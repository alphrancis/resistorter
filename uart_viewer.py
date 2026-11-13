"""
PC viewer for the ESP32-CAM UART stream, with crop + software sharpening.

  pip install pyserial opencv-python numpy
  python uart_viewer.py COM5              (Windows)
  python uart_viewer.py /dev/ttyUSB0      (Linux)
  python uart_viewer.py COM5 --baud 2000000

Keys:
  c     select a crop region (drag a box, press ENTER/SPACE; ESC cancels)
  x     clear crop
  m     sharpening mode: off -> unsharp mask -> deblur (Richardson-Lucy) -> off
  + / - sharpening strength
  f     toggle flash LED on the ESP32-CAM
  s     save raw JPEG + processed PNG
  0-3   dataset capture: save the cropped raw frame (no sharpening) to
        <out>/<label>/  labels: 0=empty 1=good 2=burnt 3=bent
  b     toggle burst: a label key then saves the next BURST_N frames
  a     toggle AUTO capture: digit keys now pick the class, and a frame is
        saved automatically each time a new part is placed and stops moving
  e     record the empty slot (do this first) so removing a part isn't saved
  q     quit

Dataset capture:  python uart_viewer.py COM5 --out dataset
The crop (c key) is saved to roi.json and reloaded next time, so every
session and the deployed model use the same crop.

The "sharp" number on screen is the variance of the Laplacian: higher = sharper.
Use it to compare settings (only compare numbers for the same crop).
"""
import argparse
import json
import os
import struct
import time

import cv2
import numpy as np
import serial

MAGIC = b"\xA5\x5A\xC3\x3C"
MAX_FRAME = 300_000  # sanity limit (bytes)
DISPLAY_MIN_W = 640  # small crops are upscaled to at least this width

MODES = ["off", "unsharp", "deblur"]

LABELS = ["empty", "good", "burnt", "bent"]  # keys 0..3
BURST_N = 10
# Auto capture tuning (mean abs pixel diff on a 64x64 gray thumbnail, 0..255)
AUTO_STILL = 3.0          # below this frame-to-frame diff = scene is still
AUTO_STABLE_FRAMES = 5    # still for this many frames before saving
AUTO_NEW = 8.0            # must differ this much from the last saved image
ROI_FILE = "roi.json"


def load_roi():
    try:
        with open(ROI_FILE) as f:
            r = json.load(f)
        return tuple(int(v) for v in r)
    except (OSError, ValueError, TypeError):
        return None


def save_roi(roi):
    with open(ROI_FILE, "w") as f:
        json.dump(list(roi) if roi else None, f)


def count_images(out_dir):
    counts = {}
    for lab in LABELS:
        d = os.path.join(out_dir, lab)
        counts[lab] = len([n for n in os.listdir(d) if n.lower().endswith(".jpg")]) if os.path.isdir(d) else 0
    return counts


def save_sample(out_dir, label, raw, roi):
    img = raw
    if roi is not None:
        x, y, w, h = roi
        img = raw[y:y + h, x:x + w]
    d = os.path.join(out_dir, label)
    os.makedirs(d, exist_ok=True)
    name = f"{label}_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.jpg"
    cv2.imwrite(os.path.join(d, name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])


def read_exact(ser, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if not chunk:
            return None  # timeout
        buf += chunk
    return bytes(buf)


def sync(ser):
    """Skip bytes until the 4-byte magic header is found."""
    window = b""
    while True:
        b = ser.read(1)
        if not b:
            continue
        window = (window + b)[-4:]
        if window == MAGIC:
            return


def read_packet(ser):
    """Returns (seq, jpeg_bytes) or (None, None) for a bad packet."""
    sync(ser)
    hdr = read_exact(ser, 6)
    if hdr is None:
        return None, None
    seq, length = struct.unpack("<HI", hdr)
    if length == 0 or length > MAX_FRAME:
        return None, None
    data = read_exact(ser, length)
    trailer = read_exact(ser, 2)
    if data is None or trailer is None:
        return None, None
    (checksum,) = struct.unpack("<H", trailer)
    if (sum(data) & 0xFFFF) != checksum:
        return None, None
    return seq, data


# ---------------------------------------------------------------- sharpening

def unsharp(img, amount):
    """Classic unsharp mask. amount 0.5..3 is a sensible range."""
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


def disk_psf(radius):
    """Model of out-of-focus blur: a flat disk."""
    r = max(1, int(round(radius)))
    size = 2 * r + 1
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    psf = (xx * xx + yy * yy <= r * r).astype(np.float32)
    return psf / psf.sum()


def richardson_lucy(channel, psf, iterations=15):
    """Richardson-Lucy deconvolution on one float32 channel in 0..1."""
    est = channel.copy()  # start from the blurry image itself
    psf_flip = psf[::-1, ::-1]
    eps = 1e-4
    for _ in range(iterations):
        conv = cv2.filter2D(est, -1, psf, borderType=cv2.BORDER_REFLECT)
        ratio = channel / (conv + eps)
        est *= cv2.filter2D(ratio, -1, psf_flip, borderType=cv2.BORDER_REFLECT)
        np.clip(est, 0, 1, out=est)
    return est


def deblur(img, radius):
    """Deblur only the brightness (Y) channel, so colors don't get noisy."""
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y = ycc[:, :, 0].astype(np.float32) / 255.0
    y = cv2.GaussianBlur(y, (3, 3), 0)  # tame JPEG noise before deconvolution
    y = richardson_lucy(y, disk_psf(radius))
    ycc[:, :, 0] = np.clip(y * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def process(img, roi, mode, strength):
    if roi is not None:
        x, y, w, h = roi
        img = img[y:y + h, x:x + w]
    # Sharpen at the camera's real resolution (blur radius is in camera pixels)
    if mode == "unsharp":
        img = unsharp(img, strength)
    elif mode == "deblur":
        img = deblur(img, strength)
    sharp = img
    # Then upscale small crops just so they're easier to see
    if img.shape[1] < DISPLAY_MIN_W:
        scale = DISPLAY_MIN_W / img.shape[1]
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return sharp, img


# ---------------------------------------------------------------- main loop

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--out", default="dataset", help="dataset folder for 0-3 keys")
    ap.add_argument("--auto", action="store_true", help="start with auto capture on")
    args = ap.parse_args()
    counts = count_images(args.out)
    burst, burst_label, burst_left = False, None, 0
    auto, active = args.auto, 0
    prev_small, last_saved, stable, flash_until = None, None, 0, 0.0
    background = None  # thumbnail of the empty slot, set with the e key

    ser = serial.Serial(args.port, args.baud, timeout=1)
    # Keep the board running: don't hold EN/GPIO0 via DTR/RTS
    ser.dtr = False
    ser.rts = False
    print(f"Listening on {args.port} @ {args.baud}...")

    frames, bad, t0, fps = 0, 0, time.time(), 0.0
    roi, mode_i = load_roi(), 0
    if roi:
        print(f"loaded crop {roi} from {ROI_FILE}")
    strength = {"unsharp": 1.5, "deblur": 2.0}  # amount / PSF radius in px

    while True:
        seq, data = read_packet(ser)
        if data is None:
            bad += 1
            continue
        raw = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if raw is None:
            bad += 1
            continue

        frames += 1
        now = time.time()
        if now - t0 >= 1.0:
            fps = frames / (now - t0)
            frames, t0 = 0, now

        mode = MODES[mode_i]
        sharp, out = process(raw, roi, mode, strength.get(mode, 0))
        score = sharpness(sharp)  # measured at native resolution

        s = f" str:{strength[mode]:.1f}" if mode in strength else ""
        lines = [
            f"#{seq} {raw.shape[1]}x{raw.shape[0]} {len(data)/1024:.1f}KB {fps:.1f}fps bad:{bad}",
            f"mode:{mode}{s} sharp:{score:.0f}" + ("  [crop]" if roi else ""),
            " ".join(f"{i}:{lab}={counts[lab]}" for i, lab in enumerate(LABELS))
            + ("  BURST" if burst else "") + (f" saving {burst_left}" if burst_left else ""),
        ]

        # ---- auto capture: save once each time the scene changes then settles
        crop = raw[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]] if roi else raw
        small = cv2.cvtColor(cv2.resize(crop, (64, 64)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        motion = float(np.mean(np.abs(small - prev_small))) if prev_small is not None else 999.0
        prev_small = small
        stable = stable + 1 if motion < AUTO_STILL else 0
        if auto and not burst_left and stable >= AUTO_STABLE_FRAMES:
            change = float(np.mean(np.abs(small - last_saved))) if last_saved is not None else 999.0
            # for non-empty classes, skip frames that look like the empty slot
            vs_bg = float(np.mean(np.abs(small - background))) if background is not None else 999.0
            is_part = active == 0 or vs_bg > AUTO_NEW
            if change > AUTO_NEW and is_part:
                save_sample(args.out, LABELS[active], raw, roi)
                counts[LABELS[active]] += 1
                last_saved = small
                flash_until = now + 0.3
                print(f"auto-saved {LABELS[active]} ({counts[LABELS[active]]})")
        lines.append(
            f"AUTO {'ON' if auto else 'off'} (a)  class: {active}={LABELS[active]}"
            f"  motion:{motion:.1f}" + ("" if background is not None else "  [press e on empty slot]")
            + ("  SAVED" if now < flash_until else ""))

        if burst_left:
            save_sample(args.out, burst_label, raw, roi)
            counts[burst_label] += 1
            burst_left -= 1
        shown = out.copy()
        for i, t in enumerate(lines):
            cv2.putText(shown, t, (5, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow("ESP32-CAM UART", shown)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            stamp = int(time.time())
            with open(f"frame_{stamp}.jpg", "wb") as f:
                f.write(data)
            cv2.imwrite(f"frame_{stamp}_processed.png", out)
            print(f"saved frame_{stamp}.jpg and frame_{stamp}_processed.png")
        elif key == ord("f"):
            ser.write(b"f")
        elif key == ord("c"):
            r = cv2.selectROI("select crop", raw, showCrosshair=False)
            cv2.destroyWindow("select crop")
            if r[2] > 4 and r[3] > 4:
                roi = tuple(int(v) for v in r)
                save_roi(roi)
            ser.reset_input_buffer()  # frames piled up while selecting
        elif key == ord("x"):
            roi = None
            save_roi(None)
        elif key == ord("b"):
            burst = not burst
        elif key == ord("e"):
            background = prev_small
            print("empty-slot background recorded")
        elif key == ord("a"):
            auto = not auto
            last_saved = prev_small  # don't instantly save what's already there
        elif ord("0") <= key < ord("0") + len(LABELS) and auto:
            active = key - ord("0")  # in auto mode, digit keys pick the class
            print(f"auto class -> {LABELS[active]}")
        elif ord("0") <= key < ord("0") + len(LABELS) and not burst_left:
            label = LABELS[key - ord("0")]
            if burst:
                burst_label, burst_left = label, BURST_N
            else:
                save_sample(args.out, label, raw, roi)
                counts[label] += 1
                print(f"saved {label} ({counts[label]})")
        elif key == ord("m"):
            mode_i = (mode_i + 1) % len(MODES)
        elif key in (ord("+"), ord("=")) and mode in strength:
            step = 0.25 if mode == "unsharp" else 0.5
            strength[mode] = min(strength[mode] + step, 10.0)
        elif key in (ord("-"), ord("_")) and mode in strength:
            step = 0.25 if mode == "unsharp" else 0.5
            strength[mode] = max(strength[mode] - step, 0.5)

    ser.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
