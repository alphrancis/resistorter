"""
Laptop viewer for resistor_sorter_s3.ino (Freenove ESP32-S3 WROOM CAM).
Shows the live camera image with the S3's prediction, and logs results.

  python sorter_viewer.py COM7

Keys:
  0-9   save the current frame (full 800x600 JPEG) into dataset/<label>/
        using the label order from the model (shown on screen) - use this to
        collect S3-camera images for retraining with train_model.py
  q     quit
"""
import argparse
import os
import re
import struct
import time

import cv2
import numpy as np
import serial

FRAME = b"\xA5\x5A\xC3\x3C"
RESULT = b"\xA5\x5A\xC3\x3D"
MAX_FRAME = 300_000
COLORS = {"empty": (160, 160, 160), "good": (0, 200, 0), "bent": (0, 140, 255),
          "burnt": (0, 0, 255), "unsure": (0, 220, 220)}


def model_labels():
    """Read the label list from the sketch's model_data.h."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(here, "resistor_sorter_s3", "model_data.h")) as f:
            m = re.search(r"MODEL_LABELS\[\]\s*=\s*\{([^}]*)\}", f.read())
        return [s.strip().strip('"') for s in m.group(1).split(",")]
    except (OSError, AttributeError):
        return ["empty", "good", "bent"]


def read_exact(ser, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def read_packet(ser, text):
    """Returns ("frame", jpeg) / ("result", (label, conf, ms, decided)) / (None, None).
    Plain text the board prints (boot messages) is collected into `text`."""
    window = b""
    while True:
        b = ser.read(1)
        if not b:
            return None, None
        window = (window + b)[-4:]
        if b == b"\n":
            line = text.pop() if text else ""
            if line.strip():
                print("[S3]", line.strip())
            text.append("")
        elif 32 <= b[0] < 127:
            if not text:
                text.append("")
            text[-1] += b.decode()
        if window == FRAME:
            text[:] = [""]
            hdr = read_exact(ser, 6)
            if hdr is None:
                return None, None
            _, length = struct.unpack("<HI", hdr)
            if not 0 < length <= MAX_FRAME:
                return None, None
            data, trailer = read_exact(ser, length), read_exact(ser, 2)
            if data is None or trailer is None:
                return None, None
            if (sum(data) & 0xFFFF) != struct.unpack("<H", trailer)[0]:
                return None, None
            return "frame", data
        if window == RESULT:
            text[:] = [""]
            p = read_exact(ser, 5)
            if p is None:
                return None, None
            label, conf, ms, decided = struct.unpack("<BBHB", p)
            return "result", (label, conf, ms, bool(decided))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port")
    ap.add_argument("--baud", type=int, default=115200)  # ignored by USB CDC
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--log", default="sorter_log.csv")
    args = ap.parse_args()

    labels = model_labels()
    ser = serial.Serial(args.port, args.baud, timeout=2)
    print(f"listening on {args.port}, labels {labels}")
    log = open(args.log, "a")

    jpeg, img, text = None, None, [""]
    last_decided, fps, t0, n = "-", 0.0, time.time(), 0
    while True:
        kind, val = read_packet(ser, text)
        if kind == "frame":
            jpeg = val
            img = cv2.imdecode(np.frombuffer(val, np.uint8), cv2.IMREAD_COLOR)
            continue
        if kind != "result" or img is None:
            if kind is None and cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        label_i, conf, ms, decided = val
        name = labels[label_i] if label_i < len(labels) else "unsure"
        if decided:
            last_decided = name
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"{stamp}  DECIDED: {name} ({conf}%)")
            log.write(f"{stamp},{name},{conf}\n")
            log.flush()

        n += 1
        if time.time() - t0 >= 1:
            fps, n, t0 = n / (time.time() - t0), 0, time.time()

        shown = cv2.resize(img, (800, 600))
        color = COLORS.get(name, (255, 255, 255))
        cv2.rectangle(shown, (0, 0), (800, 70), (0, 0, 0), -1)
        cv2.putText(shown, f"{name.upper()}  {conf}%", (12, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3, cv2.LINE_AA)
        cv2.putText(shown, f"inference {ms} ms  {fps:.1f} fps   last decided: {last_decided}",
                    (440, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        keys = "  ".join(f"{i}={l}" for i, l in enumerate(labels))
        cv2.putText(shown, f"save: {keys}", (440, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.imshow("Resistor sorter (ESP32-S3)", shown)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if ord("0") <= key <= ord("9") and key - ord("0") < len(labels) and jpeg:
            lab = labels[key - ord("0")]
            d = os.path.join(args.out, lab)
            os.makedirs(d, exist_ok=True)
            fn = os.path.join(d, f"{lab}_s3_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time()*1000)%1000:03d}.jpg")
            with open(fn, "wb") as f:
                f.write(jpeg)
            print("saved", fn)

    ser.close()
    log.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
