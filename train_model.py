"""
Train the resistor classifier and export it for the ESP32-S3.

  python train_model.py                 (uses ./dataset/<label>/*.jpg)

Outputs (in ./model_out and the Arduino sketch folder):
  model_out/resistor_int8.tflite        quantized model
  model_out/report.txt                  accuracy + confusion matrix on the held-out test set
  resistor_sorter_s3/model_data.h       model as a C array for the sketch
"""
import argparse
import glob
import os
import random

import cv2
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402

IMG_W, IMG_H = 160, 120          # model input (keeps the camera's 4:3 shape)
LABEL_ORDER = ["empty", "good", "bent", "burnt"]  # only folders that exist are used
BLOCK = 20                       # consecutive shots kept together when splitting
SEED = 42


# ------------------------------------------------------------------ data

def wire_score(img_bgr):
    """How many thin bright (wire/body) pixels the frame contains, per mille.
    Empty frames score ~0, a fully visible resistor ~5-10."""
    g = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), (200, 150)).astype(np.float32)
    return float(((g - cv2.GaussianBlur(g, (0, 0), 8)) > 25).mean() * 1000)


def load_image(path):
    img = cv2.imread(path)
    return img, cv2.cvtColor(cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA),
                             cv2.COLOR_BGR2RGB)


def load_dataset(root, min_part_score):
    labels = [l for l in LABEL_ORDER if glob.glob(os.path.join(root, l, "*.jpg"))]
    split = {"train": ([], []), "val": ([], []), "test": ([], [])}
    skipped = []
    pattern = ["train"] * 5 + ["val", "test"]  # ~70/15/15 by blocks of consecutive shots
    for ci, lab in enumerate(labels):
        files = sorted(glob.glob(os.path.join(root, lab, "*.jpg")))  # names sort by time
        blocks = [files[i:i + BLOCK] for i in range(0, len(files), BLOCK)]
        rng = random.Random(SEED + ci)
        order = pattern * (len(blocks) // len(pattern) + 1)
        order = order[:len(blocks)]
        rng.shuffle(order)
        for blk, part in zip(blocks, order):
            for f in blk:
                raw, img = load_image(f)
                if lab != "empty" and wire_score(raw) < min_part_score:
                    skipped.append(f)  # empty / part out of frame: mislabeled
                    continue
                split[part][0].append(img)
                split[part][1].append(ci)
    out = {k: (np.array(x, np.uint8), np.array(y, np.int32)) for k, (x, y) in split.items()}
    return labels, out, skipped


# ------------------------------------------------------------------ model

def build_model(n_classes):
    inp = tf.keras.Input((IMG_H, IMG_W, 3))          # raw 0..255 RGB
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1)(inp)
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_H, IMG_W, 3), alpha=0.35, include_top=False, weights="imagenet")
    base.trainable = False
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out), base


def augmenter():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.08, fill_mode="reflect"),
        tf.keras.layers.RandomTranslation(0.08, 0.08, fill_mode="reflect"),
        tf.keras.layers.RandomBrightness(0.15, value_range=(0, 255)),
        tf.keras.layers.RandomContrast(0.15),
    ])


def make_ds(x, y, aug=None, shuffle=False, bs=32):
    ds = tf.data.Dataset.from_tensor_slices((x.astype(np.float32), y))
    if shuffle:
        ds = ds.shuffle(len(x), seed=SEED)
    ds = ds.batch(bs)
    if aug is not None:
        ds = ds.map(lambda a, b: (tf.clip_by_value(aug(a, training=True), 0, 255), b))
    return ds.prefetch(tf.data.AUTOTUNE)


# ------------------------------------------------------------------ export

def to_int8_tflite(model, rep_images):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]

    def rep():
        for im in rep_images:
            yield [im[None].astype(np.float32)]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def tflite_predict(tfl, images):
    it = tf.lite.Interpreter(model_content=tfl)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    s, z = i["quantization"]
    preds = []
    for im in images:
        q = np.clip(np.round(im.astype(np.float32) / s + z), -128, 127).astype(np.int8)
        it.set_tensor(i["index"], q[None])
        it.invoke()
        preds.append(int(np.argmax(it.get_tensor(o["index"])[0])))
    return np.array(preds), (s, z)


def confusion(y, p, n):
    m = np.zeros((n, n), int)
    for a, b in zip(y, p):
        m[a, b] += 1
    return m


def write_header(path, tfl, labels, in_q):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("// Generated by train_model.py - do not edit\n#pragma once\n#include <stdint.h>\n\n")
        f.write(f"#define MODEL_IN_W {IMG_W}\n#define MODEL_IN_H {IMG_H}\n")
        f.write(f"#define MODEL_IN_SCALE {in_q[0]:.8f}f\n#define MODEL_IN_ZERO {int(in_q[1])}\n")
        f.write(f"#define MODEL_NUM_CLASSES {len(labels)}\n")
        f.write("static const char *MODEL_LABELS[] = {" + ", ".join(f'"{l}"' for l in labels) + "};\n\n")
        f.write(f"alignas(16) static const unsigned char model_tflite[] = {{\n")
        for i in range(0, len(tfl), 16):
            f.write("  " + ", ".join(f"0x{b:02x}" for b in tfl[i:i + 16]) + ",\n")
        f.write(f"}};\nstatic const unsigned int model_tflite_len = {len(tfl)};\n")


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--epochs", type=int, default=15, help="epochs per phase")
    ap.add_argument("--min-part-score", type=float, default=3.0,
                    help="skip non-empty images with less visible wire than this")
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    labels, d, skipped = load_dataset(args.data, args.min_part_score)
    (xtr, ytr), (xva, yva), (xte, yte) = d["train"], d["val"], d["test"]
    print("classes:", labels)
    for k in d:
        print(f"{k:5s}", {labels[c]: int((d[k][1] == c).sum()) for c in range(len(labels))})
    print(f"skipped {len(skipped)} empty/out-of-frame images from non-empty classes")

    # balance classes
    counts = np.bincount(ytr, minlength=len(labels))
    cw = {i: len(ytr) / (len(labels) * c) for i, c in enumerate(counts)}

    model, base = build_model(len(labels))
    aug = augmenter()
    tr = make_ds(xtr, ytr, aug, shuffle=True)
    va = make_ds(xva, yva)

    model.compile(tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(tr, validation_data=va, epochs=args.epochs, class_weight=cw, verbose=2)

    # fine-tune the top of the backbone
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False
    model.compile(tf.keras.optimizers.Adam(1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(tr, validation_data=va, epochs=args.epochs, class_weight=cw, verbose=2,
              callbacks=[tf.keras.callbacks.EarlyStopping("val_loss", patience=5,
                                                          restore_best_weights=True)])

    os.makedirs("model_out", exist_ok=True)
    model.save("model_out/resistor_float.keras")
    pf = np.argmax(model.predict(xte.astype(np.float32), verbose=0), 1)

    rep = xtr[np.random.default_rng(SEED).choice(len(xtr), min(200, len(xtr)), replace=False)]
    tfl = to_int8_tflite(model, rep)
    with open("model_out/resistor_int8.tflite", "wb") as f:
        f.write(tfl)
    pq, in_q = tflite_predict(tfl, xte)

    n = len(labels)
    lines = [f"classes: {labels}",
             f"test images: {len(yte)} (never used for training)",
             f"float model accuracy: {(pf == yte).mean():.3f}",
             f"int8 model accuracy (what runs on the ESP32-S3): {(pq == yte).mean():.3f}",
             f"int8 model size: {len(tfl) / 1024:.0f} KB",
             "", "confusion matrix, int8 (rows = true, cols = predicted):",
             "            " + "".join(f"{l:>8s}" for l in labels)]
    m = confusion(yte, pq, n)
    for i in range(n):
        lines.append(f"{labels[i]:>10s}  " + "".join(f"{v:8d}" for v in m[i]))
    lines += ["", "skipped images:"] + skipped
    report = "\n".join(lines)
    print("\n" + "\n".join(lines[:9 + n]))
    with open("model_out/report.txt", "w") as f:
        f.write(report + "\n")

    # save misclassified test images for review
    os.makedirs("model_out/wrong", exist_ok=True)
    for k, (im, t, p) in enumerate(zip(xte, yte, pq)):
        if t != p:
            cv2.imwrite(f"model_out/wrong/{k:03d}_true-{labels[t]}_pred-{labels[p]}.png",
                        cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

    write_header("resistor_sorter_s3/model_data.h", tfl, labels, in_q)
    print("wrote resistor_sorter_s3/model_data.h")


if __name__ == "__main__":
    main()
