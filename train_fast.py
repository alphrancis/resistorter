"""
Fast real-time resistor classifier for the AI-Thinker ESP32-CAM.

  python train_fast.py --clean      find mislabeled images (5-fold cross-validation)
                                    and move them to dataset_removed/<label>/
  python train_fast.py              train + export

The camera runs at QVGA 320x240 (fast JPEG decode); the model sees 160x120
(2x2 block average), so training images are prepared the same way:
  800x600 jpg -> 320x240 -> JPEG q12 round trip (like the camera) -> 2x2 average.

Outputs:
  model_out/fast_int8.tflite, model_out/fast_report.txt, model_out/fast_wrong/
  resistor_sorter_cam/model_data.h
"""
import argparse
import glob
import os
import random
import shutil

import cv2
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402

CAM_W, CAM_H = 320, 240
IMG_W, IMG_H = 160, 120
LABELS = ["empty", "good", "bent", "burnt"]
BLOCK = 20
SEED = 42
CACHE = "model_out/fast_cache.npz"


def prep(path):
    img = cv2.resize(cv2.imread(path), (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    img = img.reshape(IMG_H, 2, IMG_W, 2, 3).mean((1, 3)).round().astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def wire_score(path):
    g = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (200, 150)).astype(np.float32)
    return float(((g - cv2.GaussianBlur(g, (0, 0), 8)) > 25).mean() * 1000)


def load(root):
    files, ys, groups = [], [], []
    for ci, lab in enumerate(LABELS):
        fs = sorted(glob.glob(os.path.join(root, lab, "*.jpg")))
        for i, f in enumerate(fs):
            files.append(f); ys.append(ci); groups.append(f"{lab}{i // BLOCK}")
    key = "|".join(files)
    if os.path.exists(CACHE):
        c = np.load(CACHE, allow_pickle=True)
        if str(c["key"]) == key:
            return files, c["x"], np.array(ys), groups, c["score"]
    print(f"preparing {len(files)} images ...")
    x = np.stack([prep(f) for f in files])
    score = np.array([wire_score(f) for f in files])
    os.makedirs("model_out", exist_ok=True)
    np.savez(CACHE, x=x, score=score, key=key)
    return files, x, np.array(ys), groups, score


def build_model(n):
    L = tf.keras.layers
    inp = tf.keras.Input((IMG_H, IMG_W, 3))
    x = L.Rescaling(1 / 127.5, offset=-1)(inp)

    def cbr(x, f, s=1):
        x = L.Conv2D(f, 3, s, padding="same", use_bias=False)(x)
        return L.ReLU(6.0)(L.BatchNormalization()(x))

    def ds(x, f, s):
        x = L.DepthwiseConv2D(3, s, padding="same", use_bias=False)(x)
        x = L.ReLU(6.0)(L.BatchNormalization()(x))
        x = L.Conv2D(f, 1, use_bias=False)(x)
        return L.ReLU(6.0)(L.BatchNormalization()(x))

    # Activations kept small so the whole tensor arena (~100 KB) fits in the ESP32's
    # fast internal RAM; PSRAM on the original ESP32 makes inference ~50x slower.
    x = cbr(x, 8, 2)        # 60x80x8  = 38 KB
    x = ds(x, 16, 2)        # 30x40
    x = ds(x, 32, 1)
    x = ds(x, 48, 2)        # 15x20
    x = ds(x, 64, 1)
    x = ds(x, 96, 2)        # 8x10
    x = ds(x, 128, 1)
    x = ds(x, 128, 1)
    # max pool keeps small local evidence (a burn mark) that averaging would wash out
    x = L.GlobalMaxPooling2D()(x)
    x = L.Dropout(0.3)(x)
    out = L.Dense(n, activation="softmax")(x)
    return tf.keras.Model(inp, out)


def augment(x, y):
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    x = tf.image.random_brightness(x, 20.0)
    x = tf.image.random_contrast(x, 0.85, 1.15)
    x = tf.image.random_saturation(x, 0.85, 1.15)
    # random shift (resistors appear anywhere in the frame)
    x = tf.image.resize_with_crop_or_pad(x, IMG_H + 16, IMG_W + 16)
    x = tf.image.random_crop(x, (tf.shape(x)[0], IMG_H, IMG_W, 3))
    return tf.clip_by_value(x, 0, 255), y


def ds_from(x, y, train):
    d = tf.data.Dataset.from_tensor_slices((x.astype(np.float32), y))
    if train:
        d = d.shuffle(len(x), seed=SEED)
    d = d.batch(32)
    if train:
        d = d.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return d.prefetch(tf.data.AUTOTUNE)


def fit(xtr, ytr, xva, yva, epochs):
    tf.keras.utils.set_random_seed(SEED)
    m = build_model(len(LABELS))
    cnt = np.bincount(ytr, minlength=len(LABELS))
    cw = {i: len(ytr) / (len(LABELS) * max(c, 1)) for i, c in enumerate(cnt)}
    sched = tf.keras.optimizers.schedules.CosineDecay(3e-3, epochs * int(np.ceil(len(xtr) / 32)))
    m.compile(optimizer=tf.keras.optimizers.Adam(sched), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cb = [tf.keras.callbacks.ModelCheckpoint("model_out/fast_best.keras", monitor="val_accuracy",
                                             save_best_only=True, verbose=0)]
    m.fit(ds_from(xtr, ytr, True), validation_data=ds_from(xva, yva, False),
          epochs=epochs, class_weight=cw, verbose=2, callbacks=cb)
    return tf.keras.models.load_model("model_out/fast_best.keras")


def group_split(groups, ys, k):
    """Assign whole blocks of consecutive shots to folds (stratified by label)."""
    ug = sorted(set(groups))
    rng = random.Random(SEED)
    fold_of = {}
    for lab in LABELS:
        g = [u for u in ug if u.startswith(lab) and u[len(lab)].isdigit()]
        rng.shuffle(g)
        for i, u in enumerate(g):
            fold_of[u] = i % k
    return np.array([fold_of[g] for g in groups])


def clean(args):
    files, x, y, groups, score = load(args.data)
    folds = group_split(groups, y, 5)
    prob = np.zeros((len(y), len(LABELS)), np.float32)
    for f in range(5):
        tr, te = folds != f, folds == f
        m = fit(x[tr], y[tr], x[te], y[te], args.epochs)
        prob[te] = m.predict(x[te].astype(np.float32), verbose=0)
        print(f"fold {f}: acc {(prob[te].argmax(1) == y[te]).mean():.3f}")
    pred = prob.argmax(1)
    print(f"cross-validated accuracy: {(pred == y).mean():.3f}")
    bad = []
    for i, f in enumerate(files):
        if y[i] != 0 and score[i] < args.min_part_score:
            bad.append((f, "no part visible"))
        elif pred[i] != y[i] and prob[i, pred[i]] >= args.flag_conf:
            bad.append((f, f"looks like {LABELS[pred[i]]} ({prob[i, pred[i]]:.0%})"))
    print(f"moving {len(bad)} images to dataset_removed/")
    with open("model_out/removed.txt", "w") as fh:
        for f, why in bad:
            lab = os.path.basename(os.path.dirname(f))
            dst = os.path.join("dataset_removed", lab)
            os.makedirs(dst, exist_ok=True)
            shutil.move(f, os.path.join(dst, os.path.basename(f)))
            fh.write(f"{f}  {why}\n")
    from collections import Counter
    print(Counter(os.path.basename(os.path.dirname(f)) for f, _ in bad))


def to_int8(model, rep):
    c = tf.lite.TFLiteConverter.from_keras_model(model)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset = lambda: ([r[None].astype(np.float32)] for r in rep)
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type = tf.int8
    c.inference_output_type = tf.int8
    return c.convert()


def tfl_predict(tfl, xs):
    it = tf.lite.Interpreter(model_content=tfl)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    s, z = i["quantization"]
    out = []
    for im in xs:
        it.set_tensor(i["index"], np.clip(np.round(im / s + z), -128, 127).astype(np.int8)[None])
        it.invoke()
        out.append(int(np.argmax(it.get_tensor(o["index"])[0])))
    return np.array(out), (s, z)


def write_header(path, tfl, in_q):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("// Generated by train_fast.py - do not edit\n#pragma once\n#include <stdint.h>\n\n")
        f.write(f"#define MODEL_IN_W {IMG_W}\n#define MODEL_IN_H {IMG_H}\n")
        f.write(f"#define MODEL_IN_SCALE {in_q[0]:.8f}f\n#define MODEL_IN_ZERO {int(in_q[1])}\n")
        f.write(f"#define MODEL_NUM_CLASSES {len(LABELS)}\n")
        f.write("static const char *MODEL_LABELS[] = {" + ", ".join(f'"{l}"' for l in LABELS) + "};\n\n")
        f.write("alignas(16) static const unsigned char model_tflite[] = {\n")
        for i in range(0, len(tfl), 16):
            f.write("  " + ", ".join(f"0x{b:02x}" for b in tfl[i:i + 16]) + ",\n")
        f.write(f"}};\nstatic const unsigned int model_tflite_len = {len(tfl)};\n")


def train(args):
    files, x, y, groups, score = load(args.data)
    folds = group_split(groups, y, 7)
    te, va = folds == 0, folds == 1
    tr = ~(te | va)
    print("train", np.bincount(y[tr]), "val", np.bincount(y[va]), "test", np.bincount(y[te]))
    m = fit(x[tr], y[tr], x[va], y[va], args.epochs)
    m.save("model_out/fast_float.keras")
    pf = m.predict(x[te].astype(np.float32), verbose=0).argmax(1)
    rep = x[tr][np.random.default_rng(SEED).choice(tr.sum(), 300, replace=False)]
    tfl = to_int8(m, rep)
    open("model_out/fast_int8.tflite", "wb").write(tfl)
    pq, in_q = tfl_predict(tfl, x[te])
    yt = y[te]
    cm = np.zeros((4, 4), int)
    for a, b in zip(yt, pq):
        cm[a, b] += 1
    lines = [f"classes: {LABELS}", f"test images: {len(yt)}",
             f"float accuracy: {(pf == yt).mean():.3f}",
             f"int8 accuracy (runs on ESP32-CAM): {(pq == yt).mean():.3f}",
             f"model size: {len(tfl) / 1024:.0f} KB", "",
             "confusion matrix (rows = true, cols = predicted):",
             "          " + "".join(f"{l:>8s}" for l in LABELS)]
    lines += [f"{LABELS[i]:>8s}  " + "".join(f"{v:8d}" for v in cm[i]) for i in range(4)]
    print("\n".join(lines))
    open("model_out/fast_report.txt", "w").write("\n".join(lines) + "\n")
    shutil.rmtree("model_out/fast_wrong", ignore_errors=True)
    os.makedirs("model_out/fast_wrong")
    tf_files = np.array(files)[te]
    for f, t, p in zip(tf_files, yt, pq):
        if t != p:
            shutil.copy(f, f"model_out/fast_wrong/true-{LABELS[t]}_pred-{LABELS[p]}_{os.path.basename(f)}")
    write_header("resistor_sorter_cam/model_data.h", tfl, in_q)
    print("wrote resistor_sorter_cam/model_data.h")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--flag-conf", type=float, default=0.9)
    ap.add_argument("--min-part-score", type=float, default=3.0)
    a = ap.parse_args()
    clean(a) if a.clean else train(a)
