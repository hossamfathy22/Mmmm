#!/usr/bin/env python3
"""The engine's EYE, part 1: does a generated shape READ as its letter?

A classifier of the 18 basic Arabic letter bodies (rasm, dots removed),
trained on the 49 OFL typefaces of the corpus with heavy augmentation
(weight from hairline to ultra-fat, slant, squash, rotation), so that it
accepts display styles and not only text faces.

Input features of a glyph body:
  * its shape, normalised to a 32x32 box (aspect ratio kept, centred)
  * 5 numbers measured in "noon heights" (the noon bowl of the same style):
    width, height, top and bottom relative to the baseline, log aspect

Honest validation: 5-fold with whole FONTS held out.
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import skeleton_brush as sb

CLASSES = {"alef": "ا", "beh": "ب", "hah": "ح", "dal": "د", "reh": "ر", "seen": "س", "sad": "ص", "tah": "ط",
           "ain": "ع", "feh": "ف", "qaf": "ق", "kaf": "ك", "lam": "ل", "meem": "م", "noon": "ن", "heh": "ه",
           "waw": "و", "yeh": "ى"}
NAMES = list(CLASSES)
SIZE = 220
MODEL = sb.OUT / "legibility_model.joblib"


def body_of(mask):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == i


def render_body(font_path, ch):
    font = ImageFont.truetype(str(font_path), SIZE, layout_engine=ImageFont.Layout.RAQM)
    W = SIZE * 3
    base = int(W * 0.6)
    img = Image.new("L", (W, W), 0)
    ImageDraw.Draw(img).text((W // 2, base), ch, font=font, fill=255, anchor="ms", direction="rtl", language="ar")
    b = body_of(np.array(img) > 127)
    return b, base


def featurize(body, base_y, noon_h, size=32):
    """body: bool mask; base_y: baseline row; noon_h: height of the style's noon body."""
    ys, xs = np.nonzero(body)
    if len(ys) < 10:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    crop = body[y0:y1, x0:x1].astype(np.uint8) * 255
    h, w = crop.shape
    s = (size - 2) / max(h, w)
    r = cv2.resize(crop, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), np.float32)
    oy, ox = (size - r.shape[0]) // 2, (size - r.shape[1]) // 2
    canvas[oy:oy + r.shape[0], ox:ox + r.shape[1]] = r / 255.0
    canvas = cv2.GaussianBlur(canvas, (0, 0), 0.8)
    sc = np.array([w / noon_h, h / noon_h, (base_y - y0) / noon_h, (base_y - y1) / noon_h, np.log(w / h)], np.float32)
    return np.r_[canvas.ravel(), 2.0 * sc]


def augment(body, rng):
    ys, xs = np.nonzero(body)
    m = body.astype(np.uint8) * 255
    H, W = m.shape
    cx, cy = float(xs.mean()), float(ys.max())
    ang = rng.uniform(-7, 7)
    sx, sy = rng.uniform(0.85, 1.2), rng.uniform(0.85, 1.2)
    shear = rng.uniform(-0.2, 0.2)
    A = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    S = np.array([[sx, shear, cx - sx * cx - shear * cy], [0, sy, cy - sy * cy]])
    M = (np.vstack([S, [0, 0, 1]]) @ np.vstack([A, [0, 0, 1]]))[:2]
    m = cv2.warpAffine(m, M, (W, H), flags=cv2.INTER_LINEAR)
    k = int(rng.choice([0, 0, 2, 4, 7, 11, 16, 22]))          # up to ultra-fat
    if k:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    elif rng.random() < 0.3:
        m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    if rng.random() < 0.5:                                      # rounding (soft display styles)
        s = rng.uniform(2, 7)
        m = (cv2.GaussianBlur(m, (0, 0), s) > 127).astype(np.uint8) * 255
    return m > 127


def font_samples(args):
    path, n_aug, seed = args
    rng = np.random.default_rng(seed)
    out = []
    try:
        nb, nbase = render_body(path, "ن")
        if nb is None:
            return out
        noon_h = np.ptp(np.nonzero(nb)[0]) + 1
        for ci, (name, ch) in enumerate(CLASSES.items()):
            b, base = render_body(path, ch)
            if b is None or b.sum() < 80:
                continue
            for a in range(n_aug + 1):
                bb = b if a == 0 else augment(b, rng)
                if bb.sum() < 80:
                    continue
                bb = body_of(bb)
                f = featurize(bb, base, noon_h)
                if f is not None:
                    out.append((f, ci))
    except Exception as ex:
        print("font failed", path, type(ex).__name__, ex, file=sys.stderr)
    return out


def build_dataset(n_aug=14):
    fonts = sorted((sb.HERE / "corpus").glob("*.ttf"))
    with ProcessPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(font_samples, [(f, n_aug, i) for i, f in enumerate(fonts)]))
    X, y, g = [], [], []
    for gi, r in enumerate(res):
        for f, c in r:
            X.append(f); y.append(c); g.append(gi)
    return np.array(X, np.float32), np.array(y), np.array(g), [f.stem for f in fonts]


def make_clf():
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-3, max_iter=60,
                                                         early_stopping=True, random_state=0))


def train():
    from sklearn.model_selection import GroupKFold
    X, y, g, names = build_dataset()
    print("samples", X.shape, "fonts", len(set(g)))
    accs, per = [], np.zeros((len(NAMES), 2))
    for tr, te in GroupKFold(5).split(X, y, g):
        clf = make_clf().fit(X[tr], y[tr])
        p = clf.predict(X[te])
        accs.append(float((p == y[te]).mean()))
        for c in range(len(NAMES)):
            m = y[te] == c
            per[c] += [(p[m] == c).sum(), m.sum()]
    clf = make_clf().fit(X, y)
    joblib.dump({"clf": clf, "names": NAMES}, MODEL)
    rep = {"held_out_font_accuracy": round(float(np.mean(accs)), 4),
           "per_letter_accuracy": {NAMES[c]: round(float(per[c, 0] / max(per[c, 1], 1)), 3) for c in range(len(NAMES))},
           "samples": int(len(X)), "fonts": len(set(g))}
    import json
    (sb.OUT / "legibility_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


_cache = {}


def predict_proba(feats):
    if "m" not in _cache:
        _cache["m"] = joblib.load(MODEL)
    return _cache["m"]["clf"].predict_proba(np.asarray(feats, np.float32))


if __name__ == "__main__":
    train()
