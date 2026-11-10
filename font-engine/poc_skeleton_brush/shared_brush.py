#!/usr/bin/env python3
"""The simplest brush that explains the image, and the paths that make it enough.

Until now: path = middle of the ink, brush = whatever is left (complex, tied
to one letter). Here both are searched together:

  ONE brush for all letters of the image, ~10 settings
      nib      shape (round / soft square / rectangle), half-width, aspect, angle
      pressure heavier lower in the line (c_y), change in bends (c_k)
      ends     length and shape of the start and of the end (round .. cut)
  ONE path per letter (spline through a few control points), moved so that
      this shared brush redraws the letter as well as possible, while staying
      smooth.

Alternate: fit the brush with paths fixed, then the paths with the brush
fixed. Tests:
  image fit   IoU of the redrawn word
  hold-out    brush fitted on two letters, third letter drawn on its plain
              middle-of-ink path (same protocol as before; best so far 0.79)
"""
from __future__ import annotations

import json
import sys
import time

import cv2
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

import compose as cp
import nib_brush as nb
import skeleton_brush as sb

Q = 4
KEYS = ("meem", "zain", "noon")
SHAPES = (2.0, 3.0, 8.0)
#          a     aspect angle  c_y   c_k   Ls   Le   e_s   e_e
LO = np.array([4.0, 0.25, 0.0, -1.2, -0.6, 0.0, 0.0, 0.05, 0.05])
HI = np.array([80.0, 1.0, 180.0, 1.2, 0.6, 4.0, 4.0, 1.5, 1.5])


def load(style):
    S = cp.load_style(style)
    H, W = S["mask"].shape[0] // Q + 1, S["mask"].shape[1] // Q + 1
    letters = {}
    for k in KEYS:
        q, wl, wr = S["strokes"][k]
        m = cv2.resize(S["comps"][S["ids"][k]]["mask"].astype(np.uint8), (W, H), interpolation=cv2.INTER_AREA) > 0
        letters[k] = dict(q0=q[::Q] / Q, mask=m, w0=float(np.median(wl + wr)) / 2 / Q)
    ys = np.nonzero(np.any([l["mask"] for l in letters.values()], axis=0))[0]
    noon_h = np.ptp(np.nonzero(letters["noon"]["mask"])[0]) + 1
    return letters, (ys.min(), ys.max()), float(noon_h), (H, W)


def ctrl_from_path(q, K=11):
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
    t = np.linspace(0, L[-1], K)
    return np.stack([np.interp(t, L, q[:, 0]), np.interp(t, L, q[:, 1])], 1)


def path_from_ctrl(C):
    t = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
    if np.any(np.diff(t) <= 1e-6):
        t = np.arange(len(C), dtype=float)
    cs = CubicSpline(t, C)
    P = cs(np.linspace(0, t[-1], max(int(t[-1] / 1.5), 8)))
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    s = np.arange(0, L[-1], 1.0)
    return np.stack([np.interp(s, L, P[:, 0]), np.interp(s, L, P[:, 1])], 1)


def scales(q, p, yr, unit):
    a, asp, ang, cy, ck, Ls, Le, es, ee = p
    d = np.gradient(q, axis=0)
    th = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    kap = np.abs(np.gradient(th)) * unit
    y = (q[:, 1] - yr[0]) / max(yr[1] - yr[0], 1)
    s = np.clip(1 + cy * (y - 0.5) + ck * np.minimum(kap, 3.0), 0.2, 3.0)
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
    for dist, Lc, e in ((L, Ls * a, es), (L[-1] - L, Le * a, ee)):
        if Lc > 0.5:
            x = np.clip(dist / Lc, 0, 1)
            s = s * np.clip(1 - (1 - x) ** 2, 0, 1) ** e
    return a * np.maximum(s, 0.02)


def draw(q, p, n, yr, unit, shape):
    canvas = np.zeros(shape, np.uint8)
    nb.sweep(canvas, q, scales(q, p, yr, unit), nb.nib_polygon(n, p[1], p[2]))
    return canvas > 127


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


class Fit:
    def __init__(self, style, use=KEYS):
        self.letters, self.yr, self.unit, self.shape = load(style)
        self.use = list(use)
        self.ctrl = {k: ctrl_from_path(self.letters[k]["q0"]) for k in KEYS}
        w0 = np.mean([self.letters[k]["w0"] for k in self.use])
        self.p = np.array([w0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.5, 0.5])
        self.n = 2.0

    def letter_iou(self, k, p=None, n=None, C=None):
        q = path_from_ctrl(self.ctrl[k] if C is None else C)
        return iou(self.letters[k]["mask"], draw(q, self.p if p is None else p, self.n if n is None else n,
                                                 self.yr, self.unit, self.shape))

    def word_iou(self, keys=None):
        keys = keys or self.use
        A = np.zeros(self.shape, bool)
        B = np.zeros(self.shape, bool)
        for k in keys:
            A |= self.letters[k]["mask"]
            B |= draw(path_from_ctrl(self.ctrl[k]), self.p, self.n, self.yr, self.unit, self.shape)
        return iou(A, B)

    def fit_brush(self, shapes=SHAPES, starts=(0.0, 60.0, 120.0), maxfev=350):
        best = (-1, None, None)
        for n in shapes:
            for a0 in starts:
                x0 = self.p.copy()
                x0[2] = a0
                f = lambda x: -np.mean([self.letter_iou(k, np.clip(x, LO, HI), n) for k in self.use])
                r = minimize(f, x0, method="Powell", bounds=list(zip(LO, HI)), options=dict(maxfev=maxfev, xtol=1e-2, ftol=1e-3))
                if -r.fun > best[0]:
                    best = (-r.fun, np.clip(r.x, LO, HI), n)
        _, self.p, self.n = best
        return best[0]

    def fit_paths(self, keys=None, lam=0.02, maxfev=900):
        for k in keys or self.use:
            C0 = self.ctrl[k]

            def f(x):
                C = x.reshape(-1, 2)
                bend = np.diff(C, 2, axis=0)
                smooth = float((bend ** 2).sum()) / (self.unit ** 2)
                return -(self.letter_iou(k, C=C) - lam * smooth)

            r = minimize(f, C0.ravel(), method="Powell", options=dict(maxfev=maxfev, xtol=0.5, ftol=1e-3))
            if r.fun < f(C0.ravel()):
                self.ctrl[k] = r.x.reshape(-1, 2)

    def describe(self):
        a, asp, ang, cy, ck, Ls, Le, es, ee = self.p
        end = lambda e, L: "cut (flat)" if (e < 0.2 or L < 0.3) else "round" if e < 0.8 else "pointed"
        return {"nib_shape": nb.SHAPES[self.n], "nib_half_width_px_original": round(float(a * Q / 2), 1),
                "nib_aspect": round(float(asp), 2), "nib_angle_deg": round(float(ang), 0),
                "pressure_top_to_bottom": round(float(cy), 2), "pressure_in_bends": round(float(ck), 2),
                "start": end(es, Ls), "start_length_in_nib_widths": round(float(Ls), 2),
                "end": end(ee, Le), "end_length_in_nib_widths": round(float(Le), 2)}


def run(style, rounds=3, log=print):
    F = Fit(style)
    hist = []
    b = F.fit_brush()
    hist.append({"step": "brush on middle-of-ink paths", "word_iou": round(F.word_iou(), 4)})
    log(style, hist[-1])
    for r in range(rounds):
        F.fit_paths()
        hist.append({"step": f"round {r + 1}: paths moved", "word_iou": round(F.word_iou(), 4)})
        log(style, hist[-1])
        F.fit_brush(shapes=(F.n,), starts=(F.p[2],))
        hist.append({"step": f"round {r + 1}: brush refitted", "word_iou": round(F.word_iou(), 4)})
        log(style, hist[-1])
    per = {k: round(F.letter_iou(k), 4) for k in KEYS}
    # hold-out: brush from two letters (their paths optimised), third on its plain path
    hold = {}
    for k in KEYS:
        H = Fit(style, use=[x for x in KEYS if x != k])
        H.fit_brush(starts=(0.0, 90.0), maxfev=250)
        H.fit_paths(maxfev=600)
        H.fit_brush(shapes=(H.n,), starts=(H.p[2],), maxfev=250)
        fair = H.letter_iou(k)                      # plain path of the hidden letter
        H.fit_paths(keys=[k], maxfev=600)           # (not a fair test: the hidden letter's path is fitted)
        hold[k] = {"held_out_iou_plain_path": round(fair, 4), "held_out_iou_path_refitted": round(H.letter_iou(k), 4)}
        log(style, "hold-out", k, hold[k])
    # picture
    canvas = np.zeros((*F.shape, 3), np.uint8)
    A = np.any([F.letters[k]["mask"] for k in KEYS], axis=0)
    B = np.zeros(F.shape, bool)
    for k in KEYS:
        B |= draw(path_from_ctrl(F.ctrl[k]), F.p, F.n, F.yr, F.unit, F.shape)
    canvas[A & B] = (235, 235, 235); canvas[A & ~B] = (230, 60, 60); canvas[~A & B] = (60, 140, 255)
    big = cv2.resize(canvas, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    for k in KEYS:
        q0 = (F.letters[k]["q0"] * 3).astype(np.int32)
        q1 = (path_from_ctrl(F.ctrl[k]) * 3).astype(np.int32)
        cv2.polylines(big, [q0], False, (150, 150, 150), 2, cv2.LINE_AA)
        cv2.polylines(big, [q1], False, (20, 160, 60), 3, cv2.LINE_AA)
        for c in (F.ctrl[k] * 3).astype(int):
            cv2.circle(big, tuple(c), 5, (20, 100, 40), -1, cv2.LINE_AA)
    cv2.imwrite(str(sb.OUT / f"{style}_9_shared_brush.png"), big[:, :, ::-1])
    return {"brush": F.describe(), "history": hist, "per_letter_iou": per, "word_iou": round(F.word_iou(), 4),
            "hold_out": hold, "mean_hold_out_plain_path": round(float(np.mean([v["held_out_iou_plain_path"] for v in hold.values()])), 4)}


def main():
    styles = sys.argv[1:] or ["style1", "style2"]
    out = {}
    for s in styles:
        t = time.time()
        out[s] = run(s)
        print(s, "done in", round(time.time() - t), "s", json.dumps(out[s]["brush"]), "word", out[s]["word_iou"],
              "hold-out", out[s]["mean_hold_out_plain_path"], flush=True)
        (sb.OUT / "shared_brush_report.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
