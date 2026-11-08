#!/usr/bin/env python3
"""A small, interpretable brush model learned from measured strokes.

For every sample on a pen path we know the measured left/right half-widths.
We explain them with a few physical-ish terms:

    w_side = cap(d_start) * cap(d_end) * (b0 + b1*y + b2*|k| + b3*(+/-k))

  y        vertical position in the word frame (0 = top of ink, 1 = bottom)
  |k|      curvature magnitude (1/px) -> thinner in tight turns
  +/-k     signed curvature, mirrored for the left/right side -> the outside
           of a curve bulges more than the inside
  cap(d)   round terminal: a circle of radius Rc at each free stroke end

Fitted once from the strokes of the uploaded word, it can paint ANY new pen
path (a letter that is not in the image) with the same brush behaviour.
"""
from __future__ import annotations

import json
import math

import numpy as np
from scipy.ndimage import gaussian_filter1d

import skeleton_brush as sb


def stroke_features(q, y_top, y_bot):
    T = sb._tangents(q)
    ang = np.unwrap(np.arctan2(T[:, 1], T[:, 0]))
    k = gaussian_filter1d(np.gradient(ang), 5)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
    y = (q[:, 1] - y_top) / max(y_bot - y_top, 1e-9)
    return y, k, s, s[-1] - s


def cap(d, Rc):
    d = np.clip(d / Rc, 0, 1)
    return np.sqrt(np.clip(1 - (1 - d) ** 2, 0, 1))


class BrushModel:
    def __init__(self):
        self.beta = None
        self.Rc = None

    def _design(self, y, k, d0, d1, Rb):
        one = np.ones_like(y)
        left = np.stack([one, y, np.abs(k), k], 1)
        right = np.stack([one, y, np.abs(k), -k], 1)
        return left, right

    def fit(self, strokes, y_top, y_bot):
        """strokes: dicts with q, wl, wr, attached_start (bool)."""
        raw = []
        for st in strokes:
            y, k, d0, d1 = stroke_features(st["q"], y_top, y_bot)
            if st.get("attached_start"):
                d0 = np.full_like(d0, 1e9)
            raw.append((y, k, d0, d1, st["wl"], st["wr"]))
        best = None
        for Rb in (0,):  # (a terminal-ball term was tried; it did not help on held-out letters)
            for Rc in np.arange(20, 601, 10):
                A, b = [], []
                for y, k, d0, d1, wl, wr in raw:
                    Xl, Xr = self._design(y, k, d0, d1, Rb)
                    c = cap(d0, Rc) * cap(d1, Rc)
                    A += [Xl * c[:, None], Xr * c[:, None]]
                    b += [wl, wr]
                A, b = np.vstack(A), np.concatenate(b)
                beta, *_ = np.linalg.lstsq(A, b, rcond=None)
                err = np.mean((A @ beta - b) ** 2)
                if best is None or err < best[0]:
                    best = (err, Rc, Rb, beta, 1 - err / np.var(b))
        _, self.Rc, self.Rb, self.beta, self.r2 = best
        self.y_top, self.y_bot = y_top, y_bot
        return self

    def predict(self, q, attached_start=False):
        y, k, d0, d1 = stroke_features(q, self.y_top, self.y_bot)
        if attached_start:
            d0 = np.full_like(d0, 1e9)
        Xl, Xr = self._design(y, k, d0, d1, self.Rb)
        c = cap(d0, self.Rc) * cap(d1, self.Rc)
        return np.clip(Xl @ self.beta * c, 1, None), np.clip(Xr @ self.beta * c, 1, None)

    def describe(self, px_per_unit=1.0):
        b0, b1, b2, b3 = self.beta
        return {
            "base_half_width_px": round(float(b0) / px_per_unit, 2),
            "extra_half_width_top_to_bottom_px": round(float(b1) / px_per_unit, 2),
            "thinning_per_curvature": round(float(b2) / px_per_unit, 1),
            "outer_side_bulge_per_curvature": round(float(b3) / px_per_unit, 1),
            "round_terminal_radius_px": round(float(self.Rc) / px_per_unit, 1),
            "r2_on_training_strokes": round(float(self.r2), 3),
        }

    def to_json(self):
        return json.dumps({"beta": [float(v) for v in self.beta], "Rc": float(self.Rc), "Rb": float(self.Rb),
                           "y_top": float(self.y_top), "y_bot": float(self.y_bot)})
