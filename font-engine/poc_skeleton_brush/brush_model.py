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
    # Longest terminal taper searched. A cap of 160 px (about one stroke width)
    # was tried: held-out IoU fell from 0.789 to 0.764, so the long tapering
    # tails are a real feature of this style and the search stays wide.
    max_cap = 600

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
            for Rc in np.arange(20, self.max_cap + 1, 10):
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


# ----------------------------------------------------------------------------
# Brush library: the three ways letters are actually drawn
# ----------------------------------------------------------------------------


class TabletBrush(BrushModel):
    """Tablet / pressure brush (the model above): width from pressure along
    the stroke, heavier low in the letter, thinner in tight turns, outer side
    of a curve fuller (pen tilt)."""

    kind = "tablet (pressure + tilt)"


class QalamBrush:
    """Broad-nib reed pen (qalam). The mark is the nib dragged along the path:
    full width = W * |sin(theta - alpha)| + w0, where theta is the direction of
    travel and alpha the nib angle. As in calligraphy practice the nib may be
    rotated gradually during a stroke: alpha(u) = a0 + a1 * u."""

    kind = "qalam (broad nib, rotating)"

    def fit(self, strokes, y_top=None, y_bot=None):
        rows = []
        for st in strokes:
            T = sb._tangents(st["q"])
            th = np.arctan2(T[:, 1], T[:, 0])
            u, _ = sb.arclen_u(st["q"])
            rows.append((th, u, st["wl"] + st["wr"]))
        th = np.concatenate([r[0] for r in rows])
        u = np.concatenate([r[1] for r in rows])
        w = np.concatenate([r[2] for r in rows])
        best = None
        for a0 in np.radians(np.arange(0, 180, 3)):
            for a1 in np.radians(np.arange(-60, 61, 10)):
                x = np.abs(np.sin(th - (a0 + a1 * u)))
                A = np.stack([x, np.ones_like(x)], 1)
                coef, *_ = np.linalg.lstsq(A, w, rcond=None)
                err = np.mean((A @ coef - w) ** 2)
                if best is None or err < best[0]:
                    best = (err, a0, a1, coef)
        err, self.a0, self.a1, (self.W, self.w0) = best
        self.r2 = 1 - err / np.var(w)
        return self

    def predict(self, q, attached_start=False):
        T = sb._tangents(q)
        th = np.arctan2(T[:, 1], T[:, 0])
        u, _ = sb.arclen_u(q)
        w = np.clip(self.W * np.abs(np.sin(th - (self.a0 + self.a1 * u))) + self.w0, 2, None)
        return w / 2, w / 2

    def describe(self):
        return {"nib_angle_deg": round(math.degrees(self.a0), 1),
                "nib_rotation_along_stroke_deg": round(math.degrees(self.a1), 1),
                "nib_width_px": round(float(self.W + self.w0), 1), "hairline_px": round(float(self.w0), 1),
                "r2_on_training_strokes": round(float(self.r2), 3)}


class MonolineBrush:
    """Even-width pen with round ends."""

    kind = "monoline"

    def fit(self, strokes, y_top=None, y_bot=None):
        self.h = float(np.median(np.concatenate([np.r_[s["wl"], s["wr"]] for s in strokes])))
        return self

    def predict(self, q, attached_start=False):
        _, _, d0, d1 = stroke_features(q, 0, 1)
        if attached_start:
            d0 = d0 + 1e9
        c = cap(d0, self.h) * cap(d1, self.h)
        return self.h * c, self.h * c

    def describe(self):
        return {"half_width_px": round(self.h, 1)}


BRUSHES = {"tablet": TabletBrush, "qalam": QalamBrush, "monoline": MonolineBrush}
