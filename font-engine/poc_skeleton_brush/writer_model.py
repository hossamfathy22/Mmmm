#!/usr/bin/env python3
"""Put the system in the writer's place: how was this lettering written?

Stroke theory (Gerrit Noordzij, "The Stroke: Theory of Writing" / "The Shape
of the Stroke", 1991) says contrast — where a stroke is thick and where thin —
comes from three mechanisms:

  translation  broad nib at a CONSTANT angle: thickness depends only on the
               direction of travel (horizontal vs vertical strokes differ);
  rotation     the nib angle CHANGES along the stroke (as in the waw lesson);
  expansion    pressure (pointed pen, brush, tablet): thickness follows the
               hand's pressure along the stroke, independent of direction.

Arabic calligraphy adds: nib cut at an angle (≈70° in Sakkal's Naskh chart),
rotated during the stroke, proportions measured in nuqtas.

This module measures, from the exact brush of every stroke:
  * how much of the thickness each mechanism explains (R²), i.e. the writer's
    tool and hand;
  * thickness of horizontal vs vertical travel, and the stress angle;
  * the terminals: how each stroke starts and ends (shape, size relative to
    the body, cut angle) — read from the art-brush caps.
"""
from __future__ import annotations

import json
import math

import numpy as np
from scipy.ndimage import gaussian_filter1d

import exact_brush as eb
import skeleton_brush as sb


def _r2(y, yhat):
    return float(1 - np.var(y - yhat) / (np.var(y) + 1e-9))


def contrast_analysis(strokes, y_top, y_bot):
    """strokes: list of (q, wl, wr). Returns R² per mechanism + stress."""
    th, w, u, k, y = [], [], [], [], []
    for q, wl, wr in strokes:
        T = sb._tangents(q)
        ang = np.unwrap(np.arctan2(T[:, 1], T[:, 0]))
        uu, _ = sb.arclen_u(q)
        # body only: terminals belong to the caps, not to the contrast
        body = (uu > 0.08) & (uu < 0.92)
        th.append(np.arctan2(T[:, 1], T[:, 0])[body])
        w.append((wl + wr)[body])
        u.append(uu[body])
        k.append(np.abs(gaussian_filter1d(np.gradient(ang), 5))[body])
        y.append(((q[:, 1] - y_top) / (y_bot - y_top))[body])
    th, w, u, k, y = map(np.concatenate, (th, w, u, k, y))
    res = {}
    # translation: fixed nib angle
    best = (-9, 0, None)
    for a in np.radians(np.arange(0, 180, 2)):
        X = np.stack([np.ones_like(th), np.abs(np.sin(th - a))], 1)
        c, *_ = np.linalg.lstsq(X, w, rcond=None)
        r2 = _r2(w, X @ c)
        if r2 > best[0]:
            best = (r2, a, c)
    res["translation_r2"] = round(best[0], 3)
    nib = best[1]
    res["nib_angle_deg (0 = horizontal)"] = round(math.degrees(nib), 1)
    # rotation: nib angle drifting along the stroke
    bestr = -9
    for a0 in np.radians(np.arange(0, 180, 4)):
        for a1 in np.radians(np.arange(-90, 91, 15)):
            X = np.stack([np.ones_like(th), np.abs(np.sin(th - (a0 + a1 * u)))], 1)
            c, *_ = np.linalg.lstsq(X, w, rcond=None)
            bestr = max(bestr, _r2(w, X @ c))
    res["rotation_r2"] = round(bestr, 3)
    # expansion: pressure along the stroke (position, height in the letter, curvature)
    X = np.stack([np.ones_like(u), u, u ** 2, y, k], 1)
    c, *_ = np.linalg.lstsq(X, w, rcond=None)
    res["expansion_r2"] = round(_r2(w, X @ c), 3)
    # thickness by direction of travel (what the user asked: horizontal vs vertical)
    deg = np.degrees(np.abs(np.arctan2(np.sin(th), np.abs(np.cos(th)))))  # 0 = horizontal, 90 = vertical
    hor, ver = w[deg < 25], w[deg > 65]
    res["median_thickness_horizontal_px"] = round(float(np.median(hor)), 1) if len(hor) else None
    res["median_thickness_vertical_px"] = round(float(np.median(ver)), 1) if len(ver) else None
    if len(hor) and len(ver):
        res["vertical_to_horizontal_ratio"] = round(float(np.median(ver) / np.median(hor)), 2)
    res["stress_angle_deg (direction of the thickest strokes)"] = round((math.degrees(nib) + 90) % 180, 1)
    shares = {m: max(res[f"{m}_r2"], 0) for m in ("translation", "rotation", "expansion")}
    res["dominant_mechanism"] = max(shares, key=shares.get)
    return res


def terminal_analysis(q, wl, wr, caps):
    """How each stroke starts and ends, read from the last stretch of the
    stroke (1.5 body widths): the taper shape of the width profile and its
    left/right balance (a terminal that bends to one side)."""
    n = len(q)
    w = wl + wr
    body_w = float(np.median(w[n // 5: n - n // 5])) if n > 10 else float(np.median(w))
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
    out = {}
    for name, dist in (("start", s), ("end", s[-1] - s)):
        zone = dist <= 1.5 * body_w
        t = dist[zone] / body_w
        f = w[zone] / body_w
        o = np.argsort(t)
        t, f = t[o], f[o]
        at = lambda x: float(np.interp(x, t, f))
        bal = float(np.mean((wl[zone] - wr[zone]) / (w[zone] + 1e-9)))
        if np.max(f) > 1.15 and at(0.6) > 1.0:
            shape = "ball (swells before the end)"
        elif at(0.1) > 0.8:
            shape = "flat / cut"
        elif at(0.25) >= 0.45:
            shape = "round"
        else:
            shape = "pointed / tapered"
        out[name] = {"shape": shape,
                     "width_at_quarter_body_from_tip": round(at(0.25), 2),
                     "width_at_one_body_from_tip": round(at(1.0), 2),
                     "leans": "left" if bal > 0.15 else "right" if bal < -0.15 else "centred"}
    out["body_thickness_px"] = round(body_w, 1)
    return out


def analyse_style(style, box):
    img = sb.HERE / "input" / "muzn_two_styles.jpg"
    mask, _, model, vis = sb.analyse(img, box, style)
    comps = sb.split_components(mask)
    ys = np.nonzero(mask)[0]
    strokes, terms = [], {}
    for v in vis:
        if v["kind"] != "stroke":
            continue
        q, m = v["p"], comps[v["comp"]]["mask"]
        wl, wr, caps = eb.measure(q, m)
        strokes.append((q, wl, wr))
        terms[f"component {v['comp']} ({comps[v['comp']]['kind']})"] = terminal_analysis(q, wl, wr, caps)
    return {"contrast": contrast_analysis(strokes, ys.min(), ys.max()), "terminals": terms}


def main():
    out = {"style1": analyse_style("style1", (1000, 360, 1850, 880)),
           "style2": analyse_style("style2", (120, 360, 860, 880))}
    (sb.OUT / "writer_model.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for s, r in out.items():
        print(s, json.dumps(r["contrast"], ensure_ascii=False))
        for k, v in r["terminals"].items():
            print("   ", k, v)


if __name__ == "__main__":
    main()
