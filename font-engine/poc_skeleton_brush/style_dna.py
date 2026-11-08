#!/usr/bin/env python3
"""Understand the style, then create — do not copy.

The "DNA" of the uploaded lettering is measured by comparing each of its
letters with the SAME letter in the closest typeface (the donor), drawn with
the same single-pen-path rules:

  curvature_gain   how much more (or less) the pen turns per nuqta of travel
  terminal_curl    how much more the stroke ends curl (hooks, swashes)
  proportions      letter boxes in nuqtas (proportions.py)
  brush            tablet / qalam / monoline (brush_model.py)
  nodes            how economical the drawing is (pen_path.py)

New letters are then CREATED in one of two ways (arabic_anatomy.SIMILARITY):
  * similarity  — the letter shares strokes with an uploaded letter (beh ← noon,
                  reh ← zain): those strokes are re-proportioned to the new
                  letter's predicted nuqta box (Sakkal rule 8);
  * transfer    — otherwise: the donor's pen path gets the style's curvature
                  gain and terminal curl, then the predicted proportions.
"""
from __future__ import annotations

import json
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d

import pen_path as pp
import skeleton_brush as sb


def donor_path(font_path, text, size=700):
    """Single smooth pen path of a letter form in a typeface (raqm shaping,
    so ZWJ selects initial/medial/final forms). Returns path, dot centres,
    body bbox (x, y, w, h) and the nuqta (side of a dot's area) in px."""
    font = ImageFont.truetype(str(font_path), size, layout_engine=ImageFont.Layout.RAQM)
    W = size * 3
    img = Image.new("L", (W, W), 0)
    ImageDraw.Draw(img).text((W - size, int(W * 0.55)), text, font=font, fill=255, anchor="rs",
                             direction="rtl", language="ar")
    m = np.array(img) > 127
    comps = sb.split_components(m, min_area=30)
    body = max(comps, key=lambda c: c["area"])
    dots = [c for c in comps if c is not body and c["area"] < 0.2 * body["area"]]
    d = ndimage.distance_transform_edt(body["mask"])
    coords, _ = pp.single_walk(body["mask"], d)
    p = sb.smooth_open(sb.resample(sb.orient_rtl(coords), 1.0), 8.0)
    p = sb.extend_to_tips(p, body["mask"])
    centres = [tuple(np.argwhere(c["mask"]).mean(0)[::-1]) for c in dots]
    nuqta = float(np.sqrt(np.median([c["area"] for c in dots]))) if dots else None
    return p, centres, body["bbox"], nuqta


def _turning(q):
    d = np.diff(sb.resample(q, 1.0), axis=0)
    th = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    return gaussian_filter1d(np.diff(th), 3)


def turning_per_nuqta(q, D):
    k = _turning(q)
    return float(np.abs(k).sum() / (len(k) / D))


def terminal_curl(q, D, length_nuqta=1.0):
    k = _turning(q)
    n = max(3, int(length_nuqta * D))
    return float((abs(k[:n].sum()) + abs(k[-n:].sum())) / 2)


def measure_dna(style_paths, D_style, donor_font):
    """style_paths: {"meem.init": q, "zain.fina": q, "noon.isol": q} in px."""
    texts = {"meem.init": "م‍", "zain.fina": "‍ز", "noon.isol": "ن"}
    _, _, _, D_donor = donor_path(donor_font, "ن")
    rows = {}
    for k, q in style_paths.items():
        qd, _, _, _ = donor_path(donor_font, texts[k])
        rows[k] = {
            "turning_per_nuqta_style_deg": round(math.degrees(turning_per_nuqta(q, D_style)), 1),
            "turning_per_nuqta_donor_deg": round(math.degrees(turning_per_nuqta(qd, D_donor)), 1),
            "terminal_curl_style_deg": round(math.degrees(terminal_curl(q, D_style)), 1),
            "terminal_curl_donor_deg": round(math.degrees(terminal_curl(qd, D_donor)), 1),
        }
    g = np.median([r["turning_per_nuqta_style_deg"] / max(r["turning_per_nuqta_donor_deg"], 1e-3) for r in rows.values()])
    cg = np.median([r["terminal_curl_style_deg"] / max(r["terminal_curl_donor_deg"], 1e-3) for r in rows.values()])
    return {"per_letter": rows,
            "curvature_gain": round(float(np.clip(g, 0.5, 2.5)), 3),
            "terminal_curl_gain": round(float(np.clip(cg, 0.5, 3.0)), 3),
            "donor_nuqta_px": D_donor}


def transfer_curvature(q, gain, curl_gain, D, curl_len_nuqta=1.0):
    """Re-draw a donor pen path with the style's character: every bend turns
    `gain` times as much, and the last/first nuqta of each end `curl_gain`
    times as much (hooks). The path is integrated again from its curvature,
    so the result is a new curve, not a distorted copy."""
    q = sb.resample(q, 1.0)
    d = np.diff(q, axis=0)
    th = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    k = np.diff(th)
    s = np.arange(len(k))
    L = len(k)
    lc = max(5.0, curl_len_nuqta * D)
    sm = lambda x: np.clip(x, 0, 1) ** 2 * (3 - 2 * np.clip(x, 0, 1))
    w_end = np.maximum(sm(1 - s / lc), sm(1 - (L - 1 - s) / lc))
    k2 = k * (gain + (curl_gain - gain) * w_end)
    th2 = th[0] + np.r_[0, np.cumsum(k2)]
    steps = np.stack([np.cos(th2), np.sin(th2)], 1)
    return q[0] + np.vstack([[0, 0], np.cumsum(steps, axis=0)])


def reference_lines(user, preds, corpus):
    """Zoghbi's levels for the uploaded style, in image y (2x px)."""
    D, ref = user["nuqta_px"], user["noon_bottom_y"]
    ratio = np.median([v["baseline_above_noon_bottom"] / v["noon.isol"]["h"] for v in corpus.values()])
    base_up = float(ratio * user["noon.isol"]["h"])          # nuqtas above the noon bottom
    lv = {
        "sky2": max(preds["alef.isol"]["top"], preds["lam.isol"]["top"]),
        "eye1": user["meem.init"]["top"],
        "tooth3": user["noon.isol"]["top"],
        "base": base_up,
        "earth1": user["zain.fina"]["bottom"],
        "earth2": min(v["bottom"] for v in preds.values()),
    }
    return {k: float(ref - v * D) for k, v in lv.items()}, {k: round(v, 2) for k, v in lv.items()}
