#!/usr/bin/env python3
"""How did the designer make Sirang / سیرنگ? Measure it.

  grid      baseline, x-height (flat stem tops) and descender of each line
  slant     lean of the stems
  pen       stem thickness and hairline thickness (two clusters of the
            stroke-width distribution), contrast = stem / hairline
  parts     dots (diameter, roundness), hairline bars (thickness, length)
  rhythm    distance between neighbouring stems on the x-height middle line
All lengths in px of the original image; the image is upscaled 4x for measuring.
Output: output/sirang_measure.json, output/sirang_measure.png
"""
from __future__ import annotations

import json
import math

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

import skeleton_brush as sb
import style_metric as sm

UP = 4
IMG = sb.HERE / "input" / "sirang" / "sirang.jpg"


def load():
    g = cv2.imread(str(IMG), cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, None, fx=UP, fy=UP, interpolation=cv2.INTER_CUBIC)
    return g < 128


def edge_rows(m):
    """Rows where many ink tops (no ink above) or bottoms sit: flat tops / baseline."""
    top = m & ~np.roll(m, 1, 0)
    bot = m & ~np.roll(m, -1, 0)
    return top.sum(1).astype(float), bot.sum(1).astype(float)


def grid(m):
    ys = np.nonzero(m.any(1))[0]
    top, bot = edge_rows(m)
    k = np.ones(2 * UP + 1)
    top, bot = np.convolve(top, k, "same"), np.convolve(bot, k, "same")
    rows = np.arange(len(top))
    # baseline = strongest bottom-edge row in the lower half of the line, x-height = strongest top-edge row above it
    lo, hi = ys.min(), ys.max()
    mid = (lo + hi) // 2
    base = int(rows[mid:hi + 1][np.argmax(bot[mid:hi + 1])])
    xh = int(rows[lo:base - 10 * UP][np.argmax(top[lo:base - 10 * UP])])
    return {"top_of_ink": int(lo), "x_height_line": xh, "baseline": base, "bottom_of_ink": int(hi),
            "x_height": base - xh, "descender": int(hi) - base, "above_x_height": xh - int(lo)}


def deskew(m, slant_deg, base):
    t = math.tan(math.radians(slant_deg))
    # slant < 0 here means the stems lean right; x' = x + t * (base - y) stands them upright
    M = np.float32([[1, -t, t * base], [0, 1, 0]])
    return cv2.warpAffine(m.astype(np.uint8), M, (m.shape[1], m.shape[0]), flags=cv2.INTER_NEAREST) > 0


def pen(m):
    d = ndimage.distance_transform_edt(m)
    sk = skeletonize(m)
    w = 2 * d[sk]
    w = w[w > 1.5]
    # two clusters on log width (1-D k-means)
    lw = np.log(w)
    c = np.percentile(lw, [15, 85])
    for _ in range(50):
        lab = np.abs(lw[:, None] - c[None]).argmin(1)
        c = np.array([lw[lab == i].mean() for i in (0, 1)])
    thin, thick = np.exp(c)
    return {"hairline": thin / UP, "stem": float(np.percentile(w[lab == 1], 75)) / UP,
            "stem_median": thick / UP, "contrast_stem_over_hairline": float(np.percentile(w[lab == 1], 75) / thin)}


def stems_on_line(md, y):
    """Runs of ink along one row of the deskewed mask: stems are the wide runs."""
    row = md[y].astype(np.int8)
    d = np.diff(np.r_[0, row, 0])
    s, e = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    return [(int(a), int(b)) for a, b in zip(s, e)]


def parts(m):
    n, lab, st, cen = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    dots, bars = [], []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 30 * UP * UP / 4:
            continue
        cs, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        per = cv2.arcLength(cs[0], True)
        circ = 4 * math.pi * a / max(per ** 2, 1)
        if circ > 0.8 and 0.8 < w / h < 1.25:
            dots.append({"diameter": round(float((w + h) / 2 / UP), 1), "roundness": round(circ, 2),
                         "center": [round(float(cen[i][0] / UP), 1), round(float(cen[i][1] / UP), 1)]})
        elif w > 4 * h:
            bars.append({"thickness": round(float(a / w / UP), 2), "length": round(float(w / UP), 1),
                         "y": round(float(cen[i][1] / UP), 1)})
    return dots, bars


def main():
    m = load()
    top, bot, _ = sm.split_lines(m)
    lines = {"arabic": top, "latin": bot}
    rep, draw = {}, []
    for name, L in lines.items():
        G = grid(L)
        dna = sm.dna(L, float(G["x_height"]))
        slant = dna["slant"]
        md = deskew(L, slant, G["baseline"])
        P = pen(md)
        y_mid = (G["x_height_line"] + G["baseline"]) // 2
        runs = stems_on_line(md, y_mid)
        stem_runs = [r for r in runs if (r[1] - r[0]) > 0.6 * P["stem_median"] * UP]
        centers = [0.5 * (a + b) for a, b in stem_runs]
        gaps = np.diff(sorted(centers)) / UP
        dots, bars = parts(L)
        rep[name] = {"grid_px": {k: round(v / UP, 1) for k, v in G.items()},
                     "slant_deg": round(slant, 2),
                     "pen": {k: round(v, 2) for k, v in P.items()},
                     "stem_over_x_height": round(P["stem"] / (G["x_height"] / UP), 3),
                     "stems_on_middle_line": {"count": len(stem_runs),
                                              "widths": [round((b - a) / UP, 1) for a, b in stem_runs],
                                              "spacing": [round(float(g), 1) for g in gaps]},
                     "dots": dots, "hairline_bars": bars}
        draw.append((name, L, G, stem_runs, y_mid, slant))
    a, l = rep["arabic"], rep["latin"]
    rep["arabic_vs_latin"] = {
        "x_height": f"{a['grid_px']['x_height']} vs {l['grid_px']['x_height']}",
        "slant_deg": f"{a['slant_deg']} vs {l['slant_deg']}",
        "stem": f"{a['pen']['stem']} vs {l['pen']['stem']}",
        "hairline": f"{a['pen']['hairline']} vs {l['pen']['hairline']}",
        "contrast": f"{a['pen']['contrast_stem_over_hairline']} vs {l['pen']['contrast_stem_over_hairline']}",
        "stem_over_x_height": f"{a['stem_over_x_height']} vs {l['stem_over_x_height']}",
        "median_stem_spacing": f"{np.median(a['stems_on_middle_line']['spacing']) if a['stems_on_middle_line']['spacing'] else None} vs "
                               f"{np.median(l['stems_on_middle_line']['spacing']) if l['stems_on_middle_line']['spacing'] else None}",
        "dot_diameter": f"{[d['diameter'] for d in a['dots']]} vs {[d['diameter'] for d in l['dots']]}",
        "descender": f"{a['grid_px']['descender']} vs {l['grid_px']['descender']}"}
    (sb.OUT / "sirang_measure.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps(rep["arabic_vs_latin"], ensure_ascii=False, indent=1))
    for k in ("arabic", "latin"):
        print(k, rep[k]["stems_on_middle_line"], rep[k]["dots"], rep[k]["hairline_bars"], rep[k]["grid_px"])
    # picture: guides, stems, slant
    img = cv2.cvtColor((~m).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    for name, L, G, stem_runs, y_mid, slant in draw:
        for key, col in (("x_height_line", (0, 140, 255)), ("baseline", (220, 60, 0)), ("bottom_of_ink", (150, 150, 150))):
            cv2.line(img, (0, G[key]), (img.shape[1], G[key]), col, UP)
        t = math.tan(math.radians(slant))
        for a_, b_ in stem_runs:
            xc = 0.5 * (a_ + b_)
            # back from deskewed to image: x = x' - t * (base - y)
            p0 = (int(xc - t * (G["baseline"] - G["x_height_line"])), G["x_height_line"])
            p1 = (int(xc), G["baseline"])
            cv2.line(img, p0, p1, (0, 170, 0), UP)
    img = cv2.resize(img, None, fx=1 / UP * 1.5, fy=1 / UP * 1.5, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(sb.OUT / "sirang_measure.png"), img)


if __name__ == "__main__":
    main()
