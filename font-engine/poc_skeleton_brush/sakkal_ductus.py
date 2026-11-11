#!/usr/bin/env python3
"""Read the pen strokes straight from Mamoun Sakkal's chart (vector PDF).

The chart (c) Dr. Mamoun Sakkal, sakkal.com, 2007/2016 — "can be printed and
distributed without charge, contents not to be altered". Only measurements
(stroke counts, start points, directions) are extracted here; the chart
itself is not redistributed in the outputs.

In the PDF the arrows are purple stroked paths, the stroke starts are purple
dots and the stroke ends purple arrowheads; every table row has an orange
baseline. For each letter row and each arrow diagram this script recovers:

  strokes   ordered list of pen strokes (polyline, page units)
  start     the dot the stroke starts from (if drawn)
  end       the arrowhead where it stops
  direction overall direction of travel, and for every stroke whether it
            starts at its TOP (as teeth and stems do)

Output: output/sakkal_ductus.json and a check picture.
"""
from __future__ import annotations

import json

import numpy as np
import pymupdf

import skeleton_brush as sb

PDF = sb.HERE / "input" / "sakkal_chart.pdf"
PURPLE = (0.41, 0.17, 0.57)
ORANGE = (0.97, 0.54, 0.12)
ROWS = ["alef", "beh", "teh", "theh", "jeem", "hah", "khah", "dal", "thal", "reh", "zain", "seen", "sheen", "sad",
        "dad", "tah", "zah", "ain", "ghain", "feh", "qaf", "kaf", "lam", "meem", "noon", "heh", "waw", "yeh"]


def close(c, ref):
    return c is not None and all(abs(a - b) < 0.03 for a, b in zip(c, ref))


def flatten(items):
    pts = []
    for it in items:
        if it[0] == "l":
            pts += [it[1], it[2]]
        elif it[0] == "c":
            p0, p1, p2, p3 = [np.array([p.x, p.y]) for p in it[1:5]]
            for t in np.linspace(0, 1, 12):
                pts.append(tuple((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t * t * p2 + t ** 3 * p3))
    out = []
    for p in pts:
        q = (float(p[0]), float(p[1])) if not hasattr(p, "x") else (p.x, p.y)
        if not out or abs(q[0] - out[-1][0]) + abs(q[1] - out[-1][1]) > 1e-3:
            out.append(q)
    return np.array(out)


def parse():
    page = pymupdf.open(PDF)[0]
    strokes, dots, heads, base = [], [], [], []
    for d in page.get_drawings():
        col, fill = d.get("color"), d.get("fill")
        r = d["rect"]
        if close(col, PURPLE) and fill is None:
            P = flatten(d["items"])
            if len(P) >= 2:
                strokes.append(P)
        elif close(fill, PURPLE):
            n_curves = sum(1 for it in d["items"] if it[0] == "c")
            c = np.array([(r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2])
            (dots if n_curves >= 3 else heads).append(c)
        elif close(col, ORANGE) and r.width > 100:
            base.append((r.y0 + r.y1) / 2)
    base = sorted(set(round(b, 1) for b in base))
    return page, strokes, np.array(dots), np.array(heads), base


def build():
    page, strokes, dots, heads, base = parse()
    rows = base[-len(ROWS):] if len(base) >= len(ROWS) else base
    gap = float(np.median(np.diff(rows)))
    out = {}
    for P in strokes:
        cy = P[:, 1].mean()
        # a diagram sits above the baseline of its row
        i = int(np.argmin([abs((b - 0.45 * gap) - cy) for b in rows]))
        letter = ROWS[i] if i < len(ROWS) else f"row{i}"
        a, b = P[0], P[-1]
        da = np.linalg.norm(dots - a, axis=1).min() if len(dots) else 9
        db = np.linalg.norm(dots - b, axis=1).min() if len(dots) else 9
        ha = np.linalg.norm(heads - a, axis=1).min() if len(heads) else 9
        hb = np.linalg.norm(heads - b, axis=1).min() if len(heads) else 9
        # orient: start at the dot, or away from the arrowhead
        if (db < da and db < 3) or (ha < hb and ha < 3 and not da < 3):
            P = P[::-1]
            da, hb = db, ha
        out.setdefault(letter, []).append({"points": P.round(2).tolist(), "starts_at_dot": bool(da < 3),
                                           "ends_at_arrow": bool(hb < 3), "baseline_y": rows[i]})
    # group the strokes of a row into diagrams (clusters along x), order right to left
    res = {}
    for letter, ss in out.items():
        ss.sort(key=lambda s: -np.array(s["points"])[:, 0].max())
        groups, cur, last_min = [], [], None
        for s in ss:
            P = np.array(s["points"])
            if cur and last_min - P[:, 0].max() > 14:
                groups.append(cur)
                cur = []
            cur.append(s)
            last_min = P[:, 0].min() if last_min is None or not cur[:-1] else min(last_min, P[:, 0].min())
        if cur:
            groups.append(cur)
        diagrams = []
        for g in groups:
            info = []
            for s in g:
                P = np.array(s["points"])
                dx, dy = P[-1] - P[0]
                info.append({"starts_at_dot": s["starts_at_dot"], "starts_at_top": bool(P[0, 1] <= P[:, 1].min() + 0.25 * max(np.ptp(P[:, 1]), 1e-6)),
                             "goes_left": bool(dx < 0), "goes_down": bool(dy > 0),
                             "length": round(float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()), 1),
                             "points": s["points"]})
            xs = np.concatenate([np.array(s["points"])[:, 0] for s in g])
            diagrams.append({"x_center": round(float(xs.mean()), 1), "n_pieces": len(g),
                             "n_strokes": int(sum(1 for s in g if s["starts_at_dot"])) or 1, "pieces": info})
        res[letter] = diagrams
    return page, res, {"rows": rows, "n_strokes_paths": len(strokes), "n_dots": len(dots), "n_arrowheads": len(heads)}


def main():
    page, res, meta = build()
    (sb.OUT / "sakkal_ductus.json").write_text(json.dumps({"meta": meta, "letters": res}, indent=0))
    print(meta["n_strokes_paths"], "arrow paths,", meta["n_dots"], "start dots,", meta["n_arrowheads"], "arrowheads,", len(meta["rows"]), "rows")
    from ductus import DUCTUS
    for letter in ROWS:
        ds = res.get(letter, [])
        if not ds:
            continue
        mine = DUCTUS.get(letter, {}).get("strokes")
        print(f"{letter:6s} diagrams {len(ds)} | strokes per diagram (dots): {[d['n_strokes'] for d in ds]} | my table: {mine}")
    # check picture: page render with recovered strokes (green start, red end)
    import cv2
    z = 2.0
    pix = page.get_pixmap(matrix=pymupdf.Matrix(z, z))
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].copy()
    for letter, ds in res.items():
        for d in ds:
            for s in d["pieces"]:
                P = (np.array(s["points"]) * z).astype(int)
                cv2.circle(img, tuple(P[0]), 5, (0, 170, 0), -1)
                cv2.circle(img, tuple(P[-1]), 4, (220, 0, 0), -1)
    crop = img[int(150 * z):int(1200 * z), int(430 * z):int(770 * z)]
    cv2.imwrite(str(sb.OUT / "sakkal_ductus_check.png"), crop[:, :, ::-1])


if __name__ == "__main__":
    main()
