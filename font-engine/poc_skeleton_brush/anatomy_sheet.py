#!/usr/bin/env python3
"""Anatomy sheet: the uploaded word and the generated letters on Zoghbi's
reference levels (Sky / Tooth-Eye / Base / Earth), with every pen path and
its nodes drawn like a type designer's working file."""
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch

import skeleton_brush as sb
from arabic_anatomy import LEVELS

AR = FontProperties(fname="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")  # has Arabic presentation forms
ar = lambda t: t  # matplotlib >= 3.11 shapes and orders Arabic itself
LEVEL_AR = {k: a for k, a, _ in LEVELS}
LEVEL_AR["earth2"] = "أرض ٢"


def to_mpath(contours, dx=0.0):
    verts, codes = [], []
    for c in contours:
        for cmd in c:
            if cmd[0] == "moveTo":
                verts.append((cmd[1][0] + dx, cmd[1][1])); codes.append(MPath.MOVETO)
            elif cmd[0] == "curveTo":
                for pt in cmd[1:]:
                    verts.append((pt[0] + dx, pt[1])); codes.append(MPath.CURVE4)
            elif cmd[0] == "lineTo":
                verts.append((cmd[1][0] + dx, cmd[1][1])); codes.append(MPath.LINETO)
            elif cmd[0] == "closePath":
                verts.append(verts[-1]); codes.append(MPath.CLOSEPOLY)
    return MPath(verts, codes)


def draw_pen(ax, segs, dx=0.0, col="#ff5a36"):
    for P in segs:
        P = np.array(P) + [dx, 0]
        t = np.linspace(0, 1, 60)[:, None]
        c = (1 - t) ** 3 * P[0] + 3 * (1 - t) ** 2 * t * P[1] + 3 * (1 - t) * t * t * P[2] + t ** 3 * P[3]
        ax.plot(c[:, 0], c[:, 1], color=col, lw=1.6)
        ax.plot([P[0][0], P[1][0]], [P[0][1], P[1][1]], color="#3a86ff", lw=0.8)
        ax.plot([P[3][0], P[2][0]], [P[3][1], P[2][1]], color="#3a86ff", lw=0.8)
        ax.plot([P[1][0], P[2][0]], [P[1][1], P[2][1]], "o", color="#3a86ff", ms=2.5)
    for P in segs + [segs[-1]]:
        pass
    pts = [np.array(segs[0][0]) + [dx, 0]] + [np.array(P[3]) + [dx, 0] for P in segs]
    for x, y in pts:
        ax.plot(x, y, "s", color="white", mec="black", ms=4.5)


def main():
    g = json.loads((sb.OUT / "glyphs_px.json").read_text())
    lv = g["levels_y"]
    order = ["uni0633", "uni0627", "uni0644", "uni062F", "uni0649", "uni062B", "uni062A", "uni0628", "uni0631"]
    word = ["uni0646", "uni0632.fina", "uni0645.init"]
    fig, ax = plt.subplots(figsize=(26, 9))
    x = 0.0
    placed = []
    for name in order + word:
        cont = g["contours"][name]
        xs = [pt[0] for c in cont for cmd in c for pt in cmd[1:] if isinstance(pt, list)]
        x0, x1 = min(xs), max(xs)
        dx = x - x0
        ax.add_patch(PathPatch(to_mpath(cont, dx), fc="#d9d9d9" if name in g["centerlines"] else "#bfbfbf",
                               ec="none"))
        placed.append((name, dx))
        x += (x1 - x0) + 120
    for name, dx in placed:
        if name in g["centerlines"]:
            draw_pen(ax, g["centerlines"][name], dx)
    # word pen paths (from the image) on their glyphs
    wc = [s for s in g["word_centerlines"] if s]
    word_dx = {n: d for n, d in placed if n in word}
    # word strokes are in image coordinates; their glyph contours too -> same dx
    reh_dx = dict(placed)["uni0631"]
    for segs, name, dx in [(wc[0], "uni0645.init", word_dx["uni0645.init"]), (wc[2], "uni0632.fina", word_dx["uni0632.fina"]),
                           (wc[3], "uni0646", word_dx["uni0646"]), (wc[2], "uni0631", reh_dx)]:
        draw_pen(ax, segs, dx, col="#e4572e")
    last = None
    for k, y in sorted(lv.items(), key=lambda kv: kv[1]):
        ax.axhline(y, color="#e4572e", lw=0.8, alpha=0.8)
        ty = y if last is None or y - last > 45 else last + 45   # keep close labels apart
        ax.text(-60, ty, ar(LEVEL_AR.get(k, k)), fontproperties=AR, fontsize=13, color="#e4572e",
                ha="right", va="center")
        last = ty
    ax.set_xlim(-520, x)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Generated letters (left) and the uploaded word (right) on the anatomy levels — "
                 "one smooth pen path per letter, nodes = start, end and one per bend", fontsize=13)
    fig.tight_layout()
    fig.savefig(sb.OUT / "ANATOMY_SHEET.png", dpi=90)
    print("saved", sb.OUT / "ANATOMY_SHEET.png")


if __name__ == "__main__":
    main()
