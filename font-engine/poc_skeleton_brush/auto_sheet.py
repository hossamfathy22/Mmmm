#!/usr/bin/env python3
"""Sheet of the automatically designed letters (full resolution), with the
eye's verdict under each, next to the uploaded word's letters."""
import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import arabic_anatomy as aa
import auto_design as ad
import compose as cp
import skeleton_brush as sb
from generate_and_build import ribbon_polygon

AR = FontProperties(fname="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
CH = dict(ad.lg.CLASSES)


def build_full(style):
    ctx = ad.Ctx(style, scale=1)
    data = json.loads((sb.OUT / f"auto_design_{style}.json").read_text())
    out = {}
    for t, r in data["auto"].items():
        if "error" in r:
            continue
        part = ad.build(r["genes"], ctx)
        body = ribbon_polygon(part.q, np.clip(part.wl, 1, None), np.clip(part.wr, 1, None))
        # vertical placement chosen by the eye
        minx, miny, maxx, maxy = body.bounds
        dy = ctx.S["base_y"] - r["info"]["bottom_vs_baseline"] * ctx.noon_h - maxy
        from shapely import affinity
        body = affinity.translate(body, 0, dy)
        part = cp.translate(part, (0, dy))
        out[t] = dict(part=part, body=body, info=r["info"], genes=r["genes"], score=r["score"])
    return ctx, out, data


def main():
    style = sys.argv[1] if len(sys.argv) > 1 else "style1"
    ctx, out, data = build_full(style)
    S = ctx.S
    order = [t for t in ad.lg.NAMES if t in out]
    rows = [order[:9], order[9:]]
    fig, axs = plt.subplots(2, 1, figsize=(30, 15))
    for ax, row in zip(axs, rows):
        x = 0.0
        for t in row:
            o = out[t]
            g = o["body"]
            minx, miny, maxx, maxy = g.bounds
            dx = x - minx
            for pg in ([g] if g.geom_type == "Polygon" else list(g.geoms)):
                xs, ys = pg.exterior.xy
                ax.fill(np.array(xs) + dx, ys, color="#222", lw=0)
                for hole in pg.interiors:
                    hx, hy = hole.xy
                    ax.fill(np.array(hx) + dx, hy, color="white", lw=0)
            ax.plot(o["part"].q[:, 0] + dx, o["part"].q[:, 1], color="#ff5a36", lw=1)
            src = " + ".join(f"{s['src']}[{s['a']:.2f}-{s['b']:.2f}]" + ("R" if s["rev"] else "") + ("M" if s["mir"] else "")
                             for s in o["genes"])
            ax.text((minx + maxx) / 2 + dx, S["base_y"] + 560, f"{CH[t]}  {t}\nshape {o['info']['p']:.2f} | path {o['info'].get('p_path', 0):.2f}\n{src}",
                    fontproperties=AR, fontsize=11, ha="center", va="top")
            x += (maxx - minx) + 160
        ax.axhline(S["base_y"], color="#e4572e", lw=0.8)
        ax.set_ylim(S["base_y"] + 800, S["base_y"] - 700)
        ax.set_aspect("equal")
        ax.axis("off")
    axs[0].set_title(f"{style}: letters designed automatically from parts of the uploaded meem / zain / noon "
                     "(red = pen path, line = baseline, p = the eye's confidence that it reads as that letter)", fontsize=14)
    fig.tight_layout()
    fig.savefig(sb.OUT / f"{style}_AUTO_DESIGN.png", dpi=60)
    print("saved", sb.OUT / f"{style}_AUTO_DESIGN.png")


if __name__ == "__main__":
    main()
