#!/usr/bin/env python3
"""Several ways to build each missing letter from the writer's own parts.

The corpus study (recipe_learning.py) showed that WHICH parts build a letter
depends on the style: in most typefaces beh is not the noon, in the uploaded
bubbly style it is. Three uploaded letters cannot decide that for every
letter, so the engine proposes alternatives (each a different design rule)
and the user picks; a pick becomes a rule of the style.

All variants use the writer's real parts (pen path + exact brush widths),
keep each part's angles, and join with smooth bridges (compose.py).
"""
from __future__ import annotations

import json

import numpy as np

import compose as cp
import skeleton_brush as sb


def mirror(p):
    out = p.copy()
    out.q = p.q.copy()
    out.q[:, 0] = 2 * p.q[0, 0] - p.q[:, 0]
    out.wl, out.wr = p.wr.copy(), p.wl.copy()   # mirroring swaps the sides of the brush
    return out


def reverse(p):
    out = cp.Part(p.q[::-1], p.wr[::-1], p.wl[::-1], p.src, p.end_term, p.start_term)
    out.labels = p.labels[::-1]
    return out


def variants(S, cap="round"):
    st = S["strokes"]
    N = cp.split_noon(st["noon"], S["base_y"])
    Z = cp.split_zain(st["zain"])
    noon = cp.sub(st["noon"], 0, len(st["noon"][0]) - 1, "noon")
    zh, zn, zt = Z["head"], Z["neck"], Z["tail"]
    tooth = N["tooth"]
    mq = st["meem"][0]
    meem_end = cp.sub(st["meem"], int(0.72 * len(mq)), len(mq) - 1, "meem.end")
    noon_h = np.ptp(st["noon"][0][:, 1])

    def stem(head, height):
        s0 = cp.chain([head, zn])
        f = height * noon_h / max(np.ptp(s0.q[:, 1]), 1)
        return cp.chain([head, cp.scale(zn, f)])

    V = {}
    V["alef"] = [
        ("A: zain head + long neck", cp.end_cap(stem(zh, 1.65), cap, "end"), "base"),
        ("B: noon tooth start MIRRORED + long neck (user's idea)", cp.end_cap(stem(mirror(tooth), 1.65), cap, "end"), "base"),
        ("C: noon tooth start + long neck", cp.end_cap(stem(tooth, 1.65), cap, "end"), "base"),
    ]
    V["dal"] = [(f"{lab}: zain head + neck x{f} + zain tail", cp.chain([zh, cp.scale(zn, f), zt]), "base")
                for lab, f in (("A", 0.4), ("B", 0.55), ("C", 0.7))]
    V["lam"] = [
        ("A: zain head + neck to alef height + noon bowl", cp.chain([zh, cp.scale(zn, 1.0), N["bowl"]]), "bowl"),
        ("B: mirrored noon tooth + long neck + noon bowl", cp.chain([stem(mirror(tooth), 1.45), N["bowl"]]), "bowl"),
    ]
    V["alefmaksura"] = [
        ("A: zain head + short neck + noon bowl", cp.chain([zh, cp.scale(zn, 0.45), N["bowl"]]), "bowl"),
        ("B: mirrored noon tooth + short neck + noon bowl", cp.chain([mirror(tooth), cp.scale(zn, 0.45), N["bowl"]]), "bowl"),
    ]
    t_noon = cp.scale(tooth, 0.8)
    t_zain = cp.scale(cp.chain([zh, reverse(zh)], bridge=20), 0.8)
    t_meem = cp.scale(cp.chain([meem_end, reverse(meem_end)], bridge=20), 0.7)
    V["seen"] = [
        ("A: teeth = noon's tooth", cp.chain([cp.end_cap(t_noon, cap, "start"), t_noon, noon], bridge=45), "bowl"),
        ("B: teeth = zain's head (up and back)", cp.chain([t_zain, t_zain, noon], bridge=40), "bowl"),
        ("C: teeth = meem's ending (up and back)", cp.chain([t_meem, t_meem, noon], bridge=40), "bowl"),
    ]
    return V


def draw(S, V, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from generate_and_build import ribbon_polygon
    rows = list(V)
    fig, axs = plt.subplots(len(rows), 1, figsize=(18, 4.2 * len(rows)))
    for ax, letter in zip(axs, rows):
        x = 0
        for lab, part, mode in V[letter]:
            part = cp.place(cp.resample_part(part), mode, S)
            g = ribbon_polygon(part.q, np.clip(part.wl, 1, None), np.clip(part.wr, 1, None))
            minx, miny, maxx, maxy = g.bounds
            dx = x - minx
            polys = [g] if g.geom_type == "Polygon" else list(g.geoms)
            for pg in polys:
                xs, ys = pg.exterior.xy
                ax.fill(np.array(xs) + dx, ys, color="#e6e6e6", lw=0)
            ax.plot(part.q[:, 0] + dx, part.q[:, 1], color="#e4572e", lw=1)
            ax.text((minx + maxx) / 2 + dx, S["base_y"] + 300, lab, ha="center", fontsize=9)
            x += (maxx - minx) + 250
        ax.axhline(S["base_y"], color="#999", lw=0.6)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(letter, fontsize=14, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def main():
    S = cp.load_style("style1")
    V = variants(S)
    draw(S, V, sb.OUT / "style1_VARIANTS.png")
    (sb.OUT / "variants.json").write_text(json.dumps({k: [lab for lab, _, _ in v] for k, v in V.items()},
                                                     ensure_ascii=False, indent=1))
    print("saved", sb.OUT / "style1_VARIANTS.png")


if __name__ == "__main__":
    main()
