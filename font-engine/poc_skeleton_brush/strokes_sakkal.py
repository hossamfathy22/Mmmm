#!/usr/bin/env python3
"""Strokes as in Sakkal's chart: every tooth / stem is its OWN pen stroke.

The single continuous skeleton goes up a tooth and back down (a hairpin).
Sakkal's arrows show something else: each tooth starts at a dot at its TOP,
comes down, turns left along the base and stops where the next tooth stands;
the pen is lifted and the next stroke starts at the top of the next tooth.

  1. split   : find the hairpins of the Naskh skeleton whose tip is above
               their foot; cut there. Stroke i ends at the foot, stroke i+1
               starts at the tip and comes down the middle of the tooth.
  2. style   : every stroke is restyled as one curve (curve_style.py). A
               stroke that starts at a tooth gets the writer's tooth entry
               (from the uploaded noon); only the last stroke gets the exit.
  3. assemble: each stroke is placed so that the foot of its tooth sits where
               the previous stroke ended.
  4. paint   : each stroke with the writer's brush; strokes that end inside
               the letter keep their full width there (no terminal).
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d

import auto_design as ad
import compose as cp
import curve_style as cs
import skeleton_brush as sb
from ductus import DUCTUS, orient


def split_strokes(N, tol):
    """N: dense path (1 px). Returns [(points, foot_index)], foot_index = where
    the stroke meets the end of the previous one (None for the first)."""
    d = np.diff(N, axis=0)
    th = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    w = int(max(6, tol))
    turn = np.abs(th[w:] - th[:-w])
    cand = [i + w // 2 for i in range(1, len(turn) - 1) if turn[i] > np.radians(140) and turn[i] >= turn[i - 1]
            and turn[i] >= turn[i + 1]]
    apexes = []
    for m in cand:
        if not apexes or m - apexes[-1] > 2 * w:
            apexes.append(m)
    strokes, start, lead = [], 0, None
    for m in apexes:
        k = 0
        while m - k - 1 > start and m + k + 1 < len(N) and np.linalg.norm(N[m - k - 1] - N[m + k + 1]) < 1.6 * tol:
            k += 1
        if k < 1.5 * tol or N[m, 1] > N[m - k, 1] - 0.5 * tol:      # not a tooth standing above its foot
            continue
        body = N[start:m - k + 1]
        strokes.append((body if lead is None else np.vstack([lead, body[1:]]), None if lead is None else len(lead) - 1))
        lead = np.array([(N[m + j] + N[m - j]) / 2 for j in range(k + 1)])     # down the middle of the tooth
        start = m + k
    body = N[start:]
    strokes.append((body if lead is None else np.vstack([lead, body[1:]]), None if lead is None else len(lead) - 1))
    return strokes


def brush(C, G, key, open_end):
    q, wl, wr = C.R.S["strokes"][key]
    w = (wl + wr) / 2
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
    L = s[-1]
    s_new = np.r_[0, np.cumsum(np.linalg.norm(np.diff(G, axis=0), axis=1))]
    Ln = s_new[-1]
    g = min(1.2 * np.median(w), 0.35 * min(L, Ln))
    if open_end:      # start terminal + body only: full width where it joins the next stroke
        sm = np.where(s_new < g, s_new, g + (s_new - g) * (0.5 * L - g) / max(Ln - g, 1e-6))
    else:
        sm = np.where(s_new < g, s_new, np.where(s_new > Ln - g, L - (Ln - s_new),
                      g + (s_new - g) * (L - 2 * g) / max(Ln - 2 * g, 1e-6)))
    wn = np.clip(gaussian_filter1d(np.interp(sm, s, w), 5, mode="nearest"), 1, None)
    return cp.Part(G, wn, wn.copy(), key, True, not open_end)


def build(C, form, letter, thick):
    R = C.R
    d = DUCTUS[letter]
    N = cs.dense(orient(R.ref[form] * R.k, letter))
    parts = split_strokes(N, max(6.0, thick))
    zeros = np.zeros(C.Ls)
    G_list, N_list, end_prev = [], [], None
    for i, (Ns, foot) in enumerate(parts):
        Ns = cs.dense(Ns)
        last = i == len(parts) - 1
        entry = C.entry_by.get(d["entry"] if i == 0 else "noon", C.entry)
        exit_ = (zeros if d["exit"] is None else C.exit_by.get(d["exit"], C.exit)) if last else zeros
        L_s = np.linalg.norm(np.diff(Ns, axis=0), axis=1).sum()
        tooth = (not last) and L_s < 1.5 * C.unit
        if tooth:
            # a tooth: too short for the entry/exit operator. Keep its Naskh shape,
            # softened, and give it the height of the writer's own tooth (noon).
            # Sakkal's tooth stroke: from the dot at the top straight down, then a turn
            # to the left along the base as far as the next tooth. Height = the writer's
            # own tooth (noon); travel along the base = one brush width, so that the
            # teeth stay apart under this writer's heavy brush.
            qn, wln, wrn = R.S["strokes"]["noon"]
            tooth_h = max(R.base - qn[:, 1].min(), 0.3 * C.unit)
            sp = 1.15 * float(np.median(wln + wrn))
            t_ = np.linspace(0, 1, 80)[:, None]
            P0, P1, P2, P3 = np.array([0, -tooth_h]), np.array([0, -0.25 * tooth_h]), np.array([-0.35 * sp, 0]), np.array([-sp, 0])
            G = cs.dense((1 - t_) ** 3 * P0 + 3 * (1 - t_) ** 2 * t_ * P1 + 3 * (1 - t_) * t_ ** 2 * P2 + t_ ** 3 * P3)
            foot = 0                      # the next stroke simply starts where this one ended
        else:
            G = C.apply_path(Ns, entry, exit_)
        if end_prev is None:
            G = G - G[0]
        else:
            if tooth:      # top of the tooth directly above the end of the previous stroke
                G = G + (end_prev - np.array([G[0, 0], G[:, 1].max()]))
            else:          # the point where this stroke comes down onto the base
                fi = min(int(round((foot or 0) * (len(G) - 1) / max(len(Ns) - 1, 1))), len(G) - 1)
                G = G + (end_prev - G[fi])
        end_prev = G[-1]
        G_list.append(G)
        N_list.append(Ns)
    allG = np.vstack(G_list)
    shift = np.array([N[:, 0].mean() - allG[:, 0].mean(), N[:, 1].max() - allG[:, 1].max() + R.base])
    G_list = [G + shift for G in G_list]
    N_list = [Ns + [0, R.base] for Ns in N_list]
    strokes = []
    for i, G in enumerate(G_list):
        last = i == len(G_list) - 1
        key = (d["exit"] or d["entry"]) if last else "noon"
        strokes.append(brush(C, G, key if key in C.R.S["strokes"] else "noon", open_end=not last))
    return strokes, N_list


def shape_score(strokes, target, ctxq, eye):
    import cv2
    qs = [(p.q[::4] / 4, p.wl[::4] / 4, p.wr[::4] / 4) for p in strokes]
    lo = np.min([q.min(0) - w.max() - 3 for q, w, _ in qs], axis=0)
    hi = np.max([q.max(0) + w.max() + 3 for q, w, _ in qs], axis=0)
    canvas = np.zeros((int(hi[1] - lo[1]) + 1, int(hi[0] - lo[0]) + 1), np.uint8)
    for q, wl, wr in qs:
        sb.render_ribbon(canvas, q - lo, wl, wr)
    m = canvas > 127
    ys = np.nonzero(m)[0]
    y1 = ys.max() + 1
    F = np.array([ad.lg.featurize(m, y1 + t * ctxq.noon_h, ctxq.noon_h) for t in ad.OFFSETS], np.float32)
    return float(eye.proba(F)[:, ad.lg.NAMES.index(target)].max())


def main():
    style = sys.argv[1] if len(sys.argv) > 1 else "style1"
    C = cs.CurveStyle(style)
    data = json.loads((sb.OUT / "skeleton_corpus.json").read_text())[cs.nr.REF_FONT]
    ctxq = ad.Ctx(style)
    eye = ad.Eye(ctxq)
    prev = json.loads((sb.OUT / f"curve_style_{style}.json").read_text())["letters"]
    out, built = {}, {}
    for t in ad.TARGETS:
        form = {"yeh": "alefmaksura.isol"}.get(t, t + ".isol")
        strokes, N_list = build(C, form, t, data[form]["thickness"] * C.R.k)
        p = shape_score(strokes, t, ctxq, eye)
        out[t] = {"strokes_found": len(strokes), "strokes_in_sakkal": DUCTUS[t]["strokes"], "shape_p": round(p, 2),
                  "shape_p_single_path": prev[t]["shape_p"]}
        built[t] = (strokes, N_list)
        print(f"{t:6s} strokes {len(strokes)} (Sakkal {DUCTUS[t]['strokes']})  shape {p:.2f}  (single path: {prev[t]['shape_p']})")
    (sb.OUT / f"strokes_sakkal_{style}.json").write_text(json.dumps(out, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from generate_and_build import ribbon_polygon
    from shapely.ops import unary_union
    AR = FontProperties(fname="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    cols = ["#6a2fb5", "#e4572e", "#1f9d55", "#2b7bba"]
    fig, axs = plt.subplots(3, 1, figsize=(34, 17))
    x = 0.0
    for t, (strokes, N_list) in built.items():
        g = unary_union([ribbon_polygon(p.q, p.wl, p.wr) for p in strokes])
        lo = min(g.bounds[0], min(n[:, 0].min() for n in N_list))
        hi = max(g.bounds[2], max(n[:, 0].max() for n in N_list))
        dx = x - lo
        for i, (p, n) in enumerate(zip(strokes, N_list)):
            for ax, P in ((axs[0], n), (axs[1], p.q)):
                ax.plot(P[:, 0] + dx, P[:, 1], color=cols[i % 4], lw=2.5)
                ax.plot(P[0, 0] + dx, P[0, 1], "o", color=cols[i % 4], ms=10)
                k = len(P) - 1
                ax.annotate("", xy=(P[k, 0] + dx, P[k, 1]), xytext=(P[max(k - 10, 0), 0] + dx, P[max(k - 10, 0), 1]),
                            arrowprops=dict(arrowstyle="-|>", color=cols[i % 4], lw=2))
        for pg in ([g] if g.geom_type == "Polygon" else list(g.geoms)):
            px, py = pg.exterior.xy
            axs[2].fill(np.array(px) + dx, py, color="#ff5a36", lw=0)
            for h in pg.interiors:
                hx, hy = h.xy
                axs[2].fill(np.array(hx) + dx, hy, color="white", lw=0)
        axs[2].text((lo + hi) / 2 + dx, C.R.base + 520, f"{ad.lg.CLASSES[t]}\n{len(strokes)} stroke(s) | shape {out[t]['shape_p']}",
                    fontproperties=AR, fontsize=13, ha="center", va="top")
        x += (hi - lo) + 150
    for ax, title in zip(axs, ("Naskh skeleton split into pen strokes (dot = where the pen starts, arrow = where it stops)",
                               "Each stroke restyled", "Painted with the writer's brush")):
        ax.axhline(C.R.base, color="#bbb", lw=0.8)
        ax.set_xlim(-50, x)
        ax.set_ylim(C.R.base + 700, C.R.base - 700)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=15, loc="left")
    fig.tight_layout()
    fig.savefig(sb.OUT / f"{style}_STROKES_SAKKAL.png", dpi=55)
    print("saved", sb.OUT / f"{style}_STROKES_SAKKAL.png")


if __name__ == "__main__":
    main()
