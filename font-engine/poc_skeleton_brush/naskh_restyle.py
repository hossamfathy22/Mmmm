#!/usr/bin/env python3
"""Naskh as the reference, the writer's letters as the style (the user's diagram).

  Naskh reh  ->  writer's reh      (same letter in both: this pair TEACHES)
  Naskh seen ->  ?                 (new letter: the lesson is APPLIED)

1. Every Naskh pen path (Amiri) is cut into ARCS at its natural turning
   points (where the pen changes between rightwards/leftwards or up/down).
2. For the letters the writer gave (noon, zain = reh with a dot, meem), each
   Naskh arc is matched to the piece of the writer's stroke that plays the
   same role (DTW). Result: a library of rules
        "an arc like THIS in Naskh is drawn like THAT by this writer"
   where THAT is a real piece of the writer's stroke, with its brush widths.
3. A new letter: its Naskh arcs are looked up in the library (nearest arc by
   direction, turning, length, height, start/middle/end role; mirrored and
   reversed pieces allowed), each chosen piece is fitted to the arc's chord,
   and the pieces are joined smoothly. Structure comes from Naskh, shape and
   brush from the writer.
4. Both eyes (legibility.py, PathEye) judge the result.

Honest test: hide one of the writer's letters, rebuild it from the other two.
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

import auto_design as ad
import compose as cp
import skeleton_brush as sb
import style_deviation as sd

lg = ad.lg
KNOWN = {"noon": "noon.isol", "zain": "zain.fina", "meem": "meem.init"}
REF_FONT = "Amiri"


def arcs_of(P, min_frac=0.1):
    """Cut a path at x/y extrema; merge very short arcs."""
    d = np.diff(P, axis=0)
    from scipy.ndimage import gaussian_filter1d
    d = np.stack([gaussian_filter1d(d[:, 0], 1.5), gaussian_filter1d(d[:, 1], 1.5)], 1)
    cuts = [0]
    for i in range(1, len(d)):
        if np.sign(d[i - 1, 0]) != np.sign(d[i, 0]) or np.sign(d[i - 1, 1]) != np.sign(d[i, 1]):
            cuts.append(i)
    cuts.append(len(P) - 1)
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    out = [cuts[0]]
    for c in cuts[1:]:
        if L[c] - L[out[-1]] >= min_frac * L[-1] or c == cuts[-1]:
            out.append(c)
    if len(out) > 2 and L[out[-1]] - L[out[-2]] < min_frac * L[-1]:
        out.pop(-2)
    return [(a, b) for a, b in zip(out[:-1], out[1:]) if b - a >= 2]


def descriptor(P, a, b, n_arcs, k, unit):
    seg = P[a:b + 1]
    d = np.diff(seg, axis=0)
    th = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    chord = seg[-1] - seg[0]
    cl = np.linalg.norm(chord) + 1e-9
    L = np.linalg.norm(d, axis=1).sum()
    return np.array([chord[0] / cl, chord[1] / cl,
                     math.cos(th[0]), math.sin(th[0]), math.cos(th[-1]), math.sin(th[-1]),
                     (th[-1] - th[0]) / math.pi,
                     0.7 * L / unit, 0.5 * seg[:, 1].mean() / unit,
                     0.8 * (k == 0), 0.8 * (k == n_arcs - 1)])


class Restyler:
    def __init__(self, style="style1", exclude=()):
        self.ctx = ad.Ctx(style, scale=1)
        self.S = self.ctx.S
        data = json.loads((sb.OUT / "skeleton_corpus.json").read_text())[REF_FONT]
        self.ref = {k: np.array(v["path"]) for k, v in data.items() if isinstance(v, dict)}
        hn = np.ptp(self.ref["noon.isol"][:, 1])
        self.unit = float(np.ptp(self.S["strokes"]["noon"][0][:, 1]))       # writer's noon path height (px)
        self.k = self.unit / hn                                              # Naskh nuqta -> writer px
        self.base = self.S["base_y"]
        self.lib = []
        for key, form in KNOWN.items():
            if key in exclude:
                continue
            q, wl, wr = self.S["strokes"][key]
            N = self.ref[form] * self.k
            U64 = self._r64(q)
            Uc = U64 - [U64[:, 0].mean(), self.base]
            N = sd.orient_like(N, Uc)
            match, _ = sd.dtw_cost_match(N, Uc)
            arcs = arcs_of(N)
            for k, (a, b) in enumerate(arcs):
                j0 = int(round(min(match[a]) * (len(q) - 1) / 63))
                j1 = int(round(max(match[b]) * (len(q) - 1) / 63))
                if j1 - j0 < 12:
                    continue
                piece = cp.sub((q, wl, wr), j0, j1, f"{key}.arc{k}")
                for mir in (False, True):
                    for rev in (False, True):
                        Nv = N[a:b + 1].copy()
                        pv = piece
                        kk, na = k, len(arcs)
                        if rev:
                            Nv = Nv[::-1]
                            pv = ad.reverse(pv)
                            kk = na - 1 - k
                        if mir:
                            Nv[:, 0] = -Nv[:, 0]
                            pv = ad.transform(pv, True, 1.0, 1.0, 0.0)
                        desc = descriptor(Nv, 0, len(Nv) - 1, na, kk, self.unit)
                        self.lib.append(dict(desc=desc, piece=pv, src=f"{key}.arc{k}" + ("R" if rev else "") + ("M" if mir else ""),
                                             plain=not (mir or rev)))

    @staticmethod
    def _r64(q):
        L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
        t = np.linspace(0, L[-1], 64)
        return np.stack([np.interp(t, L, q[:, 0]), np.interp(t, L, q[:, 1])], 1)

    def fit_piece(self, piece, chord):
        c0 = piece.q[-1] - piece.q[0]
        s = np.linalg.norm(chord) / (np.linalg.norm(c0) + 1e-9)
        rot = math.degrees(math.atan2(chord[1], chord[0]) - math.atan2(c0[1], c0[0]))
        rot = (rot + 180) % 360 - 180
        out = ad.transform(piece, False, s, s, rot)
        f = float(np.clip(s ** 0.5, 0.55, 1.15))          # the brush follows the size only partly
        out.wl, out.wr = out.wl * f, out.wr * f
        return out, s, rot

    def generate(self, form):
        N = self.ref[form] * self.k
        arcs = arcs_of(N)
        parts, used = [], []
        for k, (a, b) in enumerate(arcs):
            d = descriptor(N, a, b, len(arcs), k, self.unit)
            best = None
            for e in self.lib:
                piece, s, rot = None, None, None
                dist = np.linalg.norm(e["desc"] - d) + (0.0 if e["plain"] else 0.15)
                if best is None or dist < best[0]:
                    best = (dist, e)
            piece, s, rot = self.fit_piece(best[1]["piece"], N[b] - N[a])
            parts.append(piece)
            used.append(f"{best[1]['src']} x{s:.2f} rot{rot:+.0f}")
        out = parts[0]
        for p in parts[1:]:
            br = int(min(self.ctx.bridge, len(out.q) // 3, len(p.q) // 3))
            out = cp.attach(out, p, max(br, 2))
        kk = int(min(2 * self.ctx.bridge, len(out.q) // 3))
        ramp = np.linspace(1, 0, kk)
        for end in ("start", "end"):
            sl = slice(0, kk) if end == "start" else slice(-kk, None)
            r = ramp if end == "start" else ramp[::-1]
            mid = (out.wl[sl] + out.wr[sl]) / 2
            out.wl[sl] = r * mid + (1 - r) * out.wl[sl]
            out.wr[sl] = r * mid + (1 - r) * out.wr[sl]
            out = cp.end_cap(out, self.ctx.cap, end)
        # one clean ribbon: the brush centred on the new path and its width smoothed
        # across the pieces (rotated pieces otherwise leave lumps at the joins)
        from scipy.ndimage import gaussian_filter1d
        w = gaussian_filter1d((out.wl + out.wr) / 2, 12.0, mode="nearest")
        cap_n = int(min(len(w) // 4, 1.2 * np.median(w)))
        taper = np.ones(len(w))
        t_ = np.linspace(0, 1, cap_n)
        taper[:cap_n] = np.sqrt(1 - (1 - t_) ** 2)
        taper[-cap_n:] = np.sqrt(1 - (1 - t_[::-1]) ** 2)
        out.wl = out.wr = np.maximum(w * taper, 1.0)
        # put it where the Naskh letter sits (x free, y from the baseline)
        tgt = N[:, 1].max() + self.base
        out = cp.translate(out, (0, tgt - out.q[:, 1].max()))
        return out, used, N + [0, self.base]


def quarter(p):
    return cp.resample_part(cp.Part(p.q[::4] / 4, p.wl[::4] / 4, p.wr[::4] / 4, p.src, True, True))


def path_err(A, B, unit):
    a = sd.sbl.path_err.__wrapped__ if hasattr(sd.sbl.path_err, "__wrapped__") else sd.sbl.path_err
    A64, B64 = Restyler._r64(A), Restyler._r64(B)
    A64 = A64 - [A64[:, 0].mean(), 0]
    B64 = B64 - [B64[:, 0].mean(), 0]
    return a(A64, B64) / unit


def main():
    style = sys.argv[1] if len(sys.argv) > 1 else "style1"
    # --- honest test: rebuild each of the writer's letters from the other two
    loo = {}
    for key, form in (("noon", "noon.isol"), ("zain", "reh.isol"), ("meem", "meem.init")):
        R = Restyler(style, exclude=(key,))
        part, used, N = R.generate(form if key != "zain" else "zain.fina")
        truth = R.S["strokes"][key][0]
        loo[key] = {"restyled_err_noon_heights": round(path_err(part.q, truth, R.unit), 3),
                    "plain_naskh_err_noon_heights": round(path_err(N, truth, R.unit), 3), "pieces": used}
        print("LOO", key, loo[key])
    # --- all letters with the full library, judged by both eyes
    R = Restyler(style)
    ctxq = ad.Ctx(style)
    eye, peye = ad.Eye(ctxq), ad.PathEye(ctxq)
    auto = json.loads((sb.OUT / f"auto_design_{style}.json").read_text())["auto"] if (sb.OUT / f"auto_design_{style}.json").exists() else {}
    res, parts = {}, {}
    for t in ad.TARGETS:
        form = {"yeh": "alefmaksura.isol"}.get(t, t + ".isol")
        try:
            part, used, N = R.generate(form)
        except Exception as ex:
            res[t] = {"error": str(ex)}
            continue
        (s, info), = ad.score_batch([quarter(part)], [None], t, ctxq, eye, peye)
        res[t] = {"score": round(float(s), 2), "shape_p": round(info["p"], 2) if info else None,
                  "path_p": round(info["p_path"], 2) if info else None, "pieces": used,
                  "auto_search_score": auto.get(t, {}).get("score")}
        parts[t] = (part, N)
        print(f"{t:6s} restyle score {res[t]['score']:7.2f} shape {res[t]['shape_p']} path {res[t]['path_p']} | search {res[t]['auto_search_score']}")
    (sb.OUT / f"naskh_restyle_{style}.json").write_text(json.dumps({"leave_one_out": loo, "letters": res}, ensure_ascii=False, indent=1))
    # --- sheet
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from generate_and_build import ribbon_polygon
    AR = FontProperties(fname="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    names = list(parts)
    fig, axs = plt.subplots(2, 1, figsize=(30, 14))
    for ax, row in zip(axs, (names[:9], names[9:])):
        x = 0.0
        for t in row:
            part, N = parts[t]
            g = ribbon_polygon(part.q, np.clip(part.wl, 1, None), np.clip(part.wr, 1, None))
            minx, _, maxx, _ = g.bounds
            dx = x - minx
            for pg in ([g] if g.geom_type == "Polygon" else list(g.geoms)):
                xs, ys = pg.exterior.xy
                ax.fill(np.array(xs) + dx, ys, color="#222", lw=0)
                for h in pg.interiors:
                    hx, hy = h.xy
                    ax.fill(np.array(hx) + dx, hy, color="white", lw=0)
            ax.plot(part.q[:, 0] + dx, part.q[:, 1], color="#ff5a36", lw=1)
            nx = N[:, 0] - N[:, 0].mean() + (minx + maxx) / 2 + dx
            ax.plot(nx, N[:, 1], "--", color="#2b7bba", lw=1.2)
            ax.text((minx + maxx) / 2 + dx, R.base + 560, f"{lg.CLASSES[t]}  {t}\nshape {res[t]['shape_p']} | path {res[t]['path_p']}",
                    fontproperties=AR, fontsize=12, ha="center", va="top")
            x += (maxx - minx) + 160
        ax.axhline(R.base, color="#e4572e", lw=0.8)
        ax.set_ylim(R.base + 760, R.base - 760)
        ax.set_aspect("equal")
        ax.axis("off")
    axs[0].set_title(f"{style}: Naskh skeleton (blue dashes) redrawn arc by arc with the writer's own stroke pieces (red = new pen path)", fontsize=14)
    fig.tight_layout()
    fig.savefig(sb.OUT / f"{style}_NASKH_RESTYLE.png", dpi=60)
    print("saved", sb.OUT / f"{style}_NASKH_RESTYLE.png")


if __name__ == "__main__":
    main()
