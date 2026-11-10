#!/usr/bin/env python3
"""Change the whole pen path the way the writer changed it (the user's diagram).

  Naskh reh skeleton  ->  writer's reh skeleton      (learn the change)
  Naskh seen skeleton ->  styled seen skeleton       (apply the change)

The path is treated as ONE curve, described by its direction theta along its
length. From the writer's letters versus the same Naskh letters the engine
learns a style operator:

  entry   how the stroke starts: the extra turning over the first part
          (the hooked, rounded entry of this writer)
  exit    how the stroke ends: the extra turning over the last part
  body    how soft the middle is (smoothing of the direction) and how much
          its bends are exaggerated (gain)
  size    width and height scale

The operator is applied to any Naskh skeleton, the new direction function is
integrated back into a path, and the writer's brush is laid on it.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d

import auto_design as ad
import compose as cp
import naskh_restyle as nr
import skeleton_brush as sb
import style_deviation as sd

PAIRS = {"noon": "noon.isol", "zain": "zain.fina", "meem": "meem.init"}


def dense(P, step=1.0):
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    t = np.arange(0, L[-1], step)
    return np.stack([np.interp(t, L, P[:, 0]), np.interp(t, L, P[:, 1])], 1)


def theta(P):
    d = np.diff(P, axis=0)
    return np.unwrap(np.arctan2(d[:, 1], d[:, 0]))


def integrate(p0, th, step=1.0):
    return p0 + np.vstack([[0, 0], np.cumsum(np.stack([np.cos(th), np.sin(th)], 1) * step, axis=0)])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class CurveStyle:
    def __init__(self, style="style1", exclude=()):
        self.R = nr.Restyler(style)
        R = self.R
        self.unit = R.unit
        self.Ls = int(0.9 * R.unit)                       # length of the entry / exit zones (px)
        pairs = {}
        for key, form in PAIRS.items():
            if key in exclude:
                continue
            U = dense(R.S["strokes"][key][0])
            N = R.ref[form] * R.k
            Uc = nr.Restyler._r64(U) - [0, R.base]
            Uc[:, 0] -= Uc[:, 0].mean()
            N = dense(sd.orient_like(N, Uc))
            pairs[key] = (N, U)
        self.pairs = pairs
        # entry and exit motifs: mean extra direction of the writer over the zone
        ent, ext = [], []
        for N, U in pairs.values():
            tn, tu = theta(N), theta(U)
            n = min(self.Ls, len(tn) // 2, len(tu) // 2)
            d0 = wrap(tu[:n] - tn[:n])
            d1 = wrap(tu[-n:] - tn[-n:])
            ent.append(np.pad(np.unwrap(d0), (0, self.Ls - n), mode="edge"))
            ext.append(np.pad(np.unwrap(d1), (self.Ls - n, 0), mode="edge"))
        self.entry = gaussian_filter1d(np.mean(ent, axis=0), 6)
        self.exit = gaussian_filter1d(np.mean(ext, axis=0), 6)
        # body softness / gain / length ratio: grid search on the known pairs
        self.len_ratio = float(np.mean([len(U) / len(N) for N, U in pairs.values()]))
        best = None
        for sig in (0.0, 0.04, 0.08, 0.14, 0.22):
            for gain in (0.8, 1.0, 1.25, 1.5):
                self.sigma, self.gain = sig, gain
                self.sx = self.sy = 1.0
                e = np.mean([self._err(self.apply_path(N), U) for N, U in pairs.values()])
                if best is None or e < best[0]:
                    best = (e, sig, gain)
        _, self.sigma, self.gain = best
        self.sx = self.sy = 1.0
        gx, gy = [], []
        for N, U in pairs.values():
            G = self.apply_path(N)
            gx.append(np.ptp(U[:, 0]) / max(np.ptp(G[:, 0]), 1))
            gy.append(np.ptp(U[:, 1]) / max(np.ptp(G[:, 1]), 1))
        self.sx, self.sy = float(np.exp(np.mean(np.log(gx)))), float(np.exp(np.mean(np.log(gy))))

    def _err(self, G, U):
        return nr.path_err(G, U, self.unit)

    def apply_path(self, N):
        """Naskh path (px, dense) -> styled path."""
        # stretch the path to the writer's typical length, keep its direction function
        n_new = max(int(len(N) * self.len_ratio), 2 * 8)
        th = theta(N)
        th = np.interp(np.linspace(0, len(th) - 1, n_new), np.arange(len(th)), th)
        if self.sigma > 0:
            soft = gaussian_filter1d(th, self.sigma * self.unit, mode="nearest")
        else:
            soft = th
        lin = np.linspace(th[0], th[-1], len(th))
        body = lin + self.gain * (soft - lin)             # exaggerate the bends around the straight trend
        n = min(self.Ls, len(th) // 2)
        w = np.linspace(1, 0, n) ** 2 * (3 - 2 * np.linspace(1, 0, n))
        out = body.copy()
        out[:n] += w * self.entry[:n]
        out[-n:] += w[::-1] * self.exit[-n:]
        G = integrate(np.zeros(2), out)
        G = G * [self.sx, self.sy]
        return dense(G)

    def styled(self, form):
        R = self.R
        N = dense(R.ref[form] * R.k)
        G = self.apply_path(N)
        G += [N[:, 0].mean() - G[:, 0].mean(), N[:, 1].max() - G[:, 1].max()]
        return G + [0, R.base], N + [0, R.base]

    def brush(self, G):
        """Writer's brush: mean of the uploaded strokes' half-widths, real
        terminals kept (stretch between guides), symmetric about the path."""
        s_new = np.r_[0, np.cumsum(np.linalg.norm(np.diff(G, axis=0), axis=1))]
        Ln = s_new[-1]
        ws = []
        for key in PAIRS:
            q, wl, wr = self.R.S["strokes"][key]
            w = (wl + wr) / 2
            s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
            L = s[-1]
            g = min(1.2 * np.median(w), 0.3 * min(L, Ln))
            sm = np.where(s_new < g, s_new, np.where(s_new > Ln - g, L - (Ln - s_new),
                          g + (s_new - g) * (L - 2 * g) / max(Ln - 2 * g, 1e-6)))
            ws.append(np.interp(sm, s, w))
        w = gaussian_filter1d(np.mean(ws, axis=0), 8, mode="nearest")
        cap_n = int(min(len(w) // 4, np.median(w)))
        t = np.linspace(0, 1, cap_n)
        w[:cap_n] = np.minimum(w[:cap_n], np.median(w) * np.sqrt(1 - (1 - t) ** 2) + 1)
        w[-cap_n:] = np.minimum(w[-cap_n:], np.median(w) * np.sqrt(1 - (1 - t[::-1]) ** 2) + 1)
        return cp.Part(G, w, w.copy(), "styled", True, True)

    def describe(self):
        return {"entry_extra_turn_deg": round(float(np.degrees(self.entry[0])), 0),
                "exit_extra_turn_deg": round(float(np.degrees(self.exit[-1])), 0),
                "body_softness": self.sigma, "bend_gain": self.gain, "length_ratio": round(self.len_ratio, 2),
                "width_scale": round(self.sx, 2), "height_scale": round(self.sy, 2)}


def main():
    style = sys.argv[1] if len(sys.argv) > 1 else "style1"
    loo = {}
    for key, form in PAIRS.items():
        C = CurveStyle(style, exclude=(key,))
        full = CurveStyle.__new__(CurveStyle)
        G, N = C.styled(form)
        U = C.R.S["strokes"][key][0]
        loo[key] = {"styled_err": round(nr.path_err(G, U, C.unit), 3), "plain_naskh_err": round(nr.path_err(N, U, C.unit), 3)}
        print("LOO", key, loo[key])
    C = CurveStyle(style)
    print("style operator:", C.describe())
    ctxq = ad.Ctx(style)
    eye, peye = ad.Eye(ctxq), ad.PathEye(ctxq)
    res, parts = {}, {}
    for t in ad.TARGETS:
        form = {"yeh": "alefmaksura.isol"}.get(t, t + ".isol")
        G, N = C.styled(form)
        part = C.brush(G)
        (s, info), = ad.score_batch([nr.quarter(part)], [None], t, ctxq, eye, peye)
        res[t] = {"score": round(float(s), 2), "shape_p": round(info["p"], 2), "path_p": round(info["p_path"], 2)}
        parts[t] = (part, N)
        print(f"{t:6s} score {res[t]['score']:7.2f} shape {res[t]['shape_p']} path {res[t]['path_p']}")
    (sb.OUT / f"curve_style_{style}.json").write_text(json.dumps({"leave_one_out_noon_heights": loo, "operator": C.describe(),
                                                                  "letters": res}, ensure_ascii=False, indent=1))
    # sheet in the layout of the user's diagram: Naskh skeleton -> styled skeleton -> brushed
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from generate_and_build import ribbon_polygon
    AR = FontProperties(fname="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    names = list(parts)
    fig, axs = plt.subplots(3, 1, figsize=(34, 17))
    x = 0.0
    xs_ = {}
    for t in names:
        part, N = parts[t]
        g = ribbon_polygon(part.q, np.clip(part.wl, 1, None), np.clip(part.wr, 1, None))
        minx, _, maxx, _ = g.bounds
        lo = min(minx, N[:, 0].min(), part.q[:, 0].min())
        hi = max(maxx, N[:, 0].max(), part.q[:, 0].max())
        dx = x - lo
        axs[0].plot(N[:, 0] + dx, N[:, 1], color="black", lw=2)
        axs[1].plot(part.q[:, 0] + dx, part.q[:, 1], color="#ff5a36", lw=2.5)
        for pg in ([g] if g.geom_type == "Polygon" else list(g.geoms)):
            px, py = pg.exterior.xy
            axs[2].fill(np.array(px) + dx, py, color="#ff5a36", lw=0)
            for h in pg.interiors:
                hx, hy = h.xy
                axs[2].fill(np.array(hx) + dx, hy, color="white", lw=0)
        axs[2].text((lo + hi) / 2 + dx, C.R.base + 520, f"{ad.lg.CLASSES[t]}\nshape {res[t]['shape_p']} | path {res[t]['path_p']}",
                    fontproperties=AR, fontsize=13, ha="center", va="top")
        x += (hi - lo) + 150
    for ax, title in zip(axs, ("Naskh skeleton (reference)", "Skeleton after the writer's style operator (entry, exit, softness, size)",
                               "Painted with the writer's brush")):
        ax.axhline(C.R.base, color="#bbb", lw=0.8)
        ax.set_xlim(-50, x)
        ax.set_ylim(C.R.base + 700, C.R.base - 700)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=15, loc="left")
    fig.tight_layout()
    fig.savefig(sb.OUT / f"{style}_CURVE_STYLE.png", dpi=55)
    print("saved", sb.OUT / f"{style}_CURVE_STYLE.png")


if __name__ == "__main__":
    main()
