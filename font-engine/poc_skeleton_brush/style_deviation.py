#!/usr/bin/env python3
"""Change every letter the way the writer changed the letters we can see.

  normal noon : writer's noon  ::  normal X : ?

1. NORMAL letters: the average single pen path of each letter form over the
   49-font corpus (orientations aligned) — the most typical shape.
2. DEVIATION of each uploaded letter from its normal letter, in two layers:
     global  an affine map (size, width, slant) fitted normal -> writer;
     local   at every point of the path: extra turning (Δθ), stretch
             (log ds ratio) and the writer's brush widths (left/right).
3. TRANSFER: every point of a normal target letter borrows the deviation of
   the most similar points of the uploaded letters (k nearest neighbours in a
   feature space: height above the baseline, direction of travel, curvature,
   position in the stroke, distance to a terminal). Bowl bottoms borrow from
   bowl bottoms, stroke starts from stroke starts — no hand-written recipes,
   so any uploaded word teaches the engine.
4. The new path is integrated from its new turning and stretch, placed on the
   writer's baseline, and painted with the borrowed widths.

Benchmark: each corpus font plays the "writer" (only its meem/zain/noon are
given), the others make the normal letters; predicted paths are compared with
the font's real letters (x-free mean point error, nuqtas).
"""
from __future__ import annotations

import json

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

import skeleton_blend as sbl
import skeleton_brush as sb

KNOWN = sbl.KNOWN
TARGETS = sbl.TARGETS


# ----------------------------------------------------------------------------
# normal letters
# ----------------------------------------------------------------------------


def align_orientations(data):
    """Same drawing direction for every font's path of a form (iterative)."""
    forms = {f for d in data.values() for f in d}
    for form in forms:
        fonts = [f for f in data if form in data[f]]
        mean = data[fonts[0]][form]
        for _ in range(4):
            for f in fonts:
                P = data[f][form]
                if np.linalg.norm(P[::-1] - mean, axis=1).mean() < np.linalg.norm(P - mean, axis=1).mean():
                    data[f][form] = P[::-1]
            mean = np.mean([data[f][form] for f in fonts], axis=0)
    return data


def normal(data, form, exclude=None):
    paths = [d[form] for f, d in data.items() if form in d and f != exclude]
    return np.mean(paths, axis=0) if paths else None


def orient_like(P, ref):
    return P[::-1] if np.linalg.norm(P[::-1] - ref, axis=1).mean() < np.linalg.norm(P - ref, axis=1).mean() else P


# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------


def turning(P):
    d = np.diff(P, axis=0)
    return np.unwrap(np.arctan2(d[:, 1], d[:, 0])), np.linalg.norm(d, axis=1)


def affine_fit(N, U):
    X = np.c_[N, np.ones(len(N))]
    M, *_ = np.linalg.lstsq(X, U, rcond=None)
    return M  # (3, 2)


def apply_affine(M, P):
    return np.c_[P, np.ones(len(P))] @ M


def features(P):
    th, ds = turning(P)
    mid = (P[1:] + P[:-1]) / 2
    k = np.gradient(th) / np.maximum(ds, 1e-6)
    u = np.r_[0, np.cumsum(ds)][:-1] / max(ds.sum(), 1e-9)
    term = np.minimum(u, 1 - u)
    return np.stack([mid[:, 1] / 1.0,                 # height (nuqta, baseline = 0)
                     np.cos(th), np.sin(th),          # direction of travel
                     np.clip(k, -3, 3) / 2.0,         # curvature (rad / nuqta)
                     0.5 * u,                         # position along the stroke
                     2.0 * np.exp(-term / 0.08)], 1)  # near a terminal


# ----------------------------------------------------------------------------
# deviation and transfer
# ----------------------------------------------------------------------------


def frames(P):
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
    return T, np.stack([-T[:, 1], T[:, 0]], 1)


def scale_fit(N, U):
    """Global change = width and height only (no rotation, no shear): x about
    the letter's centre, y about the baseline (y = 0) so the baseline stays."""
    sx = (U[:, 0].std() + 1e-6) / (N[:, 0].std() + 1e-6)
    sy = np.abs(U[:, 1]).mean() / max(np.abs(N[:, 1]).mean(), 1e-6)
    return float(np.clip(sx, 0.3, 3.0)), float(np.clip(sy, 0.3, 3.0))


def apply_scale(N, sx, sy, cx=None):
    cx = N[:, 0].mean() if cx is None else cx
    return np.stack([(N[:, 0] - cx) * sx + cx, N[:, 1] * sy], 1)


def dtw_match(A, B):
    """Dynamic time warping: for each point of A the matching point of B
    (position + direction), order preserved."""
    TA, _ = frames(A)
    TB, _ = frames(B)
    C = np.linalg.norm(A[:, None] - B[None], axis=2) + 0.5 * np.linalg.norm(TA[:, None] - TB[None], axis=2)
    n, m = C.shape
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = C[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    i, j = n, m
    pairs = []
    while i > 0 and j > 0:
        pairs.append((i - 1, j - 1))
        k = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
        i, j = (i - 1, j - 1) if k == 0 else (i - 1, j) if k == 1 else (i, j - 1)
    match = [[] for _ in range(n)]
    for a, b_ in pairs:
        match[a].append(b_)
    return match


def deviations(normals, writer, widths=None):
    """Per known letter: width/height scale + local displacement of every
    point of the (scaled) normal letter towards the writer's letter,
    expressed in the normal path's own frame (along, across)."""
    out = {}
    for form, U in writer.items():
        N = normals[form]
        flipped = np.linalg.norm(U[::-1] - N, axis=1).mean() < np.linalg.norm(U - N, axis=1).mean()
        U = U[::-1] if flipped else U
        sx, sy = scale_fit(N, U)
        AN = apply_scale(N, sx, sy)
        AN += [U[:, 0].mean() - AN[:, 0].mean(), 0]
        match = dtw_match(AN, U)
        T, Nn = frames(AN)
        disp = np.array([U[m].mean(0) - AN[i] for i, m in enumerate(match)])
        loc = np.stack([(disp * T).sum(1), (disp * Nn).sum(1)], 1)
        rec = {"sx": sx, "sy": sy, "AN": AN, "U": U, "loc": loc, "feat": features_pts(AN)}
        if widths is not None and form in widths:
            wl, wr = widths[form]
            if flipped:  # reversing the drawing swaps the brush's left and right
                wl, wr = wr[::-1], wl[::-1]
            W = np.stack([wl, wr], 1)
            rec["w"] = np.array([W[m].mean(0) for m in match])
        out[form] = rec
    return out


def features_pts(P):
    f = features(P)
    return np.vstack([f[:1], (f[1:] + f[:-1]) / 2, f[-1:]])


def global_scale(dev):
    return float(np.exp(np.mean([np.log(d["sx"]) for d in dev.values()]))), \
        float(np.exp(np.mean([np.log(d["sy"]) for d in dev.values()])))


def transfer(normal_X, dev, k=10, smooth=2.0):
    sx, sy = global_scale(dev)
    AN = apply_scale(normal_X, sx, sy)
    F = np.vstack([d["feat"] for d in dev.values()])
    LOC = np.vstack([d["loc"] for d in dev.values()])
    W = np.vstack([d["w"] for d in dev.values()]) if all("w" in d for d in dev.values()) else None
    tree = cKDTree(F)
    dist, idx = tree.query(features_pts(AN), k=k)
    wts = 1.0 / (dist + 1e-3)
    wts /= wts.sum(1, keepdims=True)
    loc = (LOC[idx] * wts[..., None]).sum(1)
    loc = np.stack([gaussian_filter1d(loc[:, 0], smooth), gaussian_filter1d(loc[:, 1], smooth)], 1)
    T, Nn = frames(AN)
    P = AN + loc[:, :1] * T + loc[:, 1:] * Nn
    widths = (W[idx] * wts[..., None]).sum(1) if W is not None else None
    return P, AN, widths


def benchmark(data):
    rows = {t: [] for t in TARGETS}
    for F in data:
        if not all(k in data[F] for k in KNOWN):
            continue
        normals = {f: normal(data, f, exclude=F) for f in KNOWN + TARGETS}
        dev = deviations(normals, {k: data[F][k] for k in KNOWN})
        for t in TARGETS:
            if t in data[F] and normals[t] is not None:
                P, _, _ = transfer(normals[t], dev)
                rows[t].append(sbl.path_err(P - [P[:, 0].mean(), 0], data[F][t] - [data[F][t][:, 0].mean(), 0]))
    return {t: round(float(np.median(v)), 3) for t, v in rows.items() if v}


def describe(dev):
    """Human-readable: how the writer changed each letter from normal."""
    out = {}
    for form, d in dev.items():
        L_U = np.linalg.norm(np.diff(d["U"], axis=0), axis=1).sum()
        L_N = np.linalg.norm(np.diff(d["AN"], axis=0), axis=1).sum()
        th_u, _ = turning(d["U"])
        th_n, _ = turning(d["AN"])
        ang = lambda P, a, b: float(np.degrees(np.arctan2(*(P[b] - P[a])[::-1])))
        out[form] = {"width_scale": round(d["sx"], 2), "height_scale": round(d["sy"], 2),
                     "mean_local_shift_nuqta": round(float(np.linalg.norm(d["loc"], axis=1).mean()), 2),
                     "total_turning_writer_vs_normal_deg": [round(float(np.degrees(np.abs(np.diff(th_u)).sum())), 0),
                                                            round(float(np.degrees(np.abs(np.diff(th_n)).sum())), 0)],
                     "path_length_change_pct": round(float(100 * (L_U / max(L_N, 1e-9) - 1)), 1),
                     "start_direction_writer_vs_normal_deg": [round(ang(d["U"], 0, 4), 0), round(ang(d["AN"], 0, 4), 0)],
                     "end_direction_writer_vs_normal_deg": [round(ang(d["U"], -5, -1), 0), round(ang(d["AN"], -5, -1), 0)]}
    return out


def main():
    data = align_orientations(sbl.load())
    bench = benchmark(data)
    prev = json.loads((sb.OUT / "recipe_learning_report.json").read_text())
    table = {t.split(".")[0]: {"deviation_transfer": bench.get(t), "template_blend": prev[t.split(".")[0]]["blend_err"],
                               "nearest_font": prev[t.split(".")[0]]["nearest_err"]} for t in TARGETS}
    U, D, S = sbl.user_paths()
    normals = {f: normal(data, f) for f in KNOWN + TARGETS}
    dev = deviations(normals, U)
    (sb.OUT / "style_deviation_report.json").write_text(json.dumps(
        {"corpus_benchmark_err_nuqta": table, "how_the_writer_changed_each_letter": describe(dev)},
        ensure_ascii=False, indent=1))
    for t, v in table.items():
        print(f"{t:12s}", v)
    print(json.dumps(describe(dev), indent=1))


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------
# part-wise analogy: carry the deviation only between letters that share
# anatomy (arabic_anatomy): beh <- noon, reh/dal/alef <- zain, lam/yeh <- both
# ----------------------------------------------------------------------------

RELATED = {"beh.isol": ["noon.isol"], "reh.isol": ["zain.fina"], "dal.isol": ["zain.fina"],
           "alef.isol": ["zain.fina"], "lam.isol": ["zain.fina", "noon.isol"],
           "alefmaksura.isol": ["zain.fina", "noon.isol"], "seen.isol": ["noon.isol"]}


def dtw_cost_match(A, B):
    """DTW match of A onto B plus the local matching cost per point of A."""
    TA, _ = frames(A)
    TB, _ = frames(B)
    # compare shapes, not positions: centre both, scale to unit size
    a = (A - A.mean(0)) / (A.std() + 1e-9)
    b = (B - B.mean(0)) / (B.std() + 1e-9)
    C = np.linalg.norm(a[:, None] - b[None], axis=2) + 0.7 * np.linalg.norm(TA[:, None] - TB[None], axis=2)
    n, m = C.shape
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = C[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    i, j = n, m
    match = [[] for _ in range(n)]
    while i > 0 and j > 0:
        match[i - 1].append(j - 1)
        k = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
        i, j = (i - 1, j - 1) if k == 0 else (i - 1, j) if k == 1 else (i, j - 1)
    cost = np.array([C[ii, m_].mean() for ii, m_ in enumerate(match)])
    return match, cost


def transfer_parts(target, normals, dev, smooth=2.0):
    sx, sy = global_scale(dev)
    AN = apply_scale(normals[target], sx, sy)
    T, Nn = frames(AN)
    best_cost = np.full(len(AN), np.inf)
    loc = np.zeros((len(AN), 2))
    W = np.zeros((len(AN), 2)) if all("w" in d for d in dev.values()) else None
    for src in RELATED.get(target, []):
        if src not in dev:
            continue
        # correspondence between the two NORMAL letters (same anatomy, typical shapes)
        match, cost = dtw_cost_match(AN, dev[src]["AN"])
        for i, mm in enumerate(match):
            if cost[i] < best_cost[i]:
                best_cost[i] = cost[i]
                loc[i] = dev[src]["loc"][mm].mean(0)
                if W is not None:
                    W[i] = dev[src]["w"][mm].mean(0)
    if not np.isfinite(best_cost).any():
        P, AN2, Wk = transfer(normals[target], dev)
        return P, AN2, Wk
    loc = np.stack([gaussian_filter1d(loc[:, 0], smooth), gaussian_filter1d(loc[:, 1], smooth)], 1)
    P = AN + loc[:, :1] * T + loc[:, 1:] * Nn
    return P, AN, W


def benchmark_parts(data):
    rows = {t: [] for t in TARGETS}
    for F in data:
        if not all(k in data[F] for k in KNOWN):
            continue
        normals = {f: normal(data, f, exclude=F) for f in KNOWN + TARGETS}
        dev = deviations(normals, {k: data[F][k] for k in KNOWN})
        for t in TARGETS:
            if t in data[F] and normals[t] is not None:
                P, _, _ = transfer_parts(t, normals, dev)
                rows[t].append(sbl.path_err(P - [P[:, 0].mean(), 0], data[F][t] - [data[F][t][:, 0].mean(), 0]))
    return {t: round(float(np.median(v)), 3) for t, v in rows.items() if v}



def user_widths(S, D):
    """The writer's exact brush widths at the 64 path points (px)."""
    import exact_brush as eb
    out = {}
    for key, form in (("meem", "meem.init"), ("zain", "zain.fina"), ("noon", "noon.isol")):
        q = S["strokes"][key][0]
        ci = S["ids"][key]
        wl, wr, _ = eb.measure(q, S["comps"][ci]["mask"])
        L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
        t = np.linspace(0, L[-1], 64)
        out[form] = (np.interp(t, L, wl), np.interp(t, L, wr))
    return out


def deviation_sheet():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.interpolate import CubicSpline
    import compose as cp
    from generate_and_build import ribbon_polygon
    data = align_orientations(sbl.load())
    U, D, S = sbl.user_paths()
    normals = {f: normal(data, f) for f in KNOWN + TARGETS}
    dev = deviations(normals, U, user_widths(S, D))
    fig, axs = plt.subplots(2, 1, figsize=(26, 13))
    ax = axs[0]
    x = 0
    for form in ("noon.isol", "zain.fina", "meem.init"):
        d = dev[form]
        dx = x - d["AN"][:, 0].min()
        ax.plot(d["AN"][:, 0] + dx, d["AN"][:, 1], "--", color="#888", lw=2, label="normal" if x == 0 else None)
        ax.plot(d["U"][:, 0] + dx, d["U"][:, 1], color="#e4572e", lw=2.5, label="writer" if x == 0 else None)
        T, Nn = frames(d["AN"])
        disp = d["loc"][:, :1] * T + d["loc"][:, 1:] * Nn
        for i in range(0, 64, 3):
            ax.annotate("", xy=(d["AN"][i, 0] + disp[i, 0] + dx, d["AN"][i, 1] + disp[i, 1]),
                        xytext=(d["AN"][i, 0] + dx, d["AN"][i, 1]), arrowprops=dict(arrowstyle="->", color="#2b7bba", lw=0.8))
        ax.text(d["AN"][:, 0].mean() + dx, 2.2, form, ha="center", fontsize=12)
        x += np.ptp(np.r_[d["AN"][:, 0], d["U"][:, 0]]) + 1.5
    ax.axhline(0, color="#ccc")
    ax.invert_yaxis(); ax.set_aspect("equal"); ax.axis("off"); ax.legend(fontsize=12, loc="upper right")
    ax.set_title("How the writer changed each letter from the NORMAL letter (blue arrows = local displacement)", fontsize=14)
    ax = axs[1]
    x = 0
    for t in TARGETS:
        P, AN, W = transfer_parts(t, normals, dev)
        Px = P * D
        ANx = AN * D
        s_ = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Px, axis=0), axis=1))]
        ss = np.linspace(0, s_[-1], int(s_[-1]))
        q = np.stack([CubicSpline(s_, Px[:, 0])(ss), CubicSpline(s_, Px[:, 1])(ss)], 1)
        wl = np.interp(ss, s_, W[:, 0]) if W is not None else np.full(len(ss), 60.0)
        wr = np.interp(ss, s_, W[:, 1]) if W is not None else np.full(len(ss), 60.0)
        g = ribbon_polygon(q, np.clip(wl, 1, None), np.clip(wr, 1, None))
        letter = t.split(".")[0]
        ds_ = cp.dots_for(letter, g, S) if letter in ("beh",) else []
        dx = x - min(g.bounds[0], ANx[:, 0].min())
        for pg in ([g] if g.geom_type == "Polygon" else list(g.geoms)) + ds_:
            xs, ys = pg.exterior.xy
            ax.fill(np.array(xs) + dx, ys, color="#d9d9d9", lw=0)
        ax.plot(ANx[:, 0] + dx, ANx[:, 1], "--", color="#888", lw=1.5)
        ax.plot(q[:, 0] + dx, q[:, 1], color="#e4572e", lw=1.5)
        ax.text((g.bounds[0] + g.bounds[2]) / 2 + dx, 2.6 * D, letter, ha="center", fontsize=12)
        x += (max(g.bounds[2], ANx[:, 0].max()) - min(g.bounds[0], ANx[:, 0].min())) + 150
    ax.axhline(0, color="#ccc")
    ax.invert_yaxis(); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("New letters: normal letter (dashed) changed the same way the writer changed noon/zain, painted with the writer's brush", fontsize=14)
    fig.tight_layout()
    fig.savefig(sb.OUT / "DEVIATION_SHEET.png", dpi=70)
    print("saved", sb.OUT / "DEVIATION_SHEET.png")


if __name__ == "__main__" and False:
    pass
