#!/usr/bin/env python3
"""Nib brush: a real pen tip with settings, fitted automatically to an image.

Inspired by the references (Farsi brush pack, Procreate qalam brushes,
Polanco's constructed letters): the skeleton is one thing, the BRUSH makes
the style. The brush here is a nib swept along the pen path:

  shape      superellipse exponent n   (2 = round/oval, 3 = soft square, 8 = rectangle)
  aspect     nib thickness / nib width (1 = circle or square, 0.3 = flat reed)
  angle      nib angle to the horizontal (qalam angle)
  pressure   scale of the nib along the stroke, K keypoints (K = 1: constant)

Fitting: for each candidate nib, the pressure that explains the measured
stroke width is computed directly (width / nib extent across the path); the
nib that needs the least pressure variation and gives the best IoU wins.
The fitted brush is then used on any skeleton (here: Naskh skeletons).
"""
from __future__ import annotations

import json
import sys

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline

import compose as cp
import skeleton_brush as sb

Q = 4
SHAPES = {2.0: "round / oval", 3.0: "soft square", 8.0: "rectangle"}


def nib_polygon(n, aspect, angle_deg, m=28):
    t = np.linspace(0, 2 * np.pi, m, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    x = np.sign(c) * np.abs(c) ** (2 / n)
    y = aspect * np.sign(s) * np.abs(s) ** (2 / n)
    a = np.radians(angle_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return np.stack([x, y], 1) @ R.T          # unit nib: half-width 1 along its long axis


def extent(poly, normals):
    """Half extent of the nib across the path, for each path normal."""
    return np.abs(normals @ poly.T).max(1)


def sweep(canvas, q, scale, poly):
    pts_prev = None
    for p, s in zip(q, scale):
        pts = p + s * poly
        if pts_prev is not None:
            hull = cv2.convexHull(np.round(np.vstack([pts, pts_prev]) * 16).astype(np.int32))
            cv2.fillConvexPoly(canvas, hull, 255, lineType=cv2.LINE_AA, shift=4)
        pts_prev = pts


def keys_profile(u, s, K):
    if K == 1:
        return np.full_like(s, np.median(s)), [[0.5, float(np.median(s))]]
    ku = np.linspace(0, 1, K)
    ks = np.array([np.median(s[np.abs(u - k) <= 0.5 / (K - 1)]) for k in ku])
    return PchipInterpolator(ku, ks)(u), [[float(a), float(b)] for a, b in zip(ku, ks)]


def load(style):
    S = cp.load_style(style)
    strokes, mask = [], np.zeros((S["mask"].shape[0] // Q + 1, S["mask"].shape[1] // Q + 1), bool)
    for k in ("meem", "zain", "noon"):
        q, wl, wr = S["strokes"][k]
        strokes.append((q[::Q] / Q, ((wl + wr) / 2)[::Q] / Q))
        m = S["comps"][S["ids"][k]]["mask"]
        small = cv2.resize(m.astype(np.uint8), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA) > 0
        mask |= small
    return S, strokes, mask


def render(strokes, mask_shape, poly, K):
    canvas = np.zeros(mask_shape, np.uint8)
    profs, rough = [], []
    for q, w in strokes:
        T = sb._tangents(q)
        N = np.stack([-T[:, 1], T[:, 0]], 1)
        s = w / np.maximum(extent(poly, N), 1e-3)
        u, _ = sb.arclen_u(q)
        sm, keys = keys_profile(u, s, K)
        body = (u > 0.1) & (u < 0.9)
        rough.append(float(np.std(s[body]) / (np.mean(s[body]) + 1e-9)))
        sweep(canvas, q, sm, poly)
        profs.append(keys)
    return canvas > 127, profs, float(np.mean(rough))


def fit(style):
    S, strokes, mask = load(style)
    results = []
    for n in SHAPES:
        for aspect in (1.0, 0.8, 0.6, 0.45, 0.3):
            for ang in (range(0, 180, 10) if (aspect < 1 or n > 2) else [0]):
                poly = nib_polygon(n, aspect, ang)
                r1, _, rough = render(strokes, mask.shape, poly, 1)
                results.append((sb.iou(mask, r1), n, aspect, ang, rough))
    results.sort(reverse=True)
    best = results[0]
    poly = nib_polygon(*best[1:4])
    out = {"nib_shape": SHAPES[best[1]], "exponent": best[1], "aspect": best[2], "angle_deg": best[3],
           "pressure_variation_needed": round(best[4], 3), "iou": {}}
    for K in (1, 4, 8, 16):
        r, profs, _ = render(strokes, mask.shape, poly, K)
        out["iou"][f"pressure_keys_{K}"] = round(sb.iou(mask, r), 4)
        if K == 8:
            out["pressure_profiles_8"] = profs
            diff = np.zeros((*mask.shape, 3), np.uint8)
            diff[mask & r] = (235, 235, 235); diff[mask & ~r] = (230, 60, 60); diff[~mask & r] = (60, 140, 255)
            cv2.imwrite(str(sb.OUT / f"{style}_8_nib_brush_overlap.png"), diff[:, :, ::-1])
    # the plain round pen for comparison
    r0, _, _ = render(strokes, mask.shape, nib_polygon(2.0, 1.0, 0), 1)
    out["iou"]["round_pen_constant"] = round(sb.iou(mask, r0), 4)
    out["nib_width_px_original_image"] = round(float(np.median([np.median(w) for _, w in strokes]) * Q / 2 * 2
                                                     / max(extent(poly, np.array([[0.0, 1.0]]))[0], 1e-3)), 1)
    out["top5"] = [dict(iou=round(a, 4), shape=SHAPES[n], aspect=asp, angle=ang) for a, n, asp, ang, _ in results[:5]]
    return S, strokes, poly, out


def naskh_sheet(fits):
    """Same Naskh skeletons, three looks: bare skeleton, style-1 brush, style-2 brush."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = json.loads((sb.OUT / "skeleton_corpus.json").read_text())["Amiri"]
    letters = ["alef", "beh", "hah", "dal", "reh", "seen", "sad", "ain", "feh", "qaf", "lam", "meem", "noon", "heh", "waw",
               "alefmaksura"]
    hn = np.ptp(np.array(data["noon.isol"]["path"])[:, 1])
    fig, axs = plt.subplots(len(fits) + 1, 1, figsize=(26, 4.2 * (len(fits) + 1)))
    rows = [("Naskh skeleton (Amiri), pen path only", None)] + [(f"{s}: fitted nib = {o['nib_shape']}, aspect {o['aspect']}, "
                                                               f"angle {o['angle_deg']}°", (S, st, poly, o)) for s, (S, st, poly, o) in fits.items()]
    for ax, (title, f) in zip(axs, rows):
        x = 0.0
        for L in letters:
            P = np.array(data[L + ".isol"]["path"]) / hn
            if f is None:
                unit = 60.0
                q = P * unit
                ax.plot(q[:, 0] + x - q[:, 0].min(), q[:, 1], color="black", lw=1.5)
                x += np.ptp(q[:, 0]) + 60
                continue
            S, strokes, poly, o = f
            unit = float(np.ptp(strokes[2][0][:, 1]))          # the writer's noon path height
            P = P * unit
            s_ = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
            ss = np.arange(0, s_[-1], 1.0)
            q = np.stack([CubicSpline(s_, P[:, 0])(ss), CubicSpline(s_, P[:, 1])(ss)], 1)
            u = ss / ss[-1]
            prof = np.mean([PchipInterpolator(*np.array(k).T)(u) for k in o["pressure_profiles_8"]], axis=0)
            pad = prof.max() * 1.3 + 4
            off = q.min(0) - pad
            size = np.ceil(q.max(0) - off + pad).astype(int)
            canvas = np.zeros((size[1], size[0]), np.uint8)
            sweep(canvas, q - off, prof, poly)
            ax.imshow(255 - canvas, cmap="gray", vmin=0, vmax=255,
                      extent=(x, x + size[0], off[1] + size[1], off[1]), zorder=1)
            x += size[0] + 25
        ax.set_xlim(-20, x)
        ax.set_ylim(*( (150, -160) if f is None else (2.2 * unit, -2.6 * unit)))
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=14, loc="left")
    fig.tight_layout()
    fig.savefig(sb.OUT / "NIB_BRUSH_NASKH.png", dpi=60)
    plt.close(fig)


def main():
    fits, report = {}, {}
    for style in ("style1", "style2"):
        S, strokes, poly, out = fit(style)
        fits[style] = (S, strokes, poly, out)
        report[style] = out
        print(style, {k: v for k, v in out.items() if k not in ("pressure_profiles_8", "top5")})
    (sb.OUT / "nib_brush_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    naskh_sheet(fits)
    print("saved", sb.OUT / "NIB_BRUSH_NASKH.png")


if __name__ == "__main__":
    main()
