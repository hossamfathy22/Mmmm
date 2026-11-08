#!/usr/bin/env python3
"""Exact brush: reproduce ANY uploaded lettering from (pen path + brush).

A brush stroke's outline is, by construction, two offset curves of the pen
path (left and right edges) closed by two end caps. So instead of guessing
the width with rays, every point of the real outline is projected onto its
nearest point of the pen path:

  * projects inside the path  -> it is the left or right edge at that position
                                 (nearest point => the offset is along the normal)
  * projects past an end      -> it belongs to that end's CAP shape

That gives the brush exactly, like Illustrator's Width Tool with Alt (each
side separately) plus a custom cap profile. The brush is then compressed to a
few width points per side and a few points per cap, and the word is rebuilt
from (pen path nodes + brush) only, to measure how close it is.

Brush properties stored per stroke:
  width_left(u), width_right(u)   asymmetric width profile (PCHIP keypoints)
  cap_start(angle), cap_end(angle) end shape as radius per angle (round, flat,
                                   teardrop, ball ...), relative to the tangent
"""
from __future__ import annotations

import json
import math
import sys

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

import skeleton_brush as sb


def outline_points(mask, step=0.5):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    pts = []
    for c in cnts:
        c = c[:, 0, :].astype(float)
        if len(c) < 8:
            continue
        c = np.stack([gaussian_filter1d(c[:, 0], 1.0, mode="wrap"), gaussian_filter1d(c[:, 1], 1.0, mode="wrap")], 1)
        c = sb.resample(np.vstack([c, c[:1]]), step)
        pts.append(c)
    return np.vstack(pts)


def visible_nearest(q, c, mask, k=60, n_check=24):
    """For every outline point, the nearest pen-path sample it can SEE through
    the ink (the straight segment between them stays inside the letter).
    Plain nearest-point assignment fails at narrow gaps: the inner edge of a
    thick bowl is geometrically closer to the thin stroke on the other side of
    a slit than to its own centerline."""
    from scipy import ndimage
    tree = cKDTree(q)
    k = min(k, len(q))
    _, cand = tree.query(c, k=k)
    cand = cand.reshape(len(c), k)
    m = ndimage.binary_dilation(mask, iterations=2).astype(np.float32)
    idx = cand[:, 0].copy()
    todo = np.ones(len(c), bool)
    t = np.linspace(0.05, 0.95, n_check)
    for r in range(k):
        if not todo.any():
            break
        sel = np.nonzero(todo)[0]
        a, b = c[sel], q[cand[sel, r]]
        pts = a[:, None, :] + (b - a)[:, None, :] * t[None, :, None]
        inside = ndimage.map_coordinates(m, [pts[..., 1].ravel(), pts[..., 0].ravel()], order=0).reshape(len(sel), n_check)
        ok = inside.min(1) > 0.5
        idx[sel[ok]] = cand[sel[ok], r]
        todo[sel[ok]] = False
    return idx


def measure(q, mask):
    """Exact brush of one stroke whose pen path samples are q (1 px apart)."""
    T = sb._tangents(q)
    N = np.stack([-T[:, 1], T[:, 0]], 1)
    c = outline_points(mask, step=1.0)
    idx = visible_nearest(q, c, mask)
    v = c - q[idx]
    along = (v * T[idx]).sum(1)
    side = (v * N[idx]).sum(1)
    n = len(q)
    wl = np.full(n, np.nan)
    wr = np.full(n, np.nan)
    interior = ~(((idx == 0) & (along < -0.5)) | ((idx == n - 1) & (along > 0.5)))
    for arr, sel in ((wl, interior & (side > 0)), (wr, interior & (side <= 0))):
        ii, dd = idx[sel], np.abs(side[sel])
        order = np.argsort(ii)
        ii, dd = ii[order], dd[order]
        if len(ii):
            uniq, start = np.unique(ii, return_index=True)
            med = [np.median(dd[s:e]) for s, e in zip(start, list(start[1:]) + [len(dd)])]
            arr[uniq] = med
    k = np.arange(n)
    for arr in (wl, wr):
        ok = ~np.isnan(arr)
        arr[:] = np.interp(k, k[ok], arr[ok]) if ok.any() else 0.0
    caps = {}
    for name, sel, i0, sgn in (("start", (idx == 0) & (along < -0.5), 0, -1), ("end", (idx == n - 1) & (along > 0.5), n - 1, 1)):
        pts = c[sel] - q[i0]
        t = T[i0] * sgn                                   # direction pointing out of the stroke
        nn = np.array([-t[1], t[0]])
        ang = np.degrees(np.arctan2((pts * nn).sum(1), (pts * t).sum(1)))
        r = np.linalg.norm(pts, axis=1)
        o = np.argsort(ang)
        caps[name] = (ang[o], r[o])
    return wl, wr, caps


def rdp(x, y, tol):
    """Ramer–Douglas–Peucker on a width profile: the fewest corner points
    whose straight segments stay within tol (angular brushes)."""
    keep = np.zeros(len(x), bool)
    keep[[0, -1]] = True
    stack = [(0, len(x) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        t = (x[a + 1:b] - x[a]) / max(x[b] - x[a], 1e-9)
        yl = y[a] + t * (y[b] - y[a])
        k = int(np.argmax(np.abs(y[a + 1:b] - yl))) + a + 1
        if abs(y[k] - yl[k - a - 1]) > tol:
            keep[k] = True
            stack += [(a, k), (k, b)]
    return np.nonzero(keep)[0]


def compress_width(u, w, L, tol=1.0, max_keys=40):
    """Two brush interpolation modes, chosen per stroke side:
    'smooth' (PCHIP through width points, round brushes) or 'linear'
    (corner points, angular brushes) — whichever needs fewer points."""
    ks, _, es = sb.fit_width(u, w, tol=tol, max_keys=max_keys, smooth=0.7)
    idx = rdp(u * L, gaussian_filter1d(w, 0.7), tol)
    lin = [(float(u[i]), float(w[i])) for i in idx]
    if len(lin) < len(ks) or es > 2 * tol:
        return "linear", lin
    return "smooth", ks


def eval_width(mode, keys, u):
    ku = np.array([k[0] for k in keys])
    kw = np.array([k[1] for k in keys])
    return np.clip(np.interp(u, ku, kw) if mode == "linear" else PchipInterpolator(ku, kw)(u), 0, None)


def compress_cap(ang, r, n_keys=9):
    if len(ang) < 4:
        return []
    grid = np.linspace(ang.min(), ang.max(), n_keys)
    rr = np.interp(grid, ang, gaussian_filter1d(r, 2))
    return [[round(float(a), 1), round(float(b), 2)] for a, b in zip(grid, rr)]


def cap_polygon(q, T, i0, sgn, keys):
    if not keys:
        return None
    t = T[i0] * sgn
    nn = np.array([-t[1], t[0]])
    a = np.radians(np.linspace(keys[0][0], keys[-1][0], 60))
    r = np.interp(np.degrees(a), [k[0] for k in keys], [k[1] for k in keys])
    pts = q[i0] + (np.cos(a)[:, None] * t + np.sin(a)[:, None] * nn) * r[:, None]
    return np.vstack([q[i0], pts])


def render_sides(canvas, q, wl, wr, jump=3.0):
    """Fill each side as quads between consecutive cross-sections, but never
    bridge a jump in the edge (a narrow gap / slit end): there only the two
    cross-sections themselves are drawn, so ink never spills across a gap."""
    T = sb._tangents(q)
    N = np.stack([-T[:, 1], T[:, 0]], 1)
    for w, sgn in ((wl, 1), (wr, -1)):
        E = q + sgn * N * w[:, None]
        for i in range(len(q) - 1):
            if np.linalg.norm(E[i + 1] - E[i]) <= jump + np.linalg.norm(q[i + 1] - q[i]) * (1 + 0.02 * max(w[i], w[i + 1])):
                quad = np.array([q[i], q[i + 1], E[i + 1], E[i]]) * 16
                cv2.fillConvexPoly(canvas, np.round(quad).astype(np.int32), 255, lineType=cv2.LINE_AA, shift=4)
            else:
                for a, b in ((q[i], E[i]), (q[i + 1], E[i + 1])):
                    cv2.line(canvas, tuple(np.round(a * 16).astype(int)), tuple(np.round(b * 16).astype(int)), 255, 2,
                             lineType=cv2.LINE_AA, shift=4)


def render(canvas, q, wl, wr, caps_keys):
    render_sides(canvas, q, wl, wr)
    T = sb._tangents(q)
    for name, i0, sgn in (("start", 0, -1), ("end", len(q) - 1, 1)):
        poly = cap_polygon(q, T, i0, sgn, caps_keys.get(name))
        if poly is not None:
            cv2.fillPoly(canvas, [np.round(poly * 16).astype(np.int32)], 255, lineType=cv2.LINE_AA, shift=4)


def run(style, box, width_tol=1.0, max_keys=40, cap_keys=9):
    img = sb.HERE / "input" / "muzn_two_styles.jpg"
    mask, rec_old, model, vis = sb.analyse(img, box, style)
    comps = sb.split_components(mask)
    raw = np.zeros(mask.shape, np.uint8)
    comp_ = np.zeros(mask.shape, np.uint8)
    brushes = []
    for v in vis:
        if v["kind"] == "dot":
            for canv in (raw, comp_):
                cx, cy = v["center"]
                cv2.circle(canv, (int(round(cx * 16)), int(round(cy * 16))), int(round(v["r"] * 16)), 255, -1,
                           lineType=cv2.LINE_AA, shift=4)
            continue
        q, m = v["p"], comps[v["comp"]]["mask"]
        wl, wr, caps = measure(q, m)
        ck = {k: compress_cap(*caps[k], n_keys=cap_keys) for k in caps}
        render(raw, q, wl, wr, {k: compress_cap(*caps[k], n_keys=33) for k in caps})
        u, _ = sb.arclen_u(q)
        _, L = sb.arclen_u(q)
        ml, kl = compress_width(u, wl, L, width_tol, max_keys)
        mr, kr = compress_width(u, wr, L, width_tol, max_keys)
        render(comp_, q, eval_width(ml, kl, u), eval_width(mr, kr, u), ck)
        brushes.append({"component": v["comp"], "path_nodes": len(v["segs"]) + 1,
                        "width_interpolation": {"left": ml, "right": mr},
                        "width_points_left": len(kl), "width_points_right": len(kr),
                        "cap_points": {k: len(x) for k, x in ck.items()},
                        "cap_shape": {k: describe_cap(x) for k, x in ck.items()}})
    res = {"style": style,
           "iou_previous_ray_brush": round(model["iou_word"], 4),
           "iou_exact_brush_uncompressed": round(sb.iou(mask, raw > 127), 4),
           "iou_exact_brush_compressed": round(sb.iou(mask, comp_ > 127), 4),
           "pen_path_nodes": model["summary"]["total_path_nodes"],
           "strokes": brushes}
    H, W = mask.shape
    diff = np.zeros((H, W, 3), np.uint8)
    rc = comp_ > 127
    diff[mask & rc] = (235, 235, 235)
    diff[mask & ~rc] = (230, 60, 60)
    diff[~mask & rc] = (60, 140, 255)
    cv2.imwrite(str(sb.OUT / f"{style}_6_exact_brush_overlap.png"), diff[:, :, ::-1])
    return res


def describe_cap(keys):
    if not keys:
        return "none"
    r = np.array([k[1] for k in keys])
    spread = keys[-1][0] - keys[0][0]
    if spread < 120:
        return "pointed / teardrop"
    if r.std() / (r.mean() + 1e-9) < 0.15:
        return "round"
    if r[len(r) // 2] < 0.6 * r.max():
        return "flat / square"
    return "asymmetric"


def main():
    styles = {"style1": (1000, 360, 1850, 880), "style2": (120, 360, 860, 880)}
    out = {s: run(s, b) for s, b in styles.items() if len(sys.argv) < 2 or s in sys.argv[1:]}
    (sb.OUT / "exact_brush_report.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for s, r in out.items():
        print(s, {k: v for k, v in r.items() if k != "strokes"})


if __name__ == "__main__":
    main()
