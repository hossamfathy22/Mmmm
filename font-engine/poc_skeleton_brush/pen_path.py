#!/usr/bin/env python3
"""ONE smooth pen path per letter, drawn the way a type designer would.

Rules implemented here (each one is checked by `path_quality`):

  1. One path per letter, from the start of the stroke to its end. Lobes and
     bulbs that hang off the main stroke are entered and left with a smooth
     hairpin (up one side of the lobe, around, down the other side) instead of
     a second path.
  2. The path runs through the middle of the ink: it follows the medial axis
     (centres of the largest inscribed circles); inside a hairpin each leg runs
     through the middle of its half of the lobe.
  3. Every node is smooth: tangents are shared (G1, no kinks) and the curvature
     is matched on both sides of every node (G2), by a least-squares fairing of
     the handle lengths.
  4. Few nodes, in designer positions: stroke ends, horizontal/vertical extrema
     (handles snapped flat), inflections; extra nodes only where the fit
     tolerance demands it.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.sparse.csgraph import connected_components, dijkstra
from skimage.morphology import skeletonize

import skeleton_brush as sb

# ----------------------------------------------------------------------------
# 1. single walk on the medial axis
# ----------------------------------------------------------------------------


def _branch_info(G, leaf, deg):
    """Length from a leaf to the first junction, and that junction."""
    prev, cur, length = -1, leaf, 0.0
    while True:
        nbrs = G.indices[G.indptr[cur]:G.indptr[cur + 1]]
        w = G.data[G.indptr[cur]:G.indptr[cur + 1]]
        nxt = [(n, ww) for n, ww in zip(nbrs, w) if n != prev]
        if deg[cur] >= 3 or not nxt:
            return length, cur
        if cur != leaf and deg[cur] != 2:
            return length, cur
        prev, (cur, ww) = cur, nxt[0]
        length += ww


def single_walk(mask, dist, max_leaves=8, hairpin_cost=0.004):
    """Best single walk over the skeleton: leaf -> [lobe leaf ->] leaf, scored
    by how much ink its medial discs cover, minus a small cost per hairpin."""
    skel = skeletonize(mask)
    G, pts = sb.skeleton_graph(skel)
    rad = dist[pts[:, 1].astype(int), pts[:, 0].astype(int)]
    _, labels = connected_components(G, directed=False)
    keep = labels == np.argmax(np.bincount(labels))
    deg = np.asarray((G > 0).sum(1)).ravel()
    leaves = [int(l) for l in np.nonzero(keep & (deg == 1))[0]]
    info = {l: _branch_info(G, l, deg) for l in leaves}
    # significant leaves: branch clearly longer than the radius where it joins
    sig = [l for l in leaves if info[l][0] > 0.5 * rad[info[l][1]] or deg[info[l][1]] < 3]
    sig = sorted(sig, key=lambda l: -info[l][0])[:max_leaves]
    if len(sig) < 2:
        sig = sorted(leaves, key=lambda l: -info[l][0])[:2]
    if len(sig) < 2:
        # closed shapes (an eye / ring) have fewer than two free ends: start at
        # the existing end (or anywhere) and add the farthest point of the ring
        a = sig[0] if sig else int(np.nonzero(keep)[0][0])
        da = dijkstra(G, indices=a)
        da[~np.isfinite(da)] = -1
        sig = [a, int(np.argmax(da))]

    f = 4
    import cv2
    small = cv2.resize(mask.astype(np.uint8), (mask.shape[1] // f + 2, mask.shape[0] // f + 2),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    total = small.sum()
    D, P = dijkstra(G, indices=sig, return_predecessors=True)

    def path(i, j):
        return sb._backtrack(P[i], sig[j])[::-1]

    best = None
    n = len(sig)
    for a in range(n):
        for b in range(n):
            if a == b or not np.isfinite(D[a, sig[b]]):
                continue
            mids = [None] + [m for m in range(n) if m not in (a, b)]
            for m in mids:
                if m is None:
                    if b < a:
                        continue  # same walk reversed
                    walk, pins = path(a, b), []
                else:
                    p1, p2 = path(a, m), path(m, b)
                    k = 0
                    while k < min(len(p1), len(p2)) and p1[-1 - k] == p2[k]:
                        k += 1
                    walk = p1 + p2[1:]
                    pins = [(len(p1) - k, len(p1) - 1, len(p1) - 1 + k - 1)]
                cov = (sb._disc_cover([walk], pts, rad, mask.shape, f) & small).sum() / total
                retrace = sum(e_ - s_ for s_, _, e_ in pins) / max(len(walk), 1)
                # prefer few hairpins, and short ones (a lobe), never a whole bowl retraced
                score = cov - hairpin_cost * len(pins) - 0.03 * retrace
                if best is None or score > best[0]:
                    best = (score, walk, pins, cov)
    _, walk, pins, cov = best
    coords = pts[walk].astype(float)
    r = rad[walk].astype(float)
    for s, mtip, e in pins:
        coords = _open_hairpin(coords, r, s, mtip, e)
    return coords, {"hairpins": len(pins), "disc_coverage": float(cov), "candidate_leaves": len(sig)}


def _open_hairpin(c, r, s, m, e):
    """The walk goes junction(s) -> lobe tip(m) -> junction(e) on the same pixels.
    Push the two legs apart so the path makes a clean U-turn inside the lobe:
    the leg going in stays on the side the path came from, the leg coming out
    on the side it leaves to (otherwise the legs cross -> figure-8)."""
    out = c.copy()
    J, tip = c[s], c[m]
    d = sb._unit(tip - J)
    n_left = np.array([-d[1], d[0]])
    look = int(1.5 * r[s]) + 10  # judge sides well outside the junction blob
    A = c[max(s - look, 0)]
    B = c[min(e + look, len(c) - 1)]
    side_in = np.sign(np.dot(A - J, n_left)) or 1.0
    side_out = np.sign(np.dot(B - J, n_left)) or -side_in
    if side_out == side_in:
        side_out = -side_in
    # legs keep their full separation through the junction (no pinch -> no figure-8);
    # the separation fades in before the junction and fades out after it
    smooth = lambda x: x * x * (3 - 2 * x)
    N = len(c)
    off = np.zeros(N)
    sign = np.zeros(N)
    off[s:m + 1], sign[s:m + 1] = 0.5 * r[s:m + 1], side_in
    off[m:e + 1], sign[m:e + 1] = 0.5 * r[m:e + 1], side_out
    for k in range(max(0, s - look), s):
        off[k], sign[k] = 0.5 * r[k] * smooth((k - (s - look)) / look), side_in
    for k in range(e + 1, min(N, e + 1 + look)):
        off[k], sign[k] = 0.5 * r[k] * smooth(1 - (k - e) / look), side_out
    out = c + (sign * off)[:, None] * n_left[None, :]
    return out


# ----------------------------------------------------------------------------
# 2. smooth fit: G1 everywhere + G2 fairing
# ----------------------------------------------------------------------------


def _fit_with_u(pts, t0, t1):
    P, err = sb.fit_cubic(pts, t0, t1)
    # recover parameters for fairing
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    u = np.r_[0, np.cumsum(seg)] / max(seg.sum(), 1e-9)
    for _ in range(6):
        diff = sb.bez(P, u) - pts
        d1, d2 = sb.bez_d1(P, u), sb.bez_d2(P, u)
        num = (diff * d1).sum(1)
        den = (d1 * d1).sum(1) + (diff * d2).sum(1)
        u = np.clip(u - np.where(np.abs(den) > 1e-9, num / den, 0), 0, 1)
        u[0], u[-1] = 0, 1
        u = np.maximum.accumulate(u)
    return P, err, u


def fit_smooth(p, tol=4.0, min_gap=30.0, tol_reduce=None, max_turn=110.0):
    """Fit, then (if tol_reduce) remove every node the drawing does not need:
    the path keeps only its start, its end and the nodes that carry a bend.
    A node goes if one cubic can replace its two neighbours within tol_reduce
    (chosen as a fraction of the stroke width, so the change is invisible
    under the brush)."""
    nodes = sb.detect_nodes(p, min_gap=min_gap, corner_deg=1e9)  # no corners: all smooth
    stack = list(zip(nodes[:-1], nodes[1:]))[::-1]
    segs, data, out_nodes = [], [], [nodes[0]]
    while stack:
        a, b = stack.pop()
        pts = p[a.i:b.i + 1]
        P, err, u = _fit_with_u(pts, a.t_out, b.t_in)
        k = int(np.argmax(err))
        if err[k] > tol and len(pts) > 8 and 3 <= k <= len(pts) - 4:
            mi = a.i + k
            t = sb._unit(p[min(mi + 3, len(p) - 1)] - p[max(mi - 3, 0)])
            mid = sb.Node(mi, "split", t, t)
            stack.append((mid, b))
            stack.append((a, mid))
            continue
        segs.append(P)
        data.append((pts, u))
        out_nodes.append(b)
    if tol_reduce:
        keep = out_nodes[:]
        while len(keep) > 2:
            best = None
            for j in range(1, len(keep) - 1):
                a, b = keep[j - 1], keep[j + 1]
                pts = p[a.i:b.i + 1]
                # drawing rules: one segment turns <= max_turn degrees and its
                # handles never reach past the segment (no "balloon" handles)
                d = np.diff(pts, axis=0)
                turn = math.degrees(abs(np.diff(np.unwrap(np.arctan2(d[:, 1], d[:, 0]))).sum()))
                if turn > max_turn:
                    continue
                P, err, _ = _fit_with_u(pts, a.t_out, b.t_in)
                chord = np.linalg.norm(P[3] - P[0])
                if max(np.linalg.norm(P[1] - P[0]), np.linalg.norm(P[3] - P[2])) > 0.9 * chord:
                    continue
                if best is None or err.max() < best[0]:
                    best = (err.max(), j)
            if best is None:
                break
            if best[0] > tol_reduce:
                break
            keep.pop(best[1])
        segs, data = [], []
        for a, b in zip(keep[:-1], keep[1:]):
            pts = p[a.i:b.i + 1]
            P, _, u = _fit_with_u(pts, a.t_out, b.t_in)
            segs.append(P)
            data.append((pts, u))
        out_nodes = keep
    segs = fair_g2(segs, data)
    return out_nodes, segs


def _kappa0(P):
    a, b = P[1] - P[0], P[2] - P[1]
    return (2 / 3) * (a[0] * b[1] - a[1] * b[0]) / max(np.linalg.norm(a), 1e-9) ** 3


def _kappa1(P):
    a, b = P[3] - P[2], P[1] - P[2]
    return (2 / 3) * (a[0] * b[1] - a[1] * b[0]) / max(np.linalg.norm(a), 1e-9) ** 3


def fair_g2(segs, data, w=4000.0):
    """Adjust handle lengths (directions and nodes fixed) so that curvature is
    continuous across every node while staying close to the measured path."""
    n = len(segs)
    if n < 2:
        return segs
    t0 = [sb._unit(P[1] - P[0]) for P in segs]
    t1 = [sb._unit(P[3] - P[2]) for P in segs]
    L = [max(np.linalg.norm(np.diff(d[0], axis=0), axis=1).sum(), 1.0) for d in data]
    x0 = np.array([[np.linalg.norm(P[1] - P[0]), np.linalg.norm(P[3] - P[2])] for P in segs]).ravel()
    lo = np.array([[0.05 * l, 0.05 * l] for l in L]).ravel()
    hi = np.array([[1.2 * l, 1.2 * l] for l in L]).ravel()
    x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)

    def build(x):
        out = []
        for i, P in enumerate(segs):
            a, b = x[2 * i], x[2 * i + 1]
            out.append(np.array([P[0], P[0] + a * t0[i], P[3] - b * t1[i], P[3]]))
        return out

    def resid(x):
        S = build(x)
        r = [(sb.bez(S[i], data[i][1]) - data[i][0]).ravel() for i in range(n)]
        g = [w * (_kappa1(S[i]) - _kappa0(S[i + 1])) for i in range(n - 1)]
        return np.concatenate(r + [np.array(g)])

    res = least_squares(resid, x0, bounds=(lo, hi), method="trf", max_nfev=400)
    return build(res.x)


# ----------------------------------------------------------------------------
# 3. quality tests
# ----------------------------------------------------------------------------


def path_quality(segs, mask, wl=None, wr=None, q=None):
    kinks, g2 = [], []
    kmax = 1e-9
    for P in segs:
        t = np.linspace(0, 1, 60)
        d1, d2 = sb.bez_d1(P, t), sb.bez_d2(P, t)
        kk = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / (np.linalg.norm(d1, axis=1) ** 3 + 1e-9)
        kmax = max(kmax, float(np.percentile(np.abs(kk), 95)))
    for A, B in zip(segs[:-1], segs[1:]):
        u, v = sb._unit(A[3] - A[2]), sb._unit(B[1] - B[0])
        kinks.append(math.degrees(math.acos(np.clip(np.dot(u, v), -1, 1))))
        # curvature jump at the node, relative to the sharpest curve of the letter
        g2.append(abs(_kappa1(A) - _kappa0(B)) / kmax)
    wiggles = 0
    handle_ratio_bad = 0
    for P in segs:
        t = np.linspace(0, 1, 120)
        d1, d2 = sb.bez_d1(P, t), sb.bez_d2(P, t)
        k = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / (np.linalg.norm(d1, axis=1) ** 3 + 1e-9)
        dk = np.diff(gaussian_filter1d(k, 2, mode="nearest")[4:-4])
        ext = np.sum(np.sign(dk[1:]) != np.sign(dk[:-1]))
        wiggles += max(0, ext - 1)
        chord = np.linalg.norm(P[3] - P[0]) + 1e-9
        for h in (np.linalg.norm(P[1] - P[0]), np.linalg.norm(P[3] - P[2])):
            if not (0.1 <= h / chord <= 0.9):
                handle_ratio_bad += 1
    samples = sb.sample_beziers(segs, 1.0)
    inside = ndimage.map_coordinates(mask.astype(np.float32), [samples[:, 1], samples[:, 0]], order=1) > 0.5
    out = {
        "paths": 1,
        "nodes": len(segs) + 1,
        "max_kink_deg (G1, 0 = smooth)": round(max(kinks) if kinks else 0.0, 3),
        "max_curvature_jump (G2, 0 = smooth)": round(max(g2) if g2 else 0.0, 3),
        "mean_curvature_jump": round(float(np.mean(g2)) if g2 else 0.0, 3),
        "segments_with_2plus_curvature_peaks": int(wiggles),
        "handles_outside_10_90pct_of_chord": int(handle_ratio_bad),
        "path_inside_ink_pct": round(100 * float(inside.mean()), 1),
    }
    if wl is not None:
        out["centering_mean |L-R|/(L+R)"] = round(float(np.mean(np.abs(wl - wr) / (wl + wr + 1e-9))), 3)
    return out
