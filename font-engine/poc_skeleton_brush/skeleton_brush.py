#!/usr/bin/env python3
"""Proof of concept: recover the "pen path + brush" that drew a piece of lettering.

Pipeline (for one image region):
  1. binarize + split into connected components (bodies, dots, marks)
  2. medial axis: skeleton + distance transform (= local half-width of the ink)
  3. turn the skeleton into ordered strokes (main stroke = longest path, extra
     strokes only when they are much longer than the local radius -> no spurs)
  4. fit each stroke with as few cubic Beziers as possible, putting nodes where a
     type designer would: stroke ends, horizontal/vertical extrema, inflections
     and corners; handles at extrema are snapped horizontal/vertical
  5. measure the brush: half-width r along the stroke, compressed to a few
     width keypoints (like Illustrator's Width Tool), and test whether the
     width is explained by a broad-nib pen angle or by "pressure" along the stroke
  6. re-render the word from (Bezier path + brush) only and compare to the
     original with IoU

Outputs go to ./output (JSON model + diagnostic PNGs).
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from skimage.morphology import skeletonize

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"

# ----------------------------------------------------------------------------
# 1. image -> components
# ----------------------------------------------------------------------------


def load_region(path: Path, box, scale: int = 2):
    """Crop, upscale (smooth) and binarize. Returns bool mask (ink=True)."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    x0, y0, x1, y1 = box
    crop = img[y0:y1, x0:x1].astype(np.float32)
    if scale != 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        crop = cv2.GaussianBlur(crop, (0, 0), 0.6 * scale)
    return crop > 128


def split_components(mask, min_area=80):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    comps = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < min_area:
            continue
        m = lab == i
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        per = cv2.arcLength(max(cnts, key=len), True)
        comps.append(
            dict(mask=m, bbox=(int(x), int(y), int(w), int(h)), area=int(a),
                 circularity=float(4 * math.pi * a / (per * per)))
        )
    amax = max(c["area"] for c in comps)
    for c in comps:
        if c["circularity"] > 0.85 and c["area"] < 0.25 * amax:
            c["kind"] = "dot"
        elif c["area"] < 0.08 * amax:
            c["kind"] = "mark"
        else:
            c["kind"] = "body"
    # right-to-left reading order
    comps.sort(key=lambda c: -(c["bbox"][0] + c["bbox"][2] / 2))
    return comps


# ----------------------------------------------------------------------------
# 2-3. skeleton -> ordered strokes
# ----------------------------------------------------------------------------


def skeleton_graph(skel):
    ys, xs = np.nonzero(skel)
    idx = -np.ones(skel.shape, dtype=np.int64)
    idx[ys, xs] = np.arange(len(ys))
    H, W = skel.shape
    rows, cols, wts = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        y2, x2 = ys + dy, xs + dx
        ok = (y2 >= 0) & (y2 < H) & (x2 >= 0) & (x2 < W)
        j = np.full(len(ys), -1)
        j[ok] = idx[y2[ok], x2[ok]]
        good = np.nonzero(j >= 0)[0]
        rows.append(good)
        cols.append(j[good])
        wts.append(np.full(len(good), math.hypot(dy, dx)))
    r, c, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(wts)
    n = len(ys)
    G = coo_matrix((np.r_[w, w], (np.r_[r, c], np.r_[c, r])), shape=(n, n)).tocsr()
    return G, np.stack([xs, ys], 1).astype(float)


def _backtrack(pred, node):
    path = [node]
    while pred[node] >= 0:
        node = pred[node]
        path.append(node)
    return path


def _disc_cover(paths, pts, rad, shape, f=4, step=3):
    """Rasterise the union of medial discs along skeleton paths (coarse grid)."""
    canvas = np.zeros((shape[0] // f + 2, shape[1] // f + 2), np.uint8)
    for path in paths:
        for i in path[::step] + [path[-1]]:
            x, y = pts[i]
            cv2.circle(canvas, (int(x / f), int(y / f)), int(rad[i] / f + 0.5), 1, -1)
    return canvas


def extract_strokes(mask, dist, min_gain=0.02):
    """Choose pen strokes on the medial axis by how much ink they explain.

    Main stroke: among all pairs of skeleton end points, the path whose
    medial discs cover the most ink (so the stroke runs from tip to tip, into
    bulbous terminals, instead of into a spur). Extra strokes are added only if
    they explain at least `min_gain` of the remaining ink (spurs explain ~0).
    """
    skel = skeletonize(mask)
    G, pts = skeleton_graph(skel)
    rad = dist[pts[:, 1].astype(int), pts[:, 0].astype(int)]
    _, labels = connected_components(G, directed=False)
    keep = labels == np.argmax(np.bincount(labels))
    deg = np.asarray((G > 0).sum(1)).ravel()
    leaves = np.nonzero(keep & (deg == 1))[0]
    if len(leaves) < 2:
        leaves = np.nonzero(keep)[0][[0, -1]]
    f = 4
    small = cv2.resize(mask.astype(np.uint8), (mask.shape[1] // f + 2, mask.shape[0] // f + 2),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    total = small.sum()

    D, P = dijkstra(G, indices=leaves, return_predecessors=True)
    best, best_path = -1, None
    for a in range(len(leaves)):
        for b in range(a + 1, len(leaves)):
            if not np.isfinite(D[a, leaves[b]]):
                continue
            path = _backtrack(P[a], leaves[b])[::-1]
            cov = (_disc_cover([path], pts, rad, mask.shape, f) & small).sum()
            if cov > best:
                best, best_path = cov, path
    strokes = [best_path]
    covered = set(best_path)
    cur = best
    while True:
        src = np.fromiter(covered, dtype=np.int64)
        _, pred, _ = dijkstra(G, indices=src, min_only=True, return_predecessors=True)
        cand = None
        for leaf in leaves:
            if leaf in covered or pred[leaf] < 0:
                continue
            br = _backtrack(pred, leaf)[::-1]  # attach -> tip
            cov = (_disc_cover(strokes + [br], pts, rad, mask.shape, f) & small).sum()
            if cand is None or cov > cand[0]:
                cand = (cov, br)
        if cand is None or (cand[0] - cur) < min_gain * total:
            break
        cur = cand[0]
        strokes.append(cand[1])
        covered.update(cand[1])
    return [pts[s] for s in strokes], skel


def orient_rtl(poly):
    """Arabic: start a stroke at its right-most (then top-most) end."""
    a, b = poly[0], poly[-1]
    if (b[0] - a[0]) > 2 or (abs(b[0] - a[0]) <= 2 and b[1] < a[1]):
        return poly[::-1]
    return poly


def resample(poly, step=1.0):
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    n = max(int(math.ceil(s[-1] / step)), 2)
    t = np.linspace(0, s[-1], n + 1)
    return np.stack([np.interp(t, s, poly[:, 0]), np.interp(t, s, poly[:, 1])], 1)


def smooth_open(poly, sigma):
    """Gaussian smoothing that keeps end points and end tangents (point reflection padding)."""
    k = int(3 * sigma) + 1
    if len(poly) < 2 * k + 2:
        k = max(1, len(poly) // 2 - 1)
    head = 2 * poly[0] - poly[1:k + 1][::-1]
    tail = 2 * poly[-1] - poly[-k - 1:-1][::-1]
    p = np.vstack([head, poly, tail])
    p = np.stack([gaussian_filter1d(p[:, 0], sigma), gaussian_filter1d(p[:, 1], sigma)], 1)
    return p[k:-k]


# ----------------------------------------------------------------------------
# 4. Bezier fitting with designer-style node placement
# ----------------------------------------------------------------------------


def bez(P, t):
    t = np.asarray(t)[:, None]
    mt = 1 - t
    return mt**3 * P[0] + 3 * mt * mt * t * P[1] + 3 * mt * t * t * P[2] + t**3 * P[3]


def bez_d1(P, t):
    t = np.asarray(t)[:, None]
    mt = 1 - t
    return 3 * mt * mt * (P[1] - P[0]) + 6 * mt * t * (P[2] - P[1]) + 3 * t * t * (P[3] - P[2])


def bez_d2(P, t):
    t = np.asarray(t)[:, None]
    return 6 * (1 - t) * (P[2] - 2 * P[1] + P[0]) + 6 * t * (P[3] - 2 * P[2] + P[1])


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def fit_cubic(pts, t0, t1, iters=8):
    """Fit one cubic with fixed end points and fixed tangent directions
    (Schneider-style least squares + Newton re-parameterisation)."""
    p0, p3 = pts[0], pts[-1]
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    L = seg.sum()
    u = np.r_[0, np.cumsum(seg)] / max(L, 1e-9)
    P = None
    for _ in range(iters):
        b0, b1, b2, b3 = (1 - u) ** 3, 3 * (1 - u) ** 2 * u, 3 * (1 - u) * u * u, u**3
        A = np.stack([(b1[:, None] * t0).ravel(), (-b2[:, None] * t1).ravel()], 1)
        rhs = (pts - (b0 + b1)[:, None] * p0 - (b2 + b3)[:, None] * p3).ravel()
        (ha, hb), *_ = np.linalg.lstsq(A, rhs, rcond=None)
        if not (0.02 * L < ha < 1.2 * L) or not (0.02 * L < hb < 1.2 * L):
            ha = hb = L / 3
        P = np.array([p0, p0 + ha * t0, p3 - hb * t1, p3])
        diff = bez(P, u) - pts
        d1, d2 = bez_d1(P, u), bez_d2(P, u)
        num = (diff * d1).sum(1)
        den = (d1 * d1).sum(1) + (diff * d2).sum(1)
        u = u - np.where(np.abs(den) > 1e-9, num / den, 0)
        u = np.clip(u, 0, 1)
        u[0], u[-1] = 0, 1
        u = np.maximum.accumulate(u)
    err = np.linalg.norm(bez(P, u) - pts, axis=1)
    return P, err


@dataclass
class Node:
    i: int
    kind: str  # end | x-extremum | y-extremum | inflection | corner | split
    t_in: np.ndarray = field(default_factory=lambda: np.zeros(2))
    t_out: np.ndarray = field(default_factory=lambda: np.zeros(2))


def detect_nodes(p, min_gap=14.0, corner_deg=40.0):
    N = len(p)
    T = np.gradient(p, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    Ts = np.stack([gaussian_filter1d(T[:, 0], 2), gaussian_filter1d(T[:, 1], 2)], 1)
    Ts /= np.linalg.norm(Ts, axis=1, keepdims=True) + 1e-12
    ang = np.unwrap(np.arctan2(Ts[:, 1], Ts[:, 0]))
    kappa = gaussian_filter1d(np.gradient(ang), 3)

    cands = []  # (priority, index, kind)
    w = 6
    for i in range(w, N - w):  # corners: big turn over a short window
        turn = math.degrees(abs(ang[i + w] - ang[i - w]))
        if turn > corner_deg and abs(kappa[i]) == np.max(np.abs(kappa[i - w:i + w + 1])):
            cands.append((1, i, "corner"))
    for comp, kind in ((0, "x-extremum"), (1, "y-extremum")):
        s = Ts[:, comp]
        for i in range(1, N - 2):
            if s[i] == 0 or np.sign(s[i]) != np.sign(s[i + 1]):
                j = i if abs(s[i]) < abs(s[i + 1]) else i + 1
                cands.append((2, j, kind))
    thr = 0.15 * np.max(np.abs(kappa)) + 1e-6
    for i in range(1, N - 2):
        if np.sign(kappa[i]) != np.sign(kappa[i + 1]):
            lo, hi = max(0, i - 25), min(N, i + 25)
            if np.max(kappa[lo:i + 1] * np.sign(kappa[i])) > thr and \
               np.max(kappa[i + 1:hi] * np.sign(kappa[i + 1])) > thr:
                cands.append((3, i, "inflection"))

    chosen = [(0, "end"), (N - 1, "end")]
    for _, i, kind in sorted(cands):
        if all(abs(i - j) >= min_gap for j, _ in chosen):
            chosen.append((i, kind))
    chosen.sort()

    nodes = []
    for i, kind in chosen:
        lo, hi = max(0, i - 4), min(N - 1, i + 4)
        tin = _unit(p[i] - p[lo]) if i > 0 else _unit(p[min(N - 1, 4)] - p[0])
        tout = _unit(p[hi] - p[i]) if i < N - 1 else _unit(p[-1] - p[max(0, N - 5)])
        if kind != "corner":
            t = _unit(tin + tout) if i not in (0, N - 1) else (tout if i == 0 else tin)
            if kind == "x-extremum":
                t = np.array([0.0, math.copysign(1, t[1])])
            elif kind == "y-extremum":
                t = np.array([math.copysign(1, t[0]), 0.0])
            tin = tout = t
        nodes.append(Node(i, kind, tin, tout))
    return nodes


def fit_stroke(p, tol=4.0, min_gap=30.0):
    """Returns (nodes, list of 4x2 Bezier control arrays). Adds 'split' nodes
    only where the error tolerance demands it."""
    nodes = detect_nodes(p, min_gap=min_gap)
    out_nodes, segs = [nodes[0]], []
    stack = list(zip(nodes[:-1], nodes[1:]))[::-1]
    while stack:
        a, b = stack.pop()
        pts = p[a.i:b.i + 1]
        P, err = fit_cubic(pts, a.t_out, b.t_in)
        k = int(np.argmax(err))
        if err[k] > tol and len(pts) > 8 and 3 <= k <= len(pts) - 4:
            m = a.i + k
            t = _unit(p[min(m + 3, len(p) - 1)] - p[max(m - 3, 0)])
            mid = Node(m, "split", t, t)
            stack.append((mid, b))
            stack.append((a, mid))
            continue
        segs.append(P)
        out_nodes.append(b)
    return out_nodes, segs


def sample_beziers(segs, step=0.5):
    pts = []
    for P in segs:
        L = np.linalg.norm(np.diff(bez(P, np.linspace(0, 1, 50)), axis=0), axis=1).sum()
        n = max(int(L / step), 2)
        q = bez(P, np.linspace(0, 1, n + 1))
        pts.append(q if not pts else q[1:])
    return np.vstack(pts)


# ----------------------------------------------------------------------------
# 5. brush (width profile)
# ----------------------------------------------------------------------------


def arclen_u(p):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    return s / max(s[-1], 1e-9), s[-1]


def fit_width(u, r, tol=2.0, max_keys=24):
    """Compress r(u) into width keypoints (u_k, r_k) interpolated with PCHIP —
    the automated equivalent of placing points with Illustrator's Width Tool."""
    rs = gaussian_filter1d(r, 2)
    keys = {0, len(u) - 1}
    d = np.diff(rs)
    for i in range(1, len(d)):  # local extrema of width
        if np.sign(d[i - 1]) != np.sign(d[i]):
            keys.add(i)
    keys = sorted(keys)
    # drop extrema that are not significant (< tol change) then refine by max error
    pruned = [keys[0]]
    for k in keys[1:-1]:
        if abs(rs[k] - rs[pruned[-1]]) > tol:
            pruned.append(k)
    pruned.append(keys[-1])
    keys = sorted(set(pruned))
    while True:
        f = PchipInterpolator(u[keys], rs[keys])
        err = np.abs(f(u) - r)
        k = int(np.argmax(err))
        if err[k] <= tol or len(keys) >= max_keys or k in keys:
            break
        keys = sorted(set(keys) | {k})
    return [(float(u[k]), float(rs[k])) for k in keys], float(np.mean(err)), float(np.max(err))


def width_fn(keys):
    ku = np.array([k[0] for k in keys])
    kr = np.array([k[1] for k in keys])
    return PchipInterpolator(ku, kr)


def nib_test(p, r):
    """Is width explained by a broad-nib pen (w depends on stroke direction)?
    Fit r = a + b*|sin(theta - alpha)| over alpha; report best R^2."""
    T = np.gradient(p, axis=0)
    theta = np.arctan2(T[:, 1], T[:, 0])
    best = (-1, 0)
    var = np.var(r) + 1e-9
    for alpha in np.radians(np.arange(0, 180, 2)):
        x = np.abs(np.sin(theta - alpha))
        A = np.stack([np.ones_like(x), x], 1)
        coef, *_ = np.linalg.lstsq(A, r, rcond=None)
        r2 = 1 - np.var(r - A @ coef) / var
        if r2 > best[0]:
            best = (r2, math.degrees(alpha))
    return {"r2": float(best[0]), "nib_angle_deg": float(best[1])}


# ----------------------------------------------------------------------------
# 6. rendering + comparison
# ----------------------------------------------------------------------------


def _tangents(p):
    T = np.gradient(p, axis=0)
    T = np.stack([gaussian_filter1d(T[:, 0], 2), gaussian_filter1d(T[:, 1], 2)], 1)
    return T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)


def _march(m, origins, dirs, maxd, step=0.5):
    """Distance from each origin along its direction until leaving the ink."""
    steps = np.arange(0, maxd, step)
    c = origins[:, None, :] + dirs[:, None, :] * steps[None, :, None]
    v = ndimage.map_coordinates(m.astype(np.float32), [c[..., 1].ravel(), c[..., 0].ravel()],
                                order=1, cval=0).reshape(c.shape[:2])
    out = ~(v > 0.5)
    idx = np.where(out.any(1), out.argmax(1), len(steps) - 1)
    return steps[idx]


def extend_to_tips(p, m, start=True, end=True, maxd=600):
    """The skeleton stops ~one radius short of a stroke's real tip. Continue the
    pen path along its end tangent until it reaches the edge of the ink, so the
    path really starts where the letter starts and ends where it ends."""
    T = _tangents(p)
    if start:
        d = float(_march(m, p[:1], -T[:1], maxd)[0]) - 1.0
        if d > 1:
            ext = p[0] - T[0] * np.arange(d, 0, -1.0)[:, None]
            p = np.vstack([ext, p])
    if end:
        T = _tangents(p)
        d = float(_march(m, p[-1:], T[-1:], maxd)[0]) - 1.0
        if d > 1:
            ext = p[-1] + T[-1] * np.arange(1.0, d, 1.0)[:, None]
            p = np.vstack([p, ext])
    return p


def side_widths(q, m, owners=None, sid=0, maxd=600, step=0.5):
    """Asymmetric brush: distance from the pen path to the ink edge on the left
    and on the right of the direction of travel (Width Tool with Alt).

    A ray stops at the ink edge OR as soon as it enters ink that is closer to
    another part of the pen path (Voronoi ownership). Without this, rays near
    junctions and slit ends run lengthwise through the letter."""
    from scipy.spatial import cKDTree

    T = _tangents(q)
    N = np.stack([-T[:, 1], T[:, 0]], 1)
    if owners is None:
        owners = [(0, q)]
    allpts = np.vstack([o[1] for o in owners])
    sids = np.concatenate([np.full(len(o[1]), o[0]) for o in owners])
    idxs = np.concatenate([np.arange(len(o[1])) for o in owners])
    ok = np.isfinite(allpts).all(1)
    if sid != 0:  # a secondary stroke always owns its own samples
        ok |= sids == sid
        allpts = np.where(ok[:, None], allpts, 0)
        allpts[sids == sid] = q
    allpts, sids, idxs = allpts[ok], sids[ok], idxs[ok]
    tree = cKDTree(allpts)
    steps = np.arange(0, maxd, step)
    res = []
    for sign in (1, -1):
        c = q[:, None, :] + sign * N[:, None, :] * steps[None, :, None]
        v = ndimage.map_coordinates(m.astype(np.float32), [c[..., 1].ravel(), c[..., 0].ravel()],
                                    order=1, cval=0).reshape(c.shape[:2]) > 0.5
        _, nn = tree.query(c.reshape(-1, 2))
        nn = nn.reshape(c.shape[:2])
        own = (sids[nn] == sid) & (np.abs(idxs[nn] - np.arange(len(q))[:, None])
                                     <= np.maximum(4, 0.25 * steps[None, :]))
        stop = ~(v & own)
        idx = np.where(stop.any(1), stop.argmax(1), len(steps) - 1)
        res.append(steps[idx])
    return res[0], res[1], N


def render_ribbon(canvas, q, wl, wr):
    """Fill the stroke envelope as a chain of quads between left/right offsets."""
    T = _tangents(q)
    N = np.stack([-T[:, 1], T[:, 0]], 1)
    L = q + N * wl[:, None]
    R = q - N * wr[:, None]
    for i in range(len(q) - 1):
        quad = np.array([L[i], L[i + 1], R[i + 1], R[i]]) * 16
        cv2.fillConvexPoly(canvas, np.round(quad).astype(np.int32), 255, lineType=cv2.LINE_AA, shift=4)
    return L, R


def render_disc_stroke(canvas, centers, radii):
    """Round brush: union of discs along the path (sub-pixel, anti-aliased)."""
    for (x, y), r in zip(centers, radii):
        if r <= 0.3:
            continue
        cv2.circle(canvas, (int(round(x * 16)), int(round(y * 16))), int(round(r * 16)),
                   255, -1, lineType=cv2.LINE_AA, shift=4)


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------


def analyse(image_path: Path, box, name: str, scale=2, tol=4.0, smooth_sigma=16.0, single_path=True):
    mask = load_region(image_path, box, scale)
    comps = split_components(mask)
    dist = ndimage.distance_transform_edt(mask)
    recon = np.zeros(mask.shape, np.uint8)
    model = {"name": name, "scale": scale, "box": box, "components": []}
    vis = []
    for ci, c in enumerate(comps):
        m = c["mask"]
        entry = {"index": ci, "kind": c["kind"], "bbox": c["bbox"], "area": c["area"],
                 "circularity": round(c["circularity"], 3)}
        if c["kind"] == "dot":
            ys, xs = np.nonzero(m)
            cx, cy = xs.mean(), ys.mean()
            rr = math.sqrt(c["area"] / math.pi)
            cv2.circle(recon, (int(round(cx * 16)), int(round(cy * 16))), int(round(rr * 16)),
                       255, -1, lineType=cv2.LINE_AA, shift=4)
            entry.update(stamp="circle", center=[float(cx), float(cy)], radius=float(rr))
            model["components"].append(entry)
            vis.append({"kind": "dot", "center": (cx, cy), "r": rr})
            continue
        dist_c = ndimage.distance_transform_edt(m)
        strokes_out = []
        if single_path:
            # one smooth pen path per letter (hairpins into lobes, all nodes smooth, G2 faired)
            import pen_path as pp
            coords, walk_info = pp.single_walk(m, dist_c)
            p = smooth_open(resample(orient_rtl(coords), 1.0), smooth_sigma)
            p = extend_to_tips(p, m)
            nodes, segs = pp.fit_smooth(p, tol=tol, min_gap=30.0)
            fitted = [(p, nodes, segs, sample_beziers(segs, 1.0))]
            owners = [(0, fitted[0][3])]
            entry["walk"] = walk_info
        else:
            raw, _ = extract_strokes(m, dist_c)
            strokes_out = []
            fitted = []
            for si, poly in enumerate(raw):
                if si == 0:
                    poly = orient_rtl(poly)
                p = smooth_open(resample(poly, 1.0), smooth_sigma)
                p = extend_to_tips(p, m, start=(si == 0), end=True)
                nodes, segs = fit_stroke(p, tol=tol, min_gap=30.0)
                fitted.append((p, nodes, segs, sample_beziers(segs, 1.0)))
            # A secondary stroke starts inside its parent stroke; its first part
            # (within one brush radius of the attach point) must not "steal" ink
            # from the parent, or both strokes truncate each other -> cracks.
            owners = [(0, fitted[0][3])]
            for si, f in enumerate(fitted[1:], start=1):
                q0 = f[3]
                r_att = float(dist_c[int(q0[0][1]), int(q0[0][0])])
                keep_pts = np.linalg.norm(q0 - q0[0], axis=1) > 1.1 * r_att
                owners.append((si, np.where(keep_pts[:, None], q0, np.nan)))
        for si, (p, nodes, segs, q) in enumerate(fitted):
            uq, L = arclen_u(q)
            wl, wr, _ = side_widths(q, m, owners, si)
            kl, el_mean, el_max = fit_width(uq, wl)
            kr, er_mean, er_max = fit_width(uq, wr)
            wl_m = np.clip(width_fn(kl)(uq), 0, None)
            wr_m = np.clip(width_fn(kr)(uq), 0, None)
            render_ribbon(recon, q, wl_m, wr_m)
            r = (wl + wr) / 2
            strokes_out.append({
                "length_px": float(L),
                "nodes": [{"x": float(p[n.i][0]), "y": float(p[n.i][1]), "kind": n.kind} for n in nodes],
                "beziers": [P.round(2).tolist() for P in segs],
                "width_keys_left": [[round(a, 4), round(b, 2)] for a, b in kl],
                "width_keys_right": [[round(a, 4), round(b, 2)] for a, b in kr],
                "width_fit_err_px": {"mean": round((el_mean + er_mean) / 2, 2), "max": round(max(el_max, er_max), 2)},
                "nib_test": nib_test(q, r),
                "half_width_stats_px": {"min": float(r.min()), "max": float(r.max()), "mean": float(r.mean())},
                "asymmetry_mean": float(np.mean((wl - wr) / (wl + wr + 1e-9))),
            })
            if single_path:
                strokes_out[-1]["quality"] = pp.path_quality(segs, m, wl, wr)
            vis.append({"kind": "stroke", "p": q, "wl": wl, "wr": wr, "u": uq, "nodes": nodes, "segs": segs,
                        "kl": kl, "kr": kr, "comp": ci, "stroke": si})
        entry["strokes"] = strokes_out
        model["components"].append(entry)

    rec = recon > 127
    model["iou_word"] = iou(mask, rec)
    for e, c in zip(model["components"], comps):
        # IoU restricted to the neighbourhood of this component
        region = ndimage.binary_dilation(c["mask"], iterations=12)
        e["iou_local"] = iou(c["mask"], rec & region)
    n_nodes = sum(len(s["nodes"]) for e in model["components"] for s in e.get("strokes", []))
    n_keys = sum(len(s["width_keys_left"]) + len(s["width_keys_right"])
                 for e in model["components"] for s in e.get("strokes", []))
    model["summary"] = {"total_path_nodes": n_nodes, "total_width_keys": n_keys,
                        "components": [e["kind"] for e in model["components"]]}
    return mask, rec, model, vis


def draw_diagnostics(mask, rec, vis, model, prefix: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, W = mask.shape
    colors = ["#e4572e", "#29a3a3", "#f3a712", "#a8c686", "#669bbc", "#c03fc0"]

    # --- (a) pen-tool view: centerline Beziers with nodes and handles
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.imshow(np.where(mask, 0.82, 0.08), cmap="gray", vmin=0, vmax=1)
    for k, v in enumerate([v for v in vis if v["kind"] == "stroke"]):
        col = colors[k % len(colors)]
        q = sample_beziers(v["segs"], 1.0)
        ax.plot(q[:, 0], q[:, 1], color=col, lw=2.2)
        for P in v["segs"]:
            ax.plot([P[0][0], P[1][0]], [P[0][1], P[1][1]], color="#3a86ff", lw=1)
            ax.plot([P[3][0], P[2][0]], [P[3][1], P[2][1]], color="#3a86ff", lw=1)
            ax.plot([P[1][0], P[2][0]], [P[1][1], P[2][1]], "o", color="#3a86ff", ms=3.5)
        for P in v["segs"]:
            ax.plot(P[0][0], P[0][1], "s", color="white", mec="black", ms=6)
        ax.plot(v["segs"][-1][3][0], v["segs"][-1][3][1], "s", color="white", mec="black", ms=6)
        ax.annotate("start", v["p"][0], color=col, fontsize=11, xytext=(6, -8), textcoords="offset points")
        ax.arrow(*v["p"][len(v["p"]) // 2], *(v["p"][len(v["p"]) // 2 + 4] - v["p"][len(v["p"]) // 2]) * 3,
                 color=col, head_width=12, length_includes_head=True)
    for v in vis:
        if v["kind"] == "dot":
            ax.add_patch(plt.Circle(v["center"], v["r"], fill=False, color="#ff006e", lw=2))
            ax.plot(*v["center"], "+", color="#ff006e", ms=10)
    ax.set_axis_off()
    ax.set_title(f"Pen paths: {model['summary']['total_path_nodes']} nodes total", color="black")
    fig.tight_layout()
    fig.savefig(prefix.with_name(prefix.name + "_1_pen_paths.png"))
    plt.close(fig)

    # --- (b) brush: width profile along each stroke
    strokes = [v for v in vis if v["kind"] == "stroke"]
    fig, axs = plt.subplots(len(strokes), 1, figsize=(9, 2.4 * len(strokes)), squeeze=False)
    for k, v in enumerate(strokes):
        a = axs[k][0]
        uu = np.linspace(0, 1, 400)
        a.plot(v["u"], v["wl"], color="#bbb", lw=3, label="measured left")
        a.plot(v["u"], -v["wr"], color="#ddd", lw=3, label="measured right")
        a.plot(uu, width_fn(v["kl"])(uu), color=colors[k % len(colors)], lw=1.6, label="brush model")
        a.plot(uu, -width_fn(v["kr"])(uu), color=colors[k % len(colors)], lw=1.6)
        a.plot([kk[0] for kk in v["kl"]], [kk[1] for kk in v["kl"]], "D", color="black", ms=3.5,
               label="width points")
        a.plot([kk[0] for kk in v["kr"]], [-kk[1] for kk in v["kr"]], "D", color="black", ms=3.5)
        a.axhline(0, color="#888", lw=0.8)
        a.set_ylabel("half-width px")
        a.set_title(f"component {v['comp']} stroke {v['stroke']}", fontsize=9)
        a.legend(fontsize=7, loc="upper right")
    axs[-1][0].set_xlabel("position along stroke (0 = start, 1 = end)")
    fig.tight_layout()
    fig.savefig(prefix.with_name(prefix.name + "_2_brush_profile.png"), dpi=110)
    plt.close(fig)

    # --- (c) reconstruction vs original
    diff = np.zeros((H, W, 3), np.uint8)
    diff[mask & rec] = (235, 235, 235)
    diff[mask & ~rec] = (230, 60, 60)      # missing ink (red)
    diff[~mask & rec] = (60, 140, 255)     # extra ink (blue)
    cv2.imwrite(str(prefix.with_name(prefix.name + "_3_overlap.png")), diff[:, :, ::-1])
    side = np.hstack([np.where(mask, 255, 15).astype(np.uint8), np.full((H, 12), 90, np.uint8),
                      np.where(rec, 255, 15).astype(np.uint8)])
    cv2.imwrite(str(prefix.with_name(prefix.name + "_4_original_vs_rebuilt.png")), side)


def main():
    OUT.mkdir(exist_ok=True)
    img = HERE / "input" / "muzn_two_styles.jpg"
    box = tuple(int(v) for v in sys.argv[1:5]) if len(sys.argv) >= 5 else (1000, 360, 1850, 880)
    name = sys.argv[5] if len(sys.argv) >= 6 else "style1"
    mask, rec, model, vis = analyse(img, box, name)
    (OUT / f"{name}_model.json").write_text(json.dumps(model, ensure_ascii=False, indent=1))
    draw_diagnostics(mask, rec, vis, model, OUT / name)
    print(json.dumps({"iou_word": round(model["iou_word"], 4), **model["summary"]}, ensure_ascii=False))
    for e in model["components"]:
        for s in e.get("strokes", []):
            print(e["index"], e["kind"], "nodes", len(s["nodes"]), [n["kind"] for n in s["nodes"]],
                  "| width keys", len(s["width_keys_left"]) + len(s["width_keys_right"]), "| width err", s["width_fit_err_px"],
                  "| nib", s["nib_test"])
        if e["kind"] == "dot":
            print(e["index"], "dot r=", round(e["radius"], 1))


if __name__ == "__main__":
    main()
