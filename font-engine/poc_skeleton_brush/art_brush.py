#!/usr/bin/env python3
"""Art brush (skeletal stroke): 100 % reconstruction from pen path + brush.

The simple idea (Illustrator's Art Brush; Hsu & Lee, "Skeletal Strokes",
1993): the brush is not a width number, it is the stroke's whole outline
expressed in the PEN PATH'S OWN COORDINATES:

    s = position along the path (px, <0 before the start, >L after the end)
    d = signed distance from the path (left > 0, right < 0)

Every outline point of the real letter is stored as (s, d) relative to the
point of the pen path it SEES through the ink (exact_brush.visible_nearest).
Drawing = mapping (s, d) back around the path:  p = q(s) + d * N(s).
Holes (counters) are kept, caps are just the parts with s < 0 or s > L.

Compression: the outline in (s, d) space is a band, very simple to describe;
each continuous run is reduced with Ramer–Douglas–Peucker at `tol` px.

Because the brush lives in stroke coordinates, the SAME brush can later be
laid on another pen path (a new letter): its starts, ends, swellings and
thickness come with it.
"""
from __future__ import annotations

import json

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

import exact_brush as eb
import skeleton_brush as sb


class PathFrame:
    """Arc-length parameterised pen path with tangent/normal, extrapolated
    straight past both ends (for the caps)."""

    def __init__(self, q):
        self.q = q
        self.T = sb._tangents(q)
        self.N = np.stack([-self.T[:, 1], self.T[:, 0]], 1)
        self.s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
        self.L = self.s[-1]

    def to_stroke(self, pts, idx):
        """(s_anchor, along, d): position of the path sample the point is
        attached to, plus the offset in THAT sample's frame. Exact inverse."""
        v = pts - self.q[idx]
        along = (v * self.T[idx]).sum(1)
        d = (v * self.N[idx]).sum(1)
        return np.stack([self.s[idx], along, d], 1)

    def to_image(self, sad):
        if sad.shape[1] == 3:
            s0, along, d = sad[:, 0], sad[:, 1], sad[:, 2]
            sc = np.clip(s0, 0, self.L)
            x = np.interp(sc, self.s, self.q[:, 0])
            y = np.interp(sc, self.s, self.q[:, 1])
            T = np.stack([np.interp(sc, self.s, self.T[:, 0]), np.interp(sc, self.s, self.T[:, 1])], 1)
            T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
            N = np.stack([-T[:, 1], T[:, 0]], 1)
            return np.stack([x, y], 1) + along[:, None] * T + d[:, None] * N
        s, d = sad[:, 0], sad[:, 1]
        sc = np.clip(s, 0, self.L)
        x = np.interp(sc, self.s, self.q[:, 0])
        y = np.interp(sc, self.s, self.q[:, 1])
        T = np.stack([np.interp(sc, self.s, self.T[:, 0]), np.interp(sc, self.s, self.T[:, 1])], 1)
        T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
        N = np.stack([-T[:, 1], T[:, 0]], 1)
        extra = (s - sc)[:, None]                      # past an end: continue straight
        return np.stack([x, y], 1) + extra * T + d[:, None] * N


def contours(mask):
    """Exact boundary between ink and paper (marching squares at level 0.5),
    i.e. on the pixel EDGES, not on the boundary pixels' centres (which loses
    half a pixel all round — a lot on thin strokes)."""
    from skimage.measure import find_contours
    m = np.pad(mask.astype(float), 1)
    out = []
    for c in find_contours(m, 0.5):
        c = c[:, ::-1] - 1.0                      # (row, col) -> (x, y), undo padding
        if len(c) < 6:
            continue
        if np.allclose(c[0], c[-1]):
            c = c[:-1]
        out.append(c)
    return out


def build(q, mask, tol=0.5):
    """Art brush of one stroke: list of runs, each an (k, 2) array in (s, d)."""
    fr = PathFrame(q)
    brush = []
    for c in contours(mask):
        idx = eb.visible_nearest(q, c, mask)
        sad = fr.to_stroke(c, idx)
        # simplify the outline in image space (exact geometry), keep the
        # stroke coordinates of the surviving points
        keep = rdp2(np.vstack([c, c[:1]]), tol)
        keep = keep[keep < len(c)]
        brush.append([sad[keep]])
    return brush


def rdp2(P, tol):
    keep = np.zeros(len(P), bool)
    keep[[0, -1]] = True
    stack = [(0, len(P) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        A, B = P[a], P[b]
        AB = B - A
        L = np.linalg.norm(AB)
        seg = P[a + 1:b]
        if L < 1e-9:
            dist = np.linalg.norm(seg - A, axis=1)
        else:
            dist = np.abs((seg[:, 0] - A[0]) * AB[1] - (seg[:, 1] - A[1]) * AB[0]) / L
        k = int(np.argmax(dist))
        if dist[k] > tol:
            k += a + 1
            keep[k] = True
            stack += [(a, k), (k, b)]
    return np.nonzero(keep)[0]


def render(canvas, q, brush):
    """Exact rasterisation, even-odd rule (counters cancel), OR-ed into a
    boolean canvas. (cv2.fillPoly with anti-aliasing loses ~3 % at edges.)"""
    from skimage.draw import polygon as skpoly
    fr = PathFrame(q)
    acc = np.zeros(canvas.shape, bool)
    for contour_runs in brush:
        pts = np.vstack([fr.to_image(r) for r in contour_runs if len(r)])
        rr, cc = skpoly(pts[:, 1], pts[:, 0], canvas.shape)
        t = np.zeros(canvas.shape, bool)
        t[rr, cc] = True
        acc ^= t
    canvas |= acc


def n_points(brush):
    return int(sum(len(r) for c in brush for r in c))


def rebuild(img_path, box, name, tol=0.5):
    mask, _, model, vis = sb.analyse(img_path, box, name)
    comps = sb.split_components(mask)
    canv = np.zeros(mask.shape, bool)
    pts = 0
    for v in vis:
        if v["kind"] == "dot":
            cx, cy = v["center"]
            # dots are art-brushed too: a zero-length stroke at their centre
            m = comps[[i for i, c in enumerate(comps) if c["kind"] == "dot"
                       and abs(np.argwhere(c["mask"])[:, 1].mean() - cx) < 2][0]]["mask"]
            q = np.array([[cx - 0.5, cy], [cx + 0.5, cy]])
        else:
            q, m = v["p"], comps[v["comp"]]["mask"]
        b = build(q, m, tol)
        pts += n_points(b)
        render(canv, q, b)
    rc = canv
    d = np.zeros((*mask.shape, 3), np.uint8)
    d[mask & rc] = (235, 235, 235)
    d[mask & ~rc] = (230, 60, 60)
    d[~mask & rc] = (60, 140, 255)
    cv2.imwrite(str(sb.OUT / f"{name}_7_art_brush_overlap.png"), cv2.resize(d[:, :, ::-1], None, fx=0.5, fy=0.5))
    return {"iou": round(sb.iou(mask, rc), 4), "pen_path_nodes": model["summary"]["total_path_nodes"],
            "art_brush_points": pts, "pixels_wrong": int((mask ^ rc).sum()), "pixels_ink": int(mask.sum())}


def main():
    import anyimage_test as at
    img = sb.HERE / "input" / "muzn_two_styles.jpg"
    out = {"style1": rebuild(img, (1000, 360, 1850, 880), "style1"),
           "style2": rebuild(img, (120, 360, 860, 880), "style2")}
    for fam, txt, desc in at.TESTS:
        p = sb.HERE / "input" / f"test_{fam}.png"
        if not p.exists():
            continue
        g = np.array(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)) > 128
        ys, xs = np.nonzero(g)
        box = (max(0, xs.min() - 40), max(0, ys.min() - 40), min(g.shape[1], xs.max() + 40), min(g.shape[0], ys.max() + 40))
        out[f"{fam} «{txt}»"] = rebuild(p, box, f"test_{fam}")
    (sb.OUT / "art_brush_report.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for k, v in out.items():
        print(k, v)


if __name__ == "__main__":
    main()
