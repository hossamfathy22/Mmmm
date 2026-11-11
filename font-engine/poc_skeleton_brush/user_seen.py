#!/usr/bin/env python3
"""A fourth real letter: the seen the user drew in the diagram.

1. Extract the orange seen (and the orange reh, to get the scale: the reh is
   the word's zain without its dot) from input/user_diagram.png and bring it
   to the word's frame.
2. GROUND-TRUTH TEST: how close were the engine's earlier seens to the seen
   the designer actually drew? (IoU at the same scale, best translation.)
3. Run the engine on it: one pen path, exact brush, 99.9 % art brush check.
"""
from __future__ import annotations

import json

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

import compose as cp
import skeleton_brush as sb

SEEN_BOX = (217, 398, 68, 57)      # orange seen in the diagram (x, y, w, h)
REH_BOX = (516, 390, 54, 62)       # orange reh in the diagram


def orange(im):
    return (im[..., 0] > 150) & (im[..., 1] > 40) & (im[..., 1] < 170) & (im[..., 2] < 130)


def extract():
    im = np.array(Image.open(sb.HERE / "input" / "user_diagram.png").convert("RGB")).astype(np.float32)
    S = cp.load_style("style1")
    zb = S["comps"][S["ids"]["zain"]]["bbox"]
    k = zb[3] / REH_BOX[3]                                  # diagram px -> word frame (2x) px
    x, y, w, h = SEEN_BOX
    pad = 6
    crop = im[y - pad:y + h + pad, x - pad:x + w + pad]
    # soft orange-ness, upscaled smoothly, then thresholded: keeps the curve despite the low resolution
    soft = np.clip((crop[..., 0] - crop[..., 2]) / 180.0, 0, 1) * (crop[..., 1] < 190)
    big = cv2.resize(soft, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    big = cv2.GaussianBlur(big, (0, 0), 0.35 * k)
    m = big > 0.5
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    m = lab == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))
    return m, S, float(k)


def best_iou(a, b, search=40, step=4):
    """IoU of two masks at the same scale, best translation (centroids aligned first)."""
    H = max(a.shape[0], b.shape[0]) + 2 * search + 40
    W = max(a.shape[1], b.shape[1]) + 2 * search + 40

    def centre(m):
        c = np.zeros((H, W), bool)
        ys, xs = np.nonzero(m)
        oy, ox = H // 2 - int(ys.mean()), W // 2 - int(xs.mean())
        c[ys + oy, xs + ox] = True
        return c
    A, B = centre(a), centre(b)
    best = 0.0
    for dy in range(-search, search + 1, step):
        for dx in range(-search, search + 1, step):
            Bs = np.roll(np.roll(B, dy, 0), dx, 1)
            best = max(best, float((A & Bs).sum() / max((A | Bs).sum(), 1)))
    return best


def poly_mask(geom, scale=0.25):
    minx, miny, maxx, maxy = geom.bounds
    W, H = int((maxx - minx) * scale) + 6, int((maxy - miny) * scale) + 6
    m = np.zeros((H, W), np.uint8)
    for pg in ([geom] if geom.geom_type == "Polygon" else list(geom.geoms)):
        pts = ((np.array(pg.exterior.coords) - [minx, miny]) * scale + 3).astype(np.int32)
        cv2.fillPoly(m, [pts], 1)
        for hole in pg.interiors:
            cv2.fillPoly(m, [((np.array(hole.coords) - [minx, miny]) * scale + 3).astype(np.int32)], 0)
    return m > 0


def candidates(S):
    """Earlier seens of the engine, as masks at quarter scale."""
    from generate_and_build import ribbon_polygon
    from shapely.ops import unary_union
    out = {}
    noon = S["comps"][S["ids"]["noon"]]["mask"]
    out["the word's noon used as seen (no generation)"] = cv2.resize(noon.astype(np.uint8), None, fx=0.25, fy=0.25) > 0
    try:
        R, _, _ = cp.recipes(S, "round")
        p = R["seen"][0]
        out["hand recipe: 2 teeth + noon (compose.py)"] = poly_mask(ribbon_polygon(p.q, np.clip(p.wl, 1, None), np.clip(p.wr, 1, None)))
    except Exception as ex:
        print("skip compose", ex)
    try:
        import auto_sheet as ash
        _, auto, _ = ash.build_full("style1")
        out["automatic search over parts (auto_design.py)"] = poly_mask(auto["seen"]["body"])
    except Exception as ex:
        print("skip auto", ex)
    try:
        import curve_style as cs
        C = cs.CurveStyle("style1")
        G, _ = C.styled("seen.isol", "seen")
        p = C.brush(G, "seen")
        out["Naskh path restyled, one stroke (curve_style.py)"] = poly_mask(ribbon_polygon(p.q, p.wl, p.wr))
        import json as js
        import strokes_sakkal as ss
        data = js.loads((sb.OUT / "skeleton_corpus.json").read_text())[cs.nr.REF_FONT]
        strokes, _ = ss.build(C, "seen.isol", "seen", data["seen.isol"]["thickness"] * C.R.k)
        out["Sakkal strokes: each tooth its own stroke (strokes_sakkal.py)"] = poly_mask(unary_union([ribbon_polygon(s.q, s.wl, s.wr) for s in strokes]))
    except Exception as ex:
        print("skip curve", ex)
    return out


def main():
    import art_brush as ab
    import exact_brush as eb
    import pen_path as pp
    m, S, k = extract()
    cv2.imwrite(str(sb.OUT / "user_seen_mask.png"), (m * 255).astype(np.uint8))
    truth = cv2.resize(m.astype(np.uint8), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA) > 0
    test = {name: round(best_iou(truth, c), 3) for name, c in candidates(S).items()}
    # the engine on the real seen: pen path + brushes
    d = ndimage.distance_transform_edt(m)
    coords, walk = pp.single_walk(m, d)
    p = sb.smooth_open(sb.resample(sb.orient_rtl(coords), 1.0), 16.0)
    p = sb.extend_to_tips(p, m)
    hw = float(np.median(ndimage.map_coordinates(d, [p[:, 1], p[:, 0]], order=1)))
    nodes, segs = pp.fit_smooth(p, tol=4.0, min_gap=30.0, tol_reduce=0.25 * hw)
    q = sb.sample_beziers(segs, 1.0)
    wl, wr, _ = eb.measure(q, m)
    canv = np.zeros(m.shape, bool)
    ab.render(canv, q, ab.build(q, m))
    rep = {"diagram_to_word_scale": round(k, 2), "seen_size_px_2x": [int(m.shape[1]), int(m.shape[0])],
           "ground_truth_test_iou_vs_user_seen": test,
           "engine_on_real_seen": {"pen_path_nodes": len(segs) + 1, "hairpins": walk["hairpins"],
                                   "art_brush_iou": round(sb.iou(m, canv), 4),
                                   "median_stroke_width_px": round(float(np.median(wl + wr)), 1)}}
    np.savez(sb.OUT / "user_seen_stroke.npz", q=q, wl=wl, wr=wr, mask=m)
    (sb.OUT / "user_seen_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    # picture: truth and each candidate
    tiles = [("user's seen (truth)", truth)] + [(f"{v:.2f}  {n[:34]}", c) for (n, c), v in zip(candidates(S).items(), test.values())]
    Hm = max(t.shape[0] for _, t in tiles) + 30
    row = []
    for name, t in tiles:
        tile = np.full((Hm, t.shape[1] + 20, 3), 255, np.uint8)
        tile[20:20 + t.shape[0], 10:10 + t.shape[1]][t] = (255, 90, 54) if "truth" in name else (40, 40, 40)
        cv2.putText(tile, name, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1, cv2.LINE_AA)
        row.append(tile)
    cv2.imwrite(str(sb.OUT / "user_seen_test.png"), np.hstack(row)[:, :, ::-1])


if __name__ == "__main__":
    main()
