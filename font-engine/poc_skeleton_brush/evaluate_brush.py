#!/usr/bin/env python3
"""Honest test of brush transfer: hide one letter, learn the brush from the
other letters only, then paint the hidden letter's pen path with the learned
brush and compare with the real letter (IoU). Baseline: a constant-width
round brush with the median width of the training letters."""
import json

import numpy as np
from scipy import ndimage

import skeleton_brush as sb
from brush_model import BrushModel

LETTER = {0: "meem (init)", 2: "zain (fina)", 4: "noon (isol)"}


def main():
    img = sb.HERE / "input" / "muzn_two_styles.jpg"
    mask, rec, model, vis = sb.analyse(img, (1000, 360, 1850, 880), "style1")
    comps = sb.split_components(mask)
    ys = np.nonzero(mask)[0]
    y_top, y_bot = ys.min(), ys.max()
    strokes = [dict(q=v["p"], wl=v["wl"], wr=v["wr"], attached_start=v["stroke"] > 0, comp=v["comp"])
               for v in vis if v["kind"] == "stroke"]
    body_ids = sorted({s["comp"] for s in strokes if comps[s["comp"]]["kind"] == "body"})
    results = {}
    for hold in body_ids:
        train = [s for s in strokes if s["comp"] != hold and comps[s["comp"]]["kind"] == "body"]
        test = [s for s in strokes if s["comp"] == hold]
        bm = BrushModel().fit(train, y_top, y_bot)
        med = float(np.median(np.concatenate([np.r_[s["wl"], s["wr"]] for s in train])))
        canv_m = np.zeros(mask.shape, np.uint8)
        canv_c = np.zeros(mask.shape, np.uint8)
        canv_o = np.zeros(mask.shape, np.uint8)
        for s in test:
            wl, wr = bm.predict(s["q"], s["attached_start"])
            sb.render_ribbon(canv_m, s["q"], wl, wr)
            sb.render_ribbon(canv_o, s["q"], s["wl"], s["wr"])
            from brush_model import cap, stroke_features
            _, _, d0, d1 = stroke_features(s["q"], y_top, y_bot)
            c = cap(d0 if not s["attached_start"] else d0 + 1e9, bm.Rc) * cap(d1, bm.Rc)
            sb.render_ribbon(canv_c, s["q"], med * c, med * c)
        truth = comps[hold]["mask"]
        region = ndimage.binary_dilation(truth, iterations=40)
        results[LETTER.get(hold, hold)] = {
            "learned_brush_iou": round(sb.iou(truth, (canv_m > 127) & region), 3),
            "constant_round_brush_iou": round(sb.iou(truth, (canv_c > 127) & region), 3),
            "own_measured_brush_iou (upper bound)": round(sb.iou(truth, (canv_o > 127) & region), 3),
            "brush_learned_from": [LETTER.get(i, i) for i in body_ids if i != hold],
            "brush_params": bm.describe(),
        }
        # picture: truth vs prediction
        H, W = mask.shape
        x, y, w, h = comps[hold]["bbox"]
        pad = 40
        sl = (slice(max(0, y - pad), y + h + pad), slice(max(0, x - pad), x + w + pad))
        diff = np.zeros((H, W, 3), np.uint8)
        pm = canv_m > 127
        diff[truth & pm] = (235, 235, 235)
        diff[truth & ~pm] = (230, 60, 60)
        diff[~truth & pm & region] = (60, 140, 255)
        import cv2
        cv2.imwrite(str(sb.OUT / f"style1_5_holdout_{hold}.png"), diff[sl][:, :, ::-1])
    full = BrushModel().fit([s for s in strokes if comps[s["comp"]]["kind"] == "body"], y_top, y_bot)
    out = {"holdout": results, "brush_from_all_letters": full.describe(),
           "note": "pixel units are 2x the uploaded image"}
    (sb.OUT / "style1_brush_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
