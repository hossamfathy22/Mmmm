#!/usr/bin/env python3
"""Which drawing tool made this lettering?  Honest held-out test.

For every brush in the library (qalam broad nib, tablet pressure, monoline):
hide one letter, fit the brush on the other letters only, paint the hidden
letter's single pen path with it, and score IoU against the real letter.
The brush with the best mean held-out IoU is selected for the style.
"""
import json
import sys

import cv2
import numpy as np
from scipy import ndimage

import skeleton_brush as sb
from brush_model import BRUSHES

STYLES = {"style1": (1000, 360, 1850, 880), "style2": (120, 360, 860, 880)}
NAMES = {"style1": {0: "meem (init)", 2: "zain (fina)", 4: "noon (isol)"}}


def evaluate(style):
    box = STYLES[style]
    img = sb.HERE / "input" / "muzn_two_styles.jpg"
    mask, rec, model, vis = sb.analyse(img, box, style)
    comps = sb.split_components(mask)
    ys = np.nonzero(mask)[0]
    y_top, y_bot = ys.min(), ys.max()
    strokes = [dict(q=v["p"], wl=v["wl"], wr=v["wr"], attached_start=v["stroke"] > 0, comp=v["comp"])
               for v in vis if v["kind"] == "stroke"]
    bodies = sorted({s["comp"] for s in strokes if comps[s["comp"]]["kind"] == "body"})
    names = NAMES.get(style, {})
    res = {"iou_rebuilt_with_measured_brush": round(model["iou_word"], 4), "brushes": {}}
    for bname, B in BRUSHES.items():
        per = {}
        for hold in bodies:
            train = [s for s in strokes if s["comp"] != hold and comps[s["comp"]]["kind"] == "body"]
            test = [s for s in strokes if s["comp"] == hold]
            bm = B().fit(train, y_top, y_bot)
            canv = np.zeros(mask.shape, np.uint8)
            for s in test:
                wl, wr = bm.predict(s["q"], s["attached_start"])
                sb.render_ribbon(canv, s["q"], wl, wr)
            truth = comps[hold]["mask"]
            region = ndimage.binary_dilation(truth, iterations=40)
            per[names.get(hold, f"letter {hold}")] = round(sb.iou(truth, (canv > 127) & region), 3)
            if bname == "tablet" and style == "style1":
                diff = np.zeros((*mask.shape, 3), np.uint8)
                pm = canv > 127
                diff[truth & pm] = (235, 235, 235)
                diff[truth & ~pm] = (230, 60, 60)
                diff[~truth & pm & region] = (60, 140, 255)
                x, y, w, h = comps[hold]["bbox"]
                cv2.imwrite(str(sb.OUT / f"{style}_5_holdout_{hold}.png"),
                            diff[max(0, y - 40):y + h + 40, max(0, x - 40):x + w + 40][:, :, ::-1])
        full = B().fit([s for s in strokes if comps[s["comp"]]["kind"] == "body"], y_top, y_bot)
        res["brushes"][bname] = {"tool": getattr(B, "kind", bname), "holdout_iou": per,
                                 "mean_holdout_iou": round(float(np.mean(list(per.values()))), 3),
                                 "params_from_all_letters": full.describe()}
    best = max(res["brushes"], key=lambda k: res["brushes"][k]["mean_holdout_iou"])
    res["selected_brush"] = best
    return res


def main():
    styles = sys.argv[1:] or list(STYLES)
    out = {s: evaluate(s) for s in styles}
    (sb.OUT / "brush_library_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for s, r in out.items():
        print(s, "selected:", r["selected_brush"])
        for b, v in r["brushes"].items():
            print(f"   {b:9s} mean {v['mean_holdout_iou']}  {v['holdout_iou']}")


if __name__ == "__main__":
    main()
