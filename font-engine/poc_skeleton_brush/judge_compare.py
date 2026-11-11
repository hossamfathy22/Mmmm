#!/usr/bin/env python3
"""Did the style judge make the generated letters better?

Compares the automatic designs found WITHOUT the judge (auto_design_style1.json)
and WITH it (auto_design_style1_judge.json) on:
  eye_p      the legibility eye (still reads as the right letter?)
  path_p     the path eye
  d_style    style distance from the uploaded letters (1 = a real designer letter's typical stray;
             the uploaded letters themselves reach `thr`)
  seen_iou   the user's own hand-drawn seen (ground truth never seen by either search)
Output: output/judge_compare.json, output/judge_compare.png
"""
from __future__ import annotations

import json

import cv2
import numpy as np

import auto_design as ad
import skeleton_brush as sb
import user_seen as us
from generate_and_build import ribbon_polygon


def main():
    runs = {"without judge": "auto_design_style1.json", "with judge": "auto_design_style1_judge.json"}
    ctx = ad.add_judge(ad.Ctx("style1"))
    ctx1 = ad.Ctx("style1", scale=1)
    truth_full, _, _ = us.extract()
    truth = cv2.resize(truth_full.astype(np.uint8), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA) > 0
    rep, masks = {"thr_uploaded_letters": round(ctx.judge_thr, 2)}, {}
    for run, fn in runs.items():
        data = json.loads((sb.OUT / fn).read_text())["auto"]
        rows = {}
        for t, r in data.items():
            if "genes" not in r:
                continue
            m = ad.raster(ad.build(r["genes"], ctx))[0]
            rows[t] = {"eye_p": round(r["info"]["p"], 2), "path_p": round(r["info"]["p_path"], 2),
                       "d_style": round(ctx.judge.distance(m), 2)}
            masks.setdefault(t, {})[run] = m
        p = ad.build(data["seen"]["genes"], ctx1)
        sm = us.poly_mask(ribbon_polygon(p.q, np.clip(p.wl, 1, None), np.clip(p.wr, 1, None)))
        n = len(rows)
        rep[run] = {"eye_accepts": f"{sum(v['eye_p'] >= 0.5 for v in rows.values())}/{n}",
                    "path_eye_accepts": f"{sum(v['path_p'] >= 0.5 for v in rows.values())}/{n}",
                    "within_style_tolerance": f"{sum(v['d_style'] <= ctx.judge_thr for v in rows.values())}/{n}",
                    "median_d_style": float(np.median([v["d_style"] for v in rows.values()])),
                    "seen_iou_vs_user_seen": round(us.best_iou(truth, sm), 3), "letters": rows}
        print(run, {k: v for k, v in rep[run].items() if k != "letters"})
    (sb.OUT / "judge_compare.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    # picture: each letter, without (top) and with (bottom) the judge
    tiles = []
    for t, d in masks.items():
        col = []
        for run in runs:
            m = d.get(run, np.zeros((10, 10), bool))
            tile = np.full((230, 210), 255, np.uint8)
            mm = m[:200, :200]
            tile[25:25 + mm.shape[0], 5:5 + mm.shape[1]][mm] = 0
            v = rep[run]["letters"].get(t, {})
            cv2.putText(tile, f"{t} p{v.get('eye_p', 0):.2f} d{v.get('d_style', 0):.1f}", (3, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, 0, 1, cv2.LINE_AA)
            col.append(tile)
        tiles.append(np.vstack(col))
    rows_ = [np.hstack(tiles[i:i + 9]) for i in range(0, len(tiles), 9)]
    W = max(r.shape[1] for r in rows_)
    img = np.vstack([np.pad(r, ((0, 10), (0, W - r.shape[1])), constant_values=255) for r in rows_])
    cv2.imwrite(str(sb.OUT / "judge_compare.png"), img)


if __name__ == "__main__":
    main()
