#!/usr/bin/env python3
"""Generate a new letter the way the designer does: start from the closest
letter that already exists in the word and change as LITTLE as possible.

Lesson of the first real test (user_seen.py): the designer's seen is almost
the word's noon plus a small extra tooth; every seen rebuilt from the Naskh
skeleton was further from it than the untouched noon (0.63).

For each target letter:
  base      the closest uploaded letter (seen, beh <- noon; reh, dal <- zain)
  edits     a family of small edits, ordered from the smallest to the largest
            seen : add k copies of the writer's own tooth to the right of the
                   noon (k from Sakkal's chart: up to 2 more), at several
                   sizes and spacings
            beh  : flatten / widen the noon's path (brush unchanged)
            dal  : shorten the zain's neck
            reh  : the zain itself
  choice    the SMALLEST edit that the legibility eye accepts as the target
            letter (p >= 0.5). The choice never looks at the designer's seen.
  test      the chosen seen is then compared with the designer's real seen
            (must beat 0.63, the untouched noon).
"""
from __future__ import annotations

import itertools
import json

import cv2
import numpy as np

import auto_design as ad
import compose as cp
import skeleton_brush as sb
import strokes_sakkal as ss
import user_seen as us


def tooth_copy(tooth, s, dx):
    t = cp.scale(tooth, s)
    f = s ** 0.5                                    # the brush follows the size only partly
    t.wl, t.wr = t.wl * f, t.wr * f
    # stand on the same base as the original tooth, dx to the right
    return cp.translate(t, (dx, tooth.q[-1, 1] - t.q[-1, 1]))


def families(S):
    st = S["strokes"]
    noon = cp.sub(st["noon"], 0, len(st["noon"][0]) - 1, "noon")
    zain = cp.sub(st["zain"], 0, len(st["zain"][0]) - 1, "zain")
    N = cp.split_noon(st["noon"], S["base_y"])
    Z = cp.split_zain(st["zain"])
    tooth = N["tooth"]
    lobe = float(np.median(tooth.wl + tooth.wr))
    fam = {}
    # seen: noon + k teeth
    cands = []
    for k, s, sp in itertools.product((1, 2), (0.5, 0.65, 0.8, 1.0), (0.6, 0.8, 1.0)):
        extra = [tooth_copy(tooth, s, (i + 1) * sp * lobe * (0.5 + 0.5 * s)) for i in range(k)]
        added = sum(float(((t.wl + t.wr)[1:] * np.linalg.norm(np.diff(t.q, axis=0), axis=1)).sum()) for t in extra)
        cands.append((added, f"noon + {k} tooth copies (size {s}, spacing {sp} lobe)", [noon] + extra))
    fam["seen"] = sorted(cands, key=lambda c: c[0])
    # beh: flatten / widen the noon path
    cands = []
    c0 = noon.q.mean(0)
    for sx, sy in itertools.product((1.0, 1.2, 1.4, 1.6), (1.0, 0.85, 0.7, 0.55)):
        p = noon.copy()
        p.q = c0 + (noon.q - c0) * [sx, sy]
        cands.append((abs(np.log(sx)) + abs(np.log(sy)), f"noon path widened x{sx}, flattened x{sy}", [cp.resample_part(p)]))
    fam["beh"] = sorted(cands, key=lambda c: c[0])
    # dal: shorten the zain's neck
    fam["dal"] = [(1 - f, f"zain with neck x{f}", [zain if f == 1.0 else cp.chain([Z["head"], cp.scale(Z["neck"], f), Z["tail"]])])
                  for f in (1.0, 0.85, 0.7, 0.55, 0.4, 0.3)]
    fam["reh"] = [(0.0, "zain without its dot", [zain])]
    return fam


def mask_of(parts):
    from generate_and_build import ribbon_polygon
    from shapely.ops import unary_union
    g = unary_union([ribbon_polygon(p.q, np.clip(p.wl, 1, None), np.clip(p.wr, 1, None)) for p in parts])
    return us.poly_mask(g), g


def main():
    S = cp.load_style("style1")
    ctxq = ad.Ctx("style1")
    eye = ad.Eye(ctxq)
    fam = families(S)
    truth_full, _, _ = us.extract()
    truth = cv2.resize(truth_full.astype(np.uint8), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA) > 0
    report, chosen = {}, {}
    for letter, cands in fam.items():
        rows, pick = [], None
        for size, name, parts in cands:
            p = ss.shape_score(parts, letter, ctxq, eye)
            rows.append({"edit": name, "edit_size": round(float(size), 3), "eye_p": round(p, 2)})
            if pick is None and p >= 0.5:
                pick = len(rows) - 1
        if pick is None:
            pick = int(np.argmax([r["eye_p"] for r in rows]))
        chosen[letter] = cands[pick][2]
        report[letter] = {"chosen": rows[pick], "candidates_tried": len(rows), "all": rows}
        print(f"{letter:5s} chosen: {rows[pick]['edit']}  (eye {rows[pick]['eye_p']}, {pick + 1} of {len(rows)} by size)")
    # ground truth: the designer's seen (never used for the choice)
    m, _ = mask_of(chosen["seen"])
    iou_sel = us.best_iou(truth, m)
    sens = []
    for size, name, parts in fam["seen"]:
        mm, _ = mask_of(parts)
        sens.append({"edit": name, "iou_vs_user_seen": round(us.best_iou(truth, mm), 3)})
    report["seen_ground_truth"] = {"chosen_edit_iou": round(iou_sel, 3), "untouched_noon_iou": 0.626,
                                   "best_previous_generator_iou": 0.585,
                                   "all_edits_iou (not used for the choice)": sens,
                                   "best_possible_in_family": max(s["iou_vs_user_seen"] for s in sens)}
    print("seen vs the designer's seen: chosen edit", round(iou_sel, 3), "| untouched noon 0.626 | best in family",
          report["seen_ground_truth"]["best_possible_in_family"])
    (sb.OUT / "minimal_edit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    # picture
    tiles = [("designer's seen", truth, (255, 90, 54))]
    for letter in ("seen", "beh", "dal", "reh"):
        mm, _ = mask_of(chosen[letter])
        tiles.append((f"{letter}: {report[letter]['chosen']['edit'][:38]}", mm, (40, 40, 40)))
    Hm = max(t.shape[0] for _, t, _ in tiles) + 30
    row = []
    for name, t, col in tiles:
        tile = np.full((Hm, max(t.shape[1] + 20, 260), 3), 255, np.uint8)
        tile[20:20 + t.shape[0], 10:10 + t.shape[1]][t] = col
        cv2.putText(tile, name, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1, cv2.LINE_AA)
        row.append(tile)
    cv2.imwrite(str(sb.OUT / "minimal_edit.png"), np.hstack(row)[:, :, ::-1])


if __name__ == "__main__":
    main()
