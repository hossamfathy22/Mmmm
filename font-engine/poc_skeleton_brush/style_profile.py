#!/usr/bin/env python3
"""The engine's memory: one style profile per uploaded lettering.

Everything learned from an image is collected in one JSON file so the next
image ADDS to it instead of starting again:

  brush        which tool drew it (qalam / tablet / monoline), its settings,
               thickness of horizontal vs vertical travel, contrast type
  letters      for each uploaded letter: start and end shape and direction,
               size in nuqtas, number of pen nodes, how it deviates from the
               normal letter (scale, local shift, turning, length)
  parts        the reusable parts found (noon tooth/bowl, zain head/neck/tail)
  proportions  nuqta size, baseline, predicted boxes of the missing letters
  sources      which analysis produced each value (files in output/)
"""
from __future__ import annotations

import datetime
import json

import skeleton_brush as sb


def load(name):
    p = sb.OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def main(style="style1"):
    writer = load("writer_model.json") or {}
    brushes = load("brush_library_eval.json") or {}
    dev = load("style_deviation_report.json") or {}
    prop = load("proportions_report.json") or {}
    art = load("art_brush_report.json") or {}
    model = load(f"{style}_model.json") or {}
    profile_path = sb.OUT / "STYLE_PROFILE.json"
    old = json.loads(profile_path.read_text()) if profile_path.exists() else {"images": []}
    entry = {
        "image": "input/muzn_two_styles.jpg",
        "region": style,
        "word": "مُزن",
        "learned_at": datetime.date.today().isoformat(),
        "reconstruction_iou_art_brush": art.get(style, {}).get("iou"),
        "brush": {
            "selected_tool": brushes.get(style, {}).get("selected_brush"),
            "held_out_iou_per_tool": {k: v["mean_holdout_iou"] for k, v in brushes.get(style, {}).get("brushes", {}).items()},
            "contrast": writer.get(style, {}).get("contrast"),
        },
        "letters": {
            "terminals": writer.get(style, {}).get("terminals"),
            "deviation_from_normal": dev.get("how_the_writer_changed_each_letter") if style == "style1" else None,
            "pen_nodes_total": model.get("summary", {}).get("total_path_nodes"),
        },
        "parts": ["noon.tooth", "noon.bowl", "zain.head", "zain.neck", "zain.tail", "meem.end"],
        "proportions": {
            "uploaded_letters_in_nuqtas": prop.get("uploaded_word_in_nuqtas") if style == "style1" else None,
            "predicted_missing_letters_nuqtas": prop.get("predicted_for_uploaded_style") if style == "style1" else None,
            "closest_typefaces": prop.get("nearest_typefaces") if style == "style1" else None,
        },
    }
    imgs = [e for e in old["images"] if not (e["image"] == entry["image"] and e["region"] == style)]
    imgs.append(entry)
    profile = {"engine": "pen path + art brush, Arabic anatomy, nuqta proportions",
               "images": imgs, "letters_seen": sorted({l for e in imgs for l in ["meem.init", "zain.fina", "noon.isol"]})}
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=1))
    print("saved", profile_path, "| images in memory:", len(imgs))


if __name__ == "__main__":
    for s in ("style1", "style2"):
        main(s)
