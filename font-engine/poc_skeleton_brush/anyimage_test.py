#!/usr/bin/env python3
"""Does (single smooth pen path + exact brush) rebuild ANY lettering?
Renders words in very different Arabic styles from the OFL corpus as test
images and rebuilds them; reports IoU and pen-path nodes."""
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import exact_brush as eb
import skeleton_brush as sb

TESTS = [("Amiri", "مزن", "Naskh, high contrast"), ("ReemKufi", "مزن", "geometric Kufi"),
         ("ArefRuqaa", "مزن", "Ruq'ah"), ("Lemonada", "سلام", "rounded, connected word with teeth"),
         ("Lalezar", "قلب", "heavy display, connected")]


def rebuild(img_path, box, name):
    mask, _, model, vis = sb.analyse(img_path, box, name)
    comps = sb.split_components(mask)
    canv = np.zeros(mask.shape, np.uint8)
    for v in vis:
        if v["kind"] == "dot":
            cx, cy = v["center"]
            cv2.circle(canv, (int(cx * 16), int(cy * 16)), int(v["r"] * 16), 255, -1, cv2.LINE_AA, 4)
            continue
        q, m = v["p"], comps[v["comp"]]["mask"]
        wl, wr, caps = eb.measure(q, m)
        u, L = sb.arclen_u(q)
        ml, kl = eb.compress_width(u, wl, L)
        mr, kr = eb.compress_width(u, wr, L)
        eb.render(canv, q, eb.eval_width(ml, kl, u), eb.eval_width(mr, kr, u), {k: eb.compress_cap(*caps[k]) for k in caps})
    rc = canv > 127
    d = np.zeros((*mask.shape, 3), np.uint8)
    d[mask & rc] = (235, 235, 235)
    d[mask & ~rc] = (230, 60, 60)
    d[~mask & rc] = (60, 140, 255)
    cv2.imwrite(str(sb.OUT / f"anyimage_{name}.png"), cv2.resize(d[:, :, ::-1], None, fx=0.5, fy=0.5))
    return sb.iou(mask, rc), model["summary"]["total_path_nodes"]


def main():
    out = {}
    for fam, txt, desc in TESTS:
        p = sb.HERE / "corpus" / f"{fam}.ttf"
        if not p.exists():
            continue
        W, H = 1920, 1080
        im = Image.new("L", (W, H), 0)
        f = ImageFont.truetype(str(p), 300, layout_engine=ImageFont.Layout.RAQM)
        ImageDraw.Draw(im).text((W // 2, H // 2), txt, font=f, fill=255, anchor="mm", direction="rtl", language="ar")
        path = sb.HERE / "input" / f"test_{fam}.png"
        im.convert("RGB").save(path)
        ys, xs = np.nonzero(np.array(im) > 128)
        box = (max(0, xs.min() - 40), max(0, ys.min() - 40), min(W, xs.max() + 40), min(H, ys.max() + 40))
        iou, nodes = rebuild(path, box, f"test_{fam}")
        out[f"{fam} «{txt}»"] = {"style": desc, "iou": round(iou, 4), "pen_path_nodes": nodes}
        print(fam, txt, round(iou, 4), nodes)
    (sb.OUT / "anyimage_report.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
