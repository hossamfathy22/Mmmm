#!/usr/bin/env python3
"""Proof sheet: original | rebuilt from pen+brush | typed with the generated font."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import skeleton_brush as sb

OUT = sb.OUT
W = 1800


def label(draw, xy, text, size=34):
    try:
        f = ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        f = ImageFont.load_default()
    draw.text(xy, text, fill=(150, 150, 150), font=f)


def typed(text, size, otf):
    font = ImageFont.truetype(str(otf), size, layout_engine=ImageFont.Layout.RAQM)
    tmp = Image.new("L", (W * 2, size * 3), 0)
    d = ImageDraw.Draw(tmp)
    d.text((50, size // 2), text, font=font, fill=255, direction="rtl", language="ar")
    bb = tmp.getbbox()
    return tmp.crop(bb) if bb else tmp


def main():
    img = Image.open(sb.HERE / "input" / "muzn_two_styles.jpg").convert("L")
    x0, y0, x1, y1 = (1000, 360, 1850, 880)
    orig = img.crop((x0, y0, x1, y1))
    mask, rec, model, vis = sb.analyse(sb.HERE / "input" / "muzn_two_styles.jpg", (x0, y0, x1, y1), "style1")
    rebuilt = Image.fromarray(np.where(rec, 255, 0).astype(np.uint8)).resize(orig.size, Image.LANCZOS)
    otf = OUT / "MuznBrush-Regular.otf"
    word = typed("مُزن", 420, otf)
    letters = typed("ا ب ت ث ح خ د ذ ر ز س ش ص ض", 200, otf)
    letters2 = typed("ط ظ ع غ ف ق ك ل م ن ه و ى ي", 200, otf)

    rows = [("1. uploaded image (style 1)", orig),
            (f"2. rebuilt from ONE smooth pen path per letter ({model['summary']['total_path_nodes']} nodes) + brush  (IoU {model['iou_word']:.3f})", rebuilt),
            ("3. typed with the generated font MuznBrush-Regular.otf (HarfBuzz/raqm shaping)", word),
            ("4. The alphabet (isolated letters), designed automatically from parts of the uploaded letters", letters),
            ("", letters2)]
    H = sum(r[1].height + 110 for r in rows) + 40
    sheet = Image.new("RGB", (W, H), (17, 17, 17))
    d = ImageDraw.Draw(sheet)
    y = 30
    for text, im in rows:
        label(d, (40, y), text)
        y += 60
        if im.width > W - 80:
            im = im.resize((W - 80, int(im.height * (W - 80) / im.width)), Image.LANCZOS)
        sheet.paste(Image.merge("RGB", [im] * 3), ((W - im.width) // 2, y))
        y += im.height + 50
    sheet.save(OUT / "PROOF_SHEET.png")
    print("saved", OUT / "PROOF_SHEET.png")


if __name__ == "__main__":
    main()
