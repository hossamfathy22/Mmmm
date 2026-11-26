"""The reading sheet: every letter as it was read from the drawing, with the letter and form the text gave
it. A text that does not match the drawing (a word too many, two letters swapped) shows here at a glance,
where no number catches it."""
from __future__ import annotations

from typing import List

from fontTools.pens.basePen import BasePen
from PIL import Image, ImageChops, ImageDraw, ImageFont

from . import geom as G

FORM_AR = {"isol": "منفصل", "init": "أول", "medi": "وسط", "fina": "آخر"}


class _Flat(BasePen):
    """Contours as polygons (curves cut into short lines)."""

    def __init__(self):
        super().__init__(None)
        self.polys: List[list] = []
        self.cur: list = []

    def _moveTo(self, p):
        self.cur = [p]

    def _lineTo(self, p):
        self.cur.append(p)

    def _curveToOne(self, a, b, c):
        p0 = self.cur[-1]
        for i in range(1, 13):
            t = i / 12
            u = 1 - t
            self.cur.append((u ** 3 * p0[0] + 3 * u * u * t * a[0] + 3 * u * t * t * b[0] + t ** 3 * c[0],
                             u ** 3 * p0[1] + 3 * u * u * t * a[1] + 3 * u * t * t * b[1] + t ** 3 * c[1]))

    def _qCurveToOne(self, a, b):
        p0 = self.cur[-1]
        for i in range(1, 9):
            t = i / 8
            u = 1 - t
            self.cur.append((u * u * p0[0] + 2 * u * t * a[0] + t * t * b[0], u * u * p0[1] + 2 * u * t * a[1] + t * t * b[1]))

    def _closePath(self):
        if len(self.cur) > 2:
            self.polys.append(self.cur)
        self.cur = []

    _endPath = _closePath


def _font(size: int):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.RAQM)
        except Exception:
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def reading_sheet(letters, path: str, cols: int = 10, cell: int = 230) -> str:
    """letters: layout.Drawn in reading order. Cells run right to left, as the drawing reads."""
    ch = cell + 40
    rows = max(1, (len(letters) + cols - 1) // cols)
    im = Image.new("L", (cols * cell, rows * ch), 255)
    draw = ImageDraw.Draw(im)
    font = _font(22)
    small = _font(15)
    for i, d in enumerate(letters):
        c = cols - 1 - i % cols
        r = i // cols
        x0, y0 = c * cell, r * ch
        paths = [d.body] + list(d.marks)
        boxes = [b for b in (G.bounds(p) for p in paths) if b]
        if not boxes:
            continue
        bx0 = min(b[0] for b in boxes)
        bx1 = max(b[2] for b in boxes)
        s = min((cell - 40) / max(bx1 - bx0, 1.0), (cell - 70) / 900.0)
        ox = x0 + cell / 2 - (bx0 + bx1) / 2 * s
        base = y0 + 0.68 * cell
        ink = Image.new("1", im.size, 0)
        for p in paths:
            pen = _Flat()
            p.draw(pen)
            for poly in pen.polys:
                t = Image.new("1", im.size, 0)
                ImageDraw.Draw(t).polygon([(ox + x * s, base - y * s) for x, y in poly], fill=1)
                ink = ImageChops.logical_xor(ink, t)
        im.paste(0, (0, 0), ink)
        draw.line((x0 + 8, base, x0 + cell - 8, base), fill=215)
        draw.rectangle((x0 + 2, y0 + 2, x0 + cell - 2, y0 + ch - 2), outline=200)
        label = f"{d.unit}  {FORM_AR.get(d.form, d.form)}"
        draw.text((x0 + cell / 2, y0 + cell - 18), label, fill=40, font=font, anchor="ma", direction="rtl")
        draw.text((x0 + cell / 2, y0 + cell + 12), d.where, fill=130, font=small, anchor="ma", direction="rtl")
    im.save(path)
    return path
