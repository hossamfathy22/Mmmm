"""A library of open-licensed (SIL OFL) Arabic typefaces, used as reference letterforms.

Fonts are downloaded once from the npm registry (@fontsource packages of Google Fonts families)
into a local cache; they are not shipped with this project. Text is shaped with HarfBuzz, so
every letter takes its correct joining form, and outlines are flattened into polygons.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fontTools.pens.basePen import BasePen
from fontTools.ttLib import TTFont

# SIL Open Font License families on Google Fonts with Arabic coverage, chosen to span the styles:
# naskh, kufi, geometric sans, rounded, display, calligraphic.
FAMILIES = ["amiri", "noto-naskh-arabic", "scheherazade-new", "lateef", "harmattan", "markazi-text", "mirza",
            "reem-kufi", "noto-kufi-arabic", "kufam", "changa", "cairo", "tajawal", "noto-sans-arabic", "mada",
            "ibm-plex-sans-arabic", "readex-pro", "vazirmatn", "alexandria", "almarai", "el-messiri", "lalezar",
            "lemonada", "baloo-bhaijaan-2", "marhey", "rakkas", "jomhuria", "katibeh", "aref-ruqaa", "blaka"]
WEIGHTS = (400, 700)
CACHE = Path(os.environ.get("KHATT_FONT_DIR", Path.home() / ".cache" / "khatt" / "fonts"))
LICENSE = "SIL Open Font License 1.1 (Google Fonts, via @fontsource)"

Contour = List[Tuple[float, float]]


def font_files() -> List[Path]:
    return sorted(CACHE.glob("*.woff2")) if CACHE.exists() else []


def ensure_fonts(families: Optional[List[str]] = None, weights=WEIGHTS, log=print) -> List[Path]:
    """Download the families that are not in the cache yet. Needs npm and network once."""
    CACHE.mkdir(parents=True, exist_ok=True)
    for fam in families or FAMILIES:
        want = [w for w in weights if not (CACHE / f"{fam}-{w}.woff2").exists()]
        if not want:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            try:
                out = subprocess.run(["npm", "pack", f"@fontsource/{fam}", "--silent"], cwd=tmp, capture_output=True,
                                     text=True, timeout=90)
                tgz = Path(tmp) / out.stdout.strip().splitlines()[-1]
                with tarfile.open(tgz) as tf:
                    for w in want:
                        name = f"package/files/{fam}-arabic-{w}-normal.woff2"
                        try:
                            data = tf.extractfile(name).read()
                        except (KeyError, AttributeError):
                            continue
                        (CACHE / f"{fam}-{w}.woff2").write_bytes(data)
            except Exception as e:                      # no npm, no network, unknown family: skip it
                log(f"تعذّر تنزيل الخط {fam}: {e}")
    return font_files()


@dataclass
class Face:
    name: str
    weight: int
    path: Path
    upm: int
    tt: TTFont = field(repr=False)
    hb_font: object = field(repr=False)

    @property
    def label(self) -> str:
        return f"{self.name} {self.weight}"


@lru_cache(maxsize=64)
def load(path: str) -> Face:
    import uharfbuzz as hb
    tt = TTFont(path)
    tt.flavor = None
    buf = io.BytesIO()
    tt.save(buf)
    face = hb.Face(buf.getvalue())
    stem = Path(path).stem
    fam, _, w = stem.rpartition("-")
    return Face(name=fam, weight=int(w) if w.isdigit() else 400, path=Path(path), upm=tt["head"].unitsPerEm, tt=tt,
                hb_font=hb.Font(face))


class _FlatPen(BasePen):
    """Collects contours, flattening curves into short straight segments."""

    def __init__(self, glyphset, dx: float, dy: float, steps: int = 8):
        super().__init__(glyphset)
        self.contours: List[Contour] = []
        self.cur: Contour = []
        self.dx, self.dy, self.steps = dx, dy, steps

    def _p(self, pt):
        return (pt[0] + self.dx, pt[1] + self.dy)

    def _moveTo(self, pt):
        self.cur = [self._p(pt)]

    def _lineTo(self, pt):
        self.cur.append(self._p(pt))

    def _curveToOne(self, p1, p2, p3):
        p0 = self.cur[-1]
        a, b, c = self._p(p1), self._p(p2), self._p(p3)
        for i in range(1, self.steps + 1):
            t = i / self.steps; m = 1 - t
            self.cur.append((m ** 3 * p0[0] + 3 * m * m * t * a[0] + 3 * m * t * t * b[0] + t ** 3 * c[0],
                             m ** 3 * p0[1] + 3 * m * m * t * a[1] + 3 * m * t * t * b[1] + t ** 3 * c[1]))

    def _qCurveToOne(self, p1, p2):
        p0 = self.cur[-1]
        a, b = self._p(p1), self._p(p2)
        for i in range(1, self.steps + 1):
            t = i / self.steps; m = 1 - t
            self.cur.append((m * m * p0[0] + 2 * m * t * a[0] + t * t * b[0], m * m * p0[1] + 2 * m * t * a[1] + t * t * b[1]))

    def _closePath(self):
        if len(self.cur) >= 3:
            self.contours.append(self.cur)
        self.cur = []

    _endPath = _closePath


def shape(face: Face, text: str) -> List[Tuple[int, List[Contour]]]:
    """Shape `text` and return, in visual order, (index of the source character, contours in font units)."""
    import uharfbuzz as hb
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(face.hb_font, buf, {})
    gs = face.tt.getGlyphSet()
    out = []
    x = 0.0
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        name = face.tt.getGlyphName(info.codepoint)
        pen = _FlatPen(gs, x + pos.x_offset, pos.y_offset)
        gs[name].draw(pen)
        out.append((info.cluster, pen.contours))
        x += pos.x_advance
    return out


if __name__ == "__main__":
    files = ensure_fonts()
    print(f"{len(files)} ملف خط في {CACHE} — {LICENSE}")
