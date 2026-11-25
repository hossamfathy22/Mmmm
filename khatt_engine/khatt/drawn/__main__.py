"""python -m khatt.drawn drawing.pdf --line "ا ب حـ د ر" --line "فاطمة بيدوناق" -o out/ [--page 1] [--family NAME]

Reads the designer's vector drawing with the text of each line (as it reads), builds every letter in every
form, writes the font (OTF) and a report of where each glyph came from."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from .build import build_glyphs, write_font
from .layout import read
from .source import read_pdf


def run(drawing: str, lines, out_dir: str, page: int = 1, family: str = "Khatt Drawn") -> dict:
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pieces = read_pdf(drawing, page - 1)
    R = read(pieces, lines)
    glyphs, notes = build_glyphs(R)
    font_path = out / (("".join(c for c in family if c.isalnum()) or "KhattDrawn") + ".otf")
    info = write_font(glyphs, R, str(font_path), family)
    by = Counter(g.source for g in glyphs)
    report = {
        "font": str(font_path), "seconds": round(time.time() - t0, 2),
        "measured": {"slant_deg": round(R.slant, 2), "x_height_units": round(R.x_height),
                     "connector_pt": round(R.band / R.scale, 2), "spacing_pt": round((R.spacing or 0) / R.scale, 1),
                     "dot": R.dot is not None},
        "letters_read": [{"letter": d.unit, "form": d.form, "where": d.where} for d in R.letters],
        "glyph_sources": dict(by),
        "glyphs": [{"name": g.name, "source": g.source, "how": g.how} for g in glyphs],
        "missing": [g.name for g in glyphs if g.source == "missing"],
        "notes": R.notes + notes, **info,
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    return report


def main() -> int:
    p = argparse.ArgumentParser(prog="khatt.drawn", description="a font from a designer's vector drawing")
    p.add_argument("drawing", help="PDF exported from Illustrator (vectors, not a picture)")
    p.add_argument("--line", action="append", required=True, help="the text of a line of the drawing, as it reads")
    p.add_argument("--page", type=int, default=1)
    p.add_argument("-o", "--out", default="drawn_out")
    p.add_argument("--family", default="Khatt Drawn")
    a = p.parse_args()
    r = run(a.drawing, a.line, a.out, a.page, a.family)
    m = r["measured"]
    print(f"read in {r['seconds']} s: slant {m['slant_deg']}°, connector {m['connector_pt']} pt, spacing {m['spacing_pt']} pt")
    print(f"letters read from the drawing: {len(r['letters_read'])}")
    print("glyphs: " + ", ".join(f"{k} {v}" for k, v in sorted(r["glyph_sources"].items())))
    if r["missing"]:
        print("missing: " + " ".join(r["missing"]))
    for n in r["notes"]:
        print("note:", n)
    print("font:", r["font"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
