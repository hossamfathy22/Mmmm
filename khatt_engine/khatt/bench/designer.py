"""python -m khatt.bench.designer DRAWING.pdf --line "..." [--line "..."] --logo LOGO.png --text "سیرنگ" -o OUT/ [--loo N]

The engine against a designer's own drawing of a logo's letters. A logo shows a few letters; a designer's
drawing of its alphabet (a PDF from Illustrator, read by khatt.drawn) shows the answer the engine is
asked for. For every letter form the designer drew:

  engine   the engine's letter from the logo alone (what it draws for any logo today)
  taught   the engine taught on the designer's other letters, the letter itself hidden (--loo)
  rule     the drawing path's rule: the letter hidden and rebuilt from the designer's other letters

each compared with the designer's letter: IoU at the best shift, both upright (the engine draws upright
and slants when it renders) at one x-height. The two x-heights (the drawing's rows, the engine's
measure of the logo) are tied by the letters the logo and the drawing share.

Also written: the engine taught on all the designer's letters (what its constructions can reach), and
that teaching as a profile of relations (lengths over the x-height, rules over their rule, the numbers
of the constructions, the variants, the marks) that can start the engine on another logo.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import re
import time
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

from ..drawn.build import build_glyphs
from ..drawn.layout import glyph_name, read
from ..drawn.sheet import _Flat, _font
from ..drawn.source import read_pdf
from ..editor import read_logo, style_of
from ..recipes import DECLARED, PARAMS, SKELETON, VARIANTS, rule_value
from ..style import Style
from ..typeset import flat, typeset
from . import bidonaq_fit as BF

warnings.filterwarnings("ignore")

XH = 70.0                                   # x-height of every mask, in pixels
FORM = {"isol": "i", "init": "I", "medi": "M", "fina": "F"}
FORM_AR = {"isol": "منفصل", "init": "أول", "medi": "وسط", "fina": "آخر"}
SCALES = (0.94, 0.97, 1.0, 1.03, 1.05, 1.06, 1.07, 1.08, 1.1, 1.13)
EXTRA_RATIOS = ["tail_drop", "dot_n"]       # switches of the style the designer's letters decide, beyond fit_word's


# ------------------------------------------------------------------ the two sides as masks
def _mask(polys, sc: float, tan: float = 0.0) -> np.ndarray:
    polys = [[(x - y * tan, y) for x, y in q] for q in polys if len(q) > 2]
    if not polys:
        return np.zeros((1, 1), bool)
    xs = [x for q in polys for x, _ in q]
    ys = [y for q in polys for _, y in q]
    x0, y1 = min(xs), max(ys)
    W, H = int((max(xs) - x0) * sc) + 6, int((y1 - min(ys)) * sc) + 6
    m = Image.new("1", (W, H), 0)
    for q in polys:
        t = Image.new("1", (W, H), 0)
        ImageDraw.Draw(t).polygon([((x - x0) * sc + 3, (y1 - y) * sc + 3) for x, y in q], fill=1)
        m = ImageChops.logical_xor(m, t)
    return BF.crop(np.array(m, bool))


def outline_mask(paths, sc: float, tan: float) -> np.ndarray:
    """Font outlines (khatt.drawn paths, y up) as a mask; `tan` undoes the drawing's slant."""
    polys = []
    for p in paths:
        pen = _Flat()
        p.draw(pen)
        polys += pen.polys
    return _mask(polys, sc, tan)


@dataclasses.dataclass
class Item:
    unit: str
    form: str
    token: str                               # the word it was drawn in, as the text names it
    truth: np.ndarray
    where: str


def token_of(where: str, unit: str) -> str:
    m = re.search("«(.+?)»", where)
    return m.group(1) if m else unit


def designer_items(R, scale: float) -> List[Item]:
    out, seen = [], set()
    for d in R.letters:
        key = (d.unit, d.form)
        if key in seen or len(d.unit) != 1:
            continue
        seen.add(key)
        out.append(Item(d.unit, d.form, token_of(d.where, d.unit),
                        outline_mask([d.body] + list(d.marks), scale * XH / R.x_height, R.tan), d.where))
    return out


def engine_letters(st: Style, items: List[Item]) -> Dict[Tuple[str, str], Optional[np.ndarray]]:
    """The engine's drawing of each item's letter form, set in the word the designer drew it in."""
    toks: Dict[str, List[Item]] = {}
    for it in items:
        toks.setdefault(it.token, []).append(it)
    out = {}
    for tok, its in toks.items():
        lay = typeset(tok, st)
        for it in its:
            p = next((p for p in lay.placed if p.char == it.unit and p.form == FORM[it.form]), None)
            out[(it.unit, it.form)] = BF.ras_engine(list(flat(p.glyph.shapes)), 0, XH / st.x_height) if p else None
    return out


def score_of(st: Style, items: List[Item]) -> List[float]:
    got = engine_letters(st, items)
    return [BF.iou(it.truth, got[(it.unit, it.form)]) if got.get((it.unit, it.form)) is not None
            and got[(it.unit, it.form)].any() else 0.0 for it in items]


def calibrate(R, logo: Style, text: str) -> Tuple[float, List[str]]:
    """The drawing's x-height against the engine's: the scale at which the letters the logo and the
    drawing share (the logo's word, drawn again by the designer) match best."""
    shared = [d for d in R.letters if token_of(d.where, d.unit) == text and len(d.unit) == 1]
    if not shared:
        return 1.0, []
    best = (-1.0, 1.0)
    for s in SCALES:
        its = [Item(d.unit, d.form, text, outline_mask([d.body] + list(d.marks), s * XH / R.x_height, R.tan), d.where)
               for d in shared]
        v = float(np.mean(score_of(logo, its)))
        best = max(best, (v, s))
    return best[1], list(dict.fromkeys(f"{d.unit}.{d.form}" for d in shared))


# ------------------------------------------------------------------ the drawing path's rule, held out
def rule_masks(R, items: List[Item], scale: float) -> Dict[Tuple[str, str], Tuple[Optional[np.ndarray], str]]:
    out = {}
    for it in items:
        key = (it.unit, it.form)
        R2 = dataclasses.replace(R, letters=[x for x in R.letters if (x.unit, x.form) != key])
        try:
            gl, _ = build_glyphs(R2)
        except Exception:
            out[key] = (None, "")
            continue
        g = next((g for g in gl if g.name == glyph_name(*key)), None)
        out[key] = ((outline_mask([g.path], scale * XH / R.x_height, R.tan), g.source)
                    if g is not None and g.source != "missing" else (None, ""))
    return out


# ------------------------------------------------------------------ teaching the engine
def _copy(st: Style, **kw) -> Style:
    return Style(**{**st.__dict__, "params": dict(st.params), "numbers": dict(st.numbers),
                    "variants": dict(st.variants), **kw})


def _reads(st: Style, items: List[Item]) -> List[str]:
    cache: dict = {}
    for tok in {it.token for it in items}:
        typeset(tok, st, cache=cache)
    return sorted({n for _, deps in cache.values() for n in deps if n in PARAMS or n.startswith("n:")})


def teach(start: Style, items: List[Item]) -> Tuple[Style, float]:
    """The engine's switches, variants, lengths, rules and numbers, set by the designer's letters."""
    def score(st):
        v = score_of(st, items)
        return float(np.mean(v)) if v else 0.0
    st, now = _copy(start), score(start)
    for f, v in [("dash_marks", True), ("dash_marks", False), ("angular", 0.0), ("angular", 0.6)]:
        t = _copy(st, **{f: v})
        if f == "angular" and v and not t.corner_r:
            t.corner_r = 0.3 * t.stem
        s = score(t)
        if s > now + 0.002:
            st, now = t, s
    forms = {f"{it.unit}.{FORM[it.form]}" for it in items}
    forms |= {f"{SKELETON.get(k.split('.')[0], k.split('.')[0])}.{k.split('.')[1]}" for k in forms}
    for key in sorted(forms):
        for name in [v.name for v in VARIANTS.get(key, [])] + [n for n, _ in DECLARED.get(key, ([], None))[0]]:
            t = _copy(st)
            t.variants[key] = name
            s = score(t)
            if s > now + 0.002:
                st, now = t, s
    saved = list(BF.RATIO)
    BF.RATIO[:] = list(dict.fromkeys(BF.RATIO + EXTRA_RATIOS))
    try:
        fitted, now = BF.fit_word(st, score, _reads(st, items))
    finally:
        BF.RATIO[:] = saved
    fitted.variants, fitted.dash_marks, fitted.angular, fitted.corner_r = st.variants, st.dash_marks, st.angular, st.corner_r
    return fitted, now


def profile_of(st: Style, logo: Style) -> Dict:
    """What the teaching changed, as relations a logo's own measures can carry to another logo."""
    lengths = {f: round(getattr(st, f) / st.x_height, 4) for f in BF.LEN if f != "x_height"}
    ratios = {f: round(float(getattr(st, f)), 4) for f in BF.RATIO + EXTRA_RATIOS}
    params = {}
    for name, v in st.params.items():
        r = rule_value(st, name)
        if r:
            params[name] = round(v / r, 4)
    return {"lengths_over_x_height": lengths, "ratios": ratios, "rules_over_rule": params,
            "numbers": {k: round(v, 4) for k, v in st.numbers.items() if v != 1.0},
            "variants": dict(st.variants), "dash_marks": bool(st.dash_marks), "angular": float(st.angular),
            "taught_from": "drawing", "logo_measures_kept": {f: round(getattr(logo, f), 2) for f in ("stem", "hair", "x_height", "slant")}}


# ------------------------------------------------------------------ one letter held out (in a worker)
_W: Dict = {}


def _init(state):
    _W.update(state)


def _held_out(i: int) -> Tuple[int, float]:
    items, logo = _W["items"], _W["logo"]
    rest = [it for j, it in enumerate(items) if j != i]
    st, _ = teach(logo, rest)
    got = engine_letters(st, [items[i]]).get((items[i].unit, items[i].form))
    return i, (BF.iou(items[i].truth, got) if got is not None and got.any() else 0.0)


# ------------------------------------------------------------------ the sheet
COLS = [("designer", "رسمك"), ("engine", "المحرك: الشعار"), ("taught", "المحرك: تعلّم"), ("rule", "قاعدة الرسم")]
COLOURS = {"designer": (20, 20, 20), "engine": (30, 90, 200), "taught": (0, 140, 90), "rule": (200, 120, 0)}


def sheet(rows, masks, path: str, per: int = 2, cw: int = 150, ch: int = 140) -> None:
    font = _font(16)
    n = len(rows)
    block = len(COLS) * cw + 120
    W, H = per * block, ((n + per - 1) // per) * ch + 40
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    for j in range(per):
        for c, (_, lab) in enumerate(COLS):
            d.text((W - j * block - 120 - c * cw - cw / 2, 8), lab, fill=(150, 40, 40), font=font, anchor="ma", direction="rtl")
    for i, r in enumerate(rows):
        j, row = i % per, i // per
        xr, y = W - j * block, 40 + row * ch
        d.text((xr - 10, y + 50), f"{r['letter']} {FORM_AR[r['form']]}", fill=(0, 0, 0), font=font, anchor="ra", direction="rtl")
        for c, (k, _) in enumerate(COLS):
            m = masks[(r["letter"], r["form"])].get(k)
            x0 = xr - 120 - (c + 1) * cw
            d.rectangle((x0 + 3, y + 3, x0 + cw - 3, y + ch - 3), outline=(215, 215, 215))
            if m is None or not m.any():
                d.text((x0 + cw / 2, y + 60), "—", fill=(160, 160, 160), font=font, anchor="ma")
                continue
            s = min((cw - 20) / m.shape[1], (ch - 45) / m.shape[0], 1.2)
            g = Image.fromarray((~m).astype(np.uint8) * 255).resize((max(1, int(m.shape[1] * s)), max(1, int(m.shape[0] * s))))
            im.paste(Image.new("RGB", g.size, COLOURS[k]), (int(x0 + (cw - g.width) / 2), int(y + 8)), Image.eval(g, lambda v: 255 - v))
            if k != "designer" and r.get(k) is not None:
                d.text((x0 + cw / 2, y + ch - 30), f"{r[k]:.2f}", fill=COLOURS[k], font=font, anchor="ma")
    im.save(path)


# ------------------------------------------------------------------ the examination
def run(drawing: str, lines: List[str], logo_image: str, logo_text: str, out_dir: str, page: int = 1,
        loo_workers: int = 0, loo_only: Optional[List[str]] = None) -> Dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    R = read(read_pdf(drawing, page - 1), lines)
    logo = style_of(read_logo(Path(logo_image).read_bytes(), logo_text)["style"])
    scale, shared = calibrate(R, logo, logo_text)
    items = designer_items(R, scale)
    engine = engine_letters(logo, items)
    rules = rule_masks(R, items, scale)
    taught_all, cap = teach(logo, items)
    taught_masks = engine_letters(taught_all, items)
    held: Dict[int, float] = {}
    if loo_workers:
        idx = [i for i, it in enumerate(items) if not loo_only or f"{it.unit}.{it.form}" in loo_only]
        with Pool(loo_workers, initializer=_init, initargs=({"items": items, "logo": logo},)) as pool:
            held = dict(pool.map(_held_out, idx))
    rows, masks = [], {}
    for i, it in enumerate(items):
        key = (it.unit, it.form)
        e, (rm, src), ta = engine.get(key), rules[key], taught_masks.get(key)
        row = {"letter": it.unit, "form": it.form, "token": it.token, "in_logo": it.token == logo_text,
               "engine": round(BF.iou(it.truth, e), 3) if e is not None and e.any() else None,
               "taught_all": round(BF.iou(it.truth, ta), 3) if ta is not None and ta.any() else None,
               "taught": round(held[i], 3) if i in held else None,
               "rule": round(BF.iou(it.truth, rm), 3) if rm is not None else None, "rule_source": src}
        cands = {k: row[k] for k in ("engine", "taught", "rule") if row[k] is not None}
        row["best"] = max(cands, key=cands.get) if cands else None
        rows.append(row)
        masks[key] = {"designer": it.truth, "engine": e, "taught": ta, "rule": rm}
    for r in rows:                                  # the sheet shows the honest number where there is one
        if r["taught"] is None:
            r["taught_sheet_note"] = "taught on all letters, this one included"
    sheet([{**r, "taught": r["taught"] if r["taught"] is not None else r["taught_all"]} for r in rows], masks,
          str(out / "designer_exam.png"))
    profile = profile_of(taught_all, logo)
    (out / "designer_profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=1))

    def mean(k, sel=lambda r: True):
        v = [r[k] for r in rows if r[k] is not None and sel(r)]
        return (round(float(np.mean(v)), 3), len(v)) if v else (None, 0)
    outside = lambda r: not r["in_logo"]
    summary = {"letters": len(rows), "scale": scale, "scale_from": shared,
               "engine_from_logo": mean("engine", outside), "taught_held_out": mean("taught", outside),
               "taught_all_capacity": mean("taught_all", outside), "rule_held_out": mean("rule", outside),
               "seconds": round(time.time() - t0)}
    report = {"summary": summary, "rows": rows}
    (out / "designer_exam.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    return report


def main() -> int:
    p = argparse.ArgumentParser(prog="khatt.bench.designer", description="the engine against a designer's drawing of a logo's letters")
    p.add_argument("drawing")
    p.add_argument("--line", action="append", required=True)
    p.add_argument("--logo", required=True, help="the logo the drawing was made from (picture)")
    p.add_argument("--text", required=True, help="the logo's word")
    p.add_argument("--page", type=int, default=1)
    p.add_argument("--loo", type=int, default=0, help="workers for the held-out teaching (0: skip; slow, minutes per letter)")
    p.add_argument("--loo-only", default="", help="hold out only these letter forms, e.g. \"ب.isol ص.isol\"")
    p.add_argument("-o", "--out", default="designer_exam")
    a = p.parse_args()
    r = run(a.drawing, a.line, a.logo, a.text, a.out, a.page, a.loo, a.loo_only.split() or None)
    s = r["summary"]
    print(f"{s['letters']} letter forms; x-height scale {s['scale']} (from {', '.join(s['scale_from']) or 'nothing shared'})")
    for k in ("engine_from_logo", "taught_held_out", "taught_all_capacity", "rule_held_out"):
        print(f"  {k}: {s[k][0]} over {s[k][1]} letters outside the logo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
