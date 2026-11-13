"""python -m khatt.bench [-n FONTS] [-o DIR] [--fetch]

Runs the two examinations (measuring, letters) over the cached typefaces and writes
report.md, report.json and worst.png into DIR.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from ..critic import MARK_NAMES, SAME
from ..critic import SKELETONS as _SK
from ..letters import LETTERS
from ..pipeline import analyse
from ..style import Style
from ..typeset import flat, typeset
from . import fonts
from .score import BASE_ROW, CANVAS, PX, RIGHT_COL, Shape, distance, read
from .truth import FORMS, letter_mask, place, png, word_image, x_height

LOGO = "بيدوناق"
SKELETONS = ["ي" if c == "ی" else c for c in _SK]        # every Arabic typeface has ي; few have the Persian ی
FORM_AR = {"i": "منفصل", "I": "أول", "M": "وسط", "F": "آخر"}
TOL = {"x_height": 0.03, "stem": 0.06, "slant": 1.0}
X_PX = 130.0
TRUTH_CASE = "الصورة النظيفة: ارتفاع الحروف مقابل ارتفاع السنّ في الخط (±5%)"


# ------------------------------------------------------------------ measuring: known distortions
def _shear(im: Image.Image, deg: float) -> Image.Image:
    t = math.tan(math.radians(deg))
    W, H = im.size
    extra = int(abs(t) * H) + 2
    # a point at height (H - y) above the bottom moves right by that height times t
    return im.transform((W + extra, H), Image.AFFINE, (1, t, -t * H, 0, 1, 0), resample=Image.BICUBIC, fillcolor=255)


def _outline(im: Image.Image, px: int = 3) -> Image.Image:
    a = np.asarray(im) < 128
    return Image.fromarray(np.where(a & ~ndi.binary_erosion(a, iterations=px), 0, 255).astype(np.uint8))


def _icon(im: Image.Image) -> Image.Image:
    W, H = im.size
    out = Image.new("L", (W + H, H), 255); out.paste(im, (0, 0))
    ImageDraw.Draw(out).ellipse([W + 0.15 * H, 0.2 * H, W + 0.75 * H, 0.8 * H], fill=0)
    return out


def _ground(im: Image.Image) -> Image.Image:
    a = np.asarray(im).astype(float)
    return Image.fromarray((a * np.linspace(1.0, 0.5, a.shape[1])[None, :]).astype(np.uint8))


DISTORTIONS = [
    # name, picture, what the true answer is relative to the clean picture: (scale of lengths, added slant)
    ("دوران 5°", lambda im: im.rotate(5, expand=True, fillcolor=255, resample=Image.BICUBIC), 1.0, 0.0),
    ("دوران −8°", lambda im: im.rotate(-8, expand=True, fillcolor=255, resample=Image.BICUBIC), 1.0, 0.0),
    ("إمالة 12°", lambda im: _shear(im, 12), 1.0, 12.0),
    ("تصغير إلى 40%", lambda im: im.resize((int(im.width * 0.4), int(im.height * 0.4)), Image.LANCZOS), 0.4, 0.0),
    ("رمز بجانب الكلمة", _icon, 1.0, 0.0),
    ("خلفية متدرجة", _ground, 1.0, 0.0),
    ("أبيض على أسود", lambda im: Image.fromarray(255 - np.asarray(im)), 1.0, 0.0),
    ("حروف مفرغة", _outline, 1.0, 0.0),
]


def _numbers(im: Image.Image) -> Optional[Dict[str, float]]:
    try:
        an = analyse(png(im))
    except Exception:
        return None
    L, k = an["lines"][an["src"]], an["k"]
    if not L.stems:
        return None
    return {"x_height": L.x_height * k, "stem": L.stem * k, "slant": L.slant, "confidence": an["health"]["confidence"],
            "fixes": [n["kind"] for n in an["health"]["corrections"]], "style": an["style"]}


def measuring(face: fonts.Face) -> Optional[Dict]:
    im, _ = word_image(face, LOGO, X_PX)
    clean = _numbers(im)
    if clean is None:
        return None
    out = {"clean": {k: clean[k] for k in ("x_height", "stem", "slant", "confidence")}, "style": clean["style"], "cases": {}}
    # the one length a typeface states outright: the picture was drawn with its tooth X_PX pixels high
    e = clean["x_height"] / X_PX - 1
    out["cases"][TRUTH_CASE] = {"ok": abs(e) <= 0.05, "why": "" if abs(e) <= 0.05 else "x_height",
                                "errors": {"x_height": round(float(e), 4)}, "fixes": []}
    for name, fn, scale, slant in DISTORTIONS:
        got = _numbers(fn(im))
        if got is None:
            out["cases"][name] = {"ok": False, "why": "لم يُقَس"}
            continue
        errs = {"x_height": got["x_height"] / (clean["x_height"] * scale) - 1, "stem": got["stem"] / (clean["stem"] * scale) - 1,
                "slant": got["slant"] - (clean["slant"] + slant)}
        if name == "حروف مفرغة":
            errs["stem"] *= 0.5                         # the outline moves each edge by a pixel or two
        bad = [k for k, e in errs.items() if abs(e) > TOL[k]]
        out["cases"][name] = {"ok": not bad, "why": "، ".join(bad), "errors": {k: round(float(e), 4) for k, e in errs.items()},
                              "fixes": got["fixes"]}
    return out


# ------------------------------------------------------------------ letters: drawn against the typeface
def engine_shape(char: str, form: str, st: Style) -> Optional[Shape]:
    """The engine's letter in one form, its marks left out, on the comparison canvas."""
    text = FORMS[form].format(char)
    lay = typeset(text, st)
    placed = [p for p in lay.placed if p.char == char]
    if not placed:
        return None
    s = PX / st.x_height
    H, W = CANVAS
    m = np.zeros((H, 3 * W), bool)
    ox = 2.5 * W                                         # the glyph's right edge (x = 0) sits here
    for sh in flat(placed[0].glyph.shapes):
        if getattr(sh, "name", "") in MARK_NAMES:
            continue
        part = None
        for poly in sh:
            if len(poly) < 3:
                continue
            layer = Image.new("L", (3 * W, H), 0)
            ImageDraw.Draw(layer).polygon([(ox + x * s, BASE_ROW - y * s) for x, y in poly], fill=255)
            a = np.array(layer) > 0
            part = a if part is None else part ^ a
        if part is not None:
            m |= part
    return Shape(place(m, BASE_ROW, RIGHT_COL, PX, CANVAS))


def font_shapes(face: fonts.Face) -> Dict[Tuple[str, str], Shape]:
    X = x_height(face)
    cmap = face.tt.getBestCmap()
    out = {}
    for c in SKELETONS:
        if ord(c) not in cmap:
            continue
        for f in ("iIMF" if LETTERS[c][0] == "D" else "iF"):
            m = letter_mask(face, c, f, PX, CANVAS, BASE_ROW, RIGHT_COL, X)
            if m is not None and m.any():
                out[(c, f)] = Shape(m)
    return out


JOINED_SAME = [set("بني"), set("فق")]        # one body when joined to the next letter; only the dots differ


def _same(a: str, b: str, form: str = "") -> bool:
    """Two letters that share a body are one answer: the reading looks at bodies, the dots are left out."""
    if a == b or any(a in g and b in g for g in SAME):
        return True
    return form in ("I", "M") and any(a in g and b in g for g in JOINED_SAME)


KEY_FORMS = [("ب", "M"), ("ح", "I"), ("س", "M"), ("ص", "I"), ("ع", "M"), ("م", "I"), ("ه", "M"), ("ر", "i"), ("ل", "F"), ("و", "i")]


def _kin(truth: Dict[str, Dict], names: List[str], n: int = 3) -> Dict[str, List[str]]:
    """For every typeface, the `n` typefaces of other families that draw most like it: the fair
    yardstick is another designer working in the same style, not a Naskh against a Kufi."""
    fam = lambda s: s.rsplit(" ", 1)[0]
    out = {}
    for a in names:
        ds = []
        for b in names:
            if fam(a) == fam(b):
                continue
            v = [distance(truth[a][k], truth[b][k], sizes=False) for k in KEY_FORMS if k in truth[a] and k in truth[b]]
            if len(v) >= 6:
                ds.append((float(np.mean(v)), b))
        seen, pick = set(), []
        for _, b in sorted(ds):
            if fam(b) not in seen:
                seen.add(fam(b)); pick.append(b)
            if len(pick) == n:
                break
        out[a] = pick
    return out


def letters(faces: List[fonts.Face], styles: Dict[str, Style], log=print) -> Dict:
    truth = {}
    for face in faces:
        truth[face.label] = font_shapes(face)
        log(f"  حروف {face.label}: {len(truth[face.label])} شكلًا")
    names = [f.label for f in faces]
    kin = _kin(truth, names)
    rows = defaultdict(lambda: {"engine_ok": 0, "n": 0, "d": [], "font_ok": 0, "font_n": 0, "font_d": [], "reads": Counter()})
    per_font = {}
    for i, name in enumerate(names):
        st = styles.get(name)
        if st is None:
            continue
        mine = truth[name]
        ok = tot = 0
        for (c, f), ref in mine.items():
            refs = {cc: s for (cc, ff), s in mine.items() if ff == f}
            e = engine_shape(c, f, st)
            if e is None:
                continue
            got, _, _ = read(e, refs)
            r = rows[(c, f)]
            r["n"] += 1; r["engine_ok"] += _same(got, c, f); r["d"].append(distance(e, ref))
            if not _same(got, c, f):
                r["reads"][got] += 1
            ok += _same(got, c, f); tot += 1
            # the yardstick: other typefaces drawing the same letter, read against this one
            for other_name in kin.get(name, []):
                o = truth[other_name].get((c, f))
                if o is None:
                    continue
                g2, _, _ = read(o, refs)
                r["font_n"] += 1; r["font_ok"] += _same(g2, c, f); r["font_d"].append(distance(o, ref))
        per_font[name] = (ok, tot)
    out_rows = []
    for (c, f), r in rows.items():
        if not r["n"]:
            continue
        d, fd = float(np.median(r["d"])), float(np.median(r["font_d"])) if r["font_d"] else float("nan")
        out_rows.append({"letter": c, "form": f, "fonts": r["n"], "engine_read": round(r["engine_ok"] / r["n"], 3),
                         "fonts_read": round(r["font_ok"] / r["font_n"], 3) if r["font_n"] else None,
                         "distance": round(d, 3), "fonts_distance": round(fd, 3) if fd == fd else None,
                         "ratio": round(d / fd, 2) if fd == fd and fd > 0 else None,
                         "reads_as": r["reads"].most_common(1)[0][0] if r["reads"] else ""})
    out_rows.sort(key=lambda r: (r["engine_read"], -(r["ratio"] or 0)))
    e_ok = sum(r["engine_ok"] for r in rows.values()); e_n = sum(r["n"] for r in rows.values())
    f_ok = sum(r["font_ok"] for r in rows.values()); f_n = sum(r["font_n"] for r in rows.values())
    return {"rows": out_rows, "engine_read": round(e_ok / max(e_n, 1), 3), "fonts_read": round(f_ok / max(f_n, 1), 3),
            "per_font": {k: round(a / max(b, 1), 3) for k, (a, b) in per_font.items()}, "truth": truth}


# ------------------------------------------------------------------ pictures and report
def worst_sheet(res: Dict, faces: List[fonts.Face], styles: Dict[str, Style], n: int = 16) -> Optional[Image.Image]:
    """The letters the engine draws furthest from the typefaces: the typeface's letter in blue, the
    engine's in red, on the same lines."""
    truth = res["truth"]
    names = [f.label for f in faces if f.label in styles and truth.get(f.label)]
    if not names:
        return None
    H, W = CANVAS
    cells = []
    for row in res["rows"][:n]:
        c, f = row["letter"], row["form"]
        # the typeface on which this letter is at its median distance
        ds = []
        for name in names:
            ref = truth[name].get((c, f))
            if ref is not None:
                e = engine_shape(c, f, styles[name])
                if e is not None:
                    ds.append((distance(e, ref), name, e, ref))
        if not ds:
            continue
        ds.sort(key=lambda v: v[0])
        _, name, e, ref = ds[len(ds) // 2]
        rgb = np.full((H, W, 3), 255, np.uint8)
        rgb[ref.mask & ~e.mask] = (70, 110, 220); rgb[e.mask & ~ref.mask] = (220, 70, 60); rgb[e.mask & ref.mask] = (40, 40, 60)
        rgb[BASE_ROW, :] = (150, 200, 160); rgb[BASE_ROW - PX, :] = (150, 200, 160)
        # the part of the canvas a letter can reach, enlarged; numbered as its row in the report
        crop = Image.fromarray(rgb[BASE_ROW - int(2.5 * PX):BASE_ROW + int(1.4 * PX), RIGHT_COL - int(4.4 * PX):RIGHT_COL + int(0.5 * PX)])
        crop = crop.resize((crop.width * 3, crop.height * 3), Image.NEAREST)
        ImageDraw.Draw(crop).text((8, 6), str(len(cells) + 1), fill=(90, 90, 90))
        cells.append(crop)
    if not cells:
        return None
    per = 4
    cw, ch = cells[0].size
    rows_n = (len(cells) + per - 1) // per
    sheet = Image.new("RGB", (per * (cw + 6), rows_n * (ch + 6)), (225, 228, 236))
    for i, cimg in enumerate(cells):
        sheet.paste(cimg, ((per - 1 - i % per) * (cw + 6) + 3, (i // per) * (ch + 6) + 3))
    return sheet


def report(meas: Dict[str, Dict], let: Dict) -> str:
    md = ["# امتحان المحرك على الخطوط المفتوحة", "",
          f"عدد الخطوط: {len(meas)}. كلمة الشعار في كل خط: «{LOGO}».", "",
          "## 1. القياس: الكلمة نفسها بعد تغيير معلوم", "", "الجدول للخطوط التي قال المحرك إنه يقيسها بثقة.", "",
          "الجواب الصحيح معروف: ارتفاع الحروف وسمك الجذع لا يتغيران (أو يتغيران بنسبة التصغير)، والميل يزيد بمقدار الإمالة.",
          f"المسموح: ارتفاع الحروف ±{TOL['x_height'] * 100:.0f}%، الجذع ±{TOL['stem'] * 100:.0f}%، الميل ±{TOL['slant']:.0f}°.", "",
          "| التغيير | نجح | من | أكثر ما أخطأ فيه |", "|---|---|---|---|"]
    for name in [TRUTH_CASE] + [d[0] for d in DISTORTIONS]:
        cases = [m["cases"][name] for m in meas.values() if name in m["cases"] and m["clean"]["confidence"] >= 0.8]
        ok = sum(c["ok"] for c in cases)
        why = Counter(w for c in cases if not c["ok"] for w in (c["why"].split("، ") if c["why"] else ["—"]))
        md.append(f"| {name} | {ok} | {len(cases)} | {'، '.join(f'{k} ×{v}' for k, v in why.most_common(2)) or '—'} |")
    sure = {k: m for k, m in meas.items() if m["clean"]["confidence"] >= 0.8}
    tot = sum(c["ok"] for m in sure.values() for c in m["cases"].values()); n = sum(len(m["cases"]) for m in sure.values())
    rest = {k: m for k, m in meas.items() if k not in sure}
    tot2 = sum(c["ok"] for m in rest.values() for c in m["cases"].values()); n2 = sum(len(m["cases"]) for m in rest.values())
    md += ["", f"- خطوط قال المحرك إنه يقيسها بثقة (80% فأكثر): {len(sure)} خطًا، نجح فيها {tot} من {n} ({100 * tot / max(n, 1):.0f}%).",
           f"- خطوط أعلن أن قراءتها غير مؤكدة: {len(rest)} خطًا، نجح فيها {tot2} من {n2} ({100 * tot2 / max(n2, 1):.0f}%).", "",
           "## 2. الحروف: المحرك يرسم ما لم يره، ويُقارن بحرف الخط نفسه", "",
           "«يُقرأ صحيحًا» = أقرب حرف إليه في ذلك الخط (بالموضع نفسه) هو الحرف المقصود. المسافة بين الخطين المركزيين بوحدة ارتفاع الحروف.",
           "عمود «الخطوط» هو المعيار: أقرب ثلاثة خطوط من عائلات أخرى (مصمم آخر بالطراز نفسه) ترسم الحرف نفسه وتُقرأ بالطريقة نفسها.", "",
           f"- المحرك يُقرأ صحيحًا: **{let['engine_read'] * 100:.0f}%**",
           f"- خط آخر من الطراز نفسه يُقرأ صحيحًا (المعيار): **{let['fonts_read'] * 100:.0f}%**", "",
           "| الحرف | الموضع | المحرك يُقرأ صحيحًا | الخطوط تُقرأ صحيحًا | مسافة المحرك | مسافة الخطوط | النسبة | يُقرأ غالبًا |",
           "|---|---|---|---|---|---|---|---|"]
    for r in let["rows"]:
        fr = f"{r['fonts_read'] * 100:.0f}%" if r["fonts_read"] is not None else "—"
        md.append(f"| {r['letter']} | {FORM_AR[r['form']]} | {r['engine_read'] * 100:.0f}% | {fr} | {r['distance']} | "
                  f"{r['fonts_distance'] if r['fonts_distance'] is not None else '—'} | {r['ratio'] if r['ratio'] is not None else '—'} | {r['reads_as']} |")
    return "\n".join(md) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(prog="khatt.bench", description="امتحان المحرك على خطوط مفتوحة جوابها معروف")
    p.add_argument("-n", type=int, default=0, help="عدد الخطوط (0 = كل ما في المخزن)")
    p.add_argument("-o", "--out", default="bench_out")
    p.add_argument("--fetch", action="store_true", help="نزّل الخطوط الناقصة أولًا (يحتاج npm وشبكة)")
    a = p.parse_args()
    warnings.filterwarnings("ignore")
    files = fonts.ensure_fonts() if a.fetch else fonts.font_files()
    if not files:
        print("لا خطوط في المخزن. شغّل: python -m khatt.bench --fetch", file=sys.stderr)
        return 1
    if a.n:
        files = files[:: max(1, len(files) // a.n)][: a.n]
    faces, meas, styles = [], {}, {}
    for f in files:
        face = fonts.load(str(f))
        if any(ord(c) not in face.tt.getBestCmap() for c in LOGO):
            continue
        m = measuring(face)
        if m is None:
            print(f"  {face.label}: لم تُقَس كلمة الشعار"); continue
        styles[face.label] = m.pop("style")
        meas[face.label] = m
        faces.append(face)
        ok = sum(c["ok"] for c in m["cases"].values())
        print(f"  قياس {face.label}: {ok}/{len(m['cases'])}")
    let = letters(faces, styles)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sheet = worst_sheet(let, faces, styles)
    if sheet is not None:
        sheet.save(out / "worst.png")
    let.pop("truth")
    (out / "report.md").write_text(report(meas, let), encoding="utf-8")
    (out / "report.json").write_text(json.dumps({"logo": LOGO, "measuring": meas, "letters": let}, ensure_ascii=False, indent=1), encoding="utf-8")
    tot = sum(c["ok"] for m in meas.values() for c in m["cases"].values()); n = sum(len(m["cases"]) for m in meas.values())
    print(f"القياس: {tot}/{n} · الحروف: المحرك يُقرأ صحيحًا {let['engine_read'] * 100:.0f}% (المعيار بين الخطوط {let['fonts_read'] * 100:.0f}%)")
    print(f"النتائج في: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
