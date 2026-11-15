"""The whole engine, stage by stage:

  1. load       grey image, ink/background, working scale
  2. measure    lines, slant, sub-pixel stems, pen rule (skeleton fit), grid levels, rhythm, pieces
  3. read       joins, tails, bars, script, letters (or align a typed text), observations
  4. parts      cut the logo into reusable parts and trace them as Bezier curves
  5. build      set the new word from the logo's parts (engine parts only where the logo has none)
  6. verify     measure the new word with the same engine and compare it with the logo's rules
  7. draw       logo construction sheet, part box, pen rule, word construction sheet, report
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .analyze import find_lines
from .guard import assess, prepare
from .recipes import choices, handles
from .selfcheck import check_and_take, prune_parts
from .bezier import to_path
from .features import measure_features
from .library import ACTIVE, build_library
from .measure import Line, measure_line
from .render import sheet_svg, to_png, to_svg
from .sheets import ideal_parts_sheet, logo_sheet, parts_sheet, pen_sheet, skeleton_sheet, specimen_png, thickness_png
from .critic import critique, specimen_defects
from .skeleton import read_skeleton
from .style import Style
from .typeset import typeset
from .understand import ROLE_AR, Reading, observations, read_line
from .verify import verify_and_correct

STAGES = ["القياس", "الفهم", "صندوق القطع", "البناء", "التحقق", "المخططات"]
FORM_AR = {"i": "منفصل", "I": "أول", "M": "وسط", "F": "آخر"}


def _r(v: float, k: float = 1.0, n: int = 2):
    return round(float(v) * k, n) if v == v else None


def line_json(L: Line, R: Reading, k: float) -> Dict[str, Any]:
    return {
        "line": L.idx + 1, "script": R.script, "text_guess": R.text_guess, "text_given": R.text_given,
        "slant_deg": L.slant, "stem": _r(L.stem, k), "stem_runs": _r(L.stem_runs, k), "hairline": _r(L.hair, k),
        "contrast": _r(L.stem / L.hair, 1) if L.hair == L.hair and L.hair else None, "monoline": L.monoline,
        "pen_rule": {"stem": _r(L.pen_fit.get("stem", float("nan")), k), "hair": _r(L.pen_fit.get("hair", float("nan")), k),
                     "power": _r(L.pen_fit.get("power", float("nan"))), "median_error": _r(L.pen_fit.get("median_error", float("nan")), k),
                     "samples": L.pen_fit.get("samples", 0)} if L.pen_fit else None,
        "x_height": _r(L.x_height, k), "ascender": _r(L.ascender, k), "descender": _r(L.descender, k),
        "white_between_stems": _r(L.white, k), "stem_pitch": _r(L.pitch, k),
        "levels": [{"y": _r(v.y, k), "count": v.n, "name": v.name} for v in L.levels],
        "stems": [{"x": _r(s.x, k), "width": _r(s.w, k), "top": _r(s.top, k), "bottom": _r(s.bottom, k), "role": s.role,
                   "role_ar": ROLE_AR.get(s.role, ""), "letter": s.letter} for s in L.stems],
        "joins": [{"right": j.right, "left": j.left, "white": _r(j.white, k), "bottom": _r(j.bottom, k), "kind": j.kind} for j in R.joins],
        "tails": [{"direction": t.direction, "depth": _r(-t.y0, k)} for t in R.tails],
        "bars": [{"length": _r(b.x1 - b.x0, k), "thick": _r(b.thick, k), "height": _r(b.y, k), "attached": b.attached} for b in R.bars],
        "dots": [{"diameter": _r(p.size, k), "roundness": _r(p.roundness), "center_above_baseline": _r(p.cy, k)} for p in L.dots],
        "dashes": [{"length": _r(p.size, k), "thickness": _r(p.thick, k), "center_above_baseline": _r(p.cy, k)} for p in L.dashes],
        "counters": [{"w": _r(c.x1 - c.x0, k), "h": _r(c.y1 - c.y0, k)} for c in L.counters],
        "letters": [{"letter": u.letter, "sure": u.sure, "stems": u.stems} for u in R.units],
        "ink_coverage": _r(L.coverage, 1, 3), "width": _r(L.width, k), "height": _r(L.height, k),
    }


def pick_line(lines: List[Line], readings: List[Reading], line: Optional[int]) -> int:
    usable = [i for i, L in enumerate(lines) if L.stems]
    if not usable:
        raise ValueError("لم أجد في الصورة كتابة يمكن قياسها. استعمل صورة أوضح بخلفية سادة.")
    if line is not None and line in usable:
        return line
    arabic = [i for i in usable if readings[i].script == "arabic"]
    return (arabic or usable)[0]


def style_from(L: Line, lib, k: float, feats: Optional[Dict[str, Dict]] = None) -> Style:
    X = L.x_height * k
    asc = L.ascender * k
    dot = lib.parts["dot"].info["diameter"] if "dot" in lib.parts else 1.1 * L.stem * k
    base = Style(stem=L.stem * k, hair=(L.stem if L.monoline else L.hair) * k, x_height=X,
                 ascender=asc if asc >= 0.3 * X else 0.55 * X,
                 # the descender is made equal to the ascender (vertical balance), never shorter than the logo's
                 descender=max(L.descender * k, asc if asc >= 0.3 * X else 0.55 * X),
                 slant=L.slant, white=(L.white if L.white == L.white else L.stem) * k, dot=dot,
                 dash_marks=bool(L.dashes) or not L.dots, contrast_power=float(np.clip(L.power, 0.8, 6.0)))
    f = {k_: v["value"] for k_, v in (feats or {}).items()}
    st = Style(**{**base.__dict__})
    if "top_angle" in f: st.top_angle = float(np.clip(f["top_angle"], -40, 40))
    if "top_round" in f: st.top_round = float(np.clip(f["top_round"], 0, 1))
    if "join_in" in f: st.join_in = float(np.clip(f["join_in"], 0.12, 0.7))
    if "join_out" in f: st.join_out = float(np.clip(f["join_out"], 0.12, 0.7))
    if st.join_out < st.join_in - 0.15:     # the outer curve never starts much lower than the inner one
        st.join_out = st.join_in
    if "ring_n" in f: st.ring_n = float(np.clip(f["ring_n"], 1.6, 8))
    elif st.top_round >= 0.5: st.ring_n = 2.0
    elif st.top_round == 0 and st.top_angle == 0: st.ring_n = 2.3
    if "dot_n" in f: st.dot_n = float(np.clip(f["dot_n"], 1.6, 10))
    if "dot_rot" in f: st.dot_rot = f["dot_rot"]
    if "dot_size" in f: st.dot = float(f["dot_size"])
    if "tail_reach" in f: st.tail_reach = float(np.clip(f["tail_reach"], 1.5, 5))
    if "overshoot" in f: st.overshoot = float(np.clip(f["overshoot"], 0, 0.06 * st.x_height))
    if "seen_drop" in f: st.seen_drop = float(np.clip(f["seen_drop"], 0, 0.3 * st.x_height))
    if "group_gap" in f: st.group_gap = float(np.clip(f["group_gap"], 0.5 * st.white, 2.5 * st.white))
    # the designer may set joined letters further apart than the teeth of one letter: used only when it
    # clearly differs (a sixth of the white), so a measure's noise does not move every join
    if "join_white" in f and abs(f["join_white"] - st.white) > st.white / 6:
        st.join_white = float(np.clip(f["join_white"], 0.6 * st.white, 2.5 * st.white))
    if "angular" in f: st.angular = float(np.clip(f["angular"], 0, 1))
    if "corner_r" in f: st.corner_r = float(np.clip(f["corner_r"], 0, 1.5 * st.stem))
    if "tail_drop" in f: st.tail_drop = float(np.clip(f["tail_drop"], 0, 1))
    if "top_join" in f: st.top_join = float(f["top_join"])
    if "reh_corner" in f: st.reh_corner = float(np.clip(f["reh_corner"], 0, 1.2 * st.stem))
    # the logo's own dash mark: a short hairline that is not a long bar over a letter
    own = [p.size * k for p in L.dashes if p.size <= 2.6 * L.stem and not (p.cy > 0.9 * L.x_height and p.size > 2.2 * L.stem)]
    if own:
        st.dash = float(np.median(own))
    return st


def letter_recipes(lay) -> List[Dict[str, Any]]:
    out = []
    for p in lay.placed:
        c = Counter(p.parts)
        out.append({"letter": p.char, "form": {"i": "منفصل", "I": "أول", "M": "وسط", "F": "آخر"}[p.form],
                    "from_logo": [f"{n} ×{m}" for (src, n), m in c.items() if src == "logo"],
                    "inferred": [f"{n} ×{m}" for (src, n), m in c.items() if src != "logo"]})
    return out


def report(name: str, lines_js, obs, lib, style: Style, recipes, checks, word: str, src_line: int, corrected: Optional[Style] = None, feats=None, health=None) -> str:
    md = [f"# تقرير بناء: {word}", "", f"الشعار: {name} — القواعد والقطع مأخوذة من السطر {src_line + 1}.", ""]
    if health:
        md += ["## فحص الصورة", "", f"الثقة في القراءة: {health['confidence'] * 100:.0f}%", ""]
        md += [f"- صُحّح: {n['text']}" for n in health["corrections"]] + [f"- تنبيه: {n['text']}" for n in health["warnings"]]
        if not health["corrections"] and not health["warnings"]:
            md += ["- الصورة نظيفة: لا تصحيح ولا تنبيه."]
        md += [""]
    md += ["## المرحلة 1: القياس", "", "| القياس | " + " | ".join(f"السطر {l['line']}" for l in lines_js) + " |",
           "|---|" + "---|" * len(lines_js)]
    rows = [("الميل °", "slant_deg"), ("سمك الجذع", "stem"), ("الخط الشعري", "hairline"), ("التباين", "contrast"),
            ("ارتفاع الحروف", "x_height"), ("الصعود", "ascender"), ("النزول", "descender"),
            ("الأبيض بين الجذوع", "white_between_stems"), ("خطوة الجذوع", "stem_pitch")]
    md += [f"| {n} | " + " | ".join(str(l.get(key)) for l in lines_js) + " |" for n, key in rows]
    md += ["", "## المرحلة 2: الفهم", ""]
    for l in lines_js:
        md.append(f"- السطر {l['line']}: {('عربي' if l['script'] == 'arabic' else 'لاتيني' if l['script'] == 'latin' else 'غير معروف')}"
                  + (f" — قرأ المحرك: «{l['text_guess']}»" if l['text_guess'] else "") + (f" — النص المعطى: «{l['text_given']}»" if l['text_given'] else ""))
    md += ["", "### الملاحظات", *[f"- {o['text']}" for o in obs]]
    if feats:
        md += ["", "### معاملات الشكل المقيسة (ما يحكم به المصمم بالعين)", "", "| المعامل | القيمة | عدد العينات | المعنى |", "|---|---|---|---|"]
        md += [f"| {k} | {v['value']}{v['unit']} | {v['samples']} | {v['note']} |" for k, v in feats.items()]
        if "angular" in feats:
            a = feats["angular"]["value"]
            md += ["", "### الهيكل (skeleton.svg)", "",
                   f"قُرئ مسار القلم في كل حرف وقُسّم إلى خطوط مستقيمة ومنحنيات وزوايا. الزوايا {a * 100:.0f}% من كل الانعطاف، "
                   + ("فالشعار يرسم بالزوايا: تُبنى الذيول والكؤوس من خطوط مستقيمة تلتقي بزاوية"
                      + (f" نصف قطرها الخارجي {feats['corner_r']['value']}px" if "corner_r" in feats else "") + "."
                      if a >= 0.5 else "فالشعار يرسم بالمنحنيات: تُبنى الذيول والكؤوس منحنية.")]
            if "tail_drop" in feats and feats["tail_drop"]["value"] < 0.3:
                md += ["", "ذيل حروف الراء في الشعار يستقر على السطر ولا ينزل تحته، فتُبنى الذيول على السطر، "
                       "ويرتفع رأس الواو فوق ذيلها حتى لا يُقرأ هاءً."]
    md += ["", "## المرحلة 3: صندوق القطع", "", "| القطعة | من أين | قياسات |", "|---|---|---|"]
    md += [f"| {p['name']} | {p['source']} | " + ", ".join(f"{k}={v}" for k, v in p.items() if k not in ("key", "name", "source")) + " |" for p in lib.summary()]
    md += ["", "## المرحلة 4: البناء", "", "### القواعد المستعملة", "```json", json.dumps(style.to_dict(), ensure_ascii=False, indent=2), "```",
           "", "### كيف بُني كل حرف", "| الحرف | الشكل | من الشعار | مستنتج |", "|---|---|---|---|"]
    md += [f"| {r['letter']} | {r['form']} | {'، '.join(r['from_logo']) or '—'} | {'، '.join(r['inferred']) or '—'} |" for r in recipes]
    md += ["", "## المرحلة 5: التحقق", "", "| القاعدة | الشعار | الكلمة | الفرق % | الحكم |", "|---|---|---|---|---|"]
    md += [f"| {c['name']} | {c['rule']} | {c['word']} | {c['diff']} | {c['verdict']} |" for c in checks]
    if corrected is not None and abs(corrected.white - style.white) > 0.05:
        md += ["", f"تصحيح تلقائي: المسافة الداخلية بين الجذوع صارت {corrected.white:.2f} حتى يصبح الأبيض المرئي مساويًا لأبيض الشعار ({style.white:.2f})."]
    if corrected is not None and abs(corrected.join_in - style.join_in) > 0.005:
        md += ["", f"تصحيح تلقائي: بداية منحنى الوصل في الرسم صارت {corrected.join_in:.3f} حتى يصبح العمق المقيس مساويًا لعمق الشعار ({style.join_in:.3f})."]
    return "\n".join(md) + "\n"


def _learned() -> None:
    """Load what has been learned from the designer's edits (once per process)."""
    from . import recipes
    if not getattr(_learned, "done", False):
        recipes.load_learned()
        _learned.done = True


# A reading by drawing is used only if its letters cover the logo this well. Measured: the right
# reading of the hand-made example covers it 83%; a wrong reading of the second example ("سسشی نصع"
# for "سفينة نجم") 68%, of the square kufi logo 51%.
DRAWN_MIN = 0.75
TAKE_MIN = 0.5         # letters are cut from a logo only if its measuring is trusted at least this much


def own_text(R) -> str:
    """The logo's own word, as far as it is known: typed by the designer, or read with no letter
    left unknown. A dotless tooth read by drawing is drawn as ب (the same body)."""
    if R.text_given:
        return R.text_given
    t = R.text_guess or ""
    if R.text_drawn:
        t = t.replace("ٮ", "ب")
    return t if t and not any(c in "ٮ?" for c in t) else ""


def analyse(image: bytes, *, texts: Optional[List[str]] = None, line: Optional[int] = None,
            overrides: Optional[Dict[str, Any]] = None, read: bool = True) -> Dict[str, Any]:
    """Stages 1-4 alone: guard the picture, measure and read every line, and turn the chosen line
    into drawing rules. Nothing is drawn. This is all a caller needs to typeset words in the logo's
    style, and all the benchmark needs to test the measuring."""
    _learned()
    # 1. load
    work, ink, scale, inverted, notes = prepare(image)
    k = 1.0 / scale
    g = np.asarray(work, float) / 255.0
    ink01 = g if inverted else 1.0 - g
    # 2-3. measure and read every line
    lines, readings = [], []
    for i, (a, b) in enumerate(find_lines(ink)):
        L = measure_line(ink01, ink.astype(bool), a, b, len(lines))
        if L is None:
            continue
        t = (texts or [None] * 99)[len(lines)] if texts and len(lines) < len(texts) else None
        readings.append(read_line(L, (t or "").strip() or None))
        lines.append(L)
    if not lines:
        raise ValueError("لم أجد في الصورة كتابة يمكن قياسها. استعمل صورة أوضح بخلفية سادة.")
    src = pick_line(lines, readings, line)
    health = assess(lines, readings, src, notes)
    L, R = lines[src], readings[src]
    # 4. parts
    lib = build_library(L, R, k)
    feats = measure_features(lines, readings, src, k)
    style = style_from(L, lib, k, feats)
    for key, v in (overrides or {}).items():
        if hasattr(style, key) and v is not None:
            setattr(style, key, type(getattr(style, key))(v))
    # The stem reader knows teeth, tails and tall strokes. When it leaves letters unread and the
    # designer typed nothing, the logo is read by drawing: letters drawn in the measured style are
    # matched to it. That reading is kept only if its letters cover the logo well.
    if read and R.script == "arabic" and not R.text_given and any(c in "ٮ?" for c in (R.text_guess or "?")):
        from .reading import read_by_drawing
        got = read_by_drawing(L, style, k)
        if got and got["match"] >= DRAWN_MIN:
            R.text_guess, R.text_drawn, R.drawn_match = got["text"], True, got["match"]
        elif got:
            R.drawn_match = got["match"]
        health = assess(lines, readings, src, notes)
    return {"work": work, "ink": ink, "scale": scale, "k": k, "inverted": inverted, "lines": lines, "readings": readings,
            "src": src, "health": health, "lib": lib, "feats": feats, "style": style}


def run(image: bytes, word: str, *, texts: Optional[List[str]] = None, line: Optional[int] = None,
        overrides: Optional[Dict[str, Any]] = None, use_parts: bool = False, out_dir: Optional[str] = None,
        name: str = "logo") -> Dict[str, Any]:
    an = analyse(image, texts=texts, line=line, overrides=overrides)
    work, ink, scale, k, inverted = an["work"], an["ink"], an["scale"], an["k"], an["inverted"]
    lines, readings, src, health = an["lines"], an["readings"], an["src"], an["health"]
    lib, feats, style = an["lib"], an["feats"], an["style"]
    L, R = lines[src], readings[src]
    obs = observations(lines, readings, k)
    # the self-check: the logo's own word, redrawn with the rules read from it; letters the logo draws
    # its own way are cut out of it and used for the new word
    own = own_text(R)
    # letters are taken from the logo only on a sure reading: typed, or read whole from the stems
    check = check_and_take(L, own, style, k, take=not R.text_drawn and health["confidence"] >= TAKE_MIN) if own else None
    if use_parts and own and check:
        # a piece of the logo that draws the logo's own word worse is not used
        dropped = prune_parts(lib, L, own, style, k)
        if dropped:
            from .library import PART_NAMES
            check["notes"].append("لم تُستعمل من قطع الشعار: " + "، ".join(PART_NAMES.get(q, q) for q in dropped)
                                  + " (رسمت كلمة الشعار نفسها أسوأ من صيغة المحرك).")

    # 5-6. build and verify
    def build(st: Style):
        tok = ACTIVE.set(lib if use_parts else None)
        try:
            return typeset(word, st)
        finally:
            ACTIVE.reset(tok)
    lib.used.clear()
    lay, got, checks, corrections = verify_and_correct(build, style)
    if not lay.shapes:
        raise ValueError("الكلمة لا تحتوي على حروف عربية يعرف المحرك رسمها.")
    recipes = letter_recipes(lay)
    # 7. drawings
    x0, y0, x1, y1 = lay.bbox()
    height = int(min(900, max(240, (y1 - y0 + 3 * style.stem) * 2)))
    lines_js = [line_json(a, b, k) for a, b in zip(lines, readings)]
    parts_json = {key: {"name": p.name, "source": p.source, "info": {a: _r(b, 1, 3) for a, b in p.info.items()},
                        "bezier": [[[list(map(lambda v: round(v, 3), q)) for q in seg] for seg in c] for c in p.curves],
                        "svg_path": "".join(to_path(c) for c in p.curves if c)} for key, p in lib.parts.items()}
    result = {
        "stages": STAGES,
        "image": {"width": round(work.width * k), "height": round(work.height * k), "inverted": inverted},
        "lines": lines_js, "observations": obs, "source_line": src,
        "input": health,
        "parts": lib.summary(), "parts_json": parts_json, "features": feats,
        "style": style.to_dict(), "recipes": recipes, "skipped": lay.skipped,
        "handles": handles(lay.style), "choices": choices(lay.style),
        "verification": checks, "corrections": corrections,
        "word": word, "word_svg": to_svg(lay), "word_png": to_png(lay, height=height),
        "construction_svg": sheet_svg(lay, title=word, white=style.white),
        "spacing_used": round(lay.style.white, 2), "join_used": round(lay.style.join_in, 3),
        "logo_sheet_svg": logo_sheet(thickness_png(ink, lines), (round(work.width * k), round(work.height * k)), scale, lines, readings),
        "parts_svg": parts_sheet(lib) if use_parts else ideal_parts_sheet(style),
        "cut_parts_svg": parts_sheet(lib), "pen_svg": pen_sheet(L, k), "mode": "cut" if use_parts else "ideal",
    }
    sk = read_skeleton(L)
    result["skeleton"] = sk.summary(k)
    result["skeleton_svg"] = skeleton_sheet(L, sk, k)
    # the critic: the defects a designer would see, on the new word and on the whole alphabet
    word_defects = critique(lay, word)
    alpha_defects = specimen_defects(lay.style)
    result["critic"] = {"word": [d.text() for d in word_defects], "alphabet": [d.text() for d in alpha_defects]}
    result["specimen_png"] = specimen_png(lay.style, alpha_defects)
    result["report_md"] = report(name, lines_js, obs, lib, style, recipes, checks, word, src, lay.style, feats, health)
    result["self_check_png"] = check.pop("overlay_png") if check else None
    if check:
        check.pop("overlay_after_png", None)
    result["self_check"] = check
    if check:
        result["report_md"] += "\n".join(["", "## الفحص الذاتي: كلمة الشعار نفسها بقواعد المحرك (selfcheck.png)", "",
            f"أعاد المحرك كتابة «{check['text']}» بما قاسه من الشعار ووضعها فوقه. التطابق الكلي {check['overall'] * 100:.0f}% ({check['verdict']}).",
            "الأزرق ما في الشعار وحده، والأحمر ما رسمه المحرك وحده، والداكن ما تطابق.", "",
            "| الحرف | الموضع | التطابق | الحكم | الإزاحة |", "|---|---|---|---|---|",
            *[f"| {x['letter']} | {FORM_AR[x['form']]} | {x['match'] * 100:.0f}% | {x['verdict']} | {x['shift']} |" for x in check["letters"]],
            *([f"", f"بعد أخذ ما يرسمه الشعار بطريقته من الشعار نفسه: التطابق الكلي {check['overall_after'] * 100:.0f}% "
               f"(" + "، ".join(f"{x['letter']} {x['match'] * 100:.0f}%" for x in check["letters_after"]) + ")."] if check.get("overall_after") else []),
            "", *[f"- {n}" for n in check["notes"]], ""])
    else:
        result["report_md"] += "\n".join(["", "## الفحص الذاتي", "",
            "لم يُجرَ: المحرك لم يقرأ كلمة الشعار كاملة. اكتب نص الشعار (--text) ليعيد كتابته ويطابقه عليه.", ""])
    hs, cs = result["handles"], result["choices"]
    result["report_md"] += "\n".join(["", "## المقابض: ما يستطيع المصمم تحريكه", "",
        "كل معامل له قاعدة تشتقه من قياسات الشعار. `style.params[الاسم] = قيمة` يبدّل القاعدة، و`style.variants[\"ه.I\"] = \"اسم الصيغة\"` يبدّل الصيغة.", "",
        "| المعامل | الشكل | القيمة لهذا الشعار | القاعدة | المصدر |", "|---|---|---|---|---|",
        *[f"| `{x['name']}` | {x['shape']} | {x['value']} | {x['rule']} | {x['source']} |" for x in hs], "",
        "| الحرف | الموضع | الصيغة المختارة | الصيغ المتاحة |", "|---|---|---|---|",
        *[f"| {x['letter']} | {x['form']} | {x['chosen']} | {'، '.join(o['name'] for o in x['options'])} |" for x in cs], ""])
    crit = result["critic"]
    result["report_md"] += "\n".join(["", "## الناقد (specimen.png)", "",
        "يرسم المحرك كل حرف بأشكاله الأربعة بأسلوب الشعار، ويفحصه بالعيوب التي يراها المصمم بعينه: "
        "شق أبيض بين جزأين، وتكدّس حبر أثقل من القلم، وخط أرفع من القلم الشعري، وجزء خارج خطوط الشبكة، "
        "وفراغ داخلي ضيق، ونقطة تلمس حرفها، وحرفان يُقرأ أحدهما كالآخر.", "",
        f"- الكلمة «{word}»: " + ("لا عيوب." if not crit["word"] else f"{len(crit['word'])} عيب"),
        *[f"  - {t}" for t in crit["word"]],
        f"- الحروف كلها: " + ("لا عيوب." if not crit["alphabet"] else f"{len(crit['alphabet'])} عيب"),
        *[f"  - {t}" for t in crit["alphabet"]], ""])
    if out_dir:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        for key, fname in [("word_svg", "word.svg"), ("construction_svg", "construction.svg"), ("logo_sheet_svg", "logo_sheet.svg"),
                           ("parts_svg", "parts.svg"), ("pen_svg", "pen_rule.svg"), ("skeleton_svg", "skeleton.svg"),
                           ("report_md", "report.md")]:
            (d / fname).write_text(result[key], encoding="utf-8")
        (d / "word.png").write_bytes(result["word_png"])
        if result.get("self_check_png"):
            (d / "selfcheck.png").write_bytes(result["self_check_png"])
        (d / "specimen.png").write_bytes(result["specimen_png"])
        (d / "parts.json").write_text(json.dumps(parts_json, ensure_ascii=False, indent=1), encoding="utf-8")
        (d / "measurements.json").write_text(json.dumps({"lines": lines_js, "observations": obs, "style": result["style"],
                                                         "verification": checks}, ensure_ascii=False, indent=1), encoding="utf-8")
    return result
