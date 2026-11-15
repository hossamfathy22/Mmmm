"""python -m khatt.bench.logos DIR [-o OUT]

The examination on real logos: a folder of pictures (one line of Arabic each, cropped) and a
texts.json beside them mapping each file's name (without extension) to the word written in it.
Real logos belong to their owners and are not part of this project: keep the folder outside it.

For every logo it reports how far the measuring is trusted, the self-check before and after letters
are taken from the logo, which letters were taken, which the logo draws longer than the engine (and
the engine now lengthens), and how the word is read when it is not typed.
Run it before and after a change: a change that helps the typefaces and hurts real logos is not
an improvement.

With --held-out it asks whether the engine understands a logo or only copies it: each letter of the
word is hidden in turn, the rules are measured on the other letters, and the hidden one is drawn
with them (see held_out).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List

from ..editor import read_logo


def examine(folder: Path, read: bool = True) -> List[Dict]:
    texts = json.loads((folder / "texts.json").read_text(encoding="utf-8"))
    rows = []
    for name, text in texts.items():
        pics = [p for p in folder.glob(name + ".*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
        if not pics:
            continue
        data = pics[0].read_bytes()
        row: Dict = {"name": name, "text": text}
        t = time.time()
        try:
            r = read_logo(data, text)
        except Exception as e:                          # a picture the engine cannot measure says so
            row.update({"error": str(e)[:120]})
            rows.append(row)
            continue
        c = r["self_check"] or {}
        row.update({"confidence": r["health"]["confidence"], "check": c.get("overall"),
                    "after": c.get("overall_after"), "taken": sorted((c.get("taken") or {}).keys()),
                    "stretch": {q: round(v / r["style"]["x_height"], 2) for q, v in (c.get("stretch") or {}).items()},
                    "angular": r["style"]["angular"], "seconds": round(time.time() - t, 1)})
        if read:
            try:
                plain = read_logo(data, "")
                row.update({"read": plain["read"], "read_how": plain["read_how"], "read_right": plain["read"] == text})
            except Exception as e:
                row.update({"read": "", "read_how": "error", "read_right": False})
        rows.append(row)
    return rows


def held_out(folder: Path) -> List[Dict]:
    """Does the engine understand a logo or copy it? Each letter of the logo's word is hidden in turn:
    its ink is left out, the rules of the letters are measured on the others (fit_letters), and the
    hidden letter is drawn with them and matched on the logo. Compared with the same letter drawn with
    the engine's general rules. Nothing is cut: a cut is the logo's own drawing, not understanding."""
    from ..pipeline import TAKE_MIN, analyse
    from ..recipes import SKELETON
    from ..selfcheck import _Trial, fit_letters, self_check
    from ..style import Style
    texts = json.loads((folder / "texts.json").read_text(encoding="utf-8"))
    rows = []
    for name, text in texts.items():
        pics = [p for p in folder.glob(name + ".*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
        if not pics:
            continue
        try:
            an = analyse(pics[0].read_bytes(), texts=[text], read=False)
        except Exception:
            continue
        if an["health"]["confidence"] < TAKE_MIN:
            continue
        L, k, st = an["lines"][an["src"]], an["k"], an["style"]
        base = self_check(L, text, st, k)
        if not base:
            continue
        general, measured, with_cuts, with_parts, covered = [], [], [], [], []
        for i, x in enumerate(base["letters"]):
            t = _Trial(L, text, st, k, hide=i)
            if not t.ok:
                continue
            fitted, chosen, traits = fit_letters(L, text, st, k, trial=t)
            # the hidden letter is not a letter of the logo: it takes the manner, not its own measure
            on = "".join(sorted({SKELETON.get(y["letter"], y["letter"]) for j, y in enumerate(base["letters"]) if j != i}))
            r = self_check(L, text, Style(**{**st.__dict__, "fitted": fitted, "chosen": chosen, "traits": traits, "measured_on": on}), k)
            if not r or len(r["letters"]) != len(base["letters"]):
                continue
            general.append(x["match"])
            measured.append(r["letters"][i]["match"])
            # and with the bodies of the other letters cut from the logo: a sibling's body (ي for ن)
            # may draw the hidden letter, its own ink may not
            own = {f"{x['letter']}.{f}" for f in (x["form"], {"i": "F", "F": "i", "I": "M", "M": "I"}[x["form"]])}
            cuts = {q: v for q, v in (r.get("taken") or {}).items() if q not in own}
            got = r["letters"][i]["match"]
            if cuts:
                r2 = self_check(L, text, Style(**{**st.__dict__, "fitted": fitted, "chosen": chosen, "traits": traits,
                                                   "measured_on": on, "taken": cuts,
                                                   "seen": " ".join(f"{y['letter']}.{y['form']}" for j, y in enumerate(base["letters"]) if j != i)}), k)
                if r2 and len(r2["letters"]) == len(base["letters"]):
                    got = r2["letters"][i]["match"]
            with_cuts.append(got)
            # and with the parts of the other letters cut from the logo (a head, a cup, a tail)
            from ..letters import LETTERS as _L
            from ..selfcheck import PART_KINDS, take_parts
            from ..typeset import flat as _flat
            st_fit = Style(**{**st.__dict__, "fitted": fitted, "chosen": chosen, "traits": traits, "measured_on": on})
            parts = take_parts(L, text, st_fit, k, skip=i)
            mine = {getattr(sh, "name", "") for sh in _flat(_L[x["letter"]][1](st_fit, x["form"]).shapes)} & set(PART_KINDS)
            seen_others = " ".join(f"{y['letter']}.{y['form']}" for j, y in enumerate(base["letters"]) if j != i)
            got_p = r["letters"][i]["match"]
            if mine and mine & set(parts):
                r3 = self_check(L, text, Style(**{**st_fit.__dict__, "parts": parts, "seen": seen_others}), k)
                if r3 and len(r3["letters"]) == len(base["letters"]):
                    got_p = r3["letters"][i]["match"]
                covered.append((x["match"], r["letters"][i]["match"], got_p))
            with_parts.append(got_p)
        if general:
            rows.append({"name": name, "text": text, "general": sum(general) / len(general),
                         "measured": sum(measured) / len(measured), "letters": len(general),
                         "with_cuts": sum(with_cuts) / len(with_cuts),
                         "with_parts": sum(with_parts) / len(with_parts), "covered": covered,
                         "better_cuts": sum(m > g + 0.02 for g, m in zip(general, with_cuts)),
                         "worse_cuts": sum(m < g - 0.02 for g, m in zip(general, with_cuts)),
                         "better": sum(m > g + 0.02 for g, m in zip(general, measured)),
                         "worse": sum(m < g - 0.02 for g, m in zip(general, measured))})
    return rows


def reference_exam(folder: Path, widths=(0.85, 1.0, 1.2)) -> List[Dict]:
    """Would the letters a logo lacks be better built on the structure of a reference typeface?
    Every letter of the word is hidden in turn; the typeface (and width) that redraws the other
    letters best is chosen, and the hidden letter, built on it with the logo's pen, is matched on the
    logo, against the engine's own letter."""
    import numpy as np
    from ..pipeline import TAKE_MIN, analyse
    from ..reference import faces
    from ..selfcheck import self_check
    from ..style import Style
    paths = [f for f in faces() if f.endswith("-400.woff2")]
    texts = json.loads((folder / "texts.json").read_text(encoding="utf-8"))
    rows = []
    for name, text in texts.items():
        pics = [p for p in folder.glob(name + ".*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
        if not pics or not paths:
            continue
        try:
            an = analyse(pics[0].read_bytes(), texts=[text], read=False)
        except Exception:
            continue
        if an["health"]["confidence"] < TAKE_MIN:
            continue
        L, k, st = an["lines"][an["src"]], an["k"], an["style"]
        base = self_check(L, text, st, k)
        if not base:
            continue
        n = len(base["letters"])
        eng = np.array([x["match"] for x in base["letters"]])
        found = []
        for path in paths:
            for w in widths:
                try:
                    r = self_check(L, text, Style(**{**st.__dict__, "reference": path, "ref_wide": w, "ref_all": True}), k)
                except Exception:
                    continue
                if r and len(r["letters"]) == n:
                    found.append((path, w, np.array([x["match"] for x in r["letters"]])))
        if not found:
            continue
        A = np.array([m for _, _, m in found])
        ref = []
        for i in range(n):
            others = [j for j in range(n) if j != i]
            ref.append(float(A[A[:, others].mean(axis=1).argmax(), i]))
        best = found[int(A.mean(axis=1).argmax())]
        rows.append({"name": name, "text": text, "engine": float(eng.mean()), "reference": float(np.mean(ref)), "letters": n,
                     "font": Path(best[0]).stem, "better": int(sum(r > e + 0.02 for r, e in zip(ref, eng))),
                     "worse": int(sum(r < e - 0.02 for r, e in zip(ref, eng)))})
    return rows


def reference_summary(rows: List[Dict]) -> str:
    lines = ["| الشعار | الحرف المخفي بصيغة المحرك | ببنية الخط المرجعي | تحسّن / ساء | أقرب خط |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['text']} | {r['engine'] * 100:.0f}% | {r['reference'] * 100:.0f}% | {r['better']} / {r['worse']} | {r['font']} |")
    n = sum(r["letters"] for r in rows)
    e = sum(r["engine"] * r["letters"] for r in rows) / max(1, n)
    f = sum(r["reference"] * r["letters"] for r in rows) / max(1, n)
    lines += ["", f"على {n} حرفًا مخفيًا: {e * 100:.1f}% بصيغة المحرك، و{f * 100:.1f}% ببنية الخط المرجعي المختار من بقية الحروف "
              f"(تحسّن {sum(r['better'] for r in rows)} وساء {sum(r['worse'] for r in rows)})."]
    return "\n".join(lines)


PART_WORDS = ("مفقود", "قلم ونور", "فن ولين", "سفر وزن",
              "صوم فطر", "يعمل قعد", "نبع وكد", "طمع وفم")      # the last four have parts few logos give


def parts_exam(words=PART_WORDS) -> Dict:
    """Do the parts of a logo's letters build the letters it lacks? The open typefaces stand in for
    logos (many more letters than sixteen logos give): each word, set in each typeface, is measured
    as a logo; every letter with a named part is hidden in turn, the parts are cut from the other
    letters, and the hidden letter is drawn: in the engine's construction, with the logo's parts as
    they are, and with parts also made from others (mirrored or turned) where the engine's
    construction, or real typefaces, draw the two alike. The typeface playing the logo is left out of
    every reference: of the parts' likeness and of the real letters the drawings are read against."""
    import numpy as np
    from ..letters import LETTERS
    from ..pipeline import TAKE_MIN, analyse
    from ..selfcheck import self_check, take_parts
    from ..style import Style
    from ..typeset import flat
    from .fonts import font_files, load
    from .truth import png, word_image
    rows = []
    for f in [p for p in font_files() if p.name.endswith("-400.woff2")]:
        face = load(str(f))
        for w in words:
            try:
                an = analyse(png(word_image(face, w, x_px=90)[0]), texts=[w], read=False)
            except Exception:
                continue
            if an["health"]["confidence"] < TAKE_MIN:
                continue
            L, k, st = an["lines"][an["src"]], an["k"], an["style"]
            from ..reference import family
            st.ref_skip = family(f.stem)          # the typeface playing the logo teaches nothing about itself
            base = self_check(L, w, st, k)
            if not base:
                continue
            for i, x in enumerate(base["letters"]):
                names = {getattr(sh, "name", "") for sh in flat(LETTERS[x["letter"]][1](st, x["form"]).shapes)} - {""}
                if not names:
                    continue
                parts = take_parts(L, w, st, k, skip=i)
                if not parts:
                    continue
                seen = " ".join(f"{y['letter']}.{y['form']}" for j, y in enumerate(base["letters"]) if j != i)
                got = {}
                for mode, derive in (("as_is", False), ("engine_subs", "engine"), ("real_subs", "real")):
                    r = self_check(L, w, Style(**{**st.__dict__, "parts": parts, "seen": seen, "derive_parts": derive}), k)
                    if r and len(r["letters"]) == len(base["letters"]):
                        got[mode] = r["letters"][i]["match"]
                if len(got) == 3 and any(abs(v - x["match"]) > 1e-9 for v in got.values()):
                    # and does the letter still read as itself, drawn the engine's way and with the parts?
                    # and does the letter read as itself among real typefaces' letters (not the one playing the logo)?
                    from ..bench.__main__ import _same
                    from ..legible import reads_as_real
                    c, fm = x["letter"], x["form"]
                    up = Style(**{**st.__dict__, "slant": 0.0})
                    with_p = Style(**{**up.__dict__, "parts": parts, "seen": seen, "derive_parts": "real"})
                    from ..typeset import typeset as _ts
                    from .truth import FORMS as _F
                    placed = [p for p in _ts(_F[fm].format(c), with_p).placed if p.char == c]
                    ok = lambda g: (lambda r: r is not None and _same(r, c, fm))(reads_as_real(g, fm, up))
                    rows.append({"font": f.stem, "word": w, "letter": f"{c}.{fm}", "engine": x["match"], **got,
                                 "reads_engine": ok(LETTERS[c][1](up, fm)),
                                 "reads_parts": ok(placed[0].glyph) if placed else None})
    out = {"rows": rows, "cases": len(rows)}
    if rows:
        a = np.array([(r["engine"], r["as_is"], r["engine_subs"], r["real_subs"]) for r in rows])
        out["reads_engine"] = sum(1 for r in rows if r["reads_engine"]) / len(rows)
        out["reads_parts"] = sum(1 for r in rows if r["reads_parts"]) / len(rows)
        out.update({"engine": float(a[:, 0].mean()), "as_is": float(a[:, 1].mean()), "engine_subs": float(a[:, 2].mean()),
                    "real_subs": float(a[:, 3].mean()),
                    "better": int((a[:, 3] > a[:, 0] + 0.02).sum()), "worse": int((a[:, 3] < a[:, 0] - 0.02).sum())})
    return out


def held_out_summary(rows: List[Dict]) -> str:
    lines = ["| الشعار | الحرف المخفي بالقواعد العامة | بالقواعد المقيسة من بقية الحروف | تحسّن / ساء | ومع أجسام الحروف الشقيقة من الشعار | تحسّن / ساء |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['text']} | {r['general'] * 100:.0f}% | {r['measured'] * 100:.0f}% | {r['better']} / {r['worse']} | "
                     f"{r['with_cuts'] * 100:.0f}% | {r['better_cuts']} / {r['worse_cuts']} |")
    n = sum(r["letters"] for r in rows)
    g = sum(r["general"] * r["letters"] for r in rows) / max(1, n)
    m = sum(r["measured"] * r["letters"] for r in rows) / max(1, n)
    c = sum(r["with_cuts"] * r["letters"] for r in rows) / max(1, n)
    lines += ["", f"على {n} حرفًا مخفيًا في {len(rows)} شعارًا: {g * 100:.1f}% بالقواعد العامة، و{m * 100:.1f}% بقواعد مقيسة "
              f"من بقية حروف الشعار (تحسّن {sum(r['better'] for r in rows)} وساء {sum(r['worse'] for r in rows)})، "
              f"و{c * 100:.1f}% حين تُرسم من أجسام الحروف الشقيقة المقصوصة من الشعار أيضًا "
              f"(تحسّن {sum(r['better_cuts'] for r in rows)} وساء {sum(r['worse_cuts'] for r in rows)})."]
    cov = [t for r in rows for t in r.get("covered", [])]
    if cov:
        import numpy as _np
        a = _np.array(cov)
        lines += ["", f"الحروف المخفية التي فيها قطعة قُصّت من حرف آخر في الشعار ({len(cov)} حرفًا): "
                  f"{a[:, 0].mean() * 100:.1f}% بالقواعد العامة، و{a[:, 1].mean() * 100:.1f}% بالقواعد المقيسة، "
                  f"و{a[:, 2].mean() * 100:.1f}% بقطع الشعار (تحسّن {int((a[:, 2] > a[:, 1] + 0.02).sum())} وساء {int((a[:, 2] < a[:, 1] - 0.02).sum())} "
                  f"عن القواعد المقيسة)."]
    return "\n".join(lines)


def summary(rows: List[Dict]) -> str:
    ok = [r for r in rows if "error" not in r]
    best = [r.get("after") or r.get("check") or 0.0 for r in ok]
    lines = ["| الشعار | الثقة | الفحص الذاتي | بعد الأخذ منه | أُخذ | مُدّ | القراءة دون نص |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['text']} | — | لم يُقَس: {r['error']} | | | | |")
            continue
        pct = lambda v: f"{v * 100:.0f}%" if v is not None else "—"
        lines.append(f"| {r['text']} | {pct(r['confidence'])} | {pct(r['check'])} | {pct(r['after'])} | "
                     f"{'، '.join(k.split('.')[0] for k in r['taken']) or '—'} | "
                     f"{'، '.join(q.split('.')[0] for q, v in r.get('stretch', {}).items() if v > 0) or '—'} | "
                     f"{r.get('read', '') or '—'}{' ✓' if r.get('read_right') else ''} |")
    mean = sum(best) / max(1, len(best))
    lines += ["", f"متوسط الفحص الذاتي بعد الأخذ: {mean * 100:.1f}% على {len(ok)} شعارًا قيس من {len(rows)}. "
              f"قُرئ دون نص صحيحًا: {sum(1 for r in ok if r.get('read_right'))}."]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(prog="khatt.bench.logos", description="امتحان المحرك على شعارات حقيقية")
    p.add_argument("folder")
    p.add_argument("-o", "--out", default="")
    p.add_argument("--no-read", action="store_true", help="لا تجرّب القراءة دون نص (أسرع)")
    p.add_argument("--held-out", action="store_true", help="امتحان الحرف المخفي: هل يفهم المحرك الشعار أم ينسخه")
    p.add_argument("--reference", action="store_true", help="امتحان الحرف المخفي ببنية أقرب خط مرجعي")
    p.add_argument("--parts", action="store_true", help="امتحان قطع الحروف على الخطوط المفتوحة بديلًا عن الشعارات")
    a = p.parse_args()
    warnings.filterwarnings("ignore")
    folder = Path(a.folder)
    if not a.parts and not (folder / "texts.json").exists():
        print("يلزم ملف texts.json في المجلد: {\"اسم الصورة\": \"النص العربي\"}", file=sys.stderr)
        return 1
    if a.parts:
        r = parts_exam()
        if a.out:
            out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
            (out / "parts.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"على {r['cases']} حرفًا مخفيًا تغيّر رسمه: {r['engine'] * 100:.1f}% بصيغة المحرك، و{r['as_is'] * 100:.1f}% بقطع "
              f"الشعار كما هي، و{r['engine_subs'] * 100:.1f}% مع قطع مبدّلة بحكم المحرك، "
              f"و{r['real_subs'] * 100:.1f}% مع قطع مبدّلة بحكم الخطوط الحقيقية "
              f"(تحسّن {r['better']} وساء {r['worse']} عن صيغة المحرك). يقرؤه قارئ الخطوط الحقيقية حرفه: "
              f"{r['reads_engine'] * 100:.0f}% بصيغة المحرك، و{r['reads_parts'] * 100:.0f}% بقطع الشعار.")
        from collections import defaultdict
        by = defaultdict(list)
        for x in r["rows"]:
            by[x["letter"].split(".")[0]].append(x)
        for c, xs in sorted(by.items(), key=lambda kv: -len(kv[1])):
            m = lambda q: sum(x[q] for x in xs) / len(xs) * 100
            print(f"  {c}: {len(xs)} — المحرك {m('engine'):.0f}%، كما هي {m('as_is'):.0f}%، "
                  f"مبدّلة بحكم المحرك {m('engine_subs'):.0f}%، بحكم الخطوط {m('real_subs'):.0f}%")
        return 0
    if a.reference:
        text = reference_summary(reference_exam(folder))
        print(text)
        return 0
    if a.held_out:
        text = held_out_summary(held_out(folder))
        print(text)
        if a.out:
            out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
            (out / "held_out.md").write_text("# امتحان الحرف المخفي\n\n" + text + "\n", encoding="utf-8")
        return 0
    rows = examine(folder, read=not a.no_read)
    text = summary(rows)
    print(text)
    if a.out:
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        (out / "logos.md").write_text("# امتحان الشعارات الحقيقية\n\n" + text + "\n", encoding="utf-8")
        (out / "logos.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
