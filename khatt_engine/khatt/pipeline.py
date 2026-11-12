"""One call that does everything: measure the image, extract rules, set a new word, draw, explain."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .analyze import analyze_image
from .render import sheet_svg, to_png, to_svg
from .style import Style
from .typeset import typeset


def pick_line(analysis: Dict[str, Any], line: Optional[int]) -> int:
    usable = [i for i, l in enumerate(analysis["lines"]) if l.get("stem")]
    if not usable:
        raise ValueError("لم أجد في الصورة كتابة يمكن قياسها. استعمل صورة أوضح بخلفية سادة.")
    if line is not None and line in usable:
        return line
    return usable[0]


def generate(word: str, style: Style) -> Dict[str, Any]:
    lay = typeset(word, style)
    if not lay.shapes:
        raise ValueError("الكلمة لا تحتوي على حروف عربية يعرف المحرك رسمها.")
    x0, y0, x1, y1 = lay.bbox()
    height = int(min(900, max(240, (y1 - y0 + 3 * style.stem) * 2)))
    return {
        "word": word,
        "forms": [f"{p.char}:{p.form}" for p in lay.placed],
        "skipped": lay.skipped,
        "svg": to_svg(lay),
        "png": to_png(lay, height=height),
        "sheet_svg": sheet_svg(lay, title=word),
    }


def report_markdown(name: str, analysis: Dict[str, Any], line: int, style: Style, out: Dict[str, Any], explanation: Optional[str]) -> str:
    L = analysis["lines"][line]
    md = [f"# تقرير: {name}", "", f"الكلمة المولّدة: **{out['word']}** — القواعد مأخوذة من السطر {line + 1}.", "",
          "## القواعد المستنتجة", *[f"- {r}" for r in analysis["rules"]], "",
          "## قياسات السطر", "| القياس | القيمة |", "|---|---|"]
    names = {"slant_deg": "الميل °", "stem": "سمك الجذع", "hairline": "الخط الشعري", "contrast": "التباين",
             "x_height": "ارتفاع الحروف", "ascender": "الصعود", "descender": "النزول", "stem_pitch": "خطوة الجذوع",
             "white_between_stems": "الأبيض بين الجذوع", "ink_coverage": "كثافة الحبر"}
    md += [f"| {v} | {L.get(k)} |" for k, v in names.items()]
    md += ["", "## قواعد التوليد", "```json", json.dumps(style.to_dict(), ensure_ascii=False, indent=2), "```"]
    if out["skipped"]:
        md += ["", f"حروف لم تُرسم: {' '.join(out['skipped'])}"]
    if explanation:
        md += ["", "---", "", "# شرح Claude", "", explanation]
    return "\n".join(md) + "\n"


def run(image: bytes, word: str, *, line: Optional[int] = None, explain: bool = False,
        overrides: Optional[Dict[str, Any]] = None, out_dir: Optional[str] = None, name: str = "logo") -> Dict[str, Any]:
    analysis = analyze_image(image)
    idx = pick_line(analysis, line)
    style = Style.from_line(analysis["lines"][idx])
    for k, v in (overrides or {}).items():
        if hasattr(style, k) and v is not None:
            setattr(style, k, type(getattr(style, k))(v))
    out = generate(word, style)
    explanation, explain_error = None, None
    if explain:
        from .explain import explain as ask_claude
        explanation, explain_error = ask_claude(image, analysis, style.to_dict(), word, idx)
    result = {"analysis": analysis, "line": idx, "style": style.to_dict(), "result": out,
              "explanation": explanation, "explain_error": explain_error,
              "report_md": report_markdown(name, analysis, idx, style, out, explanation)}
    if out_dir:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "analysis_overlay.png").write_bytes(analysis["overlay_png"])
        (d / "word.svg").write_text(out["svg"], encoding="utf-8")
        (d / "word.png").write_bytes(out["png"])
        (d / "construction.svg").write_text(out["sheet_svg"], encoding="utf-8")
        (d / "style.json").write_text(json.dumps(style.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        (d / "measurements.json").write_text(json.dumps({k: v for k, v in analysis.items() if k != "overlay_png"}, ensure_ascii=False, indent=2), encoding="utf-8")
        (d / "report.md").write_text(result["report_md"], encoding="utf-8")
    return result
