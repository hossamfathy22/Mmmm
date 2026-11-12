"""Command line:  python -m khatt logo.png "بيدوناق" -o out/ [--text "سیرنگ|Sirang"] [--line 1]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import run


def main() -> int:
    p = argparse.ArgumentParser(prog="khatt", description="قِس شعارًا وافهمه وقطّعه، ثم ابنِ كلمة جديدة من قطعه")
    p.add_argument("image", help="صورة الشعار (PNG أو JPG)")
    p.add_argument("word", help="الكلمة العربية المطلوب بناؤها")
    p.add_argument("-o", "--out", default="khatt_out", help="مجلد النتائج")
    p.add_argument("--text", default="", help="النص المكتوب في الشعار، سطر لكل سطر مفصولة بـ | (اختياري، يحسّن الفهم)")
    p.add_argument("--line", type=int, default=None, help="رقم السطر الذي تؤخذ منه القطع (يبدأ من 1)")
    p.add_argument("--cut-parts", action="store_true", help="تجريبي: قصّ قطع الشعار من بكسلاته بدل رسمها رسمًا مثاليًا بأرقامه")
    a = p.parse_args()
    try:
        r = run(Path(a.image).read_bytes(), a.word, texts=[t for t in a.text.split("|")] if a.text else None,
                line=(a.line - 1) if a.line else None, use_parts=a.cut_parts, out_dir=a.out, name=Path(a.image).name)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    for l in r["lines"]:
        print(f"السطر {l['line']} ({l['script']}): جذع {l['stem']} · شعري {l['hairline']} · ميل {l['slant_deg']}° · ارتفاع {l['x_height']}"
              + (f" · قرأ: {l['text_guess']}" if l["text_guess"] else ""))
    if r["fit"]:
        f = r["fit"]
        print(f"إعادة رسم «{f['text']}»: تطابق {f['iou_before'] * 100:.1f}% ← {f['iou_after'] * 100:.1f}% بعد الضبط")
        print("  شكل كل حرف:", "  ".join(f"{x['letter']} {x['iou'] * 100:.0f}%" + ("*" if x.get("outlier") else "") for x in f["letters"]))
        if f["outliers"]:
            print("  * مستبعد من الضبط لأن عائلات القطع ترسمه بشكل مختلف:", " ".join(f["outliers"]))
        for c in f["changes"]:
            print(f"  ضبط {c['name']}: {c['measured']} ← {c['fitted']}")
    print("القطع:", "، ".join(p["name"] for p in r["parts"]))
    for c in r["verification"]:
        print(f"  تحقق {c['name']}: الشعار {c['rule']} · الكلمة {c['word']} → {c['verdict']}")
    print(f"النتائج في: {Path(a.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
