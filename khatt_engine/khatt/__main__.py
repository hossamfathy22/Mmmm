"""Command line:  python -m khatt logo.png "بيدوناق" -o out/  [--line 2] [--explain]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import run


def main() -> int:
    p = argparse.ArgumentParser(prog="khatt", description="قِس شعارًا وولّد كلمة جديدة بنفس أسلوبه")
    p.add_argument("image", help="صورة الشعار (PNG أو JPG)")
    p.add_argument("word", help="الكلمة العربية المطلوب توليدها")
    p.add_argument("-o", "--out", default="khatt_out", help="مجلد النتائج")
    p.add_argument("--line", type=int, default=None, help="رقم السطر الذي تؤخذ منه القواعد (يبدأ من 1)")
    p.add_argument("--explain", action="store_true", help="اطلب شرح التصميم من Claude (يحتاج ANTHROPIC_API_KEY)")
    a = p.parse_args()
    data = Path(a.image).read_bytes()
    try:
        r = run(data, a.word, line=(a.line - 1) if a.line else None, explain=a.explain, out_dir=a.out, name=Path(a.image).name)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    st = r["style"]
    print(f"السطر المستعمل: {r['line'] + 1} من {len(r['analysis']['lines'])}")
    print(f"الجذع {st['stem']} · الشعري {st['hair']} · ارتفاع الحروف {st['x_height']} · الميل {st['slant']}°")
    for rule in r["analysis"]["rules"]:
        print(" -", rule)
    if r["result"]["skipped"]:
        print("حروف لم تُرسم:", " ".join(r["result"]["skipped"]))
    if a.explain and r["explain_error"]:
        print("الشرح:", r["explain_error"], file=sys.stderr)
    print(f"النتائج في: {Path(a.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
