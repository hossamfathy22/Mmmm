"""python -m khatt.bench.logos DIR [-o OUT]

The examination on real logos: a folder of pictures (one line of Arabic each, cropped) and a
texts.json beside them mapping each file's name (without extension) to the word written in it.
Real logos belong to their owners and are not part of this project: keep the folder outside it.

For every logo it reports how far the measuring is trusted, the self-check before and after letters
are taken from the logo, which letters were taken, which the logo draws longer than the engine (and
the engine now lengthens), and how the word is read when it is not typed.
Run it before and after a change: a change that helps the typefaces and hurts real logos is not
an improvement.
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
    a = p.parse_args()
    warnings.filterwarnings("ignore")
    folder = Path(a.folder)
    if not (folder / "texts.json").exists():
        print("يلزم ملف texts.json في المجلد: {\"اسم الصورة\": \"النص العربي\"}", file=sys.stderr)
        return 1
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
