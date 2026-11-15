"""Learning from the designer: when the same handle is corrected the same way on several logos,
propose to change its rule, examine the change on the open typefaces, and keep it only if the
letters read no worse.

  propose(edits)       the handles corrected the same way on at least `min_logos` logos
  judge(proposal)      score the letters of that handle on the typefaces, with and without the change
  apply(accepted)      write the factors to learned.json, beside the log; recipes.P applies them

Nothing changes on its own: `python -m khatt.learning` lists the proposals and the verdicts,
`--apply` writes the accepted ones. A rule is never rewritten in the code: a learned factor
sits on top of it, with the evidence, and can be removed by deleting one line of learned.json.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import recipes
from .editor import load_edits, summarise
from .recipes import PARAMS, SHAPE_LETTERS

MIN_LOGOS = 3          # one logo is a choice for that logo; the same choice on three is a rule
MAX_LOSS = 0.01        # a change may cost the typefaces at most one point of reading


def learned_path() -> Path:
    return Path(os.environ.get("KHATT_DATA", Path.home() / ".khatt")) / "learned.json"


def propose(edits: List[Dict[str, Any]], min_logos: int = MIN_LOGOS) -> List[Dict[str, Any]]:
    """Handles moved the same way on at least `min_logos` different logos. Edits marked 'for this
    logo only' still count: the designer may not have thought of other logos, the repetition says
    more. The factor proposed is the median of chosen / rule."""
    out = []
    for s in summarise(edits):
        if s["name"] not in PARAMS or not s["median_ratio"] or s["logos"] < min_logos or not s["same_way"]:
            continue
        if abs(s["median_ratio"] - 1) < 0.02:
            continue
        current = recipes.LEARNED.get(s["name"], 1.0)
        out.append({**s, "factor": round(current * s["median_ratio"], 4), "shape": PARAMS[s["name"]].shape})
    return out


def judge(proposal: Dict[str, Any], bench=None) -> Dict[str, Any]:
    """Score the letters the handle shapes, on every typeface, with the rule as it is and with the
    proposed factor. The change passes if the letters read no worse than MAX_LOSS."""
    from .bench.cache import Bench
    B = bench or Bench()
    name = proposal["name"]
    letters = "".join(sorted(set(SHAPE_LETTERS.get(proposal["shape"], ""))))
    if not letters:
        return {**proposal, "passed": False, "why": "لا حروف معروفة لهذا المقبض"}

    def total(r):
        n = sum(v["n"] for v in r.values())
        return sum(v["read"] * v["n"] for v in r.values()) / max(n, 1)
    before = total(B.score(letters))
    saved = dict(recipes.LEARNED)
    try:
        recipes.LEARNED[name] = proposal["factor"]
        after = total(B.score(letters))
    finally:
        recipes.LEARNED.clear(); recipes.LEARNED.update(saved)
    passed = after >= before - MAX_LOSS
    why = (f"الحروف على الخطوط: {before * 100:.1f}% قبل، {after * 100:.1f}% بعد"
           + ("" if passed else f" (هبوط أكبر من {MAX_LOSS * 100:.0f} نقطة)"))
    return {**proposal, "before": round(before, 4), "after": round(after, 4), "passed": passed, "why": why}


def apply(accepted: List[Dict[str, Any]], path: Optional[Path] = None) -> Path:
    p = path or learned_path()
    data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    for a in accepted:
        data[a["name"]] = {"factor": a["factor"], "logos": a["logos"], "times": a["times"], "before": a.get("before"),
                           "after": a.get("after"), "learned": time.strftime("%Y-%m-%d")}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    recipes.load_learned(p)
    return p


def main() -> int:
    import argparse
    import warnings
    warnings.filterwarnings("ignore")
    a = argparse.ArgumentParser(prog="khatt.learning", description="اقترح تحديث القواعد من سجلّ التعديلات، وامتحنه على الخطوط")
    a.add_argument("--apply", action="store_true", help="اكتب ما نجح في الامتحان إلى learned.json")
    a.add_argument("--min-logos", type=int, default=MIN_LOGOS)
    a.add_argument("--accept", action="append", default=[], metavar="NAME",
                   help="اقبل هذا المقترح وإن رفضه الامتحان: ذوقك قد يخالف الخطوط عن قصد")
    args = a.parse_args()
    recipes.load_learned()
    edits = load_edits()
    props = propose(edits, args.min_logos)
    print(f"السجلّ: {len(edits)} حفظ على {len({e.get('logo', '') for e in edits})} شعار. مقترحات: {len(props)}")
    verdicts = [judge(p) for p in props]
    for v in verdicts:
        print(f"  {v['name']}: × {v['factor']} (عُدّل على {v['logos']} شعار) → {'يُقبل' if v['passed'] else 'يُرفض'} · {v['why']}")
    ok = [v for v in verdicts if v["passed"] or v["name"] in args.accept]
    for v in verdicts:
        if not v["passed"] and v["name"] in args.accept:
            print(f"  {v['name']}: قُبل بطلبك رغم الامتحان")
    if args.apply and ok:
        print("كُتب إلى", apply(ok))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
