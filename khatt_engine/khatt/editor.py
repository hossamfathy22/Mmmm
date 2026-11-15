"""The judge's seat: read a logo once, then draw a word again and again as the designer moves
handles and picks variants, and keep a record of what was changed.

Nothing here knows about the web: `server.py` only passes JSON in and out. Whoever sits in the
seat (a designer today; a model, if one is ever wanted) uses the same three calls:

  read_logo(image, text)                    -> the measured style, how far to trust it, the self-check
  draw(style, word, params, variants)       -> the word as SVG, the critic's findings, the handles
  save_edit(style, word, params, ...)       -> one line in the log of edits
"""
from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .critic import critique
from .letters import LETTERS
from .pipeline import analyse, own_text
from .recipes import PARAMS, SHAPE_LETTERS, SKELETON, choices, handles, rule_value
from .render import to_svg
from .selfcheck import check_and_take
from .style import Style
from .typeset import shape_forms, typeset

FORM_AR = {"i": "منفصل", "I": "أول", "M": "وسط", "F": "آخر"}


def _plain(v: Any) -> Any:
    """JSON-safe copy (numpy numbers become Python numbers)."""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, (np.floating, np.integer)):
        return float(v) if isinstance(v, np.floating) else int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def style_of(d: Dict[str, Any], params: Optional[Dict[str, float]] = None, variants: Optional[Dict[str, str]] = None) -> Style:
    st = Style(**{k: v for k, v in d.items() if k in Style.__dataclass_fields__ and k not in ("params", "variants")})
    st.params = {k: float(v) for k, v in {**(d.get("params") or {}), **(params or {})}.items() if k in PARAMS and v is not None}
    st.variants = {str(k): str(v) for k, v in {**(d.get("variants") or {}), **(variants or {})}.items() if v}
    return st


def read_logo(image: bytes, text: str = "") -> Dict[str, Any]:
    """Guard, measure and read the logo once. `text` is the word written in the logo, if the designer
    gives it (the reading and the self-check are surer with it)."""
    an = analyse(image, texts=[text] if text.strip() else None)
    L, R, st, k = an["lines"][an["src"]], an["readings"][an["src"]], an["style"], an["k"]
    own = own_text(R)
    check = check_and_take(L, own, st, k, take=not R.text_drawn) if own else None
    overlay = None
    if check:
        overlay = base64.b64encode(check.pop("overlay_png")).decode("ascii")
        after = check.pop("overlay_after_png", None)
        if after:
            check["overlay_after_png"] = base64.b64encode(after).decode("ascii")
    return _plain({
        "style": st.to_dict(), "health": an["health"], "read": R.text_given or R.text_guess, "read_sure": bool(own),
        "read_how": "given" if R.text_given else ("drawn" if R.text_drawn else "stems"), "read_match": R.drawn_match,
        "self_check": check, "self_check_png": overlay,
        "measures": {"الجذع": st.stem, "الخط الشعري": st.hair, "ارتفاع الحروف": st.x_height, "الطالع": st.ascender,
                     "النازل": st.descender, "البياض بين الأسنان": st.white, "الميل °": st.slant, "قطر النقطة": st.dot},
    })


def draw(style: Dict[str, Any], word: str, params: Optional[Dict[str, float]] = None,
         variants: Optional[Dict[str, str]] = None, check: bool = True) -> Dict[str, Any]:
    """The word in the logo's style with the designer's changes. Fast: nothing is measured again."""
    st = style_of(style, params, variants)
    lay = typeset(word, st)
    if not lay.shapes:
        raise ValueError("الكلمة لا تحتوي على حروف عربية يعرف المحرك رسمها.")
    forms = [(c, f) for c, f in shape_forms(word) if c in LETTERS]
    in_word = {c for c, _ in forms}
    hs = handles(st)
    for h in hs:
        h["used"] = bool(in_word & set(SHAPE_LETTERS.get(h["shape"], "")))
    present = {(SKELETON.get(c, c), f) for c, f in forms}
    cs = [dict(c, form_ar=FORM_AR[c["form"]]) for c in choices(st) if (c["letter"], c["form"]) in present]
    out = {"svg": to_svg(lay, background=None), "handles": hs, "choices": cs, "skipped": lay.skipped,
           "letters": [{"letter": c, "form": FORM_AR[f]} for c, f in forms]}
    if check:
        out["critic"] = [d.text() for d in critique(lay, word)]
    return _plain(out)


# ------------------------------------------------------------------ the log of edits
def log_path() -> Path:
    return Path(os.environ.get("KHATT_DATA", Path.home() / ".khatt")) / "edits.jsonl"


def save_edit(style: Dict[str, Any], word: str, params: Dict[str, float], variants: Dict[str, str],
              scopes: Optional[Dict[str, str]] = None, logo: str = "", path: Optional[Path] = None) -> Dict[str, Any]:
    """Record what the designer changed for this logo: for every handle moved, the logo's measures,
    the rule's value, the chosen value, and whether it is meant for this logo only or always; for
    every variant picked, what was picked and what it replaced. One JSON line per save."""
    base = style_of(style)
    scopes = scopes or {}
    moved = []
    for name, v in (params or {}).items():
        if name not in PARAMS or v is None:
            continue
        rule = rule_value(base, name)
        # values reach the page rounded to three places: a handle left where the rule put it is not an edit
        if abs(float(v) - rule) <= max(2e-3, 1e-4 * abs(rule)):
            continue
        moved.append({"name": name, "rule": round(rule, 4), "chosen": round(float(v), 4),
                      "ratio": round(float(v) / rule, 4) if rule else None, "scope": scopes.get(name, "logo")})
    was = {f"{c['letter']}.{c['form']}": c["chosen"] for c in choices(base)}
    picked = [{"form": key, "chosen": name, "instead_of": was.get(key, ""), "scope": scopes.get(key, "logo")}
              for key, name in (variants or {}).items() if name and was.get(key, name) != name]
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "logo": logo, "word": word,
           "measures": {k: v for k, v in base.to_dict().items() if isinstance(v, (int, float)) and not isinstance(v, bool)},
           "params": moved, "variants": picked}
    p = path or log_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_plain(rec), ensure_ascii=False) + "\n")
    return {"saved": len(moved) + len(picked), "params": len(moved), "variants": len(picked), "file": str(p)}


def load_edits(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    p = path or log_path()
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def summarise(edits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """What the designer keeps correcting: per handle, on how many logos it was moved, by how much
    (median chosen / rule), and how often it was marked 'always'. This is what a rule would be
    re-fitted from, once the same correction recurs across logos."""
    by: Dict[str, List[Dict]] = {}
    for e in edits:
        for m in e.get("params", []):
            by.setdefault(m["name"], []).append({**m, "logo": e.get("logo", "")})
    out = []
    for name, ms in by.items():
        ratios = [m["ratio"] for m in ms if m.get("ratio")]
        out.append({"name": name, "times": len(ms), "logos": len({m["logo"] for m in ms}),
                    "median_ratio": round(float(np.median(ratios)), 3) if ratios else None,
                    "same_way": bool(ratios) and (all(r > 1 for r in ratios) or all(r < 1 for r in ratios)),
                    "always": sum(m.get("scope") == "always" for m in ms)})
    return sorted(out, key=lambda x: (-x["logos"], -x["times"]))
