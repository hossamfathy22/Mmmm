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
from .pipeline import TAKE_MIN, analyse, own_text
from .recipes import SHAPE_LETTERS, base, choices, handle_rule, handles, is_handle
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
    st.params = {k: float(v) for k, v in {**(d.get("params") or {}), **(params or {})}.items() if is_handle(k) and v is not None}
    st.variants = {str(k): str(v) for k, v in {**(d.get("variants") or {}), **(variants or {})}.items() if v}
    return st


def read_logo(image: bytes, text: str = "") -> Dict[str, Any]:
    """Guard, measure and read the logo once. `text` is the word written in the logo, if the designer
    gives it (the reading and the self-check are surer with it)."""
    an = analyse(image, texts=[text] if text.strip() else None)
    L, R, st, k = an["lines"][an["src"]], an["readings"][an["src"]], an["style"], an["k"]
    own = own_text(R)
    take = not R.text_drawn and an["health"]["confidence"] >= TAKE_MIN
    check = check_and_take(L, own, st, k, take=take) if own else None
    if check and take:
        # the logo's parts box: the new letters are built of the logo's own pieces
        from .selfcheck import adopt_library
        dropped = adopt_library(an["lib"], L, own, st, k)
        check["library"] = sorted((st.library.get("parts") or {}).keys())
        if dropped:
            from .library import PART_NAMES
            check["notes"].append("لم تُستعمل من قطع الشعار: " + "، ".join(PART_NAMES.get(q, q) for q in dropped)
                                  + " (رسمت كلمة الشعار نفسها أسوأ من صيغة المحرك).")
    ref = None
    if own and an["health"]["confidence"] >= TAKE_MIN:
        # the reference typeface nearest the logo: the letters the logo lacks may be built on it. Judged
        # on the logo's word drawn with the engine's letters (with the letters cut from the logo every
        # typeface redraws the word alike, and the first of them was taken)
        from .reference import nearest
        from .style import Style as _Style
        bare = _Style(**{**st.__dict__, "taken": {}, "parts": {}, "variants": {}})
        ref = nearest(L, own, bare, k, widths=(1.0,))
        if ref:
            st.reference, st.ref_wide = str(ref["path"]), ref["wide"]
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
        "reference": ({"font": Path(ref["path"]).stem, "match": round(ref["match"], 3), "engine": round(ref["engine"], 3)}
                      if ref else None),
        "measures": {"الجذع": st.stem, "الخط الشعري": st.hair, "ارتفاع الحروف": st.x_height, "الطالع": st.ascender,
                     "النازل": st.descender, "البياض بين الأسنان": st.white, "الميل °": st.slant, "قطر النقطة": st.dot},
    })


def with_designer_parts(st: Style, parts: Optional[Dict[str, Any]]) -> Style:
    """The kit with the designer's parts in it: a part given replaces the logo's (or fills a gap), a
    part given as None goes back to the engine's construction."""
    if not parts:
        return st
    merged = dict(st.parts)
    for name, part in parts.items():
        if part is None:
            merged.pop(name, None)
        elif isinstance(part, dict) and part.get("polys"):
            merged[name] = part
    return Style(**{**st.__dict__, "parts": merged})


def draw(style: Dict[str, Any], word: str, params: Optional[Dict[str, float]] = None,
         variants: Optional[Dict[str, str]] = None, check: bool = True,
         parts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The word in the logo's style with the designer's changes. Fast: nothing is measured again."""
    st = with_designer_parts(style_of(style, params, variants), parts)
    lay = typeset(word, st)
    if not lay.shapes:
        raise ValueError("الكلمة لا تحتوي على حروف عربية يعرف المحرك رسمها.")
    forms = [(c, f) for c, f in shape_forms(word) if c in LETTERS]
    in_word = {c for c, _ in forms}
    hs = handles(st)
    for h in hs:
        if "part" in h:
            # a part of the logo matters where a letter of the word takes it
            h["used"] = any(("logo", h["part"]) in p.parts for p in lay.placed)
            continue
        h["used"] = (any(c == h["letters"] and f == h["form"] or (c == h["letters"] and {f, h["form"]} in ({"i", "F"}, {"I", "M"}))
                         for c, f in forms) if "letters" in h and h["letters"] else bool(in_word & set(SHAPE_LETTERS.get(h["shape"], ""))))
    present = {(base(c, f), f) for c, f in forms} | set(forms)
    cs = [dict(c, form_ar=FORM_AR[c["form"]]) for c in choices(st) if (c["letter"], c["form"]) in present]
    # the letters the logo gave nothing to (no letter of its, no piece): the engine's own drawing, which
    # the user is told of, to complete in the kit if they wish
    from .typeset import flat as _flat
    marks = ("نقطة", "شرطة", "همزة")
    alone = sorted({p.char for p in lay.placed
                    if not any(getattr(sh, "source", "") in ("logo", "designer") for sh in _flat(p.glyph.shapes)
                               if getattr(sh, "name", "") not in marks)})
    out = {"svg": to_svg(lay, background=None), "handles": hs, "choices": cs, "skipped": lay.skipped,
           "letters": [{"letter": c, "form": FORM_AR[f]} for c, f in forms], "engine_only": alone}
    if check:
        out["critic"] = [d.text() for d in critique(lay, word)]
    return _plain(out)


# ------------------------------------------------------------------ the log of edits
def log_path() -> Path:
    return Path(os.environ.get("KHATT_DATA", Path.home() / ".khatt")) / "edits.jsonl"


def save_edit(style: Dict[str, Any], word: str, params: Dict[str, float], variants: Dict[str, str],
              scopes: Optional[Dict[str, str]] = None, logo: str = "", path: Optional[Path] = None,
              parts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Record what the designer changed for this logo: for every handle moved, the logo's measures,
    the rule's value, the chosen value, and whether it is meant for this logo only or always; for
    every variant picked, what was picked and what it replaced. One JSON line per save."""
    base = style_of(style)
    scopes = scopes or {}
    moved = []
    for name, v in (params or {}).items():
        if not is_handle(name) or v is None:
            continue
        rule = handle_rule(base, name)
        # values reach the page rounded to three places: a handle left where the rule put it is not an edit
        if abs(float(v) - rule) <= max(2e-3, 1e-4 * abs(rule)):
            continue
        moved.append({"name": name, "rule": round(rule, 4), "chosen": round(float(v), 4),
                      "ratio": round(float(v) / rule, 4) if rule else None, "scope": scopes.get(name, "logo")})
    was = {f"{c['letter']}.{c['form']}": c["chosen"] for c in choices(base)}
    picked = [{"form": key, "chosen": name, "instead_of": was.get(key, ""), "scope": scopes.get(key, "logo")}
              for key, name in (variants or {}).items() if name and was.get(key, name) != name]
    # the parts the designer completed the kit with: what they drew (its line, pen and ends) or chose
    made = []
    for name, part in (parts or {}).items():
        if part is None:
            made.append({"part": name, "source": "engine"})
        elif isinstance(part, dict):
            made.append({"part": name, "source": part.get("source", "designer"), "font": part.get("font", ""),
                         "from": part.get("from", ""), "line": part.get("line"), "closed": part.get("closed"),
                         "thickness": part.get("thickness"), "ends": part.get("ends")})
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "logo": logo, "word": word,
           "measures": {k: v for k, v in base.to_dict().items() if isinstance(v, (int, float)) and not isinstance(v, bool)},
           "params": moved, "variants": picked, "parts": made}
    p = path or log_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_plain(rec), ensure_ascii=False) + "\n")
    return {"saved": len(moved) + len(picked) + len(made), "params": len(moved), "variants": len(picked), "parts": len(made),
            "file": str(p)}


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



# ------------------------------------------------------------------ the parts kit
def _svg_paths(shapes_by_role, box=None, pad=6.0):
    """Outlines (y up) as an SVG: [(role, rings)] -> svg text; even-odd within a shape."""
    xs = [x for _, rings in shapes_by_role for r in rings for x, _ in r]
    ys = [y for _, rings in shapes_by_role for r in rings for _, y in r]
    if box:
        x0, y0, x1, y1 = box
    elif xs:
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    else:
        return ""
    w, h = x1 - x0 + 2 * pad, y1 - y0 + 2 * pad
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0 - pad:.1f} {-y1 - pad:.1f} {w:.1f} {h:.1f}">']
    for role, rings in shapes_by_role:
        d = "".join("M" + "L".join(f"{x:.1f},{-y:.1f}" for x, y in r) + "Z" for r in rings if len(r) >= 3)
        if d:
            out.append(f'<path class="{role}" fill-rule="evenodd" d="{d}"/>')
    out.append("</svg>")
    return "".join(out)


def kit(style: Dict[str, Any], params: Optional[Dict[str, float]] = None, variants: Optional[Dict[str, str]] = None,
        parts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The parts kit of this logo for the editor: every part, where it comes from, the letters it builds,
    a picture of it in a letter that shows it, and what the drawing board needs (the letter in the
    engine's construction, the part's frame, the logo's lines and its centre line to drag)."""
    from .kit import SAMPLE, centre_line, engine_part, status
    from .typeset import flat
    st = with_designer_parts(style_of(style, params, variants), parts)
    up = Style(**{**st.__dict__, "slant": 0.0, "seen": ""})
    X, A, D = st.x_height, st.ascender, st.descender
    lines = {"baseline": 0.0, "x_height": X, "ascender": X + A, "descender": -D}
    out = []
    for item in status(st):
        name = item["name"]
        eng = engine_part(st, name)
        if eng is None:
            continue
        c, f = SAMPLE[name].split(".")
        lay = typeset(c if f == "i" else {"I": c + "ـ", "M": "ـ" + c + "ـ", "F": "ـ" + c}[f], up)
        placed = [p for p in lay.placed if p.char == c]
        letter = []
        if placed:
            p = placed[0]
            for sh in flat(p.glyph.shapes):
                role = "part" if getattr(sh, "name", "") == name else "rest"
                letter.append((role, [[(x, y) for x, y in r] for r in sh if len(r) >= 3]))
        box = (min(-3.2 * st.pitch, min((x for _, rs in letter for r in rs for x, _ in r), default=0)),
               -D, max(0.0, max((x for _, rs in letter for r in rs for x, _ in r), default=0)), X + A)
        here = st.parts.get(name)
        line = here.get("line") if here and here.get("line") else (centre_line(here["polys"]) if here else centre_line(eng["part"]))
        out.append({**item, "preview": _svg_paths(letter, box),
                    "board": {"letter": [[list(map(float, pt)) for pt in r] for role, rs in letter if role == "rest" for r in rs],
                              "part": [[list(map(float, pt)) for pt in r] for role, rs in letter if role == "part" for r in rs],
                              "engine_part": [[[x + eng["xr"], y] for x, y in r] for r in eng["part"]],
                              "xr": eng["xr"], "box": box, "lines": lines,
                              # the line to drag, in the letter's frame (the part's own frame plus its right end)
                              "line": [[x + eng["xr"], y] for x, y in line] if here and not here.get("line") else
                                      ([list(map(float, q)) for q in line] if here else [[x + eng["xr"], y] for x, y in line]),
                              "closed": bool(here.get("closed")) if here else False,
                              "thickness": float(here.get("thickness", 1.0)) if here else 1.0,
                              "ends": here.get("ends", "round") if here else "round"}})
    from .library import PART_NAMES
    basics = [{"key": q, "label": PART_NAMES.get(q, q), "from": (v.get("source") or "")}
              for q, v in ((st.library or {}).get("parts") or {}).items()]
    return _plain({"parts": out, "pen": {"stem": st.stem, "hair": st.hair, "power": st.contrast_power},
                   "basics": basics,
                   "complete": sum(1 for o in out if o["source"] != "engine"), "total": len(out)})


def kit_candidates(style: Dict[str, Any], name: str) -> Dict[str, Any]:
    """Suggestions for one part of the kit, from real typefaces, drawn with the logo's pen."""
    from .kit import candidates, engine_part
    st = style_of(style)
    eng = engine_part(st, name)
    out = []
    for cand in candidates(st, name):
        polys = [[(x + eng["xr"], y) for x, y in r] for r in cand["polys"]]
        # its centre line in the letter's frame, as the board draws (and the designer goes on from it)
        cand["line"] = [[x + eng["xr"], y] for x, y in (cand.get("line") or [])]
        out.append({"font": cand["font"], "part": cand, "preview": _svg_paths([("part", polys)])})
    return _plain({"name": name, "candidates": out})


def kit_stroke(style: Dict[str, Any], name: str, points, closed: bool = False, thickness: float = 1.0,
               ends: str = "round") -> Dict[str, Any]:
    """The designer's line, drawn with the logo's pen into a part of the kit."""
    from .kit import designer_part
    st = style_of(style)
    part = designer_part(st, name, [(float(x), float(y)) for x, y in points], closed, thickness, ends)
    if part is None:
        raise ValueError("ارسم نقطتين على الأقل.")
    return _plain({"name": name, "part": part,
                   "outline": [[[x + 0.0, y] for x, y in r] for r in part["polys"]]})
