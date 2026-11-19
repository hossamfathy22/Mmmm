"""python -m khatt.bench.full DIR [-o OUT] [--held-out]

The full examination, run before and after every change (PLAN.md). DIR holds the pictures of logos
and a texts.json beside them: {"name": "the Arabic word"} or {"name": {"text": ..., "line": n}}, with
`line` the index (top first) of the line that word is written on, where the logo has more than one.
Real logos belong to their owners: keep the folder outside the repository.

Every logo is examined in a process of its own (the memory of one logo is given back before the next)
and its result kept in OUT/<name>.json. For each logo:
  - read: did the engine find writing; how many lines; which it took; is it the line of the word;
  - its measures (stem, hairline, x-height, ascender, descender, slant, white, pitch, dot);
  - its pen (qalam.fit_nib: the width, thickness and angle of a flat nib, and how well it fits);
  - the self-check: the logo's word redrawn on it, before and after letters are taken from it;
  - what was taken from it (letters, parts);
  - with --held-out, every letter hidden in turn and drawn from the others (the measure of understanding);
  - a picture: the logo, a new word and the 18 skeletons as the engine draws them in its style.
OUT/index.html puts them side by side, OUT/summary.md gives the table.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List

SKELETONS = "ا ب ح د ر س ص ط ع ف ق ك ل م ن ه و ى".split()
WORD = "مدرسة"
PICS = (".png", ".jpg", ".jpeg", ".webp")


def entries(folder: Path) -> List[Dict]:
    texts = json.loads((folder / "texts.json").read_text(encoding="utf-8"))
    out = []
    for name, v in texts.items():
        v = v if isinstance(v, dict) else {"text": v}
        pics = [p for p in folder.glob(name + ".*") if p.suffix.lower() in PICS]
        if pics:
            out.append({"name": name, "path": str(pics[0]), "text": v["text"], "line": v.get("line")})
    return out


def examine_one(path: str, text: str, line, held_out: bool) -> Dict:
    """Everything about one logo (run in a process of its own)."""
    import io
    import numpy as np
    from scipy import ndimage as ndi
    from ..pipeline import TAKE_MIN, analyse, own_text
    from ..qalam import fit_nib
    from ..selfcheck import check_and_take
    from ..render import to_png
    from ..typeset import typeset
    r: Dict = {"text": text}
    data = Path(path).read_bytes()
    t0 = time.time()
    try:
        an = analyse(data, texts=[text])
    except Exception as e:                                   # no writing found, or the picture is not read
        r.update(read=False, error=str(e)[:200]); return r
    L, R, k, st = an["lines"][an["src"]], an["readings"][an["src"]], an["k"], an["style"]
    r.update(read=True, lines=len(an["lines"]), src=an["src"], script=R.script, typed_here=bool(R.text_given),
             confidence=round(float(an["health"]["confidence"]), 2), stems=len(L.stems))
    if line is not None:
        r["line_ok"] = an["src"] == int(line)
    r["measures"] = {q: round(float(getattr(st, q)), 2) for q in
                     ("stem", "hair", "x_height", "ascender", "descender", "slant", "white", "pitch", "dot", "contrast_power")}
    # the pen, on the line's own ink (dots and dashes left out)
    B = np.asarray(L.B, bool)
    lab, n = ndi.label(B, structure=np.ones((3, 3)))
    if n:
        sizes = ndi.sum(B, lab, range(1, n + 1))
        body = np.isin(lab, 1 + np.flatnonzero(sizes > 0.05 * sizes.max()))
        try:
            nib = fit_nib(body, 1)
            r["nib"] = {"width": round(nib["width"] * k, 2), "thick": round(nib["thick"] * k, 2),
                        "angle": round(nib["angle"], 1), "error": round(nib["error"] * k, 2)}
        except Exception as e:
            r["nib"] = {"error_text": str(e)[:100]}
    own = own_text(R)
    if own:
        check = check_and_take(L, own, st, k, take=not R.text_drawn and an["health"]["confidence"] >= TAKE_MIN)
        if check:
            r["self_check"] = round(float(check["overall"]), 3)
            r["self_check_after"] = round(float(check.get("overall_after") or check["overall"]), 3)
            r["taken"] = sorted(st.taken)
            r["parts"] = sorted(st.parts)
            r["fitted"] = len(st.fitted)
            r["traits"] = {a: round(b, 2) for a, b in st.traits.items()}
    if held_out and own:
        from ..recipes import SKELETON
        from ..selfcheck import _Trial, fit_letters, self_check
        from ..style import Style
        base = self_check(L, own, st, k)
        if base:
            got = []
            for i, x in enumerate(base["letters"]):
                t = _Trial(L, own, st, k, hide=i)
                if not t.ok:
                    continue
                fitted, chosen, traits = fit_letters(L, own, st, k, trial=t)
                on = " ".join(sorted({f"{SKELETON.get(y['letter'], y['letter'])}.{y['form']}" for j, y in enumerate(base["letters"]) if j != i}))
                rr = self_check(L, own, Style(**{**st.__dict__, "fitted": fitted, "chosen": chosen, "traits": traits,
                                                 "measured_on": on, "taken": {}, "parts": {}}), k)
                if rr and len(rr["letters"]) == len(base["letters"]):
                    got.append(rr["letters"][i]["match"])
            if got:
                r["held_out"] = round(float(np.mean(got)), 3)
                r["held_out_n"] = len(got)
    # the picture: a new word, and the 18 skeletons, as the engine draws them for this logo
    pics = {}
    for w in [WORD] + SKELETONS:
        try:
            lay = typeset(w, st)
            if lay.shapes:
                pics[w] = base64.b64encode(to_png(lay, height=120 if w == WORD else 80)).decode()
        except Exception:
            pass
    r["pics"] = pics
    r["seconds"] = round(time.time() - t0, 1)
    return r


def html(folder: Path, out: Path, rows: List[Dict]) -> str:
    cards = []
    for e, r in rows:
        logo = base64.b64encode(Path(e["path"]).read_bytes()).decode()
        mime = "png" if e["path"].lower().endswith(".png") else "jpeg"
        if not r.get("read"):
            body = f'<p class="bad">لم يقرأ المحرك الشعار: {r.get("error", "")}</p>'
        else:
            m = r["measures"]; nb = r.get("nib", {})
            line = "" if "line_ok" not in r else (" ✓ السطر الصحيح" if r["line_ok"] else " ✗ سطر خاطئ")
            body = (f'<p>الثقة {r["confidence"]:.0%} · السطر {r["src"] + 1} من {r["lines"]}{line} · جذوع {r["stems"]}</p>'
                    f'<p>جذع {m["stem"]} · شعري {m["hair"]} · ارتفاع {m["x_height"]} · ميل {m["slant"]}° · خطوة {m["pitch"]}</p>'
                    f'<p>القلم: عرض {nb.get("width", "—")} · سمك {nb.get("thick", "—")} · زاوية {nb.get("angle", "—")}° · خطأ {nb.get("error", "—")}</p>'
                    f'<p>الفحص الذاتي {r.get("self_check", "—")} ← {r.get("self_check_after", "—")}'
                    + (f' · الحرف المخفي {r["held_out"]}' if "held_out" in r else "") + "</p>")
            pics = r.get("pics", {})
            if WORD in pics:
                body += f'<img class="w" src="data:image/png;base64,{pics[WORD]}">'
            body += '<div class="sk">' + "".join(
                f'<span><img src="data:image/png;base64,{pics[c]}"><b>{c}</b></span>' for c in SKELETONS if c in pics) + "</div>"
        cards.append(f'<div class="card"><img class="logo" src="data:image/{mime};base64,{logo}"><div><h2>{e["name"]} — {e["text"]}</h2>{body}</div></div>')
    return ('<!doctype html><html dir="rtl"><head><meta charset="utf-8"><style>'
            'body{font-family:sans-serif;background:#f4f1ea;margin:0;padding:16px}'
            '.card{display:flex;gap:16px;background:#fff;border-radius:12px;padding:12px;margin-bottom:12px}'
            '.logo{width:180px;height:180px;object-fit:contain;background:#eee;border-radius:8px}'
            'h2{margin:0 0 4px;font-size:18px} p{margin:2px 0;color:#444;font-size:14px} .bad{color:#b00}'
            '.w{height:70px;margin:6px 0} .sk{display:flex;flex-wrap:wrap;gap:6px} .sk span{text-align:center}'
            '.sk img{height:48px;display:block} .sk b{font-size:11px;color:#999}'
            '</style></head><body>' + "".join(cards) + "</body></html>")


def summary(rows: List[Dict]) -> str:
    lines = ["| الشعار | قُرئ | السطر | الثقة | جذع | شعري | ارتفاع | ميل | قلم: عرض/سمك/زاوية (خطأ) | الفحص الذاتي | بعد الأخذ | الحرف المخفي |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for e, r in rows:
        if not r.get("read"):
            lines.append(f"| {e['name']} | ✗ | — | — | — | — | — | — | — | — | — | — |"); continue
        m = r["measures"]; nb = r.get("nib", {})
        ln = f"{r['src'] + 1}/{r['lines']}" + ("" if "line_ok" not in r else (" ✓" if r["line_ok"] else " ✗"))
        pen = f"{nb.get('width', '—')}/{nb.get('thick', '—')}/{nb.get('angle', '—')}° ({nb.get('error', '—')})"
        lines.append(f"| {e['name']} | ✓ | {ln} | {r['confidence']:.0%} | {m['stem']} | {m['hair']} | {m['x_height']} | {m['slant']}° | {pen} | "
                     f"{r.get('self_check', '—')} | {r.get('self_check_after', '—')} | {r.get('held_out', '—')} |")
    read = [r for _, r in rows if r.get("read")]
    ok = [r for r in read if r.get("line_ok", True)]
    ho = [r["held_out"] for r in read if "held_out" in r]
    lines += ["", f"قُرئ {len(read)} من {len(rows)} شعارًا؛ على السطر الصحيح {len(ok)}."
              + (f" الحرف المخفي (وسط الشعارات): {sum(ho) / len(ho):.3f} على {len(ho)} شعارًا." if ho else "")]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(prog="khatt.bench.full")
    p.add_argument("folder", nargs="?")
    p.add_argument("-o", "--out", default="")
    p.add_argument("--held-out", action="store_true")
    p.add_argument("--one", nargs=4, metavar=("PATH", "TEXT", "LINE", "JSON"), help=argparse.SUPPRESS)
    a = p.parse_args()
    warnings.filterwarnings("ignore")
    if a.one:
        path, text, line, dst = a.one
        r = examine_one(path, text, None if line == "-" else int(line), a.held_out)
        Path(dst).write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
        return 0
    folder = Path(a.folder)
    out = Path(a.out or (folder / "_full"))
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for e in entries(folder):
        dst = out / (e["name"] + ".json")
        cmd = [sys.executable, "-m", "khatt.bench.full", "--one", e["path"], e["text"],
               "-" if e["line"] is None else str(e["line"]), str(dst)] + (["--held-out"] if a.held_out else [])
        res = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[2]), capture_output=True, text=True)
        if res.returncode != 0 or not dst.exists():
            r = {"read": False, "error": "تعطّل الامتحان: " + (res.stderr.strip().splitlines() or ["?"])[-1][:160]}
        else:
            r = json.loads(dst.read_text(encoding="utf-8"))
        rows.append((e, r))
        print(e["name"], "✓" if r.get("read") else "✗", r.get("self_check_after", ""), r.get("held_out", ""), flush=True)
    text = summary(rows)
    (out / "summary.md").write_text(text, encoding="utf-8")
    (out / "index.html").write_text(html(folder, out, rows), encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
