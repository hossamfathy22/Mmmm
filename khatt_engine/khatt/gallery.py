"""The gallery: every logo the engine has read, each with its own record.

A logo given to the gallery is read afresh (editor.read_logo: the guard, the measures, the reading, the
construction chosen by the logo, the self-check) and kept in a folder of its own:

  logo.<ext>        the picture as given
  record.json       its name and word, the construction and its evidence, every measure with where it
                    came from (the logo, the open typefaces, the engine's rule), the self-check, the
                    professional exam, the style that draws its letters
  specimen.png      every letter in every form, in the logo's style

Nothing in a record is used for another logo: the engine reads each logo from nothing (README, «المحرك
لا يحفظ شعارًا»). The gallery lives with the user's data (KHATT_DATA, ~/.khatt by default), never in
the code: a logo is its designer's work.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .editor import _plain, draw, read_logo


def root() -> Path:
    return Path(os.environ.get("KHATT_DATA", Path.home() / ".khatt")) / "gallery"


def _id(name: str, image: bytes) -> str:
    slug = re.sub(r"[^0-9A-Za-z؀-ۿ]+", "-", name.strip()).strip("-")[:40] or "logo"
    return f"{slug}-{hashlib.sha1(image).hexdigest()[:8]}"


def _ext(image: bytes) -> str:
    if image[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if image[:3] == b"\xff\xd8\xff":
        return "jpg"
    if image[:4] == b"RIFF" and image[8:12] == b"WEBP":
        return "webp"
    return "img"


def _sources(st) -> List[Dict[str, Any]]:
    """Every named measure of the style with its value and where its rule came from."""
    from .recipes import handles
    out = []
    for h in handles(st):
        if "name" in h and "value" in h and "part" not in h and "letters" not in h:
            out.append({"name": h["name"], "shape": h.get("shape", ""), "value": h["value"], "rule": h.get("rule", ""),
                        "source": h.get("source", "")})
    return out


def add(image: bytes, name: str, word: str = "", exam: bool = True) -> Dict[str, Any]:
    """Read a logo afresh and keep its record. The same picture under the same name is read again
    and its record replaced."""
    from .professional import exam as run_exam
    from .sheets import specimen_png
    from .style import Style
    rid = _id(name, image)
    folder = root() / rid
    got = read_logo(image, word.strip())
    st = Style(**{k: v for k, v in got["style"].items() if k in Style.__dataclass_fields__})
    record = {
        "id": rid, "name": name.strip() or rid, "word": word.strip(), "created": time.strftime("%Y-%m-%d %H:%M"),
        "construction": got.get("construction"), "read": got.get("read"), "read_sure": got.get("read_sure"),
        "health": got.get("health"), "measures": got.get("measures"),
        "sources": _sources(st) if st.construction != "block" else
                   [{"name": k, "value": v, "source": "مقيس أو مضبوط على كلمة الشعار" if k in (st.block or {}) else "من مقاسات الشعار الأساسية"}
                    for k, v in sorted((st.block or {}).items()) if isinstance(v, (int, float))],
        "self_check": {k: v for k, v in (got.get("self_check") or {}).items() if k in ("overall", "letters", "notes")},
        "reference": got.get("reference"), "style": got["style"],
    }
    if exam:
        record["exam"] = run_exam(st, fonts=False)
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("logo.*"):
        old.unlink()
    (folder / f"logo.{_ext(image)}").write_bytes(image)
    (folder / "specimen.png").write_bytes(specimen_png(st))
    (folder / "record.json").write_text(json.dumps(_plain(record), ensure_ascii=False, indent=1), encoding="utf-8")
    return _plain(record)


def items() -> List[Dict[str, Any]]:
    """The records, newest first, in brief."""
    out = []
    for f in sorted(root().glob("*/record.json")):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        ex = r.get("exam") or []
        out.append({"id": r["id"], "name": r.get("name"), "word": r.get("word") or r.get("read"), "created": r.get("created"),
                    "construction": (r.get("construction") or {}).get("construction", "pen"),
                    "exam": f"{sum(x['ok'] for x in ex)}/{len(ex)}" if ex else None,
                    "match": (r.get("self_check") or {}).get("overall")})
    return sorted(out, key=lambda r: r.get("created") or "", reverse=True)


def get(rid: str) -> Optional[Dict[str, Any]]:
    f = _folder(rid) / "record.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def _folder(rid: str) -> Path:
    if not re.fullmatch(r"[0-9A-Za-z؀-ۿ-]+", rid or ""):
        raise ValueError("معرّف غير صالح.")
    return root() / rid


def file(rid: str, kind: str) -> Optional[Path]:
    folder = _folder(rid)
    if kind == "logo":
        return next(iter(folder.glob("logo.*")), None)
    if kind == "specimen":
        p = folder / "specimen.png"
        return p if p.exists() else None
    return None


def write(rid: str, text: str) -> Dict[str, Any]:
    """Anything typed, in the logo's style (nothing measured again)."""
    r = get(rid)
    if r is None:
        raise KeyError(rid)
    return draw(r["style"], text, check=False)


def remove(rid: str) -> bool:
    folder = _folder(rid)
    if folder.exists():
        shutil.rmtree(folder)
        return True
    return False
