"""Web interface:  uvicorn server:app --port 8010   then open http://localhost:8010"""
from __future__ import annotations

import base64
import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from khatt.pipeline import generate, run
from khatt.style import Style

app = FastAPI(title="Khatt — logo analysis and word generation")
STATIC = Path(__file__).parent / "static"
MAX_BYTES = 15 * 1024 * 1024


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _pack(out: Dict[str, Any]) -> Dict[str, Any]:
    return {**out, "png": _b64(out["png"])}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/run")
async def api_run(image: UploadFile = File(...), word: str = Form(...), line: int = Form(-1),
                  explain: bool = Form(False), overrides: str = Form("")) -> Dict[str, Any]:
    data = await image.read()
    if not data:
        raise HTTPException(400, "الملف فارغ.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "الصورة أكبر من 15 ميغابايت.")
    try:
        ov = json.loads(overrides) if overrides else None
    except json.JSONDecodeError:
        raise HTTPException(400, "قيم التعديل ليست JSON صالحًا.")
    try:
        r = run(data, word, line=None if line < 0 else line, explain=explain, overrides=ov, name=image.filename or "logo")
    except ValueError as e:
        raise HTTPException(422, str(e))
    except OSError:
        raise HTTPException(400, "تعذّر فتح الصورة. استعمل PNG أو JPG أو WebP.")
    analysis = dict(r["analysis"])
    overlay = analysis.pop("overlay_png")
    return {"analysis": analysis, "overlay_png": _b64(overlay), "line": r["line"], "style": r["style"],
            "result": _pack(r["result"]), "explanation": r["explanation"], "explain_error": r["explain_error"],
            "report_md": r["report_md"]}


@app.post("/api/typeset")
def api_typeset(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Set a word with hand-edited rules, without measuring an image again."""
    word = str(payload.get("word", "")).strip()
    raw = payload.get("style") or {}
    allowed = {f.name: f.type for f in fields(Style)}
    try:
        st = Style(**{k: (bool(v) if k == "dash_marks" else float(v)) for k, v in raw.items() if k in allowed})
    except (TypeError, ValueError):
        raise HTTPException(400, "قيم القواعد غير صالحة.")
    if st.stem <= 0 or st.hair <= 0 or st.x_height <= 0:
        raise HTTPException(400, "السمك والارتفاع يجب أن تكون أكبر من صفر.")
    try:
        return {"style": st.to_dict(), "result": _pack(generate(word, st))}
    except ValueError as e:
        raise HTTPException(422, str(e))
