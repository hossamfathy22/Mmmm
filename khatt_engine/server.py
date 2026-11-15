"""Web interface:  uvicorn server:app --port 8010   then open http://localhost:8010

/           the editor: read a logo, write a word, move the handles, save the edits
/report     the full report of one run (measuring sheets, parts, construction)
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from khatt import editor
from khatt.pipeline import run

app = FastAPI(title="Khatt — logo construction engine")
STATIC = Path(__file__).parent / "static"
MAX_BYTES = 15 * 1024 * 1024


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "editor.html")


@app.get("/report")
def report_page() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


async def _image(image: UploadFile) -> bytes:
    data = await image.read()
    if not data:
        raise HTTPException(400, "الملف فارغ.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "الصورة أكبر من 15 ميغابايت.")
    return data


# ------------------------------------------------------------------ the editor
@app.post("/api/read")
async def api_read(image: UploadFile = File(...), text: str = Form("")) -> Dict[str, Any]:
    """Read a logo once: the measured style, how far to trust it, and the self-check."""
    data = await _image(image)
    try:
        return editor.read_logo(data, text.strip())
    except ValueError as e:
        raise HTTPException(422, str(e))
    except OSError:
        raise HTTPException(400, "تعذّر فتح الصورة. استعمل PNG أو JPG أو WebP.")


@app.post("/api/draw")
def api_draw(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Draw a word in a measured style with the designer's changes. Nothing is measured again."""
    word = str(body.get("word", "")).strip()
    if not word or not isinstance(body.get("style"), dict):
        raise HTTPException(400, "الكلمة أو قياسات الشعار ناقصة.")
    try:
        return editor.draw(body["style"], word, body.get("params") or {}, body.get("variants") or {}, bool(body.get("check", True)))
    except (ValueError, TypeError) as e:
        raise HTTPException(422, str(e))


@app.post("/api/save")
def api_save(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """Add the designer's changes for this logo to the log of edits."""
    if not isinstance(body.get("style"), dict):
        raise HTTPException(400, "قياسات الشعار ناقصة.")
    return editor.save_edit(body["style"], str(body.get("word", "")), body.get("params") or {}, body.get("variants") or {},
                            body.get("scopes") or {}, str(body.get("logo", "")))


@app.get("/api/edits")
def api_edits() -> Dict[str, Any]:
    """What has been corrected so far, handle by handle."""
    edits = editor.load_edits()
    return {"count": len(edits), "logos": len({e.get("logo", "") for e in edits}), "handles": editor.summarise(edits)}


# ------------------------------------------------------------------ the full report
@app.post("/api/run")
async def api_run(image: UploadFile = File(...), word: str = Form(...), texts: str = Form(""), line: int = Form(-1),
                  overrides: str = Form(""), use_parts: bool = Form(False)) -> Dict[str, Any]:
    data = await _image(image)
    try:
        ov: Optional[Dict[str, Any]] = json.loads(overrides) if overrides else None
    except json.JSONDecodeError:
        raise HTTPException(400, "قيم التعديل ليست JSON صالحًا.")
    lines = [t.strip() for t in texts.splitlines()] if texts.strip() else None
    try:
        r = run(data, word.strip(), texts=lines, line=None if line < 0 else line, overrides=ov,
                use_parts=use_parts, name=image.filename or "logo")
    except ValueError as e:
        raise HTTPException(422, str(e))
    except OSError:
        raise HTTPException(400, "تعذّر فتح الصورة. استعمل PNG أو JPG أو WebP.")
    r["word_png"] = base64.b64encode(r["word_png"]).decode("ascii")
    r["specimen_png"] = base64.b64encode(r["specimen_png"]).decode("ascii")
    if r.get("self_check_png"):
        r["self_check_png"] = base64.b64encode(r["self_check_png"]).decode("ascii")
    return editor._plain(r)
