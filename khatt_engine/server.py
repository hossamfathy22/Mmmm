"""Web interface:  uvicorn server:app --port 8010   then open http://localhost:8010"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from khatt.pipeline import run

app = FastAPI(title="Khatt — logo construction engine")
STATIC = Path(__file__).parent / "static"
MAX_BYTES = 15 * 1024 * 1024


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/run")
async def api_run(image: UploadFile = File(...), word: str = Form(...), texts: str = Form(""), line: int = Form(-1),
                  overrides: str = Form(""), use_parts: bool = Form(True)) -> Dict[str, Any]:
    data = await image.read()
    if not data:
        raise HTTPException(400, "الملف فارغ.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "الصورة أكبر من 15 ميغابايت.")
    try:
        ov = json.loads(overrides) if overrides else None
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
    return r
