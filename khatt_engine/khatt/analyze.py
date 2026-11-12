"""Loading an image: grey levels, ink/background, scale, and the text lines."""
from __future__ import annotations

import io
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

MAX_SIDE, MIN_SIDE = 1600, 900


# ------------------------------------------------------------------ helpers
def _otsu(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    total = gray.size
    sum_all = float(np.dot(np.arange(256), hist))
    w_b = s_b = 0.0
    best, th = -1.0, 128
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        s_b += t * hist[t]
        m_b, m_f = s_b / w_b, (sum_all - s_b) / w_f
        v = w_b * w_f * (m_b - m_f) ** 2
        if v > best:
            best, th = v, t
    return th


def _runs(row: np.ndarray) -> List[Tuple[int, int]]:
    """(start, length) of every run of 1s."""
    if not row.any():
        return []
    d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), (ends - starts).tolist()))


def _mode(lengths: List[int], lo: float, hi: float) -> Tuple[float, int]:
    """Peak of a smoothed histogram, refined by the median of the lengths around it."""
    vals = [v for v in lengths if lo <= v <= hi]
    if len(vals) < 3:
        return float("nan"), len(vals)
    hist = np.bincount(vals, minlength=int(hi) + 3).astype(float)
    sm = np.convolve(hist, [1, 2, 1], mode="same")
    peak = int(np.argmax(sm[1:]) + 1)
    near = [v for v in vals if abs(v - peak) <= max(1.5, peak * 0.2)]
    return float(np.median(near)), len(vals)


def _median(v: List[float]) -> float:
    return float(np.median(v)) if v else float("nan")


# ------------------------------------------------------------------ main entry
def load_binary(data: bytes) -> Tuple[Image.Image, np.ndarray, float, bool]:
    img = Image.open(io.BytesIO(data))
    img.load()
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    img = img.convert("L")
    w, h = img.size
    scale = min(1.0, MAX_SIDE / max(w, h))
    if max(w, h) * scale < MIN_SIDE:
        scale = min(3.0, MIN_SIDE / max(w, h))
    work = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    gray = np.asarray(work, dtype=np.uint8)
    th = _otsu(gray)
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    inverted = (border <= th).mean() > 0.5
    ink = (gray > th) if inverted else (gray <= th)
    return work, ink.astype(np.uint8), scale, bool(inverted)


def find_lines(ink: np.ndarray) -> List[Tuple[int, int]]:
    H, W = ink.shape
    rows = ink.sum(axis=1)
    min_ink = max(1, W * 0.0015)
    groups, y = [], 0
    while y < H:
        if rows[y] >= min_ink:
            s = y
            while y < H and rows[y] >= min_ink:
                y += 1
            groups.append([s, y - 1])
        else:
            y += 1
    gap = max(3, H * 0.012)
    merged: List[List[int]] = []
    for gr in groups:
        if merged and gr[0] - merged[-1][1] <= gap:
            merged[-1][1] = gr[1]
        else:
            merged.append(list(gr))
    if not merged:
        return []
    tallest = max(b - a for a, b in merged)
    big = [m for m in merged if m[1] - m[0] >= tallest * 0.35]
    for m in merged:  # dots and underlines join the nearest real line
        if m in big:
            continue
        best = min(big, key=lambda b: b[0] - m[1] if m[1] < b[0] else m[0] - b[1])
        best[0], best[1] = min(best[0], m[0]), max(best[1], m[1])
    return sorted((a, b) for a, b in big)
