"""Measure a logotype image: grid, slant, pen widths, rhythm, dots, dashes and counters.

All numbers returned are in pixels of the original image.
"""
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


def measure_line(ink: np.ndarray, top: int, bot: int) -> Optional[Dict[str, Any]]:
    H, W = ink.shape
    region = ink[top:bot + 1]
    h = bot - top + 1
    total = int(region.sum())
    if total < 50:
        return None
    cols = np.flatnonzero(region.any(axis=0))
    x0, x1 = int(cols[0]), int(cols[-1])
    rows = region.sum(axis=1)
    # x-height: the longest run of dense rows (row counts do not change with slant)
    # dense rows: compare with the 90th percentile, not the maximum, so that a few very dense
    # baseline rows (long joins) do not hide the teeth
    thr = float(np.percentile(rows[rows > 0], 90)) * 0.45
    dense = rows >= thr
    best = (0, h - 1, -1)
    i = 0
    while i < h:
        if dense[i]:
            j, miss, end = i, 0, i
            while j < h and miss <= 2:
                if dense[j]:
                    end, miss = j, 0
                else:
                    miss += 1
                j += 1
            if end - i > best[2]:
                best = (i, end, end - i)
            i = j
        else:
            i += 1
    x_top, base = top + best[0], top + best[1]
    xh = base - x_top + 1
    # slant: the shear that makes vertical strokes sharpest inside the band
    ys, xs = np.nonzero(ink[x_top:base + 1, x0:x1 + 1])
    ys, xs = ys + x_top, xs + x0
    if len(xs) > 300_000:
        k = len(xs) // 300_000 + 1
        ys, xs = ys[::k], xs[::k]
    pad = int(math.tan(math.radians(36)) * h) + 4

    def score(a: float) -> float:
        t = math.tan(math.radians(a))
        idx = np.round(xs - (base - ys) * t).astype(int) + pad
        hist = np.bincount(idx, minlength=W + 2 * pad)
        return float(np.dot(hist, hist))

    angles = np.arange(-35, 36, 1.0)
    a_best = float(angles[int(np.argmax([score(a) for a in angles]))])
    fine = np.arange(a_best - 1, a_best + 1.01, 0.1)
    a_best = float(fine[int(np.argmax([score(a) for a in fine]))])
    slant = 0.0 if abs(a_best) < 0.35 else round(a_best, 1)
    t = math.tan(math.radians(slant))
    # deskewed raster of the line
    off = int(abs(t) * h) + 2
    DW = W + 2 * off
    D = np.zeros((h, DW), dtype=np.uint8)
    ly, lx = np.nonzero(region[:, x0:x1 + 1])
    lx = lx + x0
    nx = np.round(lx - (base - (ly + top)) * t).astype(int) + off
    ok = (nx >= 0) & (nx < DW)
    D[ly[ok], nx[ok]] = 1
    # stem: most common horizontal run in the middle of the band
    lens: List[int] = []
    for yy in range(x_top + round(xh * 0.3), x_top + round(xh * 0.7) + 1):
        lens += [L for _, L in _runs(D[yy - top]) if L >= 2]
    stem, _ = _mode(lens, 2, xh)
    # hairline: most common short vertical run
    hair, n_h = float("nan"), 0
    if stem == stem:
        vl: List[int] = []
        for c in range(DW):
            col = D[:, c]
            if col.any():
                vl += [L for _, L in _runs(col) if 2 <= L <= stem * 0.6]
        hair, n_h = _mode(vl, 2, max(3, stem * 0.6))
    monoline = not (hair == hair) or n_h < max(8, (x1 - x0) * 0.04)
    if monoline:
        hair = stem
    # stems on the middle row
    mid = (x_top + base) // 2
    mr = [(s, L) for s, L in _runs(D[mid - top]) if stem * 0.6 <= L <= stem * 1.6] if stem == stem else []
    pitches, whites = [], []
    for (s0, l0), (s1, l1) in zip(mr, mr[1:]):
        p = (s1 + l1 / 2) - (s0 + l0 / 2)
        if p < stem * 3.2:
            pitches.append(p)
            whites.append(s1 - (s0 + l0))
    stems = [{"x": s - off + (base - mid) * t, "w": L} for s, L in mr]
    # components: dots and dashes
    lab, n = ndi.label(region, structure=np.ones((3, 3)))
    dots, dashes = [], []
    if stem == stem:
        for i, sl in enumerate(ndi.find_objects(lab), start=1):
            if sl is None:
                continue
            comp = lab[sl] == i
            area = int(comp.sum())
            ch, cw = comp.shape
            fill, asp = area / (cw * ch), cw / ch
            cy, cx = ndi.center_of_mass(comp)
            if (stem * 0.5 <= cw <= stem * 2.2 and stem * 0.5 <= ch <= stem * 2.2 and 0.7 < asp < 1.43 and 0.66 < fill < 0.88):
                dots.append({"cx": sl[1].start + cx, "cy": top + sl[0].start + cy,
                             "diameter": math.sqrt(4 * area / math.pi), "roundness": min(1.0, fill / 0.785)})
            elif cw >= 3 * ch and ch <= max(3 * hair, stem * 0.45) and cw >= stem * 0.8:
                dashes.append({"x0": sl[1].start, "y0": top + sl[0].start, "length": cw, "thickness": area / cw})
    # counters: white pockets that do not reach the line box edge
    by0, by1 = max(0, top - 2), min(H - 1, bot + 2)
    bx0, bx1 = max(0, x0 - 2), min(W - 1, x1 + 2)
    bg = 1 - ink[by0:by1 + 1, bx0:bx1 + 1]
    blab, bn = ndi.label(bg)
    counters = []
    if stem == stem:
        edge = set(np.unique(np.concatenate([blab[0], blab[-1], blab[:, 0], blab[:, -1]])).tolist())
        for i, sl in enumerate(ndi.find_objects(blab), start=1):
            if sl is None or i in edge:
                continue
            area = int((blab[sl] == i).sum())
            if area >= stem * stem * 0.15:
                counters.append({"x0": bx0 + sl[1].start, "y0": by0 + sl[0].start,
                                 "w": sl[1].stop - sl[1].start, "h": sl[0].stop - sl[0].start, "area": area})
    return {
        "top": top, "bottom": bot, "x0": x0, "x1": x1, "x_top": x_top, "baseline": base,
        "slant_deg": slant, "tan": t, "mid": mid,
        "stem": stem, "hairline": hair, "monoline": bool(monoline),
        "contrast": (stem / hair) if hair else float("nan"),
        "x_height": xh, "ascender": x_top - top, "descender": bot - base,
        "stem_pitch": _median(pitches), "white_between_stems": _median(whites), "stems": stems,
        "dots": dots, "dashes": dashes, "counters": counters,
        "ink_coverage": total / ((x1 - x0 + 1) * h), "width": x1 - x0 + 1, "height": h,
    }


def _scaled(line: Dict[str, Any], s: float) -> Dict[str, Any]:
    """Convert a measured line from working pixels to original-image pixels."""
    L = {"slant_deg": line["slant_deg"], "monoline": line["monoline"],
         "contrast": round(line["contrast"], 2) if line["contrast"] == line["contrast"] else None,
         "ink_coverage": round(line["ink_coverage"], 3), "stems_counted": len(line["stems"])}
    for k in ("stem", "hairline", "x_height", "ascender", "descender", "stem_pitch", "white_between_stems",
              "width", "height", "top", "bottom", "x0", "x1", "x_top", "baseline"):
        v = line[k]
        L[k] = round(v / s, 2) if v == v else None
    L["dots"] = [{"diameter": round(d["diameter"] / s, 2), "roundness": round(d["roundness"], 2),
                  "center_above_baseline": round((line["baseline"] - d["cy"]) / s, 1)} for d in line["dots"]]
    L["dashes"] = [{"length": round(d["length"] / s, 1), "thickness": round(d["thickness"] / s, 2),
                    "top_above_baseline": round((line["baseline"] - d["y0"]) / s, 1)} for d in line["dashes"]]
    L["counters"] = [{"w": round(c["w"] / s, 1), "h": round(c["h"] / s, 1)} for c in line["counters"]]
    return L


def rules(lines: List[Dict[str, Any]], composition: Dict[str, Any]) -> List[str]:
    """Plain-language rules read off the numbers (Arabic)."""
    out: List[str] = []
    L = [l for l in lines if l.get("stem")]
    near = lambda a, b, tol=0.08: a and b and abs(a / b - 1) <= tol
    if len(L) > 1:
        if all(near(l["stem"], L[0]["stem"]) for l in L):
            out.append(f"قلم واحد لكل الأسطر: سمك الجذع {L[0]['stem']:.1f} بكسل.")
        if all(abs(l["slant_deg"] - L[0]["slant_deg"]) <= 1 for l in L):
            out.append(f"ميل موحّد: {L[0]['slant_deg']}° في كل الأسطر.")
        if all(near(l["width"], L[0]["width"], 0.03) for l in L):
            out.append("عرض موحّد: الأسطر بنفس العرض تقريبًا، فهي كتلة واحدة.")
        if all(near(l["x_height"], L[0]["x_height"], 0.06) for l in L):
            out.append("شبكة مشتركة: ارتفاع الحروف متساوٍ في الأسطر.")
    for i, l in enumerate(L):
        tag = f" (السطر {i + 1})" if len(L) > 1 else ""
        if not l["monoline"] and l["contrast"] and l["contrast"] >= 3.5:
            out.append(f"تباين عالٍ{tag}: الجذع أسمك من الخط الشعري {l['contrast']:.1f} مرة.")
        elif l["monoline"]:
            out.append(f"سمك موحّد{tag}: لا يوجد خط شعري واضح.")
        if l["white_between_stems"] and near(l["white_between_stems"], l["stem"], 0.15):
            out.append(f"إيقاع 1:1{tag}: الأبيض بين الجذوع ({l['white_between_stems']:.1f}) يساوي الجذع تقريبًا.")
        ds = [d["diameter"] for d in l["dots"]]
        if ds:
            dd = float(np.median(ds))
            out.append(f"النقطة{tag}: قطرها {dd / l['stem']:.2f} من الجذع.")
        if l["dashes"] and not l["monoline"] and all(d["thickness"] <= l["hairline"] * 1.6 for d in l["dashes"]):
            out.append(f"شرطات بالقلم الشعري{tag}: الأرجح أنها بديل عن النقاط أو العلامات.")
        if l["ascender"] and l["descender"] and near(l["ascender"], l["descender"], 0.1):
            out.append(f"توازن عمودي{tag}: الصعود يساوي النزول تقريبًا.")
    if composition and near(composition["margin_left"] or 1e-9, composition["margin_right"] or 1e-9, 0.05):
        out.append("تمركز أفقي: الهامش الأيمن يساوي الأيسر.")
    return out


def overlay_png(work: Image.Image, ink: np.ndarray, lines: List[Dict[str, Any]]) -> bytes:
    """Ink coloured by pen (thick navy / thin red) with grid lines, stem boxes and dots."""
    H, W = ink.shape
    rgb = np.full((H, W, 3), 255, np.uint8)
    for l in lines:
        y0, y1 = max(0, l["top"] - 2), min(H - 1, l["bottom"] + 2)
        sub = ink[y0:y1 + 1].astype(bool)
        if l["stem"] != l["stem"]:
            rgb[y0:y1 + 1][sub] = (28, 44, 74)
            continue
        dist = ndi.distance_transform_edt(sub)
        core = dist * 2 >= l["stem"] * 0.62
        thick = sub & (ndi.distance_transform_edt(~core) <= l["stem"] * 0.58) if core.any() else np.zeros_like(sub)
        if l["monoline"]:
            thick = sub
        part = rgb[y0:y1 + 1]
        part[sub] = (207, 63, 83)
        part[thick] = (28, 44, 74)
    img = Image.fromarray(rgb)
    dr = ImageDraw.Draw(img)
    lw = max(1, W // 700)
    for l in lines:
        for y, col in ((l["top"], (101, 81, 208)), (l["x_top"], (35, 137, 95)), (l["baseline"] + 1, (35, 137, 95)), (l["bottom"] + 1, (101, 81, 208))):
            for xx in range(0, W, 14):
                dr.line([(xx, y), (min(W, xx + 8), y)], fill=col, width=lw)
        if l["stem"] == l["stem"]:
            t = l["tan"]
            for s in l["stems"]:
                xb = s["x"] + (l["mid"] - l["baseline"]) * t
                xt = xb + (l["baseline"] - l["x_top"]) * t
                dr.polygon([(xb, l["baseline"] + 1), (xb + s["w"], l["baseline"] + 1), (xt + s["w"], l["x_top"]), (xt, l["x_top"])], outline=(11, 138, 150), width=lw + 1)
        for d in l["dots"]:
            r = d["diameter"] / 2 + 3 * lw
            dr.ellipse([d["cx"] - r, d["cy"] - r, d["cx"] + r, d["cy"] + r], outline=(101, 81, 208), width=lw + 1)
        for d in l["dashes"]:
            dr.rectangle([d["x0"] - 3, d["y0"] - 3, d["x0"] + d["length"] + 3, d["y0"] + d["thickness"] + 3], outline=(207, 63, 83), width=lw + 1)
        for c in l["counters"]:
            dr.rectangle([c["x0"], c["y0"], c["x0"] + c["w"], c["y0"] + c["h"]], outline=(196, 106, 0), width=lw)
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def analyze_image(data: bytes) -> Dict[str, Any]:
    """Full measurement of an image. Returns JSON-ready numbers plus an overlay PNG (bytes)."""
    work, ink, scale, inverted = load_binary(data)
    raw = [m for m in (measure_line(ink, a, b) for a, b in find_lines(ink)) if m]
    H, W = ink.shape
    lines = [_scaled(l, scale) for l in raw]
    comp: Dict[str, Any] = {}
    if raw:
        x0, x1 = min(l["x0"] for l in raw), max(l["x1"] for l in raw)
        comp = {"margin_left": round(x0 / scale, 1), "margin_right": round((W - 1 - x1) / scale, 1),
                "margin_top": round(raw[0]["top"] / scale, 1), "margin_bottom": round((H - 1 - raw[-1]["bottom"]) / scale, 1),
                "line_gaps": [round((b["top"] - a["bottom"] - 1) / scale, 1) for a, b in zip(raw, raw[1:])],
                "baseline_to_baseline": [round((b["baseline"] - a["baseline"]) / scale, 1) for a, b in zip(raw, raw[1:])]}
    return {
        "image": {"width": round(work.width / scale), "height": round(work.height / scale), "inverted": inverted},
        "lines": lines,
        "composition": comp,
        "rules": rules(lines, comp),
        "overlay_png": overlay_png(work, ink, raw),
    }
