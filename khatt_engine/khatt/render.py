"""Turn a Layout into SVG, PNG and an annotated construction sheet."""
from __future__ import annotations

import io
import math
from typing import List, Optional, Tuple

from PIL import Image, ImageChops, ImageDraw

from .typeset import Layout

ARABIC_FONTS = "Vazirmatn, 'Noto Naskh Arabic', 'Noto Sans Arabic', Tahoma, 'Segoe UI', sans-serif"


def _sk(lay: Layout, x: float, y: float) -> Tuple[float, float]:
    return x + y * lay.style.tan, -y


def _bounds(lay: Layout, pad: float) -> Tuple[float, float, float, float]:
    pts = [_sk(lay, x, y) for s in lay.shapes for poly in s for x, y in poly]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs) - pad, min(ys) - pad, max(xs) - min(xs) + 2 * pad, max(ys) - min(ys) + 2 * pad


def _d(lay: Layout, shape) -> str:
    out = []
    for poly in shape:
        pts = [_sk(lay, x, y) for x, y in poly]
        out.append("M" + " L".join(f"{a:.2f},{b:.2f}" for a, b in pts) + "Z")
    return "".join(out)


def to_svg(lay: Layout, fill: str = "#111111", background: Optional[str] = "#ffffff", pad: Optional[float] = None) -> str:
    pad = lay.style.stem * 1.5 if pad is None else pad
    vx, vy, vw, vh = _bounds(lay, pad)
    bg = f'<rect x="{vx:.1f}" y="{vy:.1f}" width="{vw:.1f}" height="{vh:.1f}" fill="{background}"/>' if background else ""
    body = "".join(f'<path fill-rule="evenodd" d="{_d(lay, s)}"/>' for s in lay.shapes)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vx:.1f} {vy:.1f} {vw:.1f} {vh:.1f}" '
            f'width="{vw:.0f}" height="{vh:.0f}">{bg}<g fill="{fill}">{body}</g></svg>')


def to_png(lay: Layout, height: int = 360, ink: Tuple[int, int, int] = (17, 17, 17), supersample: int = 3) -> bytes:
    pad = lay.style.stem * 1.5
    vx, vy, vw, vh = _bounds(lay, pad)
    s = height / vh
    W, H = max(1, int(vw * s)), max(1, int(vh * s))
    S = supersample
    mask = Image.new("L", (W * S, H * S), 0)
    for shape in lay.shapes:
        polys = [[((a - vx) * s * S, (b - vy) * s * S) for a, b in (_sk(lay, x, y) for x, y in poly)] for poly in shape]
        if len(polys) == 1:
            ImageDraw.Draw(mask).polygon(polys[0], fill=255)
            continue
        part = Image.new("L", mask.size, 0)
        for i, p in enumerate(polys):
            layer = Image.new("L", mask.size, 0)
            ImageDraw.Draw(layer).polygon(p, fill=255)
            part = ImageChops.logical_xor(part.convert("1"), layer.convert("1")).convert("L") if i else layer
        mask = ImageChops.lighter(mask, part)
    mask = mask.resize((W, H), Image.LANCZOS)
    img = Image.new("RGB", (W, H), (255, 255, 255))
    img.paste(Image.new("RGB", (W, H), ink), mask=mask)
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def sheet_svg(lay: Layout, title: str = "", white: Optional[float] = None) -> str:
    """Construction drawing: grid lines, stem boxes, pitch and white, with Arabic labels."""
    st = lay.style
    U, X, A, D, h = st.stem, st.x_height, st.ascender, st.descender, st.hair
    x0, y0, x1, y1 = lay.bbox()
    fs = max(9.0, U * 0.42)
    pad_l, pad_r, pad_t, pad_b = U * 1.5, U * 9.5, U * 5.5, U * 5.5
    left, right = x0 - pad_l, x1 + pad_r + abs(st.tan) * (X + A)
    top, bottom = -(y1 + pad_t), -(y0 - pad_b)
    vw, vh = right - left, bottom - top
    C = {"ink": "#1e2d5a", "stem": "#0b8a96", "guide": "#6551d0", "base": "#23895f", "dim": "#c46a00", "hair": "#cf3f53", "engine": "#8e9bb0"}
    o: List[str] = []

    def text(x, y, s, col, size=fs, anchor="end", weight=700):
        # labels are right-to-left: "end" puts the text to the right of x, "start" to the left
        o.append(f'<text x="{x:.1f}" y="{y:.1f}" fill="{col}" font-size="{size:.1f}" font-weight="{weight}" '
                 f'text-anchor="{anchor}" font-family="{ARABIC_FONTS}" direction="rtl" unicode-bidi="plaintext">{s}</text>')

    guides = [(X + A, C["guide"], f"الصعود +{A:.0f}"), (X, C["base"], f"ارتفاع الحروف {X:.0f}"),
              (0, C["base"], "خط القاعدة 0"), (-D, C["guide"], f"النزول −{D:.0f}")]
    gx1 = x1 + abs(st.tan) * (X + A) + U * 1.2
    for y, col, lab in guides:
        o.append(f'<line x1="{left:.1f}" x2="{gx1:.1f}" y1="{-y:.1f}" y2="{-y:.1f}" stroke="{col}" stroke-width="{max(0.6, h * 0.18):.2f}" stroke-dasharray="{U*0.25:.1f} {U*0.18:.1f}"/>')
        text(gx1 + U * 0.4, -y + fs * 0.35, lab, col)
    order = sorted(lay.shapes, key=lambda sh: getattr(sh, "source", "engine") == "logo")
    mixed = any(getattr(sh, "source", "") == "logo" for sh in lay.shapes)
    colour = lambda sh: C["ink"] if (not mixed or getattr(sh, "source", "engine") == "logo") else C["engine"]
    body = "".join(f'<path fill-rule="evenodd" fill="{colour(sh)}" d="{_d(lay, sh)}"/>' for sh in order)
    o.insert(0, f'<g>{body}</g>')
    has_logo = any(getattr(sh, "source", "") == "logo" for sh in lay.shapes)
    if has_logo:
        ly = -(y0 - pad_b * 0.35)
        for i, (col, t) in enumerate([(C["ink"], "قطعة من الشعار"), (C["engine"], "قطعة مستنتجة بقواعد الشعار")]):
            xx = gx1 - i * U * 7
            o.append(f'<rect x="{xx - U * 0.5:.1f}" y="{ly - fs * 0.8:.1f}" width="{fs * .9:.1f}" height="{fs * .9:.1f}" fill="{col}"/>')
            text(xx - U * 0.7, ly, t, "#16202c", size=fs * 0.85, anchor="start")
    sw = max(0.8, h * 0.22)
    centres = []
    for p in lay.placed:
        for xc, t, b in p.glyph.stems:
            ax = p.x + xc
            pts = [_sk(lay, ax - U / 2, t), _sk(lay, ax + U / 2, t), _sk(lay, ax + U / 2, b), _sk(lay, ax - U / 2, b)]
            o.append('<polygon points="' + " ".join(f"{a:.1f},{c:.1f}" for a, c in pts) + f'" fill="none" stroke="{C["stem"]}" stroke-width="{sw:.2f}"/>')
            if abs(t - X) < 1e-6:
                centres.append(ax)
    if centres:
        c0 = max(centres)
        a, b = _sk(lay, c0 - U / 2, X + U * 0.35), _sk(lay, c0 + U / 2, X + U * 0.35)
        o.append(f'<line x1="{a[0]:.1f}" x2="{b[0]:.1f}" y1="{a[1]:.1f}" y2="{b[1]:.1f}" stroke="{C["stem"]}" stroke-width="{sw*1.6:.2f}"/>')
        text((a[0] + b[0]) / 2, a[1] - fs * 0.45, f"جذع {U:.1f}", C["stem"], anchor="middle")
        near = sorted(c for c in centres if 0 < c0 - c < 1.6 * st.pitch)
        if near:
            c1 = near[-1]
            ya = X * 0.5
            p, q = _sk(lay, c1 + U / 2, ya), _sk(lay, c0 - U / 2, ya)
            o.append(f'<line x1="{p[0]:.1f}" x2="{q[0]:.1f}" y1="{p[1]:.1f}" y2="{q[1]:.1f}" stroke="{C["dim"]}" stroke-width="{sw*1.6:.2f}"/>')
            text((p[0] + q[0]) / 2, p[1] - fs * 0.4, f"أبيض {(white if white is not None else st.white):.1f}", C["dim"], size=fs * 0.85, anchor="middle")
    # one label per kind of part, with a leader line
    LBL = {"كأس الوصل": f"قاع الكأس {h:.1f}", "رأس": f"رأس {2 * U + st.white:.0f}×{X:.0f}", "ذيل": f"ذيل −{D:.0f}",
           "كأس سفلي": f"كأس سفلي −{D:.0f}", "كأس أيمن": "كأس يمين", "نقطة": f"نقطة ⌀{st.dot:.1f}",
           "شرطة": f"شرطة {1.5 * st.dot:.0f}×{h:.1f}", "راية الدال": f"راية الدال {0.57 * U:.0f}×{h:.1f}",
           "قاعدة الدال": f"قاعدة الدال {h:.1f}"}
    seen = set()
    used_y: List[float] = []
    for sh in lay.shapes:
        nm = getattr(sh, "name", "")
        if nm not in LBL or nm in seen:
            continue
        seen.add(nm)
        pts = [_sk(lay, x, y) for poly in sh for x, y in poly]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax, ay = (min(xs) + max(xs)) / 2, max(ys) if nm in ("كأس الوصل", "ذيل", "كأس سفلي", "قاعدة الدال") else min(ys)
        below = ay > -0.1 * X
        ty = (D + U * 1.6) if below else -(X + A + U * 0.9)
        while any(abs(ty - u) < fs * 1.3 and True for u in used_y):
            ty += fs * 1.3 if below else -fs * 1.3
        used_y.append(ty)
        col = C["hair"] if nm in ("كأس الوصل", "شرطة", "راية الدال", "قاعدة الدال") else (C["guide"] if nm == "نقطة" else C["ink"])
        o.append(f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{ax:.1f}" y2="{ty - (fs * 0.9 if below else -fs * 0.3):.1f}" stroke="{col}" stroke-width="{sw * 0.8:.2f}"/>')
        text(ax, ty, LBL[nm], col, size=fs * 0.8, anchor="middle")
    # gaps between letters that do not join, measured at mid height
    for a, b in zip(lay.placed, lay.placed[1:]):
        if b.form in ("M", "F"):
            continue
        xa = a.x + a.glyph.left
        xb = b.x + b.glyph.entry_edge
        if xa - xb <= 0.5:
            continue
        p1, p2 = _sk(lay, xb, X * 0.5), _sk(lay, xa, X * 0.5)
        o.append(f'<line x1="{p1[0]:.1f}" x2="{p2[0]:.1f}" y1="{p1[1]:.1f}" y2="{p2[1]:.1f}" stroke="{C["dim"]}" stroke-width="{sw * 1.6:.2f}"/>')
        text((p1[0] + p2[0]) / 2, p1[1] - fs * 0.4, f"{xa - xb:.0f}", C["dim"], size=fs * 0.8, anchor="middle")
    if len(centres) >= 2:
        cs = sorted(centres, reverse=True)
        steps = [a - b for a, b in zip(cs, cs[1:]) if a - b < 1.6 * st.pitch]
        if steps:
            c0 = cs[0]
            p1, p2 = _sk(lay, c0 - steps[0], X + U * 1.2), _sk(lay, c0, X + U * 1.2)
            o.append(f'<line x1="{p1[0]:.1f}" x2="{p2[0]:.1f}" y1="{p1[1]:.1f}" y2="{p2[1]:.1f}" stroke="{C["dim"]}" stroke-width="{sw * 1.4:.2f}"/>')
            text((p1[0] + p2[0]) / 2, p1[1] - fs * 0.4, f"خطوة {steps[0]:.0f}", C["dim"], size=fs * 0.8, anchor="middle")
    sx = gx1 - U * 0.2
    o.append(f'<line x1="{sx:.1f}" y1="0" x2="{sx + st.tan * (X + A):.1f}" y2="{-(X + A):.1f}" stroke="{C["stem"]}" stroke-width="{sw*1.4:.2f}"/>')
    text(sx, D * 0.55, f"ميل {st.slant:.1f}°", C["stem"], anchor="middle")
    if title:
        text(left + U * 0.5, top + fs * 1.6, title, "#16202c", size=fs * 1.3, anchor="end")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{left:.1f} {top:.1f} {vw:.1f} {vh:.1f}" '
            f'width="{vw * 2:.0f}" height="{vh * 2:.0f}"><rect x="{left:.1f}" y="{top:.1f}" width="{vw:.1f}" height="{vh:.1f}" fill="#ffffff"/>{"".join(o)}</svg>')
