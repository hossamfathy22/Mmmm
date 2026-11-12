"""Stage 7: the drawings — logo construction sheet, part box, pen rule, word construction sheet."""
from __future__ import annotations

import base64
import io
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from .bezier import to_path
from .library import Library
from .measure import Line
from .understand import Reading

FONT = "Vazirmatn, 'Noto Naskh Arabic', 'Noto Sans Arabic', Tahoma, 'Segoe UI', sans-serif"
C = {"ink": "#1e2d5a", "thin": "#cf3f53", "stem": "#0b8a96", "guide": "#6551d0", "base": "#23895f",
     "low": "#c46a00", "dim": "#c46a00", "engine": "#9aa6b5", "text": "#16202c"}


def _esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Labels:
    """Places text without overlapping earlier labels (simple greedy search)."""

    def __init__(self, fs: float):
        self.fs = fs
        self.boxes: List[Tuple[float, float, float, float]] = []
        self.out: List[str] = []

    def _box(self, x, y, w, anchor):
        x0 = x - w if anchor == "left" else (x - w / 2 if anchor == "middle" else x)
        return (x0, y - self.fs, x0 + w, y + 0.25 * self.fs)

    def add(self, x: float, y: float, text: str, color: str, anchor: str = "right", size: float = 1.0,
            weight: int = 700, tries: Tuple[float, ...] = (0, -1.25, 1.25, -2.5, 2.5, -3.75, 3.75)) -> None:
        fs = self.fs * size
        w = 0.56 * fs * len(text) + 4
        for t in tries:
            yy = y + t * fs
            b = self._box(x, yy, w, anchor)
            if not any(b[0] < o[2] and o[0] < b[2] and b[1] < o[3] and o[1] < b[3] for o in self.boxes):
                break
        self.boxes.append(b)
        # labels are right-to-left: text-anchor "end" grows to the right of x, "start" to the left
        a = {"right": "end", "left": "start", "middle": "middle"}[anchor]
        self.out.append(f'<text x="{x:.1f}" y="{yy:.1f}" fill="{color}" font-size="{fs:.1f}" font-weight="{weight}" '
                        f'text-anchor="{a}" direction="rtl" font-family="{FONT}" stroke="#fff" stroke-width="{fs * 0.22:.1f}" '
                        f'paint-order="stroke">{_esc(text)}</text>')


def _f(v: float) -> str:
    return f"{v:.1f}" if abs(v) < 100 else f"{v:.0f}"


# ------------------------------------------------------------------ pen colouring
def thickness_png(ink: np.ndarray, lines: List[Line]) -> bytes:
    H, W = ink.shape
    rgb = np.full((H, W, 3), 255, np.uint8)
    rgb[ink.astype(bool)] = (190, 196, 206)
    for L in lines:
        if not (L.stem == L.stem):
            continue
        y0, y1 = max(0, L.top - 2), min(H - 1, L.bot + 2)
        sub = ink[y0:y1 + 1].astype(bool)
        dist = ndi.distance_transform_edt(sub)
        core = dist * 2 >= 0.62 * L.stem
        thick = sub & (ndi.distance_transform_edt(~core) <= 0.58 * L.stem) if core.any() else np.zeros_like(sub)
        if L.monoline:
            thick = sub
        part = rgb[y0:y1 + 1]
        part[sub] = (207, 63, 83)
        part[thick] = (30, 45, 90)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "PNG", optimize=True)
    return buf.getvalue()


# ------------------------------------------------------------------ 1. the logo construction sheet
def logo_sheet(png: bytes, size: Tuple[int, int], scale: float, lines: List[Line], readings: List[Reading]) -> str:
    k = 1.0 / scale
    W, H = size
    stems = [L.stem for L in lines if L.stem == L.stem]
    fs = max(9.0, 0.42 * (np.median(stems) * k if stems else 20))
    margin_r, margin_t, margin_b = fs * 15, fs * 4, fs * 6
    lab = Labels(fs)
    o: List[str] = [f'<image href="data:image/png;base64,{base64.b64encode(png).decode()}" x="0" y="0" width="{W}" height="{H}"/>']
    sw = max(0.8, fs * 0.09)
    dots_all = [(L.idx, p.size * k) for L in lines for p in L.dots]
    for L, R in zip(lines, readings):
        if not L.stems:
            continue
        P = lambda x, y, L=L: (L.to_image(x, y)[0] * k, L.to_image(x, y)[1] * k)
        rowy = lambda y, L=L: (L.top + L.row_of_y(y)) * k
        X = L.x_height
        guides = [(X + L.ascender, C["guide"], f"الصعود +{_f(L.ascender * k)}"),
                  (X, C["base"], f"ارتفاع الحروف {_f(X * k)}")]
        guides += [(v.y, C["low"], f"أسنان منخفضة −{_f((X - v.y) * k)}") for v in L.levels if v.name == "lowered"]
        guides += [(0, C["base"], "خط القاعدة 0"), (-L.descender, C["guide"], f"النزول −{_f(L.descender * k)}")]
        for y, col, text in guides:
            yy = rowy(y)
            o.append(f'<line x1="0" x2="{W + fs:.1f}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{col}" stroke-width="{sw:.2f}" stroke-dasharray="{fs * .6:.1f} {fs * .4:.1f}"/>')
            lab.add(W + fs * 1.3, yy + fs * 0.35, text, col, "right", 0.9, tries=(0, -0.9, 0.9, -1.8, 1.8))
        # stems
        for i, s in enumerate(L.stems):
            pts = [P(s.x - s.w / 2, s.top), P(s.x + s.w / 2, s.top), P(s.x + s.w / 2, max(s.bottom, 0)), P(s.x - s.w / 2, max(s.bottom, 0))]
            o.append('<polygon points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + f'" fill="none" stroke="{C["stem"]}" stroke-width="{sw * 1.3:.2f}"/>')
        s0 = L.stems[0]
        a, b = P(s0.x, s0.top)
        lab.add(a, b - fs * 0.5, f"جذع {_f(L.stem * k)}", C["stem"], "middle", 0.9)
        # letters under the line
        for u in R.units:
            if not u.stems:
                continue
            xm = float(np.mean([L.stems[i].x for i in u.stems]))
            a, b = P(xm, -L.descender)
            lab.add(a, b + fs * 1.6, u.letter + ("" if u.sure else "؟"), C["text"], "middle", 1.25)
        # joins: show one typical white
        js = [j for j in R.joins if j.kind == "join"]
        if js:
            med = float(np.median([j.white for j in js]))
            j = min(js, key=lambda v: abs(v.white - med))
            ra, lb = L.stems[j.right], L.stems[j.left]
            y = 0.55 * X
            p1, p2 = P(lb.x + lb.w / 2, y), P(ra.x - ra.w / 2, y)
            o.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" stroke="{C["dim"]}" stroke-width="{sw * 2:.2f}"/>')
            lab.add((p1[0] + p2[0]) / 2, p1[1] - fs * 0.45, f"أبيض {_f(L.white * k)}", C["dim"], "middle", 0.85)
        # dots, dashes, bars, tails, counters
        for p in L.dots:
            cx, cy = P(p.cx, p.cy)
            r = p.size * k / 2 + 2 * sw
            o.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="none" stroke="{C["guide"]}" stroke-width="{sw * 1.4:.2f}"/>')
            others = [i for i, d in dots_all if i != L.idx and abs(d / (p.size * k) - 1) <= 0.04]
            extra = f" = نقطة السطر {others[0] + 1}" if others else ""
            lab.add(cx + r + fs * 0.3, cy - r * 0.2, f"نقطة ⌀{_f(p.size * k)}{extra}", C["guide"], "right", 0.85)
        top_y = rowy(X + L.ascender) - fs * 0.5          # free space above the line
        bot_y = rowy(-L.descender) + fs * 2.9            # free space under the letter names
        for p in L.dashes:
            q = [P(p.x0, p.y0), P(p.x1, p.y0), P(p.x1, p.y1), P(p.x0, p.y1)]
            o.append('<polygon points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in q) + f'" fill="none" stroke="{C["thin"]}" stroke-width="{sw * 1.2:.2f}"/>')
            text = f"شرطة {_f(p.size * k)}×{p.thick * k:.1f}"
            if p.cy > X / 2:
                lab.add((q[0][0] + q[1][0]) / 2, top_y - fs * 0.4, text, C["thin"], "middle", 0.85, tries=(0, -1.25, -2.5))
            else:
                lab.add(q[1][0] + fs * 0.3, (q[0][1] + q[2][1]) / 2 + fs * 0.3, text, C["thin"], "right", 0.85)
        for bar in R.bars:
            if bar.attached:
                a, b = P((bar.x0 + bar.x1) / 2, bar.y)
                lab.add(a, top_y - fs * 0.4, f"خط علوي {_f((bar.x1 - bar.x0) * k)}×{bar.thick * k:.1f}", C["thin"], "middle", 0.85, tries=(0, -1.25, -2.5))
        for t in R.tails:
            a, b = P(t.x0, t.y0)
            lab.add(a - fs * 0.3, b + fs * 0.2, "ذيل" + (" يسار" if t.direction == "left" else " يمين" if t.direction == "right" else ""), C["ink"], "left", 0.85)
        areas = [c.area for c in L.counters]
        for c in L.counters:
            q = [P(c.x0, c.y0), P(c.x1, c.y0), P(c.x1, c.y1), P(c.x0, c.y1)]
            o.append('<polygon points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in q) + f'" fill="none" stroke="{C["low"]}" stroke-width="{sw:.2f}" stroke-dasharray="{fs * .3:.1f} {fs * .2:.1f}"/>')
            twin = sum(abs(a / c.area - 1) <= 0.05 for a in areas) > 1
            lab.add((q[0][0] + q[1][0]) / 2, bot_y, f"فراغ {_f((c.x1 - c.x0) * k)}×{_f((c.y1 - c.y0) * k)}" + (" متطابق" if twin else ""), C["low"], "middle", 0.8, tries=(0, 1.25, 2.5))
        # slant
        last = L.stems[-1]
        xs = last.x - 2.2 * L.stem
        p1, p2 = P(xs, -0.2 * L.stem), P(xs, X + 0.2 * L.stem)
        o.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" stroke="{C["stem"]}" stroke-width="{sw * 1.6:.2f}"/>')
        lab.add(p1[0], p1[1] + fs * 1.2, f"ميل {L.slant:.1f}°", C["stem"], "middle", 0.85)
    legend = [(C["ink"], "قلم سميك"), (C["thin"], "قلم رفيع"), (C["stem"], "صندوق الجذع المائل"), (C["guide"], "الصعود والنزول"),
              (C["base"], "ارتفاع الحروف والقاعدة"), (C["low"], "مستوى منخفض / فراغ")]
    lx = W + margin_r - fs
    for i, (col, text) in enumerate(legend):
        y = H + margin_b * 0.55
        x = lx - i * (W + margin_r) / len(legend)
        o.append(f'<rect x="{x - fs:.1f}" y="{y - fs * 0.8:.1f}" width="{fs * .9:.1f}" height="{fs * .9:.1f}" fill="{col}"/>')
        o.append(f'<text x="{x - fs * 1.4:.1f}" y="{y:.1f}" font-size="{fs * .85:.1f}" fill="{C["text"]}" text-anchor="start" direction="rtl" font-family="{FONT}">{_esc(text)}</text>')
    vw, vh = W + margin_r, H + margin_t + margin_b
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 {-margin_t:.1f} {vw:.1f} {vh:.1f}" width="{vw * 1.6:.0f}" height="{vh * 1.6:.0f}">'
            f'<rect x="0" y="{-margin_t:.1f}" width="{vw:.1f}" height="{vh:.1f}" fill="#fff"/>' + "".join(o) + "".join(lab.out) + "</svg>")


# ------------------------------------------------------------------ 2. the part box
def parts_sheet(lib: Library) -> str:
    parts = list(lib.parts.values())
    if not parts:
        return ""
    U = lib.stem
    fs = max(9.0, 0.45 * U)
    cell_w, cell_h = 6.5 * U, 7.5 * U
    cols = min(4, len(parts))
    rows = math.ceil(len(parts) / cols)
    o: List[str] = []
    for n, p in enumerate(parts):
        cx0 = (cols - 1 - n % cols) * cell_w          # right to left
        cy0 = (n // cols) * cell_h
        xs = [x for poly in p.polys for x, _ in poly]; ys = [y for poly in p.polys for _, y in poly]
        bw, bh = max(xs) - min(xs), max(ys) - min(ys)
        s = min(1.0, (cell_w - 1.2 * U) / max(bw, 1), (cell_h - 3.2 * U) / max(bh, 1))
        ox = cx0 + cell_w / 2 - (min(xs) + bw / 2) * s
        oy = cy0 + 0.8 * U + max(ys) * s
        tf = lambda x, y: (ox + x * s, oy - y * s)
        d = "".join(to_path([tuple(tf(*q) for q in seg) for seg in c]) for c in p.curves if c)
        o.append(f'<rect x="{cx0 + 4:.1f}" y="{cy0 + 4:.1f}" width="{cell_w - 8:.1f}" height="{cell_h - 8:.1f}" rx="6" fill="#f6f8fa" stroke="#d5dce4"/>')
        o.append(f'<path fill-rule="evenodd" d="{d}" fill="{C["ink"]}"/>')
        # Bezier skeleton: on-curve points and handles
        for c in p.curves:
            for seg in c:
                (a, b, cc, e) = [tf(*q) for q in seg]
                o.append(f'<path d="M{a[0]:.1f},{a[1]:.1f} L{b[0]:.1f},{b[1]:.1f} M{e[0]:.1f},{e[1]:.1f} L{cc[0]:.1f},{cc[1]:.1f}" stroke="{C["stem"]}" stroke-width=".6" fill="none"/>')
                o.append(f'<circle cx="{a[0]:.1f}" cy="{a[1]:.1f}" r="1.6" fill="{C["thin"]}"/>')
        nseg = sum(len(c) for c in p.curves)
        AR = {"width": "عرض", "white_r": "أبيض يمين", "white_l": "أبيض يسار", "depth": "عمق", "reach": "امتداد", "top": "قمة",
              "diameter": "قطر", "roundness": "استدارة", "length": "طول", "thick": "سمك", "w": "عرض", "h": "ارتفاع"}
        vals = [f"{AR.get(k, k)} {v:.1f}" for k, v in p.info.items()
                if isinstance(v, (int, float)) and k in AR and not (k.startswith("white") and v == 0)]
        info = " · ".join(vals)
        ty = cy0 + cell_h - 2.3 * U
        for i, (t, col, size) in enumerate([(p.name, C["text"], 1.0), (p.source, "#5b6878", 0.78),
                                            (info, "#5b6878", 0.72), (f"{nseg} منحنى بيزيه", C["stem"], 0.72)]):
            o.append(f'<text x="{cx0 + cell_w / 2:.1f}" y="{ty + i * fs * 1.2:.1f}" font-size="{fs * size:.1f}" fill="{col}" text-anchor="middle" '
                     f'direction="rtl" font-family="{FONT}" font-weight="{700 if i == 0 else 400}">{_esc(t)}</text>')
    vw, vh = cols * cell_w, rows * cell_h
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {vw:.1f} {vh:.1f}" width="{vw * 1.4:.0f}" height="{vh * 1.4:.0f}"><rect width="{vw:.1f}" height="{vh:.1f}" fill="#fff"/>{"".join(o)}</svg>'


# ------------------------------------------------------------------ 3. the pen rule
def pen_sheet(L: Line, k: float) -> str:
    s, w = L.pen_points
    if not len(s) or not L.pen_fit:
        return ""
    w = w * k
    U, h, pw = L.pen_fit["stem"] * k, L.pen_fit["hair"] * k, L.pen_fit["power"]
    Wd, Hd, m = 640.0, 360.0, 56.0
    ymax = max(float(np.percentile(w, 99)), U) * 1.15
    X = lambda v: m + v * (Wd - 2 * m)
    Y = lambda v: Hd - m - v / ymax * (Hd - 2 * m)
    o = [f'<rect width="{Wd}" height="{Hd + 70}" fill="#fff"/>']
    for gy in np.linspace(0, ymax, 5):
        o.append(f'<line x1="{m}" x2="{Wd - m}" y1="{Y(gy):.1f}" y2="{Y(gy):.1f}" stroke="#e3e9ef"/>'
                 f'<text x="{m - 8}" y="{Y(gy) + 4:.1f}" font-size="11" text-anchor="end" fill="#5b6878" font-family="{FONT}">{gy:.0f}</text>')
    idx = np.linspace(0, len(s) - 1, min(len(s), 900)).astype(int)
    o += [f'<circle cx="{X(a):.1f}" cy="{Y(b):.1f}" r="1.6" fill="{C["thin"]}" fill-opacity=".35"/>' for a, b in zip(s[idx], w[idx])]
    curve = " ".join(f"{X(v):.1f},{Y(h + (U - h) * v ** pw):.1f}" for v in np.linspace(0, 1, 60))
    o.append(f'<polyline points="{curve}" fill="none" stroke="{C["ink"]}" stroke-width="2.5"/>')
    for deg in (0, 30, 45, 60, 90):
        v = math.sin(math.radians(deg))
        o.append(f'<text x="{X(v):.1f}" y="{Hd - m + 18:.1f}" font-size="11" text-anchor="middle" fill="#5b6878" font-family="{FONT}">{deg}°</text>')
    o.append(f'<text x="{Wd / 2}" y="{Hd - m + 38:.1f}" font-size="12" text-anchor="middle" fill="{C["text"]}" direction="rtl" font-family="{FONT}">اتجاه الخط (0° أفقي ← 90° عمودي)</text>')
    o.append(f'<text x="{Wd - m}" y="24" font-size="13" text-anchor="start" fill="{C["text"]}" direction="rtl" font-weight="700" font-family="{FONT}">'
             f'السمك = {h:.1f} + ({U:.1f} − {h:.1f}) × |جيب الزاوية|^{pw:.2f}</text>')
    o.append(f'<text x="{Wd - m}" y="44" font-size="11" text-anchor="start" fill="#5b6878" direction="rtl" font-family="{FONT}">'
             f'{L.pen_fit["samples"]} نقطة مقيسة على الهيكل العظمي للحروف · الخطأ الوسيط {L.pen_fit["median_error"] * k:.2f}</text>')
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {Wd} {Hd + 70}" width="{Wd * 1.4:.0f}" height="{(Hd + 70) * 1.4:.0f}">{"".join(o)}</svg>'


# ------------------------------------------------------------------ 2b. the ideal part box (the six parts, drawn from the measured numbers)
def ideal_parts_sheet(st) -> str:
    from . import geometry as g
    from .letters import tail
    from .typeset import flat
    U, h, X, D, P, d = st.stem, st.hair, st.x_height, st.descender, st.pitch, st.dot
    hw = 2 * U + st.white
    parts = [
        ("الجذع", f"{U:.1f} × {X:.1f}", [g._fillet_stem(st, 0, X, 0, False, False, False, False)]),
        ("كأس الوصل", f"جانبان {U:.1f} · قاع {h:.1f} · أبيض {st.white:.1f}",
         list(flat([g._engine_stem(st, P, X, 0, False, True, False, False), g._engine_stem(st, 0, X, 0, True, False, False, False)]))),
        ("الرأس (حلقة)", f"{hw:.1f} × {X:.1f} · فراغ {st.white:.1f}", [g.ring(st, -hw, 0, 0, X)]),
        ("الذيل", f"ينزل −{D:.1f} ويرقّ كلما مال", list(flat([tail(st, 0, 0.62 * X)]))),
        ("النقطة", f"⌀{d:.1f}", [g.circle(0, X / 2, d / 2, st.tan)]),
        ("الشرطة", f"{1.5 * d:.1f} × {h:.1f}", [g.rect(-0.75 * d, 0.75 * d, X / 2 - h / 2, X / 2 + h / 2)]),
    ]
    fs = max(9.0, 0.45 * U)
    cell = 5.2 * U + st.tan * X
    top, bot = X + 0.6 * U, -D - 0.4 * U
    o = []
    for i, (name, info, shapes) in enumerate(parts):
        xs = [x + y * st.tan for s in shapes for poly in s for x, y in poly]
        ox = (len(parts) - 1 - i) * cell + cell / 2 - (min(xs) + max(xs)) / 2
        for s in shapes:
            dd = "".join("M" + " L".join(f"{ox + x + y * st.tan:.1f},{-y:.1f}" for x, y in poly) + "Z" for poly in s)
            o.append(f'<path fill-rule="evenodd" fill="{C["ink"]}" d="{dd}"/>')
        cx = (len(parts) - 1 - i) * cell + cell / 2
        o.append(f'<text x="{cx:.1f}" y="{-bot + 1.4 * fs:.1f}" font-size="{fs:.1f}" font-weight="700" text-anchor="middle" direction="rtl" font-family="{FONT}" fill="{C["text"]}">{_esc(name)}</text>')
        o.append(f'<text x="{cx:.1f}" y="{-bot + 2.6 * fs:.1f}" font-size="{fs * .8:.1f}" text-anchor="middle" direction="rtl" font-family="{FONT}" fill="#5b6878">{_esc(info)}</text>')
    W = len(parts) * cell
    for y, col in ((X, C["base"]), (0, C["base"]), (-D, C["guide"])):
        o.insert(0, f'<line x1="0" x2="{W:.1f}" y1="{-y:.1f}" y2="{-y:.1f}" stroke="{col}" stroke-width="{max(.6, h * .15):.2f}" stroke-dasharray="{U * .3:.1f} {U * .2:.1f}"/>')
    # the pen rule as five strokes
    by = -bot + 5.6 * fs + 1.8 * U
    for j, deg in enumerate((90, 60, 45, 30, 0)):
        w = st.width_at(math.sin(math.radians(deg)))
        cx = W - (j + 0.5) * W / 5
        a = math.radians(deg); L2 = 1.2 * U
        dx, dy = math.cos(a) * L2, math.sin(a) * L2
        nx, ny = -math.sin(a) * w / 2, math.cos(a) * w / 2
        pts = [(cx - dx + nx, by + dy - ny), (cx + dx + nx, by - dy - ny), (cx + dx - nx, by - dy + ny), (cx - dx - nx, by + dy + ny)]
        o.append('<polygon points="' + " ".join(f"{p:.1f},{q:.1f}" for p, q in pts) + f'" fill="{C["ink"]}"/>')
        o.append(f'<text x="{cx:.1f}" y="{by + 1.7 * U:.1f}" font-size="{fs * .85:.1f}" text-anchor="middle" font-family="{FONT}" fill="{C["text"]}">{deg}°</text>')
        o.append(f'<text x="{cx:.1f}" y="{by + 1.7 * U + 1.2 * fs:.1f}" font-size="{fs * .8:.1f}" text-anchor="middle" direction="rtl" font-family="{FONT}" fill="{C["thin"]}">سمك {w:.1f}</text>')
    o.append(f'<text x="{W - 4:.1f}" y="{by - 1.9 * U:.1f}" font-size="{fs * .9:.1f}" text-anchor="start" direction="rtl" font-family="{FONT}" font-weight="700" fill="{C["text"]}">'
             f'قاعدة القلم: السمك = {h:.1f} + ({U:.1f} − {h:.1f}) × |جيب الزاوية|^{st.contrast_power:.2f}</text>')
    vh = top - bot + 5.6 * fs + 4.4 * U + 2 * fs
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 {-top:.1f} {W:.1f} {vh:.1f}" width="{W * 1.6:.0f}" height="{vh * 1.6:.0f}">'
            f'<rect x="0" y="{-top:.1f}" width="{W:.1f}" height="{vh:.1f}" fill="#fff"/>{"".join(o)}</svg>')
