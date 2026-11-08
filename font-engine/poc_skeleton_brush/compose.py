#!/usr/bin/env python3
"""Compose the missing letters from PARTS of the uploaded letters.

Arabic letters are built from shared parts (Sakkal rule 8; Zoghbi's anatomy).
The uploaded word already contains real, hand-made parts:

  noon  ->  tooth (سن)  +  bowl (كاسة)
  zain  ->  head (رأس)  +  neck (رقبة)  +  tail (ذنب)

Each part keeps its own pen path, its exact brush widths (exact_brush.py) and
therefore its own start/end angles and terminal shapes. A letter is a recipe:

  reh   = zain without its dot
  dal   = zain head + zain neck shortened + zain tail
  beh   = noon body, one dot below      teh = two dots above   theh = three
  lam   = stem (zain head + neck, stretched up) + noon bowl
  alefmaksura = zain head + short neck + noon bowl
  seen  = two small teeth (noon's tooth, scaled) + noon (tooth + bowl)
  alef  = zain head + zain neck stretched to the alef height, round end

Parts are joined end to start WITHOUT rotating them (their angles are the
writer's), through a short smooth bridge (cubic Hermite, G1) with the width
blended across it. Dots are copies of the word's own dot.
"""
from __future__ import annotations

import json
import math

import cv2
import numpy as np

import arabic_anatomy as aa
import exact_brush as eb
import skeleton_brush as sb

STYLES = {"style1": (1000, 360, 1850, 880), "style2": (120, 360, 860, 880)}


class Part:
    def __init__(self, q, wl, wr, src, start_term=False, end_term=False):
        self.q, self.wl, self.wr = np.asarray(q, float), np.asarray(wl, float), np.asarray(wr, float)
        self.src, self.start_term, self.end_term = src, start_term, end_term
        self.labels = [src] * len(self.q)

    def copy(self):
        p = Part(self.q.copy(), self.wl.copy(), self.wr.copy(), self.src, self.start_term, self.end_term)
        p.labels = list(self.labels)
        return p


def resample_part(p, step=1.0):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p.q, axis=0), axis=1))]
    n = max(int(s[-1] / step), 2)
    t = np.linspace(0, s[-1], n + 1)
    out = p.copy()
    out.q = np.stack([np.interp(t, s, p.q[:, 0]), np.interp(t, s, p.q[:, 1])], 1)
    out.wl, out.wr = np.interp(t, s, p.wl), np.interp(t, s, p.wr)
    idx = np.clip(np.searchsorted(s, t), 0, len(p.labels) - 1)
    out.labels = [p.labels[i] for i in idx]
    return out


def sub(stroke, i0, i1, name):
    q, wl, wr = stroke
    n = len(q)
    return Part(q[i0:i1 + 1], wl[i0:i1 + 1], wr[i0:i1 + 1], name, i0 == 0, i1 == n - 1)


def scale(p, f, about="start"):
    """Change the part's size along its own drawing; the brush (writer's
    thickness) is NOT scaled."""
    o = p.q[0] if about == "start" else p.q[-1]
    out = p.copy()
    out.q = o + (p.q - o) * f
    return resample_part(out)


def translate(p, d):
    out = p.copy()
    out.q = p.q + np.asarray(d)
    return out


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def attach(a, b, bridge=40):
    """b continues a: moved so it starts where a ends (never rotated: each
    part keeps the writer's angles), then the joint is replaced by a smooth
    cubic bridge tangent to both parts, widths blended across it."""
    b2 = translate(b, a.q[-1] - b.q[0])
    q = np.vstack([a.q, b2.q[1:]])
    wl = np.r_[a.wl, b2.wl[1:]]
    wr = np.r_[a.wr, b2.wr[1:]]
    labels = a.labels + b2.labels[1:]
    j = len(a.q) - 1
    i0, i1 = max(j - bridge, 2), min(j + bridge, len(q) - 3)
    P0, P3 = q[i0], q[i1]
    t0, t1 = _unit(q[i0] - q[i0 - 2]), _unit(q[i1 + 2] - q[i1])
    L = np.linalg.norm(P3 - P0)
    t = np.linspace(0, 1, i1 - i0 + 1)[:, None]
    P1, P2 = P0 + t0 * L / 3, P3 - t1 * L / 3
    q[i0:i1 + 1] = (1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * P1 + 3 * (1 - t) * t * t * P2 + t ** 3 * P3
    tt = t[:, 0]
    wl[i0:i1 + 1] = (1 - tt) * wl[i0] + tt * wl[i1]
    wr[i0:i1 + 1] = (1 - tt) * wr[i0] + tt * wr[i1]
    out = Part(q, wl, wr, a.src + "+" + b.src, a.start_term, b.end_term)
    out.labels = labels
    return resample_part(out)


def chain(parts, bridge=40):
    out = parts[0]
    for p in parts[1:]:
        out = attach(out, p, bridge)
    return out


def end_cap(p, kind="round", at="end"):
    """A part cut out of the middle of a stroke has a flat cut. Give it the
    writer's terminal: 'round' (style 1) = a half-disc grown out of the end,
    'flat' (geometric style) = leave the cut."""
    if kind != "round":
        return p
    out = p.copy()
    if at == "end":
        T = _unit(p.q[-1] - p.q[-4])
        h = 0.5 * (p.wl[-1] + p.wr[-1])
        ext = p.q[-1] + T * np.arange(1, h)[:, None]
        d = np.arange(1, h)
        shape = np.sqrt(np.clip(1 - (d / h) ** 2, 0, 1))
        out.q = np.vstack([p.q, ext])
        out.wl = np.r_[p.wl, p.wl[-1] * shape]
        out.wr = np.r_[p.wr, p.wr[-1] * shape]
        out.labels = p.labels + [p.labels[-1]] * len(ext)
        out.end_term = True
    else:
        r = end_cap(Part(p.q[::-1], p.wr[::-1], p.wl[::-1], p.src), kind, "end")
        out = Part(r.q[::-1], r.wr[::-1], r.wl[::-1], p.src, True, p.end_term)
        out.labels = (p.labels[:1] * (len(r.q) - len(p.q))) + p.labels
    return out


# ----------------------------------------------------------------------------
# reading the uploaded letters
# ----------------------------------------------------------------------------


def load_style(style):
    box = STYLES[style]
    img = sb.HERE / "input" / "muzn_two_styles.jpg"
    mask, _, model, vis = sb.analyse(img, box, style)
    comps = sb.split_components(mask)
    bodies = [i for i, c in enumerate(comps) if c["kind"] == "body"]      # RTL: meem, zain, noon
    meem_i, zain_i, noon_i = bodies
    st = {}
    for key, ci in (("meem", meem_i), ("zain", zain_i), ("noon", noon_i)):
        v = [v for v in vis if v["kind"] == "stroke" and v["comp"] == ci][0]
        wl, wr, _ = eb.measure(v["p"], comps[ci]["mask"])
        st[key] = (v["p"], wl, wr)
    # the word's own dot (circle in style 1, rectangle in style 2): the one over noon
    others = [c for i, c in enumerate(comps) if c["kind"] != "body"]
    nx = comps[noon_i]["bbox"][0] + comps[noon_i]["bbox"][2] / 2
    dot = min([c for c in others if c["area"] > 0.02 * comps[noon_i]["area"]],
              key=lambda c: abs(c["bbox"][0] + c["bbox"][2] / 2 - nx))
    cnt, _ = cv2.findContours(dot["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    dot_poly = max(cnt, key=len)[:, 0, :].astype(float)
    dot_poly -= dot_poly.mean(0)
    # dot clearance in the word: dot edge to noon body
    from shapely.geometry import Polygon
    dp = Polygon(dot_poly + np.argwhere(dot["mask"]).mean(0)[::-1])
    nb_c, _ = cv2.findContours(comps[noon_i]["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    clearance = Polygon(max(nb_c, key=len)[:, 0, :]).distance(dp)
    # baseline: corpus ratio of baseline height to noon height (proportions corpus)
    corpus = json.loads((sb.OUT / "corpus_nuqta_metrics.json").read_text())
    ratio = np.median([v["baseline_above_noon_bottom"] / v["noon.isol"]["h"] for v in corpus.values()])
    x, y, w, h = comps[noon_i]["bbox"]
    base_y = (y + h) - ratio * h
    return dict(strokes=st, dot_poly=dot_poly, clearance=float(clearance), base_y=float(base_y),
                mask=mask, comps=comps, ids=dict(meem=meem_i, zain=zain_i, noon=noon_i))


def split_noon(st, base_y):
    """noon = tooth (start -> up -> back down to the base) + bowl (rest)."""
    q = st[0]
    n = len(q)
    apex = int(np.argmin(q[: int(0.6 * n), 1]))
    after = np.nonzero(q[apex:, 1] >= base_y)[0]
    k = apex + int(after[0]) if len(after) else (apex + int(np.argmax(q[apex:, 1]))) // 2
    return {"tooth": sub(st, 0, k, "noon.tooth"), "bowl": sub(st, k, n - 1, "noon.bowl")}


def split_zain(st):
    """zain = head (start -> top -> until it descends steeply) + neck (down to
    the lowest point) + tail (lowest point -> end)."""
    q = st[0]
    n = len(q)
    T = sb._tangents(q)
    top = int(np.argmin(q[: int(0.5 * n), 1]))
    steep = np.nonzero(T[top:, 1] > 0.6)[0]
    k1 = top + int(steep[0]) if len(steep) else top + 10
    k2 = k1 + int(np.argmax(q[k1:, 1]))
    return {"head": sub(st, 0, k1, "zain.head"), "neck": sub(st, k1, k2, "zain.neck"),
            "tail": sub(st, k2, n - 1, "zain.tail")}


# ----------------------------------------------------------------------------
# recipes
# ----------------------------------------------------------------------------


def recipes(S, cap):
    st = S["strokes"]
    N = split_noon(st["noon"], S["base_y"])
    Z = split_zain(st["zain"])
    noon = sub(st["noon"], 0, len(st["noon"][0]) - 1, "noon")
    zain = sub(st["zain"], 0, len(st["zain"][0]) - 1, "zain")
    zh = Z["head"]
    zn = Z["neck"]
    zt = Z["tail"]
    neck_len = np.linalg.norm(zn.q[-1] - zn.q[0])
    noon_h = np.ptp(st["noon"][0][:, 1])
    out = {}
    out["reh"] = (zain.copy(), "zain without its dot", "same")
    out["dal"] = (chain([zh, scale(zn, 0.55), zt]), "zain head + neck shortened to 55% + zain tail", "base")
    for k in ("beh", "teh", "theh"):
        out[k] = (noon.copy(), "the noon body itself; only the dots differ", "same")
    stem = end_cap(chain([zh, scale(zn, 1.0)]), cap, "end")  # alef: head + neck, round end
    alef_target = 1.65 * noon_h                                 # alef ~ tall stem (Sky level)
    f = alef_target / max(np.ptp(stem.q[:, 1]), 1)
    out["alef"] = (end_cap(chain([zh, scale(zn, f)]), cap, "end"), "zain head + zain neck stretched to the alef height", "base")
    out["lam"] = (chain([zh, scale(zn, f * 0.9), N["bowl"]]), "stem (zain head + stretched neck) + noon bowl", "bowl")
    out["alefmaksura"] = (chain([zh, scale(zn, 0.45), N["bowl"]]), "zain head + short neck + noon bowl", "bowl")
    tooth_small = scale(N["tooth"], 0.8)
    out["seen"] = (chain([end_cap(tooth_small, cap, "start"), tooth_small, noon], bridge=45),
                   "two teeth (noon's tooth at 80%) + noon (tooth + bowl)", "bowl")
    return out, N, Z


def place(part, mode, S):
    """Vertical placement on the writer's levels."""
    q_noon = S["strokes"]["noon"][0]
    if mode == "same":
        return part
    if mode == "bowl":                       # bowl bottom where the noon's bowl is
        return translate(part, (0, q_noon[:, 1].max() - part.q[:, 1].max()))
    if mode == "base":                       # ink bottom on the baseline
        bottom = (part.q[:, 1] + np.maximum(part.wl, part.wr)).max()
        return translate(part, (0, S["base_y"] - bottom))
    return part


def dots_for(letter, geom, S):
    from shapely.geometry import Polygon
    from shapely import affinity
    spec = aa.LETTERS[letter].get("dots")
    if not spec:
        return []
    nd, side = spec
    poly = Polygon(S["dot_poly"])
    w = np.ptp(S["dot_poly"][:, 0])
    minx, miny, maxx, maxy = geom.bounds
    cx = (minx + maxx) / 2
    gap = w * 1.25
    offs = {1: [(0, 0)], 2: [(-gap / 2, 0), (gap / 2, 0)],
            3: [(-gap / 2, 0), (gap / 2, 0), (0, -gap * 0.9 if side == "above" else gap * 0.9)]}[nd]
    hgt = np.ptp(S["dot_poly"][:, 1])
    y0 = miny - S["clearance"] - hgt / 2 if side == "above" else maxy + S["clearance"] + hgt / 2
    ds = [affinity.translate(poly, cx + dx, y0 + dy) for dx, dy in offs]
    step = -2.0 if side == "above" else 2.0
    for _ in range(800):
        if min(geom.distance(d) for d in ds) >= S["clearance"]:
            break
        ds = [affinity.translate(d, 0, step) for d in ds]
    return ds


def compose_style(style, cap=None):
    S = load_style(style)
    if cap is None:
        cap = "round" if style == "style1" else "flat"
    from generate_and_build import ribbon_polygon
    from shapely.ops import unary_union
    import pen_path as pp
    R, N, Z = recipes(S, cap)
    out = {}
    for letter, (part, how, mode) in R.items():
        part = place(resample_part(part), mode, S)
        body = ribbon_polygon(part.q, np.clip(part.wl, 1, None), np.clip(part.wr, 1, None))
        ds = dots_for(letter, body, S)
        geom = unary_union([body] + ds)
        h0 = float(np.median(part.wl + part.wr)) / 2
        _, segs = pp.fit_smooth(part.q, tol=4.0, min_gap=30.0, tol_reduce=0.25 * h0)
        out[letter] = dict(part=part, geom=geom, body=body, how=how, segs=segs)
    return S, out, N, Z


def draw_sheet(style, S, out, N, Z):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    AR = FontProperties(fname="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    colors = {"noon.tooth": "#e4572e", "noon.bowl": "#2b7bba", "noon": "#8e44ad", "zain.head": "#f3a712",
              "zain.neck": "#29a36a", "zain.tail": "#c0392b", "zain": "#7f8c8d"}
    order = ["alef", "lam", "dal", "reh", "alefmaksura", "seen", "theh", "teh", "beh"]
    fig, ax = plt.subplots(figsize=(30, 8))
    x = 0.0
    for letter in order:
        o = out[letter]
        g = o["geom"]
        minx, miny, maxx, maxy = g.bounds
        dx = x - minx
        polys = [g] if g.geom_type == "Polygon" else list(g.geoms)
        for pg in polys:
            xs, ys = pg.exterior.xy
            ax.fill(np.array(xs) + dx, ys, color="#dedede", lw=0)
            for hole in pg.interiors:
                hx, hy = hole.xy
                ax.fill(np.array(hx) + dx, hy, color="white", lw=0)
        p = o["part"]
        labs = np.array([l.split("+")[0] for l in p.labels])
        for lab in np.unique(labs):
            m = labs == lab
            ax.scatter(p.q[m, 0] + dx, p.q[m, 1], s=1.5, color=colors.get(lab, "black"))
        for P in o["segs"]:
            ax.plot(P[0][0] + dx, P[0][1], "s", color="white", mec="black", ms=4)
        ax.plot(o["segs"][-1][3][0] + dx, o["segs"][-1][3][1], "s", color="white", mec="black", ms=4)
        ax.text((minx + maxx) / 2 + dx, S["base_y"] + 330, aa.LETTERS[letter]["ch"], fontproperties=AR, fontsize=26,
                ha="center")
        x += (maxx - minx) + 140
    ax.axhline(S["base_y"], color="#e4572e", lw=0.8)
    ax.text(-40, S["base_y"], "قاعدة", fontproperties=AR, color="#e4572e", ha="right", va="center", fontsize=14)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in colors.items()]
    ax.legend(handles=handles, loc="upper right", fontsize=10, ncol=7)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{style}: letters composed from parts of the uploaded noon and zain "
                 "(path colour = which part it came from)", fontsize=14)
    fig.tight_layout()
    fig.savefig(sb.OUT / f"{style}_COMPOSED_LETTERS.png", dpi=80)
    plt.close(fig)


def main():
    report = {}
    for style in STYLES:
        S, out, N, Z = compose_style(style)
        draw_sheet(style, S, out, N, Z)
        report[style] = {k: {"recipe": v["how"], "path_nodes": len(v["segs"]) + 1} for k, v in out.items()}
        print(style, {k: v["path_nodes"] for k, v in report[style].items()})
    (sb.OUT / "composed_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
