#!/usr/bin/env python3
"""Generate letters that are NOT in the uploaded image with the learned brush,
then assemble everything into a real, installable OpenType font.

  * glyphs from the image   : rebuilt from (pen path + measured brush)
  * derived glyph (free)    : reh = zain body without its dot (same rasm)
  * generated glyphs        : beh, teh, theh, alef maksura
        pen path  <- centerline of a reference OFL font (Lemonada), mapped into
                     the uploaded word's frame using noon as the anchor letter
        brush     <- BrushModel learned from the uploaded word only
        dots      <- the uploaded word's own round dot (same radius)

Outputs (./output): MuznBrush-Regular.otf, centerline + outline SVGs, and a
proof sheet PNG where the text is shaped by HarfBuzz with the new font.
"""
from __future__ import annotations

import json
import math

import cv2
import numpy as np
import pathops
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from shapely.geometry import Polygon
from shapely.ops import unary_union

import skeleton_brush as sb
from brush_model import BrushModel

OUT = sb.OUT
REF_FONT = sb.HERE / "ref_fonts" / "Lemonada[wght].ttf"
BOX = (1000, 360, 1850, 880)
KAPPA = 0.5522847498


# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------


def ribbon_polygon(q, wl, wr):
    T = sb._tangents(q)
    N = np.stack([-T[:, 1], T[:, 0]], 1)
    L, R = q + N * wl[:, None], q - N * wr[:, None]
    quads = [Polygon([L[i], L[i + 1], R[i + 1], R[i]]).buffer(0) for i in range(len(q) - 1)]
    g = unary_union([p for p in quads if p.area > 0])
    return g.buffer(6, join_style=1).buffer(-6, join_style=1)  # heal slivers/notches at stroke joins


def circle_contour(cx, cy, r):
    k = KAPPA * r
    return [("moveTo", (cx + r, cy)),
            ("curveTo", (cx + r, cy + k), (cx + k, cy + r), (cx, cy + r)),
            ("curveTo", (cx - k, cy + r), (cx - r, cy + k), (cx - r, cy)),
            ("curveTo", (cx - r, cy - k), (cx - k, cy - r), (cx, cy - r)),
            ("curveTo", (cx + k, cy - r), (cx + r, cy - k), (cx + r, cy)),
            ("closePath",)]


def fit_closed(coords, tol=2.0):
    """Closed outline -> cubic Beziers with nodes at extrema/inflections/corners,
    starting at the top-most point with a horizontal handle."""
    p = np.asarray(coords, float)
    if np.allclose(p[0], p[-1]):
        p = p[:-1]
    p = sb.resample(np.vstack([p, p[:1]]), 1.0)[:-1]
    p = np.stack([ndimage.gaussian_filter1d(p[:, 0], 1.5, mode="wrap"),
                  ndimage.gaussian_filter1d(p[:, 1], 1.5, mode="wrap")], 1)
    i0 = int(np.argmin(p[:, 1]))
    p = np.roll(p, -i0, axis=0)
    p = np.vstack([p, p[:1]])
    nodes, segs = sb.fit_stroke(p, tol=tol, min_gap=16.0)
    t = np.array([math.copysign(1.0, p[3, 0] - p[0, 0]), 0.0])
    segs[0][1] = segs[0][0] + np.linalg.norm(segs[0][1] - segs[0][0]) * t
    segs[-1][2] = segs[-1][3] - np.linalg.norm(segs[-1][3] - segs[-1][2]) * t
    cmds = [("moveTo", tuple(segs[0][0]))]
    for P in segs:
        cmds.append(("curveTo", tuple(P[1]), tuple(P[2]), tuple(P[3])))
    cmds.append(("closePath",))
    return cmds, len(segs)


def geom_to_contours(geom, tol=2.0):
    polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    contours, nodes = [], 0
    for pg in polys:
        if pg.area < 50:
            continue
        for ring in [pg.exterior, *pg.interiors]:
            if Polygon(ring).area < max(300, 0.02 * pg.area):  # union artefacts, not real counters
                continue
            c, n = fit_closed(np.asarray(ring.coords), tol)
            contours.append(c)
            nodes += n
    return contours, nodes


def transform_contours(contours, fn):
    out = []
    for c in contours:
        cc = []
        for cmd in c:
            cc.append((cmd[0], *[fn(pt) for pt in cmd[1:]]))
        out.append(cc)
    return out


def contours_to_pathops(contours):
    path = pathops.Path()
    pen = path.getPen()
    for c in contours:
        for cmd in c:
            getattr(pen, cmd[0])(*cmd[1:])
    path.simplify(fix_winding=True, keep_starting_points=False)
    return path


def svg_d(contours):
    parts = []
    for c in contours:
        for cmd in c:
            if cmd[0] == "moveTo":
                parts.append("M%.1f %.1f" % cmd[1])
            elif cmd[0] == "curveTo":
                parts.append("C" + " ".join("%.1f %.1f" % pt for pt in cmd[1:]))
            elif cmd[0] == "lineTo":
                parts.append("L%.1f %.1f" % cmd[1])
            elif cmd[0] == "closePath":
                parts.append("Z")
    return " ".join(parts)


def beziers_d(segs):
    d = "M%.1f %.1f" % tuple(segs[0][0])
    for P in segs:
        d += " C" + " ".join("%.1f %.1f" % tuple(P[i]) for i in (1, 2, 3))
    return d


# ----------------------------------------------------------------------------
# reference skeletons for letters that are not in the image
# ----------------------------------------------------------------------------


def ensure_ref_font():
    """Lemonada (SIL OFL 1.1) from the google/fonts repository; not committed."""
    if not REF_FONT.exists():
        import urllib.request
        REF_FONT.parent.mkdir(exist_ok=True)
        url = "https://raw.githubusercontent.com/google/fonts/main/ofl/lemonada/Lemonada%5Bwght%5D.ttf"
        urllib.request.urlretrieve(url, REF_FONT)


def ref_glyph(ch, size=700):
    ensure_ref_font()
    font = ImageFont.truetype(str(REF_FONT), size, layout_engine=ImageFont.Layout.BASIC)
    img = Image.new("L", (size * 3, size * 3), 0)  # big canvas: tails must not be clipped
    ImageDraw.Draw(img).text((size, size // 2), ch, fill=255, font=font)
    m = np.array(img) > 127
    comps = sb.split_components(m, min_area=30)
    body = max(comps, key=lambda c: c["area"])
    dots = [c for c in comps if c is not body and c["area"] < 0.2 * body["area"]]
    d = ndimage.distance_transform_edt(body["mask"])
    raw, _ = sb.extract_strokes(body["mask"], d)
    p = sb.smooth_open(sb.resample(sb.orient_rtl(raw[0]), 1.0), 4.0)
    p = sb.extend_to_tips(p, body["mask"])
    dot_c = [tuple(np.argwhere(c["mask"]).mean(0)[::-1]) for c in dots]
    return p, dot_c


def spread_dots(centers, r, gap=0.3):
    c = np.array(centers, float)
    if len(c) < 2:
        return c
    mean = c.mean(0)
    for _ in range(200):
        dmin = min(np.linalg.norm(c[i] - c[j]) for i in range(len(c)) for j in range(i + 1, len(c)))
        if dmin >= 2 * r * (1 + gap):
            break
        c = mean + (c - mean) * 1.05
    return c


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------


def main():
    img = sb.HERE / "input" / "muzn_two_styles.jpg"
    mask, rec, model, vis = sb.analyse(img, BOX, "style1")
    comps = sb.split_components(mask)
    ys = np.nonzero(mask)[0]
    y_top, y_bot = ys.min(), ys.max()
    strokes = [dict(q=v["p"], wl=v["wl"], wr=v["wr"], segs=v["segs"], kl=v["kl"], kr=v["kr"],
                    attached_start=v["stroke"] > 0, comp=v["comp"]) for v in vis if v["kind"] == "stroke"]
    bm = BrushModel().fit([s for s in strokes if comps[s["comp"]]["kind"] == "body"], y_top, y_bot)

    # ---- outlines of glyphs that exist in the image (model widths = width keypoints)
    def comp_geom(ci):
        geoms = []
        for s in strokes:
            if s["comp"] != ci:
                continue
            q = sb.sample_beziers(s["segs"], 1.0)
            uq, _ = sb.arclen_u(q)
            wl = np.clip(sb.width_fn(s["kl"])(uq), 0, None)
            wr = np.clip(sb.width_fn(s["kr"])(uq), 0, None)
            geoms.append(ribbon_polygon(q, wl, wr))
        return unary_union(geoms)

    dot_comps = [i for i, c in enumerate(comps) if c["kind"] == "dot"]
    dot_info = {}
    for i in dot_comps:
        yy, xx = np.nonzero(comps[i]["mask"])
        dot_info[i] = (xx.mean(), yy.mean(), math.sqrt(comps[i]["area"] / math.pi))
    r_dot = float(np.mean([v[2] for v in dot_info.values()]))
    bodies = [i for i, c in enumerate(comps) if c["kind"] == "body"]  # RTL: meem, zain, noon
    meem_i, zain_i, noon_i = bodies
    mark_i = [i for i, c in enumerate(comps) if c["kind"] == "mark"][0]

    def nearest_dot(ci):
        x, y, w, h = comps[ci]["bbox"]
        cx = x + w / 2
        return min(dot_comps, key=lambda d: abs(dot_info[d][0] - cx))

    glyph_px = {}  # name -> (contours in px, n_nodes, kind)
    stats = {}
    for name, ci, with_dot in [("uni0645.init", meem_i, False), ("uni0632.fina", zain_i, True),
                               ("uni0646", noon_i, True), ("uni0631", zain_i, False)]:
        cont, n = geom_to_contours(comp_geom(ci))
        if with_dot:
            dx, dy, dr = dot_info[nearest_dot(ci)]
            cont.append(circle_contour(dx, dy, dr))
            n += 4
        glyph_px[name] = cont
        stats[name] = {"source": "from image" if name != "uni0631" else "derived: zain without dot",
                       "outline_nodes": n}
    damma_cont, n = geom_to_contours(comp_geom(mark_i))
    glyph_px["uni064F"] = damma_cont
    stats["uni064F"] = {"source": "from image", "outline_nodes": n}

    # measured dot clearance in the uploaded word (dot edge -> body edge)
    from shapely.geometry import Point
    clear = []
    for ci in (zain_i, noon_i):
        dx, dy, dr = dot_info[nearest_dot(ci)]
        clear.append(comp_geom(ci).distance(Point(dx, dy)) - dr)
    clearance = float(np.mean(clear))
    stats["measured_dot_clearance_px"] = round(clearance, 1)

    # ---- generated letters: reference centerline -> style frame -> learned brush
    noon_q = [s for s in strokes if s["comp"] == noon_i][0]["q"]
    ref_noon, _ = ref_glyph("ن")
    s_scale = (noon_q[:, 1].max() - noon_q[:, 1].min()) / (ref_noon[:, 1].max() - ref_noon[:, 1].min())
    ref_bottom, sty_bottom = ref_noon[:, 1].max(), noon_q[:, 1].max()
    centerlines = {}
    for name, ch in [("uni0628", "ب"), ("uni062A", "ت"), ("uni062B", "ث"), ("uni0649", "ى")]:
        p, dots = ref_glyph(ch)
        q = np.stack([(p[:, 0] - p[:, 0].min()) * s_scale + 100, (p[:, 1] - ref_bottom) * s_scale + sty_bottom], 1)
        q = sb.resample(q, 1.0)
        nodes, segs = sb.fit_stroke(q, tol=4.0, min_gap=30.0)
        qq = sb.sample_beziers(segs, 1.0)
        wl, wr = bm.predict(qq)
        cont, n = geom_to_contours(ribbon_polygon(qq, wl, wr))
        dc = [((x - p[:, 0].min()) * s_scale + 100, (y - ref_bottom) * s_scale + sty_bottom) for x, y in dots]
        body_geom = ribbon_polygon(qq, wl, wr)
        if dc:
            dc = spread_dots(dc, r_dot)
            # keep the same clearance between dot and body as in the uploaded word
            body_cy = qq[:, 1].mean()
            direction = -1.0 if np.mean([c[1] for c in dc]) < body_cy else 1.0
            from shapely.geometry import Point
            for _ in range(400):
                if min(body_geom.distance(Point(cx, cy)) for cx, cy in dc) >= r_dot + clearance:
                    break
                dc = dc + np.array([0.0, direction * 2.0])
        for cx, cy in dc if len(dc) else []:
            cont.append(circle_contour(cx, cy, r_dot))
            n += 4
        glyph_px[name] = cont
        centerlines[name] = segs
        stats[name] = {"source": "generated: Lemonada centerline + learned brush", "path_nodes": len(segs) + 1,
                       "outline_nodes": n}

    # ---- px -> font units
    word_h = (y_bot - y_top)
    S = 900.0 / word_h
    baseline_px = sty_bottom  # bottom of noon's centerline = baseline
    to_units = lambda pt: (pt[0] * S, (baseline_px - pt[1]) * S)

    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.t2CharStringPen import T2CharStringPen
    from fontTools.pens.boundsPen import BoundsPen
    from fontTools.feaLib.builder import addOpenTypeFeaturesFromString

    order = [".notdef", "space", "uni0645", "uni0645.init", "uni0632", "uni0632.fina", "uni0646",
             "uni0631", "uni0628", "uni062A", "uni062B", "uni0649", "uni064F"]
    glyph_px["uni0645"] = glyph_px["uni0645.init"]   # placeholder isolated meem (not in image)
    glyph_px["uni0632"] = glyph_px["uni0632.fina"]   # isolated zain ~ final zain in this style
    charstrings, metrics, bounds = {}, {}, {}
    lsb_rsb = 40
    for g in order:
        if g == ".notdef":
            pen = T2CharStringPen(500, None)
            pen.moveTo((50, 0)); pen.lineTo((450, 0)); pen.lineTo((450, 700)); pen.lineTo((50, 700))
            pen.closePath()
            charstrings[g], metrics[g] = pen.getCharString(), (500, 50)
            continue
        if g == "space":
            pen = T2CharStringPen(250, None)
            charstrings[g], metrics[g] = pen.getCharString(), (250, 0)
            continue
        cont = transform_contours(glyph_px[g], to_units)
        path = contours_to_pathops(cont)
        bp = BoundsPen(None)
        path.draw(bp)
        xmin, ymin, xmax, ymax = bp.bounds
        is_mark = g == "uni064F"
        shift = -xmin + (0 if is_mark else lsb_rsb)
        adv = 0 if is_mark else int(round(xmax - xmin + 2 * lsb_rsb))
        moved = pathops.Path()
        mp = moved.getPen()
        from fontTools.pens.transformPen import TransformPen
        path.draw(TransformPen(mp, (1, 0, 0, 1, shift, 0)))
        pen = T2CharStringPen(adv, None)
        moved.draw(pen)
        charstrings[g] = pen.getCharString()
        metrics[g] = (adv, int(round(xmin + shift)))
        bounds[g] = (xmin + shift, ymin, xmax + shift, ymax)

    # damma placement: keep its position relative to meem as in the image
    m_units = transform_contours(glyph_px["uni0645.init"], to_units)
    d_units = transform_contours(glyph_px["uni064F"], to_units)

    def raw_xmin(conts):
        bp = BoundsPen(None)
        contours_to_pathops(conts).draw(bp)
        return bp.bounds

    mb, db = raw_xmin(m_units), raw_xmin(d_units)
    anchor_mark = (int(round((db[2] - db[0]) / 2)), int(round(db[1])))           # in damma glyph coords
    anchor_base = (int(round(db[0] - mb[0] + lsb_rsb + (db[2] - db[0]) / 2)), int(round(db[1])))

    asc = int(max(b[3] for b in bounds.values())) + 20
    desc = int(min(b[1] for b in bounds.values())) - 20
    fb = FontBuilder(1000, isTTF=False)
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap({0x20: "space", 0x645: "uni0645", 0x632: "uni0632", 0x646: "uni0646", 0x631: "uni0631",
                          0x628: "uni0628", 0x62A: "uni062A", 0x62B: "uni062B", 0x649: "uni0649",
                          0x64F: "uni064F"})
    fb.setupCFF("MuznBrush-Regular", {"FullName": "Muzn Brush Regular"}, charstrings, {})
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=asc, descent=desc)
    fb.setupNameTable({"familyName": "Muzn Brush", "styleName": "Regular",
                       "uniqueFontIdentifier": "MuznBrush-Regular-POC", "fullName": "Muzn Brush Regular",
                       "psName": "MuznBrush-Regular", "version": "Version 0.1 (proof of concept)"})
    fb.setupOS2(sTypoAscender=asc, sTypoDescender=desc, sTypoLineGap=0, usWinAscent=asc,
                usWinDescent=-desc, version=4, fsSelection=0x40 | 0x80)
    fb.setupPost()
    fea = f"""
languagesystem DFLT dflt;
languagesystem arab dflt;
@BASES = [uni0645 uni0645.init uni0632 uni0632.fina uni0646 uni0631 uni0628 uni062A uni062B uni0649];
table GDEF {{ GlyphClassDef @BASES, , [uni064F], ; }} GDEF;
feature init {{ sub uni0645 by uni0645.init; }} init;
feature fina {{ sub uni0632 by uni0632.fina; }} fina;
markClass uni064F <anchor {anchor_mark[0]} {anchor_mark[1]}> @TOP;
feature mark {{ pos base uni0645.init <anchor {anchor_base[0]} {anchor_base[1]}> mark @TOP; }} mark;
"""
    addOpenTypeFeaturesFromString(fb.font, fea)
    otf = OUT / "MuznBrush-Regular.otf"
    fb.save(str(otf))

    # ---- SVGs: editable centerlines (pen tool) + expanded outlines
    W, H = mask.shape[1], mask.shape[0]
    pen_svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
               '<rect width="100%" height="100%" fill="#111"/>',
               '<g fill="none" stroke="#ff5a36" stroke-width="4" stroke-linecap="round">']
    for s in strokes:
        pen_svg.append(f'<path d="{beziers_d(s["segs"])}"/>')
    pen_svg.append("</g></svg>")
    (OUT / "style1_centerlines_pen_tool.svg").write_text("\n".join(pen_svg))
    gen_svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1400 900">',
               '<rect width="100%" height="100%" fill="#111"/>']
    x_off = 0
    for name in ["uni0628", "uni062A", "uni062B", "uni0649"]:
        gen_svg.append(f'<g transform="translate({x_off} 0)"><path fill="#eee" d="{svg_d(glyph_px[name])}"/>'
                       f'<path fill="none" stroke="#ff5a36" stroke-width="4" d="{beziers_d(centerlines[name])}"/></g>')
    gen_svg.append("</svg>")
    (OUT / "generated_letters_overlay.svg").write_text("\n".join(gen_svg))

    # ---- verify shaping with HarfBuzz and render proof sheet
    import uharfbuzz as hb
    blob = hb.Blob.from_file_path(str(otf))
    hbfont = hb.Font(hb.Face(blob))

    def shape(text):
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(hbfont, buf, {})
        names = [hbfont.glyph_to_string(i.codepoint) for i in buf.glyph_infos]
        return names, buf

    tests = {t: shape(t)[0] for t in ["مُزن", "ب ت ث ى ر", "زن"]}
    report = {"word_iou_rebuilt_from_pen_and_brush": round(model["iou_word"], 4),
              "pen_path_nodes_word": model["summary"]["total_path_nodes"],
              "brush_model": bm.describe(), "glyphs": stats, "harfbuzz_shaping": tests,
              "font": otf.name, "units_per_px2x": round(S, 4)}
    (OUT / "style1_font_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return otf


if __name__ == "__main__":
    main()
