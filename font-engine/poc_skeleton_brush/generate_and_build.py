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
GENERATED = [("uni0628", "ب", "beh.isol"), ("uni062A", "ت", "teh.isol"), ("uni062B", "ث", "theh.isol"),
             ("uni0649", "ى", "alefmaksura.isol"), ("uni062F", "د", "dal.isol"), ("uni0644", "ل", "lam.isol"),
             ("uni0627", "ا", "alef.isol"), ("uni0633", "س", "seen.isol")]


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


def ref_glyph(ch, font_path=None, size=700):
    """Centerline of a letter in a donor typeface, extracted with the same
    single-smooth-path rules as the uploaded word."""
    import pen_path as pp
    if font_path is None:
        ensure_ref_font()
        font_path = REF_FONT
    font = ImageFont.truetype(str(font_path), size, layout_engine=ImageFont.Layout.BASIC)
    img = Image.new("L", (size * 3, size * 3), 0)  # big canvas: tails must not be clipped
    ImageDraw.Draw(img).text((size, size // 2), ch, fill=255, font=font)
    m = np.array(img) > 127
    comps = sb.split_components(m, min_area=30)
    body = max(comps, key=lambda c: c["area"])
    dots = [c for c in comps if c is not body and c["area"] < 0.2 * body["area"]]
    d = ndimage.distance_transform_edt(body["mask"])
    coords, _ = pp.single_walk(body["mask"], d)
    p = sb.smooth_open(sb.resample(sb.orient_rtl(coords), 1.0), 8.0)
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

    # ---- generated letters: understand the style, then create (style_dna.py)
    #   proportions : predicted in nuqtas from meem/zain/noon (proportions.py)
    #   pen path    : similarity derivation from an uploaded letter (beh <- noon) or,
    #                 when no uploaded letter shares its strokes, the closest typeface's
    #                 path re-drawn with the style's curvature gain and terminal curl
    #   brush       : the tablet brush learned from the uploaded word
    import pen_path as pp
    import style_dna as sd
    import arabic_anatomy as aa
    from proportions import load as load_corpus, measure_image, predict_for
    corpus = load_corpus()
    user = measure_image()
    preds, nearest, _ = predict_for(user, corpus)
    donor_name = nearest[0][0]
    donor = sb.HERE / "corpus" / (donor_name.replace(" ", "") + ".ttf")
    D, ref_y = user["nuqta_px"], user["noon_bottom_y"]
    style_paths = {"meem.init": [s for s in strokes if s["comp"] == meem_i][0]["q"],
                   "zain.fina": [s for s in strokes if s["comp"] == zain_i][0]["q"],
                   "noon.isol": [s for s in strokes if s["comp"] == noon_i][0]["q"]}
    dna = sd.measure_dna(style_paths, D, donor)
    lines_y, lines_nuqta = sd.reference_lines(user, preds, corpus)
    # ---- the writer's brush (writer_model.py / exact_brush.py): exact widths of
    # the uploaded strokes; "stretch between guides" keeps each terminal's real
    # size and shape, only the body adapts to the new letter's length
    import exact_brush as eb
    src_w = {}
    for key, ci in (("noon", noon_i), ("zain", zain_i), ("meem", meem_i)):
        q_src = [s for s in strokes if s["comp"] == ci][0]["q"]
        wl_s, wr_s, _ = eb.measure(q_src, comps[ci]["mask"])
        s_src = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q_src, axis=0), axis=1))]
        src_w[key] = (s_src, wl_s, wr_s)
    BRUSH_SRC = {"beh": "noon", "teh": "noon", "theh": "noon", "alefmaksura": "noon", "seen": "noon",
                 "alef": "zain", "dal": "zain", "lam": "zain"}
    from writer_model import contrast_analysis
    _ca = contrast_analysis([(st["q"], *eb.measure(st["q"], comps[st["comp"]]["mask"])[:2])
                             for st in strokes if comps[st["comp"]]["kind"] == "body"], y_top, y_bot)
    v_ratio = _ca.get("vertical_to_horizontal_ratio", 1.0) or 1.0

    def writer_widths(qq, letter, full_profile):
        s_src, wl_s, wr_s = src_w[BRUSH_SRC[letter]]
        L = s_src[-1]
        s_new = np.r_[0, np.cumsum(np.linalg.norm(np.diff(qq, axis=0), axis=1))]
        Ln = s_new[-1]
        body = float(np.median((wl_s + wr_s)[len(wl_s) // 5: -len(wl_s) // 5]))
        g = min(1.5 * body, 0.3 * min(L, Ln))                    # guides: terminal zones
        sm = np.where(s_new < g, s_new, np.where(s_new > Ln - g, L - (Ln - s_new),
                      g + (s_new - g) * (L - 2 * g) / max(Ln - 2 * g, 1e-6)))
        wl_n, wr_n = np.interp(sm, s_src, wl_s), np.interp(sm, s_src, wr_s)
        if not full_profile:  # body = the writer's typical body width, terminals = real ones
            mid = (s_new > g) & (s_new < Ln - g)
            bl, br = np.median(wl_s[len(wl_s) // 5: -len(wl_s) // 5]), np.median(wr_s[len(wr_s) // 5: -len(wr_s) // 5])
            ramp = np.clip(np.minimum(s_new - g, Ln - g - s_new) / max(0.5 * g, 1), 0, 1)
            wl_n = np.where(mid, (1 - ramp) * wl_n + ramp * bl, wl_n)
            wr_n = np.where(mid, (1 - ramp) * wr_n + ramp * br, wr_n)
            # translation contrast measured on the writer: vertical travel is thinner
            T = sb._tangents(qq)
            vert = np.abs(T[:, 1])
            f = 1 - (1 - v_ratio) * vert
            wl_n, wr_n = wl_n * f, wr_n * f
        return np.clip(wl_n, 1, None), np.clip(wr_n, 1, None)

    LETTER_OF = {"uni0628": "beh", "uni062A": "teh", "uni062B": "theh", "uni0649": "alefmaksura",
                 "uni062F": "dal", "uni0644": "lam", "uni0627": "alef", "uni0633": "seen"}
    SRC_FORM = {"noon": "noon.isol", "zain": "zain.fina", "meem": "meem.init"}
    centerlines = {}
    from shapely.geometry import Point
    h0 = bm.beta[0]
    for name, ch, form in GENERATED:
        letter = LETTER_OF[name]
        pd_, dots_d, (bx, by, bw, bh), _ = sd.donor_path(donor, ch)
        if letter in aa.SIMILARITY:
            src = aa.SIMILARITY[letter][0]
            p = style_paths[SRC_FORM[src]].copy()
            method = f"similarity: {src} strokes re-proportioned (Sakkal rule 8) — {aa.SIMILARITY[letter][1]}"
        else:
            p = sd.transfer_curvature(pd_, dna["curvature_gain"], dna["terminal_curl_gain"], dna["donor_nuqta_px"])
            method = (f"transfer: {donor_name} path re-drawn with curvature x{dna['curvature_gain']} "
                      f"and terminal curl x{dna['terminal_curl_gain']}")
        W_t, H_t = preds[form]["w"] * D, preds[form]["h"] * D
        bottom_t = ref_y - preds[form]["bottom"] * D           # image y of the body bottom
        pw, ph = np.ptp(p[:, 0]), np.ptp(p[:, 1])
        wide = pw > 0.15 * ph
        sy = max(H_t - 2 * h0, 0.25 * H_t) / max(ph, 1)
        sx = max(W_t - 2 * h0, 0.25 * W_t) / max(pw, 1) if wide else sy
        for _ in range(6):  # scale the pen path until the brushed INK matches the predicted box
            q = np.stack([(p[:, 0] - p[:, 0].min()) * sx + 100, (p[:, 1] - p[:, 1].max()) * sy + bottom_t - h0], 1)
            q = sb.resample(q, 1.0)
            nodes, segs = pp.fit_smooth(q, tol=4.0, min_gap=30.0, tol_reduce=0.25 * h0)
            qq = sb.sample_beziers(segs, 1.0)
            wl, wr = writer_widths(qq, letter, letter in aa.SIMILARITY)
            body_geom = ribbon_polygon(qq, wl, wr)
            minx, miny, maxx, maxy = body_geom.bounds
            dy = bottom_t - maxy
            if wide:
                sx *= (W_t / max(maxx - minx, 1)) ** 0.8
            sy *= (H_t / max(maxy - miny, 1)) ** 0.8
            if not wide:
                sx = sy
        segs = [P + np.array([0.0, dy]) for P in segs]
        qq = sb.sample_beziers(segs, 1.0)
        wl, wr = writer_widths(qq, letter, letter in aa.SIMILARITY)
        body_geom = ribbon_polygon(qq, wl, wr)
        cont, n = geom_to_contours(body_geom)
        minx, miny, maxx, maxy = body_geom.bounds
        # dots: count and side from the anatomy knowledge base (ت = 2 above, ث = 3 above ...),
        # drawn with the style's own round dot at the style's own clearance
        dc = []
        if "dots" in aa.LETTERS[letter]:
            nd, side = aa.LETTERS[letter]["dots"]
            cx0 = (minx + maxx) / 2
            gap = 2 * r_dot * 1.25
            if nd == 1:
                dc = [(cx0, 0.0)]
            elif nd == 2:
                dc = [(cx0 - gap / 2, 0.0), (cx0 + gap / 2, 0.0)]
            else:  # three: two below, one on top (pyramid)
                dc = [(cx0 - gap / 2, 0.0), (cx0 + gap / 2, 0.0), (cx0, -gap * 0.85 if side == "above" else gap * 0.85)]
            base_y = (miny - clearance - r_dot) if side == "above" else (maxy + clearance + r_dot)
            dc = np.array([(x, base_y + y) for x, y in dc])
            direction = -1.0 if side == "above" else 1.0
            for _ in range(600):
                if min(body_geom.distance(Point(cx, cy)) for cx, cy in dc) >= r_dot + clearance:
                    break
                dc = dc + np.array([0.0, direction * 2.0])
        for cx, cy in dc if len(dc) else []:
            cont.append(circle_contour(cx, cy, r_dot))
            n += 4
        glyph_px[name] = cont
        centerlines[name] = segs
        ink_len = np.r_[0, np.cumsum(np.linalg.norm(np.diff(qq, axis=0), axis=1))][-1]
        stats[name] = {"letter": letter, "method": method, "anatomy": aa.LETTERS[letter]["parts"],
                       "brush": f"writer brush from {BRUSH_SRC[letter]} (stretch between guides)",
                       "mean_stroke_thickness_px": round(float(body_geom.area / max(ink_len, 1)), 1),
                       "shared_parts_with_uploaded": aa.shared_parts(letter, ["meem", "zain", "noon"]),
                       "path_nodes": len(segs) + 1, "outline_nodes": n,
                       "predicted_nuqta_wxh": [preds[form]["w"], preds[form]["h"]],
                       "achieved_nuqta_wxh": [round((maxx - minx) / D, 2), round((maxy - miny) / D, 2)],
                       "path_quality": pp.path_quality(segs, np.ones((4000, 4000), bool))}
    # ---- composition from PARTS of the uploaded letters (compose.py) replaces
    # the donor-based paths: the user's own tooth, bowl, head, neck and tail
    import compose as cp
    _S, comp_out, _, _ = cp.compose_style("style1")
    NAME_OF = {"beh": "uni0628", "teh": "uni062A", "theh": "uni062B", "alefmaksura": "uni0649", "dal": "uni062F",
               "lam": "uni0644", "alef": "uni0627", "seen": "uni0633", "reh": "uni0631"}
    for letter, o in comp_out.items():
        name = NAME_OF[letter]
        cont, n = geom_to_contours(o["geom"])
        glyph_px[name] = cont
        centerlines[name] = o["segs"]
        stats[name] = {"letter": letter, "method": "composed from the uploaded letters: " + o["how"],
                       "path_nodes": len(o["segs"]) + 1, "outline_nodes": n}
    # ---- letters designed automatically (auto_design.py): thousands of
    # compositions of the writer's stroke parts, judged by two eyes. They
    # replace the hand recipes; reh stays "zain without its dot" (user's rule).
    EXTRA = {}
    if (OUT / "auto_design_style1.json").exists():
        import auto_sheet as ash
        _ctx, auto_out, _ = ash.build_full("style1")
        plan = [("alef", "ا", None), ("beh", "ب", "beh"), ("beh", "ت", "teh"), ("beh", "ث", "theh"),
                ("hah", "ح", None), ("hah", "خ", "khah"), ("dal", "د", None), ("dal", "ذ", "thal"),
                ("seen", "س", None), ("seen", "ش", "sheen"), ("sad", "ص", None), ("sad", "ض", "dad"),
                ("tah", "ط", None), ("tah", "ظ", "zah"), ("ain", "ع", None), ("ain", "غ", "ghain"),
                ("feh", "ف", "feh"), ("qaf", "ق", "qaf"), ("kaf", "ك", None), ("lam", "ل", None),
                ("meem", "م", None), ("heh", "ه", None), ("waw", "و", None), ("yeh", "ى", None), ("yeh", "ي", "yeh")]
        for body, ch, dot_letter in plan:
            if body not in auto_out:
                continue
            o = auto_out[body]
            ds = cp.dots_for(dot_letter, o["body"], _S) if dot_letter else []
            cont, n = geom_to_contours(unary_union([o["body"]] + ds))
            name = "uni%04X" % ord(ch)
            glyph_px[name] = cont
            h0_ = float(np.median(o["part"].wl + o["part"].wr)) / 2
            _, centerlines[name] = pp.fit_smooth(o["part"].q, tol=4.0, min_gap=30.0, tol_reduce=0.25 * h0_)
            EXTRA[name] = ch
            stats[name] = {"letter": body, "method": "auto-designed from parts: " + " + ".join(
                               f"{g_['src']}[{g_['a']:.2f}-{g_['b']:.2f}]" + ("R" if g_["rev"] else "") + ("M" if g_["mir"] else "")
                               for g_ in o["genes"]),
                           "eye_shape_p": round(o["info"]["p"], 2), "eye_path_p": round(o["info"].get("p_path", 0), 2),
                           "path_nodes": len(centerlines[name]) + 1, "outline_nodes": n}
    stats["proportions_donor_typefaces"] = nearest
    stats["style_dna"] = dna
    stats["reference_levels_nuqta_above_noon_bottom"] = lines_nuqta
    # everything the anatomy sheet needs, in the uploaded image's px frame
    (OUT / "glyphs_px.json").write_text(json.dumps({
        "contours": {k: [[[c[0], *[list(map(float, pt)) for pt in c[1:]]] for c in cont] for cont in v]
                     for k, v in glyph_px.items()},
        "centerlines": {k: [P.tolist() for P in v] for k, v in centerlines.items()},
        "word_centerlines": [s["segs"] and [P.tolist() for P in s["segs"]] for s in strokes],
        "levels_y": lines_y, "nuqta_px": D}))
    (OUT / "style_dna.json").write_text(json.dumps({"dna": dna, "levels": lines_nuqta, "levels_y_px": lines_y},
                                                   ensure_ascii=False, indent=1))

    # ---- px -> font units
    word_h = (y_bot - y_top)
    S = 900.0 / word_h
    baseline_px = lines_y["base"]  # baseline (قاعدة) from the anatomy reference levels
    to_units = lambda pt: (pt[0] * S, (baseline_px - pt[1]) * S)

    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.t2CharStringPen import T2CharStringPen
    from fontTools.pens.boundsPen import BoundsPen
    from fontTools.feaLib.builder import addOpenTypeFeaturesFromString

    gen_names = [g[0] for g in GENERATED] + [n_ for n_ in EXTRA if n_ not in [g[0] for g in GENERATED]
                                               and n_ not in ("uni0645", "uni0632", "uni0646", "uni0631")]
    order = [".notdef", "space", "uni0645", "uni0645.init", "uni0632", "uni0632.fina", "uni0646",
             "uni0631"] + gen_names + ["uni064F"]
    glyph_px.setdefault("uni0645", glyph_px["uni0645.init"])   # isolated meem: auto-designed if available
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
    cmap = {0x20: "space", 0x645: "uni0645", 0x632: "uni0632", 0x646: "uni0646", 0x631: "uni0631", 0x64F: "uni064F"}
    cmap.update({ord(ch): g for g, ch, _ in GENERATED})
    cmap.update({ord(ch): n_ for n_, ch in EXTRA.items()})
    fb.setupCharacterMap(cmap)
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
@BASES = [uni0645 uni0645.init uni0632 uni0632.fina uni0646 uni0631 {" ".join(gen_names)}];
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
    for name in [g[0] for g in GENERATED]:
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

    tests = {t: shape(t)[0] for t in ["مُزن", "ا ب ت ث ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ك ل م ن ه و ى ي", "زن"]}
    report = {"word_iou_rebuilt_from_pen_and_brush": round(model["iou_word"], 4),
              "pen_path_nodes_word": model["summary"]["total_path_nodes"],
              "brush_model": bm.describe(), "glyphs": stats, "harfbuzz_shaping": tests,
              "font": otf.name, "units_per_px2x": round(S, 4)}
    (OUT / "style1_font_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return otf


if __name__ == "__main__":
    main()
