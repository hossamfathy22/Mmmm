"""Step 6: the same-family reference font: shape a word, take the new letter's outline, bring it to the logo's size and place."""
import numpy as np
import uharfbuzz as hb
from fontTools.ttLib import TTFont
from fontTools.pens.recordingPen import DecomposingRecordingPen
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely import affinity
import io

def _outline(GS, g):
    pen = DecomposingRecordingPen(GS); GS[g].draw(pen); polys = []; cur = []
    for op, args in pen.value:
        if op == 'moveTo':
            cur = [np.array(args[0], float)]
        elif op == 'lineTo':
            cur.append(np.array(args[0], float))
        elif op == 'qCurveTo':
            pts = [cur[-1]]+[np.array(a, float) for a in args if a is not None]; offs = pts[1:-1]; end = pts[-1]; p0 = pts[0]
            if not offs:
                cur.append(end); continue
            for i, o in enumerate(offs):
                nxt = end if i == len(offs)-1 else (o+offs[i+1])/2
                t = np.linspace(0, 1, 12)[1:, None]; cur += list((1-t)**2*p0+2*(1-t)*t*o+t**2*nxt); p0 = nxt
        elif op == 'curveTo':
            p0 = cur[-1]; c1, c2, p3 = [np.array(a, float) for a in args]; t = np.linspace(0, 1, 14)[1:, None]
            cur += list((1-t)**3*p0+3*(1-t)**2*t*c1+3*(1-t)*t**2*c2+t**3*p3)
        elif op in ('closePath', 'endPath'):
            if len(cur) > 2:
                polys.append(np.array(cur))
            cur = []
    return polys

def _even_odd(rings):
    """glyph contours -> shape with the font's own fill rule (non-zero: outer contours turn one way, holes the other;
    overlapping outer contours stay filled)"""
    if not rings:
        return Polygon()
    sa = lambda r: 0.5*np.sum(r[:, 0]*np.roll(r[:, 1], -1)-np.roll(r[:, 0], -1)*r[:, 1])
    areas = [sa(r) for r in rings]
    main = np.sign(areas[int(np.argmax(np.abs(areas)))])
    outer = [Polygon(r).buffer(0) for r, a in zip(rings, areas) if np.sign(a) == main]
    holes = [Polygon(r).buffer(0) for r, a in zip(rings, areas) if np.sign(a) != main]
    g = unary_union(outer)
    if holes:
        g = g.difference(unary_union(holes))
    return g

def shape_word(font_bytes, text, features=None):
    """shape a word with the font; returns glyphs in reading order of the shaped run (font units, y up)"""
    face = hb.Face(font_bytes); font = hb.Font(face)
    buf = hb.Buffer(); buf.add_str(text); buf.guess_segment_properties(); hb.shape(font, buf, features or {})
    tt = TTFont(io.BytesIO(font_bytes)); GS = tt.getGlyphSet(); order = tt.getGlyphOrder()
    x = 0; out = []
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        g = order[info.codepoint]
        poly = _even_odd(_outline(GS, g))
        if not poly.is_empty:
            poly = affinity.translate(poly, x+pos.x_offset, pos.y_offset)
        out.append(dict(index=len(out), name=g, cluster=int(info.cluster), char=text[info.cluster] if info.cluster < len(text) else '', poly=poly))
        x += pos.x_advance
    return out, tt['head'].unitsPerEm

def to_logo(poly, s, dx, dy):
    """font units (y up) -> logo frame (pixels, y down): x*s+dx, -y*s+dy"""
    return affinity.affine_transform(poly, [s, 0, 0, -s, dx, dy])

def place(glyphs, match_index, match_height_px, target_index, right_x, baseline_y):
    """scale so that glyph `match_index` stands `match_height_px` above the baseline; put the right edge of the new letter at right_x"""
    gm = glyphs[match_index]['poly']; s = match_height_px/max(gm.bounds[3], 1e-9)
    gt = glyphs[target_index]['poly']
    dx = right_x-gt.bounds[2]*s; dy = baseline_y
    return s, dx, dy
