"""Step 7: build the new letter from the unified numbers on the reference's paths, put it in the logo, connect, check, export."""
import numpy as np, cv2
from shapely.geometry import Polygon, LineString, Point, box, MultiPolygon
from shapely.ops import unary_union
from shapely import affinity
from scipy.interpolate import splprep, splev
from .geom import stroke_body, smooth_path, arclen, cut, recentre, centrelines
from .measure import h_widths, v_widths, h_shape
from .bezier import fit_closed, resample_closed, svg_d

def chain(edges, ids, reverse=False):
    """join centre-line edges (in the given order) into one path, each turned to continue from the previous one"""
    P = None
    for i in ids:
        Q = np.asarray(edges[i]['P'], float)
        if P is None:
            if len(ids) > 1:
                Nx = np.asarray(edges[ids[1]]['P'], float)
                dq0 = min(np.linalg.norm(Q[0]-Nx[0]), np.linalg.norm(Q[0]-Nx[-1])); dq1 = min(np.linalg.norm(Q[-1]-Nx[0]), np.linalg.norm(Q[-1]-Nx[-1]))
                if dq0 < dq1:
                    Q = Q[::-1]
            P = Q
        else:
            if np.linalg.norm(Q[-1]-P[-1]) < np.linalg.norm(Q[0]-P[-1]):
                Q = Q[::-1]
            P = np.vstack([P, Q])
    return P[::-1] if reverse else P

def spline(points, n=400):
    pts = np.asarray(points, float)
    if len(pts) < 2:
        return pts
    k = min(3, len(pts)-1)
    keep = np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-6]; pts = pts[keep]
    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0, k=min(k, len(pts)-1)); t = np.linspace(0, 1, n)
    return np.stack(splev(t, tck), 1)

def end_width(kind, spec):
    return spec['H_STEM'] if kind == 'stem' else spec['H_HAIR']

def make_stroke(P, st, spec, rise, fall, U):
    """one stroke from its path and the designer's choices -> (polygon, widths, thickest position)"""
    P = np.asarray(P, float)
    if st.get('extend_before'):
        P = np.vstack([np.asarray(st['extend_before'], float), P])
    if st.get('extend_after'):
        P = np.vstack([P, np.asarray(st['extend_after'], float)])
    P = smooth_path(P, s=st.get('smooth', 4), step=max(U/95, 0.05))
    if st.get('type', 'H') == 'H':
        if P[0][0] < P[-1][0] and st.get('writing', 'rtl') == 'rtl':
            P = P[::-1]                                                        # widths run in writing order: right -> left
        W, at = h_widths(P, spec, end_width(st.get('start', 'hair'), spec), end_width(st.get('end', 'hair'), spec), rise, fall)
    else:
        if P[0][1] > P[-1][1]:
            P = P[::-1]
        top = spec['HEAD_TOP'] if st.get('top') == 'head' else spec['V_TOP']
        W = v_widths(P, top, spec['V_MID'], spec['V_BOT']); at = 0.0
    return stroke_body(P, W, U), P, W, at

def build(project, spec, sp, U):
    """project: dict(parts, polys, glyph_edges, strokes, connections, erase, hide, copies)"""
    rise, fall = h_shape(sp)
    made = []
    for st in project.get('strokes', []):
        P = chain(project['glyph_edges'], st['edges'], st.get('reverse', False))
        poly, P2, W, at = make_stroke(P, st, spec, rise, fall, U); made.append(dict(kind='stroke', spec=st, P=P2, W=W, at=at, poly=poly))
    for cn in project.get('connections', []):
        P = spline(cn['points'])
        poly, P2, W, at = make_stroke(P, dict(cn, type='H'), spec, rise, fall, U); made.append(dict(kind='connection', spec=cn, P=P2, W=W, at=at, poly=poly))
    hidden = set(project.get('hide', []))
    base = {k: v for k, v in project['polys'].items() if k not in hidden}
    for r in project.get('erase', []):                    # erase inside one part only (the designer picks it), or everywhere
        rect = box(*r['rect']) if isinstance(r, dict) else box(*r)
        for k in ([r['part']] if isinstance(r, dict) and r.get('part') else list(base)):
            if k in base:
                base[k] = base[k].difference(rect)
    logo = unary_union(list(base.values()))
    extra = []
    for cp in project.get('copies', []):
        src = project['polys'][cp['part']]
        if 'above_stroke' in cp:
            m = made[cp['above_stroke']]; j = int(np.argmax(m['W'])); top = m['P'][j]-[0, m['W'][j]*U/2]
            g = affinity.translate(src, top[0]-src.centroid.x, top[1]-cp.get('gap', 0.2)*U-src.bounds[3])
        else:
            g = affinity.translate(src, cp.get('dx', 0), cp.get('dy', 0))
        extra.append(g)
    full = unary_union([logo]+[m['poly'] for m in made]+extra)
    # check: every built stroke measured on the finished shape, the logo's way
    checks = []
    for i, m in enumerate(made):
        P = m['P']; js = np.linspace(0, len(P)-1, 21).astype(int); js[0] = min(2, len(P)-1); js[-1] = max(len(P)-3, 0)
        ws = np.array([cut(full, P, j, 3*U)[0]/U for j in js]); own = np.array([cut(m['poly'], P, j, 3*U)[0]/U for j in js])
        ws = np.where(ws > own+0.03, np.nan, ws); ok = np.where(np.isfinite(ws))[0]
        if len(ok) < 2:
            checks.append(dict(i=i, kind=m['kind'], clean=0)); continue
        checks.append(dict(i=i, kind=m['kind'], type=m['spec'].get('type', 'H'), length=round(float(arclen(P)[-1]/U), 2),
                           start=round(float(ws[ok[0]]), 3), thickest=round(float(np.nanmax(ws)), 3), thickest_at=int(np.nanargmax(ws)*5),
                           end=round(float(ws[ok[-1]]), 3), clean=int(len(ok)),
                           designed=[round(float(m['W'][0]), 3), round(float(np.max(m['W'])), 3), round(float(m['W'][-1]), 3)]))
    return full, made, checks

def polygon_svg(geom, size, U=None, tol_px=0.15, title='letter'):
    """finished shape -> SVG of cubic Bezier outlines (fitted again, so the designer gets clean editable curves)"""
    geoms = [geom] if geom.geom_type == 'Polygon' else list(getattr(geom, 'geoms', []))
    subs = []
    for q in geoms:
        if q.is_empty:
            continue
        for r in [q.exterior]+list(q.interiors):
            P = np.array(r.coords)[:-1]
            if len(P) < 4:
                continue
            S = 8.0
            Q = resample_closed(P*S, 1.0, 1.0)
            segs, _ = fit_closed(Q, tol_px*S)
            subs.append([np.asarray(c)/S for c in segs])
    W, H = size
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %g %g" width="%g" height="%g">\n <rect width="100%%" height="100%%" fill="#fff"/>\n'
            ' <path id="%s" fill="#000" fill-rule="evenodd" d="%s"/>\n</svg>\n') % (W, H, W*4, H*4, title, svg_d(subs))

def render_png(geoms_colors, bounds, scale=10, bg=255):
    x0, y0, x1, y1 = bounds; W, H = int((x1-x0)*scale), int((y1-y0)*scale)
    im = np.full((H, W, 3), bg, np.uint8)
    for g, col in geoms_colors:
        for q in ([g] if g.geom_type == 'Polygon' else list(getattr(g, 'geoms', []))):
            if q.is_empty or q.geom_type != 'Polygon':
                continue
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [np.round((np.array(q.exterior.coords)-[x0, y0])*scale*16).astype(np.int32)], 1, shift=4)
            for h in q.interiors:
                cv2.fillPoly(m, [np.round((np.array(h.coords)-[x0, y0])*scale*16).astype(np.int32)], 0, shift=4)
            im[m > 0] = col
    return im
