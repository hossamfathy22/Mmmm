"""Geometry on the curves: polygons, straight cuts across a stroke, centre-lines (true middle of each cut)."""
import numpy as np, cv2
from shapely.geometry import Polygon, LineString, Point, MultiPolygon
from shapely.ops import unary_union
from scipy.ndimage import gaussian_filter1d
from .bezier import ring
from .skeleton import graph, resample

def area(P):
    return 0.5*np.sum(P[:, 0]*np.roll(P[:, 1], -1)-np.roll(P[:, 0], -1)*P[:, 1])

def part_polygon(outlines):
    """outlines -> polygon (even-odd: rings inside the biggest ring are holes)"""
    rings = [ring(np.asarray(s, float)) for s in outlines]
    rings = [r for r in rings if len(r) >= 3]
    if not rings:
        return Polygon()
    rings.sort(key=lambda r: -abs(area(r)))
    out = []
    used = [False]*len(rings)
    for i, r in enumerate(rings):
        if used[i]:
            continue
        outer = Polygon(r).buffer(0); holes = []
        for j in range(i+1, len(rings)):
            if not used[j] and outer.contains(Point(rings[j][0])):
                holes.append(rings[j]); used[j] = True
        used[i] = True
        out.append(Polygon(r, holes).buffer(0))
    return unary_union(out)

def polys_of(parts, names=None):
    return {p['name']: part_polygon(p['outlines']) for p in parts if names is None or p['name'] in names}

def cut(poly, P, j, maxlen):
    """straight cut across the stroke at centre-line point P[j], perpendicular to it; returns (length, segment)"""
    j = int(min(max(j, 0), len(P)-1))
    d = P[min(j+3, len(P)-1)]-P[max(j-3, 0)]; d = d/(np.linalg.norm(d)+1e-12); n = np.array([-d[1], d[0]]); c = P[j]
    g = LineString([c-n*maxlen, c+n*maxlen]).intersection(poly)
    segs = [g] if g.geom_type == 'LineString' else [s for s in getattr(g, 'geoms', []) if s.geom_type == 'LineString']
    pt = Point(c)
    for s in segs:
        if s.distance(pt) < 1e-6:
            return s.length, np.array(s.coords)
    return 0.0, None

def raster(poly, R):
    x0, y0, x1, y1 = poly.bounds; x0 -= 2; y0 -= 2
    W, H = int((x1-x0+4)*R), int((y1-y0+4)*R); m = np.zeros((H, W), np.uint8)
    geoms = [poly] if poly.geom_type == 'Polygon' else list(getattr(poly, 'geoms', []))
    for q in geoms:
        if q.is_empty:
            continue
        cv2.fillPoly(m, [np.round((np.array(q.exterior.coords)-[x0, y0])*R*16).astype(np.int32)], 1, shift=4)
        for h in q.interiors:
            cv2.fillPoly(m, [np.round((np.array(h.coords)-[x0, y0])*R*16).astype(np.int32)], 0, shift=4)
    return m > 0, (x0, y0)

def recentre(P, poly, U, it=3):
    """move each centre-line point to the middle of the straight cut through it"""
    for _ in range(it):
        M = []
        for j in range(len(P)):
            w, seg = cut(poly, P, j, 3*U)
            M.append(seg.mean(0) if seg is not None and 0 < w < 0.9*U else P[j])
        M = np.array(M)
        P = np.stack([gaussian_filter1d(M[:, 0], 3, mode='nearest'), gaussian_filter1d(M[:, 1], 3, mode='nearest')], 1)
    return P

def centrelines(poly, U, recentred=True, prune=0.02):
    """all centre-lines of a shape, in its own coordinates (full skeleton edges, untrimmed)"""
    R = float(np.clip(450.0/max(U, 1e-6), 6, 40))
    m, (x0, y0) = raster(poly, R)
    sk, nodes, edges = graph(m, U*R, prune=prune)
    out = []
    for e in edges:
        P = resample(e['path'], step=max(R/8, 1.0), sig=4)
        if P is None or len(P) < 4:
            continue
        P = P/R+[x0, y0]
        if recentred:
            P = recentre(P, poly, U)
        ka = nodes[e['a']]['kind'] if e['a'] is not None else 'loop'; kb = nodes[e['b']]['kind'] if e['b'] is not None else 'loop'
        out.append(dict(P=P, ends=[ka, kb], a=e['a'], b=e['b']))
    return out, [dict(x=nd['x']/R+x0, y=nd['y']/R+y0, kind=nd['kind']) for nd in nodes]

def arclen(P):
    return np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]

def stroke_body(P, W, U):
    """a stroke from its centre-line and widths (in U): the two edges, unioned slice by slice, straight end cuts"""
    P = np.asarray(P, float); W = np.asarray(W, float)
    d = np.gradient(P, axis=0); d /= np.linalg.norm(d, axis=1, keepdims=True)+1e-12; n = np.stack([-d[:, 1], d[:, 0]], 1)
    L = P+n*(W[:, None]*U/2); R_ = P-n*(W[:, None]*U/2)
    quads = [Polygon([L[i], L[i+1], R_[i+1], R_[i]]).buffer(0.02*U/23.8) for i in range(len(P)-1)]
    return unary_union(quads).buffer(-0.02*U/23.8)

def smooth_path(P, s=4, step=0.25):
    P = np.asarray(P, float)
    d = arclen(P); t = np.arange(0, d[-1]+1e-9, step)
    Q = np.stack([np.interp(t, d, P[:, 0]), np.interp(t, d, P[:, 1])], 1)
    if s:
        Q = np.stack([gaussian_filter1d(Q[:, 0], s*4, mode='nearest'), gaussian_filter1d(Q[:, 1], s*4, mode='nearest')], 1)
    return Q
