"""Complete measurements of a logo, automatically, on the curves:
strokes (straight runs: angle, length, width all along), bends, outline corners (angle, radius, inner/outer),
free ends (cut angle, width, shape), horizontal levels, counters, letter spacing. Then each kind unified by its mean."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from shapely.geometry import LineString, Point, Polygon
from .geom import cut, arclen, centrelines

def _angle(v):
    """direction in degrees, 0 = horizontal, positive = rising to the right (y up), folded to (-90, 90]"""
    a = np.degrees(np.arctan2(-v[1], v[0]))
    while a <= -90:
        a += 180
    while a > 90:
        a -= 180
    return float(a)

def _kind(a):
    o = abs(a)
    return 'H' if o < 35 else ('V' if o > 55 else 'D')

def _circle(P):
    """least-squares circle through points -> (centre, radius)"""
    x, y = P[:, 0], P[:, 1]; A = np.c_[2*x, 2*y, np.ones(len(P))]; b = x**2+y**2
    c, *_ = np.linalg.lstsq(A, b, rcond=None); r = np.sqrt(max(c[2]+c[0]**2+c[1]**2, 0))
    return c[:2], float(r)

# ---------------------------------------------------------------- strokes
def straight_runs(P, U, max_turn_per_U=60.0, min_len=0.15):
    """split a centre-line into straight runs (direction changes slowly) and bends (it turns)"""
    if len(P) < 8:
        return [], []
    s = arclen(P); step = max(np.median(np.diff(s)), 1e-6)
    d = np.gradient(P, axis=0); ang = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    ang = gaussian_filter1d(ang, max(int(0.05*U/step), 1))
    rate = np.abs(np.gradient(ang))/step*U                                    # radians per unit length
    straight = np.degrees(rate) < max_turn_per_U
    runs, bends = [], []
    i = 0
    while i < len(P):
        j = i
        while j+1 < len(P) and straight[j+1] == straight[i]:
            j += 1
        L = (s[j]-s[i])/U
        if straight[i] and L >= min_len:
            runs.append((i, j))
        elif not straight[i] and j > i:
            bends.append((i, j))
        i = j+1
    return runs, bends

def strokes(polys, cls, U):
    out = []; bends = []
    for part, lst in cls.items():
        poly = polys[part]
        for ei, e in enumerate(lst):
            P = e['P']; runs, bnds = straight_runs(P, U)
            for (i, j) in runs:
                Q = P[i:j+1]; js = np.linspace(0, len(Q)-1, 21).astype(int)
                W = np.array([cut(poly, Q, k, 3*U)[0]/U for k in js])
                med = np.median(W[W > 0]) if (W > 0).any() else 0
                W = np.where((W <= 0) | (W > 1.8*med), np.nan, W)
                if not np.isfinite(W).any():
                    continue
                a = _angle(Q[-1]-Q[0])
                ok = np.where(np.isfinite(W))[0]
                if len(ok) < 11:
                    continue
                w0, w1, wm = W[ok[0]], W[ok[-1]], np.nanmax(W)
                touches_end = (i == 0 and e['ends'][0] == 'end') or (j == len(P)-1 and e['ends'][1] == 'end')
                if touches_end and min(w0, w1) < 0.35*wm and arclen(Q)[-1]/U < 0.4:
                    continue                                            # the centre-line running into the corner of an end cut, not a stroke
                out.append(dict(part=part, edge=ei, P=Q[::3].tolist(), angle=round(a, 1), kind=_kind(a), length=round(float(arclen(Q)[-1]/U), 3),
                                start=round(float(W[ok[0]]), 3), middle=round(float(np.nanmean(W[9:12])), 3), end=round(float(W[ok[-1]]), 3),
                                thinnest=round(float(np.nanmin(W)), 3), thickest=round(float(np.nanmax(W)), 3), thickest_at=int(np.nanargmax(W)*5),
                                W=[None if not np.isfinite(v) else round(float(v), 3) for v in W]))
            for (i, j) in bnds:
                Q = P[max(i-2, 0):j+3]
                if len(Q) < 5:
                    continue
                d0 = Q[min(3, len(Q)-1)]-Q[0]; d1 = Q[-1]-Q[max(-4, -len(Q))]
                turn = abs(np.degrees(np.arctan2(d0[0]*d1[1]-d0[1]*d1[0], np.dot(d0, d1))))
                if turn < 15:
                    continue
                c, r = _circle(Q)
                bends.append(dict(part=part, at=Q[len(Q)//2].tolist(), turn=round(float(turn), 1), radius=round(r/U, 3)))
    return out, bends

# ---------------------------------------------------------------- outline corners
def corners(polys, U, min_turn=25.0, rings_out=None):
    """every place where the outline turns: its angle, the radius of its rounding, convex (outer) or concave (inner)"""
    out = []
    for part, poly in polys.items():
        geoms = [poly] if poly.geom_type == 'Polygon' else list(poly.geoms)
        for q in geoms:
            for ri, ring in enumerate([q.exterior]+list(q.interiors)):
                L = ring.length; n = max(int(L/(0.01*U)), 50)
                P = np.array([ring.interpolate(t*L).coords[0] for t in np.linspace(0, 1, n, endpoint=False)])
                rkey = '%s/%d/%d' % (part, geoms.index(q), ri)
                if rings_out is not None:
                    rings_out[rkey] = P
                d = np.roll(P, -1, 0)-P; a = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
                a = np.r_[a, a[0]+(a[-1]-a[0]) if False else a[:1]+2*np.pi*np.sign(a[-1]-a[0])][:n]
                da = np.angle(np.exp(1j*(np.roll(np.arctan2(d[:, 1], d[:, 0]), -1)-np.arctan2(d[:, 1], d[:, 0]))))
                da_s = da
                k = max(int(0.04*U/(L/n)), 2)
                acc = np.convolve(np.r_[da_s[-k:], da_s, da_s[:k]], np.ones(2*k+1), 'valid')     # turning within +-0.04 U
                area_sign = np.sign(Polygon(P).area) if False else np.sign(0.5*np.sum(P[:, 0]*np.roll(P[:, 1], -1)-np.roll(P[:, 0], -1)*P[:, 1]))
                used = np.zeros(n, bool)
                for i in np.argsort(-np.abs(acc)):
                    if abs(np.degrees(acc[i])) < min_turn or used[i]:
                        continue
                    # grow the corner while the outline keeps turning the same way
                    sgn = np.sign(da_s[i]) if da_s[i] != 0 else np.sign(acc[i]); i0 = i; i1 = i
                    thr = np.radians(0.6)*(L/n)/(0.01*U)
                    while (i0-1) % n != i and np.sign(da_s[(i0-1) % n]) == sgn and abs(da_s[(i0-1) % n]) > thr and not used[(i0-1) % n]:
                        i0 -= 1
                    while (i1+1) % n != i and np.sign(da_s[(i1+1) % n]) == sgn and abs(da_s[(i1+1) % n]) > thr and not used[(i1+1) % n]:
                        i1 += 1
                    idx = [t % n for t in range(i0, i1+1)]
                    turn = float(np.degrees(np.sum(da_s[idx])))
                    if abs(turn) < min_turn:
                        used[idx] = True; continue
                    used[[t % n for t in range(i0-k, i1+k+1)]] = True
                    pts = P[idx]
                    r = _circle(pts)[1] if len(pts) >= 4 else 0.0
                    arc = len(idx)*L/n
                    if r > 0 and arc < 0.03*U:
                        r = 0.0                                                     # a sharp corner, not a rounding
                    convex = (turn*area_sign) > 0
                    out.append(dict(part=part, hole=ri > 0, at=P[i].tolist(), turn=round(abs(turn), 1), radius=round(r/U, 3),
                                    kind='outer' if (convex != (ri > 0)) else 'inner', ring=rkey, i=int(i % n), i0=int(i0 % n), i1=int(i1 % n), n=int(n)))
    return out

# ---------------------------------------------------------------- free ends (end cuts)
def terminals(co, rings, U, width_hint):
    """an end cut = two neighbouring outer corners joined by a straight piece of outline about one stroke wide.
    Measured: the cut's own direction, its angle against the stroke, the stroke's width there, the two corner angles."""
    out = []
    by = {}
    for c in co:
        by.setdefault(c['ring'], []).append(c)
    for rk, cs in by.items():
        P = rings[rk]; n = len(P); cs = sorted(cs, key=lambda c: c['i'])
        for k in range(len(cs)):
            a, b = cs[k], cs[(k+1) % len(cs)]
            if a['kind'] != 'outer' or b['kind'] != 'outer' or a is b:
                continue
            ia, ib = a['i'], b['i']; idx = [(ia+t) % n for t in range(((ib-ia) % n)+1)]
            A, B = P[ia], P[ib]; d = np.linalg.norm(B-A)
            if not (0.6*width_hint*U <= d <= 1.45*width_hint*U) or len(idx) < 3:
                continue
            seg = P[idx[len(idx)//4: 3*len(idx)//4+1]] if len(idx) >= 8 else P[idx]
            ch = (B-A)/d; dev = np.abs((seg-A)@np.array([-ch[1], ch[0]])).max()/U
            if dev > max(0.04, 0.25*width_hint):
                continue
            # the stroke's direction: the outline leaving the two corners away from the cut
            m = max(int(0.25*U/(np.linalg.norm(P[1]-P[0])+1e-9)), 3)
            ua = P[(a['i0']-m) % n]-P[a['i0'] % n]; ub = P[(b['i1']+m) % n]-P[b['i1'] % n]
            ua /= np.linalg.norm(ua)+1e-12; ub /= np.linalg.norm(ub)+1e-12
            if np.dot(ua, ub) < 0.7:                                   # the two sides are not parallel: not one stroke's end
                continue
            sd = (ua+ub); sd /= np.linalg.norm(sd)+1e-12
            perp = np.array([-sd[1], sd[0]])
            width = abs(np.dot(B-A, perp))/U
            rel = float(np.degrees(np.arccos(min(1, abs(np.dot(ch, perp))))))
            out.append(dict(part=rk.split('/')[0], at=((A+B)/2).tolist(), cut_length=round(d/U, 3), width=round(width, 3),
                            cut_direction=round(_angle(ch), 1), stroke_direction=round(_angle(-sd), 1), cut_vs_square=round(rel, 1),
                            corners=[a['turn'], b['turn']]))
    return out

# ---------------------------------------------------------------- levels, counters, spacing
def levels(polys, U, baseline, tol=0.03):
    """the heights where horizontal edges of the outline lie (tops and bottoms of bars), above the baseline"""
    ys = []
    for poly in polys.values():
        for q in ([poly] if poly.geom_type == 'Polygon' else list(poly.geoms)):
            for ring in [q.exterior]+list(q.interiors):
                C = np.array(ring.coords)
                for a, b in zip(C[:-1], C[1:]):
                    v = b-a; L = np.linalg.norm(v)
                    if L > 0 and abs(np.degrees(np.arctan2(v[1], v[0])) % 180) in () or (L > 0 and min(abs(np.degrees(np.arctan2(v[1], v[0]))) % 180, 180-abs(np.degrees(np.arctan2(v[1], v[0]))) % 180) < 6):
                        ys.append(((a[1]+b[1])/2, L))
    if not ys:
        return []
    ys.sort(); groups = []
    for y, L in ys:
        if groups and abs(y-groups[-1]['y'])/U < tol:
            g = groups[-1]; g['y'] = (g['y']*g['len']+y*L)/(g['len']+L); g['len'] += L
        else:
            groups.append(dict(y=y, len=L))
    return [dict(height=round((baseline-g['y'])/U, 3), y=round(g['y'], 2), total_length=round(g['len']/U, 2)) for g in groups if g['len'] >= 0.25*U]

def counters(polys, U):
    out = []
    for part, poly in polys.items():
        for q in ([poly] if poly.geom_type == 'Polygon' else list(poly.geoms)):
            for h in q.interiors:
                hp = Polygon(h); x0, y0, x1, y1 = hp.bounds
                out.append(dict(part=part, width=round((x1-x0)/U, 3), height=round((y1-y0)/U, 3), area=round(hp.area/U**2, 3)))
    return out

def spacing(polys, parts_meta, U):
    rows = {}
    for p in parts_meta:
        rows.setdefault(p.get('row', 0), []).append(p['name'])
    out = []
    for r, names in rows.items():
        names = sorted(names, key=lambda n: -polys[n].bounds[2])
        big = [n for n in names if polys[n].area > 0.15*max(polys[m].area for m in names)]
        for a, b in zip(big[:-1], big[1:]):
            out.append(dict(row=r, between=[a, b], gap=round(polys[a].distance(polys[b])/U, 3)))
    return out

def summarize(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)])
    if not len(v):
        return None
    return dict(mean=round(float(v.mean()), 3), sd=round(float(v.std()), 3), min=round(float(v.min()), 3), max=round(float(v.max()), 3), n=int(len(v)))

def groups(vals, gap=None, rel=0.08, floor=0.012):
    """families of nearly equal values (e.g. small roundings 0.03 and big bowls 0.16): sorted values chained while each step is small"""
    v = np.sort(np.array([x for x in vals if x is not None and np.isfinite(x)]))
    if not len(v):
        return []
    out = [[v[0]]]
    for x in v[1:]:
        tol = gap if gap is not None else max(floor, rel*out[-1][-1])
        if x-out[-1][-1] > tol:
            out.append([x])
        else:
            out[-1].append(x)
    return [summarize(g) for g in out]

def full_measure(polys, parts_meta, U, baseline, cls=None):
    cls = cls or {k: centrelines(p, U, recentred=False)[0] for k, p in polys.items()}
    st, bd = strokes(polys, cls, U)
    rings = {}
    co = corners(polys, U, rings_out=rings)
    mids = [s['middle'] for s in st if np.isfinite(s['middle'])]
    if mids:                                                            # runs inside a junction (much wider than the strokes) are not strokes
        med = float(np.median(mids)); st = [s for s in st if s['middle'] <= 1.6*med]
    body = [s['middle'] for s in st if s['length'] >= 0.3]
    wh = float(np.median(body)) if body else 0.3
    te = terminals(co, rings, U, wh); lv = levels(polys, U, baseline); cn = counters(polys, U); spc = spacing(polys, parts_meta, U)
    unified = dict(
        stroke_width_all=summarize(body),
        stroke_width_H=summarize([s['middle'] for s in st if s['kind'] == 'H' and s['length'] >= 0.3]),
        stroke_width_V=summarize([s['middle'] for s in st if s['kind'] == 'V' and s['length'] >= 0.3]),
        stroke_width_D=summarize([s['middle'] for s in st if s['kind'] == 'D' and s['length'] >= 0.3]),
        change_along_stroke=summarize([(s['thickest']-s['thinnest']) for s in st if s['length'] >= 0.3]),
        angle_V=summarize([s['angle'] if s['angle'] > 0 else s['angle']+180 for s in st if s['kind'] == 'V']),
        angle_H=summarize([s['angle'] for s in st if s['kind'] == 'H']),
        angle_D=summarize([s['angle'] for s in st if s['kind'] == 'D']),
        corner_radius_outer=groups([c['radius'] for c in co if c['kind'] == 'outer' and c['radius'] > 0]),
        corner_radius_inner=groups([c['radius'] for c in co if c['kind'] == 'inner' and c['radius'] > 0]),
        corner_angle=groups([c['turn'] for c in co], gap=12.0),
        sharp_corners=int(sum(1 for c in co if c['radius'] == 0)),
        end_cut_direction=summarize([t['cut_direction'] if t['cut_direction'] > 0 else t['cut_direction']+180 for t in te]),
        end_cut_vs_square=summarize([t['cut_vs_square'] for t in te]),
        end_width=summarize([t['width'] for t in te]),
        bend_turn=summarize([b['turn'] for b in bd]), bend_radius=summarize([b['radius'] for b in bd]),
        letter_gap=summarize([s['gap'] for s in spc]))
    return dict(strokes=st, bends=bd, corners=co, terminals=te, levels=lv, counters=cn, spacing=spc, unified=unified)

def sheet(polys, r, U, baseline, SC=5, pad=10):
    """one picture with everything measured: strokes (angle, width), corners (radius), end cuts (angle), height levels"""
    import cv2
    from shapely.ops import unary_union
    g = unary_union(list(polys.values())); x0, y0, x1, y1 = g.bounds; x0 -= pad; y0 -= pad; x1 += pad; y1 += pad
    W, H = int((x1-x0)*SC), int((y1-y0)*SC); im = np.full((H, W, 3), 255, np.uint8)
    q = lambda p: (int(round((p[0]-x0)*SC)), int(round((p[1]-y0)*SC)))
    for poly in ([g] if g.geom_type == 'Polygon' else list(g.geoms)):
        m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [np.round((np.array(poly.exterior.coords)-[x0, y0])*SC*16).astype(np.int32)], 1, shift=4)
        for h in poly.interiors:
            cv2.fillPoly(m, [np.round((np.array(h.coords)-[x0, y0])*SC*16).astype(np.int32)], 0, shift=4)
        im[m > 0] = (232, 232, 232)
        for ring in [poly.exterior]+list(poly.interiors):
            cv2.polylines(im, [np.round((np.array(ring.coords)-[x0, y0])*SC*16).astype(np.int32)], True, (90, 90, 90), 1, cv2.LINE_AA, shift=4)
    for lv in r['levels']:                                                  # height levels
        y = int(round((lv['y']-y0)*SC)); cv2.line(im, (0, y), (W, y), (120, 200, 120), 1, cv2.LINE_AA)
        cv2.putText(im, '%.2f' % lv['height'], (4, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 140, 40), 1, cv2.LINE_AA)
    col = {'H': (220, 110, 30), 'V': (0, 170, 240), 'D': (170, 60, 170)}
    for s in r['strokes']:                                                  # strokes: centre-line, angle and width
        P = np.array(s['P'])
        cv2.polylines(im, [np.array([q(p) for p in P], np.int32)], False, col[s['kind']], 3, cv2.LINE_AA)
        m = q(P[len(P)//2]); cv2.putText(im, '%s %.0f° w%.2f' % (s['kind'], s['angle'] if s['kind'] != 'V' or s['angle'] > 0 else s['angle']+180, s['middle']),
                                          (m[0]+6, m[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    for c in r['corners']:                                                  # corners: radius
        p = q(c['at']); rr = max(int(c['radius']*U*SC), 3)
        cv2.circle(im, p, rr, (40, 40, 220) if c['kind'] == 'outer' else (200, 40, 160), 2, cv2.LINE_AA)
        cv2.putText(im, 'r%.2f' % c['radius'], (p[0]+4, p[1]+14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 220) if c['kind'] == 'outer' else (200, 40, 160), 1, cv2.LINE_AA)
    for t in r['terminals']:                                                # end cuts: angle against square
        p = q(t['at']); cv2.drawMarker(im, p, (0, 120, 0), cv2.MARKER_DIAMOND, 14, 2)
        cv2.putText(im, 'cut %.0f°' % t['cut_vs_square'], (p[0]+8, p[1]+4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 0), 1, cv2.LINE_AA)
    return im
