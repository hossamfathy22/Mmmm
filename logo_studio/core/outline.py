"""Measuring on the outline, and drawing back from the numbers.

Measuring: every outline is split, by curvature, into straight edges and the corners between them, so a corner is
found whatever its rounding radius. An S-shaped stretch between two edges (in, then out) is two corners, split at the
inflection. A corner's rounding radius comes from its tip: how far the outline stays from the point where its two
edges meet (r = gap / (1/sin(half the opening) - 1)), which stays right for narrow notches where a fitted circle fails.
Stroke width = the distance between two opposite parallel edges with ink between them.
Families: edge angles, widths, and corners (by kind, then by turn, then by radius), each split where neighbouring
values are further apart than a gap.

Drawing back: every edge becomes a line (its angle, its position), neighbouring lines are intersected, and every corner
is rounded with its radius. Done in levels (as measured, then each kind unified) and compared with the logo, so a
wrong measurement shows up as a place where the drawing leaves the logo."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from shapely.geometry import Polygon, Point, LinearRing
from shapely.ops import unary_union

LEVELS = ['edges as measured, sharp corners', '+ each corner rounded with its own radius', '+ edge angles unified to their family',
          '+ stroke widths unified to their family', '+ corner radii unified to their family']

def _circle(P):
    x, y = P[:, 0], P[:, 1]; A = np.c_[2*x, 2*y, np.ones(len(P))]; b = x**2+y**2
    c, *_ = np.linalg.lstsq(A, b, rcond=None); return c[:2], float(np.sqrt(max(c[2]+c[0]**2+c[1]**2, 0)))

def _ang(u):
    """0 = horizontal, positive = rising to the right (y down in the picture), folded to (-45, 135]"""
    a = np.degrees(np.arctan2(-u[1], u[0])) % 180
    return float(a-180 if a > 135 else a)

def _adiff(a, b):
    return abs(((a-b)+90) % 180-90)

def _nrm(u):
    return np.array([-u[1], u[0]])

def _inter(c1, u1, c2, u2):
    M = np.array([u1, -u2]).T
    if abs(np.linalg.det(M)) < 1e-6:
        return None
    t = np.linalg.solve(M, c2-c1); return c1+u1*t[0]

def clusters(vals, weights=None, gap=1.0):
    """1-D families: sorted, a new family starts wherever two neighbours are more than gap apart"""
    if not len(vals):
        return []
    weights = np.ones(len(vals)) if weights is None else np.asarray(weights, float)
    o = np.argsort(vals); v = np.asarray(vals, float)[o]; w = np.maximum(weights[o], 1e-9); out = []; cur = [0]
    for t in range(1, len(v)+1):
        if t == len(v) or v[t]-v[t-1] > gap:
            vv, ww = v[cur], w[cur]
            out.append(dict(mean=round(float(np.average(vv, weights=ww)), 3), sd=round(float(vv.std()), 3),
                            min=round(float(vv.min()), 3), max=round(float(vv.max()), 3), n=len(cur), weight=round(float(ww.sum()), 3)))
            cur = [t]
        else:
            cur.append(t)
    tot = sum(f['weight'] for f in out)
    for f in out:
        f['share'] = round(f['weight']/tot, 3) if tot else 0
    return sorted(out, key=lambda f: -f['weight'])

# ---------------------------------------------------------------- measuring
def _rings(polys, U, step):
    for part, poly in polys.items():
        geoms = [poly] if poly.geom_type == 'Polygon' else list(poly.geoms)
        for gi, q in enumerate(geoms):
            for ri, ring in enumerate([q.exterior]+list(q.interiors)):
                L = ring.length; n = max(int(L/(step*U)), 60)
                P = np.array([ring.interpolate(t*L).coords[0] for t in np.linspace(0, 1, n, endpoint=False)])
                yield part, gi, ri, '%s/%d/%d' % (part, gi, ri), P

def measure(polys, U, step=0.01, win=0.03, straight_r=1.0, min_edge=0.05, min_turn=8.0,
            angle_gap=3.0, width_gap=0.012, turn_gap=12.0, radius_gap=0.03):
    """straight edges, corners, widths and their families for the given parts (shapely polygons, picture units)"""
    edges, corners, rings = [], [], {}
    for part, gi, ri, rk, P in _rings(polys, U, step):
        n = len(P); ds = np.mean(np.linalg.norm(np.diff(np.r_[P, P[:1]], axis=0), axis=1)); k = max(int(win*U/ds), 2)
        d = np.roll(P, -k, 0)-np.roll(P, k, 0); th = np.arctan2(d[:, 1], d[:, 0])
        kap = np.angle(np.exp(1j*(np.roll(th, -1)-th)))/ds
        straight = np.abs(kap)*straight_r*U < 1.0
        rings[rk] = dict(P=P, part=part, g=gi, hole=ri > 0, edges=[], corners=[])
        if straight.all() or not straight.any():
            continue                                                    # a ring with no straight edge (a round dot): kept as it is
        st = int(np.argmin(straight)); idx = np.roll(np.arange(n), -st); sv = straight[idx]
        runs = []; a = 0
        while a < n:
            b = a
            while b+1 < n and sv[b+1] == sv[a]:
                b += 1
            if sv[a] and (b-a+1)*ds >= min_edge*U:
                runs.append((int(idx[a]), int(idx[b])))
            a = b+1
        if len(runs) < 2:
            continue
        span = lambda a, b: [t % n for t in range(a, b+1 if b >= a else b+n+1)]
        ext = []                                                         # S-shaped stretches: split at the inflection
        for (a0, b0), (a1, b1) in zip(runs, runs[1:]+runs[:1]):
            ext.append((a0, b0)); ci = span(b0, a1)
            kz = gaussian_filter1d(kap[ci], max(k//2, 1), mode='nearest'); sg = np.sign(kz)
            pieces = []; cur = sg[0]; s0 = 0
            for t in range(1, len(ci)+1):
                if t == len(ci) or (sg[t] != cur and sg[t] != 0):
                    pieces.append((s0, t-1, float(np.degrees(np.sum(kz[s0:t])*ds)))); s0 = t
                    cur = sg[t] if t < len(ci) else cur
            big = [q for q in pieces if abs(q[2]) >= min_turn]
            for q0, q1 in zip(big, big[1:]):
                if np.sign(q0[2]) != np.sign(q1[2]):
                    m = ci[(q0[1]+q1[0])//2]; ext.append((m, m))
        runs = ext
        area = 0.5*np.sum(P[:, 0]*np.roll(P[:, 1], -1)-np.roll(P[:, 0], -1)*P[:, 1])
        # edges: a line fitted to the middle 80 % of each straight run (a zero-length run takes the outline's tangent)
        E = []
        for (a, b) in runs:
            if a == b:
                u = P[(a+2) % n]-P[a-2]; u = u/np.linalg.norm(u); c = P[a]; L = 0.0; dev = 0.0
            else:
                ii = span(a, b); m = len(ii); core = ii[int(0.1*m):max(int(0.9*m), int(0.1*m)+2)]
                Q = P[core]; c = Q.mean(0); u = np.linalg.svd(Q-c)[2][0]
                if np.dot(P[ii[-1]]-P[ii[0]], u) < 0:
                    u = -u
                L = float(np.linalg.norm(P[ii[-1]]-P[ii[0]])); dev = float(np.abs((Q-c)@_nrm(u)).max())
            E.append(dict(ring=rk, part=part, i0=a, i1=b, c=c, u=u, angle=_ang(u), length=L/U, straight_dev=dev/U))
        ring_line = LinearRing(P)
        for e0, e1 in zip(E, E[1:]+E[:1]):                               # the corner after each edge
            u0, u1 = e0['u'], e1['u']
            turn = float(np.degrees(np.arctan2(u0[0]*u1[1]-u0[1]*u1[0], u0@u1)))
            ci = span(e0['i1'], e1['i0']); at = P[ci[len(ci)//2]]
            c = dict(part=part, ring=rk, at=at, turn=round(abs(turn), 1), hole=ri > 0)
            if abs(turn) < min_turn:
                c.update(kind='flat', radius=0.0)
            else:
                convex = (turn*np.sign(area)) > 0
                c['kind'] = 'outer' if (convex != (ri > 0)) else 'inner'
                arc = max(len(ci)-1-2*k, 0)*ds
                inner = P[ci[k:len(ci)-k]] if len(ci) > 2*k+3 else None
                r = (_circle(inner)[1] if (inner is not None and arc >= 0.02*U) else (arc/np.radians(abs(turn)) if arc > 0 else 0.0))/U
                c['radius_circle'] = round(r, 3)
                V = _inter(e0['c'], u0, e1['c'], u1)
                if abs(turn) >= 20 and V is not None and np.linalg.norm(V-at) < 0.8*U:
                    h = ring_line.distance(Point(*V)); half = np.radians(180-abs(turn))/2
                    r = h/(1/np.sin(half)-1)/U; c['tip_gap'] = round(h/U, 3)
                c['radius'] = round(float(r), 4) if (r >= 0.008 or c.get('tip_gap', 0) >= 0.01) else 0.0   # a narrow notch: a tiny radius still moves its tip a lot
            corners.append(c)
        for e in E:
            edges.append(e)
        rings[rk]['edges'] = list(range(len(edges)-len(E), len(edges)))
        rings[rk]['corners'] = list(range(len(corners)-len(E), len(corners)))
    # widths: opposite parallel edges, facing each other, with ink between
    ink = unary_union(list(polys.values())).buffer(0); pairs = []
    pts = {}
    for i, e in enumerate(edges):
        P = rings[e['ring']]['P']; n = len(P)
        pts[i] = P[[t % n for t in range(e['i0'], e['i1']+1 if e['i1'] >= e['i0'] else e['i1']+n+1)]]
    for i, s in enumerate(edges):
        if s['length'] < 0.03:
            continue
        best = None
        for j, t in enumerate(edges):
            if j == i or t['length'] < 0.03 or _adiff(s['angle'], t['angle']) > 4:
                continue
            d = (t['c']-s['c'])@_nrm(s['u'])
            if abs(d) < 0.05*U or abs(d) > 0.6*U:
                continue
            ps, pt = pts[i]@s['u'], pts[j]@s['u']; ov = min(ps.max(), pt.max())-max(ps.min(), pt.min())
            if ov < 0.3*min(np.ptp(ps), np.ptp(pt)) or not ink.contains(Point(*((s['c']+t['c'])/2))):
                continue
            if best is None or abs(d) < abs(best[1]):
                best = (j, d)
        if best and (best[0], i) not in [(p['a'], p['b']) for p in pairs]:
            pairs.append(dict(a=i, b=best[0], width=round(abs(best[1])/U, 4), length=round(min(s['length'], edges[best[0]]['length']), 3),
                              angle=round(s['angle'], 1), at=((s['c']+edges[best[0]]['c'])/2).tolist(), part=s['part']))
    angle_f = clusters([e['angle'] for e in edges if e['length'] > 0], [e['length'] for e in edges if e['length'] > 0], gap=angle_gap)
    width_f = clusters([p['width'] for p in pairs], [p['length'] for p in pairs], gap=width_gap)
    corner_f = []
    for kind in ('outer', 'inner'):
        cc = [c for c in corners if c['kind'] == kind and c['radius'] > 0]
        for tf in clusters([c['turn'] for c in cc], gap=turn_gap):
            mem = [c for c in cc if tf['min'] <= c['turn'] <= tf['max']]
            corner_f.append(dict(kind=kind, turn=tf, radii=clusters([c['radius'] for c in mem], gap=radius_gap)))
        sharp = [c for c in corners if c['kind'] == kind and c['radius'] == 0]
        if sharp:
            corner_f.append(dict(kind=kind, turn=clusters([c['turn'] for c in sharp], gap=turn_gap)[0], radii=[dict(mean=0.0, sd=0.0, min=0.0, max=0.0, n=len(sharp))], sharp=True))
    tot = sum(e['length'] for e in edges); on_edges = tot*U/max(sum(len(r['P']) for r in rings.values())*step*U, 1e-9)
    for e in edges:
        e['c'] = e['c'].tolist(); e['u'] = e['u'].tolist(); e['angle'] = round(e['angle'], 2); e['length'] = round(e['length'], 3); e['straight_dev'] = round(e['straight_dev'], 4)
    for c in corners:
        c['at'] = np.asarray(c['at']).tolist()
    return dict(edges=edges, corners=corners, pairs=pairs, rings={k: dict(part=v['part'], g=v['g'], hole=v['hole'], edges=v['edges'], corners=v['corners']) for k, v in rings.items()},
                families=dict(edge_angles=angle_f, widths=width_f, corners=corner_f), straight_share=round(float(min(on_edges, 1.0)), 3))

# ---------------------------------------------------------------- drawing back
def _fillet(V, R, steps=10):
    out = []; n = len(V)
    for k in range(n):
        A, B, C = V[k-1], V[k], V[(k+1) % n]
        a, b = A-B, C-B; la, lb = np.linalg.norm(a), np.linalg.norm(b)
        if R[k] <= 0 or la < 1e-6 or lb < 1e-6:
            out.append(B); continue
        a, b = a/la, b/lb; th = np.arccos(np.clip(a@b, -1, 1))
        if th < 1e-3 or th > np.pi-1e-3:
            out.append(B); continue
        d = min(R[k]/np.tan(th/2), 0.48*la, 0.48*lb); rr = d*np.tan(th/2)
        bis = (a+b)/np.linalg.norm(a+b); O = B+bis*rr/np.sin(th/2)
        p0, p1 = B+a*d, B+b*d; s0 = np.arctan2(*(p0-O)[::-1]); s1 = np.arctan2(*(p1-O)[::-1]); dd = np.angle(np.exp(1j*(s1-s0)))
        for t in np.linspace(0, 1, steps):
            out.append(O+rr*np.array([np.cos(s0+dd*t), np.sin(s0+dd*t)]))
    return np.array(out)

def _snap_angle(a, fams, tol=4.0):
    ms = [f['mean'] for f in fams if f['n'] >= 2]
    if not ms:
        return a
    m = min(ms, key=lambda m: _adiff(a, m)); return m if _adiff(a, m) < tol else a

def _snap_width(w, fams, tol=0.03):
    ms = [f['mean'] for f in fams if f['n'] >= 2]
    if not ms:
        return w
    m = min(ms, key=lambda m: abs(m-w)); return m if abs(m-w) < tol else w

def _snap_radius(c, fams, turn_tol=12.0):
    """the radius of the corner family of this kind and turn, the radius family nearest to the corner's own"""
    if c['kind'] == 'flat':
        return 0.0
    gs = [g for g in fams if g['kind'] == c['kind'] and (g['turn']['min']-turn_tol/2 <= c['turn'] <= g['turn']['max']+turn_tol/2)]
    if not gs:
        return c['radius']
    g = min(gs, key=lambda g: abs(g['turn']['mean']-c['turn']) + (0 if (g.get('sharp') == (c['radius'] == 0)) else 1000))
    return min((r['mean'] for r in g['radii']), key=lambda m: abs(m-c['radius']))

def redraw(polys, m, U, level=5, families=None):
    """draw the parts back from the measurements m; level 1..5 as in LEVELS; families: use another logo's families
    (to draw a new letter with the logo's numbers), default the parts' own"""
    fam = families or m['families']; E = m['edges']; C = m['corners']
    L = [dict(c=np.array(e['c'], float), u=np.array(e['u'], float)) for e in E]
    if level >= 3:
        for l, e in zip(L, E):
            a = _snap_angle(e['angle'], fam['edge_angles']); r = np.radians(a); u = np.array([np.cos(r), -np.sin(r)])
            l['u'] = u if u@np.array(e['u']) >= 0 else -u
    if level >= 4:
        done = set()
        for p in sorted(m['pairs'], key=lambda p: -p['length']):
            i, j = p['a'], p['b']
            if i in done or j in done:
                continue
            W = _snap_width(p['width'], fam['widths'])*U
            n_ = _nrm(L[i]['u']); d = (L[j]['c']-L[i]['c'])@n_; sh = d/2-np.sign(d)*W/2
            L[i]['c'] = L[i]['c']+n_*sh; L[j]['c'] = L[j]['c']-n_*sh; done |= {i, j}
    shapes = {}
    for part, poly in polys.items():
        geoms = [poly] if poly.geom_type == 'Polygon' else list(poly.geoms)
        out = []
        for gi, q in enumerate(geoms):
            rs = []
            for ri, ring in enumerate([q.exterior]+list(q.interiors)):
                rk = '%s/%d/%d' % (part, gi, ri); info = m['rings'].get(rk)
                if not info or len(info['edges']) < 3:                      # no straight edges to draw from: the ring as it is
                    rs.append(Polygon(ring.coords)); continue
                ks = info['edges']; cs = info['corners']; V = []; R = []
                for t, ka in enumerate(ks):                              # vertex = the corner after edge ka
                    kb = ks[(t+1) % len(ks)]; cc = C[cs[t]]
                    v = _inter(L[ka]['c'], L[ka]['u'], L[kb]['c'], L[kb]['u'])
                    if v is None or np.linalg.norm(v-np.array(cc['at'])) > 0.4*U:
                        v = np.array(cc['at'], float)
                    V.append(v)
                    R.append(0.0 if level < 2 else (cc['radius'] if level < 5 else _snap_radius(cc, fam['corners']))*U)
                V = np.array(V); P = _fillet(V, np.array(R)) if level >= 2 else V
                g = Polygon(P).buffer(0) if len(P) >= 3 else Polygon()
                rs.append(g if not g.is_empty else Polygon(ring.coords))
            ext = rs[0]; holes = rs[1:]
            out.append(ext.difference(unary_union(holes)) if holes else ext)
        shapes[part] = unary_union(out)
    return shapes

def check(polys, m, U, levels=(1, 2, 3, 4, 5), samples=3000):
    """draw back at each level and compare with the parts: overlap (IoU), the worst distance between the outlines and where"""
    orig = unary_union(list(polys.values())); out = []
    for lv in levels:
        g = unary_union(list(redraw(polys, m, U, lv).values()))
        iou = g.intersection(orig).area/max(g.union(orig).area, 1e-9); worst = (0.0, [0, 0])
        for A, B in ((g.boundary, orig.boundary), (orig.boundary, g.boundary)):
            for t in np.linspace(0, 1, samples):
                q = A.interpolate(t, normalized=True); d = B.distance(q)
                if d > worst[0]:
                    worst = (d, [q.x, q.y])
        out.append(dict(level=lv, name=LEVELS[lv-1], iou=round(float(iou), 4), max_off=round(worst[0]/U, 3), worst_at=worst[1],
                        missing=round(orig.difference(g).area/U/U, 4), extra=round(g.difference(orig).area/U/U, 4)))
    return out

def check_sheet(polys, m, U, SC=4, pad=8):
    """the five levels, each with the logo: grey = both, red = in the logo but not drawn, blue = drawn but not in the logo"""
    import cv2
    from .build import render_png
    orig = unary_union(list(polys.values())); x0, y0, x1, y1 = orig.bounds; box = (x0-pad, y0-pad, x1+pad, y1+pad)
    tiles = []
    for lv in range(1, 6):
        g = unary_union(list(redraw(polys, m, U, lv).values()))
        t = render_png([(orig.intersection(g), (205, 205, 205)), (orig.difference(g), (40, 40, 220)), (g.difference(orig), (220, 120, 30))], box, scale=SC)
        for q in getattr(g, 'geoms', [g]):
            if q.geom_type != 'Polygon':
                continue
            for ring in [q.exterior]+list(q.interiors):
                cv2.polylines(t, [((np.array(ring.coords)-[box[0], box[1]])*SC).astype(np.int32)], True, (20, 20, 20), 1, cv2.LINE_AA)
        iou = g.intersection(orig).area/max(g.union(orig).area, 1e-9)
        cv2.putText(t, '%d %s   IoU %.3f' % (lv, LEVELS[lv-1], iou), (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 160), 2, cv2.LINE_AA)
        tiles.append(np.pad(t, ((0, 10), (0, 0), (0, 0)), constant_values=255))
    return np.vstack(tiles)

def apply(polys, logo_m, U, tol_angle=4.0, tol_width=0.03, turn_tol=12.0):
    """draw new parts (a drafted letter or word) with the logo's numbers: measured on their own outline, then drawn
    back with the logo's families. Returns the new shapes and what in the draft has no family in the logo
    (edges at an angle the logo never uses, widths it never uses, corners of a kind/turn it never has): those are the
    designer's to decide."""
    m = measure(polys, U); F = logo_m['families']
    shapes = redraw(polys, m, U, 5, families=F)
    angs = [f['mean'] for f in F['edge_angles']]                        # anything the logo has, even once, is the logo's
    wids = [f['mean'] for f in F['widths']]
    odd = dict(edges=[], widths=[], corners=[])
    for e in m['edges']:
        if e['length'] >= 0.1 and all(_adiff(e['angle'], a) >= tol_angle for a in angs):
            odd['edges'].append(dict(angle=e['angle'], length=e['length'], at=e['c'], part=e['part']))
    for p in m['pairs']:
        if all(abs(p['width']-w) >= tol_width for w in wids):
            odd['widths'].append(dict(width=p['width'], at=p['at'], part=p['part']))
    for c in m['corners']:
        if c['kind'] == 'flat':
            continue
        if not any(g['kind'] == c['kind'] and g['turn']['min']-turn_tol/2 <= c['turn'] <= g['turn']['max']+turn_tol/2 for g in F['corners']):
            odd['corners'].append(dict(kind=c['kind'], turn=c['turn'], radius=c['radius'], at=c['at'], part=c['part']))
    return shapes, m, odd
