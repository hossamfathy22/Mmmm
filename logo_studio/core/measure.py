"""Steps 3-5: measure at the designer's marks, read whole strokes between marks, unify by kind of place, test the numbers."""
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from .geom import cut, arclen, stroke_body, recentre

CATS = {
    'h_mid':      dict(ar='وسط خط عرضي (الأتخن)', color='#1f6fe0', family='H'),
    'h_end_hair': dict(ar='طرف عرضي عند شعيرة / ملتقى حرفين', color='#58a6ff', family='H'),
    'h_end_stem': dict(ar='طرف عرضي عند جذع تخين', color='#0b3d91', family='H'),
    'head_top':   dict(ar='أعلى الرأس', color='#d62828', family='V'),
    'v_top':      dict(ar='أعلى عمودي (قمة مقطوعة)', color='#f4b400', family='V'),
    'v_mid':      dict(ar='وسط عمودي', color='#e09100', family='V'),
    'v_bot':      dict(ar='أسفل عمودي', color='#ffd75e', family='V'),
    'joint':      dict(ar='وصلة', color='#a347c2', family='J'),
}
HAIR = ('h_end_hair', 'v_bot', 'joint')

def mark_centre(m):
    return (np.asarray(m['p0'], float)+np.asarray(m['p1'], float))/2

def measure_mark(m, polys, cls, U):
    """the point where the designer's line crosses a stroke's centre-line; the width there straight across the stroke"""
    a = np.asarray(m['p0'], float); b = np.asarray(m['p1'], float); c = (a+b)/2
    ax = (b-a)/(np.linalg.norm(b-a)+1e-12); half = max(np.linalg.norm(b-a)*0.75, 0.12*U)
    line = LineString([c-ax*half, c+ax*half]); best = None
    for part, lst in cls.items():
        for ei, e in enumerate(lst):
            P = e['P']
            X = line.intersection(LineString(P))
            pts = [X] if X.geom_type == 'Point' else [g for g in getattr(X, 'geoms', []) if g.geom_type == 'Point']
            for pt in pts:
                p = np.array(pt.coords[0]); j = int(np.argmin(np.linalg.norm(P-p, axis=1)))
                t = P[min(j+2, len(P)-1)]-P[max(j-2, 0)]; t /= np.linalg.norm(t)+1e-12
                ang = np.degrees(np.arccos(min(1, abs(np.dot(t, ax)))))
                if ang < 45:
                    continue
                d = np.linalg.norm(p-c)
                sc = d/U+(90-ang)/90                       # the stroke his line crosses closest to its middle and most squarely
                if best is None or sc < best[0]:
                    best = (sc, part, ei, j)
    if best is not None:
        _, part, ei, j = best; P = cls[part][ei]['P']
        w, seg = cut(polys[part], P, j, 3*U)
        return dict(width=w/U, part=part, edge=ei, j=j, at=P[j].tolist(), seg=None if seg is None else seg.tolist(), how='across the stroke')
    part = min(polys, key=lambda k: polys[k].distance(Point(c)))
    L = LineString([c-ax*3*U, c+ax*3*U]).intersection(polys[part])
    segs = [L] if L.geom_type == 'LineString' else [g for g in getattr(L, 'geoms', []) if g.geom_type == 'LineString']
    if not segs:
        return dict(width=float('nan'), part=part, edge=None, j=None, at=c.tolist(), seg=None, how='not on the logo')
    sg = min(segs, key=lambda g: g.distance(Point(c)))
    return dict(width=sg.length/U, part=part, edge=None, j=None, at=c.tolist(), seg=np.array(sg.coords).tolist(), how='along your line')

def profile(poly, P, U, n=21):
    js = np.linspace(0, len(P)-1, n).astype(int)
    return np.array([cut(poly, P, j, 3*U)[0]/U for j in js])

def _proj(cls, part, p):
    """nearest point of each centre-line of a part to p: list of (distance, edge index, point index)"""
    out = []
    for ei, e in enumerate(cls[part]):
        d = np.linalg.norm(e['P']-p, axis=1); j = int(d.argmin()); out.append((float(d[j]), ei, j))
    return out

def _span_pts(cls, part, pts, U):
    """the stretch of one centre-line that passes all the given points (first -> last); None if no centre-line passes them all"""
    best = None
    for ei, e in enumerate(cls[part]):
        P = e['P']; idx = []; tot = 0; far = False
        for p in pts:
            d = np.linalg.norm(P-p, axis=1); j = int(d.argmin()); idx.append(j); tot += d[j]; far |= d[j] > 0.3*U
        if far or abs(idx[-1]-idx[0]) < 4:
            continue
        if len(idx) == 3 and not (min(idx[0], idx[2]) < idx[1] < max(idx[0], idx[2])):
            continue
        if best is None or tot < best[0]:
            best = (tot, ei, idx)
    if best is None:
        return None
    _, ei, idx = best; P = cls[part][ei]['P']; i0, i1 = idx[0], idx[-1]
    return P[i0:i1+1] if i1 > i0 else P[i1:i0+1][::-1]

def spans(marks, res, polys, cls, U):
    """whole strokes between marks: horizontals (end - middle - end), verticals (top - bottom), joints (vertical bottom -> horizontal hairline end)"""
    out = []
    byid = {m['id']: m for m in marks}
    pt = lambda k: np.asarray(res[k]['at'], float)
    ids = lambda cats: [m['id'] for m in marks if m['cat'] in cats and np.isfinite(res[m['id']]['width'])]
    ends = ids(('h_end_hair', 'h_end_stem'))
    # horizontals: each middle mark with the end marks around it on one centre-line (closest pair that brackets it)
    for mid in ids(('h_mid',)):
        part = res[mid]['part']; best = None
        cand = [e for e in ends if res[e]['part'] == part]
        for i, a in enumerate(cand):
            for b in cand[i+1:]:
                P = _span_pts(cls, part, [pt(a), pt(mid), pt(b)], U)
                if P is None:
                    continue
                L = arclen(P)[-1]
                if best is None or L < best[0]:
                    best = (L, a, b, P)
        if best is None:
            continue
        _, a, b, P = best
        if P[0][0] < P[-1][0]:                                         # writing order: right -> left
            P = P[::-1]
        if np.linalg.norm(P[0]-pt(a)) > np.linalg.norm(P[0]-pt(b)):
            a, b = b, a
        W = profile(polys[part], P, U)
        s_ = arclen(P)/max(arclen(P)[-1], 1e-9); js = np.linspace(0, len(P)-1, 21).astype(int)
        out.append(dict(kind='H', start=a, middle=mid, end=b, P=P.tolist(), W=W.tolist(), length=float(arclen(P)[-1]/U),
                        thick=float(np.nanmax(W)), thick_at=float(s_[js[int(np.nanargmax(W))]]), lowest_at=float(s_[int(np.argmax(P[:, 1]))]),
                        start_cat=byid[a]['cat'], end_cat=byid[b]['cat']))
    # verticals: a top mark and the nearest bottom mark on one centre-line
    for top in ids(('v_top', 'head_top')):
        part = res[top]['part']; best = None
        for bt in ids(('v_bot',)):
            if res[bt]['part'] != part:
                continue
            P = _span_pts(cls, part, [pt(top), pt(bt)], U)
            if P is None:
                continue
            L = arclen(P)[-1]
            if L < 1.5*U and (best is None or L < best[0]):
                best = (L, bt, P)
        if best:
            _, bt, P = best
            if P[0][1] > P[-1][1]:
                P = P[::-1]
            W = profile(polys[part], P, U)
            out.append(dict(kind='V', start=top, end=bt, P=P.tolist(), W=W.tolist(), length=float(arclen(P)[-1]/U), middle_w=float(W[10]),
                            top_cat=byid[top]['cat']))
    # joints: a vertical's bottom to the nearest horizontal hairline end on one centre-line
    for bt in ids(('v_bot',)):
        part = res[bt]['part']; best = None
        for e in ids(('h_end_hair',)):
            if res[e]['part'] != part:
                continue
            P = _span_pts(cls, part, [pt(bt), pt(e)], U)
            if P is None:
                continue
            L = arclen(P)[-1]
            if L < 0.6*U and (best is None or L < best[0]):
                best = (L, e, P)
        if best:
            _, e, P = best
            W = profile(polys[part], P, U)
            out.append(dict(kind='J', start=bt, end=e, P=P.tolist(), W=W.tolist(), length=float(arclen(P)[-1]/U)))
    return out

def unify(marks, res, sp, U, hairline_one=True, overrides=None):
    """same kind of place -> one number (mean); verticals' middles and joints are added from the whole-stroke readings"""
    vals = {k: [] for k in CATS}
    for m in marks:
        w = res[m['id']]['width']
        if m['cat'] in vals and np.isfinite(w):
            vals[m['cat']].append(dict(v=float(w), src='mark %s' % m['id']))
    for s in sp:
        if s['kind'] == 'V':
            vals['v_mid'].append(dict(v=s['middle_w'], src='middle of vertical %s-%s' % (s['start'], s['end'])))
        if s['kind'] == 'J':
            W = s['W']
            for k, lab in ((0, 'start'), (10, 'middle'), (20, 'end')):
                if np.isfinite(W[k]) and W[k] > 0:
                    vals['joint'].append(dict(v=float(W[k]), src='joint %s-%s %s' % (s['start'], s['end'], lab)))
    table = {}
    for k, lst in vals.items():
        v = np.array([x['v'] for x in lst])
        if len(v):
            m_ = float(v.mean()); far = max(lst, key=lambda x: abs(x['v']-m_))
            table[k] = dict(value=round(m_, 3), n=len(v), min=round(float(v.min()), 3), max=round(float(v.max()), 3),
                            farthest=far['src'], farthest_gap=round(far['v']-m_, 3), items=lst)
    if hairline_one:
        allh = [x for k in HAIR for x in vals[k]]
        if allh:
            v = np.array([x['v'] for x in allh]); table['hair'] = dict(value=round(float(v.mean()), 3), n=len(v), min=round(float(v.min()), 3),
                                                                      max=round(float(v.max()), 3), items=allh)
    Hs = [s for s in sp if s['kind'] == 'H']
    rule = [dict(start=s['start'], end=s['end'], thick_at=round(s['thick_at'], 2), lowest_at=round(s['lowest_at'], 2)) for s in Hs]
    agree = [abs(r['thick_at']-r['lowest_at']) <= 0.1 for r in rule]
    spec = dict(U=U, pixel=round(1.0/U, 4),
                H_MID=table.get('h_mid', {}).get('value'), H_HAIR=(table.get('hair') or table.get('h_end_hair', {})).get('value'),
                H_STEM=table.get('h_end_stem', {}).get('value'), HEAD_TOP=table.get('head_top', {}).get('value'),
                V_TOP=table.get('v_top', {}).get('value'), V_MID=table.get('v_mid', {}).get('value'),
                V_BOT=(table.get('hair') or table.get('v_bot', {})).get('value'), JOINT=(table.get('hair') or table.get('joint', {})).get('value'),
                THICK_POS=round(float(np.mean([s['thick_at'] for s in Hs])), 3) if Hs else 0.57,
                lowest_point_rule=dict(holds=int(sum(agree)), of=len(agree), strokes=rule))
    if overrides:
        spec.update({k: v for k, v in overrides.items() if v is not None})
    return table, spec

# ---------- the shape of a horizontal between start, thickest and end (normalised logo profiles)
def h_shape(sp, exclude=None):
    rs = lambda a, n: np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(a)), a)
    R, F = [], []
    for s in sp:
        if s['kind'] != 'H' or s is exclude:
            continue
        W = np.array(s['W'], float)
        if not np.all(np.isfinite(W)):
            ok = np.isfinite(W); W = np.interp(np.arange(len(W)), np.where(ok)[0], W[ok])
        k = int(np.argmax(W)); T = W[k]
        if k < 2 or k > len(W)-3 or T <= max(W[0], W[-1]):
            continue
        R.append(rs((W[:k+1]-W[0])/(T-W[0]), 50)); F.append(rs((W[k:]-W[-1])/(T-W[-1]), 50))
    if not R:                                                                          # no logo horizontals: a smooth default
        t = np.linspace(0, 1, 50); return np.sin(t*np.pi/2), np.cos(t*np.pi/2)
    return np.mean(R, axis=0), np.mean(F, axis=0)

def h_widths(P, spec, start, end, rise, fall, thick_at=None):
    """widths along a horizontal: start -> thickest (at the lowest point) -> end"""
    P = np.asarray(P, float); u = arclen(P)/max(arclen(P)[-1], 1e-9)
    if thick_at is None:
        low = float(u[int(np.argmax(P[:, 1]))])
        thick_at = low if 0.2 <= low <= 0.8 else spec['THICK_POS']
    T = spec['H_MID']; g = np.linspace(0, 1, len(rise))
    W = np.where(u <= thick_at, start+(T-start)*np.interp(u/thick_at, g, rise), end+(T-end)*np.interp((u-thick_at)/(1-thick_at), g, fall))
    return W, thick_at

def v_widths(P, top, mid, bot):
    u = arclen(np.asarray(P, float)); u = u/max(u[-1], 1e-9)
    return np.interp(u, [0, 0.5, 1], [top, mid, bot])

def gate(sp, spec, polys, res, marks, U):
    """rebuild every measured stroke from the numbers alone, on its own centre-line; compare with the real one"""
    byid = {m['id']: m for m in marks}; out = []
    for s in sp:
        if s['kind'] not in ('H', 'V'):
            continue
        P = np.asarray(s['P'], float); part = res[s.get('middle', s['start'])]['part']; real = polys[part]
        if s['kind'] == 'H':
            rise, fall = h_shape(sp, exclude=s)
            st = spec['H_STEM'] if s['start_cat'] == 'h_end_stem' else spec['H_HAIR']
            en = spec['H_STEM'] if s['end_cat'] == 'h_end_stem' else spec['H_HAIR']
            W, at = h_widths(P, spec, st, en, rise, fall)
        else:
            top = spec['HEAD_TOP'] if s['top_cat'] == 'head_top' else spec['V_TOP']
            W = v_widths(P, top, spec['V_MID'], spec['V_BOT'])
        Pc = recentre(P, real, U)                                        # draw around the true middle of the stroke
        mine = stroke_body(Pc, W, U); P = Pc
        band = LineString(P).buffer(0.6*U, cap_style=2)
        r = real.intersection(band); m_ = mine.intersection(band)
        iou = r.intersection(m_).area/max(r.union(m_).area, 1e-9)
        Wr = np.array(s['W']); js = np.linspace(0, len(P)-1, 21).astype(int)
        js = np.linspace(0, len(P)-1, 21).astype(int); js[0] = min(2, len(P)-1); js[-1] = max(len(P)-3, 0)
        Wm = np.array([cut(mine, P, j, 3*U)[0]/U for j in js]); Wm = np.where(Wm > 1.8*np.nanmedian(Wm), np.nan, Wm)
        out.append(dict(kind=s['kind'], start=s['start'], end=s['end'], iou=round(float(iou), 3),
                        real=[round(float(Wr[0]), 3), round(float(np.nanmax(Wr)), 3), round(float(Wr[-1]), 3)],
                        rebuilt=[round(float(np.nan_to_num(Wm[0])), 3), round(float(np.nanmax(Wm)), 3), round(float(np.nan_to_num(Wm[-1])), 3)]))
    allr = [g['iou'] for g in out]
    return dict(strokes=out, mean_iou=round(float(np.mean(allr)), 3) if allr else None)
