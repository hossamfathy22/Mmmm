"""Cubic Bezier helpers: sampling, Schneider fitting with sharp corners, SVG path text."""
import numpy as np

def bez(c, t):
    t = np.asarray(t, float)[:, None]
    return (1-t)**3*c[0] + 3*(1-t)**2*t*c[1] + 3*(1-t)*t**2*c[2] + t**3*c[3]

def _bez1(c, t):
    t = t[:, None]; return 3*(1-t)**2*(c[1]-c[0]) + 6*(1-t)*t*(c[2]-c[1]) + 3*t**2*(c[3]-c[2])

def _bez2(c, t):
    t = t[:, None]; return 6*(1-t)*(c[2]-2*c[1]+c[0]) + 6*t*(c[3]-2*c[2]+c[1])

def ring(segs, n=40):
    """closed outline (list of cubic segments) -> polyline points"""
    return np.vstack([bez(np.asarray(c), np.linspace(0, 1, n, endpoint=False)) for c in segs])

def _unit(v):
    return v/(np.linalg.norm(v)+1e-12)

def _chord(P):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    return d/max(d[-1], 1e-12)

def _gen(P, u, t1, t2):
    A1 = t1[None, :]*(3*(1-u)**2*u)[:, None]; A2 = t2[None, :]*(3*(1-u)*u**2)[:, None]
    C = np.array([[np.sum(A1*A1), np.sum(A1*A2)], [np.sum(A1*A2), np.sum(A2*A2)]])
    tmp = P - bez(np.array([P[0], P[0], P[-1], P[-1]]), u)
    X = np.array([np.sum(A1*tmp), np.sum(A2*tmp)]); det = C[0, 0]*C[1, 1]-C[0, 1]*C[1, 0]
    seg = np.linalg.norm(P[-1]-P[0])
    if abs(det) > 1e-12:
        a1 = (X[0]*C[1, 1]-X[1]*C[0, 1])/det; a2 = (C[0, 0]*X[1]-C[1, 0]*X[0])/det
    else:
        a1 = a2 = seg/3
    if a1 < 1e-6*seg or a2 < 1e-6*seg:
        a1 = a2 = seg/3
    return np.array([P[0], P[0]+t1*a1, P[-1]+t2*a2, P[-1]])

def _reparam(c, P, u):
    d = bez(c, u)-P; n = np.sum(d*_bez1(c, u), 1); den = np.sum(_bez1(c, u)**2, 1)+np.sum(d*_bez2(c, u), 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        u2 = u-np.where(np.abs(den) > 1e-12, n/den, 0)
    return np.clip(u2, 0, 1)

def _fit(P, t1, t2, tol, out, depth=0):
    if len(P) == 2 or depth > 40:
        d = np.linalg.norm(P[-1]-P[0])/3; out.append(np.array([P[0], P[0]+t1*d, P[-1]+t2*d, P[-1]])); return
    u = _chord(P); c = _gen(P, u, t1, t2)
    for _ in range(20):
        e = np.linalg.norm(bez(c, u)-P, axis=1)
        if e.max() <= tol:
            out.append(c); return
        u = _reparam(c, P, u); c = _gen(P, u, t1, t2)
    e = np.linalg.norm(bez(c, u)-P, axis=1)
    if e.max() <= tol:
        out.append(c); return
    k = int(np.clip(np.argmax(e), 1, len(P)-2)); tc = _unit(P[k-1]-P[k+1])
    _fit(P[:k+1], t1, tc, tol, out, depth+1); _fit(P[k:], -tc, t2, tol, out, depth+1)

def fit_closed(P, tol, corner_deg=38.0, k=10):
    """closed polyline (evenly sampled, ~1 unit apart) -> cubic segments; sharp corners only where the outline really turns"""
    n = len(P)
    a = np.roll(P, k, 0)-P; b = np.roll(P, -k, 0)-P
    ang = np.degrees(np.arccos(np.clip(np.sum(a*b, 1)/(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1)+1e-9), -1, 1)))
    turn = 180-ang
    cor = [i for i in range(n) if turn[i] > corner_deg and turn[i] == max(turn[(i+j) % n] for j in range(-k, k+1))]
    cor2 = []
    for i in cor:
        if not cor2 or i-cor2[-1] > k:
            cor2.append(i)
    if len(cor2) > 1 and cor2[0]+n-cor2[-1] <= k:
        cor2.pop()
    segs = []
    if not cor2:
        idx = [0, n//4, n//2, 3*n//4]
        for j in range(4):
            i0, i1 = idx[j], idx[(j+1) % 4]; Q = P[i0:i1+1] if i1 > i0 else np.vstack([P[i0:], P[:i1+1]])
            _fit(Q, _unit(P[(i0+3) % n]-P[(i0-3) % n]), _unit(P[(i1-3) % n]-P[(i1+3) % n]), tol, segs)
    else:
        for j in range(len(cor2)):
            i0 = cor2[j]; i1 = cor2[(j+1) % len(cor2)]
            Q = P[i0:i1+1] if i1 > i0 else np.vstack([P[i0:], P[:i1+1]])
            if len(Q) < 2:
                continue
            _fit(Q, _unit(Q[min(4, len(Q)-1)]-Q[0]), _unit(Q[max(-5, -len(Q))]-Q[-1]), tol, segs)
    return segs, [P[i] for i in cor2]

def resample_closed(P, step=1.0, smooth=1.5):
    from scipy.ndimage import gaussian_filter1d
    Pc = np.vstack([P, P[:1]])
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Pc, axis=0), axis=1))]
    s = np.arange(0, d[-1], step)
    Q = np.stack([np.interp(s, d, Pc[:, 0]), np.interp(s, d, Pc[:, 1])], 1)
    if smooth:
        Q = np.stack([gaussian_filter1d(Q[:, 0], smooth, mode='wrap'), gaussian_filter1d(Q[:, 1], smooth, mode='wrap')], 1)
    return Q

def svg_d(subpaths, scale=1.0, fmt='%.3f'):
    """list of outlines (each a list of 4x2 cubic segments) -> SVG path data"""
    f = lambda p: (fmt+' '+fmt) % (p[0]*scale, p[1]*scale)
    out = []
    for segs in subpaths:
        if not len(segs):
            continue
        out.append(' '.join(['M'+f(segs[0][0])]+['C%s %s %s' % (f(c[1]), f(c[2]), f(c[3])) for c in segs])+'Z')
    return ' '.join(out)
