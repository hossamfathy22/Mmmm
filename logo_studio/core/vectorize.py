"""Step 1: any logo picture -> Bezier outlines, one part per connected shape, checked against the picture."""
import numpy as np, cv2
from skimage.measure import find_contours
from scipy.spatial import cKDTree
from .bezier import fit_closed, resample_closed, bez, svg_d

def ink_map(img):
    """how far each pixel is from the background colour (0..1); background = median colour of the border"""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:                          # transparent PNG: alpha is the ink
        a = img[:, :, 3].astype(np.float32)/255
        if a.min() < 0.9:
            return a
        img = img[:, :, :3]
    lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.vstack([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    bg = np.median(border, axis=0)
    d = np.linalg.norm(lab-bg, axis=2)
    d8 = np.clip(d/max(d.max(), 1e-6)*255, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ink = np.median(d[d8 > t]) if (d8 > t).any() else d.max()
    return np.clip(d/max(ink, 1e-6), 0, 1.5)

def vectorize(img, S=8, tol_px=0.15, min_area_px=3.0, margin=6, box=None):
    """returns dict(parts=[{name,outlines,bbox}], crop=(x0,y0), size=(W,H), report, ink=upsampled mask, S)"""
    m = ink_map(img)
    if box:                                        # the designer's crop: x0, y0, x1, y1 in picture pixels
        bx0, by0, bx1, by1 = [int(round(v)) for v in box]
        mm = np.zeros_like(m); mm[by0:by1, bx0:bx1] = m[by0:by1, bx0:bx1]; m = mm
    ys, xs = np.where(m > 0.5)
    if not len(ys):
        raise ValueError('no ink found in the picture')
    y0, y1 = max(ys.min()-margin, 0), min(ys.max()+margin+1, m.shape[0]); x0, x1 = max(xs.min()-margin, 0), min(xs.max()+margin+1, m.shape[1])
    g = m[y0:y1, x0:x1].astype(np.float32)
    up = cv2.GaussianBlur(cv2.resize(g, None, fx=S, fy=S, interpolation=cv2.INTER_CUBIC), (0, 0), 3*S/8)
    THR = 0.5                                      # the edge = half way between background and ink
    up = np.pad(up, 2, constant_values=0)
    ink = up[2:-2, 2:-2] > THR
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8))
    keep = [i for i in range(1, n) if st[i, 4] >= min_area_px*S*S]
    cs = [c for c in find_contours(up, THR) if len(c) >= 5*S]
    outlines = []; corners = []
    for c in cs:
        P = c[:, ::-1]-2.0
        if np.allclose(P[0], P[-1]):
            P = P[:-1]
        P = resample_closed(P, 1.0, 1.5)
        segs, cor = fit_closed(P, tol_px*S)
        if segs:
            outlines.append(segs); corners += cor
    groups = {}
    for segs in outlines:
        p = segs[0][0]; x, y = int(p[0]), int(p[1]); w = lab[max(y-4, 0):y+5, max(x-4, 0):x+5]; v = w[np.isin(w, keep)]
        if not len(v):
            continue
        groups.setdefault(int(np.bincount(v).argmax()), []).append(segs)
    # order: rows from top to bottom (big shapes that overlap vertically share a row; small marks join the nearest row),
    # right to left inside a row (Arabic reading order)
    ks = list(groups)
    hs = {k: st[k, 3] for k in ks}; hmed = float(np.median(list(hs.values()))) if ks else 1.0
    big = sorted([k for k in ks if hs[k] >= 0.6*hmed], key=lambda k: st[k, 1])
    rows_rng = []
    row_of = {}
    for k in big:
        y0k, y1k = st[k, 1], st[k, 1]+st[k, 3]; placed = False
        for ri, (a0, a1) in enumerate(rows_rng):
            ov = min(a1, y1k)-max(a0, y0k)
            if ov > 0.3*min(a1-a0, y1k-y0k):
                rows_rng[ri] = (min(a0, y0k), max(a1, y1k)); row_of[k] = ri; placed = True; break
        if not placed:
            rows_rng.append((y0k, y1k)); row_of[k] = len(rows_rng)-1
    for k in ks:
        if k in row_of:
            continue
        cy = st[k, 1]+st[k, 3]/2
        row_of[k] = int(np.argmin([0 if a0 <= cy <= a1 else min(abs(cy-a0), abs(cy-a1)) for a0, a1 in rows_rng])) if rows_rng else 0
    rank = {ri: i for i, ri in enumerate(sorted(range(len(rows_rng)), key=lambda r: rows_rng[r][0]))}
    row_of = {k: rank.get(v, 0) for k, v in row_of.items()}
    order = sorted(ks, key=lambda k: (row_of[k], -(st[k, 0]+st[k, 2]/2)))
    parts = []
    for i, k in enumerate(order):
        segs_list = [[(np.asarray(c)/S).tolist() for c in segs] for segs in groups[k]]
        x, y, w, h, _ = st[k]
        parts.append(dict(name='part_%02d' % (i+1), outlines=segs_list, bbox=[float(x/S), float(y/S), float(w/S), float(h/S)], row=int(row_of[k])))
    # ---- check the curves against the picture
    H, W = ink.shape
    acc = np.zeros((H, W), np.uint8)
    for segs in outlines:
        Q = np.vstack([bez(np.asarray(c), np.linspace(0, 1, 24)) for c in segs])
        mm = np.zeros_like(acc); cv2.fillPoly(mm, [np.round(Q*16).astype(np.int32)], 1, shift=4); acc ^= mm
    r8 = acc > 0
    iou = float((r8 & ink).sum()/max((r8 | ink).sum(), 1))
    E = np.vstack([c[:, ::-1]-2.0 for c in cs]) if cs else np.zeros((1, 2))
    smp = np.vstack([bez(np.asarray(c), np.linspace(0, 1, 30)) for segs in outlines for c in segs]) if outlines else np.zeros((1, 2))
    de = cKDTree(E).query(smp)[0]/S
    r1 = cv2.resize(r8.astype(np.float32), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_AREA)
    gg = np.clip(g, 0, 1); sel = (r1 > 0.02) | (gg > 0.02)
    report = dict(parts=len(parts), outlines=len(outlines), curve_segments=int(sum(len(s) for s in outlines)), sharp_corners=len(corners),
                  match_vs_picture=round(iou, 4), edge_error_mean_px=round(float(de.mean()), 3), edge_error_max_px=round(float(de.max()), 3),
                  grey_error=round(float(np.abs(r1-gg)[sel].mean()), 3) if sel.any() else 0.0, source_size=[int(m.shape[1]), int(m.shape[0])])
    return dict(parts=parts, crop=[int(x0), int(y0)], size=[int(x1-x0), int(y1-y0)], report=report)

def parts_svg(parts, size, fill='#000', bg='#fff', scale=4):
    W, H = size
    body = '\n'.join('  <path id="%s" d="%s"/>' % (p['name'], svg_d(p['outlines'])) for p in parts)
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %g %g" width="%g" height="%g">\n <rect width="100%%" height="100%%" fill="%s"/>\n'
            ' <g fill="%s" fill-rule="evenodd">\n%s\n </g>\n</svg>\n') % (W, H, W*scale, H*scale, bg, fill, body)
