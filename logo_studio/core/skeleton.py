"""Skeleton graph of a mask: nodes (ends / junctions) and edges (pixel paths); centre-line resampling."""
import numpy as np, cv2
from skimage.morphology import skeletonize
from scipy.ndimage import gaussian_filter1d
N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

def trace(sk):
    deg = (cv2.filter2D(sk.astype(np.uint8), -1, np.ones((3, 3), np.uint8))-1)*sk; H, W = sk.shape
    nodepix = (deg == 1) | (deg >= 3); n, lab = cv2.connectedComponents(nodepix.astype(np.uint8), connectivity=8)
    nodes = []
    for i in range(1, n):
        ys, xs = np.where(lab == i); d = deg[ys, xs]
        nodes.append(dict(y=float(ys.mean()), x=float(xs.mean()), kind='end' if (len(ys) == 1 and d[0] == 1) else 'junction',
                          pix=list(zip(ys.tolist(), xs.tolist()))))
    vis = np.zeros_like(sk, bool); edges = []
    for ni, nd in enumerate(nodes):
        for (y, x) in nd['pix']:
            for dy, dx in N8:
                yy, xx = y+dy, x+dx
                if not (0 <= yy < H and 0 <= xx < W) or not sk[yy, xx] or lab[yy, xx] or vis[yy, xx]:
                    continue
                path = [(y, x), (yy, xx)]; vis[yy, xx] = True; cy, cx = yy, xx; end = None
                while True:
                    nxt = None
                    for ddy, ddx in N8:
                        a, b = cy+ddy, cx+ddx
                        if not (0 <= a < H and 0 <= b < W) or not sk[a, b] or (a, b) == path[-2]:
                            continue
                        if lab[a, b] and ((a, b) != path[0] or len(path) > 3):
                            end = lab[a, b]-1; path.append((a, b)); break
                        if not vis[a, b] and not lab[a, b]:
                            nxt = (a, b)
                    if end is not None or nxt is None:
                        break
                    vis[nxt] = True; path.append(nxt); cy, cx = nxt
                edges.append(dict(a=ni, b=end, path=path))
    # closed loops without any node (e.g. an 'o'): trace them as one edge
    rest = sk & ~vis & (lab == 0)
    while rest.any():
        ys, xs = np.where(rest); y, x = int(ys[0]), int(xs[0]); path = [(y, x)]; rest[y, x] = False
        while True:
            nxt = None
            for dy, dx in N8:
                a, b = path[-1][0]+dy, path[-1][1]+dx
                if 0 <= a < H and 0 <= b < W and rest[a, b]:
                    nxt = (a, b); break
            if nxt is None:
                break
            rest[nxt] = False; path.append(nxt)
        if len(path) > 5:
            edges.append(dict(a=None, b=None, path=path))
    return nodes, edges

def graph(mask, U, prune=0.08):
    """skeleton graph, short spurs (< prune*U) removed"""
    sk = skeletonize(mask)
    for _ in range(8):
        nodes, edges = trace(sk); rm = False
        for e in edges:
            if e['a'] is None or e['b'] is None:
                continue
            ka, kb = nodes[e['a']]['kind'], nodes[e['b']]['kind']
            if len(e['path']) < prune*U and ((ka == 'end') != (kb == 'end')):
                en = e['a'] if ka == 'end' else e['b']
                for (y, x) in e['path'][1:-1]:
                    sk[y, x] = False
                for (y, x) in nodes[en]['pix']:
                    sk[y, x] = False
                rm = True
        if not rm:
            break
        sk = skeletonize(sk)
    nodes, edges = trace(sk)
    return sk, nodes, edges

def resample(path, step=2.0, sig=6):
    P = np.array([(x, y) for y, x in path], float)
    if len(P) < 5:
        return P
    P = np.stack([gaussian_filter1d(P[:, 0], sig, mode='nearest'), gaussian_filter1d(P[:, 1], sig, mode='nearest')], 1)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]; ss = np.arange(0, s[-1]+1e-6, step)
    return np.stack([np.interp(ss, s, P[:, 0]), np.interp(ss, s, P[:, 1])], 1)
