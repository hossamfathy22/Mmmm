"""أدوات هندسية مشتركة: المقطع العرضي للسُمك، مناطق الالتقاء، الاتجاه، التطبيع."""
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist

def normals(c):
    t = np.gradient(c, axis=0); t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    return np.column_stack([-t[:, 1], t[:, 0]])

def march(mask, p, d, maxlen):
    H, W = mask.shape
    for s in np.arange(0, maxlen, 0.5):
        xi, yi = int(round(p[0] + s * d[0])), int(round(p[1] + s * d[1]))
        if not (0 <= xi < W and 0 <= yi < H) or not mask[yi, xi]:
            return s
    return maxlen

def resample(c, n):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
    u = np.linspace(0, s[-1], n)
    return np.column_stack([np.interp(u, s, c[:, 0]), np.interp(u, s, c[:, 1])])

def refine(mask, c, iters=4, maxlen=200, max_shift=6.0):
    """المقطع العرضي: المركز = منتصف الوتر العمودي، السُمك = طوله."""
    c0 = c.copy()
    for it in range(iters + 1):
        n = normals(c)
        a = np.array([march(mask, p, d, maxlen) for p, d in zip(c, n)])
        b = np.array([march(mask, p, -d, maxlen) for p, d in zip(c, n)])
        if it == iters:
            break
        mid = ndimage.gaussian_filter1d(c + 0.5 * (a - b)[:, None] * n, 4, axis=0, mode="nearest")
        mid[[0, -1]] = c[[0, -1]]
        d = mid - c0; L = np.linalg.norm(d, axis=1, keepdims=True)
        c = resample(c0 + d * np.minimum(1, max_shift / (L + 1e-9)), len(c))
    w = ndimage.median_filter(np.minimum(a + b, 2 * np.minimum(a, b) + 3), 5)
    return c, ndimage.gaussian_filter1d(w, 3)

def junction_zones(raw, pad=4):
    zones = []
    for i, (ci, wi) in enumerate(raw):
        z = np.zeros(len(ci), bool)
        si = np.r_[0, np.cumsum(np.linalg.norm(np.diff(ci, axis=0), axis=1))]
        for j, (cj, wj) in enumerate(raw):
            thr = 0.5 * (np.median(wi) + np.median(wj)) + 2
            D = cdist(ci, cj)
            if i == j:
                D[np.abs(si[:, None] - si[None]) < 3 * np.median(wi)] = np.inf
            z |= D.min(1) < thr
        zones.append(ndimage.binary_dilation(z, iterations=pad))
    return zones

def fill_zones(w, z):
    if z.all() or not z.any():
        return w
    idx = np.arange(len(w))
    return np.where(z, np.interp(idx, idx[~z], w[~z]), w)

def total_turning(c):
    d = np.diff(c, axis=0)
    return float(np.unwrap(np.arctan2(d[:, 1], d[:, 0]))[-1] - np.unwrap(np.arctan2(d[:, 1], d[:, 0]))[0])
