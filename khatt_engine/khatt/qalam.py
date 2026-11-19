"""The pen of a logo: a flat nib of a width, a thickness and an angle, measured from the logo's strokes
(fit_nib), and moved along centre lines to draw (stroke): the union of the hulls of consecutive prints
of the nib, an exact sweep that never folds on itself in a tight turn (the fault that made the hand-made
words give up "centre line + thickness"). One tool for every style: a nib at 0 degrees is the pen of
stems thick and joins thin (سیرنگ: 22.9 x 4.9 at 0.5), thickness near the width a monoline (ماکیتا),
an angle of about 70 degrees the reed of diwani and thuluth (الإسكندرية 68, ساران 70.5, حرية 71).
Coordinates: x right, y UP, baseline y=0."""
import math
import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import splprep, splev
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely import affinity


class Nib:
    """A flat nib, `width` long and `thick` thick, held at `angle_deg`. `power`: how the thickness of a
    stroke goes from thin (along the nib) to full (across it): t + (w - t) * |sin(dir - angle)| ** power,
    measured from the logo with the rest (a reed of fixed shape is about 1; the engine's pen rule, at
    angle 0, about 2). The print of the nib is lengthened or shortened at every point of a stroke so the
    stroke takes exactly that thickness, and stays a sweep: it never folds, and its ends are cut slanted
    as a reed cuts them."""

    def __init__(self, width, thick, angle_deg, power=None):
        self.w, self.t, self.a, self.p = width, thick, math.radians(angle_deg), power

    def thickness(self, direction):
        s = abs(math.sin(direction - self.a))
        if self.p is None:
            return self.w * s + self.t * abs(math.cos(direction - self.a))
        return self.t + (self.w - self.t) * s ** self.p

    def scale_for(self, direction):
        """How long the print must be, as a share of the width, for the stroke to take its thickness."""
        if self.p is None:
            return 1.0
        s, c = abs(math.sin(direction - self.a)), abs(math.cos(direction - self.a))
        if s < 0.05:
            return 1.0                               # along the nib the length does not show
        return max(0.05, min(1.0, (self.thickness(direction) - self.t * c) / (self.w * s)))

    def poly(self, x, y, scale=1.0):
        c, s = math.cos(self.a), math.sin(self.a)
        hw, ht = self.w * scale / 2, self.t / 2
        # long edge along the nib angle
        pts = [(-hw, -ht), (hw, -ht), (hw, ht), (-hw, ht)]
        return [(x + px * c - py * s, y + px * s + py * c) for px, py in pts]


def smooth(points, n=80, s=None, closed=False):
    P = np.asarray(points, float)
    if len(P) < 4:
        return P
    k = 3
    tck, _ = splprep([P[:, 0], P[:, 1]], s=len(P) * 0.6 if s is None else s, k=k, per=closed)
    u = np.linspace(0, 1, n)
    x, y = splev(u, tck)
    return np.c_[x, y]


def stroke(points, nib, taper=(0.0, 0.0), pressure=None):
    """The outline the nib leaves along a centre line: the union of the hulls of consecutive nib
    prints (an exact sweep for a polyline). `taper`: the share of the line at each end over which the
    nib is lifted (narrower), as a pen enters and leaves the paper."""
    P = np.asarray(points, float)
    n = len(P)
    hulls = []
    prev = None
    for i, (x, y) in enumerate(P):
        f = i / max(1, n - 1)
        sc = 1.0
        if taper[0] and f < taper[0]:
            sc = 0.35 + 0.65 * (f / taper[0])
        if taper[1] and f > 1 - taper[1]:
            sc = min(sc, 0.35 + 0.65 * ((1 - f) / taper[1]))
        if pressure is not None:
            sc *= pressure(f)
        if nib.p is not None:
            a_, b_ = P[max(0, i - 1)], P[min(n - 1, i + 1)]
            sc *= nib.scale_for(math.atan2(b_[1] - a_[1], b_[0] - a_[0]))
        q = nib.poly(x, y, sc)
        if prev is not None:
            hulls.append(Polygon(prev + q).convex_hull)
        prev = q
    return unary_union(hulls).buffer(0)


def dot(x, y, nib, k=1.0):
    """The print of the nib turned once: a rhombus, as a reed leaves a dot."""
    a = nib.a
    d = nib.w * 0.62 * k
    return stroke([(x - d * math.cos(a + math.pi / 2) / 2, y - d * math.sin(a + math.pi / 2) / 2),
                   (x + d * math.cos(a + math.pi / 2) / 2, y + d * math.sin(a + math.pi / 2) / 2)], nib)


def centre_line(mask, Z=1, closed=False):
    """The centre line of a stroke drawn in a binary mask (rows down): the longest path through its
    skeleton, from end to end (or round its cycle), in mask units / Z, y up."""
    from skimage.morphology import skeletonize
    sk = skeletonize(mask)
    ys, xs = np.nonzero(sk)
    idx = {(y, x): i for i, (y, x) in enumerate(zip(ys, xs))}
    nb = [[] for _ in ys]
    for i, (y, x) in enumerate(zip(ys, xs)):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dy or dx) and (y + dy, x + dx) in idx:
                    nb[i].append(idx[(y + dy, x + dx)])
    import heapq

    def far(src):
        dist = {src: 0.0}; prev = {src: None}; h = [(0.0, src)]
        while h:
            d, u = heapq.heappop(h)
            if d > dist[u]:
                continue
            for v in nb[u]:
                w = math.hypot(ys[u] - ys[v], xs[u] - xs[v])
                if d + w < dist.get(v, 1e18):
                    dist[v] = d + w; prev[v] = u; heapq.heappush(h, (d + w, v))
        end = max(dist, key=dist.get)
        return end, prev
    a, _ = far(0)
    b, prev = far(a)
    path = []
    u = b
    while u is not None:
        path.append((xs[u] / Z, -ys[u] / Z)); u = prev[u]
    return np.array(path[::-1])


def to_svg_path(geom, flip=True):
    d = ""
    for g in getattr(geom, "geoms", [geom]):
        if g.geom_type != "Polygon" or g.is_empty:
            continue
        for ring in [g.exterior] + list(g.interiors):
            cs = list(ring.coords)
            d += "M" + " L".join(f"{x:.2f},{(-y if flip else y):.2f}" for x, y in cs) + " Z "
    return d


def fit_nib(mask, Z=1):
    """Width, thickness and angle of the flat nib that drew the strokes in a binary mask (rows down):
    the thickness of the strokes against their direction."""
    from skimage.morphology import medial_axis
    from scipy.spatial import cKDTree
    from scipy.optimize import least_squares
    # medial_axis breaks ties at random: a fixed seed gives the same nib for the same picture every time
    try:
        sk, dist = medial_axis(mask, return_distance=True, rng=0)
    except TypeError:                                    # older scikit-image
        sk, dist = medial_axis(mask, return_distance=True, random_state=0)
    ys, xs = np.nonzero(sk); pts = np.c_[xs, ys].astype(float); tree = cKDTree(pts)
    deg = ndi.convolve(sk.astype(int), np.ones((3, 3)), mode="constant")[ys, xs] - 1
    phi, w = [], []
    for k in range(0, len(pts), 2):
        if deg[k] != 2: continue
        idx = tree.query_ball_point(pts[k], 3 * Z)
        if len(idx) < 6: continue
        q = pts[idx] - pts[idx].mean(0); ev, vec = np.linalg.eigh(q.T @ q)
        if ev[0] / ev[1] > 0.05: continue
        v = vec[:, 1]; phi.append(math.atan2(-v[1], v[0]) % math.pi); w.append(2 * dist[ys[k], xs[k]] / Z)
    phi, w = np.array(phi), np.array(w)
    def res(p):
        W, h, th, pw = p; return h + (W - h) * np.abs(np.sin(phi - th)) ** pw - w
    best = None
    for th0 in np.linspace(0, math.pi, 12, endpoint=False):
        r = least_squares(res, [np.percentile(w, 90), np.percentile(w, 10), th0, 1.5], loss="soft_l1", f_scale=1.0,
                          bounds=([1, 0.3, -1, 0.4], [200, 100, 4.5, 6.0]))
        if best is None or r.cost < best.cost: best = r
    W, h, th, pw = best.x
    a = math.degrees(th) % 180
    if h > W:                                        # the fit found the nib turned a quarter: the same nib
        W, h, a = h, W, (a + 90) % 180
    return {"width": float(W), "thick": float(h), "angle": float(a), "power": float(pw),
            "error": float(np.median(np.abs(best.fun))), "samples": int(len(w))}


def ink_mask(path, light_ink, box=None, Z=6, thr=0.5):
    """The ink of a picture, smoothed and Z times finer (bicubic from the grey levels), rows down."""
    from PIL import Image
    g = np.asarray(Image.open(path).convert("L")).astype(float) / 255.0
    a = g if light_ink else 1 - g
    lo, hi = np.percentile(a, 5), np.percentile(a, 99.5)
    a = np.clip((a - lo) / max(1e-6, hi - lo), 0, 1)
    if box:
        x0, y0, x1, y1 = box; a = a[y0:y1, x0:x1]
    return ndi.zoom(a, Z, order=3) > thr


def ring_line(mask, Z=1):
    """The centre line round a closed counter: where the distance to the counter equals the distance
    to the outside, as one closed curve (mask units / Z, y up)."""
    from skimage.measure import find_contours
    filled = ndi.binary_fill_holes(mask)
    hole = filled & ~mask
    lab, n = ndi.label(hole)
    if not n:
        return None
    sizes = ndi.sum(hole, lab, range(1, n + 1)); hole = lab == (1 + int(np.argmax(sizes)))
    d_in = ndi.distance_transform_edt(~hole); d_out = ndi.distance_transform_edt(filled)
    f = np.where(filled, d_in - d_out, d_in)
    cs = find_contours(f, 0.0)
    c = max(cs, key=len)
    return np.array([(col / Z, -row / Z) for row, col in c])
