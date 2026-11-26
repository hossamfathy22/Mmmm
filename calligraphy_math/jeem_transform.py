"""
تحويل حرف الجيم (نسخ) إلى الجيم الحديثة.

1) الهيكل والسُمك من الصورة بالمقطع العرضي:  w(t) = |وتر الحبر على العمودي n(t)| ، γ(t) = منتصفه
2) تشويه الهيكل بـ Thin-Plate Spline:
      f(x) = a0 + A·x + Σ_i c_i · U(|x − p_i|) ,   U(r) = r² log r
3) الشكل:  S = ⋃_t  segment( γ'(t) ,  w'(t) ,  normal n(t) )
4) الاستيفاء:  γ_α = (1−α)γ + α f(γ) ,   w_α = (1−α)w + α w'
"""
import sys
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.interpolate import splprep, splev
from shapely.geometry import Polygon
from shapely.ops import unary_union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

UP = sys.argv[1] if len(sys.argv) > 1 else "."
SRC_IMG = f"{UP}/83a4afbb-image.jpg"
TGT_IMG = f"{UP}/f0b4c74c-image.jpg"

# ---------- الأقنعة ----------
def masks(path):
    a = np.asarray(Image.open(path).convert("RGB")).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    ink = (r + g + b) / 3 < 110                       # الحبر الأسود
    teal = (g - r > 40) & (b - r > 30)                # النقطة الملونة
    ink = ndimage.binary_opening(ink, iterations=1)
    return ink, teal

# ---------- هياكل تقريبية (إحداثيات بكسل) — تُصحَّح آلياً بالـ skeleton ----------
SRC = {
    "head": [(198, 272), (220, 230), (250, 217), (350, 226), (460, 238)],
    "bowl": [(450, 255), (370, 275), (280, 330), (225, 410), (210, 480), (235, 540),
             (300, 580), (400, 582), (480, 570), (545, 547)],
}
TGT = {
    "head": [(238, 232), (255, 290), (300, 308), (380, 310), (435, 318), (455, 360), (460, 420)],
    "bowl": [(455, 422), (350, 390), (260, 365), (220, 365), (200, 390), (195, 418), (205, 470),
             (255, 515), (310, 545), (390, 547), (460, 518), (520, 503), (575, 505), (618, 540)],
}
N = {"head": 120, "bowl": 260}

def smooth_curve(pts, n, s=0):
    pts = np.asarray(pts, float)
    k = min(3, len(pts) - 1)
    tck, _ = splprep(pts.T, s=s, k=k)
    return np.column_stack(splev(np.linspace(0, 1, n), tck))

def normals(c):
    t = np.gradient(c, axis=0); t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    return np.column_stack([-t[:, 1], t[:, 0]])

def march(mask, p, d, maxlen):
    """المسافة من p في الاتجاه d حتى الخروج من الحبر."""
    H, W = mask.shape
    for s in np.arange(0, maxlen, 0.5):
        x, y = p + s * d
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < W and 0 <= yi < H) or not mask[yi, xi]:
            return s
    return maxlen

def resample_arc(c, n):
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
    u = np.linspace(0, seg[-1], n)
    return np.column_stack([np.interp(u, seg, c[:, 0]), np.interp(u, seg, c[:, 1])])

def refine(mask, c, iters=4, maxlen=90, max_shift=6.0):
    """مقطع عرضي (cross-section): المركز = منتصف الوتر العمودي، السُمك = طول الوتر."""
    c0 = c.copy()
    for it in range(iters + 1):
        n = normals(c)
        a = np.array([march(mask, p, d, maxlen) for p, d in zip(c, n)])
        b = np.array([march(mask, p, -d, maxlen) for p, d in zip(c, n)])
        if it == iters:
            break
        mid = c + 0.5 * (a - b)[:, None] * n
        mid = ndimage.gaussian_filter1d(mid, 5, axis=0, mode="nearest")
        mid[[0, -1]] = c[[0, -1]]
        d = mid - c0                                   # لا نبتعد كثيراً عن الدليل (تقاطع الضربات)
        L = np.linalg.norm(d, axis=1, keepdims=True)
        mid = c0 + d * np.minimum(1, max_shift / (L + 1e-9))
        c = resample_arc(mid, len(c))
    w = np.minimum(a + b, 2 * np.minimum(a, b) + 4)   # تجاهل الحبر المجاور من جهة واحدة
    w = ndimage.median_filter(w, 5)
    return c, ndimage.gaussian_filter1d(w, 4)

def extract(img_path, guide):
    ink, teal = masks(img_path)
    strokes = {}
    for k, g in guide.items():
        c, w = refine(ink, smooth_curve(g, N[k]))
        u = np.linspace(0, 1, len(w))
        w *= np.clip(np.minimum(u, 1 - u) / 0.03, 0, 1) ** 0.5   # أطراف مدببة
        strokes[k] = (c, w)
    return ink, teal, strokes

def dot_of(mask):
    ys, xs = np.nonzero(mask)
    c = np.array([xs.mean(), ys.mean()])
    half = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2
    return c, half

# ---------- Thin-Plate Spline ----------
def tps_fit(P, Q, lam=1.0):
    n = len(P)
    r = np.linalg.norm(P[:, None] - P[None], axis=2)
    K = np.where(r > 0, r**2 * np.log(r + 1e-12), 0) + lam * np.eye(n)
    Pm = np.hstack([np.ones((n, 1)), P])
    L = np.zeros((n + 3, n + 3)); L[:n, :n] = K; L[:n, n:] = Pm; L[n:, :n] = Pm.T
    Y = np.vstack([Q, np.zeros((3, 2))])
    sol = np.linalg.solve(L, Y)
    return P, sol[:n], sol[n:]

def tps_apply(model, X):
    P, C, A = model
    r = np.linalg.norm(X[:, None] - P[None], axis=2)
    U = np.where(r > 0, r**2 * np.log(r + 1e-12), 0)
    return A[0] + X @ A[1:] + U @ C

# ---------- الرسم: γ ± (w/2)·n ----------
def stroke_poly(c, w):
    t = np.gradient(c, axis=0); t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    n = np.column_stack([-t[:, 1], t[:, 0]])
    L, R = c + 0.5 * w[:, None] * n, c - 0.5 * w[:, None] * n
    quads = [Polygon([L[i], R[i], R[i + 1], L[i + 1]]).buffer(0) for i in range(len(c) - 1)]
    return unary_union([q for q in quads if q.area > 1e-6])

def diamond(c, h):
    x, y = c
    return Polygon([(x, y - h), (x + h, y), (x, y + h), (x - h, y)])

def raster(geom, shape):
    from matplotlib.path import Path
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    m = np.zeros(len(pts), bool)
    for g in getattr(geom, "geoms", [geom]):
        m |= Path(np.asarray(g.exterior.coords)).contains_points(pts)
    return m.reshape(shape)

def iou(a, b):
    return (a & b).sum() / (a | b).sum()

def pen_fit(c, w):
    """أفضل زاوية قلم ثابتة θ تفسّر السُمك:  w ≈ W|sin(φ−θ)|"""
    t = np.gradient(c, axis=0)
    phi = np.degrees(np.arctan2(-t[:, 1], t[:, 0]))     # y للأعلى
    best = None
    for th in np.arange(0, 180, 1.0):
        s = np.abs(np.sin(np.radians(phi - th)))
        W = (s @ w) / (s @ s + 1e-9)
        err = ((w - W * s) ** 2).sum()
        if best is None or err < best[0]:
            best = (err, th, W)
    r2 = 1 - best[0] / ((w - w.mean()) ** 2).sum()
    return best[1], best[2], r2

# ---------- التشغيل ----------
if __name__ == "__main__":
    out = sys.argv[2] if len(sys.argv) > 2 else "jeem_transform.png"
    ink_s, _, S = extract(SRC_IMG, SRC)
    ink_t, teal_t, T = extract(TGT_IMG, TGT)
    shape = ink_t.shape

    # نقاط الربط (landmarks) بالتناسب على طول كل ضربة
    P, Q = [], []
    for k in S:
        idx_s = np.linspace(0, len(S[k][0]) - 1, 24).astype(int)
        idx_t = np.linspace(0, len(T[k][0]) - 1, 24).astype(int)
        P.append(S[k][0][idx_s]); Q.append(T[k][0][idx_t])
    dot_s_c, dot_s_h = dot_of(ink_s & (ndimage.label(ink_s)[0] == np.bincount(ndimage.label(ink_s)[0].ravel())[1:].argsort()[0] + 1))
    dot_t_c, dot_t_h = dot_of(teal_t & ndimage.binary_dilation(ndimage.binary_opening(teal_t, iterations=4), iterations=5))
    P.append(dot_s_c[None]); Q.append(dot_t_c[None])       # النقطة نفسها نقطة ربط
    model = tps_fit(np.vstack(P), np.vstack(Q), lam=2.0)

    def frame(a):
        parts = []
        for k in S:
            c0, w0 = S[k]
            c1 = tps_apply(model, c0)
            w1 = T[k][1]
            parts.append(stroke_poly((1 - a) * c0 + a * c1, (1 - a) * w0 + a * w1))
        dc = (1 - a) * dot_s_c + a * tps_apply(model, dot_s_c[None])[0]
        dh = (1 - a) * dot_s_h + a * dot_t_h
        return unary_union(parts), diamond(dc, dh)

    res, dot_res = frame(1.0)
    rec_src, _ = frame(0.0)
    score_t = iou(raster(res, shape), ink_t)
    score_s = iou(raster(rec_src, ink_s.shape), ink_s)
    print(f"IoU source reconstruction = {score_s:.3f}")
    print(f"IoU result vs target      = {score_t:.3f}")
    for name, D in [("source", S), ("target", T)]:
        c = np.vstack([D[k][0] for k in D]); w = np.concatenate([D[k][1] for k in D])
        th, W, r2 = pen_fit(c, w)
        print(f"pen fit {name}: θ={th:.0f}°  W={W:.0f}px  R²={r2:.2f}")

    BG, INK, DOT = "#fafaf7", "#111", "#5a9ea3"
    def draw(ax, g, d, title, flip=True):
        for p in getattr(g, "geoms", [g]):
            ax.fill(*p.exterior.xy, color=INK, lw=0)
        ax.fill(*d.exterior.xy, color=DOT, lw=0)
        ax.set_xlim(150, 660); ax.set_ylim(650, 160); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=10)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7.8), facecolor=BG)
    axes[0, 0].imshow(Image.open(SRC_IMG).convert("RGB")); axes[0, 0].set_xlim(150, 660); axes[0, 0].set_ylim(650, 160)
    axes[0, 0].axis("off"); axes[0, 0].set_title("input (Naskh)", fontsize=10)
    for i, a in enumerate([0.25, 0.5, 0.75]):
        g, d = frame(a); draw(axes[0, i + 1], g, d, f"α = {a}")
    draw(axes[1, 0], res, dot_res, f"result α = 1   (IoU {score_t:.2f})")
    axes[1, 1].imshow(Image.open(TGT_IMG).convert("RGB")); axes[1, 1].set_xlim(150, 660); axes[1, 1].set_ylim(650, 160)
    axes[1, 1].axis("off"); axes[1, 1].set_title("target", fontsize=10)
    # تراكب
    ax = axes[1, 2]
    ov = np.ones(shape + (3,))
    rm = raster(res, shape)
    ov[ink_t & ~rm] = (0.85, 0.3, 0.3); ov[rm & ~ink_t] = (0.3, 0.45, 0.9); ov[rm & ink_t] = (0.15, 0.15, 0.15)
    ax.imshow(ov); ax.set_xlim(150, 660); ax.set_ylim(650, 160); ax.axis("off")
    ax.set_title("overlay  (red = missed, blue = extra)", fontsize=10)
    # الهيكل والتشويه
    ax = axes[1, 3]
    for k in S:
        ax.plot(*S[k][0].T, color="#bbb", lw=2)
        ax.plot(*tps_apply(model, S[k][0]).T, color="#c0392b", lw=2)
    Pm, Qm = model[0][:-1], np.vstack(Q)[:-1]
    for p, q in zip(Pm, Qm):
        ax.annotate("", q, p, arrowprops=dict(arrowstyle="->", color="#5a9ea3", lw=0.8))
    ax.set_xlim(150, 660); ax.set_ylim(650, 160); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("skeleton γ (grey) → f(γ) (red)", fontsize=10)
    fig.savefig(out, dpi=120, facecolor=BG, bbox_inches="tight")
    print("saved", out)
