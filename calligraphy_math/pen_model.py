"""
نموذج القلم العريض (Broad-nib pen model) لتحويل حرف النسخ إلى الأسلوب الحديث.

الشكل = اتحاد (Minkowski sum) مسار الهيكل γ(t) مع مقطع القلم:
    S = ⋃_t { γ(t) + s·W(t)·(cosθ, sinθ) : s ∈ [-1/2, 1/2] }

التحويل  Naskh → Modern:
    γ'(t) = A · Π_ε( γ(t) )        (تبسيط المنحنى ثم تحويل أفيني)
    θ'    = 18°   بدلاً من  θ = 70°
    W'    = k · W
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.ops import unary_union

# ---------- هندسة أساسية ----------
def bezier(P, n=60):
    P = np.asarray(P, float)
    t = np.linspace(0, 1, n)[:, None]
    return ((1-t)**3)*P[0] + 3*((1-t)**2)*t*P[1] + 3*(1-t)*(t**2)*P[2] + (t**3)*P[3]

def path(segments, n=60):
    pts = [bezier(s, n) for s in segments]
    return np.vstack([pts[0]] + [p[1:] for p in pts[1:]])

def rdp(pts, eps):
    """Ramer–Douglas–Peucker: Π_ε  (تحويل المنحنى إلى خطوط مستقيمة)"""
    a, b = pts[0], pts[-1]
    ab = b - a
    L = np.hypot(*ab) or 1e-9
    d = np.abs(ab[0]*(pts[:, 1]-a[1]) - ab[1]*(pts[:, 0]-a[0])) / L
    i = int(np.argmax(d))
    if d[i] > eps:
        return np.vstack([rdp(pts[:i+1], eps)[:-1], rdp(pts[i:], eps)])
    return np.vstack([a, b])

def chaikin(poly, k=2):
    """تدوير خفيف للزوايا بعد التبسيط (Chaikin corner cutting)"""
    for _ in range(k):
        q = 0.75*poly[:-1] + 0.25*poly[1:]
        r = 0.25*poly[:-1] + 0.75*poly[1:]
        poly = np.vstack([poly[0], np.column_stack([q, r]).reshape(-1, 2), poly[-1]])
    return poly

def resample(poly, n=200):
    seg = np.hypot(*np.diff(poly, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    u = np.linspace(0, s[-1], n)
    return np.column_stack([np.interp(u, s, poly[:, 0]), np.interp(u, s, poly[:, 1])])

def affine(pts, sx=1.0, sy=1.0, shear=0.0):
    A = np.array([[sx, shear], [0, sy]])
    return pts @ A.T

# ---------- القلم ----------
def pen_stroke(pts, W, theta_deg, taper=None):
    """Minkowski sum للمسار مع مقطع طوله W وزاويته θ."""
    th = np.radians(theta_deg)
    e = np.array([np.cos(th), np.sin(th)])
    n = len(pts)
    w = np.full(n, W) if taper is None else W*taper(np.linspace(0, 1, n))
    L = pts - 0.5*w[:, None]*e
    R = pts + 0.5*w[:, None]*e
    quads = [Polygon([L[i], R[i], R[i+1], L[i+1]]).buffer(0) for i in range(n-1)]
    return unary_union([q for q in quads if q.area > 1e-9])

def width_law(phi_deg, W, theta_deg):
    """سُمك الخط في اتجاه φ :  w(φ) = W·|sin(φ − θ)|"""
    return W*np.abs(np.sin(np.radians(phi_deg - theta_deg)))

# ---------- هياكل تقريبية لحروف النسخ (وحدة القياس = النقطة) ----------
SKEL = {
    "ا": [ [(0, 5), (0, 3.5), (0, 1.5), (0.1, 0)] ],
    "ر": [ [(1.2, 1.4), (1.6, 0.6), (1.4, -1.2), (-1.8, -1.9)] ],
    "ن": [ [(1.9, 1.0), (2.4, -0.4), (1.6, -1.6), (0, -1.6)],
           [(0, -1.6), (-1.6, -1.6), (-2.3, -0.6), (-1.9, 0.6)] ],
}
TAPER_END = lambda u: np.clip(1.15 - 0.95*u**3, 0.15, 1.0)

def naskh(ch):
    return pen_stroke(path(SKEL[ch]), W=1.0, theta_deg=70, taper=TAPER_END if ch != "ا" else None)

def modern(ch, eps=0.18):
    g = path(SKEL[ch])
    g = resample(chaikin(rdp(g, eps)), 200)        # Π_ε
    g = affine(g, sx=0.85, sy=1.25, shear=0.08)    # A
    taper = (lambda u: 1.0 - 0.25*u) if ch == "ا" else (lambda u: np.clip(1.1 - 0.9*u**2, 0.2, 1))
    return pen_stroke(g, W=1.15, theta_deg=18, taper=taper)

def morph(ch, a, eps=0.18):
    """γ_α = (1−α)γ₀ + αγ₁ ,  θ_α = (1−α)·70 + α·18"""
    g0 = resample(path(SKEL[ch]), 200)
    g1 = affine(resample(chaikin(rdp(path(SKEL[ch]), eps)), 200), 0.85, 1.25, 0.08)
    g = (1-a)*g0 + a*g1
    th = (1-a)*70 + a*18
    W = (1-a)*1.0 + a*1.15
    return pen_stroke(g, W, th, taper=TAPER_END), th

# ---------- رسم ----------
INK, BG = "#f4efe6", "#0f2a1f"

def draw(ax, geom, title=""):
    geoms = getattr(geom, "geoms", [geom])
    for g in geoms:
        x, y = g.exterior.xy
        ax.fill(x, y, color=INK, lw=0)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_facecolor(BG)
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 5.8)
    if title:
        ax.set_title(title, color=INK, fontsize=10)

if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "pen_model.png"
    fig, axes = plt.subplots(3, 6, figsize=(13, 7.6), facecolor=BG)
    names = {"ا": "Alef", "ر": "Ra", "ن": "Nun"}
    for r, ch in enumerate(["ا", "ر", "ن"]):
        draw(axes[r, 0], naskh(ch), f"{names[ch]}  Naskh  θ=70°")
        for c, a in enumerate([0.25, 0.5, 0.75], start=1):
            g, th = morph(ch, a)
            draw(axes[r, c], g, f"α={a}  θ={th:.0f}°")
        draw(axes[r, 4], modern(ch), f"{names[ch]}  Modern  θ=18°")
        ax = axes[r, 5]; ax.set_facecolor(BG); ax.axis("off")
    # منحنى قانون السُمك
    ax = fig.add_axes([0.80, 0.12, 0.17, 0.25], facecolor=BG)
    phi = np.linspace(-90, 90, 361)
    ax.plot(phi, width_law(phi, 1, 70), color="#e0a458", label="Naskh θ=70°")
    ax.plot(phi, width_law(phi, 1, 18), color=INK, label="Modern θ=18°")
    ax.set_xlabel("stroke direction φ (deg)", color=INK, fontsize=8)
    ax.set_ylabel("w/W", color=INK, fontsize=8)
    ax.tick_params(colors=INK, labelsize=7)
    for s in ax.spines.values(): s.set_color(INK)
    ax.legend(fontsize=7, facecolor=BG, labelcolor=INK, edgecolor=INK)
    ax.set_title("w(φ) = W·|sin(φ−θ)|", color=INK, fontsize=9)
    fig.savefig(out, dpi=130, facecolor=BG, bbox_inches="tight")
    print("saved", out)
