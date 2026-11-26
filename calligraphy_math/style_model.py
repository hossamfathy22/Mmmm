"""
النسخة العلمية: نموذج أسلوب عام (لا يعتمد على الحرف) يُقدَّر بالرسم العكسي.

تمثيل الحرف: ضربات، لكل ضربة هيكل γ(s) بطول قوس موحَّد وسُمك w(s) (مقاسان من الصورة بالمقطع العرضي).

نموذج الأسلوب  p = (s, ρ, a, c, β, f, S, γ, m, θ)   — 10 معاملات:
  الهيكل:
    φ'(u) = φ(0) + c·[φ(u) − φ(0)] + σ·β·R((u − 1 + f)/f)     (c: كسب الانحناء، β: التفاف الذيل على آخر f من الضربة الأخيرة)
    γ'(u) = γ(0) + ∫ (cos φ', sin φ') ds   ثم تصحيح خطي يحفظ نقطة الالتقاء لغير الضربة الأخيرة
    γ''   = A(s, ρ, a)·γ'                                     (ميل، دوران، نسبة أبعاد)
  السُمك:
    w'(u) = τ(u) · S · (w(u)/w̄)^γ · ( m + |sin(φ''(u) − θ)| )   (تضخيم تباين النسخ γ + قلم عريض θ)
  σ = إشارة اتجاه دوران الضربة (ليكون الالتفاف «مع» أو «ضد» انحناء الحرف بالمعنى نفسه في كل الحروف)

التقدير:  p* = argmin_p  [1 − IoU(Render(p, محاذاة), الهدف)]   بـ Differential Evolution + Nelder–Mead
عدم اليقين: 5 تقديرات مستقلة (بذور مختلفة) → تشتت المعاملات + خريطة اتفاق للتنبؤ.
"""
import sys, os, json
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.interpolate import splprep, splev
from scipy.optimize import differential_evolution, minimize

UP = sys.argv[1] if len(sys.argv) > 1 else "."
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else "."
SEEDS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1,2,3,4,5").split(",")]

IMG = {
    "jeem_naskh": f"{UP}/83a4afbb-image.jpg",
    "jeem_modern": f"{UP}/f0b4c74c-image.jpg",
    "sad_naskh": f"{UP}/5bf98b4e-image.jpg",
}
# هياكل دليلية تقريبية بترتيب الكتابة (تُصحَّح آلياً)؛ الضربة الأخيرة طرفها حر (الذيل)
GUIDE = {
    "jeem_naskh": [
        [(198, 272), (220, 230), (250, 217), (350, 226), (460, 238)],
        [(450, 255), (370, 275), (280, 330), (225, 410), (210, 480), (235, 540),
         (300, 580), (400, 582), (480, 570), (545, 547)],
    ],
    "sad_naskh": [
        [(352, 530), (400, 515), (440, 490), (480, 445), (550, 383), (620, 345), (680, 340),
         (708, 380), (690, 440), (630, 490), (550, 525), (470, 540), (400, 548), (352, 555)],
        [(345, 425), (345, 520), (355, 600), (340, 670), (300, 720), (240, 748), (150, 755),
         (70, 735), (32, 670), (45, 600), (80, 530), (118, 472)],
    ],
}
N = 160

# ---------------- الصور ----------------
def ink_mask(path):
    a = np.asarray(Image.open(path).convert("RGB")).astype(int)
    ink = (a.max(2) < 140) & ((a.max(2) - a.min(2)) < 40)          # أسود فقط
    lab, k = ndimage.label(ink)
    sz = ndimage.sum(ink, lab, range(1, k + 1))
    return np.isin(lab, 1 + np.where(sz > 0.01 * sz.max())[0])

def body_mask(path):
    """جسم الحرف بلا النقطة (أكبر مكوّن)."""
    ink = ink_mask(path)
    lab, k = ndimage.label(ink)
    sz = ndimage.sum(ink, lab, range(1, k + 1))
    return lab == 1 + int(np.argmax(sz))

def fit_canvas(mask, size=260, pad=20):
    ys, xs = np.nonzero(mask)
    m = mask[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]
    f = size / max(m.shape)
    im = Image.fromarray(m.astype(np.uint8) * 255).resize(
        (round(m.shape[1] * f), round(m.shape[0] * f)), Image.BILINEAR)
    return np.asarray(im) > 127

# ---------------- الهيكل والسُمك ----------------
def spline(pts, n):
    pts = np.asarray(pts, float)
    tck, _ = splprep(pts.T, s=0, k=min(3, len(pts) - 1))
    return np.column_stack(splev(np.linspace(0, 1, n), tck))

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
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
    u = np.linspace(0, seg[-1], n)
    return np.column_stack([np.interp(u, seg, c[:, 0]), np.interp(u, seg, c[:, 1])])

def refine(mask, c, iters=4, maxlen=140, max_shift=8.0):
    c0 = c.copy()
    for it in range(iters + 1):
        n = normals(c)
        a = np.array([march(mask, p, d, maxlen) for p, d in zip(c, n)])
        b = np.array([march(mask, p, -d, maxlen) for p, d in zip(c, n)])
        if it == iters:
            break
        mid = ndimage.gaussian_filter1d(c + 0.5 * (a - b)[:, None] * n, 5, axis=0, mode="nearest")
        mid[[0, -1]] = c[[0, -1]]
        d = mid - c0; L = np.linalg.norm(d, axis=1, keepdims=True)
        c = resample(c0 + d * np.minimum(1, max_shift / (L + 1e-9)), len(c))
    w = ndimage.median_filter(np.minimum(a + b, 2 * np.minimum(a, b) + 4), 5)
    return c, ndimage.gaussian_filter1d(w, 4)

def extract(name):
    mask = body_mask(IMG[name])
    ys, xs = np.nonzero(mask)
    H = float(ys.max() - ys.min())
    ctr = np.array([(xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2])
    strokes = []
    for g in GUIDE[name]:
        c, w = refine(mask, spline(g, N))
        # أطراف الضربة تقع غالباً داخل ضربة أخرى (تقاطع) أو على قَطّ مائل، فيُقاس السُمك أكبر من حقيقته:
        # نقيّد أول وآخر 12% بقيمة السُمك الداخلية عند حدّها
        e = int(0.12 * len(w))
        w[:e] = np.minimum(w[:e], w[e]); w[-e:] = np.minimum(w[-e:], w[-e - 1])
        c = (c - ctr) / H
        c[:, 1] *= -1                                             # y للأعلى (رياضياً)
        strokes.append((c, np.maximum(w, 1.0) / H))
    return mask, strokes

# ---------------- نموذج الأسلوب ----------------
NAMES = ["shear s", "rotation ρ°", "aspect a", "curvature gain c", "swash β°", "swash span f",
         "weight S", "contrast γ", "pen floor m", "pen angle θ°"]
BOUNDS = [(-0.7, 0.7), (-40, 40), (-0.7, 0.7), (0.4, 1.8), (-150, 150), (0.05, 0.5),
          (0.02, 0.35), (0.0, 3.0), (0.0, 1.5), (0.0, 180.0)]
ALIGN_BOUNDS = [(0.6, 1.4), (-0.25, 0.25), (-0.25, 0.25)]

def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)

def apply_style(strokes, p):
    s, rho, a, c, beta, f, S, gam, m, th = p
    r = np.radians(rho)
    A = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]]) @ \
        np.array([[1, s], [0, 1]]) @ np.diag([np.exp(a / 2), np.exp(-a / 2)])
    wbar = np.mean(np.concatenate([w for _, w in strokes]))
    out = []
    for k, (g, w) in enumerate(strokes):
        last = k == len(strokes) - 1
        d = np.diff(g, axis=0)
        ds = np.linalg.norm(d, axis=1)
        phi = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        u = np.linspace(0, 1, len(phi))
        sigma = np.sign(phi[-1] - phi[0]) or 1.0
        phi2 = phi[0] + c * (phi - phi[0])
        if last:
            phi2 = phi2 + sigma * np.radians(beta) * smoothstep((u - (1 - f)) / f)
        g2 = np.vstack([g[0], g[0] + np.cumsum(ds[:, None] * np.column_stack([np.cos(phi2), np.sin(phi2)]), axis=0)])
        if not last:                                              # حفظ نقطة الالتقاء
            uu = np.linspace(0, 1, len(g2))[:, None]
            g2 = g2 + uu * (g[-1] - g2[-1])
        g3 = g2 @ A.T
        t = np.gradient(g3, axis=0)
        ang = np.arctan2(t[:, 1], t[:, 0])
        U = np.linspace(0, 1, len(g3))
        tau = np.clip(np.minimum(U, 1 - U) / 0.04, 0, 1) ** 0.5
        w2 = tau * S * (w / wbar) ** gam * (m + np.abs(np.sin(ang - np.radians(th))))
        out.append((g3, w2))
    return out

def render(styled, align, shape, scale=1.0):
    k, tx, ty = align
    H, W = shape[0] * scale, shape[1] * scale
    img = Image.new("L", (int(W), int(H)), 0)
    dr = ImageDraw.Draw(img)
    for g, w in styled:
        P = np.column_stack([g[:, 0], -g[:, 1]]) * k * H + np.array([W / 2 + tx * H, H / 2 + ty * H])
        n = normals(P)
        L, R = P + 0.5 * (w * k * H)[:, None] * n, P - 0.5 * (w * k * H)[:, None] * n
        for i in range(len(P) - 1):
            dr.polygon([tuple(L[i]), tuple(R[i]), tuple(R[i + 1]), tuple(L[i + 1])], fill=255)
    return np.asarray(img) > 127

def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)

def edge_err(a, b):
    ea = a & ~ndimage.binary_erosion(a); eb = b & ~ndimage.binary_erosion(b)
    da = ndimage.distance_transform_edt(~eb)[ea].mean()
    db = ndimage.distance_transform_edt(~ea)[eb].mean()
    return 100 * 0.5 * (da + db) / a.shape[0]

class Loss:
    """دالة الخطأ: 1 − IoU (قابلة للنقل بين العمليات المتوازية)."""
    def __init__(self, strokes, target):
        self.strokes, self.target = strokes, target
    def __call__(self, x):
        return 1 - iou(render(apply_style(self.strokes, x[:10]), x[10:], self.target.shape), self.target)

def fit_style(strokes, target, seed):
    _loss = Loss(strokes, target)
    r = differential_evolution(_loss, BOUNDS + ALIGN_BOUNDS, seed=seed, popsize=14, maxiter=160,
                               tol=1e-7, polish=False, workers=4, updating="deferred")
    r2 = minimize(_loss, r.x, method="Nelder-Mead", options=dict(maxiter=2500, xatol=1e-4, fatol=1e-7))
    x = r2.x if r2.fun < r.fun else r.x
    return x[:10], x[10:]

def fit_align(strokes, target, style):
    loss = lambda al: 1 - iou(render(apply_style(strokes, style), al, target.shape), target)
    r = differential_evolution(loss, ALIGN_BOUNDS, seed=0, popsize=12, maxiter=60, polish=False)
    return minimize(loss, r.x, method="Nelder-Mead").x

def identity_width(strokes):
    return [(g, w) for g, w in strokes]

if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    _, J = extract("jeem_naskh")
    sad_mask, Sd = extract("sad_naskh")
    T = fit_canvas(ink_mask(IMG["jeem_modern"]))     # النقطة ملونة، فكل الحبر الأسود = جسم الحرف

    results = {"fits": []}
    # خط الأساس: النسخ كما هو (محاذاة فقط)
    loss_id = lambda al: 1 - iou(render(identity_width(J), al, T.shape), T)
    r = differential_evolution(loss_id, ALIGN_BOUNDS, seed=0, popsize=12, maxiter=60, polish=False)
    al_id = minimize(loss_id, r.x, method="Nelder-Mead").x
    base = render(identity_width(J), al_id, T.shape)
    results["baseline_jeem"] = dict(IoU=float(iou(base, T)), edge_err_pct=float(edge_err(base, T)))

    for sd in SEEDS:
        p, al = fit_style(J, T, sd)
        pred = render(apply_style(J, p), al, T.shape)
        rec = dict(seed=sd, params=dict(zip(NAMES, map(float, p))), align=list(map(float, al)),
                   IoU=float(iou(pred, T)), edge_err_pct=float(edge_err(pred, T)))
        results["fits"].append(rec)
        print(json.dumps({k: rec[k] for k in ("seed", "IoU", "edge_err_pct")}), flush=True)
        with open(f"{OUTDIR}/style_fits.json", "w") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=1)
    print("done")
