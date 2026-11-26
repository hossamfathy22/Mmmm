"""
إصلاح معادلات الحل A (الذي اختاره المستخدم) دون تغيير معاملات الأسلوب المتعلَّمة من الجيم.

الإصلاحات (كلها قواعد عامة، تُطبَّق على كل الحروف بما فيها الجيم):
 E1  مناطق التقاء الضربات تُستبعد من قياس السُمك وتُملأ بالاستيفاء من الجانبين   (في style_model.extract)
 E2  سقف ناعم للسُمك متعلَّم من الجيم:   w ← W_max · tanh(w / W_max) ,  W_max = أقصى سُمك في الجيم المطابقة
 E3  حفظ الفراغ الداخلي (counter) للعُقَد المغلقة:
         أكبر f ∈ [0.35, 1] بحيث  r_hole/r_loop (f·w_loop)  ≥  κ · (النسبة نفسها في النسخ) ,  κ = 1
         r_hole: نصف قطر أكبر دائرة داخل الفراغ،  r_loop: نصف السُمك الوسيط لضربة العقدة
 E4  تنعيم السُمك والهيكل + استدقاق الأطراف + رسم مضاد للتسنّن (4× ثم تصغير)
 E5  حفظ الالتقاء: الضربة التي يبدأ طرفها على ضربة أخرى تُزاح لتبقى ملتصقة بها:
         γ_b ← γ_b + [γ'_a(i*) + A·(γ_b(0) − γ_a(i*)) − γ'_b(0)]
"""
import sys, json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

UP = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else "results_refined"
FITS = sys.argv[3] if len(sys.argv) > 3 else "results/style_fits.json"
sys.argv = [sys.argv[0], UP]
import style_model as SM

USE_CAP = False                   # E2 كان ترقيعاً لكتلة سببها خطأ القياس؛ بعد إصلاح القياس يضرّ الجيم (0.72→0.67)
KAPPA = 1.0                      # قاعدة تصميم: العقدة تبقى مفتوحة كما في النسخ

def mode_A():
    fits = json.load(open(FITS))["fits"]
    A = [f for f in fits if abs(f["params"]["rotation ρ°"]) < 10]
    return np.mean([list(f["params"].values()) for f in A], axis=0), A

def closed_strokes(strokes, thr=0.08):
    return [k for k, (c, w) in enumerate(strokes) if np.linalg.norm(c[0] - c[-1]) < thr]

def soft_cap(styled, wmax):
    return [(g, wmax * np.tanh(w / wmax)) for g, w in styled]

def smooth(styled, sig=3):
    out = []
    for g, w in styled:
        g2 = ndimage.gaussian_filter1d(g, 2, axis=0, mode="nearest"); g2[[0, -1]] = g[[0, -1]]
        u = np.linspace(0, 1, len(w))
        x = np.clip(np.minimum(u, 1 - u) / 0.06, 0, 1)
        out.append((g2, ndimage.gaussian_filter1d(w, sig, mode="nearest") * (0.05 + 0.95 * x * x * (3 - 2 * x))))
    return out

def render(styled, H=300, W=420, margin=0.08, ss=4, align=None, return_k=False):
    P = np.vstack([g for g, _ in styled]); wmax = max(w.max() for _, w in styled)
    if align is None:
        lo, hi = P.min(0) - wmax / 2, P.max(0) + wmax / 2
        ctr, span = (lo + hi) / 2, hi - lo
        k = (1 - 2 * margin) * min(H / span[1], W / span[0])
    else:
        ctr, k = align
    img = Image.new("L", (W * ss, H * ss), 0); dr = ImageDraw.Draw(img)
    for g, w in styled:
        X = np.column_stack([(g[:, 0] - ctr[0]) * k * ss + W * ss / 2, -(g[:, 1] - ctr[1]) * k * ss + H * ss / 2])
        r = w * k * ss / 2; n = SM.normals(X)
        L, R = X + r[:, None] * n, X - r[:, None] * n
        for i in range(len(X) - 1):
            dr.polygon([tuple(L[i]), tuple(R[i]), tuple(R[i + 1]), tuple(L[i + 1])], fill=255)
    img = img.filter(ImageFilter.GaussianBlur(ss * 0.4)).resize((W, H), Image.LANCZOS)
    a = np.asarray(img).astype(float) / 255
    return (a, k) if return_k else a

def hole_radius_px(mask):
    holes = ndimage.binary_fill_holes(mask) & ~mask
    return float(ndimage.distance_transform_edt(holes).max()) if holes.any() else 0.0

def openness(styled, loops):
    """انفتاح العقدة = نصف قطر أكبر دائرة داخل فراغها ÷ نصف السُمك الوسيط لضربة العقدة نفسها."""
    a, k = render(styled, return_k=True)
    r_hole = hole_radius_px(a > 0.5) / k
    r_loop = np.median(np.concatenate([styled[i][1] for i in loops])) / 2
    return r_hole / r_loop

def affine_of(p):
    s_, rho, a = p[0], np.radians(p[1]), p[2]
    R = np.array([[np.cos(rho), -np.sin(rho)], [np.sin(rho), np.cos(rho)]])
    return R @ np.array([[1, s_], [0, 1]]) @ np.diag([np.exp(a / 2), np.exp(-a / 2)])

def keep_junctions(strokes, st, p, thr=0.06):
    """E5"""
    A = affine_of(p); st = [(g.copy(), w) for g, w in st]
    for b in range(len(strokes)):
        cb = strokes[b][0]
        for end in (0, len(cb) - 1):
            best = None
            for a_ in range(len(strokes)):
                if a_ == b:
                    continue
                d = np.linalg.norm(strokes[a_][0] - cb[end], axis=1); i = int(np.argmin(d))
                if d[i] < thr and (best is None or d[i] < best[0]):
                    best = (d[i], a_, i)
            if best and best[1] < b or (best and b == 0 and end == 0):
                _, a_, i = best
                target = st[a_][0][i] + A @ (cb[end] - strokes[a_][0][i])
                st[b] = (st[b][0] + (target - st[b][0][end]), st[b][1])
                break
    return st

def refine(strokes, p, wmax, naskh_mask=None):
    st = SM.apply_style(strokes, p)
    st = keep_junctions(strokes, st, p)                              # E5
    if USE_CAP:
        st = soft_cap(st, wmax)                                     # E2 (أُلغي بعد إصلاح E1 من جذره)
    st = smooth(st)                                                 # E4
    loops = closed_strokes(strokes)
    info = {}
    if loops:                                                       # E3
        ys, _ = np.nonzero(naskh_mask)
        ref = (hole_radius_px(naskh_mask) / np.ptp(ys)) / (np.median(np.concatenate([strokes[i][1] for i in loops])) / 2)
        lo, hi = 0.35, 1.0
        def with_f(f):
            return [(g, w * (f if k in loops else 1.0)) for k, (g, w) in enumerate(st)]
        if openness(with_f(1.0), loops) < KAPPA * ref:
            for _ in range(14):
                mid = (lo + hi) / 2
                if openness(with_f(mid), loops) >= KAPPA * ref:
                    lo = mid
                else:
                    hi = mid
            f = lo
        else:
            f = 1.0
        st = with_f(f)
        info = dict(loop_strokes=loops, naskh_openness=round(float(ref), 3),
                    loop_width_factor=round(f, 3), openness_before=round(float(openness(st if f == 1.0 else with_f(1.0), loops)), 3), openness_after=round(float(openness(st, loops)), 3))
    return st, info

if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    p, A = mode_A()
    _, J = SM.extract("jeem_naskh")
    sad_mask, S = SM.extract("sad_naskh")
    T = SM.fit_canvas(SM.ink_mask(SM.IMG["jeem_modern"]))

    # W_max متعلَّم من الجيم: 97% من أعلى سُمك في الجيم المطابقة (الحل A)
    stJ_raw = SM.apply_style(J, p)
    wmax = float(np.percentile(np.concatenate([w for _, w in stJ_raw]), 97))

    # فحص الجيم: الإصلاحات يجب ألّا تفسد حرف التدريب
    def jeem_iou(styled):
        best = 0
        for f in A:
            al = np.array(f["align"])
            m = SM.render(styled, al, T.shape)
            best = max(best, (m & T).sum() / (m | T).sum())
        return best
    stJ_fix, _ = refine(J, p, wmax)
    al = SM.fit_align(J, T, p)
    def iou_at(styled):
        m = SM.render(styled, al, T.shape); return float((m & T).sum() / (m | T).sum())
    j_before, j_after = iou_at(stJ_raw), iou_at(stJ_fix)

    stS_raw = SM.apply_style(S, p)
    stS_fix, info = refine(S, p, wmax, naskh_mask=sad_mask)
    summary = dict(params_mode_A_mean=dict(zip(SM.NAMES, np.round(p, 3).tolist())), W_max=round(wmax, 4),
                   jeem_IoU_before_fixes=round(j_before, 3), jeem_IoU_after_fixes=round(j_after, 3), sad=info)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    json.dump(summary, open(f"{OUT}/summary.json", "w"), ensure_ascii=False, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), facecolor="white")
    ax[0].imshow(Image.open(SM.IMG["sad_naskh"]).convert("RGB").crop((0, 250, 736, 850))); ax[0].set_title("input: Naskh sad")
    ax[1].imshow(1 - render(stS_raw), cmap="gray", vmin=0, vmax=1); ax[1].set_title("style before fixes")
    ax[2].imshow(1 - render(stS_fix), cmap="gray", vmin=0, vmax=1); ax[2].set_title("after fixes E3–E5")
    for a in ax: a.axis("off")
    fig.savefig(f"{OUT}/sad_before_after.png", dpi=120, bbox_inches="tight")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), facecolor="white")
    ax[0].imshow(1 - render(stJ_raw, H=300, W=340), cmap="gray"); ax[0].set_title(f"jeem before fixes (IoU {j_before:.2f})")
    ax[1].imshow(1 - render(stJ_fix, H=300, W=340), cmap="gray"); ax[1].set_title(f"jeem after fixes (IoU {j_after:.2f})")
    ax[2].imshow(Image.open(SM.IMG["jeem_modern"]).convert("RGB")); ax[2].set_title("target")
    for a in ax: a.axis("off")
    fig.savefig(f"{OUT}/jeem_check.png", dpi=110, bbox_inches="tight")
    Image.fromarray((255 * (1 - render(stS_fix, H=600, W=840))).astype(np.uint8)).save(f"{OUT}/sad_refined_large.png")
    print("saved")
