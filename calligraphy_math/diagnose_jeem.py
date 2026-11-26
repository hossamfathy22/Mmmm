"""
تشخيص الفجوة بين ناتج النموذج والجيم الحديثة.

  سقف القياس:      رسم الجيم الحديثة من هيكلها وسُمكها المستخرَجين → أعلى IoU ممكن بطريقة الرسم هذه
  تبديل 1:         هيكل النموذج + سُمك الهدف     → إن ارتفع كثيراً فالمشكلة في السُمك
  تبديل 2:         هيكل الهدف  + سُمك النموذج   → إن ارتفع كثيراً فالمشكلة في الهيكل
  توزيع الخطأ:     كل بكسل خاطئ يُنسب لأقرب جزء من هيكل الهدف (رأس، خطّاف، شعرة، منحنى يسار، ذيل)
"""
import sys, json
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree

UP = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else "results_diag"
FITS = sys.argv[3] if len(sys.argv) > 3 else "results_v2/style_fits.json"
sys.argv = [sys.argv[0], UP]
import style_model as SM
import srvf

MODERN_GUIDES = [[(238, 232), (255, 290), (300, 308), (380, 310), (435, 318), (455, 360), (460, 420)],
                 [(455, 422), (350, 390), (260, 365), (220, 365), (200, 390), (195, 418), (205, 470),
                  (255, 515), (310, 545), (390, 547), (460, 518), (520, 503), (575, 505), (618, 540)]]

# ---------- لوحة الهدف (نفس fit_canvas) ----------
ink = SM.ink_mask(SM.IMG["jeem_modern"])
ys, xs = np.nonzero(ink); pad = 20
y0, x0 = max(ys.min() - pad, 0), max(xs.min() - pad, 0)
crop = ink[y0:ys.max() + pad, x0:xs.max() + pad]
SIZE = 260; f = SIZE / max(crop.shape)
T = SM.fit_canvas(ink, size=SIZE)
H, W = T.shape
to_canvas = lambda P: (P - np.array([x0, y0])) * f

# ---------- هيكل الهدف وسُمكه (مع إصلاح مناطق الالتقاء) ----------
raw = [SM.refine(ink, SM.spline(g, 200)) for g in MODERN_GUIDES]
zones = SM.junction_zones(raw)
tgt = [(to_canvas(c), SM.fill_zones(w, z) * f) for (c, w), z in zip(raw, zones)]

# ---------- ناتج النموذج في إحداثيات اللوحة ----------
fits = json.load(open(FITS))["fits"]; best = max(fits, key=lambda r: r["IoU"])
p, al = np.array(list(best["params"].values())), np.array(best["align"])
_, J = SM.extract("jeem_naskh")
k, tx, ty = al
mod = [(np.column_stack([g[:, 0] * k * H + W / 2 + tx * H, -g[:, 1] * k * H + H / 2 + ty * H]), w * k * H)
       for g, w in SM.apply_style(J, p)]

def draw(strokes):
    img = Image.new("L", (W, H), 0); dr = ImageDraw.Draw(img)
    for X, w in strokes:
        n = SM.normals(X); L, R = X + 0.5 * w[:, None] * n, X - 0.5 * w[:, None] * n
        for i in range(len(X) - 1):
            dr.polygon([tuple(L[i]), tuple(R[i]), tuple(R[i + 1]), tuple(L[i + 1])], fill=255)
    return np.asarray(img) > 127

iou = lambda a: float((a & T).sum() / (a | T).sum())

def matched(a, b, n=200):
    """نقاط b المقابلة لنقاط a (تطابق مرن SRVF)."""
    A, B = srvf.resample(a, n), srvf.resample(b, n)
    q0, _, _ = srvf.to_q(A); q1, _, _ = srvf.to_q(B)
    psi = srvf.dp_reparam(q0, q1)
    t = np.linspace(0, 1, n)
    return A, np.column_stack([np.interp(psi, t, B[:, 0]), np.interp(psi, t, B[:, 1])]), psi

def resample_w(c, w, n=200):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]; u = s / s[-1]
    return np.interp(np.linspace(0, 1, n), u, w)

rows = []
swapA, swapB = [], []
for (Xt, wt), (Xm, wm) in zip(tgt, mod):
    At, Bm, psi = matched(Xt, Xm)                          # لكل نقطة هدف: نقطة النموذج المقابلة
    wt_r = resample_w(Xt, wt); wm_r = np.interp(psi, np.linspace(0, 1, 200), resample_w(Xm, wm))
    swapA.append((Bm, wt_r))                               # هيكل النموذج + سُمك الهدف
    swapB.append((At, wm_r))                               # هيكل الهدف + سُمك النموذج
    rows.append(dict(dev=np.linalg.norm(At - Bm, axis=1), wt=wt_r, wm=wm_r, At=At))

pred = draw(mod)
res = dict(model_IoU=iou(pred), ceiling_IoU=iou(draw(tgt)),
           model_skeleton_plus_target_widths_IoU=iou(draw(swapA)),
           target_skeleton_plus_model_widths_IoU=iou(draw(swapB)))

# ---------- توزيع الخطأ على الأجزاء ----------
# أجزاء الهدف: الضربة 1 = رأس حتى الزاوية ثم خطّاف (ينزل)؛ الضربة 2 = شعرة، منحنى يسار، ذيل
def parts():
    lab = []
    A1 = rows[0]["At"]; d = np.gradient(A1, axis=0); down = np.degrees(np.arctan2(d[:, 1], d[:, 0])) > 50   # y للأسفل
    corner = int(np.argmax(down & (np.arange(len(A1)) > 60)))
    for i in range(len(A1)):
        lab.append(("head" if i < corner else "hook", A1[i]))
    A2 = rows[1]["At"]
    left = int(np.argmin(A2[:, 0])); bottom = left + int(np.argmax(A2[left:, 1]))
    for i in range(len(A2)):
        lab.append(("hairline" if A2[i, 0] > A2[left, 0] + 8 and i < left else "left curve" if i < bottom else "tail swash", A2[i]))
    return lab
lab = parts()
tree = cKDTree(np.array([p_ for _, p_ in lab])); names = [n for n, _ in lab]
err = {}
for kind, m in (("missed", T & ~pred), ("extra", pred & ~T)):
    yy, xx = np.nonzero(m)
    _, idx = tree.query(np.column_stack([xx, yy]))
    for i in idx:
        err.setdefault(names[i], {"missed": 0, "extra": 0})[kind] += 1
tot = sum(v["missed"] + v["extra"] for v in err.values())
res["error_share_by_part_pct"] = {n: round(100 * (v["missed"] + v["extra"]) / tot, 1) for n, v in
                                  sorted(err.items(), key=lambda kv: -(kv[1]["missed"] + kv[1]["extra"]))}
res["error_by_part_px"] = err
res["mean_skeleton_offset_pct_H"] = [round(100 * float(r["dev"].mean()) / H, 2) for r in rows]
res["mean_abs_width_error_pct_H"] = [round(100 * float(np.abs(r["wt"] - r["wm"]).mean()) / H, 2) for r in rows]
print(json.dumps(res, ensure_ascii=False, indent=1))

import os; os.makedirs(OUT, exist_ok=True)
json.dump(res, open(f"{OUT}/diagnosis.json", "w"), ensure_ascii=False, indent=1)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
def ov(a):
    o = np.ones(a.shape + (3,)); o[T & ~a] = (0.86, 0.3, 0.3); o[a & ~T] = (0.3, 0.45, 0.9); o[a & T] = (0.15, 0.15, 0.15)
    return o
fig = plt.figure(figsize=(17, 9), facecolor="white")
for i, (img, ttl) in enumerate([(draw(tgt), f"ceiling: target redrawn from its own skeleton\nIoU {res['ceiling_IoU']:.2f}"),
                                (pred, f"model\nIoU {res['model_IoU']:.2f}"),
                                (draw(swapA), f"model skeleton + TARGET widths\nIoU {res['model_skeleton_plus_target_widths_IoU']:.2f}"),
                                (draw(swapB), f"TARGET skeleton + model widths\nIoU {res['target_skeleton_plus_model_widths_IoU']:.2f}")]):
    ax = fig.add_subplot(2, 4, i + 1); ax.imshow(ov(img)); ax.axis("off"); ax.set_title(ttl, fontsize=9)
ax = fig.add_subplot(2, 4, 5)
cols = {"head": "#1f77b4", "hook": "#d62728", "hairline": "#ff7f0e", "left curve": "#2ca02c", "tail swash": "#9467bd"}
ax.imshow(T, cmap="Greys", alpha=0.25)
for n_, p_ in lab: ax.plot(p_[0], p_[1], ".", color=cols[n_], ms=3)
ax.axis("off"); ax.set_title("target parts", fontsize=9)
ax = fig.add_subplot(2, 4, 6)
sh = res["error_share_by_part_pct"]; ax.barh(list(sh)[::-1], list(sh.values())[::-1], color=[cols[n] for n in list(sh)[::-1]])
ax.set_xlabel("% of all wrong pixels"); ax.set_title("where the error is", fontsize=9)
for j, r in enumerate(rows):
    ax = fig.add_subplot(2, 4, 7 + j); u = np.linspace(0, 1, 200)
    ax.plot(u, 100 * r["wt"] / H, color="k", label="target width"); ax.plot(u, 100 * r["wm"] / H, color="#1f77b4", label="model width")
    ax.plot(u, 100 * r["dev"] / H, color="#d62728", ls="--", label="skeleton offset")
    ax.set_xlabel(f"stroke {j + 1} (arc length)"); ax.set_ylabel("% of letter height"); ax.legend(fontsize=7)
    ax.set_title(["stroke 1: head → hook", "stroke 2: hairline → curve → tail"][j], fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/jeem_diagnosis.png", dpi=110)
print("saved")
