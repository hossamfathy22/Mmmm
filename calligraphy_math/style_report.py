"""رسم نتائج style_model.py: ملاءمة الجيم + التنبؤ بالصاد مع خريطة الاتفاق بين التقديرات."""
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results"]
import style_model as M

UP, OUT = sys.argv[1], sys.argv[2]
R = json.load(open(f"{OUT}/style_fits.json"))
fits = sorted(R["fits"], key=lambda f: -f["IoU"])
_, J = M.extract("jeem_naskh")
_, S = M.extract("sad_naskh")
T = M.fit_canvas(M.ink_mask(M.IMG["jeem_modern"]))

def ov(a, b):
    o = np.ones(a.shape + (3,))
    o[b & ~a] = (0.86, 0.3, 0.3); o[a & ~b] = (0.3, 0.45, 0.9); o[a & b] = (0.12, 0.12, 0.12)
    return o

def show(ax, img, title, cmap=None):
    ax.imshow(img, cmap=cmap); ax.axis("off"); ax.set_title(title, fontsize=9)

# ---------- (أ) الجيم ----------
best = fits[0]
p, al = np.array(list(best["params"].values())), np.array(best["align"])
pred = M.render(M.apply_style(J, p), al, T.shape)
hi = M.render(M.apply_style(J, p), al, T.shape, scale=3)
fig, ax = plt.subplots(1, 4, figsize=(15, 4.3), facecolor="white")
show(ax[0], Image.open(M.IMG["jeem_naskh"]).convert("RGB"), "input: Naskh jeem")
show(ax[1], ~hi, f"model output (best of {len(fits)} fits)", cmap="gray")
show(ax[2], Image.open(M.IMG["jeem_modern"]).convert("RGB"), "target: modern jeem")
show(ax[3], ov(pred, T), f"overlay  IoU {best['IoU']:.2f} · edge err {best['edge_err_pct']:.1f}% H\n"
     f"(Naskh as-is: IoU {R['baseline_jeem']['IoU']:.2f})")
fig.suptitle("Fit on jeem — grey = match · red = missed · blue = extra", fontsize=10)
fig.savefig(f"{OUT}/jeem_fit.png", dpi=120, bbox_inches="tight")

# ---------- (ب) الصاد ----------
def framed(strokes, p, H=300, margin=0.08):
    """رسم داخل إطار يتسع للشكل كله (بلا قص)."""
    st = M.apply_style(strokes, p)
    pts = np.vstack([g for g, _ in st]); wmax = max(w.max() for _, w in st)
    lo, hi = pts.min(0) - wmax, pts.max(0) + wmax
    ctr, span = (lo + hi) / 2, hi - lo
    st = [(g - ctr, w) for g, w in st]
    k = (1 - 2 * margin) / max(span[1], span[0] / 1.4)
    return M.render(st, (k, 0, 0), (H, int(H * 1.4)), scale=2)

preds = [framed(S, np.array(list(f["params"].values()))) for f in fits]
agree = np.mean(preds, axis=0)
fig, ax = plt.subplots(2, 4, figsize=(16, 7.4), facecolor="white")
show(ax[0, 0], Image.open(M.IMG["sad_naskh"]).convert("RGB").crop((0, 250, 736, 850)), "input: sad")
show(ax[0, 1], ~preds[0], f"prediction (fit #1, IoU on jeem {fits[0]['IoU']:.2f})", cmap="gray")
show(ax[0, 2], ~preds[1], f"prediction (fit #2, IoU on jeem {fits[1]['IoU']:.2f})", cmap="gray")
show(ax[0, 3], 1 - agree, "agreement of all 5 fits\n(black = all agree, grey = uncertain)", cmap="gray")
for i in range(3):
    show(ax[1, i], ~preds[i + 2], f"prediction (fit #{i + 3}, IoU on jeem {fits[i + 2]['IoU']:.2f})", cmap="gray")
ax[1, 3].axis("off")
fig.suptitle("sad predicted with the style learned from jeem only (no modern sad was used)", fontsize=11)
fig.savefig(f"{OUT}/sad_prediction.png", dpi=110, bbox_inches="tight")

# ---------- جدول المعاملات ----------
P = np.array([list(f["params"].values()) for f in fits])
print(f"{'param':18s} {'mean':>8s} {'std':>7s}   values")
for j, n in enumerate(M.NAMES):
    print(f"{n:18s} {P[:, j].mean():8.2f} {P[:, j].std():7.2f}   " + " ".join(f"{v:7.2f}" for v in P[:, j]))
print("IoU:", [round(f["IoU"], 3) for f in fits], "baseline:", round(R["baseline_jeem"]["IoU"], 3))

# ---------- (ج) الشكل النهائي للصاد: متوسط معاملات كل «نمط» من الحلول ----------
modeA = [f for f in fits if abs(f["params"]["rotation ρ°"]) < 10]
modeB = [f for f in fits if abs(f["params"]["rotation ρ°"]) >= 10]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), facecolor="white")
show(ax[0], Image.open(M.IMG["sad_naskh"]).convert("RGB").crop((0, 250, 736, 850)), "input: sad")
for a, grp, name in [(ax[1], modeA, "A"), (ax[2], modeB, "B")]:
    if not grp:
        a.axis("off"); continue
    pm = np.mean([list(f["params"].values()) for f in grp], axis=0)
    iou_m = np.mean([f["IoU"] for f in grp])
    show(a, ~framed(S, pm), f"mode {name}: mean of {len(grp)} fits (jeem IoU {iou_m:.2f})", cmap="gray")
fig.suptitle("Predicted modern sad — two solutions fit the jeem almost equally well", fontsize=11)
fig.savefig(f"{OUT}/sad_final.png", dpi=120, bbox_inches="tight")
print("mode A:", len(modeA), "fits; mode B:", len(modeB), "fits")
