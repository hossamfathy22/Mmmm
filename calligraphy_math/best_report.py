"""تقرير النموذج الأفضل: ملاءمة الجيم، الانتقال المرن (SRVF)، توزيع المعاملات، والتنبؤ بالصاد مع عدم اليقين."""
import sys, json, math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

UP = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else "results_best"
import best_model as B
import kinematics as K
import srvf
SM = B.SM

post = torch.load(f"{OUT}/posterior.pt", weights_only=False)

def letter_from(name, fits):
    L = B.Letter.__new__(B.Letter); L.name = name; L.fits = []
    for f in fits:
        m = K.SigmaLognormal(f["n"], f["c"], f["t"]); m.load_state_dict(f["state"]); m.requires_grad_(False)
        L.fits.append(dict(model=m, **{k: f[k] for k in ("t", "c", "w", "n", "snr")}))
    L.wbar = float(np.mean(np.concatenate([f["w"] for f in L.fits])))
    L.junctions = L._junctions()
    return L

J = letter_from("jeem_naskh", post["J_fits"]); S = letter_from("sad_naskh", post["S_fits"])
modes, wts = post["modes"], post["weights"]
tgt_mask = SM.ink_mask(SM.IMG["jeem_modern"])
cv = B.Canvas(tgt_mask, size=150)
T_hi = SM.fit_canvas(tgt_mask, size=450)

def framed(strokes, H=300, W=420, margin=0.08, scale=2):
    P = torch.cat([s[0] for s in strokes]).detach().numpy(); wmax = max(float(s[1].max()) for s in strokes)
    lo, hi = P.min(0) - wmax / 2, P.max(0) + wmax / 2
    ctr, span = (lo + hi) / 2, hi - lo
    k = (1 - 2 * margin) / max(span[1], span[0] * H / W)
    st = [(s[0] - torch.tensor(ctr), s[1]) for s in strokes]
    return B.hard_render(st, [k, 0, 0], H, W, scale=scale)

def show(ax, img, title, cmap=None):
    ax.imshow(img, cmap=cmap); ax.axis("off"); ax.set_title(title, fontsize=9)

def ov(a, b):
    o = np.ones(a.shape + (3,))
    o[b & ~a] = (0.86, 0.3, 0.3); o[a & ~b] = (0.3, 0.45, 0.9); o[a & b] = (0.12, 0.12, 0.12)
    return o

summary = {}
# ---------------- (1) الجيم: أفضل MAP ----------------
best = modes[0]["map"]
stJ = B.generate(J, best["g"], best["delta"])
predJ = B.hard_render(stJ, best["al"], T_hi.shape[0], T_hi.shape[1])
iou_J = float((predJ & T_hi).sum() / (predJ | T_hi).sum())
summary["jeem"] = dict(IoU=round(iou_J, 3), edge_err_pct=round(float(SM.edge_err(predJ, T_hi)), 2),
                       pulses=[f["n"] for f in J.fits], snr_db=[round(f["snr"], 1) for f in J.fits])

# SRVF: مقارنة الهيكل المولَّد بهيكل الجيم الحديثة المستخرج
TGT_GUIDE = [[(238, 232), (255, 290), (300, 308), (380, 310), (435, 318), (455, 360), (460, 420)],
             [(455, 422), (350, 390), (260, 365), (220, 365), (200, 390), (195, 418), (205, 470),
              (255, 515), (310, 545), (390, 547), (460, 518), (520, 503), (575, 505), (618, 540)]]
mod_strokes = []
for g in TGT_GUIDE:
    c, w = SM.refine(tgt_mask, SM.spline(g, 160))
    c = c.copy(); c[:, 1] *= -1
    mod_strokes.append(c)
d_naskh = [srvf.elastic_distance(f["c"], m) for f, m in zip(J.fits, mod_strokes)]
d_model = [srvf.elastic_distance(s[0].detach().numpy(), m) for s, m in zip(stJ, mod_strokes)]
summary["jeem"]["srvf_distance_rad"] = dict(naskh_vs_modern=[round(x, 3) for x in d_naskh],
                                            model_vs_modern=[round(x, 3) for x in d_model])

fig, ax = plt.subplots(1, 4, figsize=(16, 4.4), facecolor="white")
show(ax[0], Image.open(SM.IMG["jeem_naskh"]).convert("RGB"), "input: Naskh jeem")
show(ax[1], ~framed(stJ), "model output (MAP)", cmap="gray")
show(ax[2], Image.open(SM.IMG["jeem_modern"]).convert("RGB"), "target: modern jeem")
show(ax[3], ov(predJ, T_hi), f"overlay  IoU {iou_J:.2f} · edge err {summary['jeem']['edge_err_pct']:.1f}% H")
fig.savefig(f"{OUT}/1_jeem_fit.png", dpi=120, bbox_inches="tight")

# ---------------- (2) الانتقال المرن (SRVF geodesic) ----------------
alphas = [0, 0.25, 0.5, 0.75, 1]
per_stroke = []
for f, s in zip(J.fits, stJ):
    fr, th = srvf.geodesic(f["c"], s[0].detach().numpy(), alphas, w0=f["w"] / J.wbar * float(np.exp(best["g"][5])),
                           w1=s[1].detach().numpy())
    per_stroke.append(fr)
fig, ax = plt.subplots(1, 5, figsize=(17, 3.8), facecolor="white")
for i, a in enumerate(alphas):
    st = [(torch.tensor(per_stroke[k][i][0]), torch.tensor(per_stroke[k][i][1])) for k in range(len(per_stroke))]
    show(ax[i], ~framed(st, H=260, W=300), f"SRVF geodesic α = {a}", cmap="gray")
fig.savefig(f"{OUT}/2_jeem_geodesic.png", dpi=110, bbox_inches="tight")

# ---------------- (3) توزيع المعاملات (اللاحقة) ----------------
def readable(g):
    g = g.numpy()
    return {"shear s": g[0], "rotation ρ°": math.degrees(g[1]), "aspect (width/height ×)": math.exp(g[2]),
            "curvature gain c": math.exp(g[3]), "swash β°": math.degrees(g[4]), "weight S": math.exp(g[5]),
            "contrast γ": g[6], "pen angle θp°": math.degrees(g[7]) % 180, "pen floor m": math.exp(g[8]),
            "pressure a1 (cos φ)": g[9], "pressure b1 (sin φ)": g[10], "pressure a2": g[11], "pressure b2": g[12],
            "taper length ℓ": math.exp(g[13])}
summary["modes"] = []
for md, w in zip(modes, wts):
    R = [readable(s["g"]) for s in md["samples"]]
    summary["modes"].append(dict(weight=round(float(w), 3), map_softIoU=round(md["map"]["iou"], 3),
                                 neg_log_post=round(md["map"]["nlp"], 2), n_samples=len(R),
                                 params={k: [round(float(np.mean([r[k] for r in R])), 3),
                                             round(float(np.std([r[k] for r in R])), 3)] for k in R[0]}))

# ---------------- (4) التنبؤ بالصاد ----------------
H, W = 300, 420
maps, mode_imgs = [], []
for md in modes:
    imgs = []
    for s in md["samples"][::2]:
        st, _ = B.predict(S, s["g"], iters=150)
        imgs.append(framed(st, H, W))
    maps.append(np.mean(imgs, 0))
    st, _ = B.predict(S, md["map"]["g"], iters=300)
    mode_imgs.append(framed(st, H, W))
pmap = sum(w * m for w, m in zip(wts, maps))

n = len(modes)
fig, ax = plt.subplots(1, 2 + n, figsize=(5.2 * (2 + n), 4.6), facecolor="white")
show(ax[0], Image.open(SM.IMG["sad_naskh"]).convert("RGB").crop((0, 250, 736, 850)), "input: Naskh sad")
for i in range(n):
    show(ax[1 + i], ~mode_imgs[i], f"mode {chr(65 + i)} (weight {wts[i]:.2f}, jeem softIoU {modes[i]['map']['iou']:.2f})", cmap="gray")
show(ax[-1], 1 - pmap, "posterior predictive\n(black = certain ink, grey = uncertain)", cmap="gray")
fig.suptitle("Modern sad predicted from the jeem only (no modern sad used)", fontsize=11)
fig.savefig(f"{OUT}/3_sad_prediction.png", dpi=110, bbox_inches="tight")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), facecolor="white")
show(ax[0], Image.open(SM.IMG["sad_naskh"]).convert("RGB").crop((0, 250, 736, 850)), "input")
show(ax[1], ~mode_imgs[0], "most probable modern sad", cmap="gray")
fig.savefig(f"{OUT}/4_sad_final.png", dpi=130, bbox_inches="tight")

json.dump(summary, open(f"{OUT}/summary.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False, indent=1))
