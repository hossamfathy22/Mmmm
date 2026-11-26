"""
تشغيل المحرّك:
    python run.py <pdf سكّال> <صورة الحرف الهدف> <اسم الحرف بالإنجليزية> <مجلد المخرجات> [--transfer]
مثال (الحرف الهدف فقط):
    python run.py sakkal.pdf modern_jeem.jpg jeem out/
النقل إلى بقية الحروف لا يحدث إلا بالخيار --transfer.
"""
import sys, os, json, math, pickle
import numpy as np
import torch
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alphabet as A
import engine as E
import geometry as GE
from glyph import Glyph

PDF, TARGET, LETTER, OUT = sys.argv[1:5]
TRANSFER = "--transfer" in sys.argv[5:]
os.makedirs(OUT, exist_ok=True)
log = lambda *x: print(*x, flush=True)

# ---------- 1) الأبجدية من جدول سكّال (مع ذاكرة مؤقتة) ----------
cache = os.path.join(OUT, "glyphs.pkl")
if os.path.exists(cache):
    glyphs = pickle.load(open(cache, "rb"))
else:
    alpha = A.extract_alphabet(PDF)
    unit = float(np.ptp(np.nonzero(alpha["jeem"]["body"])[0]))       # الوحدة: ارتفاع الجيم
    glyphs = {}
    for n in A.NAMES:
        glyphs[n] = Glyph(n, alpha[n]["body"], alpha[n]["dots"], unit=unit)
        log("glyph", n, len(glyphs[n].strokes), "strokes")
    pickle.dump(glyphs, open(cache, "wb"))

# ---------- 2) الحرف الهدف ----------
a = np.asarray(Image.open(TARGET).convert("RGB")).astype(int)
ink = (a.max(2) < 140) & ((a.max(2) - a.min(2)) < 40)
from scipy import ndimage
lab, k = ndimage.label(ink); sz = ndimage.sum(ink, lab, range(1, k + 1))
ink = np.isin(lab, 1 + np.where(sz > 0.01 * sz.max())[0])
cv = E.Canvas(ink, size=150)
cv_eval = E.Canvas(ink, size=300)        # التقييم بضعف الدقة حتى لا يُخفي التنعيم أي انقطاع
src = glyphs[LETTER]
model = E.Model(src)

# ---------- 3) الملاءمة ----------
log("stage 1: general style only")
h1, g, al1 = E.stage1(model, cv, starts=8, iters=500, log=log)
log("stage 2: letter-specific residual")
h2, res, al2 = E.stage2(model, cv, g, al1, log=log)
zero = model.zero_residual()

# ---------- 4) النقل إلى كل الحروف ----------
PX, CW, CH = 120, 3.2, 2.2               # بكسل لكل وحدة، أبعاد اللوحة بالوحدات
def render_letter(strokes, dots, colors=None, ss=3):
    W, H = int(CW * PX), int(CH * PX)
    P = torch.cat([c for c, _ in strokes]).detach().numpy()
    wm = max(float(w.max()) for _, w in strokes)
    pts = [P] + [np.array([p]) for p, _ in dots]
    allp = np.vstack(pts); ctr = (allp.min(0) + allp.max(0)) / 2
    canvas = np.ones((H, W, 3))
    for i, (c, w) in enumerate(strokes):
        cov = E.implicit_image([(c, w)], PX, ctr, W, H)[..., None]
        col = np.array(colors[i] if colors else (15, 15, 15)) / 255.0
        canvas = canvas * (1 - cov) + col * cov
    img = Image.fromarray((canvas * 255).astype(np.uint8)).resize((W * ss, H * ss), Image.LANCZOS)
    dr = ImageDraw.Draw(img)
    tf = lambda q: np.column_stack([(q[:, 0] - ctr[0]) * PX * ss + W * ss / 2, -(q[:, 1] - ctr[1]) * PX * ss + H * ss / 2])
    for p, r in dots:
        x, y = tf(np.array([p]))[0]; rr = r * PX * ss
        dr.polygon([(x, y - rr), (x + rr, y), (x, y + rr), (x - rr, y)], fill=(15, 15, 15) if not colors else (120, 120, 120))
    return img.resize((W, H), Image.LANCZOS)

def conf_color(a):
    return (46, 140, 87) if a >= 0.7 else (214, 162, 32) if a >= 0.2 else (140, 140, 140)

with torch.no_grad():
    p1e, p2e = cv_eval.hard(model.generate(g, zero), al1), cv_eval.hard(model.generate(g, res), al2)
    h1, h2 = E.hard_iou(p1e, cv_eval.Tn), E.hard_iou(p2e, cv_eval.Tn)
    e1, e2 = E.edge_err(p1e, cv_eval.Tn), E.edge_err(p2e, cv_eval.Tn)
log(f"evaluation at 300 px: stage1 IoU {h1:.3f} edge {e1:.2f}% | stage2 IoU {h2:.3f} edge {e2:.2f}%")
CRIT = dict(IoU_min=0.90, edge_err_max_pct=1.5)
report = dict(letter=LETTER, target=os.path.basename(TARGET), stage1_general_IoU=round(h1, 3),
              stage2_with_residual_IoU=round(h2, 3), stage1_edge_err_pct=round(e1, 2), stage2_edge_err_pct=round(e2, 2),
              criteria=CRIT, PASS=bool(h2 >= CRIT["IoU_min"] and e2 <= CRIT["edge_err_max_pct"]), general_style=dict(zip(E.GN, [round(float(x), 4) for x in g])),
              letters={})
naskh_imgs, modern_imgs, conf_imgs = {}, {}, {}
identity = E.GPRIOR[:, 0].clone()
for n in (A.NAMES if TRANSFER else [LETTER]):
    m = E.Model(glyphs[n], wbar=src.wbar)
    if n == LETTER:
        r_n, rep = res, [dict(stroke=k, nearest_target_stroke=k, distance=0.0, alpha=1.0) for k in range(m.n)]
    else:
        r_n, rep = E.transfer_residual(model, res, m)
    with torch.no_grad():
        st = m.generate(g, r_n)
        dots = m.dots(g, st)
        naskh = [(torch.tensor(c), torch.tensor(w)) for c, w in glyphs[n].strokes]
    naskh_imgs[n] = render_letter(naskh, glyphs[n].dots)
    modern_imgs[n] = render_letter(st, dots)
    conf_imgs[n] = render_letter(st, dots, colors=[conf_color(r["alpha"]) for r in rep])
    report["letters"][n] = rep
    log("transferred", n, [r["alpha"] for r in rep])

# ---------- 5) الأشكال ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
def ov(p, t):
    o = np.ones(t.shape + (3,)); o[t & ~p] = (0.86, 0.3, 0.3); o[p & ~t] = (0.3, 0.45, 0.9); o[p & t] = (0.12, 0.12, 0.12)
    return o
fig, ax = plt.subplots(1, 5, figsize=(19, 4), facecolor="white")
ax[0].imshow(naskh_imgs[LETTER]); ax[0].set_title(f"Sakkal {LETTER} (input)")
with torch.no_grad():
    p1, p2 = p1e, p2e
ax[1].imshow(ov(p1, cv_eval.Tn)); ax[1].set_title(f"stage 1: general style only\nIoU {h1:.2f} · edge {e1:.1f}% H")
ax[2].imshow(ov(p2, cv_eval.Tn)); ax[2].set_title(f"stage 2: + letter residual\nIoU {h2:.2f} · edge {e2:.1f}% H")
ax[3].imshow(modern_imgs[LETTER]); ax[3].set_title("result")
ax[4].imshow(Image.open(TARGET).convert("RGB")); ax[4].set_title("target")
for x in ax: x.axis("off")
fig.savefig(os.path.join(OUT, "1_fit.png"), dpi=110, bbox_inches="tight")

SHEETS = (("2_alphabet_modern.png", modern_imgs, "transformed alphabet"),
          ("3_alphabet_confidence.png", conf_imgs, "confidence: green = residual transferred (similar stroke) · yellow = partial · grey = general style only"),
          ("0_alphabet_naskh.png", naskh_imgs, "Sakkal Naskh input"))
for key, imgs, ttl in (SHEETS if TRANSFER else ()):
    fig, axes = plt.subplots(4, 7, figsize=(17, 9), facecolor="white")
    for x, n in zip(axes.flat[::-1], A.NAMES):
        x.imshow(imgs[n]); x.set_title(f"{n}  {A.ARABIC[n]}" if False else n, fontsize=9); x.axis("off")
    fig.suptitle(ttl, fontsize=11)
    fig.savefig(os.path.join(OUT, key), dpi=90, bbox_inches="tight")
json.dump(report, open(os.path.join(OUT, "report.json"), "w"), ensure_ascii=False, indent=1)
torch.save(dict(letter=LETTER, g=g, residual=res, align_stage1=al1, align_stage2=al2), os.path.join(OUT, "params.pt"))
# صورة كبيرة للنتيجة بجانب الهدف
with torch.no_grad():
    st = model.generate(g, res)
    P = torch.cat([c for c, _ in st]).numpy(); ctr = (P.min(0) + P.max(0)) / 2
    span = P.max(0) - P.min(0) + 0.3
    W_, H_ = 700, int(700 * span[1] / span[0])
    big = E.implicit_image(st, (W_ - 80) / span[0], ctr, W_, H_, ss=2)
Image.fromarray((255 * (1 - big)).astype(np.uint8)).save(os.path.join(OUT, "result_large.png"))
log("done", json.dumps({k: report[k] for k in ("stage1_general_IoU", "stage2_with_residual_IoU", "stage1_edge_err_pct", "stage2_edge_err_pct", "PASS")}))
