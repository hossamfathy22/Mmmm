"""
نقل التحويل المتعلَّم من حرف التدريب إلى حرف واحد آخر (بلا إعادة تعلّم):
    python transfer_one.py <مجلد نتائج التدريب> <اسم الحرف الجديد>
الأداة واليد تُطبَّقان كما هما؛ قواعد العناصر والبقية الخاصة تنتقل لكل ضربة بوزن α حسب شبهها بضربات حرف التدريب.
"""
import sys, os, json, pickle
import numpy as np
import torch
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E

RES, NEW = sys.argv[1], sys.argv[2]
glyphs = pickle.load(open(os.path.join(RES, "glyphs.pkl"), "rb"))
P = torch.load(os.path.join(RES, "params.pt"), weights_only=False)
src = glyphs[P["letter"]]; src_model = E.Model(src)
m = E.Model(glyphs[NEW], wbar=src.wbar)
r_new, rep = E.transfer_residual(src_model, P["residual"], m)
OUT = os.path.join(RES, NEW); os.makedirs(OUT, exist_ok=True)

def conf_color(a):
    return (46, 140, 87) if a >= 0.7 else (214, 162, 32) if a >= 0.2 else (140, 140, 140)

def draw(strokes, dots, colors=None, W=700):
    allp = np.vstack([c.detach().numpy() for c, _ in strokes] + [np.array([p]) for p, _ in dots])
    wm = max(float(w.max()) for _, w in strokes)
    lo, hi = allp.min(0) - wm, allp.max(0) + wm
    ctr, span = (lo + hi) / 2, hi - lo
    H = int(W * span[1] / span[0]); px = W / span[0]
    canvas = np.ones((H, W, 3))
    for i, (c, w) in enumerate(strokes):
        cov = E.implicit_image([(c, w)], px, ctr, W, H)[..., None]
        col = np.array(colors[i] if colors else (15, 15, 15)) / 255.0
        canvas = canvas * (1 - cov) + col * cov
    img = Image.fromarray((canvas * 255).astype(np.uint8)); dr = ImageDraw.Draw(img)
    for p, r in dots:
        x, y = (p[0] - ctr[0]) * px + W / 2, -(p[1] - ctr[1]) * px + H / 2; rr = r * px
        dr.polygon([(x, y - rr), (x + rr, y), (x, y + rr), (x - rr, y)], fill=(15, 15, 15))
    return img

with torch.no_grad():
    alpha = [r["alpha"] for r in rep]
    st_general = m.generate(P["g"], m.zero_residual(), alpha)
    st = m.generate(P["g"], r_new, alpha)
    dots = m.dots(P["g"], st)
naskh = [(torch.tensor(c), torch.tensor(w)) for c, w in glyphs[NEW].strokes]
imgs = [("Sakkal Naskh (input)", draw(naskh, glyphs[NEW].dots)),
        ("tool & hand (+ element rules × α)", draw(st_general, m.dots(P["g"], st_general))),
        ("general style + transferred residual", draw(st, dots)),
        ("confidence per stroke", draw(st, dots, colors=[conf_color(r["alpha"]) for r in rep]))]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 4, figsize=(20, 4.6), facecolor="white")
for a, (t, im) in zip(ax, imgs):
    a.imshow(im); a.set_title(t, fontsize=10); a.axis("off")
fig.suptitle(f"{NEW}: transferred from {P['letter']} · " + " · ".join(
    f"stroke {r['stroke']}→{P['letter']} stroke {r['nearest_target_stroke']} α={r['alpha']:.2f}" for r in rep) +
    "   (green ≥0.7 · yellow 0.2–0.7 · grey <0.2 = general style only)", fontsize=10)
fig.savefig(os.path.join(OUT, f"{NEW}_transfer.png"), dpi=100, bbox_inches="tight")
imgs[2][1].save(os.path.join(OUT, f"{NEW}_result_large.png"))
json.dump(dict(letter=NEW, trained_on=P["letter"], strokes=rep), open(os.path.join(OUT, "report.json"), "w"), indent=1)
print(json.dumps(rep, indent=1))
