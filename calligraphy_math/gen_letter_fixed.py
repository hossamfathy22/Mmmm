"""
توليد حرف من جدول سكّال بمعادلات v1 بعد الإصلاح (letter_rules.py)، مع مقارنة v1 الأصلية والقياسات.
    python3 gen_letter_fixed.py <مجلد الصور> <الحرف>
"""
import sys, os, json, pickle, math
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
UP, LETTER = sys.argv[1], sys.argv[2]
sys.argv = [sys.argv[0], UP]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine"))
import style_model as SM
import letter_rules as LR
import srvf

fits = json.load(open("results/style_fits.json"))["fits"]
p = np.mean([list(f["params"].values()) for f in fits if abs(f["params"]["rotation ρ°"]) < 10], axis=0)
G = pickle.load(open("results_engine_jeem/glyphs.pkl", "rb"))[LETTER]
naskh = [(c, w) for c, w in G.strokes]
loops = [LR.find_loop(c, w) for c, w in naskh]

v1 = SM.apply_style(naskh, p)
fx = LR.apply_style_fixed(naskh, p, loops)
fx, counter_info = LR.preserve_counters(naskh, fx, loops)

def measure(st):
    out = {}
    for k, lp in enumerate(loops):
        if lp is None:
            continue
        i, j = lp
        out["openness"] = round(LR.openness(st, k, lp), 3)
        out["loop_shape_change_rad"] = round(srvf.elastic_distance(naskh[k][0][i:j + 1], st[k][0][i:j + 1], n=60, rotate=True), 3)
    d = np.diff(st[-1][0][-12:], axis=0)
    out["tail_direction_deg"] = round(math.degrees(math.atan2(d[:, 1].sum(), d[:, 0].sum())), 1)
    return out

def render(styled, H=420, W=560, margin=0.1, ss=4):
    P = np.vstack([g for g, _ in styled]); wm = max(w.max() for _, w in styled)
    lo, hi = P.min(0) - wm / 2, P.max(0) + wm / 2
    ctr, span = (lo + hi) / 2, hi - lo
    k = (1 - 2 * margin) * min(H / span[1], W / span[0])
    img = Image.new("L", (W * ss, H * ss), 0); dr = ImageDraw.Draw(img)
    for g, w in styled:
        X = np.column_stack([(g[:, 0] - ctr[0]) * k * ss + W * ss / 2, -(g[:, 1] - ctr[1]) * k * ss + H * ss / 2])
        r = w * k * ss / 2; n = SM.normals(X)
        L, R = X + r[:, None] * n, X - r[:, None] * n
        for i in range(len(X) - 1):
            dr.polygon([tuple(L[i]), tuple(R[i]), tuple(R[i + 1]), tuple(L[i + 1])], fill=255)
    return 255 - np.asarray(img.filter(ImageFilter.GaussianBlur(ss * 0.4)).resize((W, H), Image.LANCZOS))

res = {"Naskh": measure(naskh), "v1": measure(v1), "fixed": measure(fx), "counter_rule": counter_info}
print(json.dumps(res, ensure_ascii=False, indent=1))
OUT = f"results_fixed_letters/{LETTER}"; os.makedirs(OUT, exist_ok=True)
json.dump(res, open(f"{OUT}/report.json", "w"), ensure_ascii=False, indent=1)
imgs = [("Sakkal Naskh", render(naskh)), ("v1 (before fix)", render(v1)), ("v1 after fixes F1–F3", render(fx))]
Image.fromarray(imgs[2][1].astype(np.uint8)).save(f"{OUT}/{LETTER}_fixed_large.png")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), facecolor="white")
for a_, (t, im), key in zip(ax, imgs, ["Naskh", "v1", "fixed"]):
    a_.imshow(im, cmap="gray", vmin=0, vmax=255); a_.axis("off")
    r = res[key]
    a_.set_title(f"{t}\nopenness {r.get('openness')} · loop change {r.get('loop_shape_change_rad', 0)} · tail {r['tail_direction_deg']}°", fontsize=9)
fig.savefig(f"{OUT}/{LETTER}_compare.png", dpi=110, bbox_inches="tight")
