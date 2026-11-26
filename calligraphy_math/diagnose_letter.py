"""
تشخيص داخلي لتوليد حرف: نطبّق كل مكوّن من المعادلات وحده (ablation) ونقيس أثره.
    python3 diagnose_letter.py <مجلد الصور> <الحرف>
المقاييس:
  openness  = نصف قطر أكبر دائرة داخل عين العقدة ÷ نصف سُمك ضربة العقدة الوسيط   (0 = العين مسدودة)
  loop_dist = المسافة المرنة (SRVF) بين جزء العقدة قبل التحويل وبعده (0 = الشكل نفسه)
  turn_loop = مجموع انعطاف جزء العقدة بالدرجات (حلقة مغلقة ≈ ±360)
"""
import sys, os, json, pickle, math
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

UP, LETTER = sys.argv[1], sys.argv[2]
sys.argv = [sys.argv[0], UP]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine"))
import style_model as SM
import srvf

G = pickle.load(open("results_engine_jeem/glyphs.pkl", "rb"))[LETTER]
strokes = [(c, w) for c, w in G.strokes]
fits = json.load(open("results/style_fits.json"))["fits"]
pA = np.mean([list(f["params"].values()) for f in fits if abs(f["params"]["rotation ρ°"]) < 10], axis=0)

def find_loops(c, w):
    """جزء [i, j] من الضربة يعود إلى نفسه: |γ_i − γ_j| < سُمك وسيط، ومسافة القوس بينهما > 4 أضعافه."""
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
    wm = np.median(w)
    D = np.linalg.norm(c[:, None] - c[None], axis=2)
    best = None
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            if s[j] - s[i] > 4 * wm and D[i, j] < 0.8 * wm:
                if best is None or (s[j] - s[i]) > (s[best[1]] - s[best[0]]):
                    best = (i, j)
    return best

def raster(styled, H=420, W=560, margin=0.1):
    P = np.vstack([g for g, _ in styled]); wm = max(w.max() for _, w in styled)
    lo, hi = P.min(0) - wm / 2, P.max(0) + wm / 2
    ctr, span = (lo + hi) / 2, hi - lo
    k = (1 - 2 * margin) * min(H / span[1], W / span[0])
    img = Image.new("L", (W, H), 0); dr = ImageDraw.Draw(img)
    for g, w in styled:
        X = np.column_stack([(g[:, 0] - ctr[0]) * k + W / 2, -(g[:, 1] - ctr[1]) * k + H / 2])
        r = w * k / 2; n = SM.normals(X)
        L, R = X + r[:, None] * n, X - r[:, None] * n
        for i in range(len(X) - 1):
            dr.polygon([tuple(L[i]), tuple(R[i]), tuple(R[i + 1]), tuple(L[i + 1])], fill=255)
    return np.asarray(img) > 127, k

def metrics(styled, loop):
    m, k = raster(styled)
    holes = ndimage.binary_fill_holes(m) & ~m
    r_hole = ndimage.distance_transform_edt(holes).max() / k if holes.any() else 0.0
    out = {}
    if loop:
        i, j = loop
        c, w = styled[0]
        out["openness"] = round(float(r_hole / (np.median(w[i:j + 1]) / 2)), 3)
        out["loop_dist"] = round(srvf.elastic_distance(strokes[0][0][i:j + 1], c[i:j + 1], n=60, rotate=True), 3)
        d = np.diff(c[i:j + 1], axis=0); ph = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        out["turn_loop"] = round(math.degrees(ph[-1] - ph[0]), 1)
        out["loop_width_ratio"] = round(float(np.median(w[i:j + 1]) / np.median(strokes[0][1][i:j + 1])), 2)
    d = np.diff(styled[0][0][-12:], axis=0); out["tail_dir"] = round(math.degrees(math.atan2(d[:, 1].sum(), d[:, 0].sum())), 1)
    return out, m

def variant(name, keep):
    """keep: أي المكوّنات تُفعَّل. الباقي = هوية."""
    s, rho, a, c, beta, f, S, gam, mm, th = pA
    q = np.array([s if "affine" in keep else 0, rho if "affine" in keep else 0, a if "affine" in keep else 0,
                  c if "curv" in keep else 1, beta if "swash" in keep else 0, f, S, gam, mm, th])
    st = SM.apply_style(strokes, q)
    if "width" not in keep:                        # السُمك الأصلي (بلا قانون القلم)
        st = [(g, w0) for (g, _), (_, w0) in zip(st, strokes)]
    return st

loop = find_loops(*strokes[0])
print("loop segment (indices):", loop)
rows, imgs = {}, {}
for name, keep in [("Naskh (identity)", set()), ("affine only", {"affine"}), ("curvature c only", {"curv"}),
                   ("swash only", {"swash"}), ("width law only", {"width"}), ("v1 full", {"affine", "curv", "swash", "width"})]:
    st = variant(name, keep)
    rows[name], imgs[name] = metrics(st, loop)
    print(f"{name:20s}", rows[name])

# الطبقة الأداة فقط (كالمحرّك عند α=0): تحويل أفيني + قانون السُمك، بلا انحناء ولا ذيل
st = variant("tool", {"affine", "width"}); rows["affine + width (tool only)"], imgs["affine + width (tool only)"] = metrics(st, loop)
print(f"{'affine + width':20s}", rows["affine + width (tool only)"])

# أين يذهب السُمك؟ نسبة السُمك الجديد إلى القديم على طول الضربة
st = variant("w", {"width"})
ratio = st[0][1] / strokes[0][1]
print("width ratio new/old: loop median %.2f | tail median %.2f | overall %.2f" %
      (np.median(ratio[loop[0]:loop[1] + 1]), np.median(ratio[loop[1]:]), np.median(ratio)))
c, w = strokes[0]
print("Naskh widths (×H): loop median %.3f, tail median %.3f, letter mean w̄ %.3f" %
      (np.median(w[loop[0]:loop[1] + 1]), np.median(w[loop[1]:]), np.mean(w)))

os.makedirs(f"results_diag_{LETTER}", exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, len(imgs), figsize=(3.2 * len(imgs), 3.4), facecolor="white")
for a_, (k_, im) in zip(ax, imgs.items()):
    a_.imshow(~im, cmap="gray"); a_.axis("off")
    r = rows[k_]; a_.set_title(f"{k_}\nopen {r.get('openness')} · loop Δ {r.get('loop_dist')}", fontsize=8)
fig.savefig(f"results_diag_{LETTER}/ablation.png", dpi=110, bbox_inches="tight")
json.dump(dict(loop=loop, rows=rows), open(f"results_diag_{LETTER}/ablation.json", "w"), ensure_ascii=False, indent=1)
