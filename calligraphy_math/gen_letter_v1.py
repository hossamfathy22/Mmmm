"""
توليد أي حرف من جدول سكّال بمعادلات النسخة v1 (معاملات الحل A المتعلَّمة من الجيم) — بلا إعادة تعلّم.
    python3 gen_letter_v1.py <مجلد الصور> <اسم الحرف> [مجلد المخرجات]
الهيكل والسُمك من المستخرِج الآلي للمحرّك (engine/glyph.py)، والتحويل من style_model.apply_style (v1 كما هي).
"""
import sys, os, json, pickle
import numpy as np
from PIL import Image, ImageFilter

UP, LETTER = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else f"results_v1_letters/{LETTER}"
sys.argv = [sys.argv[0], UP]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine"))
import style_model as SM

FITS = "results/style_fits.json"          # نتائج v1 (commit 8212ee8)
fits = json.load(open(FITS))["fits"]
A = [f for f in fits if abs(f["params"]["rotation ρ°"]) < 10]
p = np.mean([list(f["params"].values()) for f in A], axis=0)          # متوسط الحل A

glyphs = pickle.load(open("results_engine_jeem/glyphs.pkl", "rb"))
G = glyphs[LETTER]
strokes = [(c, w) for c, w in G.strokes]

def render(styled, dots=(), H=420, W=560, margin=0.1, ss=4):
    P = np.vstack([g for g, _ in styled] + [np.array([d]) for d, _ in dots]); wm = max(w.max() for _, w in styled)
    lo, hi = P.min(0) - wm / 2, P.max(0) + wm / 2
    ctr, span = (lo + hi) / 2, hi - lo
    k = (1 - 2 * margin) * min(H / span[1], W / span[0])
    from PIL import ImageDraw
    img = Image.new("L", (W * ss, H * ss), 0); dr = ImageDraw.Draw(img)
    tf = lambda q: np.column_stack([(q[:, 0] - ctr[0]) * k * ss + W * ss / 2, -(q[:, 1] - ctr[1]) * k * ss + H * ss / 2])
    for g, w in styled:
        X = tf(g); r = w * k * ss / 2; n = SM.normals(X)
        L, R = X + r[:, None] * n, X - r[:, None] * n
        for i in range(len(X) - 1):
            dr.polygon([tuple(L[i]), tuple(R[i]), tuple(R[i + 1]), tuple(L[i + 1])], fill=255)
    for d, rr in dots:
        x, y = tf(np.array([d]))[0]; rr = rr * k * ss
        dr.polygon([(x, y - rr), (x + rr, y), (x, y + rr), (x - rr, y)], fill=255)
    img = img.filter(ImageFilter.GaussianBlur(ss * 0.4)).resize((W, H), Image.LANCZOS)
    return 255 - np.asarray(img)

os.makedirs(OUT, exist_ok=True)
styled = SM.apply_style(strokes, p)
Image.fromarray(render(strokes, G.dots).astype(np.uint8)).save(f"{OUT}/{LETTER}_naskh.png")
Image.fromarray(render(styled).astype(np.uint8)).save(f"{OUT}/{LETTER}_v1.png")
json.dump(dict(letter=LETTER, params_modeA_mean=dict(zip(SM.NAMES, np.round(p, 3).tolist())),
               strokes=len(strokes)), open(f"{OUT}/info.json", "w"), ensure_ascii=False, indent=1)
print("saved", OUT)
