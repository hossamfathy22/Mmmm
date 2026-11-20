"""python -m khatt.bench.parts_exam PATH TEXT OUT.json

The hidden-letter exam of flexible parts drawn with the logo's pen.

Every letter is a skeleton of parts (stem, tooth, flat stretch, deep bowl, round loop, the tail of ر, the
head of ح, ...), each a centre line with a few handles (height, length, width, depth), drawn with the
logo's nib (qalam). The handles are fitted on each letter the logo has (located by match.locate); a
hidden letter is built with the handles its parts took on the logo's OTHER letters, and compared with
the real one (best shift, IoU, the letters' bodies: dots apart)."""
import sys, json, math, warnings; warnings.filterwarnings('ignore')
import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from khatt.own import own_word, draw, rasterise, body_of
from khatt.typeset import shape_forms

# ------------------------------------------------------------------ curves
def cubic(p0, c1, c2, p1, n=24):
    t = np.linspace(0, 1, n)[:, None]
    p0, c1, c2, p1 = map(np.array, (p0, c1, c2, p1))
    return [tuple(q) for q in ((1-t)**3*p0 + 3*(1-t)**2*t*c1 + 3*(1-t)*t*t*c2 + t**3*p1)]

def ellipse(cx, cy, a, b, n=40):
    return [(cx + a*math.cos(t), cy + b*math.sin(t)) for t in np.linspace(0, 2*math.pi, n)]

# ------------------------------------------------------------------ the parts (handles in pixels)
PARTS = {   # handle: (default as a share of the x-height X)
    'tooth_h': 1.0, 'join': 0.55, 'stem_h': 1.6, 'flat_w': 1.4, 'bowl_w': 1.4, 'bowl_d': 0.5,
    'loop_w': 0.9, 'loop_h': 0.9, 'tail_d': 0.6, 'tail_w': 0.9, 'hah_w': 1.3, 'dal_h': 0.8, 'dal_w': 0.8,
}
TOOTH = set('بتثنيیىپئ'); SEEN = set('سشصض'); ALEF = set('اأإآ'); KAF = set('كکگ')

def skeleton(c, f, P):
    """The centre lines of letter c in form f (right end at x=0, baseline y=0, y up), and the handles used."""
    J, th, sh = P['join'], P['tooth_h'], P['stem_h']
    S, used = [], set()
    def u(*k): used.update(k)
    if c in TOOTH and f in 'IM':
        u('join', 'tooth_h')
        x = 0.0 if f == 'I' else -0.5 * J
        S += [[(x, th), (x, 0)], [(0, 0), (-J if f == 'I' else -J, 0)] if f == 'M' else [(0, 0), (-J, 0)]]
    elif c in TOOTH and c in 'نیيىئ':
        u('bowl_w', 'bowl_d', 'tooth_h')
        W, d = P['bowl_w'], P['bowl_d']; y0 = 0.6 * th if f == 'i' else 0.0
        S += [cubic((0, y0), (0, -d), (-W, -d), (-W, 0.6 * th))]
    elif c in TOOTH:                                  # ب ت ث isolated and final: the flat stretch
        u('flat_w', 'tooth_h')
        W = P['flat_w']; y0 = 0.5 * th if f == 'i' else 0.0
        S += [[(0, y0), (0, 0), (-W, 0), (-W, 0.3 * th)]]
    elif c in SEEN:
        u('join', 'tooth_h')
        x0 = 0.0 if f == 'I' else -0.5 * J
        for k in range(3):
            S.append([(x0 - k * J, th * (1.0 if k == 0 else 0.85)), (x0 - k * J, 0)])
        S.append([(0, 0), (x0 - 3 * J, 0)])
    elif c in ALEF:
        u('stem_h') if f == 'i' else u('stem_h', 'join')
        x = 0.0 if f == 'i' else -0.6 * J
        S.append([(x, sh), (x, 0)])
        if f == 'F': S.append([(0, 0), (x, 0)])
    elif c == 'ل' and f in 'IM':
        u('stem_h', 'join')
        x = 0.0 if f == 'I' else -0.5 * J
        S += [[(x, sh), (x, 0)], [(0, 0), (-J if f == 'I' else x - 0.5 * J, 0)]]
    elif c in KAF:
        u('stem_h', 'flat_w')
        W = P['flat_w']; x = 0.0 if f in 'iI' else -0.3 * W
        S += [[(x, 0), (x - W, 0)], [(x, 0), (x, 0.55 * sh)], [(x - 0.1 * W, 0.55 * sh), (x - 0.8 * W, sh)]]
        if f in 'MF': S.append([(0, 0), (x, 0)])
    elif c in 'دذ':
        u('dal_h', 'dal_w')
        H, W = P['dal_h'], P['dal_w']
        S.append([(-0.3 * W, H), (0, 0.15 * H), (-W, 0)] if f == 'i' else [(0, 0), (-0.25 * W, 0), (-0.25 * W, H)])
        if f == 'F': S.append([(-0.25 * W, 0), (-W, 0)])
    elif c in 'رزژ':
        u('tail_d', 'tail_w', 'tooth_h')
        D, W = P['tail_d'], P['tail_w']; y0 = 0.6 * th if f == 'i' else 0.25 * th
        S.append(cubic((0, y0), (0, -0.3 * D), (-0.4 * W, -D), (-W, -0.9 * D)))
    elif c in 'وؤ':
        u('loop_w', 'loop_h', 'tail_d', 'tail_w')
        a, b = P['loop_w'] / 2, P['loop_h'] / 2
        S.append(ellipse(-a, b, a, b))
        S.append(cubic((0, b), (0, -0.3 * P['tail_d']), (-0.4 * P['tail_w'], -P['tail_d']), (-P['tail_w'], -0.9 * P['tail_d'])))
    elif c in 'فق':
        u('loop_w', 'loop_h', 'flat_w')
        a, b = P['loop_w'] / 2, P['loop_h'] / 2
        S += [ellipse(-a, b, a, b), [(-a, 0), (-a - P['flat_w'], 0)]]
    elif c == 'م':
        u('loop_w', 'loop_h') if f in 'IM' else u('loop_w', 'loop_h', 'tail_d')
        a, b = P['loop_w'] / 2, P['loop_h'] / 2
        x0 = 0.0 if f in 'iI' else -0.3 * P['join']
        S.append(ellipse(x0 - a, b, a, b))
        if f in 'MF': S.append([(0, 0), (x0, 0)])
        if f in 'iF': S.append([(x0 - 2 * a, b), (x0 - 2 * a, -P['tail_d'])])
        else: S.append([(x0 - 2 * a, 0), (x0 - 2 * a - 0.5 * P['join'], 0)])
    elif c in 'هة':
        u('loop_w', 'loop_h')
        a, b = P['loop_w'] / 2, P['loop_h'] / 2
        x0 = 0.0 if f in 'iI' else -0.3 * P['join']
        S.append(ellipse(x0 - a, b, a, b))
        if f in 'MF': S.append([(0, 0), (x0, 0)])
    elif c in 'حجخ':
        u('hah_w', 'tooth_h', 'join')
        W, H = P['hah_w'], P['tooth_h']
        S.append(cubic((-W, H), (-0.2 * W, H), (0.1 * W, 0.4 * H), (-0.5 * W, 0)))
        S.append([(-0.5 * W, 0), (-0.5 * W - P['join'], 0)])
    else:
        return None, set()
    return S, used

# ------------------------------------------------------------------ drawing and scoring
def crop(m):
    r = np.flatnonzero(m.any(1)); c = np.flatnonzero(m.any(0))
    return m[r[0]:r[-1] + 1, c[0]:c[-1] + 1] if len(r) else m

def best_iou(a, b):
    inter = fftconvolve(a.astype(float), b[::-1, ::-1].astype(float), mode='full')
    return float((inter / (a.sum() + b.sum() - inter)).max())

def render(c, f, P, lean, nib):
    S, used = skeleton(c, f, P)
    if not S:
        return None
    S = [[(x + y * lean, y) for x, y in s] for s in S]
    xs = [x for s in S for x, _ in s]; ys = [y for s in S for _, y in s]
    pad = nib.w * 1.5 + 4
    ox, oy = min(xs) - pad, max(ys) + pad
    g = draw([[(x - ox, y - oy) for x, y in s] for s in S], [], nib)
    W = int(max(xs) - min(xs) + 2 * pad) + 2; H = int(max(ys) - min(ys) + 2 * pad) + 2
    return crop(rasterise(g, (H, W))) if g is not None else None

def score(c, f, P, lean, nib, tgt):
    m = render(c, f, P, lean, nib)
    return best_iou(tgt, m) if m is not None and m.any() else 0.0

def fit(c, f, P0, lean0, nib, tgt, keys):
    """The handles (and the lean) that make the letter cover its ink best."""
    keys = sorted(keys)
    x0 = np.r_[np.log([P0[k] for k in keys]), lean0]
    def loss(x):
        P = dict(P0); P.update({k: float(np.exp(v)) for k, v in zip(keys, x[:-1])})
        return -score(c, f, P, float(np.clip(x[-1], -0.7, 0.7)), nib, tgt)
    r = minimize(loss, x0, method='Nelder-Mead', options={'maxiter': 120 * (len(keys) + 1), 'xatol': 0.02, 'fatol': 0.002})
    P = dict(P0); P.update({k: float(np.exp(v)) for k, v in zip(keys, r.x[:-1])})
    return P, float(np.clip(r.x[-1], -0.7, 0.7)), -float(r.fun)

# ------------------------------------------------------------------ the exam
path, text, out = sys.argv[1:4]
ow = own_word(open(path, 'rb').read(), text)
nib = ow['nib']; forms = shape_forms(text)
letters = [o for o in ow['letters'] if skeleton(o.char, forms[o.index][1], {k: 1.0 for k in PARTS})[0]]
bodies = {o.index: crop(body_of(o.mask)) for o in letters}
# the logo's x-height, roughly: the median height of its letters' bodies above their bottom
X = float(np.median([bodies[o.index].shape[0] for o in letters])) * 0.8
P0 = {k: v * X for k, v in PARTS.items()}
# 1. every letter fitted on its own ink: its handles, and how well the template can draw it at all
fitted = {}
for o in letters:
    f = forms[o.index][1]
    _, used = skeleton(o.char, f, P0)
    P, lean, s = fit(o.char, f, P0, 0.0, nib, bodies[o.index], used)
    fitted[o.index] = (P, lean, s, used)
# 2. every letter hidden in turn: built with its parts' handles taken from the other letters
res = []
for o in letters:
    f = forms[o.index][1]
    _, used = fitted[o.index][3], None
    used = fitted[o.index][3]
    P = dict(P0)
    taken = []
    for k in used:
        vals = [fitted[j][0][k] for j in fitted if j != o.index and k in fitted[j][3]]
        if vals:
            P[k] = float(np.median(vals)); taken.append(k)
    leans = [fitted[j][1] for j in fitted if j != o.index]
    lean = float(np.median(leans)) if leans else 0.0
    hid = score(o.char, f, P, lean, nib, bodies[o.index])
    base = score(o.char, f, P0, lean, nib, bodies[o.index])
    res.append({'letter': o.char, 'form': f, 'hidden': round(hid, 3), 'defaults': round(base, 3),
                'self_fit': round(fitted[o.index][2], 3), 'own': round(o.match, 3),
                'from_others': sorted(taken), 'missing': sorted(set(used) - set(taken))})
json.dump(res, open(out, 'w'), ensure_ascii=False)
for r in res:
    print(f"{r['letter']}{r['form']}  hidden {r['hidden']:.2f}  defaults {r['defaults']:.2f}  self-fit {r['self_fit']:.2f}  own {r['own']:.2f}  from {','.join(r['from_others']) or '-'}  missing {','.join(r['missing']) or '-'}")
