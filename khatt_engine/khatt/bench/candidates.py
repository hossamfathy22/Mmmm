"""python -m khatt.bench.candidates PATH TEXT OUT.json

The hidden-letter exam of candidates. For each letter of the logo's word, hidden in turn:
candidates are that letter's skeleton in 31 open typefaces, scaled by the analogy measured on the logo's
other letters (logo size / typeface size, of the same family where there is one), and drawn with the
logo's own pen. The typefaces are ranked by how well they redraw the logo's OTHER letters of the same
family (never the hidden one). Scored against the logo's hidden letter (best shift, IoU)."""
import sys, json, warnings; warnings.filterwarnings('ignore')
import numpy as np
from PIL import Image
from scipy.signal import fftconvolve
from khatt.bench import fonts
from khatt.match import ref_labels, _rows
from khatt.own import own_word, read_letter, draw, rasterise, body_of
from khatt.typeset import shape_forms

def family(c, f):
    if c in 'اأإآ' or c in 'كکگطظ' or (c == 'ل' and f in 'IM') or c in 'دذ': return 'stem'
    if c in 'بتثنيیىپئسشصض': return 'tooth' if f in 'IM' else 'bowl'
    if c == 'ل': return 'bowl'
    if c in 'فقموؤهة': return 'loop'
    if c in 'جحخعغچ': return 'arc'
    if c in 'رزژ': return 'tail'
    return 'other'

def crop(m):
    r = np.flatnonzero(m.any(1)); c = np.flatnonzero(m.any(0))
    return m[r[0]:r[-1] + 1, c[0]:c[-1] + 1] if len(r) else m

def best_iou(a, b):
    inter = fftconvolve(a.astype(float), b[::-1, ::-1].astype(float), mode='full')
    return float((inter / (a.sum() + b.sum() - inter)).max())

def size(m):
    m = crop(body_of(m)); return m.shape[1], m.shape[0]

path, text, out = sys.argv[1:4]
data = open(path, 'rb').read()
ow = own_word(data, text)
nib = ow['nib']; word = ow['line']
forms = shape_forms(text)
logo = {o.index: o for o in ow['letters']}
idx = sorted(logo)
H = _rows(word)[1] - _rows(word)[0]
files = {f.stem: f for f in fonts.font_files()}
faces = {n.rsplit('-', 1)[0]: fonts.load(str(files[n])) for n in sorted(files) if n.endswith('-400')}

def pen_letter(m):
    s, d = read_letter(m, nib)
    g = draw(s, d, nib)
    return crop(rasterise(g, m.shape)) if g is not None else None

# every typeface: each letter of the word, its mask at the logo's line height
cand = {}
for name, face in faces.items():
    try:
        lab, _ = ref_labels(face, text, H)
    except Exception:
        continue
    cand[name] = {i: crop(lab == i + 1) for i in idx if (lab == i + 1).any()}

res = []
for h in idx:
    c, f = text[h], forms[h][1]
    fam = family(c, f)
    others = [i for i in idx if i != h]
    same = [i for i in others if family(text[i], forms[i][1]) == fam] or others
    tgt = crop(logo[h].mask)
    scores = {}
    for name, L in cand.items():
        if h not in L or not all(i in L for i in same):
            continue
        # the analogy: how much larger the logo draws the hidden letter's family than this typeface
        rx = np.median([size(logo[i].mask)[0] / max(1, size(L[i])[0]) for i in same])
        ry = np.median([size(logo[i].mask)[1] / max(1, size(L[i])[1]) for i in same])
        # how well this typeface, with the logo's pen, redraws the logo's other letters of the family
        fit = []
        for i in same:
            m = L[i]; m2 = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((max(1, int(m.shape[1] * rx)), max(1, int(m.shape[0] * ry))))) > 127
            p = pen_letter(np.pad(m2, 20))
            if p is not None: fit.append(best_iou(crop(logo[i].mask), p))
        m = L[h]; m2 = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((max(1, int(m.shape[1] * rx)), max(1, int(m.shape[0] * ry))))) > 127
        p = pen_letter(np.pad(m2, 20))
        if p is None or not fit: continue
        scores[name] = (float(np.mean(fit)), best_iou(tgt, p))
    if not scores: continue
    ranked = sorted(scores, key=lambda n: -scores[n][0])
    got = [scores[n][1] for n in ranked]
    oracle = max(got)
    res.append({'letter': c, 'form': f, 'family': fam, 'top1': got[0], 'top3': max(got[:3]), 'oracle': oracle,
                'top1_font': ranked[0], 'oracle_font': max(scores, key=lambda n: scores[n][1]),
                'rank_of_best': ranked.index(max(scores, key=lambda n: scores[n][1])) + 1, 'n': len(ranked),
                'own': logo[h].match})
json.dump(res, open(out, 'w'), ensure_ascii=False)
for r in res:
    print(f"{r['letter']}{r['form']} {r['family']:5s} top1 {r['top1']:.2f} ({r['top1_font']})  top3 {r['top3']:.2f}  best {r['oracle']:.2f} ({r['oracle_font']}, rank {r['rank_of_best']}/{r['n']})  own {r['own']:.2f}")
