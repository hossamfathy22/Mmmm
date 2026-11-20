"""python -m khatt.bench.bidonaq_fit FONT WORD OUT.json

Can Bidonaq's constructions (the old engine's letters), fitted to a logo's own letters on their exact
vector outlines, draw the letters the logo lacks? A word set in a typeface is the logo; the letters it
does not have are drawn and compared with the typeface's own."""
import sys, json, math, time, warnings; warnings.filterwarnings('ignore')
import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import fftconvolve
from khatt.bench import fonts
from khatt.vector import word_shapes, dna, raster
from khatt.style import Style
from khatt.typeset import typeset, flat
from khatt.recipes import PARAMS, rule_value, SKELETON

def crop(m):
    r = np.flatnonzero(m.any(1)); c = np.flatnonzero(m.any(0))
    return m[r[0]:r[-1] + 1, c[0]:c[-1] + 1] if len(r) else m

def iou(a, b):
    if not a.any() or not b.any(): return 0.0
    inter = fftconvolve(a.astype(float), b[::-1, ::-1].astype(float), mode='full')
    return float((inter / (a.sum() + b.sum() - inter)).max())

def ras_engine(glyph_shapes, dx, sc):
    pts = [(x + dx, y) for sh in glyph_shapes for poly in sh for x, y in poly]
    if not pts: return np.zeros((1, 1), bool)
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0, y1 = min(xs), max(ys)
    W = int((max(xs) - x0) * sc) + 6; H = int((y1 - min(ys)) * sc) + 6
    out = np.zeros((H, W), bool)
    for sh in glyph_shapes:
        acc = np.zeros((H, W), bool)
        for poly in sh:
            if len(poly) < 3: continue
            im = Image.new('L', (W, H), 0)
            ImageDraw.Draw(im).polygon([((x + dx - x0) * sc + 3, (y1 - y) * sc + 3) for x, y in poly], fill=1)
            acc ^= np.asarray(im, bool)
        out |= acc
    return crop(out)

def letters_of(word, st, sc):
    lay = typeset(word, st)
    idx = [i for i, c in enumerate(word) if not c.isspace()]
    out = {}
    for j, p in enumerate(lay.placed):
        if j < len(idx):
            out[idx[j]] = ras_engine(list(flat(p.glyph.shapes)), p.x, sc)
    return out

def truth_of(shapes, sc):
    out = {}
    for i in sorted({s.index for s in shapes}):
        m, _ = raster([s for s in shapes if s.index == i], sc, pad=3)
        out[i] = crop(m)
    return out

LEN = ['stem', 'hair', 'x_height', 'ascender', 'descender', 'white', 'dot', 'overshoot', 'seen_drop']
RATIO = ['join_in', 'join_out', 'ring_n', 'contrast_power', 'tail_reach', 'top_round', 'corner_r', 'angular']
SK18 = "ا ب ح د ر س ص ط ع ف ق ك ل م ن ه و ى".split()

MEASURED = ['stem', 'hair', 'x_height', 'ascender', 'descender']   # read from the vector: within BAND
BAND = 0.15


def fit_word(meas, score, reads):
    """Coordinate search, two rounds, over the style's lengths and ratios and the rules the word reads.
    What the vector measures exactly stays within BAND of its measure; a value that is 0 (a switch that is
    off, a corner not rounded) stays 0: multiplying it does nothing, and a jump from 0 to a length turned
    every letter square (angular is a switch, not a length)."""
    st = Style(**meas.__dict__); st.params = {}
    now = score(st)
    for rnd in range(2):
        for f in LEN + RATIO + reads:
            cur = (st.params.get(f, rule_value(st, f)) if f in PARAMS else getattr(st, f))
            if f not in PARAMS and not isinstance(cur, (int, float)): continue
            if not cur: continue
            best = (now, cur)
            if f in MEASURED:
                steps = (0.87, 0.93, 1.07, 1.15) if rnd == 0 else (0.96, 1.04)
            else:
                steps = (0.6, 0.75, 0.9, 1.1, 1.25, 1.5) if rnd == 0 else (0.85, 0.95, 1.05, 1.15)
            for m in steps:
                v = cur * m
                if f in MEASURED:
                    m0 = getattr(meas, f)
                    v = min(max(v, (1 - BAND) * m0), (1 + BAND) * m0)
                trial = Style(**{**st.__dict__, 'params': dict(st.params)})
                if f in PARAMS: trial.params[f] = v
                else: setattr(trial, f, v)
                try: s = score(trial)
                except Exception: continue
                if s > best[0] + 0.002: best = (s, v)
            if best[1] != cur:
                if f in PARAMS: st.params[f] = best[1]
                else: setattr(st, f, best[1])
                now = best[0]
    return st, now


def run(name, word):
    face = fonts.load(str({f.stem: f for f in fonts.font_files()}[name]))
    sh = word_shapes(face, word)
    d = dna(sh, word)
    X = d['x_height']; sc = 70.0 / X
    truth = truth_of(sh, sc)
    # Bidonaq as it is, scaled to this x-height
    k = X / Style().x_height
    base = Style(**{**Style().__dict__, **{f: getattr(Style(), f) * k for f in LEN}})
    # measured from the vector drawing (the rest Bidonaq's, scaled)
    meas = Style(**{**base.__dict__, 'stem': d.get('stem', base.stem), 'hair': max(1.0, d.get('hair', base.hair)),
                    'x_height': X, 'ascender': max(0.2 * X, d.get('ascender', base.ascender)),
                    'descender': max(0.2 * X, d.get('descender', base.descender)), 'white': d.get('stem', base.stem)})
    def score(st):
        got = letters_of(word, st, sc)
        v = [iou(truth[i], got[i]) for i in truth if i in got]
        return float(np.mean(v)) if v else 0.0
    # the parameters the word's letters read
    cache = {}
    typeset(word, meas, cache=cache)
    reads = sorted({dname for _, deps in cache.values() for dname in deps if dname in PARAMS})
    t0 = time.time()
    st, now = fit_word(meas, score, reads)
    fit_t = time.time() - t0
    # the letters the word does not have (by skeleton), isolated, against the typeface's own
    have = {SKELETON.get(c, c) for c in word} | set(word)
    test = [c for c in SK18 if c not in have and SKELETON.get(c, c) not in have]
    res = {}
    for label, s in (('bidonaq', base), ('measured', meas), ('fitted', st)):
        vals = []
        for c in test:
            tr = truth_of(word_shapes(face, c), sc)
            g = letters_of(c, s, sc)
            if tr and g: vals.append(iou(list(tr.values())[0], list(g.values())[0]))
        res[label] = (float(np.mean(vals)) if vals else None, vals)
    own = {'bidonaq': score(base), 'measured': score(meas), 'fitted': now}
    return {'font': name, 'word': word, 'test': test, 'own_word': own, 'hidden': {k: v[0] for k, v in res.items()},
            'per_letter': {k: [round(x, 2) for x in v[1]] for k, v in res.items()}, 'fit_seconds': round(fit_t), 'params_fitted': len(st.params)}

if __name__ == '__main__':
    r = run(sys.argv[1], sys.argv[2])
    json.dump(r, open(sys.argv[3], 'w'), ensure_ascii=False)
    print(r['font'], '| logo word: bidonaq %.2f measured %.2f fitted %.2f' % tuple(r['own_word'][k] for k in ('bidonaq', 'measured', 'fitted')),
          '| letters it lacks (%d): bidonaq %.2f measured %.2f fitted %.2f' % ((len(r['test']),) + tuple(r['hidden'][k] or 0 for k in ('bidonaq', 'measured', 'fitted'))), '|', r['fit_seconds'], 's')
