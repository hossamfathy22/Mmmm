"""python -m khatt.bench.letter_rules FONT OUT.json

Each of the 18 skeletons' own rules fitted to a typeface's letter (isolated, on its vector outline),
with the typeface's measures (vector.dna) as they are: how far each rule (recipes.PARAMS) is from what
this typeface draws, as a ratio to the rule as written. Over many typefaces the ratios tell what the
rules should be for a letter no logo shows (the letters a logo lacks)."""
import sys, json, math, warnings; warnings.filterwarnings('ignore')
import numpy as np
from scipy.optimize import minimize
from khatt.bench import fonts
from khatt.vector import word_shapes, dna
from khatt.style import Style
from khatt.typeset import typeset
from khatt.recipes import PARAMS, general_rule
from khatt.bench import bidonaq_fit as E

REF = "مستقبل حديقة"

def measured(face):
    d = dna(word_shapes(face, REF), REF)
    X = d['x_height']; k = X / Style().x_height
    st = Style(**{**Style().__dict__, **{f: getattr(Style(), f) * k for f in E.LEN}})
    st = Style(**{**st.__dict__, 'stem': d.get('stem', st.stem), 'hair': max(1.0, d.get('hair', st.hair)), 'x_height': X,
                  'ascender': max(0.2 * X, d.get('ascender', st.ascender)), 'descender': max(0.2 * X, d.get('descender', st.descender)),
                  'white': d.get('stem', st.stem)})
    st.params = {}
    return st, 70.0 / X

def reads_of(c, st):
    cache = {}; typeset(c, st, cache=cache)
    return sorted({n for _, deps in cache.values() for n in deps if n in PARAMS})

def fit_letter(face, c, st, sc):
    tr = list(E.truth_of(word_shapes(face, c), sc).values())[0]
    names = reads_of(c, st)
    v0 = [max(1e-3, general_rule(st, n)) for n in names]
    def build(x):
        s = Style(**{**st.__dict__, 'params': {}})
        s.params = {n: b * math.exp(xi) for n, b, xi in zip(names, v0, x)}
        return s
    def obj(x):
        try:
            g = list(E.letters_of(c, build(np.clip(x, -1.2, 1.2)), sc).values())[0]
        except Exception:
            return 1.0
        return -E.iou(tr, g)
    start = -obj(np.zeros(len(names)))
    if not names:
        return {'start': start, 'fit': start, 'ratios': {}}
    r = minimize(obj, np.zeros(len(names)), method='Powell', options={'maxfev': 60 * len(names) + 60, 'xtol': 0.03, 'ftol': 1e-4})
    x = np.clip(r.x, -1.2, 1.2)
    return {'start': round(start, 3), 'fit': round(-float(r.fun), 3),
            'ratios': {n: round(math.exp(xi), 3) for n, xi in zip(names, x)}}

if __name__ == '__main__':
    name, out = sys.argv[1], sys.argv[2]
    face = fonts.load(str({f.stem: f for f in fonts.font_files()}[name]))
    st, sc = measured(face)
    res = {}
    for c in E.SK18:
        res[c] = fit_letter(face, c, st, sc)
        print(name, c, res[c]['start'], '->', res[c]['fit'], flush=True)
    json.dump({'font': name, 'letters': res}, open(out, 'w'), ensure_ascii=False)
