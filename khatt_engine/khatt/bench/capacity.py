"""python -m khatt.bench.capacity FONT OUT.json

Capacity of each of Bidonaq's constructions: fitted freely (its style values and its own rules) to the
same letter, isolated, in a typeface. How close can it come at all?"""
import sys, json, math, warnings; warnings.filterwarnings('ignore')
import numpy as np
from scipy.optimize import minimize
from khatt.bench import fonts
from khatt.vector import word_shapes, dna
from khatt.style import Style
from khatt.typeset import typeset
from khatt.recipes import PARAMS, rule_value
from khatt.bench import bidonaq_fit as E

name, out = sys.argv[1], sys.argv[2]
face = fonts.load(str({f.stem: f for f in fonts.font_files()}[name]))
ref = word_shapes(face, "مستقبل حديقة")
d = dna(ref, "مستقبل حديقة")
X = d['x_height']; sc = 70.0 / X
k = X / Style().x_height
meas = Style(**{**Style().__dict__, **{f: getattr(Style(), f) * k for f in E.LEN}})
meas = Style(**{**meas.__dict__, 'stem': d.get('stem', meas.stem), 'hair': max(1.0, d.get('hair', meas.hair)), 'x_height': X,
                'ascender': max(0.2 * X, d.get('ascender', meas.ascender)), 'descender': max(0.2 * X, d.get('descender', meas.descender)),
                'white': d.get('stem', meas.stem)})
meas.params = {}
res = {}
for c in E.SK18:
    tr = list(E.truth_of(word_shapes(face, c), sc).values())[0]
    cache = {}; typeset(c, meas, cache=cache)
    reads = sorted({n for _, deps in cache.values() for n in deps if n in PARAMS})
    fields = [f for f in E.LEN + E.RATIO if isinstance(getattr(meas, f), (int, float)) and getattr(meas, f) > 0]
    names = fields + reads
    v0 = [getattr(meas, f) for f in fields] + [max(1e-3, rule_value(meas, n)) for n in reads]
    def build(x):
        st = Style(**{**meas.__dict__, 'params': {}})
        for nm, b, xi in zip(names, v0, x):
            v = b * math.exp(xi)
            if nm in PARAMS: st.params[nm] = v
            else: setattr(st, nm, v)
        return st
    def obj(x):
        x = np.clip(x, -1.2, 1.2)
        try:
            g = list(E.letters_of(c, build(x), sc).values())[0]
        except Exception:
            return 1.0
        return -E.iou(tr, g)
    start = -obj(np.zeros(len(names)))
    r = minimize(obj, np.zeros(len(names)), method='Powell', options={'maxfev': 700, 'xtol': 0.03, 'ftol': 1e-4})
    res[c] = (round(start, 3), round(-r.fun, 3))
    print(name, c, res[c], flush=True)
json.dump({'font': name, 'letters': res}, open(out, 'w'), ensure_ascii=False)
