"""python -m khatt.bench.free_numbers FONT OUT.json

Bidonaq's constructions carry two kinds of numbers. The named rules (recipes.PARAMS, about 54) are the
ones the engine fits to a logo. Every other number in a builder (letters.py, geometry.py: where a curve's
control point sits, how far a tail drops, how a bowl turns, about 450 of them) is fixed: Sirang's hand,
measured once on Sirang by hadar_word.py and kept as a ratio for every logo.

Here every fixed number (a decimal literal in a builder's body) is made a handle: its value times a
multiplier the fit may change (the code is rewritten in memory, nothing in the package changes). Then each
of the 18 skeletons is fitted to the same letter of a typeface twice: its named rules only, and its named
rules with all its fixed numbers. The difference is what the fixed numbers cost."""
import math, sys, json, warnings; warnings.filterwarnings('ignore')
import numpy as np

# the engine names its numbers itself now (khatt/numbers.py): an examination sets its factors here
from khatt import numbers as _N
FREE = _N.OVERRIDE       # handle -> multiplier
USED = _N.USED           # handles read since cleared
_K = _N._K
SKIP = _N.SKIP


def free_all():
    """Every builder of letters.py and geometry.py reads its decimal numbers through _K (numbers.install)."""
    return dict(_N.install())


def coord(f, x0, idx, rounds=((-0.5, -0.22, 0.22, 0.47), (-0.1, 0.1))):
    """Coordinate search in log space over the entries `idx` of x (maximises f); a cheap stand-in for
    Powell when a table over every typeface is wanted (KHATT_FAST=1)."""
    x = np.array(x0, float); now = f(x)
    for steps in rounds:
        for i in idx:
            best = (now, x[i])
            for d in steps:
                y = x.copy(); y[i] = np.clip(x[i] + d, -1.2, 1.2)
                v = f(y)
                if v > best[0] + 0.002:
                    best = (v, y[i])
            if best[1] != x[i]:
                x[i], now = best[1], best[0]
    return x


if __name__ == '__main__':
    import os
    from scipy.optimize import minimize
    FAST = os.environ.get('KHATT_FAST') == '1'
    from khatt.bench import fonts
    from khatt.bench import bidonaq_fit as E
    from khatt.bench.letter_rules import measured, reads_of
    from khatt.style import Style
    from khatt.recipes import general_rule
    from khatt.vector import word_shapes
    count = free_all()
    name, out = sys.argv[1], sys.argv[2]
    face = fonts.load(str({f.stem: f for f in fonts.font_files()}[name]))
    st, sc = measured(face)
    res = {}
    for c in E.SK18:
        tr = list(E.truth_of(word_shapes(face, c), sc).values())[0]
        names = reads_of(c, st)
        FREE.clear(); USED.clear(); E.letters_of(c, st, sc); keys = sorted(USED)
        v0 = [max(1e-3, general_rule(st, n)) for n in names]
        def score(x, with_keys):
            s = Style(**{**st.__dict__, 'params': {}})
            s.params = {n: b * math.exp(xi) for n, b, xi in zip(names, v0, x[:len(names)])}
            FREE.clear()
            if with_keys:
                FREE.update({k: math.exp(xi) for k, xi in zip(keys, x[len(names):])})
            try:
                return E.iou(tr, list(E.letters_of(c, s, sc).values())[0])
            except Exception:
                return 0.0
        start = score(np.zeros(len(names) + len(keys)), False)
        row = {'start': round(start, 3), 'rules': len(names), 'numbers': len(keys), 'keys': keys}
        x_rules = np.zeros(len(names))
        if names and FAST:
            x_rules = coord(lambda x: score(x, False), np.zeros(len(names) + len(keys)), range(len(names)))[:len(names)]
        elif names:
            r = minimize(lambda x: -score(np.clip(np.r_[x, np.zeros(len(keys))], -1.2, 1.2), False), x_rules, method='Powell',
                         options={'maxfev': 60 * len(names) + 60, 'xtol': 0.03, 'ftol': 1e-4})
            x_rules = np.clip(r.x, -1.2, 1.2)
        row['named_rules'] = round(score(np.r_[x_rules, np.zeros(len(keys))], False), 3)
        if keys and FAST:
            xb = coord(lambda x: score(x, True), np.r_[x_rules, np.zeros(len(keys))], range(len(names) + len(keys)))
            row['with_numbers'] = round(max(row['named_rules'], score(xb, True)), 3)
            row['numbers_fit'] = {k: round(math.exp(v), 3) for k, v in zip(keys, xb[len(names):]) if abs(v) > 0.01}
        elif keys:
            x0 = np.r_[x_rules, np.zeros(len(keys))]
            r = minimize(lambda x: -score(np.clip(x, -1.2, 1.2), True), x0, method='Powell',
                         options={'maxfev': 25 * (len(names) + len(keys)) + 200, 'xtol': 0.03, 'ftol': 1e-4})
            xb = np.clip(r.x, -1.2, 1.2)
            row['with_numbers'] = round(max(row['named_rules'], score(xb, True)), 3)
            row['numbers_fit'] = {k: round(math.exp(v), 3) for k, v in zip(keys, xb[len(names):]) if abs(v) > 0.01}
        else:
            row['with_numbers'] = row['named_rules']
        res[c] = row
        print(name, c, row['start'], '-> rules', row['named_rules'], '-> +numbers', row['with_numbers'], f"({row['rules']} rules, {row['numbers']} numbers)", flush=True)
    FREE.clear()
    json.dump({'font': name, 'freed': count, 'letters': res}, open(out, 'w'), ensure_ascii=False)
