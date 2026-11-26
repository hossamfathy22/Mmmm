"""python -m khatt.bench.free_numbers FONT OUT.json

Bidonaq's constructions carry two kinds of numbers: the named rules (recipes.PARAMS, about 54), and every
other number of a builder (letters.py, geometry.py: where a curve's control point sits, how far a tail
drops, how a bowl turns: about 350), once Sirang's hand as hadar_word.py measured it, now handles too
(recipes.N, style.numbers: a factor on Bidonaq's number).

Each of the 18 skeletons is fitted to the same letter of a typeface (isolated, its vector outline, the
typeface's measures as vector.dna reads them) twice: its named rules only, then its named rules and all
its numbers. The difference is what the fixed numbers cost; the fitted values, over many typefaces, are
what the letters a logo lacks take from the typefaces nearest the logo (hidden_rules)."""
import math, sys, json, warnings; warnings.filterwarnings('ignore')
import numpy as np
from scipy.optimize import minimize
from khatt.bench import fonts
from khatt.bench import bidonaq_fit as E
from khatt.bench.letter_rules import measured
from khatt.style import Style
from khatt.recipes import PARAMS, general_rule
from khatt.typeset import typeset
from khatt.vector import word_shapes


def reads_of(c, st):
    cache = {}; typeset(c, st, cache=cache)
    deps = {n for _, d in cache.values() for n in d}
    return sorted(n for n in deps if n in PARAMS), sorted(n[2:] for n in deps if n.startswith("n:"))


if __name__ == '__main__':
    name, out = sys.argv[1], sys.argv[2]
    face = fonts.load(str({f.stem: f for f in fonts.font_files()}[name]))
    st, sc = measured(face)
    res = {}
    for c in E.SK18:
        tr = list(E.truth_of(word_shapes(face, c), sc).values())[0]
        names, keys = reads_of(c, st)
        v0 = [max(1e-3, general_rule(st, n)) for n in names]
        def score(x, with_keys):
            s = Style(**{**st.__dict__, 'params': {n: b * math.exp(xi) for n, b, xi in zip(names, v0, x[:len(names)])},
                         'numbers': {k: math.exp(xi) for k, xi in zip(keys, x[len(names):])} if with_keys else {}})
            try:
                return E.iou(tr, list(E.letters_of(c, s, sc).values())[0])
            except Exception:
                return 0.0
        start = score(np.zeros(len(names) + len(keys)), False)
        row = {'start': round(start, 3), 'rules': len(names), 'numbers': len(keys)}
        x_rules = np.zeros(len(names))
        if names:
            r = minimize(lambda x: -score(np.clip(np.r_[x, np.zeros(len(keys))], -1.2, 1.2), False), x_rules, method='Powell',
                         options={'maxfev': 60 * len(names) + 60, 'xtol': 0.03, 'ftol': 1e-4})
            x_rules = np.clip(r.x, -1.2, 1.2)
        row['named_rules'] = round(score(np.r_[x_rules, np.zeros(len(keys))], False), 3)
        xb = np.r_[x_rules, np.zeros(len(keys))]
        if keys:
            r = minimize(lambda x: -score(np.clip(x, -1.2, 1.2), True), xb, method='Powell',
                         options={'maxfev': 25 * (len(names) + len(keys)) + 200, 'xtol': 0.03, 'ftol': 1e-4})
            if -r.fun > row['named_rules']:
                xb = np.clip(r.x, -1.2, 1.2)
        row['with_numbers'] = round(max(row['named_rules'], score(xb, True)), 3)
        row['ratios'] = {n: round(math.exp(v), 3) for n, v in zip(names, xb[:len(names)])}
        row['numbers_fit'] = {k: round(math.exp(v), 3) for k, v in zip(keys, xb[len(names):])}
        res[c] = row
        print(name, c, row['start'], '-> rules', row['named_rules'], '-> +numbers', row['with_numbers'], f"({row['rules']} rules, {row['numbers']} numbers)", flush=True)
    json.dump({'font': name, 'letters': res}, open(out, 'w'), ensure_ascii=False)
