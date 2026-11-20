"""python -m khatt.bench.free_numbers FONT OUT.json

Bidonaq's constructions carry two kinds of numbers. The named rules (recipes.PARAMS, about 54) are the
ones the engine fits to a logo. Every other number in a builder (letters.py, geometry.py: where a curve's
control point sits, how far a tail drops, how a bowl turns, about 450 of them) is fixed: Sirang's hand,
measured once on Sirang by hadar_word.py and kept as a ratio for every logo.

Here every fixed number (a decimal literal in a builder's body) is made a handle: its value times a
multiplier the fit may change (the code is rewritten in memory, nothing in the package changes). Then each
of the 18 skeletons is fitted to the same letter of a typeface twice: its named rules only, and its named
rules with all its fixed numbers. The difference is what the fixed numbers cost."""
import ast, inspect, math, sys, json, warnings; warnings.filterwarnings('ignore')
import numpy as np

FREE: dict = {}          # handle -> multiplier
USED: set = set()        # handles read since cleared


def _K(key, v):
    USED.add(key)
    return v * FREE.get(key, 1.0)


# number crunching, not drawing: Bezier arithmetic, polygon cleanup, marks (dots are kept as drawn), names
SKIP = {'cubic', 'line', 'arc', '_pts', 'soften', 'fuse', 'clear_marks', '_split_at_baseline', 'family_names',
        'top_joins', 'marks', '_rings', '_named', '_lib', 'circle', 'hamza', 'tatweel', 'dot', 'dash', 'rect',
        'path', 'translate', 'box', 'flat', '_draw', '_e', '_x', 'clip_x', 'Point'}


class _Free(ast.NodeTransformer):
    def __init__(self, fname):
        self.f, self.i = fname, 0

    def visit_Compare(self, node):            # thresholds choose between variants: left as they are
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, float) and node.value != 0.0:
            key = f"{self.f}#{self.i}"; self.i += 1
            return ast.copy_location(ast.Call(ast.Name('_K', ast.Load()), [ast.Constant(key), ast.Constant(node.value)], []), node)
        return node


def free_all():
    """Rewrite every builder of letters.py and geometry.py so that its decimal numbers read _K."""
    from khatt import letters, geometry
    count = {}
    for mod in (geometry, letters):
        mod._K = _K
        tree = ast.parse(inspect.getsource(mod))
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or node.name in SKIP:
                continue
            fn = getattr(mod, node.name, None)
            if not callable(fn) or not hasattr(fn, '__code__') or fn.__code__.co_freevars:
                continue
            node.decorator_list = []
            t = _Free(node.name)
            node.body = [t.visit(b) for b in node.body]
            if not t.i:
                continue
            m = ast.fix_missing_locations(ast.Module([node], []))
            code = compile(m, inspect.getsourcefile(mod), 'exec')
            new = [c for c in code.co_consts if hasattr(c, 'co_name') and c.co_name == node.name][0]
            fn.__code__ = new
            count[node.name] = t.i
    return count


if __name__ == '__main__':
    from scipy.optimize import minimize
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
        row = {'start': round(start, 3), 'rules': len(names), 'numbers': len(keys)}
        x_rules = np.zeros(len(names))
        if names:
            r = minimize(lambda x: -score(np.clip(np.r_[x, np.zeros(len(keys))], -1.2, 1.2), False), x_rules, method='Powell',
                         options={'maxfev': 60 * len(names) + 60, 'xtol': 0.03, 'ftol': 1e-4})
            x_rules = np.clip(r.x, -1.2, 1.2)
        row['named_rules'] = round(score(np.r_[x_rules, np.zeros(len(keys))], False), 3)
        if keys:
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
