"""python -m khatt.bench.hidden_rules RULES_DIR FONT WORD OUT.json

The letters a logo lacks, drawn with the rules of the typefaces nearest the logo. A word set in a typeface
is the logo: its letters are fitted on their vector outlines (bidonaq_fit.fit_word, the vector's measures
kept); a rule the word does not fit takes the median ratio that the five typefaces nearest by their
measures (vector.dna: stem, hairline, ascender, descender to the x-height, slant, tension, corners)
needed for the same rule (letter_rules, run on every typeface into RULES_DIR; the typeface's own family
left out). Compared with the rules as written, on the letters of the 18 skeletons the word lacks."""
import sys, os, json, glob, math, warnings; warnings.filterwarnings('ignore')
import numpy as np
from khatt.bench import fonts
from khatt.vector import word_shapes, dna
from khatt.style import Style
from khatt.typeset import typeset
from khatt.recipes import PARAMS, SKELETON, general_rule
from khatt.bench import bidonaq_fit as E
from khatt.bench.letter_rules import REF

DNA_KEYS = ['stem', 'hair', 'ascender', 'descender', 'slant', 'tension', 'corners']
NEAREST = 5


def dna_vec(face, text):
    d = dna(word_shapes(face, text), text); X = d['x_height']
    return np.array([d.get(k, 0.0) / (X if k in ('stem', 'hair', 'ascender', 'descender') else 1.0) for k in DNA_KEYS], float)


def load_rules(folder):
    out = {}
    for f in glob.glob(os.path.join(folder, '*.json')):
        r = json.load(open(f))
        out[r['font']] = {n: v for L in r['letters'].values() for n, v in L['ratios'].items()}
    return out


def nearest(face, word, rules, files, skip_family):
    others = [f for f in rules if f.rsplit('-', 1)[0] != skip_family]
    me = dna_vec(face, word)
    V = {f: dna_vec(fonts.load(str(files[f])), REF) for f in others}
    sd = np.std(np.array(list(V.values())), 0) + 1e-6
    return sorted(others, key=lambda f: np.abs((V[f] - me) / sd).sum())[:NEAREST]


def with_rules(st, rules, names):
    """The rules the word did not fit, turned by the median ratio of the given typefaces."""
    out = Style(**{**st.__dict__, 'params': dict(st.params)})
    for n in set().union(*(rules[f] for f in names)):
        if n not in st.params and general_rule(st, n) > 0:
            out.params[n] = general_rule(st, n) * float(np.median([rules[f][n] for f in names if n in rules[f]]))
    return out


def hidden(face, word, st, sc):
    have = {SKELETON.get(c, c) for c in word} | set(word)
    vals = {}
    for c in E.SK18:
        if c in have or SKELETON.get(c, c) in have:
            continue
        tr = E.truth_of(word_shapes(face, c), sc); g = E.letters_of(c, st, sc)
        if tr and g:
            vals[c] = E.iou(list(tr.values())[0], list(g.values())[0])
    return vals


def run(rules_dir, name, word):
    files = {f.stem: f for f in fonts.font_files()}
    face = fonts.load(str(files[name]))
    sh = word_shapes(face, word); d = dna(sh, word)
    X = d['x_height']; sc = 70.0 / X
    truth = E.truth_of(sh, sc)
    k = X / Style().x_height
    base = Style(**{**Style().__dict__, **{f: getattr(Style(), f) * k for f in E.LEN}})
    meas = Style(**{**base.__dict__, 'stem': d.get('stem', base.stem), 'hair': max(1.0, d.get('hair', base.hair)), 'x_height': X,
                    'ascender': max(0.2 * X, d.get('ascender', base.ascender)), 'descender': max(0.2 * X, d.get('descender', base.descender)),
                    'white': d.get('stem', base.stem)})
    meas.params = {}
    def score(st):
        got = E.letters_of(word, st, sc)
        v = [E.iou(truth[i], got[i]) for i in truth if i in got]
        return float(np.mean(v)) if v else 0.0
    cache = {}; typeset(word, meas, cache=cache)
    reads = sorted({n for _, deps in cache.values() for n in deps if n in PARAMS})
    st, own = E.fit_word(meas, score, reads)
    rules = load_rules(rules_dir)
    near = nearest(face, word, rules, files, name.rsplit('-', 1)[0])
    written, learned = hidden(face, word, st, sc), hidden(face, word, with_rules(st, rules, near), sc)
    return {'font': name, 'word': word, 'own_word': own, 'nearest': near,
            'hidden_written': float(np.mean(list(written.values()))), 'hidden_nearest': float(np.mean(list(learned.values()))),
            'per_letter': {c: [round(written[c], 3), round(learned[c], 3)] for c in written}}


if __name__ == '__main__':
    r = run(*sys.argv[1:4])
    json.dump(r, open(sys.argv[4], 'w'), ensure_ascii=False)
    print(f"{r['font']:22s} {r['word']:12s} own {r['own_word']:.3f} | lacks: rules as written {r['hidden_written']:.3f}, "
          f"nearest typefaces' rules {r['hidden_nearest']:.3f} {r['nearest'][:3]}")
