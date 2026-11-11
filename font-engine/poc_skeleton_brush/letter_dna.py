#!/usr/bin/env python3
"""The style judge at LETTER level: does a single letter belong to the style
of the letters we were given?

Our real situation: the uploaded word gives 3 letters (noon, zain, meem);
every other letter is generated and must look like the same font.

1. Corpus test (honest, leave-one-font-out):
     for every open font, the 'known' letters are its noon, zain and meem;
     every other letter X of that font is a query; rank the 52 fonts by the
     DNA distance between X and each font's known letters. Real designer
     letters set the bar: how often is their own font ranked first?
     Feature weights are learned without the font being tested.
2. Our font: the same question for every GENERATED letter of
   output/MuznBrush-Regular.otf against the word's real letters, among the
   52 fonts + Muzn. A generated letter that is ranked like a real designer's
   letter "belongs" to the style.
"""
from __future__ import annotations

import json

import numpy as np
from fontTools.ttLib import TTFont
from scipy.stats import spearmanr

import skeleton_brush as sb
import style_metric as sd

KNOWN = "نزم"
QUERY = "ابتدرسصطعفقكلهوى"
SIZE = 140
# shape-independent features only: a letter's own shape decides its edge directions, not its style
FEATS = ["weight", "contrast", "width_cv", "straight", "corners", "taper", "curv_soft", "slant"]


def font_dna(path, letters):
    cm = TTFont(path).getBestCmap()
    out = {}
    for ch in letters:
        if ord(ch) not in cm:
            continue
        m = sd.render(path, ch, SIZE, rtl=True)
        if m.sum() < 50:
            continue
        try:
            d = sd.dna(m, float(SIZE))
            out[ch] = [d[k] for k in FEATS]
        except Exception:
            pass
    return out


def collect():
    cache = sb.OUT / "letter_dna_cache.json"
    data = json.loads(cache.read_text()) if cache.exists() else {}
    if len(data) < 2:
        for p in sorted((sb.HERE / "corpus").glob("*.ttf")):
            d = font_dna(p, KNOWN + QUERY)
            if all(c in d for c in KNOWN) and len(d) >= len(KNOWN) + 8:
                data[p.stem] = d
    # our font changes with every build: always measured fresh
    data["__muzn__"] = font_dna(sb.OUT / "MuznBrush-Regular.otf", KNOWN + QUERY)
    cache.write_text(json.dumps(data))
    return data


def known_mean(d):
    return np.mean([d[c] for c in KNOWN], axis=0)


def weights(data, fonts):
    """Feature weight = how well one letter's value predicts the known letters' mean across fonts."""
    w = []
    for j in range(len(FEATS)):
        a, b = [], []
        for f in fonts:
            K = known_mean(data[f])[j]
            for ch in QUERY:
                if ch in data[f]:
                    a.append(data[f][ch][j])
                    b.append(K)
        w.append(max(spearmanr(a, b)[0], 0) ** 2)
    return np.array(w)


def rank_of(q, refs, own, mu, sd_, w):
    d = {f: float(np.sqrt((((q - r) / sd_) ** 2 * w).sum())) for f, r in refs.items()}
    return 1 + sum(v < d[own] for f, v in d.items() if f != own), d


def main():
    data = collect()
    fonts = [f for f in data if f != "__muzn__"]
    allv = np.array([v for f in fonts for v in data[f].values()])
    corpus_ranks, per_letter = [], {}
    for f in fonts:                                   # leave-one-font-out
        tr = [g for g in fonts if g != f]
        w = weights(data, tr)
        tv = np.array([v for g in tr for v in data[g].values()])
        mu, s = tv.mean(0), tv.std(0) + 1e-9
        refs = {g: known_mean(data[g]) for g in fonts}
        for ch in QUERY:
            if ch in data[f]:
                r, _ = rank_of(np.array(data[f][ch]), refs, f, mu, s, w)
                corpus_ranks.append(r)
                per_letter.setdefault(ch, []).append(r)
    n = len(fonts)
    cr = np.array(corpus_ranks)
    rep = {"fonts": n, "chance_mean_rank": round((n + 1) / 2, 1),
           "real_designer_letters": {"queries": int(len(cr)), "top1": round(float((cr == 1).mean()), 3),
                                     "top5": round(float((cr <= 5).mean()), 3), "median_rank": float(np.median(cr)),
                                     "mean_rank": round(float(cr.mean()), 2)},
           "per_letter_median_rank": {ch: float(np.median(v)) for ch, v in per_letter.items()}}
    # our generated letters against the word's real letters
    w = weights(data, fonts)
    mu, s = allv.mean(0), allv.std(0) + 1e-9
    refs = {g: known_mean(data[g]) for g in fonts}
    refs["__muzn__"] = known_mean(data["__muzn__"])
    gen = {}
    for ch in QUERY:
        if ch in data["__muzn__"]:
            r, d = rank_of(np.array(data["__muzn__"][ch]), refs, "__muzn__", mu, s, w)
            near = sorted(d, key=d.get)[:3]
            gen[ch] = {"rank_of_muzn": r, "nearest_styles": near}
    gr = np.array([v["rank_of_muzn"] for v in gen.values()])
    rep["generated_letters_vs_word"] = {"letters": len(gr), "top1": round(float((gr == 1).mean()), 3),
                                        "top5": round(float((gr <= 5).mean()), 3),
                                        "median_rank": float(np.median(gr)), "mean_rank": round(float(gr.mean()), 2),
                                        "of": n + 1, "per_letter": gen}
    rep["feature_weights"] = dict(zip(FEATS, np.round(w, 3).tolist()))
    (sb.OUT / "letter_dna.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "generated_letters_vs_word"}, ensure_ascii=False, indent=1))
    g = rep["generated_letters_vs_word"]
    print("generated:", {k: g[k] for k in ("top1", "top5", "median_rank", "mean_rank", "of")})
    for ch, v in gen.items():
        print(ch, v)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- the judge inside the search
class StyleJudge:
    """Distance of a candidate letter mask from the style of the uploaded letters, in units
    of how far a REAL designer's letters stray from their own font's noon/zain/meem
    (measured on the corpus). d <= d_real: as consistent as a real letter."""

    def __init__(self, ref_masks, H):
        rep = json.loads((sb.OUT / "letter_dna.json").read_text())
        self.w = np.array([rep["feature_weights"][k] for k in FEATS])
        data = json.loads((sb.OUT / "letter_dna_cache.json").read_text())
        devs = []
        for f, d in data.items():
            if f.startswith("__"):
                continue
            K = known_mean(d)
            devs += [self._dev(np.array(d[ch]), K) for ch in QUERY if ch in d]
        devs = np.array(devs)
        self.sd = devs.std(0) + 1e-9
        self.d_real = float(np.median(self._dist(devs)))
        self.H = H
        self.K = np.mean([self.vec(m) for m in ref_masks], axis=0)

    @staticmethod
    def _dev(x, K):
        d = x - K
        d[0] = x[0] / max(K[0], 1e-6) - 1.0          # weight: relative
        return d

    def _dist(self, dev):
        return np.sqrt(((dev / self.sd) ** 2 * self.w).sum(-1))

    def vec(self, m):
        d = sd.dna(m, self.H)
        return np.array([d[k] for k in FEATS])

    def distance(self, m):
        try:
            return float(self._dist(self._dev(self.vec(m), self.K)) / self.d_real)
        except Exception:
            return 9.0
