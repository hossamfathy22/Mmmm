#!/usr/bin/env python3
"""Learn the 'dance' of the pen path from 49 typefaces.

Suveeranont & Igarashi (2010): a new style is a weighted blend of template
fonts; the weights are fitted on the glyphs you have and reused for the ones
you do not have. Here the blend is done on the SKELETON (single smooth pen
path, 64 corresponding points, nuqta units, baseline origin); the SKIN comes
from the writer's own brush (art brush / writer brush), so the result is the
user's hand on a pen path that obeys how real typefaces relate their letters.

Benchmarks (honest, nothing is seen twice):
  corpus   leave-one-font-out: from a font's meem/zain/noon only, predict its
           other letters' pen paths; mean point error in nuqtas vs
           (a) the nearest single font, (b) the average font.
  user     leave-one-letter-out on the uploaded word: fit on two of
           meem/zain/noon, predict the third, compare with the real one.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.optimize import nnls

import skeleton_brush as sb

KNOWN = ["meem.init", "zain.fina", "noon.isol"]
TARGETS = ["beh.isol", "dal.isol", "reh.isol", "alef.isol", "lam.isol", "alefmaksura.isol", "seen.isol"]


def load():
    data = json.loads((sb.OUT / "skeleton_corpus.json").read_text())
    return {f: {k: np.array(v["path"]) for k, v in d.items() if isinstance(v, dict) and "path" in v} for f, d in data.items()}


def fit_weights(templates, X, known, lam=0.05):
    """Convex blend weights (w >= 0, sum 1) of template fonts matching X."""
    fonts = [f for f in templates if all(k in templates[f] for k in known)]
    A = np.stack([np.concatenate([templates[f][k].ravel() for k in known]) for f in fonts], 1)
    b = np.concatenate([X[k].ravel() for k in known])
    big = 50.0
    A2 = np.vstack([A, big * np.ones((1, A.shape[1])), np.sqrt(lam) * np.eye(A.shape[1])])
    b2 = np.r_[b, big, np.zeros(A.shape[1])]
    w, _ = nnls(A2, b2)
    w /= w.sum() + 1e-12
    return dict(zip(fonts, w))


def predict(templates, w, target):
    tot, acc = 0.0, 0.0
    for f, wf in w.items():
        if target in templates[f] and wf > 1e-6:
            acc = acc + wf * templates[f][target]
            tot += wf
    return acc / tot if tot > 0 else None


def path_err(P, Q):
    return float(min(np.linalg.norm(P - Q, axis=1).mean(), np.linalg.norm(P[::-1] - Q, axis=1).mean()))


def benchmark(data):
    rows = {t: {"blend": [], "nearest": [], "average": []} for t in TARGETS}
    for F in data:
        if not all(k in data[F] for k in KNOWN):
            continue
        templ = {f: d for f, d in data.items() if f != F}
        w = fit_weights(templ, data[F], KNOWN)
        dist = {f: sum(np.linalg.norm(templ[f][k] - data[F][k], axis=1).mean() for k in KNOWN)
                for f in templ if all(k in templ[f] for k in KNOWN)}
        near = min(dist, key=dist.get)
        for t in TARGETS:
            if t not in data[F]:
                continue
            pb = predict(templ, w, t)
            if pb is None:
                continue
            rows[t]["blend"].append(path_err(pb, data[F][t]))
            if t in templ[near]:
                rows[t]["nearest"].append(path_err(templ[near][t], data[F][t]))
            avg = np.mean([templ[f][t] for f in templ if t in templ[f]], axis=0)
            rows[t]["average"].append(path_err(avg, data[F][t]))
    return {t: {m: round(float(np.median(v)), 3) for m, v in r.items() if v} for t, r in rows.items()}


def user_paths():
    """The uploaded word's pen paths in the same normalisation."""
    import compose as cp
    S = cp.load_style("style1")
    from proportions import measure_image
    D = measure_image()["nuqta_px"]
    out = {}
    for key, form in (("meem", "meem.init"), ("zain", "zain.fina"), ("noon", "noon.isol")):
        q = S["strokes"][key][0]
        ci = S["ids"][key]
        x, y, w, h = S["comps"][ci]["bbox"]
        L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
        t = np.linspace(0, L[-1], 64)
        P = np.stack([np.interp(t, L, q[:, 0]), np.interp(t, L, q[:, 1])], 1)
        out[form] = (P - [x + w / 2, S["base_y"]]) / D
    return out, D, S


def main():
    data = load()
    bench = benchmark(data)
    U, D, S = user_paths()
    # leave-one-letter-out on the user's own letters
    loo = {}
    for k in KNOWN:
        rest = [x for x in KNOWN if x != k]
        w = fit_weights(data, U, rest)
        p = predict(data, w, k)
        dist = {f: sum(np.linalg.norm(data[f][r] - U[r], axis=1).mean() for r in rest)
                for f in data if all(r in data[f] for r in rest + [k])}
        near = min(dist, key=dist.get)
        loo[k] = {"blend_err_nuqta": round(path_err(p, U[k]), 3),
                  "nearest_font_err_nuqta": round(path_err(data[near][k], U[k]), 3), "nearest_font": near}
    w_all = fit_weights(data, U, KNOWN)
    top = sorted(w_all.items(), key=lambda kv: -kv[1])[:6]
    preds = {t: predict(data, w_all, t).tolist() for t in TARGETS}
    out = {"corpus_leave_one_font_out_median_err_nuqta": bench, "user_leave_one_letter_out": loo,
           "user_blend_weights_top": [(f, round(float(v), 3)) for f, v in top],
           "user_predicted_paths_nuqta": preds, "user_nuqta_px": D, "user_base_y": S["base_y"]}
    (sb.OUT / "skeleton_blend_report.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "user_predicted_paths_nuqta"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
