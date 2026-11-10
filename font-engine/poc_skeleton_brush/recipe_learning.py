#!/usr/bin/env python3
"""Learn HOW to compose letters from parts, from 49 real typefaces.

FlexyFont (Phan, Fu & Chan 2015): typeface designers reuse a library of parts,
and the rules that say which part builds which letter can be LEARNED from a
font collection. Here every recipe of compose.py is run on each corpus font
using only that font's own meem / zain / noon pen paths, and compared with the
font's REAL letter (mean point error in nuqtas, translation in x removed).

  * recipe parameters (how short the neck of dal is, how tall alef is, how big
    the teeth of seen are ...) are chosen to minimise that error across fonts;
  * competing recipes are tested against each other (e.g. alef head = zain head
    vs. alef head = the start of the noon tooth MIRRORED, as the user proposed);
  * each recipe is also compared with the corpus-only methods (template blend,
    nearest font), so the engine uses, per letter, what predicts real letters
    best.
"""
from __future__ import annotations

import json
import itertools

import numpy as np

import compose as cp
import skeleton_blend as sbl

SC = 100.0  # 1 nuqta = 100 units, so compose.py's px-based bridges apply


def as_stroke(P):
    P = sb_resample(P * SC)
    z = np.zeros(len(P))
    return (P, z, z)


def sb_resample(P, step=1.0):
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    t = np.linspace(0, L[-1], max(int(L[-1] / step), 2) + 1)
    return np.stack([np.interp(t, L, P[:, 0]), np.interp(t, L, P[:, 1])], 1)


def to64(q):
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
    t = np.linspace(0, L[-1], 64)
    return np.stack([np.interp(t, L, q[:, 0]), np.interp(t, L, q[:, 1])], 1) / SC


def mirror(p):
    """Mirror a part left-right about its own start (a reversed tooth)."""
    out = p.copy()
    out.q = p.q.copy()
    out.q[:, 0] = 2 * p.q[0, 0] - p.q[:, 0]
    return out


def parts_of(F):
    st = {k: as_stroke(F[f]) for k, f in (("meem", "meem.init"), ("zain", "zain.fina"), ("noon", "noon.isol"))}
    N = cp.split_noon(st["noon"], 0.0)            # baseline is y = 0 in corpus units
    Z = cp.split_zain(st["zain"])
    noon = cp.sub(st["noon"], 0, len(st["noon"][0]) - 1, "noon")
    zain = cp.sub(st["zain"], 0, len(st["zain"][0]) - 1, "zain")
    return st, N, Z, noon, zain


def compose_letter(letter, F, prm):
    st, N, Z, noon, zain = parts_of(F)
    zh, zn, zt = Z["head"], Z["neck"], Z["tail"]
    noon_h = np.ptp(st["noon"][0][:, 1])
    if letter == "reh":
        return zain.q
    if letter == "dal":
        return cp.chain([zh, cp.scale(zn, prm["neck"]), zt]).q
    if letter == "beh":
        return noon.q
    if letter == "alef":
        if prm["head"] == "zain_head":
            stem = cp.chain([zh, zn])
        else:  # the start of noon's tooth, mirrored, then the zain neck
            stem = cp.chain([mirror(N["tooth"]), zn])
        f = prm["height"] * noon_h / max(np.ptp(stem.q[:, 1]), 1e-6)
        head = zh if prm["head"] == "zain_head" else mirror(N["tooth"])
        return cp.chain([head, cp.scale(zn, f)]).q
    if letter == "lam":
        stem = cp.chain([zh, zn])
        f = prm["height"] * noon_h / max(np.ptp(stem.q[:, 1]), 1e-6)
        return cp.chain([zh, cp.scale(zn, f), cp.scale(N["bowl"], prm["bowl"])]).q
    if letter == "alefmaksura":
        return cp.chain([zh, cp.scale(zn, prm["neck"]), cp.scale(N["bowl"], prm["bowl"])]).q
    if letter == "seen":
        t = cp.scale(N["tooth"], prm["tooth"])
        return cp.chain([t, t, noon], bridge=30).q
    raise KeyError(letter)


GRID = {
    "reh": [{}],
    "beh": [{}],
    "dal": [{"neck": v} for v in (0.3, 0.45, 0.55, 0.7, 0.85)],
    "alef": [{"head": h, "height": v} for h in ("zain_head", "noon_tooth_mirrored") for v in (1.2, 1.5, 1.8, 2.1)],
    "lam": [{"height": v, "bowl": b} for v in (1.2, 1.6, 2.0) for b in (0.8, 1.0, 1.2)],
    "alefmaksura": [{"neck": v, "bowl": b} for v in (0.3, 0.45, 0.6) for b in (0.9, 1.0, 1.2)],
    "seen": [{"tooth": v} for v in (0.4, 0.55, 0.7, 0.85)],
}
FORM = {"reh": "reh.isol", "dal": "dal.isol", "beh": "beh.isol", "alef": "alef.isol", "lam": "lam.isol",
        "alefmaksura": "alefmaksura.isol", "seen": "seen.isol"}


def err_xfree(P, Q):
    P = P - [P[:, 0].mean(), 0]
    Q = Q - [Q[:, 0].mean(), 0]
    return sbl.path_err(P, Q)


def evaluate(data):
    fonts = [f for f in data if all(k in data[f] for k in sbl.KNOWN)]
    res = {}
    for letter, grid in GRID.items():
        form = FORM[letter]
        best = None
        per_prm = []
        for prm in grid:
            errs = []
            for F in fonts:
                if form not in data[F]:
                    continue
                try:
                    errs.append(err_xfree(to64(compose_letter(letter, data[F], prm)), data[F][form]))
                except Exception:
                    pass
            if errs:
                m = float(np.median(errs))
                per_prm.append((prm, round(m, 3)))
                if best is None or m < best[1]:
                    best = (prm, m)
        # corpus-only methods with the same x-free error
        b_err, n_err = [], []
        for F in fonts:
            if form not in data[F]:
                continue
            templ = {f: d for f, d in data.items() if f != F}
            w = sbl.fit_weights(templ, data[F], sbl.KNOWN)
            pb = sbl.predict(templ, w, form)
            if pb is not None:
                b_err.append(err_xfree(pb, data[F][form]))
            dist = {f: sum(np.linalg.norm(templ[f][k] - data[F][k], axis=1).mean() for k in sbl.KNOWN)
                    for f in templ if all(k in templ[f] for k in sbl.KNOWN + [form])}
            near = min(dist, key=dist.get)
            n_err.append(err_xfree(templ[near][form], data[F][form]))
        res[letter] = {"best_recipe_params": best[0], "recipe_err": round(best[1], 3),
                       "blend_err": round(float(np.median(b_err)), 3), "nearest_err": round(float(np.median(n_err)), 3),
                       "all_params": per_prm}
        cands = {"recipe": best[1], "blend": float(np.median(b_err)), "nearest": float(np.median(n_err))}
        res[letter]["winner"] = min(cands, key=cands.get)
    return res


def main():
    data = sbl.load()
    res = evaluate(data)
    (sb_out() / "recipe_learning_report.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for k, v in res.items():
        print(f"{k:12s} recipe {v['recipe_err']:.3f} {v['best_recipe_params']}  blend {v['blend_err']:.3f}  "
              f"nearest {v['nearest_err']:.3f}  -> {v['winner']}")


def sb_out():
    import skeleton_brush as sb
    return sb.OUT


if __name__ == "__main__":
    main()
