#!/usr/bin/env python3
"""The engine designs letters by itself: try thousands of compositions of the
writer's own stroke parts, judge each one, keep the best.

  genes     a letter = 1..3 segments; a segment = a piece [a, b] of one of the
            uploaded strokes (meem / zain / noon), optionally reversed and/or
            mirrored, scaled (x, y), slightly rotated. The brush widths travel
            with the piece; pieces are joined with smooth bridges and cut ends
            get the writer's round terminal (compose.py).
  eye       legibility  : ensemble of letter classifiers (legibility.py),
                          adapted to the style with the uploaded letters,
                          + distance to the nearest real letter of that class;
            style       : no blobs (ink overlapping itself more than in the
                          writer's letters), no kinks sharper than the
                          writer's, sizes near the nuqta proportions, simple
                          recipes preferred.
  search    random start + evolution (mutate the best, keep the best).

The same eye scores the hand-written recipes of compose.py, so the report
says honestly whether the automatic designs are better.
"""
from __future__ import annotations

import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import ndimage

import compose as cp
import legibility as lg
import skeleton_brush as sb

Q = 4                      # search at quarter scale
SRC = ["meem", "zain", "noon"]
TARGETS = [n for n in lg.NAMES if n != "noon"]
FORM_OF = {"beh": "beh.isol", "dal": "dal.isol", "reh": "reh.isol", "alef": "alef.isol", "lam": "lam.isol",
           "yeh": "alefmaksura.isol", "seen": "seen.isol", "waw": "waw.isol", "heh": "heh.isol", "ain": "ain.isol",
           "hah": "hah.isol", "sad": "sad.isol", "meem": "meem.isol"}
OFFSETS = np.array([0.1, 0.0, -0.15, -0.3, -0.5, -0.7, -0.9, -1.1, -1.4])   # ink bottom vs baseline, noon heights


# ----------------------------------------------------------------------------
# style context
# ----------------------------------------------------------------------------


class Ctx:
    def __init__(self, style, scale=Q):
        S = cp.load_style(style)
        self.S, self.style, self.scale = S, style, scale
        self.strokes = {}
        for k in SRC:
            q, wl, wr = S["strokes"][k]
            self.strokes[k] = (q[::scale] / scale, wl[::scale] / scale, wr[::scale] / scale)
        noon = S["comps"][S["ids"]["noon"]]["mask"]
        self.noon_h = (np.ptp(np.nonzero(noon)[0]) + 1) / scale
        self.cap = "round" if style == "style1" else "flat"
        self.bridge = max(3, 40 // scale)
        # the writer's real terminals: width profile (tip -> inside, as a multiple of the body
        # width) at the start and at the end of each uploaded stroke. A cut end of a new letter
        # gets the terminal of the stroke it was cut from, instead of a blunt cap.
        self.writer_terminals = False   # tested: hurts legibility and the ground-truth seen (see README)
        self.term_len = 1.0
        self.terms = {}
        for k, (q, wl, wr) in self.strokes.items():
            w = ndimage.gaussian_filter1d(wl + wr, 2.0)
            n = len(w)
            body = float(np.median(w[n // 5: -n // 5]))
            s_ = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
            prof = {}
            for which, ww, ss in (("start", w, s_), ("end", w[::-1], s_[-1] - s_[::-1])):
                # the tip only: at most one stroke width from the end (a longer thin run is
                # structure, e.g. the noon's tooth, not terminal), blended into the body
                lim = int(np.searchsorted(ss, self.term_len * body))
                j = next((i for i in range(lim) if ww[i] >= 0.95 * body), lim)
                m = np.clip(ww[:j + 1] / body, 0.05, 1.5)
                r = np.clip((np.arange(j + 1) - 0.6 * j) / max(0.4 * j, 1), 0, 1)
                prof[which] = (ss[:j + 1].copy(), m * (1 - r) + r)
            self.terms[k] = prof
        # natural cut points: extrema of x and y along each stroke
        self.cuts = {}
        for k, (q, _, _) in self.strokes.items():
            d = np.diff(q, axis=0)
            idx = [i for c in (0, 1) for i in range(1, len(d)) if np.sign(d[i - 1, c]) != np.sign(d[i, c])]
            self.cuts[k] = sorted({0.0, 1.0, *[round(i / (len(q) - 1), 3) for i in idx]})
        # the writer's own limits
        ratios, kinks = [], []
        for k in SRC:
            p = cp.Part(*self.strokes[k], k, True, True)
            m, r, kk = raster(p)
            ratios.append(r)
            kinks.append(kk)
        self.ratio_min, self.kink_max = min(ratios), max(kinks)
        try:
            from proportions import load as lc, measure_image, predict_for
            user = measure_image(style, cp.STYLES[style])
            self.preds, _, _ = predict_for(user, lc())
            self.D = user["nuqta_px"] / scale
        except Exception:
            self.preds, self.D = {}, None


def raster(part):
    """Ink mask of a part + self-overlap ratio + sharpest turn (deg / 5 px)."""
    import cv2
    q, wl, wr = part.q, np.clip(part.wl, 0.3, None), np.clip(part.wr, 0.3, None)
    pad = float(max(wl.max(), wr.max())) + 3
    o = q.min(0) - pad
    size = np.ceil(q.max(0) - o + pad).astype(int)
    canvas = np.zeros((size[1], size[0]), np.uint8)
    qq = q - o
    sb.render_ribbon(canvas, qq, wl, wr)
    m = canvas > 127
    ds = np.linalg.norm(np.diff(q, axis=0), axis=1)
    strip = float((((wl + wr)[1:] + (wl + wr)[:-1]) / 2 * ds).sum())
    T = np.diff(q, axis=0)
    th = np.unwrap(np.arctan2(T[:, 1], T[:, 0]))
    kink = float(np.degrees(np.abs(th[5:] - th[:-5]).max())) if len(th) > 6 else 0.0
    raster.origin = o
    return m, m.sum() / max(strip, 1.0), kink


# ----------------------------------------------------------------------------
# genes -> part
# ----------------------------------------------------------------------------


def transform(p, mir, sx, sy, rot):
    out = p.copy()
    d = p.q - p.q[0]
    if mir:
        d[:, 0] = -d[:, 0]
        out.wl, out.wr = p.wr.copy(), p.wl.copy()
    d = d * [sx, sy]
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    out.q = p.q[0] + d @ np.array([[c, s], [-s, c]])
    return cp.resample_part(out)


def reverse(p):
    out = cp.Part(p.q[::-1], p.wr[::-1], p.wl[::-1], p.src, p.end_term, p.start_term)
    out.labels = p.labels[::-1]
    return out


def writer_terminal(p, prof, at):
    """Give a cut end the writer's own terminal: the width multiplier profile of a real
    stroke end (tip -> inside), laid along the new end by arc length; the cut is made
    symmetric first, the path is not changed."""
    s_t, mult = prof
    if len(s_t) < 3:
        return p
    out = p.copy()
    q = out.q if at == "start" else out.q[::-1]
    s_ = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
    L = s_t[-1]
    sel = s_ <= L
    if sel.sum() < 3 or sel.sum() > 0.45 * len(q):
        return p
    f = np.interp(s_[sel], s_t, mult)
    wl, wr = (out.wl, out.wr) if at == "start" else (out.wl[::-1], out.wr[::-1])
    wl, wr = wl.copy(), wr.copy()
    mid = (wl[sel] + wr[sel]) / 2
    wl[sel], wr[sel] = mid * f, mid * f
    out.wl, out.wr = (wl, wr) if at == "start" else (wl[::-1], wr[::-1])
    return out


def build(genes, ctx):
    parts = []
    for g in genes:
        q, wl, wr = ctx.strokes[g["src"]]
        n = len(q)
        i0, i1 = int(round(g["a"] * (n - 1))), int(round(g["b"] * (n - 1)))
        if i1 - i0 < 6:
            return None
        p = cp.sub((q, wl, wr), i0, i1, f"{g['src']}[{g['a']:.2f}-{g['b']:.2f}]")
        if g["rev"]:
            p = reverse(p)
        parts.append(transform(p, g["mir"], g["sx"], g["sy"], g["rot"]))
    out = parts[0]
    for p in parts[1:]:
        br = int(min(ctx.bridge, len(out.q) // 3, len(p.q) // 3))
        out = cp.attach(out, p, max(br, 2))
    k = int(min(2 * ctx.bridge, len(out.q) // 3))
    ramp = np.linspace(1, 0, k)
    if getattr(ctx, "writer_terminals", False):
        g0, g1 = genes[0], genes[-1]
        if not out.start_term:
            out = writer_terminal(out, ctx.terms[g0["src"]]["end" if g0["rev"] else "start"], "start")
        if not out.end_term:
            out = writer_terminal(out, ctx.terms[g1["src"]]["start" if g1["rev"] else "end"], "end")
    if not out.start_term:          # a cut end: make the brush symmetric there, then round it
        mid = (out.wl[:k] + out.wr[:k]) / 2
        out.wl[:k] = ramp * mid + (1 - ramp) * out.wl[:k]
        out.wr[:k] = ramp * mid + (1 - ramp) * out.wr[:k]
        out = cp.end_cap(out, ctx.cap, "start")
    if not out.end_term:
        mid = (out.wl[-k:] + out.wr[-k:]) / 2
        out.wl[-k:] = ramp[::-1] * mid + (1 - ramp[::-1]) * out.wl[-k:]
        out.wr[-k:] = ramp[::-1] * mid + (1 - ramp[::-1]) * out.wr[-k:]
        out = cp.end_cap(out, ctx.cap, "end")
    return out


def rand_seg(rng, ctx):
    src = SRC[rng.integers(3)]
    cuts = ctx.cuts[src]
    if rng.random() < 0.6 and len(cuts) > 2:
        a, b = sorted(rng.choice(cuts, 2, replace=False))
    else:
        a, b = sorted(rng.random(2))
    if b - a < 0.12:
        a, b = max(0.0, a - 0.1), min(1.0, b + 0.1)
    plain = rng.random() < 0.5
    return dict(src=src, a=float(a), b=float(b), rev=bool(rng.random() < 0.25), mir=bool(rng.random() < 0.2),
                sx=1.0 if plain else float(np.exp(rng.normal(0, 0.3))), sy=1.0 if plain else float(np.exp(rng.normal(0, 0.3))),
                rot=0.0 if plain else float(rng.normal(0, 12)))


def rand_genes(rng, ctx):
    return [rand_seg(rng, ctx) for _ in range(rng.choice([1, 2, 3], p=[0.3, 0.5, 0.2]))]


def mutate(genes, rng, ctx):
    g = [dict(s) for s in genes]
    r = rng.random()
    if r < 0.08 and len(g) < 3:
        g.insert(rng.integers(len(g) + 1), rand_seg(rng, ctx))
    elif r < 0.16 and len(g) > 1:
        g.pop(rng.integers(len(g)))
    else:
        s = g[rng.integers(len(g))]
        for key, sd in (("a", 0.05), ("b", 0.05)):
            if rng.random() < 0.6:
                s[key] = float(np.clip(s[key] + rng.normal(0, sd), 0, 1))
        if s["b"] - s["a"] < 0.1:
            s["a"], s["b"] = max(0.0, s["a"] - 0.06), min(1.0, s["b"] + 0.06)
        for key in ("sx", "sy"):
            if rng.random() < 0.5:
                s[key] = float(np.clip(s[key] * np.exp(rng.normal(0, 0.12)), 0.4, 2.2))
        if rng.random() < 0.4:
            s["rot"] = float(np.clip(s["rot"] + rng.normal(0, 6), -40, 40))
        if rng.random() < 0.05:
            s["rev"] = not s["rev"]
        if rng.random() < 0.05:
            s["mir"] = not s["mir"]
    return g


# ----------------------------------------------------------------------------
# the eye
# ----------------------------------------------------------------------------


class Eye:
    def __init__(self, ctx, n_models=3):
        import joblib
        path = sb.OUT / f"legibility_{ctx.style}.joblib"
        bank_path = sb.HERE / "corpus" / "legibility_dataset.npz"
        if bank_path.exists():
            z = np.load(bank_path)
            X, y, g = z["X"], z["y"], z["g"]
        else:
            X, y, g, _ = lg.build_dataset()
            np.savez_compressed(bank_path, X=X, y=y, g=g)
        # teach the eye this style: the uploaded letters with light augmentation
        S = ctx.S
        nh = ctx.noon_h * ctx.scale
        rng = np.random.default_rng(0)
        ux, uy = [], []
        for key, cls in (("noon", "noon"), ("zain", "reh")):
            m = S["comps"][S["ids"][key]]["mask"]
            for i in range(80):
                mm = m if i == 0 else light_aug(m, rng)
                f = lg.featurize(mm, S["base_y"], nh)
                if f is not None:
                    ux.append(f); uy.append(lg.NAMES.index(cls))
        X2 = np.vstack([X, np.array(ux, np.float32)])
        y2 = np.r_[y, uy]
        if path.exists():
            self.models = joblib.load(path)
        else:
            self.models = []
            for seed in range(n_models):
                clf = lg.make_clf()
                clf.set_params(mlpclassifier__random_state=seed)
                self.models.append(clf.fit(X2, y2))
            joblib.dump(self.models, path)
        self.bank = {c: X2[y2 == c] for c in range(len(lg.NAMES))}
        self.d0 = {}
        for c, B in self.bank.items():
            idx = rng.choice(len(B), min(120, len(B)), replace=False)
            d = np.sqrt(((B[idx][:, None] - B[None]) ** 2).sum(2))
            d[d < 1e-6] = np.inf
            self.d0[c] = float(np.median(np.sort(d, axis=1)[:, 4]))

    def proba(self, F):
        return np.mean([m.predict_proba(F) for m in self.models], axis=0)

    def nn_dist(self, F, c):
        B = self.bank[c]
        d2 = (F ** 2).sum(1)[:, None] - 2 * F @ B.T + (B ** 2).sum(1)[None]
        return np.sqrt(np.clip(d2.min(1), 0, None)) / self.d0[c]


class PathEye:
    """Second opinion, on STRUCTURE: is the candidate's pen path close to the
    real pen paths of that letter (49 typefaces + the uploaded letters)?
    Paths: 64 points, baseline origin, x centred, unit = height of the noon
    path of the same style."""

    def __init__(self, ctx, tau=0.12):
        data = json.loads((sb.OUT / "skeleton_corpus.json").read_text())
        self.tau = tau
        self.bank = {}
        for c, name in enumerate(lg.NAMES):
            form = {"yeh": "alefmaksura.isol"}.get(name, name + ".isol")
            rows = []
            for f, d in data.items():
                if form in d and "noon.isol" in d:
                    hn = np.ptp(np.array(d["noon.isol"]["path"])[:, 1])
                    P = np.array(d[form]["path"]) / max(hn, 1e-6)
                    rows.append(P - [P[:, 0].mean(), 0])
            self.bank[c] = rows
        self.hU = float(np.ptp(ctx.strokes["noon"][0][:, 1]))
        base = ctx.S["base_y"] / ctx.scale
        for key, cls in (("noon", "noon"), ("zain", "reh")):      # the style's own letters
            self.bank[lg.NAMES.index(cls)].append(self.norm(ctx.strokes[key][0], base))
        self.B = {c: np.array(v + [p[::-1] for p in v]) for c, v in self.bank.items() if v}

    def norm(self, q, base_y):
        L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
        t = np.linspace(0, L[-1], 64)
        P = np.stack([np.interp(t, L, q[:, 0]), np.interp(t, L, q[:, 1])], 1)
        P = (P - [P[:, 0].mean(), base_y]) / self.hU
        return P

    def dists(self, P):
        out = np.full(len(lg.NAMES), 9.0)
        for c, B in self.B.items():
            d = np.linalg.norm(B - P[None], axis=2).mean(1)
            out[c] = np.sort(d)[:3].mean()
        return out

    def proba(self, P):
        d = self.dists(P)
        e = np.exp(-(d - d.min()) / self.tau)
        return e / e.sum(), d


def light_aug(m, rng):
    import cv2
    ys, xs = np.nonzero(m)
    cx, cy = float(xs.mean()), float(ys.mean())
    A = cv2.getRotationMatrix2D((cx, cy), rng.uniform(-5, 5), rng.uniform(0.92, 1.08))
    A[0, 1] += rng.uniform(-0.08, 0.08)
    return cv2.warpAffine(m.astype(np.uint8) * 255, A, (m.shape[1], m.shape[0])) > 127


def candidate_features(part, ctx):
    m, ratio, kink = raster(part)
    ys, xs = np.nonzero(m)
    if len(ys) < 20:
        return None
    y1 = ys.max() + 1
    h, w = y1 - ys.min(), xs.max() + 1 - xs.min()
    feats = [lg.featurize(m, y1 + t * ctx.noon_h, ctx.noon_h) for t in OFFSETS]
    return np.array(feats, np.float32), ratio, kink, w, h, float(raster.origin[1] + y1), m


def score_batch(parts, genes_list, target, ctx, eye, peye=None):
    c = lg.NAMES.index(target)
    rows, meta = [], []
    for i, p in enumerate(parts):
        cf = candidate_features(p, ctx) if p is not None else None
        if cf is None:
            meta.append(None)
            continue
        meta.append((len(rows), *cf[1:]))
        rows.append(cf[0])
    out = [(-1e9, None)] * len(parts)
    if not rows:
        return out
    F = np.vstack(rows)
    P = eye.proba(F)[:, c].reshape(-1, len(OFFSETS))
    Fr = F.reshape(-1, len(OFFSETS), F.shape[1])
    best_t = P.argmax(1)
    Fb = Fr[np.arange(len(Fr)), best_t]
    pb = P[np.arange(len(P)), best_t]
    dn = eye.nn_dist(Fb, c)
    form = FORM_OF.get(target)
    for i, mt in enumerate(meta):
        if mt is None:
            continue
        k, ratio, kink, w, h, ink_bottom, mask = mt
        genes = genes_list[i]
        pen = 1.0 * max(0.0, dn[k] - 1.0)
        p_path, d_path = 1.0, 0.0
        if peye is not None:
            base = ink_bottom + OFFSETS[best_t[k]] * ctx.noon_h
            pp_, dd_ = peye.proba(peye.norm(parts[i].q, base))
            p_path, d_path = float(pp_[c]), float(dd_[c])
        pen += 3.0 * max(0.0, 0.95 * ctx.ratio_min - ratio)
        pen += 0.02 * max(0.0, kink - 1.2 * ctx.kink_max)
        if form and form in ctx.preds and ctx.D:
            pen += 0.5 * (abs(math.log(max(w / ctx.D, 1e-3) / ctx.preds[form]["w"]))
                          + abs(math.log(max(h / ctx.D, 1e-3) / ctx.preds[form]["h"])))
        if genes is not None:
            pen += 0.15 * (len(genes) - 1)
            pen += 0.3 * sum(abs(math.log(g["sx"])) + abs(math.log(g["sy"])) + abs(g["rot"]) / 25 for g in genes)
        d_style = None
        if getattr(ctx, "judge", None) is not None:
            # the style judge (letter_dna.py): stray from the uploaded letters' style more
            # than a real designer's letter strays from its own font -> penalty
            d_style = ctx.judge.distance(mask)
            pen += ctx.judge_weight * max(0.0, d_style - ctx.judge_thr)
        s = 2.0 * math.log(max(pb[k], 1e-4)) + 2.0 * math.log(max(p_path, 1e-4)) - pen
        out[i] = (s, dict(d_style=d_style, p=float(pb[k]), p_path=p_path, d_path=d_path, nn=float(dn[k]), overlap_ratio=float(ratio), kink=float(kink),
                          bottom_vs_baseline=float(OFFSETS[best_t[k]]), penalty=float(pen)))
    return out


# ----------------------------------------------------------------------------
# search
# ----------------------------------------------------------------------------


def add_judge(ctx, weight=1.0):
    import letter_dna as ld
    refs = [raster(cp.Part(*ctx.strokes[k], k, True, True))[0] for k in SRC]
    ctx.judge, ctx.judge_weight = ld.StyleJudge(refs, ctx.noon_h), weight
    # tolerance = how far the uploaded letters themselves are from their own style mean
    ctx.judge_thr = max(1.0, max(ctx.judge.distance(m) for m in refs))
    return ctx


def design(target, style="style1", n0=2500, gens=7, keep=40, kids=400, seed=0, judge=0.0):
    ctx = Ctx(style)
    if judge:
        add_judge(ctx, judge)
    eye = Eye(ctx)
    peye = PathEye(ctx)
    rng = np.random.default_rng(seed + lg.NAMES.index(target))
    pop = [rand_genes(rng, ctx) for _ in range(n0)]
    pop += [[dict(src=s, a=0.0, b=1.0, rev=False, mir=False, sx=1.0, sy=1.0, rot=0.0)] for s in SRC]

    def evaluate(gl):
        parts = []
        for g in gl:
            try:
                parts.append(build(g, ctx))
            except Exception:
                parts.append(None)
        return score_batch(parts, gl, target, ctx, eye, peye)

    scored = sorted(zip(evaluate(pop), pop), key=lambda t: -t[0][0])[:keep]
    for _ in range(gens):
        children = [mutate(scored[rng.integers(len(scored))][1], rng, ctx) for _ in range(kids)]
        scored = sorted(scored + list(zip(evaluate(children), children)), key=lambda t: -t[0][0])[:keep]
    (s, info), genes = scored[0]
    return {"target": target, "score": round(float(s), 3), "info": info, "genes": genes}


def score_hand(style="style1"):
    """The hand-written recipes of compose.py judged by the same eye."""
    ctx = Ctx(style)
    eye = Eye(ctx)
    peye = PathEye(ctx)
    R, _, _ = cp.recipes(ctx.S, ctx.cap)
    out = {}
    for letter, (part, how, mode) in R.items():
        cls = {"alefmaksura": "yeh", "teh": None, "theh": None}.get(letter, letter)
        if cls is None:
            continue
        p = cp.Part(part.q[::Q] / Q, part.wl[::Q] / Q, part.wr[::Q] / Q, letter, True, True)
        (s, info), = score_batch([cp.resample_part(p)], [None], cls, ctx, eye, peye)
        out[cls] = {"score": round(float(s), 3), "info": info, "recipe": how}
    return out


def _job(args):
    try:
        return design(*args)
    except Exception as ex:  # keep the batch alive, report the failure
        return {"target": args[0], "error": f"{type(ex).__name__}: {ex}"}


def main():
    style = sys.argv[1] if len(sys.argv) > 1 else "style1"
    judge = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0      # weight of the style judge
    Eye(Ctx(style))                      # train / cache the eye once before forking
    with ProcessPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(_job, [(t, style, 2500, 7, 40, 400, 0, judge) for t in TARGETS]))
    hand = score_hand(style)
    out = {"style": style, "auto": {r["target"]: r for r in res}, "hand_recipes_same_eye": hand}
    (sb.OUT / f"auto_design_{style}{'_judge' if judge else ''}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for r in res:
        if "error" in r:
            print(r["target"], "ERROR", r["error"])
            continue
        h = hand.get(r["target"])
        print(f"{r['target']:6s} auto score {r['score']:7.2f} shape p={r['info']['p']:.2f} path p={r['info']['p_path']:.2f} segs={len(r['genes'])}"
              + (f" | hand score {h['score']:7.2f} shape p={h['info']['p']:.2f} path p={h['info']['p_path']:.2f}" if h and h["info"] else ""))


if __name__ == "__main__":
    main()
