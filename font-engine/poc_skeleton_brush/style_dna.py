#!/usr/bin/env python3
"""Style DNA: script-independent numbers that describe a lettering style.

The reference logos are each ONE style written in two scripts (the designer
made the Latin word, then the Arabic word in the same style). That gives an
honest test of a style metric:

    measure the DNA of the Latin half and of the Arabic half of every logo;
    for each Arabic half, rank the 8 Latin halves by DNA distance.
    A good metric puts the Arabic half's own Latin partner first
    (chance = 1 in 8).

DNA features (all measured from the ink mask, no script knowledge):
  weight        median stroke width / height of the whole two-line logo
  contrast      stroke width spread (p90 / p10 along the medial axis)
  width_cv      irregularity of the width along each stroke
  slant         lean of the near-vertical edges (degrees, + = leans right)
  straight      share of the outline that is straight
  corners       sharp corners per 100 px of outline
  dir_h/v/d     share of outline running horizontal / vertical / diagonal
  taper         width at stroke ends / median width (1 = blunt, < 1 = pointed)
  curv_soft     mean |curvature| on the curved parts (how round the curves are)

Output: output/style_dna.json, output/style_dna.png
"""
from __future__ import annotations

import json

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize

import skeleton_brush as sb

LOGOS = sb.HERE / "input" / "logos"
# which line is Latin (the other line is Arabic)
LATIN_ON_TOP = {"direct": True, "yashm": True, "haliz": False, "karashab": False, "kivano": False,
                "makita": False, "mootanroo": False, "sirang": False}
FEATURES = ["weight", "contrast", "width_cv", "slant", "straight", "corners", "dir_h", "dir_v", "dir_d",
            "taper", "curv_soft"]


def load(name, up=3):
    a = np.array(Image.open(LOGOS / f"{name}.png").convert("L"))
    a = cv2.resize(a, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
    m = a > 160                                  # ink only (the grey shadow bar of Karashab is dropped)
    m = ndimage.binary_opening(m, iterations=1)
    return m


def split_lines(m):
    """Two text lines: cut at the widest empty row band between them."""
    rows = m.sum(1) > 0
    ys = np.nonzero(rows)[0]
    y0, y1 = ys.min(), ys.max()
    gaps, start = [], None
    for y in range(y0, y1 + 1):
        if not rows[y] and start is None:
            start = y
        if rows[y] and start is not None:
            gaps.append((y - start, start, y))
            start = None
    g = max(gaps)
    cut = (g[1] + g[2]) // 2
    top, bot = m.copy(), m.copy()
    top[cut:] = False
    bot[:cut] = False
    return top, bot, float(y1 - y0)


def contour_props(m):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    L, straight, corners, hist, curv = 0.0, 0.0, 0, np.zeros(3), []
    angles_v, wts_v = [], []
    for c in cs:
        P = c[:, 0, :].astype(float)
        if len(P) < 40:
            continue
        # smooth the outline a little (pixel steps)
        P = ndimage.gaussian_filter1d(P, 2.0, axis=0, mode="wrap")
        d = np.roll(P, -3, 0) - np.roll(P, 3, 0)
        th = np.arctan2(d[:, 1], d[:, 0])
        seg = np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1)
        L += seg.sum()
        # turning per pixel over a window of +-5 px
        k = np.angle(np.exp(1j * (np.roll(th, -5) - np.roll(th, 5)))) / 10.0
        flat = np.abs(k) < 0.012
        straight += seg[flat].sum()
        curv += list(np.abs(k[~flat]))
        # corners: local maxima of turning over a short window
        kk = np.abs(np.angle(np.exp(1j * (np.roll(th, -4) - np.roll(th, 4)))))
        peaks = (kk > np.radians(55)) & (kk >= ndimage.maximum_filter1d(kk, 15, mode="wrap"))
        corners += int(peaks.sum())
        a = np.degrees(th) % 180
        hist[0] += seg[(a < 15) | (a > 165)].sum()
        hist[1] += seg[(a > 70) & (a < 110)].sum()
        hist[2] += seg[((a >= 15) & (a <= 70)) | ((a >= 110) & (a <= 165))].sum()
        nv = (a > 50) & (a < 130) & flat
        angles_v += list(a[nv] - 90)
        wts_v += list(seg[nv])
    out = {"straight": straight / L, "corners": 100.0 * corners / L,
           "dir_h": hist[0] / L, "dir_v": hist[1] / L, "dir_d": hist[2] / L,
           "curv_soft": float(np.mean(curv)) if curv else 0.0}
    if wts_v:
        a, w = np.array(angles_v), np.array(wts_v)
        o = np.argsort(a)
        cw = np.cumsum(w[o])
        out["slant"] = float(-a[o][np.searchsorted(cw, cw[-1] / 2)])   # image y down: lean right = negative angle
    else:
        out["slant"] = 0.0
    return out


def stroke_props(m, H):
    d = ndimage.distance_transform_edt(m)
    sk = skeletonize(m)
    w = 2 * d[sk]
    w = w[w > 2]
    nb = ndimage.convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - 1
    ends = np.argwhere(sk & (nb == 1))
    med = float(np.median(w))
    # width near the stroke ends: walk 1.5 widths in from each end
    tw = []
    for y, x in ends:
        r = int(1.0 * med)
        ys, xs = np.nonzero(sk[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1])
        if len(ys):
            far = np.hypot(ys - min(r, y), xs - min(r, x))
            sel = far > 0.6 * r
            if sel.any():
                tw.append(2 * d[ys[sel] + max(0, y - r), xs[sel] + max(0, x - r)].mean())
    # irregularity: local std of width on the skeleton
    lab, n = ndimage.label(sk, np.ones((3, 3)))
    cvs = []
    for i in range(1, n + 1):
        ww = 2 * d[lab == i]
        if len(ww) > 30:
            cvs.append(np.std(ww) / max(np.mean(ww), 1e-6))
    return {"weight": med / H, "contrast": float(np.percentile(w, 90) / max(np.percentile(w, 10), 1)),
            "width_cv": float(np.median(cvs)) if cvs else 0.0,
            "taper": float(np.median(tw) / med) if tw else 1.0}


def dna(m, H):
    f = stroke_props(m, H)
    f.update(contour_props(m))
    return {k: round(float(f[k]), 4) for k in FEATURES}


def main():
    res = {}
    for name, latin_top in LATIN_ON_TOP.items():
        m = load(name)
        top, bot, H = split_lines(m)
        lat, ara = (top, bot) if latin_top else (bot, top)
        res[name] = {"latin": dna(lat, H), "arabic": dna(ara, H)}
        print(f"{name:10s}", " ".join(f"{k}={res[name]['latin'][k]:.3f}/{res[name]['arabic'][k]:.3f}" for k in FEATURES))
    names = list(res)
    X = np.array([[res[n][s][k] for k in FEATURES] for s in ("latin", "arabic") for n in names])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    Lz, Az = Z[:len(names)], Z[len(names):]

    def test(cols):
        D = np.linalg.norm(Az[:, None, cols] - Lz[None, :, cols], axis=2)
        ranks = [int((D[i] < D[i, i]).sum()) + 1 for i in range(len(names))]
        return D, ranks

    allc = list(range(len(FEATURES)))
    D, ranks = test(allc)
    # leave-one-feature-out: which features carry the style across scripts
    imp = {}
    for j, k in enumerate(FEATURES):
        _, r = test([c for c in allc if c != j])
        imp[k] = round(float(np.mean(r)) - float(np.mean(ranks)), 3)    # > 0: removing it hurts
    # per feature: how well Latin predicts Arabic across the 8 logos (rank correlation)
    from scipy.stats import spearmanr
    agree = {k: round(float(spearmanr(X[:len(names), j], X[len(names):, j])[0]), 2) for j, k in enumerate(FEATURES)}
    rep = {"what": "Arabic half -> rank of its own Latin partner among 8 Latin halves (1 = best, chance mean 4.5)",
           "rank_of_own_partner": dict(zip(names, ranks)),
           "top1": f"{sum(r == 1 for r in ranks)}/{len(names)}",
           "top3": f"{sum(r <= 3 for r in ranks)}/{len(names)}",
           "mean_rank": round(float(np.mean(ranks)), 2),
           "feature_agreement_latin_vs_arabic_spearman": agree,
           "feature_importance_mean_rank_loss_when_removed": imp,
           "dna": res}
    (sb.OUT / "style_dna.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps({k: rep[k] for k in ("rank_of_own_partner", "top1", "top3", "mean_rank",
                                         "feature_agreement_latin_vs_arabic_spearman",
                                         "feature_importance_mean_rank_loss_when_removed")}, indent=1))
    # picture: distance matrix
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(D, cmap="viridis_r")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Latin half")
    ax.set_ylabel("Arabic half")
    for i in range(len(names)):
        j = int(np.argmin(D[i]))
        ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="red" if j != i else "white", lw=2))
    ax.set_title(f"Style DNA distance (top-1 {rep['top1']}, mean rank {rep['mean_rank']})")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(sb.OUT / "style_dna.png", dpi=110)




# ---------------------------------------------------------------- harder test
LATIN_TEXT = "Handgloves"
ARABIC_TEXT = "خط جميل سهل"


def render(font_path, text, size=140, rtl=False):
    from PIL import ImageDraw, ImageFont
    f = ImageFont.truetype(str(font_path), size, layout_engine=ImageFont.Layout.RAQM)
    kw = {"direction": "rtl", "language": "ar"} if rtl else {}
    im = Image.new("L", (size * 12, size * 3), 0)
    ImageDraw.Draw(im).text((size, size // 2), text, font=f, fill=255, **kw)
    return np.array(im) > 128


def corpus_test():
    """Every open font in corpus/ that has both scripts: Latin word vs Arabic word, same size.
    For each Arabic sample, rank all Latin samples; is the own font first?"""
    from fontTools.ttLib import TTFont
    rows = {}
    for p in sorted((sb.HERE / "corpus").glob("*.ttf")):
        cm = TTFont(p).getBestCmap()
        if not all(ord(c) in cm for c in LATIN_TEXT + ARABIC_TEXT.replace(" ", "")):
            continue
        try:
            rows[p.stem] = {"latin": dna(render(p, LATIN_TEXT), 140.0), "arabic": dna(render(p, ARABIC_TEXT, rtl=True), 140.0)}
        except Exception as ex:
            print("skip", p.stem, ex)
    (sb.OUT / "style_dna_corpus_dna.json").write_text(json.dumps(rows))
    names = list(rows)
    X = np.array([[rows[n][s][k] for k in FEATURES] for s in ("latin", "arabic") for n in names])
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    Lz, Az = Z[:len(names)], Z[len(names):]
    D = np.linalg.norm(Az[:, None] - Lz[None], axis=2)
    ranks = [int((D[i] < D[i, i]).sum()) + 1 for i in range(len(names))]
    from scipy.stats import spearmanr
    agree = {k: round(float(spearmanr(X[:len(names), j], X[len(names):, j])[0]), 2) for j, k in enumerate(FEATURES)}
    n = len(names)
    rep = {"fonts": n, "top1": f"{sum(r == 1 for r in ranks)}/{n}", "top3": f"{sum(r <= 3 for r in ranks)}/{n}",
           "top5": f"{sum(r <= 5 for r in ranks)}/{n}", "mean_rank": round(float(np.mean(ranks)), 2),
           "chance_mean_rank": round((n + 1) / 2, 1), "rank_of_own_font": dict(zip(names, ranks)),
           "feature_agreement_latin_vs_arabic_spearman": agree}
    (sb.OUT / "style_dna_corpus.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "rank_of_own_font"}, indent=1))
    return rep


def learned_weights_test():
    """Same corpus test, but each feature is weighted by how well it carries across scripts
    (Spearman Latin vs Arabic), learned WITHOUT the font being tested (leave-one-font-out)."""
    from scipy.stats import spearmanr
    rep = json.loads((sb.OUT / "style_dna_corpus.json").read_text())
    names = list(rep["rank_of_own_font"])
    # re-measure is slow; keep DNA in a cache file
    cache = sb.OUT / "style_dna_corpus_dna.json"
    rows = json.loads(cache.read_text())
    X = np.array([[rows[n][s][k] for k in FEATURES] for s in ("latin", "arabic") for n in names])
    n = len(names)
    Lx, Ax = X[:n], X[n:]
    ranks = []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        w = np.array([max(spearmanr(Lx[tr, f], Ax[tr, f])[0], 0) ** 2 for f in range(len(FEATURES))])
        mu, sd = X[tr + [t + n for t in tr]].mean(0), X[tr + [t + n for t in tr]].std(0) + 1e-9
        Lz, a = (Lx - mu) / sd, (Ax[i] - mu) / sd
        d = np.sqrt((((Lz - a) ** 2) * w).sum(1))
        ranks.append(int((d < d[i]).sum()) + 1)
    out = {"top1": f"{sum(r == 1 for r in ranks)}/{n}", "top3": f"{sum(r <= 3 for r in ranks)}/{n}",
           "top5": f"{sum(r <= 5 for r in ranks)}/{n}", "mean_rank": round(float(np.mean(ranks)), 2),
           "rank_of_own_font": dict(zip(names, ranks))}
    rep["learned_weights_leave_one_font_out"] = out
    (sb.OUT / "style_dna_corpus.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rank_of_own_font"}, indent=1))


if __name__ == "__main__":
    main()
    corpus_test()
    learned_weights_test()
