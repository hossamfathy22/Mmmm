#!/usr/bin/env python3
"""Smart proportions: predict the size and position of letters you do not
have from the letters you do have, in nuqta units.

  data      corpus_nuqta_metrics.json (fonts_corpus.py): ~50 Arabic typefaces,
            every letter body measured in nuqtas (w, h, top, bottom)
  features  the letters present in the uploaded word (meem.init, zain.fina,
            noon.isol): their w/h/top/bottom + stroke weight, in nuqtas
  model     ridge regression per target letter & measure (closed form),
            in log space for sizes (proportions multiply, they do not add)
  test      leave-one-FONT-out: hide a whole typeface, learn from the others,
            predict its letters from its own meem/zain/noon; compare with
            two baselines (corpus average, and "scale by the noon width")
  extra     style retrieval: the typefaces whose meem/zain/noon proportions
            are closest to the uploaded word (used as skeleton donors)
"""
from __future__ import annotations

import json
import math

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

import skeleton_brush as sb
from fonts_corpus import ANCHORS, FORMS

MEASURES = ["w", "h", "top", "bottom"]
LOGM = {"w", "h"}


def feature_vector(m):
    f = []
    for a in ANCHORS:
        for k in MEASURES:
            if a == "noon.isol" and k == "bottom":
                continue  # 0 by definition (reference line)
            v = m[a][k]
            f.append(math.log(max(v, 1e-3)) if k in LOGM else v)
    f.append(math.log(max(m["noon.isol"]["weight"], 1e-3)))
    return np.array(f, float)


FEATURE_NAMES = [f"{a}.{k}" for a in ANCHORS for k in MEASURES if not (a == "noon.isol" and k == "bottom")] + \
                ["stroke_weight"]


def ridge_fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    A = Z.T @ Z + alpha * np.eye(Z.shape[1])
    beta = np.linalg.solve(A, Z.T @ (y - y.mean()))
    return dict(mu=mu, sd=sd, beta=beta, b0=y.mean())


def ridge_pred(m, x):
    return m["b0"] + ((x - m["mu"]) / m["sd"]) @ m["beta"]


def load():
    data = json.loads((sb.OUT / "corpus_nuqta_metrics.json").read_text())
    return {k: v for k, v in data.items() if all(a in v for a in ANCHORS)}


def evaluate(data, alpha=3.0):
    fonts = sorted(data)
    targets = [t for t in FORMS if t not in ANCHORS]
    report = {}
    for t in targets:
        for k in MEASURES:
            rows = [(feature_vector(data[f]), data[f][t][k]) for f in fonts if t in data[f]]
            if len(rows) < 12:
                continue
            X = np.array([r[0] for r in rows])
            y_raw = np.array([r[1] for r in rows])
            y = np.log(np.maximum(y_raw, 1e-3)) if k in LOGM else y_raw
            noon_w = np.exp(X[:, FEATURE_NAMES.index("noon.isol.w")])
            err_m, err_mean, err_ratio = [], [], []
            for i in range(len(rows)):
                tr = np.arange(len(rows)) != i
                mdl = ridge_fit(X[tr], y[tr], alpha)
                p = ridge_pred(mdl, X[i])
                p = math.exp(p) if k in LOGM else p
                base_mean = y_raw[tr].mean()
                base_ratio = (y_raw[tr] / noon_w[tr]).mean() * noon_w[i]
                err_m.append(abs(p - y_raw[i]))
                err_mean.append(abs(base_mean - y_raw[i]))
                err_ratio.append(abs(base_ratio - y_raw[i]))
            scale = np.mean(np.abs(y_raw)) + 1e-9
            report[f"{t}.{k}"] = {
                "fonts": len(rows),
                "model_mae_nuqta": round(float(np.mean(err_m)), 3),
                "baseline_average_mae_nuqta": round(float(np.mean(err_mean)), 3),
                "baseline_noon_ratio_mae_nuqta": round(float(np.mean(err_ratio)), 3),
                "model_rel_error_pct": round(100 * float(np.mean(err_m)) / scale, 1),
            }
    return report


def measure_image(style="style1", box=(1000, 360, 1850, 880)):
    """The uploaded word measured exactly like the corpus (bodies without dots)."""
    mask = sb.load_region(sb.HERE / "input" / "muzn_two_styles.jpg", box, 2)
    comps = sb.split_components(mask)
    bodies = [c for c in comps if c["kind"] == "body"]  # RTL: meem, zain, noon
    dots = [c for c in comps if c["kind"] == "dot"]
    meem, zain, noon = bodies
    nx = noon["bbox"][0] + noon["bbox"][2] / 2
    noon_dot = min(dots, key=lambda d: abs(d["bbox"][0] + d["bbox"][2] / 2 - nx))
    D = math.sqrt(noon_dot["area"])
    ref = noon["bbox"][1] + noon["bbox"][3]  # image y of the noon bottom

    def m(c):
        x, y, w, h = c["bbox"]
        return {"w": w / D, "h": h / D, "top": (ref - y) / D, "bottom": (ref - (y + h)) / D,
                "weight": c["area"] / max(skeletonize(c["mask"]).sum(), 1) / D}
    return {"meem.init": m(meem), "zain.fina": m(zain), "noon.isol": m(noon), "nuqta_px": D,
            "noon_bottom_y": ref}


def predict_for(user, data, alpha=3.0):
    x = feature_vector(user)
    fonts = sorted(data)
    X = np.array([feature_vector(data[f]) for f in fonts])
    z = (x - X.mean(0)) / (X.std(0) + 1e-9)
    preds = {}
    for t in [t for t in FORMS if t not in ANCHORS]:
        preds[t] = {}
        for k in MEASURES:
            rows = [(feature_vector(data[f]), data[f][t][k]) for f in fonts if t in data[f]]
            Xt = np.array([r[0] for r in rows])
            yt = np.array([r[1] for r in rows])
            y = np.log(np.maximum(yt, 1e-3)) if k in LOGM else yt
            p = ridge_pred(ridge_fit(Xt, y, alpha), x)
            preds[t][k] = round(float(math.exp(p) if k in LOGM else p), 3)
    # style retrieval: nearest typefaces in standardized proportion space
    Zf = (X - X.mean(0)) / (X.std(0) + 1e-9)
    dist = np.linalg.norm(Zf - z, axis=1)
    order = np.argsort(dist)
    nearest = [(fonts[i], round(float(dist[i]), 2)) for i in order[:5]]
    outside = {FEATURE_NAMES[i]: round(float(z[i]), 1) for i in np.nonzero(np.abs(z) > 2.5)[0]}
    return preds, nearest, outside


def draw_chart(user, preds, path):
    """A calligraphy-style proportion sheet: every predicted letter body as a
    box on the nuqta grid, placed relative to the noon bowl's bottom line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = ["noon.isol", "zain.fina", "meem.init", "beh.isol", "teh.isol", "alefmaksura.isol", "reh.isol",
            "dal.isol", "waw.isol", "lam.isol", "alef.isol", "heh.isol", "ain.isol", "seen.isol"]
    glyph = {"noon.isol": "ن", "zain.fina": "ـز", "meem.init": "مـ", "beh.isol": "ب", "teh.isol": "ت",
             "alefmaksura.isol": "ى", "reh.isol": "ر", "dal.isol": "د", "waw.isol": "و", "lam.isol": "ل",
             "alef.isol": "ا", "heh.isol": "ه", "ain.isol": "ع", "seen.isol": "س"}
    fig, ax = plt.subplots(figsize=(16, 5.5))
    x = 0.0
    for name in show:
        m = user[name] if name in user else preds[name]
        col = "#e4572e" if name in user else "#2b7bba"
        ax.add_patch(plt.Rectangle((x, m["bottom"]), m["w"], m["h"], fill=False, ec=col, lw=2))
        for gx in np.arange(0, m["w"] + 1e-6, 1.0):
            ax.plot([x + gx, x + gx], [m["bottom"], m["bottom"] + m["h"]], color=col, lw=0.3, alpha=0.5)
        for gy in np.arange(0, m["h"] + 1e-6, 1.0):
            ax.plot([x, x + m["w"]], [m["bottom"] + gy, m["bottom"] + gy], color=col, lw=0.3, alpha=0.5)
        ax.text(x + m["w"] / 2, -2.2 - 1.1 * (show.index(name) % 2), f"{name}\n{m['w']:.1f}×{m['h']:.1f}",
                ha="center", va="top", fontsize=7, color=col)
        x += m["w"] + 0.8
    ax.axhline(0, color="#888", lw=1, ls="--")
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, x)
    ax.set_title("Letter bodies in nuqtas (grid = 1 nuqta). Red = measured from the uploaded word, "
                 "blue = predicted. Dashed line = bottom of the noon bowl.")
    ax.set_ylim(-4.6, 7)
    ax.set_yticks(range(-2, 8))
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    data = load()
    rep = evaluate(data)
    better = sum(1 for v in rep.values() if v["model_mae_nuqta"] < min(v["baseline_average_mae_nuqta"],
                                                                        v["baseline_noon_ratio_mae_nuqta"]))
    user = measure_image()
    preds, nearest, outside = predict_for(user, data)
    out = {"fonts_in_corpus": len(data),
           "leave_one_font_out": rep,
           "summary": {
               "targets_x_measures": len(rep),
               "model_beats_both_baselines": better,
               "median_model_mae_nuqta": round(float(np.median([v["model_mae_nuqta"] for v in rep.values()])), 3),
               "median_avg_baseline_mae_nuqta": round(float(np.median([v["baseline_average_mae_nuqta"] for v in rep.values()])), 3),
               "median_noon_ratio_baseline_mae_nuqta": round(float(np.median([v["baseline_noon_ratio_mae_nuqta"] for v in rep.values()])), 3)},
           "uploaded_word_in_nuqtas": user,
           "predicted_for_uploaded_style": preds,
           "nearest_typefaces": nearest,
           "features_far_outside_corpus (z>2.5)": outside}
    (sb.OUT / "proportions_report.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    draw_chart(user, preds, sb.OUT / "proportions_nuqta_chart.png")
    print(json.dumps(out["summary"], indent=1))
    print("nearest:", nearest)
    print("outside corpus:", outside)


if __name__ == "__main__":
    main()
