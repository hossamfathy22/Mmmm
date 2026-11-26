"""
معادلات v1 بعد إصلاح ثلاثة أخطاء داخلية كشفها تشخيص الواو (diagnose_letter.py):

 F1  كسب الانحناء موضعي لا تراكمي خارج العُقَد:
        φ'(u_k) = φ(0) + Σ_{j<k} Δφ_j · (c إن كانت القطعة j خارج العقدة، وإلا 1)
     (بلا عقد = v1 تماماً: φ(0) + c·[φ(u) − φ(0)])
 F2  إشارة التفاف الذيل من انعطاف الذيل نفسه:  σ = sign(φ(1−f) − φ(1−2f))  بدل إشارة انعطاف الضربة كلها
 F3  حفظ انفتاح العين:  أكبر λ ∈ [0.3, 1] بحيث  r_hole/(w̃_loop/2) ≥ κ · (القيمة نفسها في النسخ) ,  κ = 1
     ويُضرب سُمك جزء العقدة وحده في λ.
 العقدة = أطول جزء [i, j] من الضربة يعود إلى نفسه: |γ_i − γ_j| < 0.8 w̃ ومسافة القوس بينهما > 4 w̃
"""
import math
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import style_model as SM

KAPPA = 1.0

def find_loop(c, w):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
    wm = np.median(w)
    D = np.linalg.norm(c[:, None] - c[None], axis=2)
    ii, jj = np.nonzero((D < 0.8 * wm) & ((s[None] - s[:, None]) > 4 * wm))
    if len(ii) == 0:
        return None
    k = int(np.argmax(s[jj] - s[ii]))
    return int(ii[k]), int(jj[k])

def _affine(p):
    s, rho, a = p[0], np.radians(p[1]), p[2]
    return np.array([[np.cos(rho), -np.sin(rho)], [np.sin(rho), np.cos(rho)]]) @ \
        np.array([[1, s], [0, 1]]) @ np.diag([np.exp(a / 2), np.exp(-a / 2)])

def apply_style_fixed(strokes, p, loops=None):
    s, rho, a, c, beta, f, S, gam, m, th = p
    A = _affine(p)
    wbar = np.mean(np.concatenate([w for _, w in strokes]))
    loops = loops if loops is not None else [find_loop(g, w) for g, w in strokes]
    out = []
    for k, (g, w) in enumerate(strokes):
        last = k == len(strokes) - 1
        d = np.diff(g, axis=0); ds = np.linalg.norm(d, axis=1)
        phi = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        u = np.linspace(0, 1, len(phi))
        gain = np.full(len(phi) - 1, c)
        if loops[k] is not None:                                  # F1
            i, j = loops[k]; gain[i:j] = 1.0
        phi2 = phi[0] + np.r_[0, np.cumsum(np.diff(phi) * gain)]
        if last:
            n = len(phi)
            a_, b_ = int((1 - 2 * f) * (n - 1)), int((1 - f) * (n - 1))
            loc = phi[b_] - phi[max(a_, 0)]                       # F2
            sigma = np.sign(loc) or np.sign(phi[-1] - phi[0]) or 1.0
            phi2 = phi2 + sigma * np.radians(beta) * SM.smoothstep((u - (1 - f)) / f)
        g2 = np.vstack([g[0], g[0] + np.cumsum(ds[:, None] * np.column_stack([np.cos(phi2), np.sin(phi2)]), axis=0)])
        if not last:
            uu = np.linspace(0, 1, len(g2))[:, None]
            g2 = g2 + uu * (g[-1] - g2[-1])
        g3 = g2 @ A.T
        t = np.gradient(g3, axis=0); ang = np.arctan2(t[:, 1], t[:, 0])
        U = np.linspace(0, 1, len(g3))
        tau = np.clip(np.minimum(U, 1 - U) / 0.04, 0, 1) ** 0.5
        w2 = tau * S * (w / wbar) ** gam * (m + np.abs(np.sin(ang - np.radians(th))))
        out.append((g3, w2))
    return out

def raster(styled, H=420, W=560, margin=0.1):
    P = np.vstack([g for g, _ in styled]); wm = max(w.max() for _, w in styled)
    lo, hi = P.min(0) - wm / 2, P.max(0) + wm / 2
    ctr, span = (lo + hi) / 2, hi - lo
    k = (1 - 2 * margin) * min(H / span[1], W / span[0])
    img = Image.new("L", (W, H), 0); dr = ImageDraw.Draw(img)
    for g, w in styled:
        X = np.column_stack([(g[:, 0] - ctr[0]) * k + W / 2, -(g[:, 1] - ctr[1]) * k + H / 2])
        r = w * k / 2; n = SM.normals(X)
        L, R = X + r[:, None] * n, X - r[:, None] * n
        for i in range(len(X) - 1):
            dr.polygon([tuple(L[i]), tuple(R[i]), tuple(R[i + 1]), tuple(L[i + 1])], fill=255)
    return np.asarray(img) > 127, k

def openness(styled, k_stroke, loop):
    m, k = raster(styled)
    holes = ndimage.binary_fill_holes(m) & ~m
    r = ndimage.distance_transform_edt(holes).max() / k if holes.any() else 0.0
    i, j = loop
    return r / (np.median(styled[k_stroke][1][i:j + 1]) / 2)

def preserve_counters(naskh, styled, loops):
    """F3"""
    styled = [(g, w.copy()) for g, w in styled]
    info = []
    for k, lp in enumerate(loops):
        if lp is None:
            continue
        ref = openness(naskh, k, lp)
        def with_l(l):
            st = [(g, w.copy()) for g, w in styled]
            st[k][1][lp[0]:lp[1] + 1] *= l
            return st
        before = openness(styled, k, lp)
        lo, hi = 0.3, 1.0
        if before < KAPPA * ref:
            for _ in range(14):
                mid = (lo + hi) / 2
                if openness(with_l(mid), k, lp) >= KAPPA * ref:
                    lo = mid
                else:
                    hi = mid
            styled = with_l(lo); lam = lo
        else:
            lam = 1.0
        info.append(dict(stroke=k, loop=lp, naskh_openness=round(ref, 3), before=round(before, 3),
                         factor=round(lam, 3), after=round(openness(styled, k, lp), 3)))
    return styled, info
