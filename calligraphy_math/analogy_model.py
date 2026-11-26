"""
تحويل أي حرف نسخ إلى الأسلوب الحديث بثلاث طبقات — التدريب على الجيم فقط.

الطبقة 1 (اليد والأداة، لكل الحروف):
    γ_A = A·γ                                      (A: تحويل أفيني عام بالمربعات الصغرى على نقاط الجيم المتطابقة)
    log w = log S + γ·log(w_n/w̄) + log(m + |sin(φ − θ)|) + Σ_{k=1,2} a_k cos kφ + b_k sin kφ

الطبقة 2 (النقل بالقياس بين العناصر):
    لكل عنصر e في الحرف الجديد:  k* = أقرب عنصر في الجيم بالمسافة المرنة d (SRVF، تراعي الاتجاه)
    α_e = exp(−d² / 2λ²)
    φ'(u) = φ_A(u) + α_e·Δφ_k*(ψ(u)) ,  |γ'(u)| = |γ_A'(u)|·exp(α_e·ρ_k*(ψ(u))) ,  log w += α_e·r_k*(ψ(u))
    (Δφ: تغيّر اتجاه المماس، ρ: تغيّر السرعة/الطول، r: بقية السُمك — تمثيل SRVF للتحوّل)

الطبقة 3 (الثقة):  لون كل عنصر بحسب α_e.

الاختبارات (عتبات محددة قبل التشغيل):
    T1 ثبات الأداة داخل الجيم: قانون القلم من الرأس يتنبأ بسُمك البطن (R² ≥ 0.5) والعكس، |Δθ| ≤ 15°.
    T2 مصدر لم يُرَ: جيم المستخدم (نسخ بيد أخرى) → التطابق مع الجيم الحديثة IoU ≥ 0.75.
"""
import sys, os, json, math, subprocess, tempfile
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.optimize import least_squares, differential_evolution, minimize

UP = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "results_analogy"
sys.argv = [sys.argv[0], UP]
import style_model as SM
import srvf

PDF = f"{UP}/88dd7f5d-sakkal_arabic_alphabet_copy.pdf"
USER_JEEM = f"{UP}/83a4afbb-image.jpg"
MODERN_JEEM = f"{UP}/f0b4c74c-image.jpg"
LAM = 0.25                       # مقياس الشبه (راديان)
NS = 160

# ---------------- بيانات الحروف: قص من جدول سكّال (1200 dpi) + هياكل بترتيب الكتابة (من أسهم الجدول) ----------------
LETTERS = {
    "jeem": dict(crop=(7700, 4420, 560, 620),
                 guides=[[(84, 157), (100, 125), (170, 103), (280, 101), (384, 99)],
                         [(262, 142), (205, 190), (152, 258), (120, 340), (114, 420), (137, 490),
                          (192, 530), (280, 548), (380, 545), (485, 534)]],
                 elements=[(0, 0.0, 1.0, "head"), (1, 0.0, 0.22, "neck"), (1, 0.22, 1.0, "bowl")], ndots=1),
    "ha": dict(crop=(7720, 5080, 560, 620),
               guides=[[(66, 117), (82, 75), (150, 57), (260, 60), (360, 62)],
                       [(330, 92), (260, 112), (180, 170), (125, 250), (100, 340), (110, 420), (160, 480),
                        (250, 515), (340, 518), (440, 503), (500, 497)]],
               elements=[(0, 0.0, 1.0, "head"), (1, 0.0, 0.22, "neck"), (1, 0.22, 1.0, "bowl")], ndots=0),
    "nun": dict(crop=(7780, 16850, 400, 560),
                guides=[[(262, 212), (290, 270), (300, 330), (295, 390), (250, 440), (150, 470),
                         (80, 445), (55, 385), (70, 305)]],
                elements=[(0, 0.0, 0.25, "stem"), (0, 0.25, 1.0, "nun-bowl")], ndots=1),
    "sad": dict(crop=(7560, 10020, 720, 520),
                guides=[[(382, 232), (430, 182), (500, 126), (570, 103), (617, 128), (614, 176), (558, 214),
                         (480, 238), (382, 250)],
                        [(330, 166), (346, 222), (350, 305), (328, 378), (265, 425), (185, 438), (116, 395),
                         (96, 330), (113, 256)]],
                elements=[(0, 0.0, 1.0, "loop"), (1, 0.0, 0.25, "stem"), (1, 0.25, 1.0, "nun-bowl")], ndots=0),
}
USER_JEEM_GUIDES = [[(198, 272), (220, 230), (250, 217), (350, 226), (460, 238)],
                    [(450, 255), (370, 275), (280, 330), (225, 410), (210, 480), (235, 540),
                     (300, 580), (400, 582), (480, 570), (545, 547)]]
MODERN_GUIDES = [[(238, 232), (255, 290), (300, 308), (380, 310), (435, 318), (455, 360), (460, 420)],
                 [(455, 422), (350, 390), (260, 365), (220, 365), (200, 390), (195, 418), (205, 470),
                  (255, 515), (310, 545), (390, 547), (460, 518), (520, 503), (575, 505), (618, 540)]]

# ---------------- الاستخراج ----------------
def pdf_rgb(crop):
    x, y, W, H = crop
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["pdftoppm", "-r", "1200", "-png", "-x", str(x), "-y", str(y), "-W", str(W), "-H", str(H),
                        PDF, f"{d}/c"], check=True)
        return np.asarray(Image.open(f"{d}/c-1.png").convert("RGB")).astype(int)

def split_mask(ink):
    """→ جسم الحرف (أكبر مكوّن) + النقاط (مركز، نصف القطر)."""
    lab, k = ndimage.label(ink)
    sz = ndimage.sum(ink, lab, range(1, k + 1))
    big = 1 + int(np.argmax(sz))
    dots = []
    for i in range(1, k + 1):
        if i != big and sz[i - 1] > 0.02 * sz.max():
            ys, xs = np.nonzero(lab == i)
            dots.append((np.array([xs.mean(), ys.mean()]), 0.5 * max(np.ptp(xs), np.ptp(ys))))
    return lab == big, dots

class Glyph:
    def __init__(self, name, body, dots, guides, elements, unit=None):
        self.name, self.elements = name, elements
        ys, xs = np.nonzero(body)
        self.H = float(ys.max() - ys.min())
        self.unit = unit or self.H
        self.ctr = np.array([(xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2])
        self.strokes = []
        for g in guides:
            c, w = SM.refine(body, SM.spline(g, NS))
            e = int(0.12 * len(w))
            w[:e] = np.minimum(w[:e], w[e]); w[-e:] = np.minimum(w[-e:], w[-e - 1])
            self.strokes.append((self.norm(c), np.maximum(w, 1.0) / self.unit))
        self.dots = [(self.norm(c[None])[0], r / self.unit) for c, r in dots]
        self.wbar = float(np.mean(np.concatenate([w for _, w in self.strokes])))

    def norm(self, p):
        q = (p - self.ctr) / self.unit
        q[:, 1] *= -1
        return q

def load_pdf_glyph(name, unit=None):
    a = pdf_rgb(LETTERS[name]["crop"])
    ink = (a.max(2) < 100) & ((a.max(2) - a.min(2)) < 40)
    ink = ndimage.binary_opening(ink, structure=np.ones((11, 1)))
    body, dots = split_mask(ink)
    dots = sorted(dots, key=lambda d: -d[1])[:LETTERS[name]["ndots"]]     # نقاط الحرف نفسه فقط (لا نقطة حرف الصف المجاور)
    return Glyph(name, body, dots, LETTERS[name]["guides"], LETTERS[name]["elements"], unit)

def load_img_glyph(path, guides, elements, name):
    ink = SM.ink_mask(path)
    body, dots = split_mask(ink)
    return Glyph(name, body, dots, guides, elements)

def modern_glyph():
    a = np.asarray(Image.open(MODERN_JEEM).convert("RGB")).astype(int)
    ink = SM.ink_mask(MODERN_JEEM)
    teal = (a[..., 1] - a[..., 0] > 40) & (a[..., 2] - a[..., 0] > 30)
    teal = ndimage.binary_opening(teal, iterations=4)
    ys, xs = np.nonzero(ink)
    G = Glyph.__new__(Glyph)
    G.name, G.elements = "modern_jeem", LETTERS["jeem"]["elements"]
    G.H = float(ys.max() - ys.min()); G.unit = G.H
    G.ctr = np.array([(xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2])
    G.strokes = []
    for g in MODERN_GUIDES:
        c, w = SM.refine(ink, SM.spline(g, NS))
        G.strokes.append((G.norm(c), np.maximum(w, 1.0) / G.unit))
    ty, tx = np.nonzero(teal)
    G.dots = [(G.norm(np.array([[tx.mean(), ty.mean()]]))[0], 0.5 * max(np.ptp(tx), np.ptp(ty)) / G.unit)]
    G.wbar = float(np.mean(np.concatenate([w for _, w in G.strokes])))
    G.mask = ink
    return G

# ---------------- أدوات المنحنيات ----------------
def arclen(c):
    return np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]

def sub(c, w, u0, u1, n=100):
    s = arclen(c); u = s / s[-1]
    v = np.linspace(u0, u1, n)
    return (np.column_stack([np.interp(v, u, c[:, 0]), np.interp(v, u, c[:, 1])]), np.interp(v, u, w))

def tangent(c):
    d = np.gradient(c, axis=0)
    return np.unwrap(np.arctan2(d[:, 1], d[:, 0])), np.linalg.norm(d, axis=1)

def match(c0, c1, n=90):
    """ψ: معامل طول القوس في c0 → المعامل المقابل في c1 (برمجة ديناميكية SRVF، بلا تدوير)."""
    a, b = srvf.resample(c0, n), srvf.resample(c1, n)
    q0, _, _ = srvf.to_q(a); q1, _, _ = srvf.to_q(b)
    psi = srvf.dp_reparam(q0, q1)
    return np.linspace(0, 1, n), psi

def elastic(c0, c1):
    return srvf.elastic_distance(c0, c1, n=70, rotate=False)

# ---------------- التعلم من الجيم ----------------
def fit_affine(X, Y):
    """Y ≈ A X + t  (مربعات صغرى)."""
    Xh = np.hstack([X, np.ones((len(X), 1))])
    M, *_ = np.linalg.lstsq(Xh, Y, rcond=None)
    return M[:2].T, M[2]

def decompose(A):
    """A = R(ρ)·[[1,s],[0,1]]·diag(sx,sy)"""
    Q, R = np.linalg.qr(A)
    D = np.diag(np.sign(np.diag(R))); Q, R = Q @ D, D @ R
    rho = math.degrees(math.atan2(Q[1, 0], Q[0, 0]))
    sx, sy = R[0, 0], R[1, 1]; s = R[0, 1] / sy
    return dict(rotation_deg=round(rho, 1), shear=round(s, 3), width_scale=round(sx, 3), height_scale=round(sy, 3),
                width_to_height=round(sx / sy, 3))

def pen_features(phi, wn_rel, p, fourier=True):
    logS, gam, th, logm = p[:4]
    out = logS + gam * np.log(wn_rel) + np.log(np.exp(logm) + np.abs(np.sin(phi - th)))
    if fourier:
        a1, b1, a2, b2 = p[4:8]
        out = out + a1 * np.cos(phi) + b1 * np.sin(phi) + a2 * np.cos(2 * phi) + b2 * np.sin(2 * phi)
    return out

def fit_pen(phi, wn_rel, logw, fourier=True):
    best = None
    for th0 in np.radians(np.arange(0, 180, 20)):
        p0 = [np.mean(logw), 1.0, th0, math.log(0.2)] + ([0, 0, 0, 0] if fourier else [])
        lo = [-10, 0, -10, math.log(0.01)] + ([-3] * 4 if fourier else [])
        hi = [10, 4, 10, math.log(3)] + ([3] * 4 if fourier else [])
        r = least_squares(lambda p: pen_features(phi, wn_rel, p, fourier) - logw, p0, bounds=(lo, hi))
        if best is None or r.cost < best.cost:
            best = r
    return best.x

def r2(y, yhat):
    return 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()

class Style:
    """كل ما نتعلمه من زوج (جيم نسخ، جيم حديثة)."""
    def __init__(self, nas, mod):
        self.nas, self.mod = nas, mod
        # 1) تطابق الضربات بـ SRVF ثم A ثم إعادة التطابق بعد A
        A, t = np.eye(2), np.zeros(2)
        for _ in range(2):
            X, Y, self.psi = [], [], []
            for (cn, _), (cm, _) in zip(nas.strokes, mod.strokes):
                u, psi = match(cn @ A.T + t, cm)
                self.psi.append((u, psi))
                X.append(sub(cn, cn[:, 0], 0, 1, len(u))[0]); Y.append(self._at(cm, psi))
            A, t = fit_affine(np.vstack(X), np.vstack(Y))
        self.A, self.t = A, t
        # 2) العينات المتطابقة: الاتجاه الحديث، سُمك النسخ، سُمك الحديث
        self.samples = []
        for j, ((cn, wn), (cm, wm)) in enumerate(zip(nas.strokes, mod.strokes)):
            u, psi = self.psi[j]
            cmm = self._at(cm, psi)
            phi_m, _ = tangent(cmm)
            wn_u = np.interp(u, arclen(cn) / arclen(cn)[-1], wn)
            wm_u = np.interp(psi, arclen(cm) / arclen(cm)[-1], wm)
            keep = (u > 0.06) & (u < 0.94)
            self.samples.append(dict(stroke=j, u=u, phi=phi_m, wn=wn_u / nas.wbar, wm=wm_u, keep=keep, cmm=cmm))
        phi = np.concatenate([s["phi"][s["keep"]] for s in self.samples])
        wn = np.concatenate([s["wn"][s["keep"]] for s in self.samples])
        lw = np.log(np.concatenate([s["wm"][s["keep"]] for s in self.samples]))
        self.pen = fit_pen(phi, wn, lw)
        self.pen_r2 = r2(lw, pen_features(phi, wn, self.pen))
        # 3) تحوّل كل عنصر:  Δφ(v), ρ(v), r(v)
        self.elem = {}
        for (j, u0, u1, name) in nas.elements:
            cn, wn = nas.strokes[j]; s = self.samples[j]
            m = (s["u"] >= u0 - 1e-9) & (s["u"] <= u1 + 1e-9)
            cA = (sub(cn, wn, 0, 1, len(s["u"]))[0] @ self.A.T + self.t)[m]
            cM = s["cmm"][m]
            phA, spA = tangent(cA); phM, spM = tangent(cM)
            dphi = phM - phA
            dphi -= 2 * math.pi * round(float(np.median(dphi)) / (2 * math.pi))
            rho = np.log(np.maximum(spM, 1e-6) / np.maximum(spA, 1e-6))
            pred = pen_features(phM, s["wn"][m], self.pen)
            res = np.log(s["wm"][m]) - pred
            sm = lambda x: ndimage.gaussian_filter1d(x, 2, mode="nearest")
            self.elem[name] = dict(v=np.linspace(0, 1, m.sum()), dphi=sm(dphi), rho=sm(np.clip(rho, -2, 2)),
                                   res=sm(np.clip(res, -3, 3)), naskh_curve=sub(cn, wn, u0, u1, 100)[0],
                                   d0=cM[0] - cA[0])                     # إزاحة بداية العنصر بعد A
        # 4) حجم النقطة
        if nas.dots and mod.dots:
            self.dot_ratio = (mod.dots[0][1] / np.median(mod.strokes[1][1])) / (nas.dots[0][1] / np.median(nas.strokes[1][1]))
        else:
            self.dot_ratio = 1.0

    @staticmethod
    def _at(c, psi):
        s = arclen(c); u = s / s[-1]
        return np.column_stack([np.interp(psi, u, c[:, 0]), np.interp(psi, u, c[:, 1])])

    def tool_consistency(self):
        """T1: قانون القلم (بلا حدود فورييه) من ضربة يتنبأ بالأخرى."""
        out = {}
        S0, S1 = self.samples
        data = [(s["phi"][s["keep"]], s["wn"][s["keep"]], np.log(s["wm"][s["keep"]])) for s in (S0, S1)]
        p0 = fit_pen(*data[0], fourier=False); p1 = fit_pen(*data[1], fourier=False)
        th = lambda p: math.degrees(p[2]) % 180
        dth = abs(th(p0) - th(p1)); dth = min(dth, 180 - dth)
        out["theta_from_head_deg"] = round(th(p0), 1); out["theta_from_bowl_deg"] = round(th(p1), 1)
        out["delta_theta_deg"] = round(dth, 1)
        out["R2_head_to_bowl"] = round(float(r2(data[1][2], pen_features(*data[1][:2], p0, False))), 3)
        out["R2_bowl_to_head"] = round(float(r2(data[0][2], pen_features(*data[0][:2], p1, False))), 3)
        out["R2_self_head"] = round(float(r2(data[0][2], pen_features(*data[0][:2], p0, False))), 3)
        out["R2_self_bowl"] = round(float(r2(data[1][2], pen_features(*data[1][:2], p1, False))), 3)
        out["PASS"] = bool(out["R2_head_to_bowl"] >= 0.5 and out["R2_bowl_to_head"] >= 0.5 and dth <= 15)
        return out

    # ---------------- التطبيق على أي حرف ----------------
    def apply(self, G):
        report = []
        new_strokes, elem_spans = [], []
        for j, (c, w) in enumerate(G.strokes):
            n = len(c)
            s_arc = arclen(c); u = s_arc / s_arc[-1]
            cA = c @ self.A.T + self.t
            phA, spA = tangent(cA)
            dphi, rho, res, alpha_u = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
            start_shift, start_alpha = np.zeros(2), 0.0
            for (jj, u0, u1, name) in G.elements:
                if jj != j:
                    continue
                m = (u >= u0 - 1e-9) & (u <= u1 + 1e-9)
                ce = sub(c, w, u0, u1, 100)[0]
                dists = {k: elastic(ce, e["naskh_curve"]) for k, e in self.elem.items()}
                k = min(dists, key=dists.get)
                a = math.exp(-dists[k] ** 2 / (2 * LAM ** 2))
                v_new, psi = match(ce, self.elem[k]["naskh_curve"])
                ve = (u[m] - u0) / max(u1 - u0, 1e-9)
                pk = np.interp(ve, v_new, psi)
                E = self.elem[k]
                dphi[m] = a * np.interp(pk, E["v"], E["dphi"])
                rho[m] = a * np.interp(pk, E["v"], E["rho"])
                res[m] = a * np.interp(pk, E["v"], E["res"])
                alpha_u[m] = a
                if u0 == 0.0:
                    start_shift, start_alpha = a * E["d0"], a
                report.append(dict(letter=G.name, element=name, nearest_jeem_element=k,
                                   distance_rad=round(dists[k], 3), alpha=round(a, 3),
                                   all_distances={kk: round(vv, 3) for kk, vv in dists.items()}))
                elem_spans.append((j, u0, u1, a))
            # الحفاظ على اتجاه بداية الضربة وفق الطبقة 1 ثم إضافة تحوّل العنصر
            phi = phA + dphi
            sp = spA * np.exp(rho)
            v = np.column_stack([sp * np.cos(phi), sp * np.sin(phi)])
            p0 = cA[0] + start_shift
            cN = np.vstack([p0, p0 + np.cumsum(0.5 * (v[1:] + v[:-1]), axis=0)])
            phN, _ = tangent(cN)
            x = np.clip(np.minimum(u, 1 - u) / 0.06, 0, 1)
            taper = 0.08 + 0.92 * x * x * (3 - 2 * x)                    # استدقاق الأطراف (الأسلوب الحديث)
            logw = pen_features(phN, w / G.wbar, self.pen) + res
            new_strokes.append([cN, np.exp(logw) * taper, alpha_u, cA, start_alpha])
        # قيود الالتقاء: كل ضربة لاحقة تُزاح لتبقى ملتصقة بما قبلها (الإزاحة المحفوظة = A·الإزاحة الأصلية)
        for b in range(1, len(G.strokes)):
            cb = G.strokes[b][0]
            best = None
            for a_ in range(b):
                ca = G.strokes[a_][0]
                for end in (0, len(cb) - 1):
                    d = np.linalg.norm(ca - cb[end], axis=1); i = int(np.argmin(d))
                    if best is None or d[i] < best[0]:
                        best = (d[i], a_, i, end)
            # أو طرف ضربة سابقة يلامس هذه الضربة
            for a_ in range(b):
                ca = G.strokes[a_][0]
                for end in (0, len(ca) - 1):
                    d = np.linalg.norm(cb - ca[end], axis=1); i = int(np.argmin(d))
                    if d[i] < best[0]:
                        best = (d[i], a_, end, i, "rev")
            if best[0] < 0.08:
                if len(best) == 4:
                    _, a_, i, end = best
                    target = new_strokes[a_][0][i] + self.A @ (cb[end] - G.strokes[a_][0][i])
                    shift = target - new_strokes[b][0][end]
                else:
                    _, a_, end, i, _ = best
                    target = new_strokes[a_][0][end] - self.A @ (G.strokes[a_][0][end] - cb[i])
                    shift = target - new_strokes[b][0][i]
                new_strokes[b][0] = new_strokes[b][0] + (1 - new_strokes[b][4]) * shift
        # النقاط: تتبع إزاحة أقرب نقطة هيكل، وحجمها بنسبة الأسلوب
        dots = []
        for p, r in G.dots:
            best = None
            for (cN, wN, _, cA0, _sa), (c, _) in zip(new_strokes, G.strokes):
                d = np.linalg.norm(c - p, axis=1); i = int(np.argmin(d))
                if best is None or d[i] < best[0]:
                    best = (d[i], cN[i] - cA0[i], np.median(wN) / np.median(_w(G)))
            pA = self.A @ p + self.t + best[1]
            dots.append((pA, r * self.dot_ratio * best[2]))
        return dict(strokes=[(s[0], s[1], s[2]) for s in new_strokes], dots=dots, report=report)

def _w(G):
    return np.concatenate([w for _, w in G.strokes])

# ---------------- الرسم ----------------
def draw(result, H=300, W=360, margin=0.1, scale=2, color_by_alpha=False, naskh=None):
    if naskh is not None:
        strokes = [(c, w, np.zeros(len(c))) for c, w in naskh.strokes]; dots = naskh.dots
    else:
        strokes, dots = result["strokes"], result["dots"]
    P = np.vstack([s[0] for s in strokes]); wm = max(float(s[1].max()) for s in strokes)
    lo, hi = P.min(0) - wm / 2, P.max(0) + wm / 2
    for p, r in dots:
        lo = np.minimum(lo, p - r); hi = np.maximum(hi, p + r)
    ctr, span = (lo + hi) / 2, hi - lo
    k = (1 - 2 * margin) * min(H / span[1], W / span[0])
    img = Image.new("RGB", (W * scale, H * scale), (255, 255, 255)); dr = ImageDraw.Draw(img)
    tf = lambda p: ((p[:, 0] - ctr[0]) * k * scale + W * scale / 2, (-(p[:, 1] - ctr[1])) * k * scale + H * scale / 2)
    def col(a):
        if not color_by_alpha:
            return (17, 17, 17)
        return (46, 140, 87) if a >= 0.7 else (214, 162, 32) if a >= 0.2 else (150, 150, 150)
    for c, w, al in strokes:
        x, y = tf(c); X = np.column_stack([x, y]); r = w * k * scale / 2
        nrm = SM.normals(X); L, R = X + r[:, None] * nrm, X - r[:, None] * nrm
        for i in range(len(X) - 1):
            dr.polygon([tuple(L[i]), tuple(R[i]), tuple(R[i + 1]), tuple(L[i + 1])], fill=col(al[i]))
        for i in range(1, len(X) - 1, 2):
            if (X[i + 1] - X[i]) @ (X[i] - X[i - 1]) < 0.9 * np.linalg.norm(X[i + 1] - X[i]) * np.linalg.norm(X[i] - X[i - 1]):
                dr.ellipse([X[i, 0] - r[i], X[i, 1] - r[i], X[i, 0] + r[i], X[i, 1] + r[i]], fill=col(al[i]))
    for p, r in dots:
        x, y = tf(p[None]); rr = r * k * scale
        dr.polygon([(x[0], y[0] - rr), (x[0] + rr, y[0]), (x[0], y[0] + rr), (x[0] - rr, y[0])],
                   fill=(90, 158, 163) if not color_by_alpha else (150, 150, 150))
    return img

def iou_vs_target(result, target_mask, size=260):
    """IoU بعد أفضل محاذاة (حجم + إزاحة) — الحجم والموضع اعتباطيان في الصور."""
    T = SM.fit_canvas(target_mask, size=size)
    H, W = T.shape
    strokes = result["strokes"]
    P = np.vstack([s[0] for s in strokes]); h0 = np.ptp(P[:, 1])
    def render(x):
        k, tx, ty = x
        img = Image.new("L", (W, H), 0); dr = ImageDraw.Draw(img)
        for c, w, _ in strokes:
            X = np.column_stack([c[:, 0] / h0 * k * H + W / 2 + tx * H, -c[:, 1] / h0 * k * H + H / 2 + ty * H])
            r = w / h0 * k * H / 2; nrm = SM.normals(X)
            L, R = X + r[:, None] * nrm, X - r[:, None] * nrm
            for i in range(len(X) - 1):
                dr.polygon([tuple(L[i]), tuple(R[i]), tuple(R[i + 1]), tuple(L[i + 1])], fill=255)
        return np.asarray(img) > 127
    loss = lambda x: 1 - (render(x) & T).sum() / max((render(x) | T).sum(), 1)
    r = differential_evolution(loss, [(0.4, 1.2), (-0.3, 0.3), (-0.3, 0.3)], seed=0, popsize=15, maxiter=80, polish=False)
    x = minimize(loss, r.x, method="Nelder-Mead").x
    P_ = render(x)
    return float((P_ & T).sum() / (P_ | T).sum()), float(SM.edge_err(P_, T)), P_, T

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    jeem = load_pdf_glyph("jeem")
    U = jeem.unit                                          # وحدة مشتركة لكل حروف الجدول (نفس القلم والحجم)
    glyphs = {n: load_pdf_glyph(n, unit=U) for n in ("ha", "nun", "sad")}
    modern = modern_glyph()
    user_jeem = load_img_glyph(USER_JEEM, USER_JEEM_GUIDES, LETTERS["jeem"]["elements"], "user_jeem")

    st = Style(jeem, modern)
    summary = dict(hand=decompose(st.A), pen_law_R2_on_jeem=round(float(st.pen_r2), 3),
                   pen_params=dict(zip(["log S", "γ", "θ(rad)", "log m", "a1", "b1", "a2", "b2"],
                                       [round(float(x), 3) for x in st.pen])),
                   dot_ratio=round(float(st.dot_ratio), 3))
    summary["T1_tool_consistency"] = st.tool_consistency()

    res = {"jeem (training, capacity)": st.apply(jeem), "user_jeem (unseen source)": st.apply(user_jeem)}
    for n, G in glyphs.items():
        res[n] = st.apply(G)

    ev = {}
    for key in ("jeem (training, capacity)", "user_jeem (unseen source)"):
        i, e, P_, T = iou_vs_target(res[key], modern.mask)
        ev[key] = dict(IoU=round(i, 3), edge_err_pct=round(e, 2)); res[key]["eval"] = (P_, T)
    base = dict(strokes=[(c, w * 1.0, np.zeros(len(c))) for c, w in user_jeem.strokes])
    i, e, _, _ = iou_vs_target(base, modern.mask)
    ev["baseline: user Naskh jeem as-is"] = dict(IoU=round(i, 3), edge_err_pct=round(e, 2))
    summary["evaluation"] = ev
    summary["T2_unseen_source_PASS"] = bool(ev["user_jeem (unseen source)"]["IoU"] >= 0.75)
    summary["element_analogies"] = [r for k in ("ha", "nun", "sad", "user_jeem (unseen source)") for r in res[k]["report"]]
    json.dump(summary, open(f"{OUT}/summary.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))

    import pickle
    pickle.dump(dict(res={k: {kk: vv for kk, vv in v.items() if kk != "eval"} for k, v in res.items()},
                     glyphs=dict(jeem=jeem, user_jeem=user_jeem, modern=modern, **glyphs)),
                open(f"{OUT}/state.pkl", "wb"))

    # ---------------- الأشكال ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def ov(a, b):
        o = np.ones(a.shape + (3,))
        o[b & ~a] = (0.86, 0.3, 0.3); o[a & ~b] = (0.3, 0.45, 0.9); o[a & b] = (0.12, 0.12, 0.12)
        return o
    fig, ax = plt.subplots(1, 5, figsize=(19, 4.2), facecolor="white")
    ax[0].imshow(Image.open(MODERN_JEEM).convert("RGB")); ax[0].set_title("target: modern jeem", fontsize=9)
    for a_, key in zip(ax[1:3], ("jeem (training, capacity)", "user_jeem (unseen source)")):
        a_.imshow(draw(res[key])); a_.set_title(f"{key}", fontsize=9)
    for a_, key in zip(ax[3:5], ("jeem (training, capacity)", "user_jeem (unseen source)")):
        P_, T = res[key]["eval"]; a_.imshow(ov(P_, T))
        a_.set_title(f"{key}\nIoU {ev[key]['IoU']:.2f} · edge {ev[key]['edge_err_pct']:.1f}% H", fontsize=9)
    for a_ in ax: a_.axis("off")
    fig.suptitle("Tests on jeem — grey = match · red = missed · blue = extra", fontsize=10)
    fig.savefig(f"{OUT}/1_jeem_tests.png", dpi=115, bbox_inches="tight")

    names = {"ha": "ha  ح", "nun": "nun  ن", "sad": "sad  ص"}
    fig, ax = plt.subplots(3, 3, figsize=(13, 12.5), facecolor="white")
    for r_, n in enumerate(("ha", "nun", "sad")):
        ax[r_, 0].imshow(draw(None, naskh=glyphs[n])); ax[r_, 0].set_title(f"{n}: Naskh input (Sakkal)", fontsize=10)
        ax[r_, 1].imshow(draw(res[n])); ax[r_, 1].set_title(f"{n}: predicted modern", fontsize=10)
        ax[r_, 2].imshow(draw(res[n], color_by_alpha=True))
        ttl = ", ".join(f"{d['element']}→{d['nearest_jeem_element']} α={d['alpha']:.2f}" for d in res[n]["report"])
        ax[r_, 2].set_title(f"confidence: {ttl}", fontsize=8)
        for a_ in ax[r_]: a_.axis("off")
    fig.suptitle("green = transferred from a matching jeem element (α≥0.7) · yellow = partial · grey = tool & hand only",
                 fontsize=10)
    fig.savefig(f"{OUT}/2_predictions.png", dpi=110, bbox_inches="tight")
    print("saved")
