"""
التحليل المرن للأشكال (Srivastava et al.):  q(t) = γ'(t) / √‖γ'(t)‖

- المسافة المرنة:  d(γ0, γ1) = min_{O, ψ} arccos ⟨q0, √ψ'·O q1∘ψ⟩   (لا تتأثر بالإزاحة والحجم والدوران وإعادة التوسيط)
  ψ: إعادة توسيط مثلى بالبرمجة الديناميكية،  O: دوران مثالي (Procrustes)
- الانتقال بين شكلين: أقصر مسار (geodesic) على الكرة:
  q_α = [sin((1−α)θ) q0 + sin(αθ) q1] / sin θ ,  θ = arccos⟨q0, q1⟩
"""
import numpy as np

def resample(c, n):
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
    u = np.linspace(0, seg[-1], n)
    return np.column_stack([np.interp(u, seg, c[:, 0]), np.interp(u, seg, c[:, 1])])

def to_q(c):
    """→ q مُطبَّعة (‖q‖=1)، وطول المنحنى، ونقطة البداية."""
    n = len(c)
    d = np.gradient(c, 1 / (n - 1), axis=0)
    sp = np.linalg.norm(d, axis=1)
    q = d / np.sqrt(np.maximum(sp, 1e-9))[:, None]
    nrm = np.sqrt(np.trapezoid((q ** 2).sum(1), dx=1 / (n - 1)))
    return q / nrm, nrm ** 2, c[0]

def from_q(q, length, start):
    n = len(q)
    v = q * np.linalg.norm(q, axis=1, keepdims=True)
    c = np.vstack([np.zeros(2), np.cumsum(0.5 * (v[1:] + v[:-1]) / (n - 1), axis=0)])
    return start + c * length

def inner(q0, q1):
    return np.trapezoid((q0 * q1).sum(1), dx=1 / (len(q0) - 1))

MOVES = [(1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)]

def dp_reparam(q0, q1):
    """ψ الأمثل الذي يعظّم ⟨q0, √ψ' q1∘ψ⟩ (برمجة ديناميكية على شبكة N×N)."""
    n = len(q0); t = np.linspace(0, 1, n)
    E = np.full((n, n), -np.inf); E[0, 0] = 0
    back = np.zeros((n, n, 2), int)
    for i in range(1, n):
        for j in range(1, n):
            for di, dj in MOVES:
                k, l = i - di, j - dj
                if k < 0 or l < 0 or E[k, l] == -np.inf:
                    continue
                slope = (t[j] - t[l]) / (t[i] - t[k])
                ii = np.arange(k, i + 1)
                jj = t[l] + (t[ii] - t[k]) * slope
                q1i = np.column_stack([np.interp(jj, t, q1[:, 0]), np.interp(jj, t, q1[:, 1])])
                val = np.trapezoid((q0[ii] * q1i).sum(1), t[ii]) * np.sqrt(slope)
                if E[k, l] + val > E[i, j]:
                    E[i, j] = E[k, l] + val; back[i, j] = (k, l)
    path = [(n - 1, n - 1)]
    while path[-1] != (0, 0):
        path.append(tuple(back[path[-1]]))
    path = np.array(path[::-1])
    return np.interp(t, t[path[:, 0]], t[path[:, 1]])

def apply_reparam(c, psi):
    t = np.linspace(0, 1, len(c))
    return np.column_stack([np.interp(psi, t, c[:, 0]), np.interp(psi, t, c[:, 1])])

def procrustes(q0, q1):
    M = (q0.T @ q1); U, _, Vt = np.linalg.svd(M)
    O = U @ Vt
    if np.linalg.det(O) < 0:
        U[:, -1] *= -1; O = U @ Vt
    return O

def elastic_distance(c0, c1, n=70, rotate=True):
    """المسافة المرنة (بالراديان، 0 = الشكل نفسه، π/2 = متعامد تماماً)."""
    c0, c1 = resample(c0, n), resample(c1, n)
    q0, _, _ = to_q(c0)
    q1, _, _ = to_q(c1)
    for _ in range(2):
        if rotate:
            q1 = q1 @ procrustes(q0, q1).T
        psi = dp_reparam(q0, q1)
        c1 = apply_reparam(c1, psi)
        q1, _, _ = to_q(c1)
    if rotate:
        q1 = q1 @ procrustes(q0, q1).T
    return float(np.arccos(np.clip(inner(q0, q1), -1, 1)))

def geodesic(c0, c1, alphas, n=160, w0=None, w1=None):
    """إطارات الانتقال من c0 إلى c1 (بلا تدوير: الاتجاه جزء من الأسلوب)، مع استيفاء السُمك على التطابق الأمثل."""
    c0r, c1r = resample(c0, n), resample(c1, n)
    q0, L0, s0 = to_q(c0r)
    q1, L1, s1 = to_q(c1r)
    psi = dp_reparam(q0, q1)
    c1m = apply_reparam(c1r, psi)
    q1, L1, s1 = to_q(c1m)
    th = np.arccos(np.clip(inner(q0, q1), -1, 1))
    t = np.linspace(0, 1, n)
    if w0 is not None:
        w0r = np.interp(t, np.linspace(0, 1, len(w0)), w0)
        w1r = np.interp(psi, np.linspace(0, 1, len(w1)), w1)
    frames = []
    for a in alphas:
        if th < 1e-6:
            qa = q0
        else:
            qa = (np.sin((1 - a) * th) * q0 + np.sin(a * th) * q1) / np.sin(th)
        qa = qa / np.sqrt(inner(qa, qa))
        La = L0 ** (1 - a) * L1 ** a
        ca = from_q(qa, La, (1 - a) * s0 + a * s1)
        wa = None if w0 is None else (1 - a) * w0r + a * w1r
        frames.append((ca, wa))
    return frames, float(th)
