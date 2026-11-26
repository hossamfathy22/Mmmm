"""
Sigma-Lognormal (Plamondon) لهيكل الحرف.

  v(t) = Σ_i D_i · Λ(t; t0_i, μ_i, σ_i) · (cos θ_i(t), sin θ_i(t))
  Λ(t) = 1/(σ√(2π)(t−t0)) · exp(−(ln(t−t0) − μ)² / 2σ²)
  θ_i(t) = θs_i + (θe_i − θs_i) · Φ((ln(t−t0_i) − μ_i)/σ_i)

الصورة ثابتة بلا زمن، فنستنتج التوقيت من «قانون القوة 2/3» (Lacquaniti 1983):
  سرعة اليد  v ∝ κ^(−1/3)   (أبطأ في المنعطفات)
ثم نلائم معاملات الدفعات على مواضع الهيكل.
"""
import math
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
NT = 600                                   # شبكة التكامل الزمني

def power_law_time(c, kappa0=2.0, ramp=0.06):
    """زمن كل نقطة من الهيكل:  dt = ds / v ,  v = (κ+κ0)^(-1/3) مع تباطؤ عند البداية والنهاية."""
    d = np.diff(c, axis=0); ds = np.linalg.norm(d, axis=1)
    s = np.r_[0, np.cumsum(ds)]; u = s / s[-1]
    t1 = np.gradient(c, s, axis=0); t2 = np.gradient(t1, s, axis=0)
    kappa = np.abs(t1[:, 0] * t2[:, 1] - t1[:, 1] * t2[:, 0])
    from scipy.ndimage import gaussian_filter1d
    kappa = gaussian_filter1d(kappa, 3)
    v = (kappa + kappa0) ** (-1 / 3)
    v *= np.clip(np.minimum(u, 1 - u) / ramp, 0.15, 1) ** 0.5
    vm = 0.5 * (v[1:] + v[:-1])
    t = np.r_[0, np.cumsum(ds / vm)]
    return t / t[-1]

class SigmaLognormal(torch.nn.Module):
    def __init__(self, n, c, t):
        super().__init__()
        # بداية: دفعات متساوية التباعد زمنياً، اتجاهها من مماس الهيكل
        tc = (np.arange(n) + 0.5) / n
        idx = np.clip(np.searchsorted(t, tc), 1, len(c) - 2)
        seg = np.linalg.norm(np.diff(c, axis=0), axis=1).sum()
        ang = np.unwrap(np.arctan2(*np.gradient(c, axis=0)[:, ::-1].T))
        lo = np.clip(np.searchsorted(t, tc - 0.5 / n), 0, len(c) - 1)
        hi = np.clip(np.searchsorted(t, tc + 0.5 / n), 0, len(c) - 1)
        sig0, mu0 = 0.35, math.log(0.6 / n)
        self.t0 = torch.nn.Parameter(torch.tensor(tc - math.exp(mu0 - sig0 ** 2)))
        self.mu = torch.nn.Parameter(torch.full((n,), mu0))
        self.ls = torch.nn.Parameter(torch.full((n,), math.log(sig0)))
        self.lD = torch.nn.Parameter(torch.full((n,), math.log(seg / n)))
        self.ths = torch.nn.Parameter(torch.tensor(ang[lo]))
        self.the = torch.nn.Parameter(torch.tensor(ang[hi]))
        self.p0 = torch.nn.Parameter(torch.tensor(c[0]))

    def params(self):
        return dict(t0=self.t0, mu=self.mu, sigma=self.ls.exp(), D=self.lD.exp(), ths=self.ths, the=self.the)

    def trajectory(self, t_query, P=None):
        """المواضع عند الأزمنة t_query، ومتجه السرعة عند الشبكة. P: معاملات (قد تكون معدّلة بالأسلوب)."""
        P = P or self.params()
        T = torch.linspace(0, 1, NT)
        dt = T[:, None] - P["t0"][None]
        ok = dt > 1e-9
        ldt = torch.log(torch.where(ok, dt, torch.ones_like(dt)))
        z = (ldt - P["mu"]) / P["sigma"]
        lam = torch.where(ok, torch.exp(-0.5 * z ** 2) / (P["sigma"] * math.sqrt(2 * math.pi) * torch.where(ok, dt, torch.ones_like(dt))), torch.zeros_like(dt))
        Phi = torch.where(ok, 0.5 * (1 + torch.erf(z / math.sqrt(2))), torch.zeros_like(dt))
        th = P["ths"] + (P["the"] - P["ths"]) * Phi
        sp = P["D"] * lam
        v = torch.stack([(sp * torch.cos(th)).sum(1), (sp * torch.sin(th)).sum(1)], 1)
        A = P.get("A")
        if A is not None:
            v = v @ A.T
        pos = torch.cat([torch.zeros(1, 2), torch.cumsum(0.5 * (v[1:] + v[:-1]) * (1 / (NT - 1)), 0)])
        p0 = P.get("p0", self.p0)
        pos = pos + p0
        # استيفاء خطي عند t_query
        tq = torch.as_tensor(t_query)
        f = tq * (NT - 1); i0 = torch.clamp(f.floor().long(), 0, NT - 2); a = (f - i0)[:, None]
        return pos[i0] * (1 - a) + pos[i0 + 1] * a, v

def fit_stroke(c, n, iters=2500, verbose=False):
    t = power_law_time(c)
    m = SigmaLognormal(n, c, t)
    target = torch.tensor(c)
    tq = torch.tensor(t)
    opt = torch.optim.Adam(m.parameters(), lr=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    for k in range(iters):
        opt.zero_grad()
        p, _ = m.trajectory(tq)
        loss = ((p - target) ** 2).sum(1).mean() + 1e-4 * ((m.ls.exp() - 0.35) ** 2).sum()
        loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        p, _ = m.trajectory(tq)
        err = (p - target).norm(dim=1)
        snr = 10 * math.log10(float(((target - target.mean(0)) ** 2).sum(1).mean()) / float((err ** 2).mean()))
    return m, t, snr, float(err.max())

def fit_letter(strokes, snr_goal=30.0, err_goal=0.012, nmax=9):
    """أقل عدد دفعات يحقق SNR ≥ snr_goal dB وأقصى خطأ ≤ err_goal من ارتفاع الحرف (مبدأ البساطة)."""
    out = []
    for c, w in strokes:
        best = None
        for n in range(2, nmax + 1):
            m, t, snr, emax = fit_stroke(c, n)
            best = (m, t, snr, emax, n)
            if snr >= snr_goal and emax <= err_goal:
                break
        out.append(dict(model=best[0], t=best[1], snr=best[2], err_max=best[3], n=best[4], c=c, w=w))
    return out
