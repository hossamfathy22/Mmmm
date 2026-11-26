"""
المحرّك العام: حرف هدف واحد → تحويل رياضي → كل الحروف.

  التحويل = أسلوب عام g (ينتقل لكل الحروف)  +  بقية خاصة بالحرف r (تنتقل فقط بالقياس)

الأسلوب العام g (18 معاملاً):
  الهيكل:  φ'(u) = φ(0) + c·[φ(u) − φ(0)]
                 + σ_main·β·R((u−1+f)/f)        التفاف ذيل الضربة الرئيسة
                 + σ_k·η·R((u−1+h)/h)           خطّاف نهاية الضربات الثانوية
           γ' = ∫(cos φ', sin φ') ds ,  ثم  A(s, ρ, a)
           استمرار القلم:  نقطة اتصال الضربة b بالضربة a تنتقل على a نحو نهايتها بمقدار κ
  السُمك:  w = τ_ℓ(u) · S · (w_n/w̄)^γ · (m + |sin(φ − θ)|) · exp(Σ a_k cos kφ + b_k sin kφ)

البقية r (لكل ضربة):  إزاحة الهيكل  Σ_k B_k(u)·D_k  و  تصحيح log السُمك  Σ_k B_k(u)·e_k   (B: متعددات برنشتاين، K=12)
  عليها أولوية:  D ~ N(0, 0.3²) ,  e ~ N(0, 0.4²)   → النموذج يفسّر بالأسلوب العام أولاً

الملاءمة (رسم قابل للاشتقاق):
  المرحلة 1: g فقط (r = 0)     → «كم يفسّر الأسلوب العام»
  المرحلة 2: r مع تثبيت g      → «الوصول إلى الهدف»
النقل: g لكل الحروف؛ r لضربة جديدة بوزن α = exp(−d²/2λ²)، d = المسافة المرنة (SRVF) لأقرب ضربة في حرف الهدف.
"""
import math, json, os, sys
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import srvf
import geometry as GE

torch.set_default_dtype(torch.float64)
K = 12
GN = ["shear", "rotation", "aspect", "log curvature gain", "swash β", "log swash span", "hook η", "log hook span",
      "continuity κ (logit)", "log weight S", "contrast γ", "pen angle θ", "log pen floor m",
      "a1", "b1", "a2", "b2", "log taper ℓ"]
GPRIOR = torch.tensor([[0, .5], [0, .5], [0, .5], [0, .3], [0, 2.], [math.log(.3), .7], [0, 2.], [math.log(.3), .7],
                       [-2, 2.], [math.log(.13), .7], [1, 1.], [math.radians(70), 1.5], [math.log(.3), 1.],
                       [0, .7], [0, .7], [0, .7], [0, .7], [math.log(.08), .7]])
D_SD, E_SD, T_LIK = 0.3, 0.4, 0.01

def bernstein(n, k=None):
    k = k or K
    u = np.linspace(0, 1, n)[:, None]; j = np.arange(k)[None]
    from scipy.special import comb
    return torch.tensor(comb(k - 1, j) * u ** j * (1 - u) ** (k - 1 - j))

def smoothstep(x):
    x = torch.clamp(x, 0, 1); return x * x * (3 - 2 * x)

def affine(g):
    s, r, a = g[0], g[1], g[2]
    R = torch.stack([torch.stack([torch.cos(r), -torch.sin(r)]), torch.stack([torch.sin(r), torch.cos(r)])])
    Sh = torch.stack([torch.stack([torch.ones(()), s]), torch.stack([torch.zeros(()), torch.ones(())])])
    D = torch.diag(torch.stack([torch.exp(a / 2), torch.exp(-a / 2)]))
    return R @ Sh @ D

def interp_point(c, x):
    """نقطة على المنحنى c عند فهرس كسري x (قابل للاشتقاق)."""
    n = c.shape[0]; x = torch.clamp(x, 0, n - 1 - 1e-6)
    i = x.floor().long(); f = x - i
    return c[i] * (1 - f) + c[i + 1] * f

class Model:
    def __init__(self, glyph, wbar=None):
        self.G = glyph
        self.wbar = wbar or glyph.wbar                           # متوسط سُمك مرجعي مشترك (من حرف التدريب)
        self.n = len(glyph.strokes)
        self.B = bernstein(len(glyph.strokes[0][0]))
        self.base = []
        for c, w in glyph.strokes:
            d = np.diff(c, axis=0)
            phi = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
            self.base.append(dict(c=torch.tensor(c), w=torch.tensor(w), phi=torch.tensor(phi),
                                  ds=torch.tensor(np.linalg.norm(d, axis=1)),
                                  sig=float(np.sign(phi[-1] - phi[0]) or 1.0)))
    def zero_residual(self):
        return [dict(D=torch.zeros(K, 2), e=torch.zeros(K)) for _ in range(self.n)]

    def generate(self, g, res):
        A = affine(g); cg = torch.exp(g[3])
        f_sw, f_hk = torch.exp(g[5]).clamp(0.03, 0.9), torch.exp(g[7]).clamp(0.03, 0.9)
        kappa = torch.sigmoid(g[8]); taper_l = torch.exp(g[17]).clamp(0.01, 0.4)
        out = []
        for k, b in enumerate(self.base):
            main = k == self.n - 1
            u = torch.linspace(0, 1, len(b["phi"]))
            phi = b["phi"][0] + cg * (b["phi"] - b["phi"][0])
            if main:
                phi = phi + b["sig"] * g[4] * smoothstep((u - 1 + f_sw) / f_sw)
            elif not self.G.closed[k]:
                phi = phi + b["sig"] * g[6] * smoothstep((u - 1 + f_hk) / f_hk)
            v = torch.stack([torch.cos(phi), torch.sin(phi)], 1) * b["ds"][:, None]
            c = torch.cat([torch.zeros(1, 2), torch.cumsum(v, 0)]) @ A.T
            att = self.G.attach[k]
            if att is None:
                start = A @ b["c"][0]
            else:
                a, i, off = att
                ca = out[a][0]
                x = torch.tensor(float(i)) + kappa * (ca.shape[0] - 1 - i)
                start = interp_point(ca, x) + (1 - kappa) * (A @ torch.tensor(off))
            c = c + start + self.B @ res[k]["D"]
            t = torch.gradient(c, dim=0)[0]
            ang = torch.atan2(t[:, 1], t[:, 0])
            up = torch.linspace(0, 1, c.shape[0])
            x = torch.clamp(torch.minimum(up, 1 - up) / taper_l, 0, 1)
            tau = 0.04 + 0.96 * x * x * (3 - 2 * x)
            if self.G.closed[k]:
                tau = torch.ones_like(tau)
            pen = torch.exp(g[12]) + torch.abs(torch.sin(ang - g[11]))
            press = torch.exp(g[13] * torch.cos(ang) + g[14] * torch.sin(ang) + g[15] * torch.cos(2 * ang) + g[16] * torch.sin(2 * ang))
            w = tau * torch.exp(g[9]) * (b["w"] / self.wbar) ** g[10] * pen * press * torch.exp(self.B @ res[k]["e"])
            out.append((c, w))
        return out

    def dots(self, g, strokes):
        """النقاط تتبع أقرب نقطة هيكل، وحجمها بنسبة وزن الأسلوب."""
        A = affine(g).detach().numpy()
        out = []
        for p, r in self.G.dots:
            best = None
            for (c0, _), (c, w) in zip(self.G.strokes, strokes):
                d = np.linalg.norm(c0 - p, axis=1); i = int(np.argmin(d))
                if best is None or d[i] < best[0]:
                    best = (d[i], c[i].detach().numpy() - A @ c0[i], float(w.detach().median()) / float(np.median(self.G.strokes[0][1])))
            out.append((A @ p + best[1], r * best[2]))
        return out

# ---------------- الرسم القابل للاشتقاق ----------------
class Canvas:
    def __init__(self, mask, size=150, pad=12):
        ys, xs = np.nonzero(mask)
        m = mask[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]
        f = size / max(m.shape)
        T = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((round(m.shape[1] * f), round(m.shape[0] * f)), Image.BILINEAR)) > 127
        self.Tn = T; self.T = torch.tensor(T, dtype=torch.float32)
        self.H, self.W = T.shape
        yy, xx = torch.meshgrid(torch.arange(self.H) + 0.5, torch.arange(self.W) + 0.5, indexing="ij")
        self.pix = torch.stack([xx.flatten(), yy.flatten()], 1).float()

    def to_pix(self, c, al):
        k, tx, ty = al[0], al[1], al[2]
        return torch.stack([c[:, 0] * k * self.H + self.W / 2 + tx * self.H, -c[:, 1] * k * self.H + self.H / 2 + ty * self.H], 1)

    def soft(self, strokes, al, tau=0.6, eps=0.6):
        X = torch.cat([self.to_pix(c, al) for c, _ in strokes]).float()
        r = torch.cat([w for _, w in strokes]).float() * al[0].float() * self.H / 2
        d = torch.cdist(self.pix, X)
        s = tau * torch.logsumexp((r[None] - d) / tau, dim=1)
        return torch.sigmoid(s / eps).reshape(self.H, self.W)

    def hard(self, strokes, al):
        """الشكل نفسه الذي حسّنه النموذج (عتبة 0.5 على التغطية) — لا رسم بديل مختلف عنه."""
        with torch.no_grad():
            return self.soft([(c.detach(), w.detach()) for c, w in strokes], al.detach()).numpy() > 0.5

def implicit_image(strokes, px_per_unit, ctr, W, H, ss=2, tau=0.6, eps=0.6, chunk=40000):
    """رسم عالي الدقة بالتعريف الرياضي نفسه المستخدم في الملاءمة (اتحاد أقراص ناعم)."""
    X = torch.cat([c.detach() for c, _ in strokes]).float()
    r = torch.cat([w.detach() for _, w in strokes]).float() / 2
    k = px_per_unit * ss
    Xp = torch.stack([(X[:, 0] - ctr[0]) * k + W * ss / 2, -(X[:, 1] - ctr[1]) * k + H * ss / 2], 1)
    rp = r * k; t = tau * ss; e = eps * ss
    yy, xx = torch.meshgrid(torch.arange(H * ss) + 0.5, torch.arange(W * ss) + 0.5, indexing="ij")
    P = torch.stack([xx.flatten(), yy.flatten()], 1).float()
    out = torch.empty(len(P))
    for i in range(0, len(P), chunk):
        d = torch.cdist(P[i:i + chunk], Xp)
        out[i:i + chunk] = torch.sigmoid(t * torch.logsumexp((rp[None] - d) / t, 1) / e)
    img = out.reshape(H * ss, W * ss).numpy()
    return np.asarray(Image.fromarray((img * 255).astype(np.uint8)).resize((W, H), Image.LANCZOS)) / 255.0

def edge_err(a, b):
    """متوسط المسافة المتناظرة بين الحواف، نسبة مئوية من ارتفاع اللوحة."""
    ea = a & ~ndimage.binary_erosion(a); eb = b & ~ndimage.binary_erosion(b)
    da = ndimage.distance_transform_edt(~eb)[ea].mean(); db = ndimage.distance_transform_edt(~ea)[eb].mean()
    return float(100 * 0.5 * (da + db) / a.shape[0])

def soft_iou(a, b):
    inter = (a * b).sum(); return inter / (a.sum() + b.sum() - inter)

def hard_iou(a, b):
    return float((a & b).sum() / (a | b).sum())

def g_prior(g):
    return (((g - GPRIOR[:, 0]) / GPRIOR[:, 1]) ** 2).sum() / 2

def r_prior(res):
    return sum(((r["D"] / D_SD) ** 2).sum() + ((r["e"] / E_SD) ** 2).sum() for r in res) / 2

# ---------------- الملاءمة ----------------
def initial_align(model, cv):
    c = torch.cat([s[0] for s in model.generate(GPRIOR[:, 0], model.zero_residual())]).detach()
    span = (c.max(0).values - c.min(0).values)
    k = 0.8 * min(1.0, float(cv.W / cv.H) / float(span[0] / span[1])) / float(span[1])
    ctr = (c.max(0).values + c.min(0).values) / 2
    return torch.tensor([k, -float(ctr[0]) * k, float(ctr[1]) * k])

def stage1(model, cv, starts=10, iters=500, seed=0, log=print):
    torch.manual_seed(seed)
    best = None
    al0 = initial_align(model, cv)
    for s in range(starts):
        g = (GPRIOR[:, 0] + (0 if s == 0 else 0.6) * GPRIOR[:, 1] * torch.randn(len(GN))).clone().requires_grad_(True)
        al = al0.clone().requires_grad_(True)
        res = model.zero_residual()
        opt = torch.optim.Adam([g, al], lr=0.04)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=0.002)
        for it in range(iters):
            opt.zero_grad()
            st = model.generate(g, res)
            iou = soft_iou(cv.soft(st, al), cv.T)
            loss = (1 - iou) / T_LIK + g_prior(g)
            loss.backward(); opt.step(); sch.step()
        h = hard_iou(cv.hard(model.generate(g, res), al), cv.Tn)
        log(f"  stage1 start {s}: hard IoU {h:.3f}")
        if best is None or h > best[0]:
            best = (h, g.detach().clone(), al.detach().clone())
    return best

def stage2(model, cv, g, al, iters=1500, log=print):
    res = [dict(D=torch.zeros(K, 2, requires_grad=True), e=torch.zeros(K, requires_grad=True)) for _ in range(model.n)]
    al = al.clone().requires_grad_(True)
    opt = torch.optim.Adam([al] + [v for r in res for v in r.values()], lr=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=0.0005)
    for it in range(iters):
        opt.zero_grad()
        st = model.generate(g, res)
        iou = soft_iou(cv.soft(st, al), cv.T)
        loss = (1 - iou) / T_LIK + r_prior(res)
        loss.backward(); opt.step(); sch.step()
    res = [{k: v.detach() for k, v in r.items()} for r in res]
    h = hard_iou(cv.hard(model.generate(g, res), al), cv.Tn)
    log(f"  stage2: hard IoU {h:.3f}")
    return h, res, al.detach()

# ---------------- النقل ----------------
LAM = 0.25
def transfer_residual(src_model, src_res, tgt_model):
    """لكل ضربة في الحرف الجديد: أقرب ضربة في حرف الهدف (SRVF بلا تدوير) → α، والبقية عبر التطابق الأمثل."""
    out, report = [], []
    n = len(tgt_model.base[0]["c"])
    for k, b in enumerate(tgt_model.base):
        cb = b["c"].numpy()
        best = None
        for j, sb in enumerate(src_model.base):
            d = srvf.elastic_distance(cb, sb["c"].numpy(), n=60, rotate=False)
            if best is None or d < best[0]:
                best = (d, j)
        d, j = best
        a = math.exp(-d ** 2 / (2 * LAM ** 2))
        A_, B_ = srvf.resample(cb, 80), srvf.resample(src_model.base[j]["c"].numpy(), 80)
        q0, _, _ = srvf.to_q(A_); q1, _, _ = srvf.to_q(B_)
        psi = srvf.dp_reparam(q0, q1)
        Bsrc = src_model.B.numpy()
        t = np.linspace(0, 1, n); t80 = np.linspace(0, 1, 80)
        pk = np.interp(t, t80, psi)
        disp = np.stack([np.interp(pk, t, Bsrc @ src_res[j]["D"].numpy()[:, q]) for q in range(2)], 1)
        logw = np.interp(pk, t, Bsrc @ src_res[j]["e"].numpy())
        # إسقاط على أساس برنشتاين للحرف الجديد
        Bt = tgt_model.B.numpy()
        D = np.linalg.lstsq(Bt, a * disp, rcond=None)[0]; e = np.linalg.lstsq(Bt, a * logw, rcond=None)[0]
        out.append(dict(D=torch.tensor(D), e=torch.tensor(e)))
        report.append(dict(stroke=k, nearest_target_stroke=j, distance=round(d, 3), alpha=round(a, 3)))
    return out, report
