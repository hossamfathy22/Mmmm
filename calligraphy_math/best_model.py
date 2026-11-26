"""
النموذج الأفضل (حرف واحد للتدريب):

1) الهيكل:  Sigma-Lognormal لكل ضربة (kinematics.py)، والأسلوب يعمل على معاملات حركة اليد:
     θs' = θs + δθs ,  θe' = θs' + c·(θe − θs) + δθe  (+ σ·β لآخر دفعة في آخر ضربة: التفاف الذيل)
     D'  = D·e^{δlnD} ,  v' = A(s,ρ,a)·v
2) القلم: ضغط يتبع اتجاه الحركة φ (قابل للنقل بين الحروف):
     w = τ · S · (w_n/w̄)^γ · (m + |sin(φ − θp)|) · exp(Σ_k a_k cos kφ + b_k sin kφ) · exp(δw)
3) الرسم القابل للاشتقاق:  coverage(x) = sigmoid( τ·LSE_i[(r_i − |x − p_i|)/τ] / ε )
4) بايز هرمي:
     p(g, δ_jeem | صورة) ∝ exp(−(1 − softIoU)/T) · N(δ_jeem; 0, Σ_δ) · p(g) · exp(−‖قيود الالتقاء‖²/2ς²)
   g = 14 معاملاً عاماً للأسلوب، δ = انحرافات خاصة بالحرف.
   MAP بـ Adam من بدايات متعددة، ثم عيّنات بـ Langevin المُكيَّف (pSGLD).
5) التنبؤ بالصاد:  g ~ اللاحقة،  δ_sad = argmin [prior + قيود الالتقاء]  (لا صاد حديثة مستخدمة).
"""
import sys, os, json, math, time
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage

torch.set_default_dtype(torch.float64)
torch.manual_seed(0); np.random.seed(0)
sys_argv = list(sys.argv)
UP = sys_argv[1] if len(sys_argv) > 1 else "."
OUT = sys_argv[2] if len(sys_argv) > 2 else "results_best"
sys.argv = [sys.argv[0], UP]
import style_model as SM
import kinematics as K

T_LIK = 0.01                     # حرارة الاحتمال: نقص 0.01 في IoU = أقل احتمالاً بعامل e
JUNC_SD = 0.006                  # تسامح قيود الالتقاء (نسبة من الارتفاع)

# ---------------- المعاملات العامة للأسلوب ----------------
GNAMES = ["shear s", "rotation ρ", "aspect a", "log curvature gain", "swash β", "log weight S",
          "contrast γ", "pen angle θp", "log pen floor m", "a1", "b1", "a2", "b2", "log taper length ℓ"]
# أولوية ضعيفة (متوسط, انحراف)
GPRIOR = torch.tensor([[0, .5], [0, .5], [0, .5], [0, .3], [0, 2.0], [math.log(.13), .7],
                       [1, 1.0], [math.radians(70), 1.5], [math.log(.25), 1.0],
                       [0, .7], [0, .7], [0, .7], [0, .7], [math.log(.08), .7]])
DSD = dict(ths=0.25, the=0.25, lD=0.2, w=0.25)        # انحرافات الحرف: τ الهرمية
NW = 6                                               # عُقد تصحيح السُمك لكل ضربة

class Letter:
    """حرف نسخ ممثَّل بالحركة + السُمك المقاس + قيود الالتقاء."""
    def __init__(self, name):
        self.name = name
        _, strokes = SM.extract(name)
        self.fits = K.fit_letter(strokes)
        self.wbar = float(np.mean(np.concatenate([f["w"] for f in self.fits])))
        for f in self.fits:
            f["model"].requires_grad_(False)
        self.junctions = self._junctions()

    def base(self, j):
        m = self.fits[j]["model"]
        return {k: v.detach() for k, v in m.params().items()}, m.p0.detach()

    def _junctions(self, thr=0.06):
        """أطراف الضربات القريبة من ضربة أخرى (أو من طرفها الآخر) → يجب أن تبقى ملتصقة."""
        J = []
        cs = [f["c"] for f in self.fits]
        for a, ca in enumerate(cs):
            for end in (0, len(ca) - 1):
                best = None
                for b, cb in enumerate(cs):
                    d = np.linalg.norm(cb - ca[end], axis=1)
                    if a == b:
                        d[max(0, end - 40):end + 40] = np.inf     # ليس جوار النقطة نفسها
                    i = int(np.argmin(d))
                    if d[i] < thr and (best is None or d[i] < best[0]):
                        best = (d[i], b, i)
                if best:
                    J.append((a, end, best[1], best[2], ca[end] - cs[best[1]][best[2]]))
        return J

def new_delta(L):
    return [dict(ths=torch.zeros(f["n"], requires_grad=True), the=torch.zeros(f["n"], requires_grad=True),
                 lD=torch.zeros(f["n"], requires_grad=True), w=torch.zeros(NW, requires_grad=True))
            for f in L.fits]

def affine(g):
    s, r, a = g[0], g[1], g[2]
    R = torch.stack([torch.stack([torch.cos(r), -torch.sin(r)]), torch.stack([torch.sin(r), torch.cos(r)])])
    Sh = torch.stack([torch.stack([torch.ones(()), s]), torch.stack([torch.zeros(()), torch.ones(())])])
    D = torch.diag(torch.stack([torch.exp(a / 2), torch.exp(-a / 2)]))
    return R @ Sh @ D

def generate(L, g, delta):
    """→ قائمة (نقاط الهيكل، السُمك) لكل ضربة، في إحداثيات مطبّعة (y للأعلى)."""
    A = affine(g)
    c_gain = torch.exp(g[3])
    out = []
    for j, f in enumerate(L.fits):
        P, p0 = L.base(j)
        d = delta[j]
        sweep = P["the"] - P["ths"]
        ths = P["ths"] + d["ths"]
        the = ths + c_gain * sweep + d["the"]
        if j == len(L.fits) - 1:
            sig = torch.sign(sweep.sum()) if float(sweep.sum()) != 0 else torch.ones(())
            the = the + torch.cat([torch.zeros(f["n"] - 1), (sig * g[4]).reshape(1)])
        Q = dict(t0=P["t0"], mu=P["mu"], sigma=P["sigma"], D=P["D"] * torch.exp(d["lD"]),
                 ths=ths, the=the, A=A, p0=A @ p0)
        pts, _ = f["model"].trajectory(torch.tensor(f["t"]), Q)
        tg = torch.gradient(pts, dim=0)[0]
        phi = torch.atan2(tg[:, 1], tg[:, 0])
        u = torch.linspace(0, 1, len(pts))
        x = torch.clamp(torch.minimum(u, 1 - u) / torch.exp(g[13]), 0, 1)
        tau = x * x * (3 - 2 * x) * 0.97 + 0.03                    # طول الاستدقاق ℓ معامل عام
        wn = torch.tensor(f["w"]) / L.wbar
        pen = torch.exp(g[8]) + torch.abs(torch.sin(phi - g[7]))
        press = torch.exp(g[9] * torch.cos(phi) + g[10] * torch.sin(phi) + g[11] * torch.cos(2 * phi) + g[12] * torch.sin(2 * phi))
        # استيفاء خطي قابل للاشتقاق للعقد
        pos = u * (NW - 1); i0 = torch.clamp(pos.floor().long(), 0, NW - 2); fr = pos - i0
        dwi = d["w"][i0] * (1 - fr) + d["w"][i0 + 1] * fr
        w = tau * torch.exp(g[5]) * wn ** g[6] * pen * press * torch.exp(dwi)
        out.append((pts, w))
    return out

def junction_penalty(L, strokes, g):
    A = affine(g)
    pen = torch.zeros(())
    for a, end, b, i, off in L.junctions:
        e = strokes[a][0][end] - strokes[b][0][i] - A @ torch.tensor(off)
        pen = pen + (e ** 2).sum()
    return pen / (2 * JUNC_SD ** 2)

def delta_prior(delta):
    return sum(((d[k] / DSD[k]) ** 2).sum() for d in delta for k in DSD) / 2

def g_prior(g):
    return (((g - GPRIOR[:, 0]) / GPRIOR[:, 1]) ** 2).sum() / 2

# ---------------- الرسم ----------------
class Canvas:
    def __init__(self, mask, size=150):
        self.T = torch.tensor(SM.fit_canvas(mask, size=size), dtype=torch.float32)
        H, W = self.T.shape
        yy, xx = torch.meshgrid(torch.arange(H) + 0.5, torch.arange(W) + 0.5, indexing="ij")
        self.pix = torch.stack([xx.flatten(), yy.flatten()], 1).float()
        self.H, self.W = H, W

    def to_pix(self, strokes, al):
        k, tx, ty = al[0], al[1], al[2]
        P = torch.cat([s[0] for s in strokes]); r = torch.cat([s[1] for s in strokes]) / 2
        X = torch.stack([P[:, 0] * k * self.H + self.W / 2 + tx * self.H,
                         -P[:, 1] * k * self.H + self.H / 2 + ty * self.H], 1)
        return X.float(), (r * k * self.H).float()

    def soft(self, strokes, al, tau=0.6, eps=0.6):
        X, r = self.to_pix(strokes, al)
        d = torch.cdist(self.pix, X)
        s = tau * torch.logsumexp((r[None] - d) / tau, dim=1)
        return torch.sigmoid(s / eps).reshape(self.H, self.W)

def soft_iou(a, b):
    inter = (a * b).sum()
    return inter / (a.sum() + b.sum() - inter)

def hard_render(strokes, al, H, W, scale=1):
    img = Image.new("L", (W * scale, H * scale), 0); dr = ImageDraw.Draw(img)
    for pts, w in strokes:
        P = pts.detach().numpy(); w = w.detach().numpy()
        k, tx, ty = [float(x) for x in al]
        X = np.column_stack([P[:, 0] * k * H + W / 2 + tx * H, -P[:, 1] * k * H + H / 2 + ty * H]) * scale
        r = w * k * H * scale / 2
        n = SM.normals(X)
        Lft, Rgt = X + r[:, None] * n, X - r[:, None] * n
        for i in range(len(X) - 1):
            dr.polygon([tuple(Lft[i]), tuple(Rgt[i]), tuple(Rgt[i + 1]), tuple(Lft[i + 1])], fill=255)
        for i in range(0, len(X), 2):
            dr.ellipse([X[i, 0] - r[i], X[i, 1] - r[i], X[i, 0] + r[i], X[i, 1] + r[i]], fill=255)
    return np.asarray(img) > 127

# ---------------- اللاحقة ----------------
def neg_log_post(L, cv, g, delta, al):
    st = generate(L, g, delta)
    pred = cv.soft(st, al)
    iou = soft_iou(pred, cv.T)
    nlp = (1 - iou) / T_LIK + delta_prior(delta) + g_prior(g) + junction_penalty(L, st, g)
    return nlp, float(iou.detach())

def init_from(vals):
    """تحويل حلول النموذج السابق (style_fits.json) إلى بداية في هذا الفضاء."""
    p = vals
    return torch.tensor([p["shear s"], math.radians(p["rotation ρ°"]), p["aspect a"], math.log(p["curvature gain c"]),
                         math.radians(p["swash β°"]), math.log(p["weight S"] * 2.2), p["contrast γ"],
                         math.radians(p["pen angle θ°"]), math.log(max(p["pen floor m"], 0.02)), 0, 0, 0, 0, math.log(.08)])

def fit_map(L, cv, g0, al0, iters=700, lr=0.03, label=""):
    g = g0.clone().requires_grad_(True)
    al = torch.tensor(al0, dtype=torch.float64, requires_grad=True)
    delta = new_delta(L)
    params = [g, al] + [v for d in delta for v in d.values()]
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr * 0.05)
    for it in range(iters):
        opt.zero_grad()
        nlp, iou = neg_log_post(L, cv, g, delta, al)
        nlp.backward(); opt.step(); sched.step()
    nlp, iou = neg_log_post(L, cv, g, delta, al)
    print(f"  MAP {label}: −log p = {float(nlp.detach()):.2f}, softIoU = {iou:.3f}", flush=True)
    return dict(g=g.detach().clone(), al=al.detach().clone(),
                delta=[{k: v.detach().clone() for k, v in d.items()} for d in delta], nlp=float(nlp.detach()), iou=iou)

def langevin(L, cv, m, steps=900, burn=300, thin=15, eps=2e-4):
    """pSGLD: θ ← θ − (ε/2)·M∇U + √(εM)·ξ ,  M = 1/(√(RMS(∇U))+λ)."""
    g = m["g"].clone().requires_grad_(True); al = m["al"].clone().requires_grad_(True)
    delta = [{k: v.clone().requires_grad_(True) for k, v in d.items()} for d in m["delta"]]
    params = [g, al] + [v for d in delta for v in d.values()]
    V = [torch.zeros_like(p) for p in params]
    samples = []
    for it in range(steps):
        for p in params:
            p.grad = None
        nlp, iou = neg_log_post(L, cv, g, delta, al)
        nlp.backward()
        with torch.no_grad():
            for p, v in zip(params, V):
                v.mul_(0.99).add_(0.01 * p.grad ** 2)
                M = 1 / (torch.sqrt(v) + 1e-3)
                p.add_(-0.5 * eps * M * p.grad + torch.sqrt(eps * M) * torch.randn_like(p))
        if it >= burn and (it - burn) % thin == 0:
            samples.append(dict(g=g.detach().clone(), al=al.detach().clone(), iou=iou, nlp=float(nlp.detach())))
    return samples

def predict(Lt, g, iters=250):
    """δ للحرف الجديد: أصغر انحراف (أولوية) يحقق قيود الالتقاء — بلا أي صورة هدف."""
    delta = new_delta(Lt)
    opt = torch.optim.Adam([v for d in delta for v in d.values()], lr=0.03)
    for _ in range(iters):
        opt.zero_grad()
        st = generate(Lt, g, delta)
        loss = delta_prior(delta) + junction_penalty(Lt, st, g)
        loss.backward(); opt.step()
    return generate(Lt, g, delta), delta

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t_start = time.time()
    J = Letter("jeem_naskh"); S = Letter("sad_naskh")
    print("lognormal pulses:", {L.name: [(f["n"], round(f["snr"], 1)) for f in L.fits] for L in (J, S)})
    print("junctions:", {L.name: [(a, e, b, i) for a, e, b, i, _ in L.junctions] for L in (J, S)}, flush=True)
    tgt = SM.ink_mask(SM.IMG["jeem_modern"])
    cv = Canvas(tgt, size=150)

    prev = json.load(open("results/style_fits.json"))["fits"]
    starts = [(init_from(f["params"]), f["align"], f"prev-seed{f['seed']}") for f in prev]
    for k in range(3):
        g0 = GPRIOR[:, 0] + GPRIOR[:, 1] * torch.randn(len(GNAMES)) * 0.5
        starts.append((g0, [1.0, 0.0, 0.0], f"random{k}"))
    maps = [fit_map(J, cv, g0, al0, label=lab) | {"label": lab} for g0, al0, lab in starts]
    maps.sort(key=lambda m: m["nlp"])

    # تجميع الحلول في أنماط (حسب الدوران والميل)
    modes = []
    for m in maps:
        key = np.array([float(m["g"][0]), float(m["g"][1])])
        for md in modes:
            if np.linalg.norm(key - md["key"]) < 0.25:
                break
        else:
            modes.append(dict(key=key, map=m))
    print(f"{len(modes)} distinct modes", flush=True)
    for md in modes[:3]:
        md["samples"] = langevin(J, cv, md["map"])
        print(f"  mode nlp={md['map']['nlp']:.2f}: {len(md['samples'])} samples, "
              f"IoU mean {np.mean([s['iou'] for s in md['samples']]):.3f}", flush=True)
    modes = modes[:3]
    # وزن الأنماط: تقريب exp(−Δ(−log p)) عند MAP
    nl = np.array([md["map"]["nlp"] for md in modes]); wts = np.exp(-(nl - nl.min())); wts /= wts.sum()

    torch.save(dict(J_fits=[{k: f[k] for k in ("t", "c", "w", "n", "snr")} | {"state": f["model"].state_dict()} for f in J.fits],
                    S_fits=[{k: f[k] for k in ("t", "c", "w", "n", "snr")} | {"state": f["model"].state_dict()} for f in S.fits],
                    modes=[dict(map=md["map"], samples=md["samples"]) for md in modes], weights=wts),
               f"{OUT}/posterior.pt")
    print("weights:", np.round(wts, 3), f"time {time.time() - t_start:.0f}s")
