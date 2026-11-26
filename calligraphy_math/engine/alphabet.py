"""
استخراج الحروف الـ 28 (الشكل المنفصل) من جدول سكّال آلياً، ثم استخراج ضرباتها آلياً.

1) عمود «الشكل المنفصل» يُرسم بدقة 600 dpi، وتُكتشف خطوط الجدول الأفقية لتقسيم الصفوف.
2) الحبر = بكسلات داكنة غير ملونة (تُستبعد الأسهم البنفسجية والخلفيات الصفراء والبرتقالية).
3) خطوط الجدول تُحذف بفتح مورفولوجي رأسي، وكل مكوّن يُنسب إلى الصف الذي يقع فيه مركزه.
4) جسم الحرف = أكبر مكوّن، والنقاط = المكوّنات الصغيرة الأخرى.
"""
import os, subprocess, tempfile
import numpy as np
from PIL import Image
from scipy import ndimage

NAMES = ["alef", "ba", "ta", "tha", "jeem", "ha", "kha", "dal", "thal", "ra", "zai", "seen", "sheen", "sad",
         "dad", "toa", "zhoa", "ain", "ghain", "fa", "qaf", "kaf", "lam", "meem", "nun", "he", "waw", "ya"]
ARABIC = dict(zip(NAMES, "ا ب ت ث ج ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ك ل م ن ه و ي".split()))
DPI = 600
COL = (3490, 930, 900, 8760)                  # x, y, W, H عند 600 dpi

def render_column(pdf):
    x, y, W, H = COL
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["pdftoppm", "-r", str(DPI), "-png", "-x", str(x), "-y", str(y), "-W", str(W), "-H", str(H),
                        pdf, f"{d}/c"], check=True)
        return np.asarray(Image.open(f"{d}/c-1.png").convert("RGB")).astype(int)

def row_bounds(a):
    dark = (a.max(2) < 90) & ((a.max(2) - a.min(2)) < 40)
    ys = np.where(dark.mean(1) > 0.5)[0]
    groups = []
    for y in ys:
        if not groups or y - groups[-1][-1] > 3:
            groups.append([y])
        else:
            groups[-1].append(y)
    lines = [int(np.mean(g)) for g in groups]
    return list(zip(lines[:-1], lines[1:]))

def extract_alphabet(pdf):
    a = render_column(pdf)
    rows = row_bounds(a)
    assert len(rows) == 28, f"expected 28 rows, got {len(rows)}"
    ink = (a.max(2) < 100) & ((a.max(2) - a.min(2)) < 40)
    ink = ndimage.binary_opening(ink, structure=np.ones((7, 1)))          # حذف خطوط الجدول
    lab, k = ndimage.label(ink)
    sizes = ndimage.sum(ink, lab, range(1, k + 1))
    cents = ndimage.center_of_mass(ink, lab, range(1, k + 1))
    out = {}
    for name, (y0, y1) in zip(NAMES, rows):
        comps = [i + 1 for i in range(k) if y0 <= cents[i][0] < y1 and sizes[i] > 150]
        # استبعاد بقايا الخطوط (مكوّن أعرض من نصف العمود وأرفع من 12 بكسل)
        sl = ndimage.find_objects(lab)
        comps = [i for i in comps if not ((sl[i - 1][1].stop - sl[i - 1][1].start) > 0.5 * a.shape[1]
                                          and (sl[i - 1][0].stop - sl[i - 1][0].start) < 12)]
        if not comps:
            continue
        body = max(comps, key=lambda i: sizes[i - 1])
        dots = [i for i in comps if i != body and sizes[i - 1] > 0.015 * sizes[body - 1]]
        ys, xs = np.nonzero(np.isin(lab, [body] + dots))
        pad = 12
        Y0, Y1, X0, X1 = max(ys.min() - pad, 0), ys.max() + pad, max(xs.min() - pad, 0), xs.max() + pad
        sub = lab[Y0:Y1, X0:X1]
        dot_list = []
        for i in dots:
            yy, xx = np.nonzero(sub == i)
            dot_list.append((np.array([xx.mean(), yy.mean()]), 0.5 * max(np.ptp(xx), np.ptp(yy)) + 0.5))
        out[name] = dict(body=sub == body, dots=dot_list, ink=np.isin(sub, [body] + dots))
    return out

def sheet(alpha, path, key="ink"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 7, figsize=(14, 9), facecolor="white")
    for ax, n in zip(axes.flat[::-1], NAMES):
        ax.imshow(~alpha[n][key], cmap="gray"); ax.set_title(n, fontsize=9); ax.axis("off")
    fig.savefig(path, dpi=90, bbox_inches="tight")

if __name__ == "__main__":
    import sys
    A = extract_alphabet(sys.argv[1])
    print(len(A), {n: (A[n]["body"].shape, len(A[n]["dots"])) for n in A})
    sheet(A, sys.argv[2] if len(sys.argv) > 2 else "alphabet_sheet.png")
