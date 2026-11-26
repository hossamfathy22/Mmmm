"""
تمثيل الحرف للمحرّك: ضربات مرتّبة ومُوجَّهة (هيكل + سُمك) + نقاط.

الاتجاه (قاعدة خطية): تبدأ الضربة من طرفها الأعلى-الأيمن  (الدرجة = 0.3·x − y  في إحداثيات الصورة)
  استثناء معروف من جدول سكّال: رأس ج ح خ يُكتب من اليسار إلى اليمين.
الضربة الرئيسة = التي تصل إلى أدنى نقطة (فيها الذيل أو النزول)، وتأتي أخيراً؛ الضربات الثانوية قبلها.
"""
import numpy as np
from scipy import ndimage
import geometry as G
import strokes as ST

HEAD_LEFT_TO_RIGHT = {"jeem", "ha", "kha"}

class Glyph:
    def __init__(self, name, body, dots=(), n=160, unit=None):
        self.name = name
        raw_c, _ = ST.extract_strokes(body, n)
        raw = []
        for c in raw_c:
            c2, w = G.refine(body, c)
            raw.append((c2, w))
        zones = G.junction_zones(raw)
        raw = [(c, G.fill_zones(w, z)) for (c, w), z in zip(raw, zones)]
        # توجيه كل ضربة
        oriented = []
        for c, w in raw:
            closed = np.linalg.norm(c[0] - c[-1]) < 0.5 * np.median(w) + 3
            if not closed:
                s0, s1 = 0.3 * c[0, 0] - c[0, 1], 0.3 * c[-1, 0] - c[-1, 1]
                if s1 > s0:
                    c, w = c[::-1], w[::-1]
            oriented.append([c, w, closed])
        main = int(np.argmax([c[:, 1].max() for c, _, _ in oriented]))   # الضربة التي تصل لأدنى نقطة (الذيل/النزول)
        if name in HEAD_LEFT_TO_RIGHT:
            for k, (c, w, cl) in enumerate(oriented):
                if k != main and c[0, 0] > c[-1, 0]:
                    oriented[k] = [c[::-1], w[::-1], cl]
        order = [k for k in range(len(oriented)) if k != main] + [main]
        oriented = [oriented[k] for k in order]
        ys, xs = np.nonzero(body)
        self.H = float(unit) if unit else float(ys.max() - ys.min())     # وحدة قياس مشتركة لكل حروف الجدول
        self.ctr = np.array([(xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2])
        self.strokes = [(self.norm(c), w / self.H) for c, w, _ in oriented]
        self.closed = [cl for _, _, cl in oriented]
        self.dots = [(self.norm(p[None])[0], r / self.H) for p, r in dots]
        self.wbar = float(np.mean(np.concatenate([w for _, w in self.strokes])))
        self.body = body
        # نقاط الالتقاء: بداية كل ضربة تقع على ضربة أخرى؟
        self.attach = []
        for b, (cb, wb) in enumerate(self.strokes):
            best = None
            for a, (ca, wa) in enumerate(self.strokes):
                if a == b:
                    continue
                d = np.linalg.norm(ca - cb[0], axis=1); i = int(np.argmin(d))
                if d[i] < 0.6 * (np.median(wa) + np.median(wb)) and (best is None or d[i] < best[1]):
                    best = (a, d[i], i)
            self.attach.append(None if best is None or best[0] > b else (best[0], best[2], cb[0] - self.strokes[best[0]][0][best[2]]))

    def norm(self, p):
        q = (p - self.ctr) / self.H
        q[:, 1] *= -1
        return q
