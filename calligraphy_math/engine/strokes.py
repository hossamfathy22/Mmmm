"""
استخراج الضربات آلياً من صورة حرف:
  1) هيكل وسطي (skeletonize) بعد تنعيم خفيف.
  2) رسم بياني: أطراف (درجة 1)، تقاطعات (درجة ≥ 3)، حواف بينها.
  3) تقليم الزوائد القصيرة (أقصر من السُمك الوسيط).
  4) تجميع الحواف في ضربات: عند كل تقاطع نصل الحافتين الأكثر استمراراً (أقل انعطافاً)، ثم نسلسل.
  5) السُمك بالمقطع العرضي + إصلاح مناطق الالتقاء (استيفاء من الجانبين).
"""
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

def skeleton_graph(mask):
    m = ndimage.gaussian_filter(mask.astype(float), 1.5) > 0.5
    sk = skeletonize(m)
    pts = set(zip(*np.nonzero(sk)))
    nbrs = {p: [(p[0] + a, p[1] + b) for a, b in NB if (p[0] + a, p[1] + b) in pts] for p in pts}
    special = {p for p, n in nbrs.items() if len(n) != 2}
    # تجميع بكسلات التقاطع المتجاورة في عقدة واحدة
    node_of, nodes = {}, []
    for p in special:
        if p in node_of:
            continue
        stack, comp = [p], []
        node_of[p] = len(nodes)
        while stack:
            q = stack.pop(); comp.append(q)
            for r in nbrs[q]:
                if r in special and r not in node_of:
                    node_of[r] = len(nodes); stack.append(r)
        nodes.append(comp)
    edges, seen = [], set()
    for p in special:
        for q in nbrs[p]:
            if q in special or (p, q) in seen:
                continue
            path = [p, q]; prev, cur = p, q
            while cur not in special:
                nxt = [r for r in nbrs[cur] if r != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]; path.append(cur)
            seen.add((path[-2], path[-1])); seen.add((p, q))
            if cur in special:
                seen.add((cur, path[-2]))
            edges.append(dict(a=node_of[p], b=node_of.get(cur, -1), pix=np.array(path)[:, ::-1].astype(float)))
    # حلقات بلا عقد (مثل الهاء المغلقة تماماً)
    covered = set(map(tuple, np.vstack([e["pix"][:, ::-1] for e in edges]).astype(int))) if edges else set()
    rest = pts - covered - special
    if rest and not edges:
        start = next(iter(rest)); path = [start]; prev = None; cur = start
        while True:
            nxt = [r for r in nbrs[cur] if r != prev and r not in path[-3:]]
            if not nxt or (len(path) > 3 and nxt[0] == start):
                break
            prev, cur = cur, nxt[0]; path.append(cur)
        nodes.append([start])
        edges.append(dict(a=len(nodes) - 1, b=len(nodes) - 1, pix=np.array(path + [start])[:, ::-1].astype(float)))
    # إزالة الحواف المكررة (نفس البكسلات بالعكس)
    uniq, keys = [], set()
    for e in edges:
        key = frozenset(map(tuple, e["pix"].astype(int)))
        if key not in keys:
            keys.add(key); uniq.append(e)
    deg = np.zeros(len(nodes), int)
    for e in uniq:
        deg[e["a"]] += 1; deg[e["b"]] += 1
    return uniq, nodes, deg

def prune(edges, deg, min_len):
    changed = True
    while changed:
        changed = False
        for e in list(edges):
            L = np.linalg.norm(np.diff(e["pix"], axis=0), axis=1).sum()
            leaf = deg[e["a"]] == 1 or deg[e["b"]] == 1
            junction = deg[e["a"]] >= 3 or deg[e["b"]] >= 3
            if leaf and junction and L < min_len:
                edges.remove(e); deg[e["a"]] -= 1; deg[e["b"]] -= 1; changed = True
    return edges, deg

def direction_out(e, node, k=12):
    P = e["pix"] if e["a"] == node else e["pix"][::-1]
    k = min(k, len(P) - 1)
    d = P[k] - P[0]
    return d / (np.linalg.norm(d) + 1e-9)

def assemble(edges, nodes, deg):
    """عند كل تقاطع: نصل أزواج الحواف الأقرب إلى الاستقامة (زاوية بين اتجاهيهما الخارجيين أقرب إلى 180°)."""
    link = {}
    for node in range(len(nodes)):
        inc = [(i, e) for i, e in enumerate(edges) if e["a"] == node or e["b"] == node]
        if len(inc) < 2:
            continue
        cand = []
        for x in range(len(inc)):
            for y in range(x + 1, len(inc)):
                i, ei = inc[x]; j, ej = inc[y]
                c = float(direction_out(ei, node) @ direction_out(ej, node))
                cand.append((c, i, j))
        used = set()
        for c, i, j in sorted(cand):
            if c > -0.35 or i in used or j in used:              # انعطاف أكبر من ~110° = لا وصل
                continue
            used |= {i, j}
            link[(i, node)] = j; link[(j, node)] = i
    strokes, done = [], set()
    def other(e, node):
        return e["b"] if e["a"] == node else e["a"]
    for i0, e0 in enumerate(edges):
        if i0 in done:
            continue
        # ارجع إلى بداية السلسلة
        i, node = i0, e0["a"]
        steps = 0
        while (i, node) in link and steps < 50:
            j = link[(i, node)]
            if j == i0:
                break
            node = other(edges[j], node); i = j; steps += 1
        # امشِ للأمام
        chain, cur_node = [], node
        while True:
            e = edges[i]
            P = e["pix"] if e["a"] == cur_node else e["pix"][::-1]
            chain.append(P if not chain else P[1:]); done.add(i)
            nxt_node = other(e, cur_node)
            if (i, nxt_node) in link and link[(i, nxt_node)] not in done:
                i = link[(i, nxt_node)]; cur_node = nxt_node
            else:
                break
        strokes.append(np.vstack(chain))
    return strokes

def resample(c, n):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
    u = np.linspace(0, s[-1], n)
    return np.column_stack([np.interp(u, s, c[:, 0]), np.interp(u, s, c[:, 1])])

def extract_strokes(mask, n=160):
    dt = ndimage.distance_transform_edt(mask)
    edges, nodes, deg = skeleton_graph(mask)
    wmed = 2 * np.median(dt[skeletonize(mask)])
    edges, deg = prune(edges, deg, 1.3 * wmed)
    strokes = assemble(edges, nodes, deg)
    out = []
    for c in strokes:
        L = np.linalg.norm(np.diff(c, axis=0), axis=1).sum()
        if L < 1.5 * wmed:
            continue
        c = ndimage.gaussian_filter1d(c, 3, axis=0, mode="nearest")
        out.append(resample(c, n))
    return out, wmed
