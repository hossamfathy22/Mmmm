"""Read a drawing the way a type designer reads it: which pieces touch (one shape), which small shapes are
dots and signs, which shapes stand on one line, which word of the designer's text each shape is, and where
inside a joined word one letter hands over to the next.

The designer gives the text of each line, as it reads (right to left), e.g. "ا ب حـ د ر" or "فاطمة بيدوناق".
Every other decision is measured on the drawing and reported."""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import arabic as A
from . import geom as G
from .source import Piece, straight_edges


@dataclass
class Blob:
    ids: List[int]
    path: G.Path                         # source units (points), y down
    rect: Tuple[float, float, float, float]
    marks: List["Blob"] = field(default_factory=list)
    fills: frozenset = frozenset()

    @property
    def w(self):
        return self.rect[2] - self.rect[0]

    @property
    def h(self):
        return self.rect[3] - self.rect[1]

    @property
    def area(self):
        return self.w * self.h


@dataclass
class Drawn:
    """One letter as the designer drew it, in font units: y up, its run's baseline at 0."""
    unit: str
    form: str
    body: G.Path
    marks: List[G.Path]
    jl: Optional[float]
    jr: Optional[float]
    where: str                           # "line 2, «فاطمة»" — for the report
    colour: bool                         # drawn in several colours (a construction drawing)
    exact: bool = True                   # False: kept whole because the join inside could not be found


@dataclass
class Reading:
    slant: float                         # degrees, the stems' lean (positive: tops to the right)
    tan: float
    scale: float                         # font units per point
    x_height: float                      # font units
    band: float                          # connector thickness, font units
    letters: List[Drawn]
    dot: Optional[G.Path]                # the designer's dot, centred on (0, 0)
    notes: List[str]
    spacing: Optional[float] = None      # white between unjoined letters along the slant, font units
    kerns: Dict[Tuple[str, str], float] = field(default_factory=dict)   # (right glyph, left glyph): white
    spec: Optional[object] = None        # the logos' measures the letters were brought to (rectify.py)
    levels: Optional[object] = None      # those measures in this drawing's units


# ------------------------------------------------------------------ measuring
def measure_slant(pieces: List[Piece]) -> float:
    hs = sorted(pc.rect[3] - pc.rect[1] for pc in pieces)
    med = hs[len(hs) // 2] if hs else 0
    edges = [(a, ln) for a, ln in straight_edges(pieces, 0.5 * med) if -35 < a < 35]
    if len(edges) < 3:
        return 0.0
    longest = max(ln for _, ln in edges)              # the stems: the longest straight edges
    edges = [(a, ln) for a, ln in edges if ln >= 0.8 * longest] or edges
    edges.sort()
    total = sum(ln for _, ln in edges)
    acc = 0.0
    for a, ln in edges:                  # length-weighted median
        acc += ln
        if acc >= total / 2:
            return a
    return edges[-1][0]


def _desl_src(p: G.Path, tan: float) -> G.Path:
    """Upright a source-space path (y down): x' = x + y*tan."""
    return G.transformed(p, (1, 0, tan, 1, 0, 0))


def blobs_of(pieces: List[Piece]) -> List[Blob]:
    paths = {pc.i: G.from_contours(pc.contours, lambda x, y: (x, y)) for pc in pieces}
    rects = {pc.i: pc.rect for pc in pieces}
    fills = {pc.i: pc.fill for pc in pieces}
    parent = {i: i for i in paths}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ids = sorted(paths)
    for k, a in enumerate(ids):
        ra = rects[a]
        for b in ids[k + 1:]:
            rb = rects[b]
            if ra[0] - 0.5 > rb[2] or rb[0] - 0.5 > ra[2] or ra[1] - 0.5 > rb[3] or rb[1] - 0.5 > ra[3]:
                continue
            if find(a) != find(b) and G.touches(paths[a], paths[b], 0.3):
                parent[find(a)] = find(b)
    groups: Dict[int, List[int]] = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    out = []
    for g in groups.values():
        p = G.union(paths[i] for i in g)
        b = G.bounds(p)
        if b:
            out.append(Blob(sorted(g), p, b, fills=frozenset(fills[i] for i in g)))
    return out


def split_marks(blobs: List[Blob], tan: float) -> List[Blob]:
    """Small shapes over or under a bigger one (dots, the kaf's sign, the gaf's bars) become its marks."""
    ds = {id(b): G.bounds(_desl_src(b.path, tan)) for b in blobs}
    letters, marks = [], []
    order = sorted(blobs, key=lambda b: b.area, reverse=True)
    big = order[: max(1, len(order) // 2)]
    xh = 0.85 * statistics.median(b.h for b in big)           # a first idea of the x-height
    for s in order:
        sd = ds[id(s)]
        best, best_ov = None, 0.0
        small = max(s.w, s.h) <= 1.1 * xh and min(s.w, s.h) <= 0.5 * xh
        for b in order:
            if b is s or b.area < (2.8 if small else 1.2) * s.area:
                continue
            bd = ds[id(b)]
            ov = min(sd[2], bd[2]) - max(sd[0], bd[0])
            gap = max(b.rect[1] - s.rect[3], s.rect[1] - b.rect[3], 0)
            if small:
                ok = ov >= 0.5 * (sd[2] - sd[0]) and gap < 0.45 * xh
            else:                       # a stroke standing inside another letter's width (the alef of لا)
                ok = ov >= 0.7 * (sd[2] - sd[0]) and gap <= 0.02 * xh
            if ok and ov > best_ov:
                best, best_ov = b, ov
        if best is not None:
            marks.append((s, best))
        else:
            letters.append(s)
    for s, b in marks:
        host = b
        while host not in letters:      # a mark of a mark belongs to the letter under both
            host = next(h for m, h in marks if m is host)
        host.marks.append(s)
    return letters


def rows_of(letters: List[Blob]) -> List[List[Blob]]:
    if not letters:
        return []
    med_h = statistics.median(b.h for b in letters)
    byc = sorted(letters, key=lambda b: (b.rect[1] + b.rect[3]) / 2)
    rows, cur, last = [], [], None
    for b in byc:
        c = (b.rect[1] + b.rect[3]) / 2
        if last is not None and c - last > 0.6 * med_h:
            rows.append(cur)
            cur = []
        cur.append(b)
        last = c
    rows.append(cur)
    return rows


def _mode_min(values: List[float], width: float) -> float:
    """The densest cluster of values (within `width`); its smallest member."""
    best, best_n = None, -1
    for v in values:
        n = sum(1 for w in values if abs(w - v) <= width)
        if n > best_n or (n == best_n and v < best):
            best, best_n = v, n
    cluster = [w for w in values if abs(w - best) <= width]
    return min(cluster)


def row_lines(row: List[Blob]) -> Tuple[float, float]:
    """Baseline (y down) and x-height top of a row: the commonest bottom and the commonest top."""
    base = _mode_min([b.rect[3] for b in row], 1.5)
    tops = [b.rect[1] for b in row if base - b.rect[1] < 1.35 * statistics.median(base - c.rect[1] for c in row)]
    top = -_mode_min([-t for t in tops], 1.5) if tops else min(b.rect[1] for b in row)
    return base, top


# ------------------------------------------------------------------ the text
def expected_runs(text: str) -> List[Tuple[str, List[Tuple[str, str]]]]:
    """(token, run) in reading order, i.e. right to left on the page."""
    out = []
    for k, tok in enumerate(text.split()):
        for run in A.runs_of(tok):
            out.append(((k, tok), run))          # the word's place in the line keeps two equal words apart
    return out


def align(blobs: List[Blob], runs, unit_w: float, tan: float):
    """Match the runs (reading order) to the shapes (right to left). A shape the text does not name is
    skipped (reported); every run must find a shape."""
    ds = [G.bounds(_desl_src(b.path, tan)) for b in blobs]
    order = sorted(range(len(blobs)), key=lambda k: -(ds[k][0] + ds[k][2]) / 2)
    bl = [blobs[k] for k in order]
    wd = [ds[k][2] - ds[k][0] for k in order]
    n, m = len(runs), len(bl)
    INF = float("inf")
    cost = [[INF] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            c = cost[i][j]
            if c == INF:
                continue
            if j < m and c + 1.5 < cost[i][j + 1]:                       # skip a shape
                cost[i][j + 1], back[i][j + 1] = c + 1.5, (i, j, None, 0)
            # one shape may hold up to three runs whose letters touch (a connector drawn into the next letter)
            for k in (1, 2, 3):
                if i + k <= n and j < m:
                    letters_n = sum(len(runs[q][1]) for q in range(i, i + k))
                    mc = abs(math.log(max(wd[j], 1e-6) / (letters_n * unit_w))) + 0.8 * (k - 1)
                    if c + mc < cost[i + k][j + 1]:
                        cost[i + k][j + 1], back[i + k][j + 1] = c + mc, (i, j, j, k)
    if cost[n][m] == INF:
        return None, bl
    groups, i, j = [], n, m
    while (i, j) != (0, 0):
        pi, pj, matched, k = back[i][j]
        if matched is not None:
            groups.append(([runs[q] for q in range(pi, pi + k)], bl[matched]))
        i, j = pi, pj
    groups.reverse()
    used = {id(b) for _, b in groups}
    extra = [b for b in bl if id(b) not in used]
    return groups, extra


# ------------------------------------------------------------------ cutting a joined run
def _columns(D: G.Path, x0: float, x1: float, step: float):
    cols = []
    x = x0 + step / 2
    while x < x1:
        r = G.intersect(D, G.rect(x - step / 2, -G.BIG, x + step / 2, G.BIG))
        spans = []
        for c in G.contours(r):
            _, b = G.contour_info(c)
            if b:
                spans.append((b[1], b[3]))
        cols.append((x, spans))
        x += step
    return cols


def connector_zones(D: G.Path, xh: float, band: Optional[float], step: float):
    """Stretches where the only ink is a thin stroke on the baseline: where one letter hands over."""
    b = G.bounds(D)
    cols = _columns(D, b[0], b[2], step)
    zones, cur = [], []
    for x, spans in cols:
        if spans:
            top = max(s[1] for s in spans)
            bot = min(s[0] for s in spans)
            thick = sum(s[1] - s[0] for s in spans)
            if band is None:
                ok = top <= 0.35 * xh and bot >= -0.3 * xh and thick <= 0.25 * xh
            else:                       # a connector lies on the baseline; a tail passing down does not
                ok = top <= 1.9 * band and abs(bot) <= 0.7 * band and thick <= 1.8 * band
        else:
            ok = False
        if ok:
            cur.append((x, thick, bot, top))
        else:
            if len(cur) >= 2:
                zones.append(cur)
            cur = []
    if len(cur) >= 2:
        zones.append(cur)
    return zones


def _work(p_src: G.Path, base: float, s: float) -> G.Path:
    """Source (points, y down) to font units, y up, baseline at 0."""
    return G.transformed(p_src, (s, 0, 0, -s, 0, base * s))


def read(pieces: List[Piece], texts: List[str], x_height_units: float = 450.0) -> Reading:
    notes: List[str] = []
    slant = measure_slant(pieces)
    tan = math.tan(math.radians(slant))
    blobs = blobs_of(pieces)
    letters = split_marks(blobs, tan)
    rows = rows_of(letters)
    if len(rows) != len(texts):
        notes.append(f"the drawing has {len(rows)} lines of letters but {len(texts)} lines of text were given; "
                     "lines are matched top to bottom")
    lines = [row_lines(r) for r in rows]
    xh_pt = statistics.median(base - top for base, top in lines)
    s = x_height_units / xh_pt
    xh = x_height_units
    singles = [b.w for r, t in zip(rows, texts) for b in r if all(len(tok) <= 2 for tok in t.split())]
    out: List[Drawn] = []
    bands: List[float] = []
    plans = []
    for li, (row, text) in enumerate(zip(rows, texts)):
        runs = expected_runs(text)
        n_letters = sum(len(r) for _, r in runs)
        unit_w = statistics.median(singles) if singles else sum(b.w for b in row) / max(1, n_letters)
        matched, extra = align(row, runs, unit_w, tan)
        if matched is None:
            notes.append(f"line {li + 1}: {len(runs)} joined pieces expected from the text, only {len(row)} shapes found")
            continue
        for b in extra:
            notes.append(f"line {li + 1}: a shape the text does not name was left out (at x {b.rect[0]:.0f} pt)")
        doubts = _dot_doubts(matched)
        for d in doubts:
            notes.append(f"line {li + 1}: {d}")
        if doubts:
            notes.append(f"line {li + 1}: the text may not match the drawing (a word too many or too few shifts "
                         "every letter after it): look at the reading sheet before using the font")
        plans.append((li, row, text, matched))
    # first pass over the joined runs: the connector's thickness
    for li, row, text, matched in plans:
        base, _ = lines[li]
        for group, blob in matched:
            if sum(len(r) for _, r in group) < 2:
                continue
            D = G.deslant(_work(blob.path, base, s), tan)
            for z in connector_zones(D, xh, None, 0.5 * s):
                bands.append(min(t for _, t, _, _ in z))
    band = statistics.median(bands) if bands else 0.08 * xh
    for li, row, text, matched in plans:
        base, _ = lines[li]
        for group, blob in matched:
            out += _cut(group, blob, base, s, tan, xh, band, li, notes)
    dot = _dot_shape(letters, s)
    spacing, kerns = _spacing(plans, lines, s, tan)
    return Reading(slant, tan, s, xh, band, out, dot, notes, spacing, kerns)


def _dark(c) -> bool:
    if not c:
        return True
    r, g, b = (int(c[i:i + 2], 16) for i in (1, 3, 5))
    return max(r, g, b) < 90


def _cut(group, blob: Blob, base_pt, s, tan, xh, band, li, notes) -> List[Drawn]:
    # a construction drawing shows its parts in colours; the letter as meant is drawn in one dark ink
    colour = len(blob.fills) > 1 or not all(_dark(c) for c in blob.fills)
    P = _work(blob.path, base_pt, s)
    marks = [_work(m.path, base_pt, s) for m in blob.marks]
    toks = " ".join(dict.fromkeys(t[1] for t, _ in group))
    where = f"line {li + 1}, «{toks}»"
    run = [lf for _, r in group for lf in r]
    # where one run ends and the next begins inside this shape (letters drawn touching): not a join
    ends = set()
    k = 0
    for _, r in group[:-1]:
        k += len(r)
        ends.add(k - 1)
    if len(run) == 1:
        u, f = run[0]
        d = Drawn(u, f, P, marks, None, None, where, colour)
        if f in ("init", "medi"):
            _left_join_of_drawn(d, tan, xh, band)
        return [d]
    D = G.deslant(P, tan)
    zones = connector_zones(D, xh, band, 0.4 * s)
    zones.sort(key=lambda z: -z[0][0])                              # right to left
    # the run's own baseline: the connectors' underside
    bots = [min(c[2] for c in z) for z in zones]
    lift = statistics.median(bots) if bots else 0.0
    if abs(lift) > 0.2 * band:
        P = G.shift(P, 0, -lift)
        marks = [G.shift(m, 0, -lift) for m in marks]
        zones = [[(x, t, b - lift, tp - lift) for x, t, b, tp in z] for z in zones]
    cuts, zi = [], 0                     # (x where the right piece ends, x where the left piece begins, joined)
    for idx, (u, f) in enumerate(run[:-1]):
        zi += A.INTERNAL_JOINED.get(A.body(u) if len(u) == 1 else "", 0) if f in ("init", "medi") else 0
        if zi >= len(zones):
            break
        z = zones[zi]
        if idx in ends:                  # a connector drawn into the next letter: all of it is this letter's
            left_end = min(c[0] for c in z)
            cuts.append((left_end, left_end, False))
        else:
            x = min(z, key=lambda c: c[1])[0]                       # the thinnest column of the stretch
            cuts.append((x, x, True))
        zi += 1
    out, pieces_out = [], []
    for k in range(len(cuts) + 1):
        xr = cuts[k - 1][1] if k > 0 else None
        xl = cuts[k][0] if k < len(cuts) else None
        jr = xr if (k > 0 and cuts[k - 1][2]) else None
        jl = xl if (k < len(cuts) and cuts[k][2]) else None
        if k < len(cuts) and not cuts[k][2] and run[k][1] in ("init", "medi"):
            jl = xl                      # the letter itself joins left (حـ): its connector ends at the cut
        letters_here = run[k:] if k == len(cuts) else [run[k]]
        body = G.clip(P, tan, xl, xr)
        pieces_out.append((letters_here, body, jl, jr, xl, xr))
    _give_marks(pieces_out, marks, tan, xh)
    for letters_here, body, jl, jr, xl, xr, ms in pieces_out:
        if len(letters_here) == 1:
            u, f = letters_here[0]
            out.append(Drawn(u, f, body, ms, jl, jr, where, colour))
        else:
            notes.append(f"{where}: the join inside «{''.join(u for u, _ in letters_here)}» is not on the baseline "
                         "(drawn as one shape, like a logo's ligature); those letters were not taken from it")
    return out


def _expected_dots(u: str, f: str):
    if len(u) != 1 or u not in A.LETTERS:
        return 0, True
    n, above = A.LETTERS[u][2], A.LETTERS[u][3]
    if u in A.DOTS_JOINED_ONLY and f in ("init", "medi"):
        n, above = A.DOTS_JOINED_ONLY[u]
    return n, above


def _is_round(m: Blob) -> bool:
    return m.h > 0 and 0.75 <= m.w / m.h <= 1.33


def _dot_doubts(matched) -> List[str]:
    """Where the text and the shape it was matched to disagree about dots. Only the plain cases count:
    a letter that needs dots on a shape with no mark at all, or a round dot on letters that carry no mark
    of any kind (a kaf's sign, a hamza or a construction piece is not round, or belongs to its letter)."""
    out = []
    for group, blob in matched:
        letters = [(u, f) for _, run in group for u, f in run]
        word = "".join(u for u, _ in letters)
        need = sum(_expected_dots(u, f)[0] for u, f in letters)
        bare = all(len(u) == 1 and u in A.LETTERS and _expected_dots(u, f)[0] == 0 and u not in A.MARKED
                   for u, f in letters)
        if need and not blob.marks:
            out.append(f"«{word}» needs {need} dot(s) but its shape has none")
        elif bare and any(_is_round(m) for m in blob.marks):
            out.append(f"«{word}» has no dots but a dot lies on its shape")
    return out


def _give_marks(pieces_out, marks, tan, xh):
    """The text says which letters carry dots, how many, above or below: each letter takes the nearest
    dots on its side; any other mark (a bar, a sign) goes to the letter it stands over."""
    info = []
    for m in marks:
        b = G.bounds(m)
        db = G.bounds(G.deslant(m, tan))
        info.append({"m": m, "cx": (db[0] + db[2]) / 2, "above": (b[1] + b[3]) / 2 > 0.45 * xh,
                     "round": 0.7 < (b[2] - b[0]) / max(b[3] - b[1], 1e-6) < 1.45, "taken": False})
    centres = []
    for letters_here, body, *_ in pieces_out:
        db = G.bounds(G.deslant(body, tan))
        centres.append(((db[0] + db[2]) / 2) if db else 0.0)
    given = [[] for _ in pieces_out]
    for k, (letters_here, *_r) in enumerate(pieces_out):
        if len(letters_here) != 1:
            continue
        n, above = _expected_dots(*letters_here[0])
        cands = sorted((i for i in info if not i["taken"] and i["round"] and i["above"] == above),
                       key=lambda i: abs(i["cx"] - centres[k]))
        for i in cands[:n]:
            i["taken"] = True
            given[k].append(i["m"])
    for i in info:
        if i["taken"]:
            continue
        k = min(range(len(pieces_out)), key=lambda q: abs(i["cx"] - centres[q]))
        given[k].append(i["m"])
    for k in range(len(pieces_out)):
        pieces_out[k] = pieces_out[k] + (given[k],)


def _left_join_of_drawn(d: Drawn, tan, xh, band):
    """A letter drawn alone with its connector to the left (حـ): the join is where that connector has
    become as thin as a connector."""
    D = G.deslant(d.body, tan)
    zones = connector_zones(D, xh, band, 0.4 * (xh / 74.3))
    b = G.bounds(D)
    left = [z for z in zones if z[0][0] - b[0] < 0.1 * xh]
    if left:
        z = left[0]
        d.jl = z[-1][0] if len(z) < 3 else z[len(z) // 2][0]
        d.body = G.clip(d.body, tan, d.jl, None)
        return
    # no clean stretch: the join is the tip of the ink lying on the baseline at the left
    run = G.run_at(D, band * 0.5)
    if run:
        d.jl = run[0] + 0.5 * (xh / 74.3)
        d.body = G.clip(d.body, tan, d.jl, None)


def _dot_shape(letters: List[Blob], s: float) -> Optional[G.Path]:
    cands = [m for b in letters for m in b.marks if 0.75 < m.w / max(m.h, 1e-6) < 1.33]
    if not cands:
        return None
    med = statistics.median(m.w for m in cands)
    best = min(cands, key=lambda m: abs(m.w - med) + abs(m.h - med))
    p = G.transformed(best.path, (s, 0, 0, -s, 0, 0))
    b = G.bounds(p)
    return G.shift(p, -(b[0] + b[2]) / 2, -(b[1] + b[3]) / 2)


def glyph_name(u: str, f: str) -> str:
    return A.unit_name(u) + ("" if f == "isol" else "." + f)


def _spacing(plans, lines, s, tan):
    """The white between two unjoined pieces of one word, along the slant: its median is the designer's
    spacing, and each pair of letters keeps its own (kerning)."""
    gaps, pairs = [], {}
    for li, row, text, matched in plans:
        base, _ = lines[li]
        by_word: Dict[tuple, list] = {}
        for group, blob in matched:
            keys = {t for t, _ in group}
            if len(keys) == 1:
                by_word.setdefault(next(iter(keys)), []).append((blob, group))
        for key, items in by_word.items():
            if len(items) < 2:
                continue
            ds = []
            for blob, group in items:
                P = _work(blob.path, base, s)
                above = G.intersect(P, G.rect(-G.BIG, 0, G.BIG, G.BIG))
                ds.append((G.bounds(G.deslant(above, tan)), group))
            ds.sort(key=lambda d: -d[0][2])
            for (a, ga), (b, gb) in zip(ds, ds[1:]):
                g = a[0] - b[2]
                if g <= 0:
                    continue
                gaps.append(g)
                right = ga[-1][1][-1]                  # the last letter of the right piece
                left = gb[0][1][0]                     # the first letter of the left piece
                pairs.setdefault((glyph_name(*right), glyph_name(*left)), []).append(g)
    spacing = statistics.median(gaps) if gaps else None
    return spacing, {k: statistics.median(v) for k, v in pairs.items()}
