"""A project on disk: the picture, the curves, the designer's marks and choices, every result. One folder per logo."""
import os, json, uuid, time, numpy as np, cv2
from shapely.geometry import Point
from .vectorize import vectorize, parts_svg
from .geom import polys_of, centrelines
from .measure import measure_mark, spans, unify, gate, CATS
from .reference import shape_word, to_logo, place
from .build import build, polygon_svg, render_png
from .fullmeasure import full_measure, sheet as full_sheet
from . import outline as OL

ROOT = os.environ.get('LOGO_STUDIO_PROJECTS', os.path.join(os.path.dirname(os.path.dirname(__file__)), 'projects'))

def _np(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, dict):
        return {k: _np(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_np(v) for v in o]
    return o

def geom_path(g):
    """shapely geometry -> SVG path data (straight segments; for the preview)"""
    out = []
    for q in ([g] if g.geom_type == 'Polygon' else list(getattr(g, 'geoms', []))):
        if q.is_empty or q.geom_type != 'Polygon':
            continue
        for r in [q.exterior]+list(q.interiors):
            c = np.array(r.coords)
            out.append('M'+' L'.join('%.2f %.2f' % (x, y) for x, y in c)+'Z')
    return ' '.join(out)

class Project:
    def __init__(self, pid):
        self.id = pid; self.dir = os.path.join(ROOT, pid)
        self.state = json.load(open(self.f('state.json'))) if os.path.exists(self.f('state.json')) else {}
        self._polys = None; self._cls = {}

    def f(self, name):
        return os.path.join(self.dir, name)

    def save(self):
        json.dump(_np(self.state), open(self.f('state.json'), 'w'))

    @staticmethod
    def create(image_bytes, filename):
        pid = time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]
        d = os.path.join(ROOT, pid); os.makedirs(d, exist_ok=True)
        ext = os.path.splitext(filename or 'logo.png')[1].lower() or '.png'
        open(os.path.join(d, 'source'+ext), 'wb').write(image_bytes)
        p = Project(pid); p.state = dict(id=pid, source='source'+ext, created=time.time(), step=1)
        img = p.image()
        if img is None:
            raise ValueError('this file is not a picture')
        p.state['picture'] = [int(img.shape[1]), int(img.shape[0])]; p.save()
        return p

    # ---------- step 1: curves
    def image(self):
        return cv2.imdecode(np.frombuffer(open(self.f(self.state['source']), 'rb').read(), np.uint8), cv2.IMREAD_UNCHANGED)

    def vectorize(self, box=None):
        img = self.image()
        r = vectorize(img, box=box)
        self.state.update(box=box, crop=r['crop'], size=r['size'], parts=r['parts'], report=r['report'], picture=[int(img.shape[1]), int(img.shape[0])])
        for k in ('marks', 'measure', 'full', 'reference', 'design', 'result'):
            self.state.pop(k, None)
        self.state.setdefault('U', None); self.state['U'] = self.suggest_unit(); self.state['baseline'] = self.suggest_baseline()
        open(self.f('logo.svg'), 'w').write(parts_svg(r['parts'], r['size']))
        self.save(); self._polys = None; self._cls = {}
        return r

    def suggest_unit(self):
        """a first guess for the unit: the height most parts share (e.g. the Latin x-height); the designer can change it"""
        hs = np.array([p['bbox'][3] for p in self.state['parts']])
        if not len(hs):
            return 1.0
        best = None
        for h in hs:
            n = int(np.sum(np.abs(hs-h) <= 0.06*h))
            if best is None or n > best[0] or (n == best[0] and h < best[1]):
                best = (n, float(h))
        return float(np.median(hs[np.abs(hs-best[1]) <= 0.06*best[1]]))

    def suggest_baseline(self):
        """first guess: the bottom line most big parts of the first row sit on"""
        ps = [p for p in self.state['parts'] if p.get('row', 0) == 0]
        if not ps:
            return 0.0
        hmax = max(p['bbox'][3] for p in ps); big = [p for p in ps if p['bbox'][3] > 0.5*hmax]
        return float(np.median([p['bbox'][1]+p['bbox'][3] for p in big]))*0.98

    @property
    def polys(self):
        if self._polys is None:
            self._polys = polys_of(self.state['parts'])
        return self._polys

    def cls(self, part):
        if part not in self._cls:
            self._cls[part] = centrelines(self.polys[part], self.state['U'], recentred=False)[0]
        return self._cls[part]

    # ---------- steps 3-5: marks, measures, unified numbers, test
    def measure(self, marks, overrides=None, hairline_one=True):
        U = self.state['U']; polys = self.polys
        used = {}
        for m in marks:
            c = (np.asarray(m['p0'], float)+np.asarray(m['p1'], float))/2
            used.setdefault(min(polys, key=lambda k: polys[k].distance(Point(c))), True)
        cls = {p: self.cls(p) for p in used}
        res = {m['id']: measure_mark(m, {k: polys[k] for k in used}, cls, U) for m in marks}
        sp = spans(marks, res, polys, cls, U)
        table, spec = unify(marks, res, sp, U, hairline_one=hairline_one, overrides=overrides)
        g = gate(sp, spec, polys, res, marks, U)
        self.state['marks'] = marks
        self.state['measure'] = dict(results=res, spans=sp, table=table, spec=spec, gate=g, overrides=overrides or {}, hairline_one=hairline_one)
        self.save()
        return self.state['measure']

    # ---------- step 5: complete measurements (automatic, the whole logo or the chosen parts)
    def full(self, parts=None):
        names = parts or [p['name'] for p in self.state['parts']]
        polys = {k: self.polys[k] for k in names}
        r = full_measure(polys, [q for q in self.state['parts'] if q['name'] in names], self.state['U'], self.state['baseline'],
                         cls={k: self.cls(k) for k in names})
        cv2.imwrite(self.f('full_measures.png'), full_sheet(polys, r, self.state['U'], self.state['baseline']))
        # the check: draw the parts back from these numbers, level by level, and compare
        r['redraw'] = OL.check(polys, r['outline'], self.state['U'])
        cv2.imwrite(self.f('redraw_check.png'), OL.check_sheet(polys, r['outline'], self.state['U']))
        r['parts'] = names
        self.state['full'] = r; self.save()
        return r

    # ---------- step 6: reference
    def set_font(self, data, filename):
        ext = os.path.splitext(filename or 'font.ttf')[1].lower() or '.ttf'
        open(self.f('reference'+ext), 'wb').write(data); self.state['font'] = 'reference'+ext; self.save()

    def shape(self, text):
        fb = open(self.f(self.state['font']), 'rb').read()
        gl, upm = shape_word(fb, text)
        return fb, gl

    def reference(self, text, match_index, match_height_u, target_index, right_x):
        U = self.state['U']; base = self.state['baseline']
        fb, gl = self.shape(text)
        s, dx, dy = place(gl, match_index, match_height_u*U, target_index, right_x, base)
        glyph = to_logo(gl[target_index]['poly'], s, dx, dy)
        word = [to_logo(g['poly'], s, dx, dy) for g in gl if not g['poly'].is_empty]
        edges, nodes = centrelines(glyph, U, recentred=False, prune=0.08)
        L = lambda P: float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))/U)
        ref = dict(text=text, match_index=match_index, match_height_u=match_height_u, target_index=target_index, right_x=right_x, scale=s,
                   glyphs=[dict(index=g['index'], name=g['name'], char=g['char']) for g in gl],
                   glyph_path=geom_path(glyph), word_path=' '.join(geom_path(w) for w in word),
                   edges=[dict(id=i, P=e['P'].tolist(), ends=e['ends'], length=round(L(e['P']), 2)) for i, e in enumerate(edges)],
                   junctions=[[n['x'], n['y']] for n in nodes if n['kind'] == 'junction'])
        self.state['reference'] = ref; self.save()
        return ref

    # ---------- step 7: build
    def build(self, design):
        U = self.state['U']; ms = self.state['measure']; ref = self.state['reference']
        project = dict(polys=self.polys, glyph_edges=[dict(P=np.asarray(e['P'], float)) for e in ref['edges']], **design)
        full, made, checks = build(project, ms['spec'], ms['spans'], U)
        b = full.bounds
        self.state['design'] = design
        self.state['result'] = dict(path=geom_path(full), checks=checks, bounds=list(b),
                                    strokes=[dict(kind=m['kind'], P=np.asarray(m['P'])[::4].tolist(), W=np.asarray(m['W'])[::4].tolist()) for m in made])
        open(self.f('result.svg'), 'w').write(polygon_svg(full, self.state['size'], title='logo_with_new_letter'))
        im = render_png([(full, (25, 25, 25))], (0, 0, self.state['size'][0], self.state['size'][1]), scale=8)
        cv2.imwrite(self.f('result.png'), im)
        self.save()
        return self.state['result']

def list_projects():
    if not os.path.isdir(ROOT):
        return []
    out = []
    for d in sorted(os.listdir(ROOT), reverse=True):
        f = os.path.join(ROOT, d, 'state.json')
        if os.path.exists(f):
            s = json.load(open(f)); out.append(dict(id=d, step=s.get('step'), parts=len(s.get('parts', [])), created=s.get('created')))
    return out
