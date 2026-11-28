"""The whole Women Tech session through the app's API: it must give back the numbers and the letter we reached by hand."""
import os, sys, json, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault('LOGO_STUDIO_PROJECTS', tempfile.mkdtemp())
from fastapi.testclient import TestClient
from shapely.geometry import LineString, box
from logo_studio.server import app
from logo_studio.core.project import Project
from logo_studio.tests import womentech_fixture as F

def run(verbose=True):
    c = TestClient(app); say = print if verbose else (lambda *a: None)
    st = c.post('/api/projects', files={'file': ('womentech.jpg', open(F.IMG, 'rb'), 'image/jpeg')}).json(); pid = st['id']
    v = c.post(f'/api/projects/{pid}/vectorize', json={'box': list(F.BOX)}).json()
    say('1 curves', v['report'])
    assert v['report']['match_vs_picture'] > 0.96 and v['report']['edge_error_mean_px'] < 0.1
    crop = v['crop']; dy = 196-crop[1]
    U = float(np.median([p['bbox'][3] for p in v['parts'] if p['row'] == 1 and 20 < p['bbox'][3] < 26]))  # Latin x-height, as we chose
    base = 59.25+dy
    c.post(f'/api/projects/{pid}/settings', json={'U': U, 'baseline': base})
    marks = F.marks(crop)
    m = c.post(f'/api/projects/{pid}/measure', json={'marks': marks}).json(); sp = m['spec']
    say('4 unified', {k: sp[k] for k in ('H_MID', 'H_HAIR', 'H_STEM', 'HEAD_TOP', 'V_TOP', 'V_MID', 'THICK_POS')}, 'lowest point rule', sp['lowest_point_rule']['holds'], '/', sp['lowest_point_rule']['of'])
    assert abs(sp['H_MID']-0.455) < 0.02 and abs(sp['H_HAIR']-0.06) < 0.015 and abs(sp['H_STEM']-0.19) < 0.03 and abs(sp['HEAD_TOP']-0.27) < 0.03
    assert sp['lowest_point_rule']['holds'] >= 4
    say('5 test', m['gate']['mean_iou']); assert m['gate']['mean_iou'] > 0.75
    c.post(f'/api/projects/{pid}/font', files={'file': ('Mirza.ttf', open(os.path.join(os.path.dirname(F.IMG), 'Mirza.ttf'), 'rb'), 'font/ttf')})
    gl = c.post(f'/api/projects/{pid}/shape', json={'text': 'تحــك'}).json()
    ti = [g['index'] for g in gl if g['name'] == 'uniFEA4'][0]; mi = [g['index'] for g in gl if g['name'] == 'uni066E.init'][0]
    p = Project(pid); word = p.polys['part_09']
    te_top = (base-word.intersection(box(106+(223-crop[0])-4, 0, 124+(223-crop[0]), 200)).bounds[1])/U
    ref = c.post(f'/api/projects/{pid}/reference', json={'text': 'تحــك', 'match_index': mi, 'match_height_u': te_top, 'target_index': ti, 'right_x': 110.5+(223-crop[0])}).json()
    E = ref['edges']; L = lambda e: e['length']
    say('6 reference: scale %.4f, %d centre-lines' % (ref['scale'], len(E)))
    spur = [e['id'] for e in E if e['ends'].count('end') == 1 and min(e['P'][0][1], e['P'][-1][1]) < base-0.8*U][0]
    head = [e['id'] for e in E if 1.0 < L(e) < 1.3][0]; exit_ = [e['id'] for e in E if L(e) > 1.2 and min(e['P'][0][0], e['P'][-1][0]) < 64][0]
    links = [e['id'] for e in E if L(e) < 0.2]
    res = m['results']; KAF = np.array(res['24']['at']); NECK = np.array(res['18']['at'])
    def centre_at(x):
        g = LineString([(x, 0), (x, 200)]).intersection(word); gs = [g] if g.geom_type == 'LineString' else [q for q in g.geoms if q.geom_type == 'LineString']
        g = max(gs, key=lambda q: q.coords[0][1]); ys = [q[1] for q in g.coords]; return [x, (min(ys)+max(ys))/2]
    ex = np.array([e for e in E if e['id'] == exit_][0]['P']); ex_end = ex[np.argmin(ex[:, 0])]
    MEET = np.mean(np.array(ref['junctions']), axis=0); lowx = MEET[0]+0.55*(NECK[0]-MEET[0])
    design = dict(strokes=[dict(edges=[spur]+links+[head], type='H', start='hair', end='hair'),
                           dict(edges=[exit_], type='H', start='hair', end='stem', extend_after=[centre_at(x) for x in np.arange(ex_end[0]-1, KAF[0], -1.0)]+[KAF.tolist()])],
                  connections=[dict(points=[NECK.tolist(), centre_at(NECK[0]-0.25*(NECK[0]-lowx)), centre_at(lowx), [MEET[0]+0.06*U, MEET[1]+0.10*U], MEET.tolist()], start='hair', end='hair')],
                  erase=[dict(part='part_09', rect=[KAF[0]+0.3, 0, NECK[0]-0.2, 200])], hide=['part_07', 'part_08'], copies=[dict(part='part_05', above_stroke=0, gap=0.2)])
    b = c.post(f'/api/projects/{pid}/build', json=design).json()
    for ch in b['checks']:
        say('7 built', ch)
    head_c = b['checks'][0]
    assert abs(head_c['thickest']-sp['H_MID']) < 0.02 and head_c['start'] < 0.1 and head_c['end'] < 0.1
    for f in ('result.svg', 'result.png', 'report.json', 'logo.svg'):
        r = c.get(f'/api/projects/{pid}/{f}'); assert r.status_code == 200 and len(r.content) > 500, f
    say('8 export ok ->', Project(pid).dir)
    return pid

def test_women_tech():
    run(verbose=False)

if __name__ == '__main__':
    run()
