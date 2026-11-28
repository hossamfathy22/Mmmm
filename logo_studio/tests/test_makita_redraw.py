"""Makita through the app's API: the complete measurements must draw the logo back (the outline check)."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault('LOGO_STUDIO_PROJECTS', tempfile.mkdtemp())
from fastapi.testclient import TestClient
from logo_studio.server import app

IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'makita.jpg')

def run(verbose=True):
    c = TestClient(app); say = print if verbose else (lambda *a: None)
    pid = c.post('/api/projects', files={'file': ('makita.jpg', open(IMG, 'rb'), 'image/jpeg')}).json()['id']
    v = c.post(f'/api/projects/{pid}/vectorize', json={}).json()
    assert v['report']['match_vs_picture'] > 0.98
    c.post(f'/api/projects/{pid}/settings', json={'U': 59.9, 'baseline': 92.37})
    arabic = [p['name'] for p in v['parts'] if p['bbox'][1] < 100]           # the Arabic row (the Latin MAKITA is below)
    fm = c.post(f'/api/projects/{pid}/full', json={'parts': arabic}).json()
    u = fm['unified']; rd = {l['level']: l for l in fm['redraw']}
    say('edge angles', [(f['mean'], f['n']) for f in u['edge_angles']][:4])
    say('widths', [(f['mean'], f['n']) for f in u['edge_widths']])
    for l in fm['redraw']:
        say(l['level'], l['name'], l['iou'], l['max_off'])
    ang = sorted(f['mean'] for f in u['edge_angles'][:2])
    assert abs(ang[0]) < 1 and abs(ang[1]-69.8) < 1                          # horizontals and the 20-degree lean
    w = sorted(f['mean'] for f in u['edge_widths'] if f['n'] >= 2)
    assert len(w) == 2 and abs(w[0]-0.267) < 0.01 and abs(w[1]-0.304) < 0.01    # dots / meem top, and the strokes
    assert rd[2]['iou'] > 0.99 and rd[2]['max_off'] < 0.03                     # each element's own numbers: one pixel
    assert rd[5]['iou'] > 0.975 and rd[5]['max_off'] < 0.08                    # all unified
    assert c.get(f'/api/projects/{pid}/redraw_check.png').status_code == 200
    return pid

def test_makita_redraw():
    run(verbose=False)

if __name__ == '__main__':
    run()
