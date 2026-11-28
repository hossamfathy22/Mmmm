"""Logo Studio web server: one endpoint per step; the page in static/ drives them."""
import os, json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from .core.project import Project, list_projects, _np, ROOT
from .core.measure import CATS

app = FastAPI(title='Logo Studio')
HERE = os.path.dirname(__file__)

def P(pid):
    p = Project(pid)
    if not p.state:
        raise HTTPException(404, 'no such project')
    return p

def J(o):
    return JSONResponse(_np(o))

@app.get('/api/categories')
def categories():
    return J(CATS)

@app.get('/api/projects')
def projects():
    return J(list_projects())

@app.post('/api/projects')
async def create(file: UploadFile = File(...)):
    data = await file.read()
    try:
        p = Project.create(data, file.filename)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return J(p.state)

@app.get('/api/projects/{pid}')
def get(pid: str):
    return J(P(pid).state)

@app.get('/api/projects/{pid}/picture')
def picture(pid: str):
    p = P(pid); return FileResponse(p.f(p.state['source']))

@app.post('/api/projects/{pid}/vectorize')
async def vec(pid: str, body: dict):
    p = P(pid)
    try:
        r = p.vectorize(box=body.get('box'))
    except ValueError as e:
        raise HTTPException(400, str(e))
    p.state['step'] = max(p.state.get('step', 1), 2); p.save()
    return J(dict(report=r['report'], parts=[dict(name=q['name'], bbox=q['bbox'], row=q['row']) for q in r['parts']], crop=r['crop'], size=r['size'],
                  U=p.state['U'], baseline=p.state['baseline']))

@app.get('/api/projects/{pid}/logo.svg')
def logo_svg(pid: str):
    return FileResponse(P(pid).f('logo.svg'), media_type='image/svg+xml')

@app.get('/api/projects/{pid}/parts')
def parts(pid: str):
    from .core.bezier import svg_d
    p = P(pid)
    return J([dict(name=q['name'], bbox=q['bbox'], row=q['row'], d=svg_d(q['outlines'])) for q in p.state.get('parts', [])])

@app.post('/api/projects/{pid}/settings')
def settings(pid: str, body: dict):
    p = P(pid)
    for k in ('U', 'baseline'):
        if body.get(k) is not None:
            p.state[k] = float(body[k])
    if 'rename' in body:
        for q in p.state['parts']:
            if q['name'] in body['rename']:
                q['name'] = body['rename'][q['name']]
    p.state['step'] = max(p.state.get('step', 1), 3); p.save()
    return J(dict(U=p.state['U'], baseline=p.state['baseline']))

@app.post('/api/projects/{pid}/measure')
def measure(pid: str, body: dict):
    p = P(pid)
    marks = body.get('marks', [])
    if not marks:
        raise HTTPException(400, 'no marks yet')
    r = p.measure(marks, overrides=body.get('overrides'), hairline_one=body.get('hairline_one', True))
    p.state['step'] = max(p.state.get('step', 1), 5); p.save()
    return J(r)

@app.post('/api/projects/{pid}/full')
def full(pid: str, body: dict):
    p = P(pid); r = p.full(body.get('parts') or None)
    p.state['step'] = max(p.state.get('step', 1), 6); p.save()
    return J({k: v for k, v in r.items()})

@app.get('/api/projects/{pid}/redraw_check.png')
def redraw_png(pid: str):
    return FileResponse(P(pid).f('redraw_check.png'), media_type='image/png', filename='redraw_check.png')

@app.get('/api/projects/{pid}/full_measures.png')
def full_png(pid: str):
    return FileResponse(P(pid).f('full_measures.png'), media_type='image/png', filename='full_measures.png')

@app.post('/api/projects/{pid}/font')
async def font(pid: str, file: UploadFile = File(...)):
    p = P(pid); p.set_font(await file.read(), file.filename); return J(dict(ok=True))

@app.post('/api/projects/{pid}/shape')
def shape(pid: str, body: dict):
    p = P(pid)
    if not p.state.get('font'):
        raise HTTPException(400, 'upload the reference font first')
    fb, gl = p.shape(body['text'])
    return J([dict(index=g['index'], name=g['name'], char=g['char'], height=float(g['poly'].bounds[3]) if not g['poly'].is_empty else 0.0) for g in gl])

@app.post('/api/projects/{pid}/reference')
def reference(pid: str, body: dict):
    p = P(pid)
    r = p.reference(body['text'], int(body['match_index']), float(body['match_height_u']), int(body['target_index']), float(body['right_x']))
    p.state['step'] = max(p.state.get('step', 1), 8); p.save()
    return J(r)

@app.post('/api/projects/{pid}/build')
def build(pid: str, body: dict):
    p = P(pid)
    if not p.state.get('measure'):
        raise HTTPException(400, 'measure the logo first')
    if not p.state.get('reference'):
        raise HTTPException(400, 'place the reference letter first')
    r = p.build(body)
    p.state['step'] = max(p.state.get('step', 1), 9); p.save()
    return J(r)

@app.get('/api/projects/{pid}/result.svg')
def result_svg(pid: str):
    return FileResponse(P(pid).f('result.svg'), media_type='image/svg+xml', filename='logo_new_letter.svg')

@app.get('/api/projects/{pid}/result.png')
def result_png(pid: str):
    return FileResponse(P(pid).f('result.png'), media_type='image/png', filename='logo_new_letter.png')

@app.get('/api/projects/{pid}/report.json')
def report(pid: str):
    p = P(pid); s = p.state
    out = dict(curves=s.get('report'), unit_px=s.get('U'), baseline=s.get('baseline'), marks=s.get('marks'),
               unified=(s.get('measure') or {}).get('spec'), table=(s.get('measure') or {}).get('table'), test=(s.get('measure') or {}).get('gate'),
               reference={k: v for k, v in (s.get('reference') or {}).items() if k not in ('glyph_path', 'word_path', 'edges')},
               design=s.get('design'), checks=(s.get('result') or {}).get('checks'),
               complete_measurements=(s.get('full') or {}).get('unified'), levels=(s.get('full') or {}).get('levels'),
               redraw_check=(s.get('full') or {}).get('redraw'))
    return Response(json.dumps(_np(out), ensure_ascii=False, indent=1), media_type='application/json',
                    headers={'Content-Disposition': 'attachment; filename="logo_report.json"'})

app.mount('/', StaticFiles(directory=os.path.join(HERE, 'static'), html=True), name='static')
