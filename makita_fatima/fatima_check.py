# the same complete measurements on فاطمة and on the logo's Arabic word: do the numbers agree?
import runpy, os, sys, json
g=runpy.run_path(''+__import__("os").path.join(__import__("os").path.dirname(__file__),"fatima_logo.py")+'')
from logo_studio.core.fullmeasure import full_measure
U=g['U']; base=g['base']; p=g['p']
def parts(geom):
    gs=getattr(geom,'geoms',[geom]); return {'w%02d'%i:q for i,q in enumerate(sorted(gs,key=lambda q:-q.bounds[2]))}
def meta(d): return [dict(id=k,bbox=list(v.bounds)) for k,v in d.items()]
rows=[]
for name,d in (('logo ماكيتا',{k:p.polys[k] for k in ('part_01','part_02','part_03','part_06')}),('فاطمة',parts(g['word']))):
    try: r=full_measure(d,meta(d),U,base)
    except Exception as e: r=full_measure(d,[],U,base)
    u=r['unified']; rows.append((name,u))
keys=['stroke_width_all','stroke_width_H','stroke_width_V','angle_V','angle_H','corner_radius_outer','corner_radius_inner','corner_angle']
for k in keys:
    print(k); [print('   ',n,json.dumps(u.get(k),ensure_ascii=False)[:160]) for n,u in rows]
print('levels'); 
