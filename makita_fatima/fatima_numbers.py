# فاطمة drafted from the logo's pieces, then drawn again with the logo's corrected numbers (outline families):
# edge angles, stroke widths and corner radii all snapped to the logo's; what has no family in the logo is listed.
import os, sys, runpy, json, numpy as np, cv2
HERE=os.path.dirname(os.path.abspath(__file__))
g=runpy.run_path(os.path.join(HERE,'fatima_logo.py'))
from shapely.ops import unary_union
from shapely import affinity
from logo_studio.core import outline as OL
from logo_studio.core.build import render_png, polygon_svg
U=g['U']; LOGO_M=g['LOGO_M']; word=g['word']; p=g['p']
parts={'w%d'%i:q for i,q in enumerate(getattr(word,'geoms',[word]))}
shapes,m,odd=OL.apply(parts,LOGO_M,U)
new=unary_union(list(shapes.values()))
iou=new.intersection(word).area/new.union(word).area
print('draft -> logo numbers: IoU %.4f (how much the draft moved)'%iou)
print('logo families: angles',[f['mean'] for f in LOGO_M['families']['edge_angles'] if f['n']>=2],'widths',[f['mean'] for f in LOGO_M['families']['widths'] if f['n']>=2])
print('draft own:     angles',[(f['mean'],f['n']) for f in m['families']['edge_angles']],'widths',[(f['mean'],f['n']) for f in m['families']['widths']])
print('no family in the logo:',json.dumps({k:[{kk:(round(vv,3) if isinstance(vv,float) else ([round(x/U,2) for x in vv] if kk=='at' else vv)) for kk,vv in d.items()} for d in v] for k,v in odd.items()},ensure_ascii=False))
# check: measure the result and compare its families with the logo's
m2=OL.measure({'w':new},U)
print('result:        angles',[(f['mean'],f['n']) for f in m2['families']['edge_angles']],'widths',[(f['mean'],f['n']) for f in m2['families']['widths']])
logo=unary_union([p.polys[k] for k in ('part_01','part_02','part_03','part_06')])
y0=min(new.bounds[1],logo.bounds[1])-8; y1=max(new.bounds[3],logo.bounds[3])+8
def tile(gm,lab,over=None):
    gx=affinity.translate(gm,-gm.bounds[0]+8,0); t=render_png([(gx,(25,25,25))],(0,y0,gx.bounds[2]+8,y1),scale=4)
    cv2.putText(t,lab,(10,28),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,200),2); return t
diff=render_png([(word.intersection(new),(205,205,205)),(word.difference(new),(40,40,220)),(new.difference(word),(220,120,30))],
                (new.bounds[0]-8,y0,new.bounds[2]+8,y1),scale=4)
cv2.putText(diff,'draft vs logo numbers: red = taken away, blue = added   IoU %.3f'%iou,(10,28),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,200),2)
T=[tile(new,'FATIMA with the logo numbers'),diff,tile(logo,'the logo')]
W=max(t.shape[1] for t in T); im=np.vstack([np.pad(t,((0,16),(0,W-t.shape[1]),(0,0)),constant_values=255) for t in T])
OUT=os.path.dirname(HERE)
cv2.imwrite(os.path.join(OUT,'fatima_makita.png'),im)
open(os.path.join(OUT,'fatima_makita.svg'),'w').write(polygon_svg(affinity.translate(new,-new.bounds[0]+5,-new.bounds[1]+5),(new.bounds[2]-new.bounds[0]+10,new.bounds[3]-new.bounds[1]+10),title='fatima'))
