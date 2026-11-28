# فاطمة from the logo's own pieces (as the HACOUPIAN hah came from the logo's kaf and waw)
import sys, os; sys.path.insert(0,'/home/user/Mmmm'); os.environ['LOGO_STUDIO_PROJECTS']='/home/user/Mmmm/logo_studio/projects'
import numpy as np, cv2
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from shapely import affinity
from logo_studio.core.project import Project
from logo_studio.core.build import render_png, polygon_svg
p=Project('20261128-035950-a4f7bf'); U=p.state['U']; base=p.state['baseline']
T=np.tan(np.radians(19.8))
def right_of(x_at_base, xpad=4000):
    """everything right of a line leaning like the logo's verticals, crossing the baseline at x_at_base"""
    top=base-3*U; return Polygon([(x_at_base,base+20),(x_at_base+(base+20-top)*0+ (base-top)*T+ 0, top),(x_at_base+xpad+(base-top)*T,top),(x_at_base+xpad,base+20)])
ma=p.polys['part_01']
BOX=ma.intersection(right_of(349.5))                    # the م's box (with its counter)
ALIF=ma.difference(right_of(349.5))                      # the ا of ما, with its foot on the baseline
STEM=max(getattr(_s:=ALIF.intersection(box(0,0,1000,base-0.75*U)),'geoms',[_s]),key=lambda q:q.area)   # the upright only (drop the box sliver the lean cut leaves)        # the upright part of the ا, above the box height
DOTS2=p.polys['part_03']                                 # the logo's two-dot bar
W_LINE=0.31*U                                            # the baseline stroke: one stroke width, sitting on the baseline
def line(xa,xb):
    a,b=min(xa,xb),max(xa,xb); h=W_LINE
    return Polygon([(a,base),(b,base),(b+h*T,base-h),(a+h*T,base-h)])      # ends lean like the logo's cuts
bx0,_,bx1,_=BOX.bounds; BW=bx1-bx0
KASH=0.67*U                                              # letter to letter along the baseline: the logo's tooth spacing in كيتا
GAP=0.292*U                                              # between the two words-parts: the logo's gap between ما and كيتا
def place(g,dx): return affinity.translate(g,dx,0)
# ---- فا : the logo's own ما, the box gets the dot of ف (one dot: the two-dot bar cut to one dot's length, ends leaning the same way)
def lean_cut(g,frac_keep):
    """keep the right part of the bar, cut along the logo's lean (a single dot from the two-dot bar)"""
    x0,y0,x1,y1=g.bounds; yc=(y0+y1)/2; xc=x0+(x1-x0)*(1-frac_keep)
    H=60; return g.intersection(Polygon([(xc-H*T,yc+H),(xc+H*T,yc-H),(x1+100,yc-H),(x1+100,yc+H)]))
DOT1=lean_cut(DOTS2,0.42)
def dot_above(dot,shape,gap=0.10*U):
    sx=shape.bounds; c=((sx[0]+sx[2])/2, sx[1]); d=dot.bounds
    return affinity.translate(dot,c[0]-(d[0]+d[2])/2+ (gap+ (d[3]-d[1])/2)*T, c[1]-gap-d[3])
fa_alef=unary_union([ma, dot_above(DOT1,BOX)])
# ---- طمة : three boxes on one baseline stroke; the ط also takes the ا's upright on its left side; the ة takes the two-dot bar
right=ma.bounds[0]-GAP                                   # the right end of طمة
ta_box=place(BOX,right-bx1)
def left_x(g,y): return g.intersection(box(-1e4,y-0.5,1e4,y+0.5)).bounds[0]
yy=base-0.85*U                                                          # inside the box's top stroke
stem=place(STEM,left_x(ta_box,yy)-left_x(STEM,yy))                      # the upright's left edge continues the box's left edge
meem_box=place(BOX,ta_box.bounds[0]-KASH-bx1)
teh_box=place(BOX,meem_box.bounds[0]-KASH-bx1)
base_line=line(teh_box.bounds[0], ta_box.bounds[2]-0.5*U)      # starts flush with the ة's left side, runs into the ط's rounded corner
tmh=unary_union([ta_box,stem,meem_box,teh_box,base_line, dot_above(DOTS2,teh_box)])
word=unary_union([fa_alef,tmh])
logo=unary_union([p.polys[k] for k in ('part_01','part_02','part_03','part_06')])
y0=min(word.bounds[1],logo.bounds[1])-8; y1=max(word.bounds[3],logo.bounds[3])+8
def tile(g,lab):
    gx=affinity.translate(g,-g.bounds[0]+8,0); t=render_png([(gx,(25,25,25))],(0,y0,gx.bounds[2]+8,y1),scale=4)
    cv2.putText(t,lab,(10,28),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,200),2); return t
A=tile(word,'FATIMA from the logo pieces'); B=tile(logo,'the logo')
W=max(A.shape[1],B.shape[1]); im=np.vstack([np.pad(t,((0,16),(0,W-t.shape[1]),(0,0)),constant_values=255) for t in (A,B)])
cv2.imwrite('/home/user/Mmmm/fatima_logo_pieces.png',im)
open('/home/user/Mmmm/fatima_logo_pieces.svg','w').write(polygon_svg(affinity.translate(word,-word.bounds[0]+5,-word.bounds[1]+5),(word.bounds[2]-word.bounds[0]+10,word.bounds[3]-word.bounds[1]+10),title='fatima'))
print('one dot area %.1f, empty %s'%(DOT1.area,DOT1.is_empty)); print('box width %.2f U, height %.2f U'%(BW/U,(BOX.bounds[3]-BOX.bounds[1])/U))
