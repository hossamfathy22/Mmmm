# anatomy measured on the Bezier outlines: exact corners from the curves, exact cuts across the strokes (polygon intersection)
import sys, re, json; sys.path.insert(0,'../wt')
import numpy as np, cv2
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union
from scipy.ndimage import gaussian_filter1d
from wt_anat import graph, resample
SVG='womentech_logo_curves.svg'
R=24                                   # raster only for the centre-line; every width is cut on the curves themselves
def parse(svg):
    parts={}
    for pid,d in re.findall(r'<path id="([^"]+)" d="([^"]+)"',open(svg).read()):
        subs=[]
        for sp in d.split('Z'):
            nums=[float(v) for v in re.findall(r'-?\d+\.?\d*',sp)]
            if not nums: continue
            P0=np.array(nums[:2]); segs=[]; k=2
            while k+6<=len(nums):
                c=np.array(nums[k:k+6]).reshape(3,2); segs.append(np.vstack([P0,c])); P0=c[2]; k+=6
            subs.append(segs)
        parts[pid]=subs
    return parts
def bez(c,t): t=t[:,None]; return (1-t)**3*c[0]+3*(1-t)**2*t*c[1]+3*(1-t)*t**2*c[2]+t**3*c[3]
def ring(segs,n=40): return np.vstack([bez(c,np.linspace(0,1,n,endpoint=False)) for c in segs])
def area(P): return 0.5*np.sum(P[:,0]*np.roll(P[:,1],-1)-np.roll(P[:,0],-1)*P[:,1])
PARTS=parse(SVG)
# unit: x-height of the Latin letters, measured on the curves (median height of the lower-case parts)
lat=[np.ptp(np.vstack([ring(s) for s in subs])[:,1]) for k,subs in PARTS.items() if k.startswith('latin') for s in subs[:1]]
U=float(np.median([h for h in lat if 20<h<26]))
print('U (Latin x-height on the curves) = %.3f px'%U)
def corners(subs):
    out=[]
    for segs in subs:
        A=area(ring(segs)); sgn=np.sign(A)
        for i in range(len(segs)):
            a=segs[i]; b=segs[(i+1)%len(segs)]
            tin=a[3]-a[2] if np.linalg.norm(a[3]-a[2])>1e-6 else a[3]-a[1]; tout=b[1]-b[0] if np.linalg.norm(b[1]-b[0])>1e-6 else b[2]-b[0]
            ang=np.degrees(np.arccos(np.clip(np.dot(tin,tout)/(np.linalg.norm(tin)*np.linalg.norm(tout)+1e-12),-1,1)))
            if ang<25: continue
            cr=tin[0]*tout[1]-tin[1]*tout[0]
            # image y points down: for a ring with positive area (in these coords) a left turn (cr<0) bends inward ... decide by testing a point
            out.append(dict(p=a[3],turn=float(ang),cr=float(cr*sgn)))
    return out
def build_poly(subs):
    rings=[ring(s) for s in subs]; rings.sort(key=lambda r:-abs(area(r)))
    outer=Polygon(rings[0]); holes=[r for r in rings[1:] if outer.contains(Point(r[0]))]
    others=[r for r in rings[1:] if not outer.contains(Point(r[0]))]
    poly=Polygon(rings[0],holes)
    if others: poly=unary_union([poly]+[Polygon(o) for o in others])
    return poly
def raster(subs,x0,y0,W,H):
    acc=np.zeros((int(H*R)+4,int(W*R)+4),np.uint8)
    for s in subs:
        Q=(ring(s,60)-[x0,y0])*R; m=np.zeros_like(acc); cv2.fillPoly(m,[np.round(Q*16).astype(np.int32)],1,shift=4); acc^=m
    return acc>0
def cut(poly,P,j,L=3*23):
    d=P[min(j+4,len(P)-1)]-P[max(j-4,0)]; d=d/(np.linalg.norm(d)+1e-12); n=np.array([-d[1],d[0]]); c=P[j]
    g=LineString([c-n*L,c+n*L]).intersection(poly)
    segs=[g] if g.geom_type=='LineString' else [s for s in getattr(g,'geoms',[]) if s.geom_type=='LineString']
    pt=Point(c)
    for s in segs:
        if s.distance(pt)<1e-6: return s.length, np.array(s.coords)
    return 0.0,None
def ang_cls(a):
    o=abs(((a+90)%180)-90); return 'H' if o<35 else ('V' if o>55 else 'J')
LET={'teh_yeh_kaf':[('kaf',0,50.5),('ye',50.5,110.5),('teh',110.5,999)],'meem_noon':[('nun',0,158),('mim',158,999)],'waw_waw':[('waw',0,999)]}
rows=[]; draw=[]
for pid in ('teh_yeh_kaf','meem_noon','waw_waw'):
    subs=PARTS[pid]; poly=build_poly(subs); allp=np.vstack([ring(s) for s in subs])
    x0,y0=allp.min(0)-2; W,H=np.ptp(allp,0)+4
    m=raster(subs,x0,y0,W,H)
    C=corners(subs)
    # concave (crotch) vs convex (tip/cut): a point just off the corner along the bisector inside or outside the shape
    for c in C:
        p=c['p']; near=[np.asarray(poly.exterior.coords)] 
        probe=[p+np.array([dx,dy])*0.6 for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(0.7,0.7),(-0.7,0.7),(0.7,-0.7),(-0.7,-0.7))]
        inside=sum(poly.contains(Point(q)) for q in probe); c['kind']='crotch' if inside>=5 else 'tip'
    sk,nodes,edges=graph(m,U*R)
    toU=lambda P:P/R+[x0,y0]
    for e in edges:
        P=resample(e['path'],step=3.0,sig=4)
        if P is None or len(P)<10: continue
        P=toU(P)                                             # centre-line in original-pixel coordinates (x,y)
        i0,i1=0,len(P)-1
        for end,which in ((e['a'],0),(e['b'],1)):
            if end is None: continue
            kind=nodes[end]['kind']; nd=toU(np.array([[nodes[end]['x'],nodes[end]['y']]]))[0]
            half=np.arange(0,len(P)//2) if which==0 else np.arange(len(P)//2,len(P))
            kn=0 if which==0 else len(P)-1
            want='crotch' if kind=='junction' else 'tip'
            w0=cut(poly,P,min(max(kn,2),len(P)-3))[0]; RAD=max(0.3*U,w0) if kind=='junction' else max(0.35*U,w0)
            best=None
            for c in C:
                if c['kind']!=want or np.linalg.norm(c['p']-nd)>RAD: continue
                k=int(half[np.argmin(np.linalg.norm(P[half]-c['p'],axis=1))])
                if np.linalg.norm(P[k]-c['p'])>0.75*max(cut(poly,P,k)[0],1e-3)+0.5: continue
                if np.sum(np.linalg.norm(np.diff(P[min(k,kn):max(k,kn)+1],axis=0),axis=1))>RAD: continue
                best=k if best is None else (max(best,k) if which==0 else min(best,k))
            if best is None:
                st=int(round(0.5*w0/ (np.linalg.norm(P[1]-P[0])+1e-9)))
                best=st if which==0 else len(P)-1-st
            if which==0: i0=max(i0,best)
            else: i1=min(i1,best)
        if i1-i0<5: continue
        Rr=P[i0:i1+1]
        d=np.gradient(Rr,axis=0); a=np.degrees(np.arctan2(-d[:,1],d[:,0]))
        cl=[ang_cls(v) for v in gaussian_filter1d(np.degrees(np.unwrap(np.radians(a))),6)]
        segs=[]; s=0
        for j in range(1,len(Rr)+1):
            if j==len(Rr) or cl[j]!=cl[s]: segs.append([s,j-1,cl[s]]); s=j
        out=[]
        for g_ in segs:
            if out and (g_[1]-g_[0])*np.linalg.norm(Rr[1]-Rr[0])<0.06*U: out[-1][1]=g_[1]
            else: out.append(g_)
        for s0,s1,c in out:
            Q=Rr[s0:s1+1]
            if c=='H' and Q[0,0]<Q[-1,0]: Q=Q[::-1]
            if c=='V' and Q[0,1]>Q[-1,1]: Q=Q[::-1]
            L=np.sum(np.linalg.norm(np.diff(Q,axis=0),axis=1))/U
            if L<0.05: continue
            js=np.linspace(0,len(Q)-1,21).astype(int); Wd=np.array([cut(poly,Q,j)[0] for j in js])/U
            med=np.median(Wd); Wd=np.where((Wd>1.8*med)|(Wd<=0),np.nan,Wd)
            if np.all(np.isnan(Wd)): continue
            ok=np.where(~np.isnan(Wd))[0]
            lt=[n_ for n_,a_,b_ in LET[pid] if a_<=Q[len(Q)//2,0]<b_][0]
            rows.append(dict(id=len(rows)+1,part=pid,letter=lt,cls=c,len=round(L,3),start=round(float(Wd[ok[0]]),3),mid=round(float(np.nanmean(Wd[9:12])),3),
                             end=round(float(Wd[ok[-1]]),3),thin=round(float(np.nanmin(Wd)),3),thick=round(float(np.nanmax(Wd)),3),thick_at=int(np.nanargmax(Wd)*5),
                             P=Q[::2].round(3).tolist(),Wp=[None if np.isnan(v) else round(float(v),4) for v in Wd]))
            jsv=js[ok]; ends=[cut(poly,Q,int(jsv[0]))[1],cut(poly,Q,int(jsv[-1]))[1]]
            draw.append((c,Q,ends,rows[-1]['id'],L))
    for c in C: draw.append(('corner',c['p'],c['kind'],None,None))
json.dump(dict(U=U,rows=rows),open('vanat.json','w'))
# ---------- picture: the curves, every piece coloured, its start (black) / end (red) cut lines exactly where they are measured
SC=10; allp=np.vstack([ring(s) for k in ('teh_yeh_kaf','meem_noon','waw_waw') for s in PARTS[k]]); X0,Y0=allp.min(0)-4; W_,H_=np.ptp(allp,0)+8
im=np.full((int(H_*SC),int(W_*SC),3),255,np.uint8)
for k in ('teh_yeh_kaf','meem_noon','waw_waw'):
    acc=np.zeros(im.shape[:2],np.uint8)
    for s in PARTS[k]:
        Q=(ring(s,60)-[X0,Y0])*SC; mm=np.zeros_like(acc); cv2.fillPoly(mm,[np.round(Q*16).astype(np.int32)],1,shift=4); acc^=mm
    im[acc>0]=(215,215,215)
q=lambda p:(int((p[0]-X0)*SC),int((p[1]-Y0)*SC))
COL={'H':(220,100,20),'V':(0,180,240),'J':(190,50,190)}
for it in draw:
    if it[0]=='corner': continue
    c,Q,ends,i,L=it
    cv2.polylines(im,[np.array([q(p) for p in Q],np.int32)],False,COL[c],5,cv2.LINE_AA)
    for k_,e_ in enumerate(ends):
        if e_ is not None: cv2.line(im,q(e_[0]),q(e_[-1]),(0,0,0) if k_==0 else (0,0,210),3,cv2.LINE_AA)
    if (c=='H' and L>=0.3) or (c=='V' and L>=0.2) or c=='J':
        pm=q(Q[len(Q)//2]); cv2.putText(im,str(i),(pm[0]-15,pm[1]+(-25 if c!='J' else 45)),cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,0,0),3)
for it in draw:
    if it[0]=='corner':
        p,kind=it[1],it[2]
        if kind=='crotch': cv2.circle(im,q(p),9,(0,0,230),3)
        else: cv2.drawMarker(im,q(p),(0,140,0),cv2.MARKER_TILTED_CROSS,14,3)
cv2.imwrite('vanat.png',im)
print(len(rows),'pieces')
for r in sorted(rows,key=lambda r:(r['cls'],-r['len'])):
    if (r['cls']=='H' and r['len']>=0.3) or (r['cls']=='V' and r['len']>=0.2) or r['cls']=='J':
        print('#%-3d %-4s %s len %.2f | start %.3f mid %.3f end %.3f | thin %.3f thick %.3f @%d%%'%(r['id'],r['letter'],r['cls'],r['len'],r['start'],r['mid'],r['end'],r['thin'],r['thick'],r['thick_at']))
