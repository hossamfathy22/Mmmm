# Draw ماكيتا back from its measurements, step by step, to see which measurement is wrong.
# Measurement on the outline: straight edges between corners (angle, position), stroke width = distance between
# opposite parallel edges, corners = turn + rounding radius.
import sys,os,json; sys.path.insert(0,'/home/user/Mmmm'); os.environ['LOGO_STUDIO_PROJECTS']='/home/user/Mmmm/logo_studio/projects'
import numpy as np, cv2
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely import affinity
from logo_studio.core.project import Project
from logo_studio.core.fullmeasure import corners, families
p=Project('20261128-035950-a4f7bf'); U=p.state['U']; base=p.state['baseline']
AR=('part_01','part_02','part_03','part_06')
polys={k:p.polys[k] for k in AR}
from logo_studio.core.fullmeasure import _circle
from scipy.ndimage import gaussian_filter1d
def corners2(polys,U,step=0.01,win=0.03,straight_r=1.0,min_edge=0.05,min_turn=8.0,rings_out=None):
    """split every outline into straight edges and the corners between them, by curvature, so a corner is found
    whatever its rounding radius: a point is on a straight edge when the outline there bends less than a circle of
    radius straight_r U would; a corner is everything between two straight edges"""
    out=[]
    for part,poly in polys.items():
        geoms=[poly] if poly.geom_type=='Polygon' else list(poly.geoms)
        for gi,q in enumerate(geoms):
            for ri,ring in enumerate([q.exterior]+list(q.interiors)):
                L=ring.length; n=max(int(L/(step*U)),60); ds=L/n
                P=np.array([ring.interpolate(t*L).coords[0] for t in np.linspace(0,1,n,endpoint=False)])
                rk='%s/%d/%d'%(part,gi,ri)
                if rings_out is not None: rings_out[rk]=P
                k=max(int(win*U/ds),2)
                d=np.roll(P,-k,0)-np.roll(P,k,0); th=np.unwrap(np.arctan2(d[:,1],d[:,0]))
                th=np.r_[th[-1:]-2*np.pi*np.sign(th[-1]-th[0]) if False else th[-1:],th,th[:1]]
                kap=np.angle(np.exp(1j*(np.roll(np.arctan2(d[:,1],d[:,0]),-1)-np.arctan2(d[:,1],d[:,0]))))/ds
                straight=np.abs(kap)*straight_r*U<1.0
                # straight runs long enough are edges; open the ring at a curved point
                if straight.all() or not straight.any(): continue
                st=int(np.argmin(straight)); idx=np.roll(np.arange(n),-st); sv=straight[idx]
                runs=[];a=0
                while a<n:
                    b=a
                    while b+1<n and sv[b+1]==sv[a]: b+=1
                    if sv[a] and (b-a+1)*ds>=min_edge*U: runs.append((idx[a],idx[b]))
                    a=b+1
                # an S-shaped zone between two straight edges is two corners: split it at the inflection (zero-length tangent edge)
                ext=[]
                for (a0,b0),(a1,b1) in zip(runs,runs[1:]+runs[:1]):
                    ext.append((a0,b0)); ci=[t%n for t in range(b0,a1+1 if a1>=b0 else a1+n+1)]
                    kz=gaussian_filter1d(kap[ci],max(k//2,1),mode='nearest'); sg=np.sign(kz)
                    cum=0; cur=sg[0]; start=0; pieces=[]
                    for t in range(1,len(ci)+1):
                        if t==len(ci) or (sg[t]!=cur and sg[t]!=0):
                            pieces.append((start,t-1,float(np.degrees(np.sum(kz[start:t])*ds)))); start=t; cur=sg[t] if t<len(ci) else cur
                    big=[q for q in pieces if abs(q[2])>=min_turn]
                    for q0,q1 in zip(big,big[1:]):
                        if np.sign(q0[2])!=np.sign(q1[2]):
                            m=ci[(q0[1]+q1[0])//2]; ext.append((m,m))
                runs=ext
                area=0.5*np.sum(P[:,0]*np.roll(P[:,1],-1)-np.roll(P[:,0],-1)*P[:,1])
                def fitdir(a,b):
                    if a==b: u=P[(a+1)%n]-P[a-1]; return u/np.linalg.norm(u)
                    ii=[t%n for t in range(a,b+1 if b>=a else b+n+1)]; Q=P[ii]; c=Q.mean(0); u=np.linalg.svd(Q-c)[2][0]
                    return u if np.dot(Q[-1]-Q[0],u)>=0 else -u
                for (a0,b0),(a1,b1) in zip(runs,runs[1:]+runs[:1]):
                    u0=fitdir(a0,b0); u1=fitdir(a1,b1)
                    turn=float(np.degrees(np.arctan2(u0[0]*u1[1]-u0[1]*u1[0],u0@u1)))
                    ci=[t%n for t in range(b0,a1+1 if a1>=b0 else a1+n+1)]
                    if abs(turn)<min_turn: 
                        out.append(dict(part=part,ring=rk,at=P[ci[len(ci)//2]].tolist(),turn=round(abs(turn),1),radius=0.0,kind='flat',i0=int(b0),i1=int(a1),n=n,hole=ri>0)); continue
                    arc=max(len(ci)-1-2*k,0)*ds                     # the curved part only (the window blurs k points into each side)
                    inner_pts=P[ci[k:len(ci)-k]] if len(ci)>2*k+3 else None
                    if arc<0.02*U or inner_pts is None: r=arc/np.radians(abs(turn))/U if arc>0 else 0.0
                    else: r=_circle(inner_pts)[1]/U
                    if arc<0.02*U and r<0.02: r=0.0
                    convex=(turn*np.sign(area))>0
                    out.append(dict(part=part,ring=rk,at=P[ci[len(ci)//2]].tolist(),turn=round(abs(turn),1),radius=round(float(r),3),
                                    kind='outer' if (convex!=(ri>0)) else 'inner',i0=int(b0),i1=int(a1),n=n,hole=ri>0))
    return out
rings={}; co=corners2(polys,U,rings_out=rings)
co=[c for c in co]
def ang(u):                                   # 0 = horizontal, rising to the right positive, folded to [0,180)
    a=np.degrees(np.arctan2(-u[1],u[0]))%180; return a-180 if a>135 else a      # 135..180 folds next to 0
# ---------- 1. straight edges between corners
segs=[]; ring_corners={}
for rk,P in rings.items():
    cs=[c for c in co if c['ring']==rk]
    ring_corners[rk]=cs; n=len(P)
    for a,b in zip(cs,cs[1:]+cs[:1]):
        i0,i1=a['i1'],b['i0']
        idx=[i0] if i1==i0 else [t%n for t in range(i0, i1 if i1>i0 else i1+n)]
        if len(idx)<4:
            i_=i0%n; c=P[i_]; u=P[(i_+2)%n]-P[i_-2]; u=u/np.linalg.norm(u); idx=[(i_-1)%n,i_,(i_+1)%n]; Q=P[idx]
        else:
            m=len(idx); core=idx[int(0.1*m):max(int(0.9*m),int(0.1*m)+2)]
            Q=P[core]; c=Q.mean(0); _,_,vt=np.linalg.svd(Q-c); u=vt[0]
        if np.dot(P[idx[-1]]-P[idx[0]],u)<0: u=-u
        res=np.abs((Q-c)@np.array([-u[1],u[0]]))
        segs.append(dict(ring=rk,c=c,u=u,ang=ang(u),len=float(np.linalg.norm(P[idx[-1]]-P[idx[0]])/U),
                         straight=float(res.max()/U),pts=P[idx]))
print('edges',len(segs),' corners',len(co))
# ---------- 1b. corner radius from the tip: how far the outline stays from where its two edges meet
from shapely.geometry import LinearRing, Point
def _inter(c1,u1,c2,u2):
    M=np.array([u1,-u2]).T
    if abs(np.linalg.det(M))<1e-6: return None
    t=np.linalg.solve(M,c2-c1); return c1+u1*t[0]
byr={}
for k_,s_ in enumerate(segs): byr.setdefault(s_['ring'],[]).append(k_)
for rk,ks in byr.items():
    cs=[c for c in co if c['ring']==rk]; ring=LinearRing(rings[rk])
    for a_,b_ in zip(ks,ks[1:]+ks[:1]):
        cc=cs[ks.index(b_)%len(cs)]; cc['radius_fit']=cc['radius']
        if cc['turn']<20: continue
        V=_inter(segs[a_]['c'],segs[a_]['u'],segs[b_]['c'],segs[b_]['u'])
        if V is None or np.linalg.norm(V-np.array(cc['at']))>0.8*U: continue
        h=ring.distance(Point(*V)); th=np.radians(180-cc['turn'])
        cc['radius']=round(float(h/(1/np.sin(th/2)-1))/U,3); cc['tip_gap']=round(h/U,3)
def clusters(vals,weights,gap):
    """1-D families: sort, start a new family wherever two neighbours are more than gap apart"""
    o=np.argsort(vals); v=np.array(vals)[o]; w=np.array(weights)[o]; out=[]; cur=[0]
    for t in range(1,len(v)+1):
        if t==len(v) or v[t]-v[t-1]>gap:
            vv=v[cur]; ww=w[cur]; out.append(dict(mean=round(float(np.average(vv,weights=ww)),3),sd=round(float(vv.std()),3),min=round(float(vv.min()),3),max=round(float(vv.max()),3),n=len(cur)))
            cur=[t]
        else: cur.append(t)
    return sorted(out,key=lambda f:-f['n'])
# ---------- 2. families of edge angles (weighted by length)
A=np.array([s['ang'] for s in segs]); Lw=np.array([s['len'] for s in segs])
fam=clusters(list(A),list(Lw),gap=3.0)
print('angle families',json.dumps(fam))
# ---------- 3. widths: opposite parallel edges
def nrm(u): return np.array([-u[1],u[0]])
pairs=[]
for i,s in enumerate(segs):
    best=None
    if s['len']<0.03: continue
    for j,t in enumerate(segs):
        if j==i or t['len']<0.03: continue
        da=abs(((s["ang"]-t["ang"])+90)%180-90)
        if da>4: continue
        d=(t['c']-s['c'])@nrm(s['u'])
        if abs(d)<0.05*U or abs(d)>0.6*U: continue
        # facing each other: overlap along the edge direction
        ps=s['pts']@s['u']; pt=t['pts']@s['u']; ov=min(ps.max(),pt.max())-max(ps.min(),pt.min())
        if ov<0.3*min(np.ptp(ps),np.ptp(pt)): continue
        mid=(s['c']+t['c'])/2                      # the gap between must be ink
        if not unary_union(list(polys.values())).buffer(0).contains(__import__('shapely.geometry',fromlist=['Point']).Point(*mid)): continue
        if best is None or abs(d)<abs(best[1]): best=(j,d)
    if best: pairs.append((i,best[0],abs(best[1])/U))
W=[w for i,j,w in pairs]
for i,j,w in sorted(pairs,key=lambda q:q[2]):
    if i<j: print('   width %.3f U  angle %.1f  len %.2f  at'%(w,segs[i]['ang'],min(segs[i]['len'],segs[j]['len'])),np.round((segs[i]['c']+segs[j]['c'])/2/U,2),segs[i]['ring'])
WF=clusters(W,[segs[i]['len'] for i,j,w in pairs],gap=0.012)
print('widths', json.dumps(WF))
json.dump(dict(segs=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in s.items() if k!='pts'} for s in segs],
               pairs=pairs,corners=co,angle_families=fam),open('/tmp/claude-0/-home-user-Mmmm/039d9313-867b-5a26-a74d-07f8024a7ca0/scratchpad/mk_edges.json','w'))
# ---------- 4. draw it back, level by level
def inter(c1,u1,c2,u2):
    M=np.array([u1,-u2]).T
    if abs(np.linalg.det(M))<1e-6: return None
    t=np.linalg.solve(M,c2-c1); return c1+u1*t[0]
def fillet(V,r,steps=10):
    out=[]; n=len(V)
    for k in range(n):
        A,B,C=V[k-1],V[k],V[(k+1)%n]
        a=A-B; b=C-B; la,lb=np.linalg.norm(a),np.linalg.norm(b)
        if r[k]<=0 or la<1e-6 or lb<1e-6: out.append(B); continue
        a/=la; b/=lb; th=np.arccos(np.clip(a@b,-1,1))
        if th<1e-3 or th>np.pi-1e-3: out.append(B); continue
        d=min(r[k]/np.tan(th/2),0.48*la,0.48*lb); rr=d*np.tan(th/2)
        bis=(a+b)/np.linalg.norm(a+b); O=B+bis*rr/np.sin(th/2)
        p0=B+a*d; p1=B+b*d; s0=np.arctan2(*(p0-O)[::-1]); s1=np.arctan2(*(p1-O)[::-1])
        ds=np.angle(np.exp(1j*(s1-s0)))
        for t in np.linspace(0,1,steps): out.append(O+rr*np.array([np.cos(s0+ds*t),np.sin(s0+ds*t)]))
    return np.array(out)
fam_means=[f['mean'] for f in fam if f['n']>=2]
def snap(a):
    f=min(fam_means,key=lambda m:abs(((a-m)+90)%180-90)); return f if abs(((a-f)+90)%180-90)<4 else a
def unit(a): r=np.radians(a); return np.array([np.cos(r),-np.sin(r)])
Wu=float(np.mean(W))*U
outer_f=[f for f in families([c['radius'] for c in co if c['kind']=='outer' and c['radius']>0],gap=0.03)]
inner_f=[f for f in families([c['radius'] for c in co if c['kind']=='inner' and c['radius']>0],gap=0.03)]
# corners are unified within their own kind (outer/inner) and their own turn family, then by radius
CG={}
for kind in ('outer','inner'):
    cc_=[c for c in co if c['kind']==kind and c['radius']>0]
    if not cc_: continue
    tf=clusters([c['turn'] for c in cc_],[1]*len(cc_),gap=12.0)
    for f in tf:
        mem=[c for c in cc_ if f['min']<=c['turn']<=f['max']]
        rf=clusters([c['radius'] for c in mem],[1]*len(mem),gap=0.03)
        for c in mem:
            r_=[g_ for g_ in rf if g_['min']<=c['radius']<=g_['max']][0]
            c['radius_unified']=r_['mean']; c['group']=(kind,f['mean'],r_['mean'],r_['n'])
        CG[(kind,f['mean'])]=dict(turn=f,radii=rf)
for (kind,t),v in CG.items(): print('corners %-5s turn %6.1f (n=%d): radii'%(kind,t,v['turn']['n']),[(r['mean'],r['n']) for r in v['radii']])
def snap_r(c): return c.get('radius_unified',c['radius'])
print('outer radius families',[(f['mean'],f['n']) for f in outer_f]); print('inner radius families',[(f['mean'],f['n']) for f in inner_f])
def rebuild(level):
    L=[dict(c=s['c'].copy(),u=s['u'].copy()) for s in segs]
    if level>=3:
        for l,s in zip(L,segs): l['u']=unit(snap(s['ang']))*np.sign(unit(snap(s['ang']))@s['u'] or 1)
    if level>=4:
        done=set()
        for i,j,w in sorted(pairs,key=lambda q:-min(segs[q[0]]['len'],segs[q[1]]['len'])):
            if i in done or j in done: continue
            fw=[f['mean'] for f in WF if f['n']>=2]
            Wu=min(fw,key=lambda m_:abs(m_-w))*U if fw else w*U
            n_=nrm(L[i]['u']); d=(L[j]['c']-L[i]['c'])@n_; m=L[i]['c']+n_*d/2
            L[i]['c']=L[i]['c']+n_*((d/2-np.sign(d)*Wu/2)); L[j]['c']=L[j]['c']-n_*((d/2-np.sign(d)*Wu/2)); done|={i,j}
    shapes={k:[] for k in polys}
    idx=0; byring={}
    for k,s in enumerate(segs): byring.setdefault(s['ring'],[]).append(k)
    for rk,ks in byring.items():
        cs=ring_corners[rk]; V=[]; R=[]
        for a_,b_ in zip(ks,ks[1:]+ks[:1]):           # corner between edge a_ and the next edge b_
            q=inter(L[a_]['c'],L[a_]['u'],L[b_]['c'],L[b_]['u'])
            cc=cs[(ks.index(b_))%len(cs)]
            if q is None or np.linalg.norm(q-np.array(cc['at']))>0.4*U: q=np.array(cc['at'])
            V.append(q)
            R.append(0 if level<2 else (cc['radius'] if level<5 else snap_r(cc))*U)
        V=np.array(V); P=fillet(V,np.array(R)) if level>=2 else V
        part,gi,ri=rk.split('/'); shapes[part].append((int(gi),int(ri),Polygon(P).buffer(0)))
    out={}
    for part,lst in shapes.items():
        g=[]
        for gi in sorted(set(x[0] for x in lst)):
            ext=[x[2] for x in lst if x[0]==gi and x[1]==0][0]
            holes=[x[2] for x in lst if x[0]==gi and x[1]>0]
            g.append(ext.difference(unary_union(holes)) if holes else ext)
        out[part]=unary_union(g)
    return out
orig=unary_union(list(polys.values()))
LEVELS=['1 edges as measured, sharp corners','2 + each corner rounded with its own radius','3 + edge angles unified to their family',
        '4 + stroke width unified to its family','5 + corner radii unified to their family']
from logo_studio.core.build import render_png
x0,y0,x1,y1=orig.bounds; x0-=8;y0-=8;x1+=8;y1+=8; SC=4
tiles=[]; summary=[]
for lv in range(1,6):
    g=unary_union(list(rebuild(lv).values()))
    iou=g.intersection(orig).area/g.union(orig).area
    miss=orig.difference(g); extra=g.difference(orig)
    hd=g.boundary.hausdorff_distance(orig.boundary)/U
    # where is the worst place: sample both outlines
    from shapely.geometry import Point
    worst=(0,None)
    for A_,B_ in ((g.boundary,orig.boundary),(orig.boundary,g.boundary)):
        for t in np.linspace(0,1,4000):
            q=A_.interpolate(t,normalized=True); d=B_.distance(q)
            if d>worst[0]: worst=(d,(q.x/U,q.y/U))
    print('  level',lv,'worst %.3f U at (%.2f, %.2f)'%(worst[0]/U,*worst[1]))
    summary.append(dict(level=LEVELS[lv-1],IoU=round(iou,4),hausdorff_U=round(hd,3),missing_U2=round(miss.area/U/U,4),extra_U2=round(extra.area/U/U,4)))
    t=render_png([(orig.intersection(g),(205,205,205)),(miss,(40,40,220)),(extra,(220,120,30))],(x0,y0,x1,y1),scale=SC)
    # rebuilt outline in black
    for q in getattr(g,'geoms',[g]):
        for ring in [q.exterior]+list(q.interiors):
            pts=((np.array(ring.coords)-[x0,y0])*SC).astype(np.int32); cv2.polylines(t,[pts],True,(20,20,20),1,cv2.LINE_AA)
    cv2.putText(t,'%s   IoU %.3f  max off %.3f U'%(LEVELS[lv-1],iou,hd),(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,160),2)
    tiles.append(t)
im=np.vstack([np.pad(t,((0,10),(0,0),(0,0)),constant_values=255) for t in tiles])
cv2.imwrite('/home/user/Mmmm/makita_redraw_levels.png',im)
for s in summary: print(s)
# edges whose angle is not in a family (single odd ones) and their straightness
print('odd edges:')
for k,s in enumerate(segs):
    if all(abs(((s['ang']-m)+90)%180-90)>=4 for m in fam_means):
        print('  edge',k,s['ring'],'angle %.1f len %.2f straight-dev %.3f U at'%(s['ang'],s['len'],s['straight']),np.round(s['c']/U,2))
json.dump(summary,open('/tmp/claude-0/-home-user-Mmmm/039d9313-867b-5a26-a74d-07f8024a7ca0/scratchpad/mk_levels.json','w'),ensure_ascii=False)
# ---------- 5. the clean drawing from the unified numbers, next to the logo; the corrected numbers
from logo_studio.core.build import polygon_svg
g5=unary_union(list(rebuild(5).values()))
def tile(gm,lab):
    t=render_png([(gm,(25,25,25))],(x0,y0,x1,y1),scale=SC); cv2.putText(t,lab,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,160),2); return t
im=np.vstack([np.pad(t,((0,10),(0,0),(0,0)),constant_values=255) for t in (tile(g5,'drawn from the unified numbers only'),tile(orig,'the logo'))])
cv2.imwrite('/home/user/Mmmm/makita_redraw.png',im)
open('/home/user/Mmmm/makita_redraw.svg','w').write(polygon_svg(affinity.translate(g5,-g5.bounds[0]+5,-g5.bounds[1]+5) if False else g5,(x1,y1),title='makita redrawn'))
numbers=dict(unit_px=U,
    edge_angle_families=[f for f in fam if f['n']>=2], stroke_width_families=WF,
    corner_families=[dict(kind=k,turn=v['turn'],radii=v['radii']) for (k,t),v in CG.items()],
    redraw_levels=summary)
json.dump(numbers,open('/home/user/Mmmm/makita_numbers_corrected.json','w'),indent=1,ensure_ascii=False)
