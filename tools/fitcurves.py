# designer-style outlines: sub-pixel edge -> sharp corners only where the logo has corners -> fewest cubic Beziers within a tolerance
import numpy as np, cv2, json
from skimage.measure import find_contours
from scipy.ndimage import gaussian_filter1d
SRC='/root/.claude/uploads/039d9313-867b-5a26-a74d-07f8024a7ca0/2a029a10-image.jpg'
Y0,Y1,X0,X1=196,322,223,511; S=8; THR=143.0
TOL=0.15*S            # max distance curve <-> edge: 0.15 original pixel
g=cv2.imread(SRC,0).astype(np.float32)[Y0:Y1,X0:X1]
up=cv2.GaussianBlur(cv2.resize(g,None,fx=S,fy=S,interpolation=cv2.INTER_CUBIC),(0,0),3)
up=np.pad(up,2,constant_values=0)
# ---------- Schneider curve fitting
def bez(c,t): t=t[:,None]; return (1-t)**3*c[0]+3*(1-t)**2*t*c[1]+3*(1-t)*t**2*c[2]+t**3*c[3]
def bez1(c,t): t=t[:,None]; return 3*(1-t)**2*(c[1]-c[0])+6*(1-t)*t*(c[2]-c[1])+3*t**2*(c[3]-c[2])
def bez2(c,t): t=t[:,None]; return 6*(1-t)*(c[2]-2*c[1]+c[0])+6*t*(c[3]-2*c[2]+c[1])
def chord(P): d=np.r_[0,np.cumsum(np.linalg.norm(np.diff(P,axis=0),axis=1))]; return d/d[-1]
def gen(P,u,t1,t2):
    A1=t1[None,:]*(3*(1-u)**2*u)[:,None]; A2=t2[None,:]*(3*(1-u)*u**2)[:,None]
    C=np.array([[np.sum(A1*A1),np.sum(A1*A2)],[np.sum(A1*A2),np.sum(A2*A2)]])
    tmp=P-bez(np.array([P[0],P[0],P[-1],P[-1]]),u)
    X=np.array([np.sum(A1*tmp),np.sum(A2*tmp)]); det=C[0,0]*C[1,1]-C[0,1]*C[1,0]
    seg=np.linalg.norm(P[-1]-P[0])
    if abs(det)>1e-12:
        a1=(X[0]*C[1,1]-X[1]*C[0,1])/det; a2=(C[0,0]*X[1]-C[1,0]*X[0])/det
    else: a1=a2=seg/3
    if a1<1e-6*seg or a2<1e-6*seg: a1=a2=seg/3
    return np.array([P[0],P[0]+t1*a1,P[-1]+t2*a2,P[-1]])
def reparam(c,P,u):
    d=bez(c,u)-P; n=np.sum(d*bez1(c,u),1); den=np.sum(bez1(c,u)**2,1)+np.sum(d*bez2(c,u),1)
    with np.errstate(divide='ignore',invalid='ignore'): u2=u-np.where(np.abs(den)>1e-12,n/den,0)
    return np.clip(u2,0,1)
def fit(P,t1,t2,out):
    if len(P)==2:
        d=np.linalg.norm(P[1]-P[0])/3; out.append(np.array([P[0],P[0]+t1*d,P[1]+t2*d,P[1]])); return
    u=chord(P); c=gen(P,u,t1,t2)
    for _ in range(20):
        e=np.linalg.norm(bez(c,u)-P,axis=1)
        if e.max()<=TOL: out.append(c); return
        u=reparam(c,P,u); c=gen(P,u,t1,t2)
    e=np.linalg.norm(bez(c,u)-P,axis=1)
    if e.max()<=TOL: out.append(c); return
    k=int(np.clip(np.argmax(e),1,len(P)-2)); tc=P[k-1]-P[k+1]; tc/=np.linalg.norm(tc)+1e-12
    fit(P[:k+1],t1,tc,out); fit(P[k:],-tc,t2,out)
def unit(v): return v/(np.linalg.norm(v)+1e-12)
# ---------- contours, corners, fitting
cs=find_contours(up,THR)
paths=[]; nodes=0; corners_all=[]
for c in cs:
    P=c[:,::-1]-2.0                                            # (x,y) in the 8x frame
    if len(P)<40: continue
    if np.allclose(P[0],P[-1]): P=P[:-1]
    # resample evenly (1 px at 8x) and smooth very lightly
    d=np.r_[0,np.cumsum(np.linalg.norm(np.diff(np.vstack([P,P[:1]]),axis=0),axis=1))]; L=d[-1]
    s=np.arange(0,L,1.0); Pc=np.vstack([P,P[:1]])
    P=np.stack([np.interp(s,d,Pc[:,0]),np.interp(s,d,Pc[:,1])],1)
    P=np.stack([gaussian_filter1d(P[:,0],1.5,mode='wrap'),gaussian_filter1d(P[:,1],1.5,mode='wrap')],1)
    n=len(P); k=10
    a=np.roll(P,k,0)-P; b=np.roll(P,-k,0)-P
    ang=np.degrees(np.arccos(np.clip(np.sum(a*b,1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)+1e-9),-1,1)))
    turn=180-ang
    cor=[i for i in range(n) if turn[i]>38 and turn[i]==max(turn[(i+j)%n] for j in range(-k,k+1))]
    # drop duplicates
    cor2=[]
    for i in cor:
        if not cor2 or i-cor2[-1]>k: cor2.append(i)
    if len(cor2)>1 and cor2[0]+n-cor2[-1]<=k: cor2.pop()
    segs=[]
    if not cor2:                                                # smooth closed loop: split into 4 tangent-continuous pieces
        idx=[0,n//4,n//2,3*n//4]
        for j in range(4):
            i0,i1=idx[j],idx[(j+1)%4]; Q=P[i0:i1+1] if i1>i0 else np.vstack([P[i0:],P[:i1+1]])
            t1=unit(P[(i0+3)%n]-P[(i0-3)%n]); t2=unit(P[(i1-3)%n]-P[(i1+3)%n]); fit(Q,t1,t2,segs)
    else:
        for j in range(len(cor2)):
            i0=cor2[j]; i1=cor2[(j+1)%len(cor2)]
            Q=P[i0:i1+1] if i1>i0 else np.vstack([P[i0:],P[:i1+1]])
            if len(Q)<2: continue
            t1=unit(Q[min(4,len(Q)-1)]-Q[0]); t2=unit(Q[max(-5,-len(Q))]-Q[-1])     # one-sided tangents: the corner stays sharp
            fit(Q,t1,t2,segs)
        corners_all+= [P[i] for i in cor2]
    paths.append(segs); nodes+=len(segs)
f=lambda p:f'{p[0]/S:.3f} {p[1]/S:.3f}'
W,H=X1-X0,Y1-Y0
d=[]
for segs in paths:
    s=[f'M{f(segs[0][0])}']+[f'C{f(c[1])} {f(c[2])} {f(c[3])}' for c in segs]; d.append(' '.join(s)+'Z')
open('curves_fit.svg','w').write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W*4}" height="{H*4}"><rect width="100%" height="100%" fill="#fff"/><path fill="#000" fill-rule="evenodd" d="{" ".join(d)}"/></svg>\n')
# ---------- check
def raster(scale=S):
    acc=np.zeros((H*scale,W*scale),np.uint8)
    for segs in paths:
        Q=np.vstack([bez(c,np.linspace(0,1,24)) for c in segs])/S*scale
        m=np.zeros_like(acc); cv2.fillPoly(m,[np.round(Q*16).astype(np.int32)],1,shift=4); acc^=m
    return acc>0
ink=(up[2:-2,2:-2]>THR)
r=raster(); iou=(r&ink).sum()/(r|ink).sum()
# edge error: distance from fitted curve samples to the sub-pixel edge, original pixels
E=np.vstack([c[:,::-1]-2.0 for c in cs if len(c)>=40]); from scipy.spatial import cKDTree; T=cKDTree(E)
Smp=np.vstack([bez(c,np.linspace(0,1,30)) for segs in paths for c in segs]); de=T.query(Smp)[0]/S
r1=cv2.resize(r.astype(np.float32),(W,H),interpolation=cv2.INTER_AREA); orig=np.clip((g-32)/(254-32),0,1)
mae=np.abs(r1-orig)[(r1>0.02)|(orig>0.02)].mean()
rep=dict(outlines=len(paths),bezier_segments=nodes,sharp_corners=len(corners_all),iou_vs_bitmap=round(float(iou),4),edge_error_mean_px=round(float(de.mean()),3),edge_error_max_px=round(float(de.max()),3),grey_error_vs_original=round(float(mae),3))
print(rep); json.dump(rep,open('fit_report.json','w'),indent=1)
np.save('paths.npy',np.array([np.array(p) for p in paths],dtype=object),allow_pickle=True); np.save('corners.npy',np.array(corners_all))
