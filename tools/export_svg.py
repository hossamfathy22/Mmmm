import numpy as np, cv2, json
S=8
paths=np.load('paths.npy',allow_pickle=True); ink=np.load('ink8.npy')
n,lab,st,_=cv2.connectedComponentsWithStats(ink.astype(np.uint8))
AR={1:'damma',9:'waw_waw',4:'bar_above_meem',7:'noon_dot',8:'meem_noon',5:'teh_dots',10:'yeh_dots',3:'chevron_above_yeh',11:'bar_below_yeh',6:'kaf_mark',2:'teh_yeh_kaf'}
lat=sorted([i for i in range(1,n) if i not in AR],key=lambda i:st[i,0])
NAME=dict(AR); NAME.update({k:f'latin_{j+1:02d}' for j,k in enumerate(lat)})
groups={}
for segs in paths:
    p=np.asarray(segs[0][0]); x,y=int(p[0]),int(p[1]); w=lab[max(y-4,0):y+5,max(x-4,0):x+5]; v=w[w>0]
    k=int(np.bincount(v).argmax()) if len(v) else 0
    groups.setdefault(k,[]).append(segs)
f=lambda q:f'{q[0]/S:.3f} {q[1]/S:.3f}'
def d_of(segs_list):
    out=[]
    for segs in segs_list:
        out.append(' '.join([f'M{f(segs[0][0])}']+[f'C{f(c[1])} {f(c[2])} {f(c[3])}' for c in segs])+'Z')
    return ' '.join(out)
Y0,Y1,X0,X1=196,322,223,511; W,H=X1-X0,Y1-Y0
rows=[]
def grp(title,keys):
    s=[f'  <g id="{title}" inkscape:groupmode="layer" inkscape:label="{title}">']
    for k in keys:
        if k not in groups: continue
        nseg=sum(len(x) for x in groups[k]); rows.append((NAME[k],len(groups[k]),nseg))
        s.append(f'    <path id="{NAME[k]}" d="{d_of(groups[k])}"/>')
    return s+['  </g>']
order_ar=[2,6,11,3,10,5,8,7,4,9,1]
svg=['<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
     f'viewBox="0 0 {W} {H}" width="{W*4}" height="{H*4}">',
     ' <!-- Women Tech logo, traced to cubic Bezier outlines. Units: pixels of the original 735 x 518 image, cropped at x=223, y=196. -->',
     ' <g fill="#000" fill-rule="evenodd">']+grp('arabic',order_ar)+grp('latin',lat)+[' </g>','</svg>']
open('womentech_logo_curves.svg','w').write('\n'.join(svg)+'\n')
for r in rows: print('%-20s outlines %d  curve segments %d'%r)
print('total segments',sum(r[2] for r in rows))
