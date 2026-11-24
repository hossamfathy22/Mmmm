import sys,os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from render import *
OUT=sys.argv[1];os.makedirs(OUT+'/svg',exist_ok=True)
# logo geometry: canvas 736x736, Arabic baseline bottom at y=340.5, horizontal centre of Arabic word = 375
CX,BY=375,339.5
order=['موسيقى','فاطمة','مصطلح','تعكس','الرحمن','صديق','كهف','عشب','نجم','هلال','ظل','فلك','رقص','نبع','خط','هاکوپیان']
for i,w in enumerate(order):
    els,W=build(w) if w!='هاکوپیان' else build(w)
    W-=GAP; x0=CX+W/2 if w!='هاکوپیان' else 580
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="736" height="736" viewBox="0 0 736 736"><rect width="736" height="736" fill="#fff"/><g fill="#000" transform="translate({x0},{BY}) scale(1,-1)">{"".join(els)}</g></svg>'
    name=f'{i+1:02d}-{w}' if w!='هاکوپیان' else '00-هاكوبيان-مرجع'
    open(f'{OUT}/svg/{name}.svg','w').write(svg)
