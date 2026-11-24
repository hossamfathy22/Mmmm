import sys,importlib.util,os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from draw import *
def d3b(cx): return [bar2(cx,-17),dia(cx,-29.5)]
WORDS['هاکوپیان']=[('heh_i',1,19),('alef_f',0,11),('kaf_i',1,3),('waw_f',0,10),('tooth_i',1,24.5),('tooth_m',1,63.5),('alef_f',0,13),('noon_f',0)]
DOTS['هاکوپیان']={4:d3b(-24),5:d2b(-40)}
def svg_word(w,scale=1.0,pad=40):
    els,W=build(w)
    top,bot=95,-45
    Wt=W+2*pad;Ht=top-bot+2*pad
    body=f'<g transform="translate({W+pad},{top+pad}) scale(1,-1)">{"".join(els)}</g>'
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{Wt*scale}" height="{Ht*scale}" viewBox="0 0 {Wt} {Ht}">{body}</svg>',Wt,Ht
if __name__=='__main__':
    out=sys.argv[1];os.makedirs(out,exist_ok=True)
    for w in list(WORDS):
        s,_,_=svg_word(w)
        open(f'{out}/{w}.svg','w').write(s.replace('<g','<g fill="#000"',1))
