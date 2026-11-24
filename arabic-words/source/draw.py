# Hand-drawn Arabic glyph system replicating the logo's construction.
# Units = logo pixels. Local glyph coords: x=0 at glyph's right edge, x<0 leftwards; y=0 baseline bottom, y up.
import sys
BT=10      # baseline bar thickness
VW=8.5     # vertical stem width
S=9.2      # curve stroke
A=70       # alef / lam / kaf height
TI=43      # initial tooth
TM=31      # medial tooth
K=18       # kashida between joined letters
GAP=11     # gap between unjoined letters
SL=2.0     # slant of top cuts (right side higher)

def stem(xc,y1,y0=0,w=VW):
    return f'<polygon points="{xc-w/2},{y0} {xc+w/2},{y0} {xc+w/2},{y1+SL/2} {xc-w/2},{y1-SL/2}"/>'
def rect(x1,x2,y1,y2): return f'<rect x="{min(x1,x2)}" y="{min(y1,y2)}" width="{abs(x2-x1)}" height="{abs(y2-y1)}"/>'
def P(d,w=S,cap='butt'): return f'<path d="{d}" fill="none" stroke="#000" stroke-width="{w}" stroke-linecap="{cap}" stroke-linejoin="round"/>'
def dia(cx,cy,r=7.6): return f'<polygon points="{cx},{cy+r} {cx+r},{cy} {cx},{cy-r} {cx-r},{cy}"/>'
def bar2(cx,cy,w=30,h=8.7): return rect(cx-w/2,cx+w/2,cy-h/2,cy+h/2)
# dots
def d1a(cx,top): return [dia(cx,top+15)]
def d2a(cx,top): return [bar2(cx,top+13)]
def d3a(cx,top): return [bar2(cx,top+13),dia(cx,top+30)]
def d1b(cx,bot=0): return [dia(cx,bot-17)]
def d2b(cx,bot=0): return [bar2(cx,bot-17)]

# ---- reusable parts ----
def bowl(xr,xl,hr,hl,depth=-12):
    """ن-style bowl: right stem centre xr (height hr), left stem centre xl (height hl), bottom at depth."""
    c=depth+S/2
    return [stem(xr,hr,4), P(f"M {xr} 8 L {xr} {c+10} Q {xr} {c} {xr-12} {c-0.5} L {xl+12} {c} Q {xl} {c} {xl} {c+12} L {xl} {hl-2}"), stem(xl,hl,c+10,7.5)]
def loop(xr,xl,h,y0=0,rtl=16,rtr=12):
    """و/ف-style loop sitting on y0 (its bottom is the baseline bar)."""
    xr2=xr-S/2; xl2=xl+S/2; t=y0+h-S/2; b=y0+BT/2
    return [P(f"M {xr2} {b} L {xr2} {t-rtr} Q {xr2} {t} {xr2-rtr} {t} L {xl2+rtl} {t} Q {xl2} {t} {xl2} {t-rtl} L {xl2} {b+6} Q {xl2} {b} {xl2+8} {b}")]
def tailflat(xr,length,depth=-17.5):
    """و/ر tail: stroke down from xr then flat bar going left."""
    x=xr-S/2
    return [P(f"M {x} 6 L {x} {depth+10} Q {x} {depth} {x-12} {depth} L {x-length} {depth}",w=9.5)]

G={}
def glyph(name):
    def deco(f): G[name]=f; return f
    return deco
# each glyph returns (elements, width, (barL,barR) or None)
@glyph('alef_s')
def _(): return [stem(-4.25,A)],8.5,None
@glyph('alef_f')
def _(): return [stem(-4.25,A,14), P("M -4.25 16 L -4.25 13 Q -4.25 5 4 5 L 8 5",w=9.6)],8.5,(0,0)
@glyph('lam_i')
def _(): return [stem(-4.25,A)],8.5,(-8.5,0)
@glyph('lam_m')
def _(): return [stem(-4.25,A)],8.5,(-8.5,0)
@glyph('lam_f')
def _(): return bowl(-4.25,-58,A,36),62,(-8,0)
@glyph('lam_s')
def _(): return bowl(-4.25,-58,A,36),62,None
@glyph('tooth_i')
def _(): return [stem(-4.25,TI)],8.5,(-8.5,0)
@glyph('tooth_m')
def _(): return [stem(-4.25,TM)],8.5,(-8.5,0)
@glyph('noon_f')
def _(): return bowl(-4.5,-61,49,38)+d1a(-32,17),65,(-8,0)
@glyph('yaa_f')
def _(): return [P("M 4 5 L -10 5 Q -26 5 -32 -6 Q -38 -18 -54 -18 L -66 -18 Q -82 -18 -82 -2 L -82 14",w=9.4)],84,None
@glyph('baa_f')
def _(): return [stem(-4.25,20), P("M -60 22 L -60 13 Q -60 5 -50 5",w=9)]+d1b(-32),64,(-52,0)
@glyph('waw_f')
def _(): return loop(0,-41,43)+tailflat(0,46),41,(-41,0)
@glyph('raa_f')
def _(): return [stem(-4.6,22,4,9.2)]+tailflat(0,34),9.2,(-4,0)
@glyph('raa_s')
def _(): return [stem(-4.6,22,4,9.2)]+tailflat(0,34),9.2,(-9.2,0)
@glyph('kaf_i')
def _():
    xs=-5; # right stem centre
    e=[P(f"M {xs} 5 L {xs} 30 Q {xs} 39 {xs-10} 39 L -46 39",w=8.5),
       P("M -48 41 L -2 66",w=9)]
    return e,50,(-50,0)
@glyph('kaf_f')
def _():
    e=[rect(-52,2,0,BT),P("M -52 5 Q -60 5 -60 14 L -60 38 L -14 66",w=9.4)]
    return e,64,(-54,0)
@glyph('heh_i')
def _():
    e=[P("M -68 52 L -14 34 Q -5 31 -5 22 L -5 6",w=9.5),
       P("M -63 47 L -63 8",w=8.5),
       P("M -60 40 Q -44 35 -40 24 L -38 6",w=8.5)]
    return e,70,(-70,0)
@glyph('teh_marb_f')
def _(): return loop(0,-34,36)+d2a(-17,36),34,(-34,0)
@glyph('feh_i')
def _(): return loop(0,-38,43)+d1a(-19,43),38,(-38,0)
@glyph('qaf_i')
def _(): return loop(0,-38,43)+d2a(-19,43),38,(-38,0)
@glyph('feh_f')
def _(): return loop(0,-36,43)+[P("M -36 5 L -72 5 Q -80 5 -80 13 L -80 20",w=BT)]+d1a(-18,43),84,(-80,0)
@glyph('qaf_f')
def _(): return loop(0,-36,43)+bowl(-4.5,-70,8,26,depth=-20)[1:]+d2a(-18,43),74,(-36,0)
@glyph('meem_i')
def _(): return loop(0,-30,27,rtl=11,rtr=10),30,(-30,0)
@glyph('meem_m')
def _(): return loop(0,-30,27,rtl=11,rtr=10),30,(-30,0)
@glyph('meem_f')
def _(): return loop(0,-30,27,rtl=11,rtr=10)+[stem(-26,8,-40,8.5)],30,(-30,0)
@glyph('seen_i')
def _(): return [stem(-4.25,22),stem(-19,22),stem(-34,22)],38.5,(-38.5,0)
@glyph('seen_f')
def _(): return [stem(-4.25,22),stem(-19,22)]+bowl(-34,-88,22,30),92,(-34,0)
@glyph('sheen_m')
def _(): return [stem(-4.25,22),stem(-19,22),stem(-34,22)]+d3a(-19,22),38.5,(-38.5,0)
@glyph('sad_i')
def _(): return loop(0,-48,27,rtl=13,rtr=13)+[stem(-55,18)],59,(-59,0)
@glyph('sad_f')
def _(): return loop(0,-48,27,rtl=13,rtr=13)+bowl(-52,-104,16,30),108,(-52,0)
@glyph('tah_i')
def _(): return loop(0,-42,30,rtl=12,rtr=12)+[stem(-34,A,20)],42,(-42,0)
@glyph('tah_f')
def _(): return loop(0,-42,30,rtl=12,rtr=12)+[stem(-34,A,20)],42,(-42,0)
@glyph('zah_i')
def _(): return loop(0,-42,30,rtl=12,rtr=12)+[stem(-34,A,20)]+d1a(-17,30),42,(-42,0)
@glyph('hah_i')
def _(): return [P("M 0 37 L -4 42 L -34 42 Q -42 42 -46 34 L -56 12 Q -60 5 -68 5",w=8.8)],60,(-64,-58)
@glyph('hah_f')
def _(): return [P("M 4 5 L -6 42 L -34 42 Q -42 42 -46 34 L -54 16 Q -58 8 -58 -2 L -58 -16 Q -58 -32 -40 -32 L -4 -32",w=8.8)],60,None
@glyph('khah_i')
def _(): return [P("M 0 37 L -4 42 L -34 42 Q -42 42 -46 34 L -56 12 Q -60 5 -68 5",w=8.8)]+d1a(-22,42),60,(-64,-58)
@glyph('jeem_m')
def _(): return [P("M 4 5 L -6 42 L -34 42 Q -42 42 -46 34 L -56 12 Q -60 5 -68 5",w=8.8)]+d1b(-30),60,(-64,-58)
@glyph('ain_i')
def _(): return [P("M -40 40 Q -34 50 -20 50 Q -4 50 -4 37 Q -4 25 -20 23 L -30 22 Q -42 20 -44 6",w=8.8)],46,(-46,-38)
@glyph('ain_m')
def _(): return [P("M -2 34 L -30 34 L -18 5 Z",w=8.6)],32,(-32,0)
@glyph('ain_f')
def _(): return [P("M -2 34 L -30 34 L -18 5 Z",w=8.6),P("M -18 5 Q -26 2 -34 -6 Q -40 -14 -40 -20 Q -40 -32 -26 -32 L -2 -32",w=8.6)],40,(-20,0)
@glyph('dal_f')
def _(): return [P("M -34 42 L -6 30 Q -2 28 -2 22 L -2 8",w=9)],36,(-34,0)

def word(seq):
    """seq: list of (glyph, joinsLeft). joinsRight derived from previous joinsLeft."""
    out=[];x=0;prevJ=False
    for it in seq:
        name,jl=it[0],it[1]; k=it[2] if len(it)>2 else (K if jl else GAP)
        e,W,bar=G[name]()
        if bar is not None or prevJ or jl:
            bl,br=(bar if bar else (-W,0))
            if prevJ: br=0
            L=-(W+k) if jl else bl
            out.append(f'<g transform="translate({x},0)">{rect(L,br,0,BT)}</g>')
        out.append(f'<g transform="translate({x},0)">{"".join(e)}</g>')
        x-= W+k
        prevJ=jl
    return out, -(x + (GAP))

WORDS={
 'موسيقى':[('meem_i',1),('waw_f',0),('seen_i',1),('tooth_m',1),('qaf_i',1),('yaa_f',0)],
 'فاطمة':[('feh_i',1),('alef_f',0),('tah_i',1),('meem_m',1),('teh_marb_f',0)],
 'مصطلح':[('meem_i',1),('sad_i',1),('tah_i',1),('lam_m',1),('hah_f',0)],
 'تعكس':[('tooth_i',1),('ain_m',1),('kaf_i',1),('seen_f',0)],
 'الرحمن':[('alef_s',0),('lam_i',1),('raa_f',0),('hah_i',1),('meem_m',1),('noon_f',0)],
 'صديق':[('sad_i',1),('dal_f',0),('tooth_i',1),('qaf_f',0)],
 'كهف':[('kaf_i',1),('heh_i',1),('feh_f',0)],
 'عشب':[('ain_i',1),('sheen_m',1),('baa_f',0)],
 'نجم':[('tooth_i',1),('jeem_m',1),('meem_f',0)],
 'هلال':[('heh_i',1),('lam_m',1),('alef_f',0),('lam_s',0)],
 'ظل':[('zah_i',1),('lam_f',0)],
 'فلك':[('feh_i',1),('lam_m',1),('kaf_f',0)],
 'رقص':[('raa_s',0),('qaf_i',1),('sad_f',0)],
 'نبع':[('tooth_i',1),('tooth_m',1),('ain_f',0)],
 'خط':[('khah_i',1),('tah_f',0)],
}
# dots that depend on word (tooth letters)
DOTS={'تعكس':{0:d2a(-4.25,TI)},'صديق':{2:d2b(-20)},'موسيقى':{3:d2b(-20)},
      'نجم':{0:d1a(-4.25,TI)},'نبع':{0:d1a(-4.25,TI),1:d1b(-20)}}
_orig=dict(G)
def build(w):
    seq=WORDS[w];extra=DOTS.get(w,{})
    for i,(n,j,*_) in enumerate(seq):
        if i in extra:
            base=_orig[n];ex=extra[i]
            G[f'{n}#{w}{i}']=(lambda base=base,ex=ex:(lambda r:(r[0]+ex,r[1],r[2]))(base()))
            seq[i]=(f'{n}#{w}{i}',)+tuple(seq[i][1:])
    return word(seq)
