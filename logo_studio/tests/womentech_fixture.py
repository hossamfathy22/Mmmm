"""The Women Tech session as a fixture: the logo picture, our crop, and the designer's marks with the kind of place he meant."""
import json, os
HERE = os.path.dirname(__file__)
IMG = os.path.join(HERE, 'data', 'womentech.jpg')
BOX = (223, 196, 511, 322)
CATS = {  # mark number (from his sheet) -> kind of place
    27: 'h_mid', 29: 'h_mid', 28: 'h_mid', 32: 'h_mid', 22: 'h_mid',
    17: 'h_end_hair', 25: 'h_end_hair', 21: 'h_end_hair', 30: 'h_end_hair', 18: 'h_end_hair', 26: 'h_end_hair',
    23: 'h_end_stem', 24: 'h_end_stem', 12: 'h_end_stem', 11: 'h_end_stem',
    1: 'v_top', 4: 'v_top', 0: 'head_top', 3: 'head_top', 9: 'head_top', 10: 'head_top',
    2: 'v_mid', 6: 'v_bot', 5: 'v_bot', 8: 'v_bot', 7: 'v_bot',
}
def marks(crop):
    """his marks were drawn in a frame whose origin is the picture point (223,196); move them into the project's frame"""
    raw = json.load(open(os.path.join(HERE, 'data', 'womentech_marks.json')))
    dx, dy = 223-crop[0], 196-crop[1]
    out = []
    for i, m in enumerate(raw):
        if i not in CATS:
            continue
        out.append(dict(id=str(i), cat=CATS[i], p0=[m['p0'][0]+dx, m['p0'][1]+dy], p1=[m['p1'][0]+dx, m['p1'][1]+dy]))
    return out
