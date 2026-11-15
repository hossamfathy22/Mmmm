"""Reading a logo by drawing it.

The reader in understand.py looks at stems alone: it knows teeth, tails and tall strokes, and a round
head, a bowl or a nose is a stranger to it. This reader asks a different question: which letters,
drawn by the engine in the style just measured from this logo, cover this piece of writing best?

Every piece of joined writing (a block: the letters between two breaks) is read on its own, from
its right end, by a beam search: a few likely beginnings are kept, each is extended by every letter
skeleton, drawn, and compared with the logo over the stretch it covers; the best complete readings
are compared with the whole block. Dots and dashes are then read off the logo to tell ب from ت,
ح from ج, and so on.

It is slower than the stem reader (seconds, not milliseconds), and is used only when that one
leaves letters unread and the designer has not typed the logo's word.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from .letters import LETTERS
from .measure import Line
from .style import Style
from .typeset import flat, typeset

# one letter per skeleton (dots come later); ق and ن are kept apart from ف and ب for their final forms
ALPHABET = "بحدرسصطعفقكلمنهوىا"
DOTS = {   # skeleton -> {(marks above, marks below): letter}
    "ب": {(0, 0): "ٮ", (0, 1): "ب", (2, 0): "ت", (3, 0): "ث", (0, 3): "پ", (1, 0): "ن", (0, 2): "ي"},
    "ح": {(0, 0): "ح", (0, 1): "ج", (1, 0): "خ", (0, 3): "چ"},
    "د": {(0, 0): "د", (1, 0): "ذ"}, "ر": {(0, 0): "ر", (1, 0): "ز", (3, 0): "ژ"},
    "س": {(0, 0): "س", (3, 0): "ش"}, "ص": {(0, 0): "ص", (1, 0): "ض"}, "ط": {(0, 0): "ط", (1, 0): "ظ"},
    "ع": {(0, 0): "ع", (1, 0): "غ"}, "ف": {(0, 0): "ف", (1, 0): "ف", (2, 0): "ق", (3, 0): "ڤ"},
    "ق": {(0, 0): "ق", (2, 0): "ق", (1, 0): "ف"}, "ن": {(0, 0): "ن", (1, 0): "ن"},
    "ى": {(0, 0): "ی", (0, 2): "ي"}, "ه": {(0, 0): "ه", (2, 0): "ة"},
}


@dataclass
class Block:
    mask: np.ndarray        # the block's ink (marks left out), on a reduced copy of the line
    x0: int                 # its left and right columns on that copy
    x1: int


def _blocks(L: Line, f: float) -> Tuple[List[Block], np.ndarray, np.ndarray]:
    """The pieces of joined writing of the line, right to left, at scale f (and the marks)."""
    body_ids = [p.comp for p in L.pieces if p.kind == "body"]
    mark_ids = [p.comp for p in L.pieces if p.kind in ("dot", "dash")]
    body = np.isin(L.lab, body_ids)
    marks = np.isin(L.lab, mark_ids)
    small = lambda m: np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(
        (max(1, int(m.shape[1] * f)), max(1, int(m.shape[0] * f))), Image.BILINEAR)) > 100
    b, mk = small(body), small(marks)
    lab, n = ndi.label(b, structure=np.ones((3, 3)))
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        m = lab == i
        if m.sum() < 6:
            continue
        cols = np.flatnonzero(m.any(axis=0))
        out.append(Block(m, int(cols[0]), int(cols[-1])))
    # a block that sits within another's columns (a tail's tip, a stray piece) joins it
    out.sort(key=lambda bl: -(bl.x1 - bl.x0))
    merged: List[Block] = []
    for bl in out:
        host = next((h for h in merged if h.x0 - 2 <= bl.x0 and bl.x1 <= h.x1 + 2), None)
        if host is not None:
            host.mask |= bl.mask
        else:
            merged.append(bl)
    merged.sort(key=lambda bl: -bl.x1)
    return merged, b, mk


class _Drawer:
    """Draws letter sequences in the logo's style onto the reduced canvas, with a cache."""

    def __init__(self, st: Style, s: float, H: int, base_row: float):
        self.st, self.s, self.H, self.base = st, s, H, base_row
        self.cache: Dict[str, Tuple[np.ndarray, float, float]] = {}

    def __call__(self, text: str) -> Tuple[np.ndarray, float, float, List[Tuple[str, float, float]]]:
        """Mask of the letters of `text` (a trailing ـ only forces the joined form and is not drawn),
        right edge at column 0 of a canvas wide enough; returns (mask, left, right, spans)."""
        if text in self.cache:
            return self.cache[text]
        lay = typeset(text, self.st)
        letters = [p for p in lay.placed if p.char != "ـ"]
        pts = [(x, y) for p in letters for sh in flat(p.glyph.shapes) for poly in sh for x, y in [(q[0] + p.x, q[1]) for q in poly]]
        if not pts:
            r = (np.zeros((self.H, 1), bool), 0.0, 0.0, [])
            self.cache[text] = r
            return r
        right = max(x for x, _ in pts)
        left = min(x for x, _ in pts)
        W = int((right - left) * self.s) + 4
        im = Image.new("L", (W, self.H), 0)
        d = ImageDraw.Draw(im)
        spans = []
        for p in letters:
            xs = []
            for sh in flat(p.glyph.shapes):
                if getattr(sh, "name", "") in ("نقطة", "شرطة", "همزة"):
                    continue
                part = Image.new("L", (W, self.H), 0)
                pd = ImageDraw.Draw(part)
                for poly in sh:
                    if len(poly) >= 3:
                        q = [((x + p.x - left) * self.s + 2, self.base - y * self.s) for x, y in poly]
                        pd.polygon(q, fill=255)
                        xs += [x + p.x for x, _ in poly]
                im.paste(255, mask=part)
            if xs:
                spans.append((p.char, min(xs), max(xs)))
        m = np.asarray(im) > 0
        r = (m, (left - right) * self.s, 0.0, spans)
        self.cache[text] = r
        return r


def _iou_at(block: Block, m: np.ndarray, x_right: int, lo: Optional[int] = None) -> float:
    """IoU of the drawing m (right edge at column x_right of the block's canvas) with the block,
    over the columns from `lo` (or the drawing's left end) to the block's right end."""
    H, W = block.mask.shape
    w = m.shape[1]
    c0 = x_right - w + 1
    canvas = np.zeros_like(block.mask)
    a0, a1 = max(0, c0), min(W, c0 + w)
    if a1 <= a0:
        return 0.0
    canvas[:, a0:a1] = m[:, a0 - c0:a1 - c0]
    left = max(0, c0 if lo is None else lo)
    win = slice(left, block.x1 + 1)
    A, B = canvas[:, win], block.mask[:, win]
    u = float((A | B).sum())
    return float((A & B).sum()) / u if u else 0.0


def _best_iou(block: Block, m: np.ndarray, reach: int, window: bool) -> float:
    best = 0.0
    for dx in range(-reach, reach + 1, max(1, reach // 4)):
        x_right = block.x1 + dx
        lo = x_right - m.shape[1] + 1 if window else block.x0
        best = max(best, _iou_at(block, m, x_right, lo))
    return best


def read_by_drawing(L: Line, st: Style, k: float, beam: int = 8, max_letters: int = 7) -> Optional[Dict]:
    """Read the line by drawing: returns {"text", "blocks": [{"text", "match"}], "match"} or None."""
    if not L.stems:
        return None
    up = Style(**{**st.__dict__, "slant": 0.0, "taken": {}})
    s_full = 1.0 / k
    f = min(1.0, 9.0 / max(st.stem * s_full, 1e-6))          # work where a stem is about 9 pixels wide
    s = s_full * f
    blocks, body, marks = _blocks(L, f)
    if not blocks:
        return None
    H = body.shape[0]
    base = L.row_of_y(0.0) * f
    draw = _Drawer(up, s, H, base)
    reach = max(2, int(0.25 * st.pitch * s))
    out_blocks, texts = [], []
    for bl in blocks:
        width = bl.x1 - bl.x0 + 1
        frontier = [("", 1.0)]
        done: List[Tuple[float, str]] = []
        for _ in range(max_letters):
            grown = []
            for seq, _ in frontier:
                if seq and LETTERS[seq[-1]][0] != "D":
                    continue
                for c in ALPHABET:
                    cand = seq + c
                    full, lft, _, _ = draw(cand)
                    if full.shape[1] <= width * 1.35 + 2 * reach:
                        done.append((_best_iou(bl, full, reach, window=False), cand))
                    if LETTERS[c][0] == "D":
                        part, _, _, spans = draw(cand + "ـ")
                        if part.shape[1] <= width * 1.2 + 2 * reach:
                            grown.append((cand, _best_iou(bl, part, reach, window=True)))
            if not grown:
                break
            grown.sort(key=lambda t: -t[1])
            # letters that share one body when joined (ب ن ى, ف ق) draw the same beginning: keep one
            frontier, seen = [], set()
            for cand, score in grown:
                m = draw(cand + "ـ")[0]
                key = (m.shape, hash(m.tobytes()))
                if key in seen:
                    continue
                seen.add(key)
                frontier.append((cand, score))
                if len(frontier) == beam:
                    break
        if not done:
            continue
        # equal drawings score equally; of those, the reading with the common letter wins (ي over ى)
        done.sort(key=lambda t: (-round(t[0], 4), sum(ch in "ىن" for ch in t[1][:-1])))
        score, text = done[0]
        out_blocks.append({"text": text, "match": round(score, 3), "x": (bl.x0, bl.x1)})
        texts.append(text)
    if not texts:
        return None
    text = _with_dots(out_blocks, draw, marks, base, s, L, gap=1.6 * st.pitch * s)
    total = float(np.mean([b["match"] for b in out_blocks]))
    return {"text": text, "skeleton": " ".join(texts), "blocks": out_blocks, "match": round(total, 3)}


def _with_dots(blocks: List[Dict], draw: _Drawer, marks: np.ndarray, base: float, s: float, L: Line, gap: float = 1e9) -> str:
    """Read each letter's dots off the logo: the marks over (or under) the stretch of the block the
    letter covers. A dash counts as two dots."""
    lab, n = ndi.label(marks, structure=np.ones((3, 3)))
    pieces = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        m = lab == i
        rows, cols = np.nonzero(m)
        h, w = rows.max() - rows.min() + 1, cols.max() - cols.min() + 1
        if w > 3.5 * h and w > 2.5 * L.stem * s:                 # a long bar over a letter (گ): not dots
            continue
        pieces.append((float(cols.mean()), float(rows.mean()), 2 if w > 2.2 * h else 1))
    words = []
    for b in blocks:
        _, left, _, spans = draw(b["text"])
        right_col = b["x"][1]
        letters = []
        for (c, x0, x1), ch in zip(spans, b["text"]):
            right = max(sp[2] for sp in spans)
            a = right_col - (right - x0) * s
            z = right_col - (right - x1) * s
            up = sum(k for x, y, k in pieces if a - 1 <= x <= z + 1 and y < base - 0.55 * L.x_height * s)
            down = sum(k for x, y, k in pieces if a - 1 <= x <= z + 1 and y > base)
            table = DOTS.get(ch, {})
            letters.append(table.get((min(up, 3), min(down, 3)), table.get((0, 0), ch)))
        words.append("".join(letters))
    # blocks are parts of one word unless the white between them is as wide as a word's gap
    out = words[0] if words else ""
    for prev, b, w in zip(blocks, blocks[1:], words[1:]):
        out += (" " if prev["x"][0] - b["x"][1] > gap else "") + w
    return out
