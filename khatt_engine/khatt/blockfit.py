"""Measuring a logo built of blocks: the engine's block construction fitted to the logo's own word.

This is the model's measuring step for a logo whose designer built every letter of straight blocks
(blocks.py). The model has looked at the logo and set the construction; this code finds its numbers
on the logo itself, as the stems and whites of سیرنگ were found by scanning its rows and columns:

1. the rules (`Style.block`): the pen, its wedge, the x-height, the head and its cuts, the tail, the
   dots, the kaf, the whites, the softened corners; one value each, for every letter;
2. the hand (`Style.block_hand`): for each stroke of the logo's word, how the designer set it off its
   rule (tilt, end cuts, length, weight, wedge, shift).

`match` is the proof: the logo's word set by the engine from these numbers alone, with nothing cut
from the logo, laid over the logo (the share of ink they have in common, IoU). كيوانو: the rules
alone 84%, with its measured hand 95.2% (README, "طريقة البناء بالكتل").
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .style import Style

# rules that are fractions (searched in small steps) and rules the fit leaves alone (spacing is
# measured by the letters' places, and the spread of the hand is read off the fitted hand)
FRACTIONS = ("k", "kv", "h", "jh", "kaf_bar", "kaf_riser", "teeth", "teeth_yeh")
FIXED = ("hand", "hand_tilt", "hand_cut", "hand_len", "hand_weight", "space", "gap", "join_white", "into")
STEP = [2.0, 4.0, 4.0, 3.0, 2.0, 0.06, 2.0, 2.0]   # tilt, cut, cut, length, weight, wedge, shift x, shift y


def raster(lay, size: Tuple[int, int], origin: Tuple[float, float], ss: int = 4) -> np.ndarray:
    """The layout as a mask of the logo's pixels: the word's right edge at origin[0], baseline at origin[1]."""
    from PIL import Image, ImageDraw
    W, H = size
    ox, base = origin
    im = Image.new("L", (W * ss, H * ss), 0)
    d = ImageDraw.Draw(im)
    for shape in lay.shapes:
        d.polygon([((ox + x) * ss, (base - y) * ss) for x, y in shape[0]], fill=255)
        for hole in shape[1:]:
            d.polygon([((ox + x) * ss, (base - y) * ss) for x, y in hole], fill=0)
    return np.array(im.resize((W, H), Image.BOX)) >= 128


def match(word: str, st: Style, ink: np.ndarray, origin: Tuple[float, float]) -> float:
    """How much of its ink the engine's drawing of `word` shares with the logo's (IoU, 0..1)."""
    from .typeset import typeset
    got = raster(typeset(word, st), (ink.shape[1], ink.shape[0]), origin)
    union = (got | ink).sum()
    return float((got & ink).sum() / union) if union else 0.0


def _hand_keys(hand: Dict[str, List[float]], letter: int):
    out = []
    for name, v in hand.items():
        if name.split(".")[0] != str(letter):
            continue
        n = 5 if name.endswith(".head") else 3 if name.endswith(".slot") else 1 if name.endswith((".dot", ".hamza")) else 8
        out += [(name, i) for i in range(min(n, len(v)))]
    return out


def fit(word: str, st: Style, ink: np.ndarray, origin: Tuple[float, float], cycles: int = 2,
        log: Callable[[str], None] = print) -> Style:
    """Fit the rules and the hand of the block construction to the logo's ink (a boolean mask), the
    logo's `word` standing with its right edge at origin[0] and its baseline at origin[1]. Each letter's
    hand is searched with its strokes together (they pull on each other), then each rule alone."""
    from scipy.optimize import minimize
    from .blocks import rules
    from .typeset import shape_forms

    block = dict(rules(st))
    hand = {k: list(v) for k, v in ((st.block_hand or {}).get("hand") or {}).items()} if (st.block_hand or {}).get("word") == word else {}
    cur = replace(st, construction="block", block=block, block_hand={"word": word, "hand": hand})
    # the strokes of the word, named by a first drawing (the hand starts at zero where not measured)
    from .typeset import typeset
    typeset(word, cur)

    def score() -> float:
        return match(word, cur, ink, origin)

    def search(get, put, x0, steps, iters):
        def f(x):
            put(x)
            return -score()
        f0 = f(x0)
        simplex = [x0] + [x0 + np.eye(len(x0))[j] * steps[j] for j in range(len(x0))]
        res = minimize(f, x0, method="Nelder-Mead",
                       options=dict(initial_simplex=np.array(simplex), maxiter=iters, xatol=0.05, fatol=1e-6))
        put(res.x if res.fun <= f0 else x0)

    log(f"start {100 * score():.2f}%")
    letters = sum(1 for _, f in shape_forms(word) if f not in (" ", "?"))
    for cycle in range(cycles):
        for i in range(letters):
            keys = _hand_keys(hand, i)
            if not keys:
                continue

            def put(x, keys=keys):
                for (n, j), v in zip(keys, x):
                    hand[n][j] = float(v)
            x0 = np.array([hand[n][j] for n, j in keys])
            steps = np.array([3.0 if n.endswith(".head") else STEP[j] for n, j in keys])
            search(None, put, x0, steps, 2500)
        log(f"cycle {cycle}: hand {100 * score():.2f}%")
        for k in [k for k in block if k not in FIXED]:
            def put(x, k=k):
                block[k] = float(x[0])
            search(None, put, np.array([block[k]]), np.array([0.03 if k in FRACTIONS else 0.5 if k == "round" else 1.5]), 40)
        log(f"cycle {cycle}: rules {100 * score():.2f}%")
    return cur


def spread(hand: Dict[str, List[float]]) -> Dict[str, float]:
    """The spread of a measured hand, as the block rules give it for words the logo does not show:
    twice the median of each measure over the strokes (a value drawn evenly from +- that has the
    same mean size)."""
    rows = np.array([v[:5] for n, v in hand.items() if len(v) >= 5 and not n.endswith((".head", ".slot", ".dot"))])
    if not len(rows):
        return {}
    med = np.median(np.abs(rows), axis=0)
    return dict(hand_tilt=float(2 * med[0]), hand_cut=float(2 * np.median(med[1:3])),
                hand_len=float(2 * med[3]), hand_weight=float(2 * med[4]))
