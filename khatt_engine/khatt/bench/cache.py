"""The typefaces' letters and the styles the engine measured from them, kept on disk, so that a
change to one letter can be examined in seconds instead of minutes.

`build()` renders every letter of every cached typeface once and measures each typeface's logo word;
`Bench()` loads them and `Bench.score(letters)` reads the engine's letters against them.
"""
from __future__ import annotations

import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..letters import LETTERS
from ..style import Style
from . import fonts
from .score import CANVAS, Shape, distance, read

FILE = fonts.CACHE / "bench_cache.npz"


def build(log=print) -> Path:
    from .__main__ import LOGO, X_PX, _numbers, font_shapes
    from .truth import word_image
    warnings.filterwarnings("ignore")
    names, styles, conf, keys, masks = [], {}, {}, [], []
    for f in fonts.font_files():
        face = fonts.load(str(f))
        if any(ord(c) not in face.tt.getBestCmap() for c in LOGO):
            continue
        m = _numbers(word_image(face, LOGO, X_PX)[0])
        if m is None:
            continue
        shapes = font_shapes(face)
        names.append(face.label)
        styles[face.label] = {k: (float(v) if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) else v)
                              for k, v in m["style"].__dict__.items()}
        conf[face.label] = m["confidence"]
        for (c, form), s in shapes.items():
            keys.append(f"{face.label}|{c}|{form}"); masks.append(np.packbits(s.mask))
        log(f"  {face.label}: {len(shapes)}")
    FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FILE, keys=np.array(keys), masks=np.array(masks),
                        meta=json.dumps({"names": names, "styles": styles, "confidence": conf, "canvas": CANVAS}, ensure_ascii=False))
    return FILE


class Bench:
    """The cached ground truth, and the engine's letters scored against it."""

    def __init__(self, only_sure: bool = False):
        if not FILE.exists():
            build()
        z = np.load(FILE, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        self.names: List[str] = [n for n in meta["names"] if not only_sure or meta["confidence"][n] >= 0.8]
        self.confidence: Dict[str, float] = meta["confidence"]
        self.styles = {n: Style(**{k: v for k, v in meta["styles"][n].items() if k in Style.__dataclass_fields__}) for n in self.names}
        n_bits = CANVAS[0] * CANVAS[1]
        self.masks: Dict[str, Dict[Tuple[str, str], np.ndarray]] = {n: {} for n in self.names}
        for key, packed in zip(z["keys"], z["masks"]):
            name, c, form = str(key).split("|")
            if name in self.masks:
                self.masks[name][(c, form)] = np.unpackbits(packed)[:n_bits].reshape(CANVAS).astype(bool)
        self._shapes: Dict[str, Dict[Tuple[str, str], Shape]] = {}

    def truth(self, name: str) -> Dict[Tuple[str, str], Shape]:
        if name not in self._shapes:
            self._shapes[name] = {k: Shape(m) for k, m in self.masks[name].items()}
        return self._shapes[name]

    def consensus(self, c: str, form: str, names: Optional[Iterable[str]] = None) -> np.ndarray:
        """How often each point of the canvas is ink, over the typefaces: the letter all of them agree on."""
        ms = [self.masks[n][(c, form)] for n in (names or self.names) if (c, form) in self.masks[n]]
        return np.mean(ms, axis=0) if ms else np.zeros(CANVAS)

    def score(self, letters: Optional[str] = None, forms: str = "iIMF", names: Optional[Iterable[str]] = None,
              overrides: Optional[Dict] = None, variants: Optional[Dict] = None) -> Dict[Tuple[str, str], Dict]:
        """Read the engine's letters against every typeface. Returns, per (letter, form): the share
        read right, the median distance to the typeface's own letter, and what it is most often
        taken for."""
        from .__main__ import SKELETONS, _same, engine_shape
        want = [c for c in SKELETONS if letters is None or c in letters]
        out: Dict[Tuple[str, str], Dict] = {}
        acc: Dict[Tuple[str, str], List] = {}
        for name in (names or self.names):
            st = self.styles[name]
            if overrides or variants:
                st = Style(**st.__dict__)
                st.params = {**st.params, **(overrides or {})}
                st.variants = {**st.variants, **(variants or {})}
            T = self.truth(name)
            for c in want:
                for f in forms:
                    if (c, f) not in T or (f in "IM" and LETTERS[c][0] != "D"):
                        continue
                    e = engine_shape(c, f, st)
                    if e is None:
                        continue
                    refs = {cc: s for (cc, ff), s in T.items() if ff == f}
                    got, _, _ = read(e, refs)
                    acc.setdefault((c, f), []).append((_same(got, c, f), distance(e, T[(c, f)]), got))
        for k, v in acc.items():
            wrong = Counter(g for ok, _, g in v if not ok)
            out[k] = {"read": float(np.mean([ok for ok, _, _ in v])), "distance": float(np.median([d for _, d, _ in v])),
                      "n": len(v), "reads_as": wrong.most_common(1)[0][0] if wrong else ""}
        return out


def summary(res: Dict[Tuple[str, str], Dict]) -> str:
    tot = sum(r["read"] * r["n"] for r in res.values()); n = sum(r["n"] for r in res.values())
    lines = [f"{c} {f}: {r['read'] * 100:3.0f}%  d={r['distance']:.3f}  {('→' + r['reads_as']) if r['reads_as'] else ''}"
             for (c, f), r in sorted(res.items(), key=lambda kv: kv[1]["read"])]
    return "\n".join(lines + [f"المجموع: {100 * tot / max(n, 1):.1f}% على {n} حالة"])


if __name__ == "__main__":
    print(build())
