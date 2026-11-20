"""A logo's picture made a vector drawing, so that it is measured as a drawing (vector.dna), not as
pixels (measure.py). The edge of the ink is followed to a fraction of a pixel (skimage's marching
squares on the grey picture), and each outline is made a designer's path: corners kept, curves fitted as
cubic Beziers, straight runs as lines (vparts.ring_to_segments). Where the typed word was found in the
picture (match.locate: which letter owns each pixel), every letter is traced on its own, as a typeface
gives its glyphs, so that the word says which shape is a tooth and which a tall letter."""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .vector import VShape


def _rings(grey: np.ndarray, level: float = 0.5) -> List[np.ndarray]:
    from skimage.measure import find_contours
    pad = np.pad(grey, 2)
    out = []
    for c in find_contours(pad, level):
        if len(c) < 6:
            continue
        rows, cols = c[:, 0] - 2, c[:, 1] - 2
        out.append(np.c_[cols, -rows])                   # x right, y up
    return out


def trace(grey: np.ndarray, owner: Optional[np.ndarray] = None, err: float = 0.35, smooth: float = 0.6) -> List[VShape]:
    """grey: ink intensity 0..1 (1 = ink), rows down. owner: the index in the typed text of the letter
    each pixel belongs to (+1; 0 none), as match.locate gives it. Returns shapes in picture pixels, y up."""
    from scipy import ndimage as ndi
    from .vparts import ring_to_segments
    g = ndi.gaussian_filter(np.asarray(grey, float), smooth) if smooth else np.asarray(grey, float)
    groups = [(None, g)]
    if owner is not None:
        ink = g >= 0.5
        groups = []
        for i in sorted(set(np.unique(owner[ink]).tolist()) - {0}):
            m = ndi.binary_dilation(owner == i, iterations=1)
            groups.append((int(i) - 1, np.where(m, g, 0.0)))
    shapes = []
    for idx, gg in groups:
        contours = []
        for r in _rings(gg):
            segs = ring_to_segments(r, err)
            if segs:
                contours.append(segs)
        if contours:
            shapes.append(VShape(contours, f"traced-{idx}", idx))
    return shapes


def measure(grey: np.ndarray, owner: Optional[np.ndarray], text: str) -> dict:
    """The style of a picture read as a drawing: the x-height and the ascender from its letters traced one
    by one (the word says which is a tooth, which a tall letter); the stem, the slant and the hairline from
    the word traced whole (a letter cut from its neighbour has a cut where the join was, not a join)."""
    from .vector import dna
    whole = dna(trace(grey), text)
    if owner is None:
        return whole
    each = dna(trace(grey, owner), text)
    out = dict(whole)
    # the letters' tops as heights (each tracing finds its own baseline: the cut letters rest elsewhere),
    # measured from the baseline of the word traced whole
    if "x_height" in each and "baseline" in each and "baseline" in whole:
        top = each["baseline"] + each["x_height"]
        out["x_height"] = top - whole["baseline"]
        out["x_from"] = each.get("x_from")
        if "ascender" in each:
            out["ascender"] = top + each["ascender"] - whole["baseline"] - out["x_height"]
    return out


def picture_to_svg(image: bytes) -> str:
    """A logo's picture as an SVG drawing (every outline of its ink, curves and lines), as a designer's
    file would hold it."""
    from .guard import prepare
    from .vector import to_svg
    work, _, _, inverted, _ = prepare(image)
    g = np.asarray(work, float) / 255.0
    return to_svg(trace(g if inverted else 1.0 - g), pad=10.0)


def measure_svg(svg, text: str, px: float = 900.0) -> dict:
    """A logo's SVG measured as a drawing, in the SVG's own units. The drawing is shown to the word finder
    (match.locate) as a picture, `px` pixels tall, to learn which letter each part of it draws; the word's
    letters are then traced and read (measure), and its pen measured on the word's own ink
    (qalam.fit_nib: the width, thickness and angle of the nib that drew it)."""
    import io
    from PIL import Image
    from scipy import ndimage as ndi
    from .bench import fonts
    from .match import locate
    from .pipeline import MATCH_FACES
    from .qalam import fit_nib
    from .vector import raster, read_svg
    shapes = read_svg(svg)
    pts = np.vstack([s.points() for s in shapes if len(s.points())])
    scale = px / float(np.ptp(pts[:, 1]) or 1.0)
    m, _ = raster(shapes, scale, pad=30)
    buf = io.BytesIO(); Image.fromarray(np.where(m, 0, 255).astype(np.uint8)).save(buf, "PNG")
    files = {f.stem: f for f in fonts.font_files()}
    faces = [fonts.load(str(files[n])) for n in MATCH_FACES if n in files]
    got = locate(buf.getvalue(), text, faces)
    out = {"scale": scale, "found": got is not None}
    grey = m.astype(float)
    owner = None
    if got is not None and np.asarray(got["owner"]).shape == m.shape:
        word = np.asarray(got["line"], bool)
        near = ndi.binary_dilation(word, iterations=2)
        grey = np.where(near, grey, 0.0)
        owner = np.where(word, np.asarray(got["owner"]), 0)
        out["word_mask"] = word
    d = measure(grey, owner, text)
    for k in ("x_height", "ascender", "descender", "stem", "hair", "hair_scan"):
        if d.get(k) is not None:
            out[k] = d[k] / scale
    out["slant"] = d.get("slant")
    out["baseline_px"] = d.get("baseline")
    body = (grey >= 0.5)
    lab, n = ndi.label(body, structure=np.ones((3, 3)))
    if n:
        sizes = ndi.sum(body, lab, range(1, n + 1))
        try:
            nb = fit_nib(np.isin(lab, 1 + np.flatnonzero(sizes > 0.05 * sizes.max())), 1)
            out["nib"] = {"width": nb["width"] / scale, "thick": nb["thick"] / scale, "angle": nb["angle"]}
            # the stem and the hairline are the pen's: its width and its thickness. Read off straight upright
            # edges and the tops of the joins (vector.dna) they hold only for a hand that draws uprights:
            # on Sirang the pen gave 22.9 and 5.0 (hadar_word measured 23 and 4.5), the edges and joins
            # 23.8 and 11.6; on Rozita's brush the edges gave a stem thinner than its hairline.
            out["stem_edges"], out["hair_joins"] = out.get("stem"), out.get("hair")
            out["stem"], out["hair"] = out["nib"]["width"], out["nib"]["thick"]
        except Exception as e:
            out["nib_error"] = str(e)[:80]
    out["mask"] = m
    return out
