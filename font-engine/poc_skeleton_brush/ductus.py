#!/usr/bin/env python3
"""Ductus table: HOW each Arabic letter body is written.

Source: Mamoun Sakkal's Naskh chart (arrows, dots at stroke starts; notes 1
and 6: right to left, generally clockwise, strokes begin at the dots) and the
stroke-direction charts the user supplied.

  start    where the pen starts: "top" (the higher end) or "right"
  strokes  number of pen strokes in formal Naskh (the engine draws one
           continuous path; where Naskh lifts the pen this is noted)
  loop     the letter has a closed eye, written clockwise
  entry    which uploaded letter teaches how this letter's stroke BEGINS
           (noon = tooth entry, zain = head entry, meem = eye/loop entry)
  exit     which uploaded letter teaches how it ENDS
           (noon = rising bowl tip, zain = descending tail, None = plain end)
"""

DUCTUS = {
    "alef": dict(start="top", strokes=1, loop=False, entry="zain", exit=None),
    "beh": dict(start="right", strokes=1, loop=False, entry="noon", exit="noon"),
    "hah": dict(start="top", strokes=1, loop=False, entry="zain", exit="zain"),
    "dal": dict(start="top", strokes=1, loop=False, entry="zain", exit=None),
    "reh": dict(start="top", strokes=1, loop=False, entry="zain", exit="zain"),
    "seen": dict(start="right", strokes=3, loop=False, entry="noon", exit="noon"),
    "sad": dict(start="right", strokes=2, loop=True, entry="meem", exit="noon"),
    "tah": dict(start="top", strokes=2, loop=True, entry="zain", exit=None),
    "ain": dict(start="top", strokes=2, loop=False, entry="zain", exit="zain"),
    "feh": dict(start="right", strokes=1, loop=True, entry="meem", exit="noon"),
    "qaf": dict(start="right", strokes=1, loop=True, entry="meem", exit="noon"),
    "kaf": dict(start="top", strokes=2, loop=False, entry="zain", exit="noon"),
    "lam": dict(start="top", strokes=1, loop=False, entry="zain", exit="noon"),
    "meem": dict(start="top", strokes=1, loop=True, entry="meem", exit="zain"),
    "noon": dict(start="right", strokes=1, loop=False, entry="noon", exit="noon"),
    "heh": dict(start="top", strokes=1, loop=True, entry="meem", exit=None),
    "waw": dict(start="top", strokes=1, loop=True, entry="meem", exit="zain"),
    "yeh": dict(start="top", strokes=1, loop=False, entry="zain", exit="noon"),
    # forms of the uploaded word
    "zain": dict(start="top", strokes=1, loop=False, entry="zain", exit="zain"),
    "meem.init": dict(start="right", strokes=1, loop=True, entry="meem", exit="meem"),
}


def orient(P, letter):
    """Order the path so it starts where the pen starts for this letter."""
    mode = DUCTUS[letter]["start"]
    a, b = P[0], P[-1]
    if mode == "top":                      # image coordinates: smaller y is higher
        return P if a[1] <= b[1] else P[::-1]
    return P if a[0] >= b[0] else P[::-1]


def check(P, letter):
    """Does a path obey the table? (start position; clockwise eye if any)"""
    import numpy as np
    a, b = P[0], P[-1]
    mode = DUCTUS[letter]["start"]
    ok_start = bool(a[1] <= b[1]) if mode == "top" else bool(a[0] >= b[0])
    out = {"start_ok": ok_start}
    if DUCTUS[letter]["loop"]:
        # total signed turning: clockwise on screen (y down) is positive
        d = np.diff(P, axis=0)
        th = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        out["eye_clockwise"] = bool(th[-1] - th[0] > 0)
    return out
