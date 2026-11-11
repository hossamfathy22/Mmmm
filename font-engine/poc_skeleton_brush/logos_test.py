#!/usr/bin/env python3
"""Reference logos (Latin + Arabic pairs) through the engine: does pen path +
art brush rebuild lettering it has never seen, in any style?"""
import json, sys, time
import numpy as np
from PIL import Image
import art_brush as ab
import skeleton_brush as sb

def main():
    out = {}
    for p in sorted((sb.HERE / "input" / "logos").glob("*.png")):
        w, h = Image.open(p).size
        t = time.time()
        try:
            r = ab.rebuild(p, (0, 0, w, h), "logo_" + p.stem)
            r["seconds"] = round(time.time() - t)
        except Exception as ex:
            r = {"error": f"{type(ex).__name__}: {ex}"}
        out[p.stem] = r
        print(p.stem, r, flush=True)
        (sb.OUT / "logos_test.json").write_text(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
