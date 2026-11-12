"""The design rules that drive the letter engine. All lengths are in pixels of the source image."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass
class Style:
    stem: float = 23.0          # thick pen: width of every vertical stem (U)
    hair: float = 4.5           # thin pen: horizontal joins, flags, dashes
    x_height: float = 66.0      # top of teeth and heads
    ascender: float = 36.0      # alef and marks rise this far above the x-height
    descender: float = 36.0     # tails and bowls go this far below the baseline
    slant: float = 14.0         # degrees, positive leans right
    white: float = 22.0         # white between two teeth (counter width)
    dot: float = 25.0           # diameter of a dot
    dash_marks: bool = True     # two dots become one hairline dash
    contrast_power: float = 2.0 # how fast a curve thins from stem to hairline
    # ---- shape parameters (measured from the logo by features.py; defaults = the hand-built بيدوناق)
    top_angle: float = 0.0      # degrees: how the top of a stem is cut
    top_round: float = 0.0      # 0 = square corners, 1 = half circle
    join_in: float = 0.455      # where the inner curve of a join starts, fraction of the x-height
    join_out: float = 0.455     # where the outer curve of a join starts, fraction of the x-height
    ring_n: float = 2.3         # squareness of heads (2 = ellipse)
    dot_n: float = 2.0          # 2 = round dot, larger = square
    dot_rot: float = 0.0        # 45 = diamond
    tail_reach: float = 3.05    # how far a tail sweeps left, in stems
    overshoot: float = 0.0      # round bottoms and tops pass the lines by this much
    seen_drop: float = 0.0      # inner teeth of س ش are this much lower
    group_gap: float = 0.0      # white between letters that do not join (0 = 1.15 x white)
    round_gap: float = 1.18     # extra white before a round head (optical spacing)

    @property
    def pitch(self) -> float:
        return self.stem + self.white

    @property
    def fillet(self) -> float:
        return max(1.0, min(self.white / 2.0, 0.6 * self.stem))

    @property
    def tan(self) -> float:
        return math.tan(math.radians(self.slant))

    def width_at(self, ty: float) -> float:
        """Pen rule: vertical strokes get the stem, horizontal ones the hairline."""
        return self.hair + (self.stem - self.hair) * abs(ty) ** self.contrast_power

    def to_dict(self) -> dict:
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in asdict(self).items()}

    @classmethod
    def from_line(cls, line: dict) -> "Style":
        """Turn one measured line (see analyze.py) into generation rules."""
        stem = line["stem"]
        xh = line["x_height"]
        hair = stem if line.get("monoline") else line["hairline"]
        white = line.get("white_between_stems") or stem
        if not (0.4 * stem <= white <= 2.5 * stem):
            white = stem
        dots = [d["diameter"] for d in line.get("dots", [])]
        dot = sorted(dots)[len(dots) // 2] if dots else 1.1 * stem
        asc = line.get("ascender") or 0.0
        desc = line.get("descender") or 0.0
        return cls(
            stem=stem,
            hair=max(1.0, min(hair, stem)),
            x_height=xh,
            ascender=asc if asc >= 0.3 * xh else 0.55 * xh,
            # bowls and dots under the baseline need room: never less than 45 % of the x-height
            descender=max(desc, 0.45 * xh),
            slant=line.get("slant_deg", 0.0),
            white=white,
            dot=dot,
            dash_marks=bool(line.get("dashes")) or not dots,
        )
