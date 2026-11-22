"""The design rules that drive the letter engine. All lengths are in pixels of the source image."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict


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
    reh_corner: float = 0.0     # outer radius of the top corner of a letter hung from the tooth before it (0 = not seen)
    join_white: float = 0.0     # white between two joined letters (0 = white, as between the teeth of one letter)
    round_gap: float = 1.18     # extra white before a round head (optical spacing)
    # ---- vocabulary read from the skeleton (skeleton.py)
    angular: float = 0.0        # share of the turning done at corners: >= 0.5 draws tails and cups with corners
    corner_r: float = 0.0       # outer radius of those corners (0 = sharp)
    tail_drop: float = 1.0      # how far tails go below the baseline, fraction of the descender
    top_join: float = 0.0       # 1 = the logo joins its ر from the top of the tooth before it
    dash: float = 0.0           # length of the logo's own dash mark (two dots drawn as one line); 0 = none seen
    # ---- the designer's hand (recipes.py): a named parameter set here replaces its rule, and a letter
    # form named here is drawn in the chosen variant
    params: Dict[str, float] = field(default_factory=dict)
    variants: Dict[str, str] = field(default_factory=dict)
    # letters the logo draws its own way, cut out of it by the self-check: "letter.form" -> outlines
    taken: Dict[str, Dict] = field(default_factory=dict)
    # parts of letters as the logo draws them (selfcheck.take_parts): part name -> outlines, relative to
    # the right end of the engine's part; they stand in for that part in every letter the logo lacks
    parts: Dict[str, Dict] = field(default_factory=dict)
    # the logo's part library (library.Library.to_dict): its own stem, the feet of its teeth where they
    # flow into a join, its tail, head, dot and dash, that every letter the engine builds is made of
    library: Dict = field(default_factory=dict)
    # a part the logo lacks made from another of its parts, mirrored or turned: tried, and off (on 551
    # hidden letters it drew worse than the parts as they are, above all the cups of ل ن ي)
    derive_parts: object = False   # False, "engine" (alike in the engine's construction) or "real" (in real typefaces)
    # a part of the logo fitted to the letter it goes into: its centre line lengthened or shortened, made
    # taller or lower to the size of the engine's part there, and drawn again with its own widths.
    # Off: on 455 hidden letters the parts as the logo draws them matched 50.3%, fitted 47.9% (the
    # logo's own proportions are nearer the truth than the engine's part they would be fitted to)
    part_fit: bool = False
    # the logo's parts may change how a letter looks, not what it reads as (legible.py)
    part_legible: bool = True
    # a part moved from one letter to another takes the difference real typefaces make between the two
    # letters' parts (reference.part_ratio), its pen kept. Off: on 266 hidden letters (the typeface
    # playing the logo left out of the references) the parts as the logo draws them matched 53.5%,
    # fitted by real typefaces 50.8% (by the engine's own proportions 47.9%): a designer keeps a
    # part one size across its letters more than the typefaces' median says
    part_ref_fit: bool = False
    ref_skip: str = ""
    # a letter the logo lacks that real typefaces do not read as itself in the engine's construction is
    # built on a reference typeface's structure instead, with the logo's pen, when that one reads right.
    # Off: the readers judge by ordinary letters, and a logo's style is often not ordinary on purpose
    # (Mootanroo's ت is a tall stem, which the readers take for ا, and the fallback drew it short).
    # Hidden letters of the 16 logos the readers misread: 7 better with it, 10 worse, likeness to the
    # logo's own letter 0.398 -> 0.366. The designer chooses it, letter by letter ("من المرجع")
    legible_fallback: bool = False
    # every letter form the logo shows is taken from the logo, not only those the engine draws badly
    # (all but those it draws all but exactly): the pixels cannot tell a cup from two stems
    # (selfcheck._take_rest)
    take_all: bool = True
    # ا and ل drawn on the logo's ر when it draws ر as an upright stem (typeset.stem_from_reh). Off: on
    # the hidden ا of the other logos it fitted worse three times in four; the designer chooses it
    # ("على الراء"), for every ا and ل or letter by letter
    stem_kin: bool = False
    # س and ش built of the logo's own tooth where the logo hangs its teeth as cups sharing an arm
    # (typeset.teeth_from_logo); for any other tooth, the designer's choice ("أسنان الشعار")
    teeth_kin: bool = True          # a typeface family left out of the references (the one playing the logo in the examination)
    seen: str = ""              # the letter forms in the logo's word ("ت.I ن.M ..."): they keep their own drawing
    # the reference typeface the logo is nearest to (reference.py): letters the logo lacks are built on
    # its structure with the logo's pen; `ref_wide` is how much wider the logo's rhythm draws them
    reference: str = ""
    ref_wide: float = 1.0
    ref_all: bool = False       # every letter on the reference's structure (to judge a reference on the logo's word)
    # how a letter is taken from the reference: "pen", its centre line (the structure) drawn with the
    # logo's pen, or, where that breaks the letter or fails the critic, its outline at the logo's
    # weight; "outline", always the outline; "strokes", always the centre line
    ref_how: str = "pen"
    # a letter of few logos (ح ص ط ع ف ق ه and their siblings) the logo lacks, drawn from the reference
    # typeface instead of the engine's recipe. Off: the typefaces are a reference for a letter's
    # structure and an examiner, not what the letters are drawn from; a letter is built of the logo's
    # parts on its recipe (the method of the hand-made words). The user may choose it, letter by letter.
    ref_auto: bool = False
    # how far the logo draws a letter longer than the engine builds it (kashida): "letter.form" -> length.
    # A joining letter lengthens its join to the next; a final or isolated letter stretches its body
    # along the baseline; a letter with neither leaves more white after it. Fitted by the self-check.
    stretch: Dict[str, float] = field(default_factory=dict)
    # the rules of the letters as measured on this logo (selfcheck.fit_letters): a parameter's value and
    # a letter form's construction read off the logo's own word. They stand in for the rule (the
    # designer's `params` and `variants` still win), and serve every letter that shares them.
    fitted: Dict[str, float] = field(default_factory=dict)
    measured_on: str = ""       # the letter forms (by skeleton, "م.I ر.F") `fitted` was measured on: it serves only them
    # the designer's general manner, measured on all the logo's letters at once, applied to every
    # letter (also those the logo does not contain): trait -> factor on the rules of that kind
    traits: Dict[str, float] = field(default_factory=dict)
    chosen: Dict[str, str] = field(default_factory=dict)
    # how the letters are built: "parts" (letters.py: stems and hairlines of a contrast pen, round heads
    # and cups, the construction of بيدوناق) or "block" (blocks.py: every part one straight block of a
    # flat pen, cut at its ends, set by the designer's hand, the construction of كيوانو). The model
    # chooses it on seeing how the logo was built, and gives the construction's measures:
    construction: str = "parts"
    block: Dict[str, float] = field(default_factory=dict)       # blocks.KIVANO's rules, as this logo measures them
    # the logo's measured parts (blockfit.cut_parts): "letter.form" -> {"polys": its outline, "marks": its
    # dots}, in the letter's coordinates. They check the measures (blockfit.rebuild) and are never
    # written into a word: words are drawn from the rules (Style.block), as بيدوناق was from سیرنگ's
    block_parts: Dict = field(default_factory=dict)

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
