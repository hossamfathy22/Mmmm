# Open-source toolchain and minimum OpenType engineering for an AI-generated Arabic font (as of Nov 2026)

> Research method note: this session's network allowed PyPI, crates.io, raw.githubusercontent.com and web search. It blocked fonttools.readthedocs.io, learn.microsoft.com, openfontlicense.org, googlefonts.github.io and unicode.org. So version and license data comes from PyPI or crates.io JSON (fetched 2026-11-08), and code facts come from the GitHub sources. Some Microsoft, Google Fonts guide and OFL FAQ statements reached me only through search-engine summaries, and each is flagged where it appears.
> **A working end-to-end prototype was built and tested locally** with fontTools 4.68.0, uharfbuzz 0.57.0 and ufo2ft 3.11.0. It compiled an Arabic TTF with init/medi/fina/rlig/curs/mark, shaped it with HarfBuzz, checked joins automatically and exported WOFF2. Its results are cited as "[Local test]".

---

## Q1. Which Python libraries to use, with links, licenses and maturity, and which is best for fully automated generation

### Takeaway
Use **fontTools (MIT)** as the core. The quickest route is to emit the glyph outlines straight into `fontTools.fontBuilder.FontBuilder` and compile a generated `.fea` string with `feaLib`. The more maintainable route is to write a **UFO with ufoLib2 (Apache-2.0)** and compile it with **ufo2ft (MIT)**. ufo2ft then *auto-generates* GDEF, `curs`, `mark`/`mkmk` and `kern` from anchors and kerning, so the AI only has to emit outlines, anchors and a short GSUB `.fea`. fontmake and glyphsLib only matter if Glyphs/designspace sources are involved. FontForge is unnecessary and adds GPL plus a C dependency.

### Cited Findings
- **fontTools 4.68.0**, released 2026-10-27, MIT, requires Python >=3.11. Recent cadence: 4.67.0 (2026-10-14), 4.66.1 (2026-09-29), 4.65.0 (2026-09-10). — [PyPI fonttools](https://pypi.org/project/fonttools/) · repo [github.com/fonttools/fonttools](https://github.com/fonttools/fonttools)
- `FontBuilder` public API: `setupGlyphOrder`, `setupCharacterMap(cmapping, uvs=None, allowFallback=False)`, `setupGlyf(glyphs, calcGlyphBounds=True)`, `setupCFF(psName, fontInfo, charStringsDict, privateDict)`, `setupCFF2`, `setupHorizontalMetrics(metrics)`, `setupHorizontalHeader(**values)`, `setupNameTable(nameStrings, windows=True, mac=True)`, `setupOS2(**values)`, `setupPost(keepGlyphNames=True, **values)`, `setupFvar/Avar/Gvar/Stat` (variable fonts), `addOpenTypeFeatures(features, filename=None, tables=None, debug=False)`, `addFeatureVariations`, `setupCOLR/CPAL`, `setupDummyDSIG`. The docstring has full TTF (`TTGlyphPen`, quadratic) and OTF (`T2CharStringPen`, cubic CFF) examples. — [fontBuilder.py source](https://raw.githubusercontent.com/fonttools/fonttools/main/Lib/fontTools/fontBuilder.py)
- `feaLib.builder.addOpenTypeFeatures(font, featurefile, tables=None, debug=False)` compiles Adobe feature syntax into an existing TTFont and replaces any existing features. — [web search summary of fontTools feaLib docs](https://fonttools.readthedocs.io/en/latest/feaLib/)
- **ufo2ft 3.11.0** (2026-11-02, MIT, Py>=3.10) is "A bridge between UFOs and FontTools". It provides `compileOTF`/`compileTTF` (the variable-font functions exist too). It accepts Adobe `.fea` or MTI feature files. "If your font's features do not contain kerning/mark/mkmk features," ufo2ft creates them from kerning and anchor data. Filters (`decomposeComponents`, `flattenComponents`, `propagateAnchors`, `removeOverlaps`, `cubicToQuadratic`, …) are configured through the lib key `com.github.googlei18n.ufo2ft.filters`. — [PyPI ufo2ft](https://pypi.org/project/ufo2ft/), [ufo2ft README](https://raw.githubusercontent.com/googlefonts/ufo2ft/main/README.rst)
- ufo2ft's built-in feature writers are `CursFeatureWriter`, `GdefFeatureWriter`, `KernFeatureWriter` and `MarkFeatureWriter`. — [featureWriters/__init__.py](https://raw.githubusercontent.com/googlefonts/ufo2ft/main/Lib/ufo2ft/featureWriters/__init__.py)
- **ufoLib2 0.19.1** (2026-10-13, Apache-2.0). **defcon 0.12.2** (2025-07-16, MIT; slower release cadence, used mostly by older tooling). **fontParts 1.1.1** (2026-07-09, MIT; a RoboFont-style scripting API on top of defcon). — [PyPI ufoLib2](https://pypi.org/project/ufoLib2/), [PyPI defcon](https://pypi.org/project/defcon/), [PyPI fontParts](https://pypi.org/project/fontParts/)
- **fontmake 3.12.2** (2026-10-09, Apache-2.0) compiles `.glyphs`, `.ufo` and `.designspace` to `.otf`/`.ttf`, including variable fonts (`fontmake MyFont.glyphs -o variable`). The default output is `-o otf ttf`. `-f` flattens nested components, recommended because "certain rendering environments do not correctly handle nested components". — [PyPI fontmake](https://pypi.org/project/fontmake/), [fontmake README](https://raw.githubusercontent.com/googlefonts/fontmake/main/README.md)
- **glyphsLib 6.17.2** (2026-11-04, Apache-2.0): converts `.glyphs` to UFO (`glyphs2ufo`). Amiri's build uses `glyphs2ufo --minimal --generate-GDEF` and then a custom Python `build.py` with an explicit `.fea`. — [PyPI glyphsLib](https://pypi.org/project/glyphsLib/), [Amiri Makefile](https://raw.githubusercontent.com/aliftype/amiri/main/Makefile)
- **fontc 1.0.0** (2026-09-01) is "A font compiler written in Rust" from the Google Fonts team, on crates.io and PyPI. Related Rust crates: `write-fonts` 0.55.0, `skrifa` 0.49.0, and **harfrust 0.15.0**, "A complete HarfBuzz shaping algorithm port to Rust" (all updated Oct 2026). — [crates.io fontc](https://crates.io/crates/fontc), [PyPI fontc](https://pypi.org/project/fontc/), [crates.io harfrust](https://crates.io/crates/harfrust)
- Geometry helpers: **skia-pathops 0.9.2** (BSD-3, 2026-02-16) for overlap removal and booleans, **booleanOperations 0.10.0** (MIT, 2026-01-15), and **cu2qu**, now inside fontTools as `fontTools.cu2qu` (the standalone PyPI package was last released in 2022). — [PyPI skia-pathops](https://pypi.org/project/skia-pathops/), [PyPI booleanOperations](https://pypi.org/project/booleanOperations/), [PyPI cu2qu](https://pypi.org/project/cu2qu/)
- [Local test] FontBuilder plus a feaLib string with GDEF, init/medi/fina, rlig, curs and mark compiled without errors under fontTools 4.68.0. `setupOS2` warns unless `version=4` is passed when fsSelection bit 7 (USE_TYPO_METRICS) is set: "fsSelection bits 7, 8 and 9 are only defined in OS/2 table version 4 and up". WOFF2 export works via `TTFont.flavor = "woff2"` once the `brotli` package is installed.
- [Local test] A ufoLib2 font with `entry`/`exit` and `_bottom`/`bottom` anchors, passed through `ufo2ft.compileTTF`, auto-generated `curs` (with `lookupflag RightToLeft IgnoreMarks`), `mark2base` with `markClass … @mark_bottom` and a `GDEF GlyphClassDef`. The categories came from `lib["public.openTypeCategories"]`. ufo2ft did **not** propagate anchors from components into the composite `beh-ar.medi`, so the AI should emit anchors on every final glyph or enable the `propagateAnchors` filter.

### Inferences
- **Recommendation for full automation: AI → (outlines + anchors JSON) → ufoLib2 UFO → ufo2ft compileTTF/compileOTF → fontTools post-processing (name, OS/2, WOFF2) → HarfBuzz tests.** The UFO stays human-inspectable (FontGoggles, RoboFont, Glyphs) and diffable. ufo2ft takes the error-prone GPOS writing off the AI, and the same UFO can later feed fontmake/fontc for variable fonts with a designspace.
- FontBuilder alone is fine for a minimal pipeline. It has no dependencies beyond fontTools, but then every GDEF, curs and mark rule has to be generated by hand. The [Local test] shows this is about 60 lines of templated `.fea`.
- TTF (quadratic) vs OTF (CFF cubic): AI vector output is usually cubic Bézier. Either build CFF with `T2CharStringPen`, or convert with `fontTools.pens.cu2quPen` (ufo2ft does this automatically for TTF). Run `skia-pathops` overlap removal before writing static fonts, because overlapping contours render badly in some rasterizers.
- FontForge (C, GPLv3 with some BSD parts, from prior knowledge) has Python scripting and can generate fonts from SVG, but it is heavy to install headless and its OpenType feature model is clumsier. It is not recommended as the primary path.

### Gaps
- Could not open fonttools.readthedocs.io directly (network-blocked), so the docs text for `feaLib` and `varLib` was not checked beyond source code and search snippets.
- FontForge's current version and license were not checked (it is not on PyPI, and GitHub API access was blocked).

---

## Q2. Arabic OpenType features: which are needed, the minimal set, example .fea, and joining behaviour

### Takeaway
The shaping engine (HarfBuzz, DirectWrite/Uniscribe, CoreText) does the **contextual joining analysis itself** from Unicode joining types. The font only maps base letters to forms through `isol`/`init`/`medi`/`fina` (GSUB single substitutions). **Minimal usable set:** a correct `cmap`, a GDEF glyph-class table, `init`/`medi`/`fina` (plus `isol` if the isolated form differs from the cmap glyph), and `rlig` for لام-ألف. Add **`mark`** for harakat if they are supported, and either `curs` or zero-sidebearing joins so connections touch. `ccmp`, `mkmk`, `kern`, `calt` and `liga` are quality add-ons.

### Cited Findings
- HarfBuzz's Arabic shaper applies features **in this order, with GSUB pauses between stages**: `stch` → `ccmp`, `locl` → `isol`, `fina`, `fin2`, `fin3`, `medi`, `med2`, `init` (each its own stage) → `rlig` (with fallback) → `calt` → `liga`, `clig` → `mset`. The code comment says "The pause between init/medi/... and rlig is required." `cswh` is commented out (off by default). — [hb-ot-shaper-arabic.cc](https://raw.githubusercontent.com/harfbuzz/harfbuzz/main/src/hb-ot-shaper-arabic.cc)
- HarfBuzz derives each character's joining type from a generated Unicode table (`joining_type(u)`). Characters not in the table fall back to the general category: Mn/Me/Cf become **T** (transparent) and everything else becomes **U**. A 7-state machine over the columns U, L, R, D, ALAPH and DALATH_RISH assigns one action (ISOL/FINA/FIN2/FIN3/MEDI/MED2/INIT) per glyph and sets the corresponding feature mask. — [same source](https://raw.githubusercontent.com/harfbuzz/harfbuzz/main/src/hb-ot-shaper-arabic.cc)
- HarfBuzz has an `arabic_fallback_shape` path, used when the font lacks the needed features for Arabic script (`F_HAS_FALLBACK` on `rlig` and the joining features). — [same source](https://raw.githubusercontent.com/harfbuzz/harfbuzz/main/src/hb-ot-shaper-arabic.cc)
- Microsoft's "Developing OpenType Fonts for Arabic Script" describes the same model: contextual analysis selects isol/init/medi/fina, and `rlig` is then applied "to compose any mandatory ligatures, like 'lam alef'". isol, fina, medi, init and rlig are listed as always-applied GSUB features, and not every feature applies to every Arabic-script language. *(Seen via search summary because the page was network-blocked.)* — [Microsoft Typography: Arabic script development](https://learn.microsoft.com/en-us/typography/script-development/arabic)
- Joining-type source data: `ArabicShaping.txt` in the UCD, with lines like `0628; BEH; D; BEH` (fields: code point; name; joining type R/L/D/C/U/T; joining group). — [Unicode UCD ArabicShaping.txt](https://www.unicode.org/Public/UCD/latest/ucd/ArabicShaping.txt) *(the format is from prior knowledge; unicode.org was network-blocked in this session)*. Python access to the latest UCD, including joining data, is available through **youseedee 0.7.0** ("Interface to the latest version of the Unicode Character Database"). `fontTools.unicodedata` provides `script()`, `script_horizontal_direction()` and `ot_tags_from_script()`, but no joining type. — [PyPI youseedee](https://pypi.org/project/youseedee/); [Local test] `fontTools.unicodedata` listing
- ufo2ft's `CursFeatureWriter` builds `curs` from glyph anchors named `entry`/`exit`, plus suffixed pairs `entry.<x>`/`exit.<x>`. The lookup flag is `IgnoreMarks` for LTR and `("IgnoreMarks","RightToLeft")` otherwise. It splits lookups into `_ltr`/`_rtl` only if the cmap contains LTR characters, and `.RTL`/`.LTR` anchor suffixes override the direction. By default it skips writing if a `curs` feature already exists. — [cursFeatureWriter.py](https://raw.githubusercontent.com/googlefonts/ufo2ft/main/Lib/ufo2ft/featureWriters/cursFeatureWriter.py)
- ufo2ft's `MarkFeatureWriter` treats anchors prefixed "_" as mark anchors (`_top`, `_bottom`) and attaches them to bases carrying the same anchor without the underscore (`top`, `bottom`). Suffixes `top_1`, `top_2`, … address ligature components (counting from 1), and an empty component uses `<anchor NULL>`. It writes `mark`, `mkmk` (and `abvm`/`blwm` for Indic/USE/Khmer shapers only). — [markFeatureWriter.py](https://raw.githubusercontent.com/googlefonts/ufo2ft/main/Lib/ufo2ft/featureWriters/markFeatureWriter.py)
- shaperglot's Arabic test requires an `init` feature whose shaped output differs with `init` on and off. It also checks that marks actually attach to bases or compose. — [shaperglot README](https://raw.githubusercontent.com/googlefonts/shaperglot/main/README.md)
- [Local test] The `.fea` below (FontBuilder plus feaLib) shaped correctly in HarfBuzz 0.57.0 (uharfbuzz). `بب` gave `beh-ar.fina` + `beh-ar.init`, `ببب` gave fina/medi/init, `لا` gave `lam_alef-ar`, `بلا` gave `lam_alef-ar.fina` + `beh-ar.init`, and `بَ` positioned the fatha at x_offset 250, y_offset 320 from the mark anchor. Note: HarfBuzz returns RTL glyphs in **visual order**, so the logically last glyph comes first.

```fea
languagesystem DFLT dflt;
languagesystem arab dflt;
# GDEF: bases, ligatures, marks (required for IgnoreMarks and mark positioning to work)
@GDEF_base = [beh-ar beh-ar.init beh-ar.medi beh-ar.fina alef-ar alef-ar.fina lam-ar lam-ar.init lam-ar.medi lam-ar.fina];
@GDEF_liga = [lam_alef-ar lam_alef-ar.fina];
@GDEF_mark = [dotbelow-ar fatha-ar];
table GDEF { GlyphClassDef @GDEF_base, @GDEF_liga, @GDEF_mark, ; } GDEF;

feature init { lookupflag IgnoreMarks; sub [beh-ar lam-ar] by [beh-ar.init lam-ar.init]; } init;
feature medi { lookupflag IgnoreMarks; sub [beh-ar lam-ar] by [beh-ar.medi lam-ar.medi]; } medi;
feature fina { lookupflag IgnoreMarks; sub [beh-ar alef-ar lam-ar] by [beh-ar.fina alef-ar.fina lam-ar.fina]; } fina;

# rlig runs AFTER the positional forms -> match the substituted forms, not the base glyphs
feature rlig { lookupflag IgnoreMarks;
  sub lam-ar.init alef-ar.fina by lam_alef-ar;        # isolated لا
  sub lam-ar.medi alef-ar.fina by lam_alef-ar.fina;   # joined ـلا
} rlig;

# cursive: pos cursive <glyph> <entry anchor (right side)> <exit anchor (left side)>
feature curs { lookup curs_rtl { lookupflag RightToLeft IgnoreMarks;
  pos cursive beh-ar.init  <anchor NULL>  <anchor 40 0>;
  pos cursive beh-ar.medi  <anchor 300 0> <anchor 0 0>;
  pos cursive beh-ar.fina  <anchor 440 0> <anchor NULL>;
  pos cursive alef-ar.fina <anchor 150 0> <anchor NULL>;
} curs_rtl; } curs;

markClass dotbelow-ar <anchor 0 0> @bottom;
markClass fatha-ar    <anchor 0 0> @top;
feature mark {
  lookup mark_base { pos base beh-ar <anchor 250 -20> mark @bottom <anchor 250 320> mark @top; } mark_base;
  lookup mark_liga { pos ligature [lam_alef-ar lam_alef-ar.fina]
      <anchor 410 760> mark @top  ligComponent  <anchor 100 760> mark @top; } mark_liga;
} mark;
```

### Inferences
- **Strict minimum for "text connects and reads correctly"**: (1) cmap for U+0621–U+064A, U+0660–0669, U+060C/061B/061F/0640 tatweel, and U+064B–0652 harakat; (2) `.init/.medi/.fina` glyphs for every dual-joining letter (type D) and `.fina` for right-joining letters (R: alef forms, dal, thal, reh, zain, waw, teh marbuta…), with no init/medi for R-type letters; (3) `rlig` with at least لا, لأ, لإ, لآ in isolated and final forms (8 ligatures); (4) GDEF classes; (5) `mark` for harakat and hamza marks. Generate the init/medi/fina tables **from ArabicShaping.txt joining types**, never by hand.
- **Joins: curs or "zero-sidebearing".** The simplest robust approach is to have the AI draw every joining form so the baseline stroke runs exactly to x=0 (left/exit side) and to x=advance (right/entry side) at the same y-band. Joins then touch without `curs`, which is how most Naskh and Kufi web fonts work. Use `curs` when the AI's join heights vary, as in Nastaliq-like or calligraphic styles. It lets each glyph declare its exit/entry and the shaper aligns them, but Word and older engines support `curs` less reliably than plain advances (prior knowledge, not verified this session).
- `ccmp` is needed if the font decomposes precomposed letters into base + dot components, or composes shadda+fatha into a single glyph. `mkmk` is needed for stacked marks (shadda + fatha/kasra). Both come almost free from ufo2ft once `_top`/`top` anchors exist on mark glyphs too.
- `kern` for Arabic is usually small (for example reh/waw/dal followed by a following letter). Keep it to class kerning and skip it in a minimal v1.
- Use the language systems `arab dflt` plus `arab URD`/`FAR` only if language-specific forms (Persian or Urdu digits) are added through `locl`.

### Gaps
- Could not fetch the Microsoft guide's full templates or its lookup-flag guidance, or the OpenType feature registry pages, directly (network-blocked).
- Did not verify Microsoft Word's and Adobe InDesign's `curs` support empirically. Adobe's "World-Ready Composer" needs to be on for Arabic shaping in InDesign/Illustrator (prior knowledge).

---

## Q3. Using composite glyphs and components to cut the number of generated shapes

### Takeaway
Have the AI generate only **dotless rasm bodies** (about 18 skeleton classes × 4 forms) plus a handful of **dot and mark glyphs**. Build every dotted letter as a **TrueType composite** (body + dot components at offsets), or use GSUB `ccmp` decomposition. Arabic letter groups share bodies: ب ت ث ن(init/medi) ي(init/medi) share one tooth body, ج ح خ share one, د ذ, ر ز, س ش, ص ض, ط ظ, ع غ, ف ق(init/medi).

### Cited Findings
- [Local test] A `beh-ar.medi` composite made of `behDotless-ar.medi` plus a `dotbelow-ar` component compiled under ufo2ft with `flattenComponents=True`. GDEF, curs and mark were generated for the body glyph, but anchors were not inherited by the composite unless set explicitly or propagated.
- fontmake's `-f` flag flattens nested components, recommended because "certain rendering environments do not correctly handle nested components" — [fontmake README](https://raw.githubusercontent.com/googlefonts/fontmake/main/README.md)
- ufo2ft has `decomposeComponents`, `flattenComponents` and `propagateAnchors` filters (configured in the UFO lib) — [ufo2ft README](https://raw.githubusercontent.com/googlefonts/ufo2ft/main/README.rst)
- The Cairo README notes it uses "connecting components" and so "font must be generated by Glyphs". In other words, Glyphs-specific smart-component and connector features may not survive open-source pipelines. — [Cairo README](https://raw.githubusercontent.com/Gue3bara/Cairo/master/README.md)
- Unicode joining groups (fourth field of ArabicShaping.txt, for example BEH, HAH, DAL, REH, SEEN) formalise which letters share a skeleton — [ArabicShaping.txt](https://www.unicode.org/Public/UCD/latest/ucd/ArabicShaping.txt) *(format from prior knowledge)*

### Inferences
- Component approach: `beh-ar.init = behDotless-ar.init + dotbelow-ar @ (x,y)`, `teh-ar.init = behDotless-ar.init + twodotsabove-ar @ (x,y)`, and so on. TrueType composites keep file size small and guarantee identical dots. Keep components one level deep, or flatten them.
- Dot placement can be computed from the body's `top`/`bottom` anchors. The AI then emits only one anchor per body form, and the build script places dot components there. The same anchors double as harakat attachment points in `mark` (offset beyond the dots, or chained with `mkmk`).
- Alternative: give dots their own glyphs and position them by GPOS (`ccmp` decomposes ب → dotless body + dot mark, then `mark` places the dot). This needs fewer glyphs, but text-extraction, PDF and search behaviour is more fragile, and it is less common. Prefer static composites.
- Shared bodies cut AI generation to roughly 18 joining groups × up to 4 forms ≈ 60 bodies, plus about 10 dot and mark shapes, plus 8 لا ligatures, instead of about 120 letter forms. It also enforces consistency: the same tooth everywhere.

### Gaps
- No primary source measured rendering differences between composite and decomposed dots across Word, Adobe and browsers.

---

## Q4. Testing and validation: HarfBuzz, fontbakery/fontspector, shaperglot, browsers, and automatic join verification

### Takeaway
Run three layers of checks automatically. (1) **Structural QA** with fontspector (the Rust successor of fontbakery) or fontbakery. (2) **Shaping tests** with uharfbuzz: assert the expected glyph names per test word, including init/medi/fina and لا, and make sure no `.notdef` appears. (3) **Geometric join tests**: place the shaped glyphs and sample the outlines either side of every seam with `PointInsidePen`, or rasterise and check pixel continuity. Add shaperglot language-support checks, and browser screenshots (Playwright with Chromium, Firefox and WebKit) for visual regression.

### Cited Findings
- **uharfbuzz 0.57.0** (2026-10-23, Apache-2.0, Py>=3.11), Cython bindings to HarfBuzz. Usage: `hb.Blob.from_file_path` → `hb.Face` → `hb.Font` → `hb.Buffer().add_str(text); buf.guess_segment_properties(); hb.shape(font, buf, features)`. Results come back as `buf.glyph_infos` (`codepoint` = glyph id, `cluster`) and `buf.glyph_positions` (`x_advance`, `y_advance`, `x_offset`, `y_offset`), and `font.glyph_to_string(gid)` gives names. — [PyPI uharfbuzz](https://pypi.org/project/uharfbuzz/), [uharfbuzz README](https://raw.githubusercontent.com/harfbuzz/uharfbuzz/main/README.md)
- **fontbakery 1.1.0** was last released 2025-10-02. **fontspector 1.9.0** ("Quality control for OpenType fonts", Rust) was updated 2026-10-09, and its wiki now documents checks such as `arabic_spacing_symbols`. — [PyPI fontbakery](https://pypi.org/project/fontbakery/), [crates.io fontspector](https://crates.io/crates/fontspector), [fontspector wiki: arabic_spacing_symbols](https://github.com/fonttools/fontspector/wiki/arabic_spacing_symbols)
- Arabic-specific QA checks: `arabic_spacing_symbols` (U+FBB2–FBC1 must not be GDEF marks), `arabic_high_hamza` (U+0675 must not be a mark; it is a raised variant of U+0621), `gdef_mark_chars` and `gdef_non_mark_chars` (mark characters must be in the GDEF mark class and vice versa). Real reports flag, for example, U+0670 and U+06DB in Zain. — [Font Bakery GDEF profile docs](https://font-bakery.readthedocs.io/en/latest/fontbakery/profiles/gdef.html), [GF dashboard IBM Plex Sans Arabic report](https://googlefonts.github.io/gf-dashboard/fontbakery-reports/ibmplexsansarabic-report.html), [Zain report](https://googlefonts.github.io/gf-dashboard/fontbakery-reports/zain-report.html) *(via search summaries)*
- **shaperglot 1.2.2** (PyPI, 2026-09-09) is now a Rust library with PyO3 Python bindings (`from shaperglot import Checker`). It is also a CLI (`cargo install --git https://github.com/googlefonts/shaperglot`, subcommands `check`/`report`/`describe`) and a client-side web app. It tests behaviour, not just coverage. For Arabic it checks that `init` exists and changes shaping, and that marks attach. — [shaperglot README](https://raw.githubusercontent.com/googlefonts/shaperglot/main/README.md), [PyPI shaperglot](https://pypi.org/project/shaperglot/)
- Other Google tooling: **gftools 0.10.8** (2026-11-04) for builders and QA, and **diffenator2 0.5.0** (2026-05-08) for comparing two fonts, including rendered glyph and word diffs. — [PyPI gftools](https://pypi.org/project/gftools/), [PyPI diffenator2](https://pypi.org/project/diffenator2/)
- The Noto Arabic repo runs `make test` (FontBakery) and `make proof` (HTML proofs) in GitHub Actions, with badges for FontBakery, Universal, Google Fonts profile, Outline Correctness and Shaping checks. This is a ready-made CI template. — [notofonts/arabic README](https://raw.githubusercontent.com/notofonts/arabic/main/README.md)
- [Local test] **Automatic join checker** (complete, works): shape with HarfBuzz, accumulate the pen x in visual order, then use `fontTools.pens.pointInsidePen.PointInsidePen` to test ink at `seam−3` in the left glyph and `seam+3` in the right glyph at the join height. Results: `ببب` passed both seams, `بلا` passed, `لب` passed, and `ب ب` reported GAPs at the space, as expected. **Pitfall found:** the seam is the *pen position* (sum of advances). `x_offset` from `curs` moves the outline only, so testing at `pen+x_offset` gave false GAPs.

```python
import uharfbuzz as hb
from fontTools.ttLib import TTFont
from fontTools.pens.pointInsidePen import PointInsidePen
tt = TTFont(path); gs = tt.getGlyphSet()
font = hb.Font(hb.Face(hb.Blob.from_file_path(path)))
buf = hb.Buffer(); buf.add_str(text); buf.guess_segment_properties(); hb.shape(font, buf, {})
x = 0; placed = []
for info, pos in zip(buf.glyph_infos, buf.glyph_positions):          # visual order (L->R)
    name = font.glyph_to_string(info.codepoint)
    placed.append((name, x + pos.x_offset, pos.y_offset, x)); x += pos.x_advance
def inked(name, ox, oy, px, py):
    pen = PointInsidePen(gs, (px - ox, py - oy)); gs[name].draw(pen); return pen.getResult()
bases = [p for p in placed if gs[p[0]].width > 0]
for (a, ax, ay, _), (b, bx, by, seam) in zip(bases, bases[1:]):
    ok = inked(a, ax, ay, seam - 3, JOIN_Y) and inked(b, bx, by, seam + 3, JOIN_Y)
    # only assert ok where joining is expected: a has a left-join (init/medi form) per ArabicShaping
```

### Inferences
- Shaping test corpus: every letter in each position (for example `ـبـ` tests built with U+0640 tatweel or a neighbour letter), all لا variants, every letter followed by each harakat, shadda+harakat stacks, ZWJ/ZWNJ cases (U+200D/U+200C), numerals and punctuation. Golden expectations can be generated from the joining table: dual-joining letter between two dual-joiners gives `.medi`.
- Join check: base the expected join set on joining types. Only assert continuity when glyph a (visual left) is init or medi and b (visual right) joins to it. Also compare the **stroke thickness and y-range** at the seam: sample several y values and require that the ink intervals overlap at least 80%. An alternative raster method uses `fontTools.pens.freetypePen` or uharfbuzz drawing to render each word and check for a continuous horizontal ink band through the seam column.
- Cross-engine: test HarfBuzz, which covers Chrome, Firefox, Android, LibreOffice and Adobe products that adopted HarfBuzz (prior knowledge). Separately smoke-test DirectWrite (Windows and Word) and CoreText (Safari and macOS), because these have their own Arabic shapers. Automate browsers with Playwright using a `@font-face` WOFF2 test page and pixel diffs.
- Microsoft Font Validator is a legacy, Windows-centric open-source tool. fontspector/fontbakery plus OTS (the OpenType Sanitizer used by browsers, which rejects malformed web fonts) are the practical modern equivalents. Run `ots-sanitize` on the WOFF2.

### Gaps
- Did not verify Font Validator's current maintenance status or OTS's Python packaging this session.
- No source found for an existing open-source "Arabic join continuity" checker. The checker above is original, built during this research.

---

## Q5. Metrics automation: advances, sidebearings, baseline and vertical metrics for Arabic, and kerning tools

### Takeaway
Derive advances from the outline bounding box plus fixed sidebearings: **0 on joining sides** so strokes meet, and a style-dependent value (about 4–8% UPM) on non-joining sides. Compute vertical metrics from the **actual font-wide yMax/yMin**, including stacked harakat over lam/alef and descenders of ي/ع/ن. Set hhea = typo, turn on USE_TYPO_METRICS (OS/2 v4, fsSelection bit 7), set lineGap to 0, and make winAscent/winDescent ≥ the bounding box so nothing clips in Windows/Word.

### Cited Findings
- FontBuilder example: `setupHorizontalMetrics({glyph: (advanceWidth, lsb)})` with lsb taken from `glyf[gn].xMin`, `setupHorizontalHeader(ascent=…, descent=…)`, and `setupOS2(sTypoAscender=…, usWinAscent=…, usWinDescent=…)`. — [fontBuilder.py](https://raw.githubusercontent.com/fonttools/fonttools/main/Lib/fontTools/fontBuilder.py)
- [Local test] `setupOS2(version=4, fsSelection=0x40|0x80, ...)`: bit 7 (USE_TYPO_METRICS) needs OS/2 version ≥ 4, otherwise fontTools warns. Arabic Unicode range: `ulUnicodeRange1` bit 13.
- `curs` positioning overrides advances at joins. HarfBuzz output for `بب` with an init exit anchor at x=40 was `beh-ar.init` advance 260 with x_offset −40, compared with the designed advance 300. — [Local test]
- With ufo2ft, vertical metrics come from UFO fontinfo (`ascender`, `descender`, `openTypeHheaAscender`, `openTypeOS2TypoAscender`, `openTypeOS2WinAscent`, …), with static fallbacks in `fontInfoData.py` — [ufo2ft README](https://raw.githubusercontent.com/googlefonts/ufo2ft/main/README.rst)

### Inferences
- Google Fonts vertical-metrics rules (from prior knowledge of the GF guide, which was network-blocked here): typoAscender − typoDescender + typoLineGap = hhea ascender − hhea descender + hhea lineGap, with USE_TYPO_METRICS on. winAscent ≥ font yMax and winDescent ≥ |font yMin|. Line height is usually about 1.2–1.5× UPM for Latin, but Arabic with full harakat often needs about 1.6–2.0× UPM (Amiri and Scheherazade-style Naskh fonts are tall). For a family that pairs with Latin, keep the Latin cap height and x-height and set the Arabic "tooth" height to roughly the Latin x-height.
- Suggested automatic layout for UPM 1000: baseline 0, Arabic join stroke centred about 0–80, tooth height ≈ x-height (~450–500), alef/lam ascender ≈ 700–750, harakat above up to about 1000–1100, descender (ي ن ع ح) ≈ −350 to −450, marks below to about −500. Then set typoAscender = max ink + small margin and typoDescender = min ink − margin.
- Non-joining side bearings: run a simple "optical spacing" pass, comparing the area between a glyph's profile and a vertical line (in the spirit of HTLetterspacer). An AI could also predict sidebearings. Kerning: generate class kerning only for right-joining letters (ر ز و د ذ) and final forms against following letters, use HarfBuzz to measure gaps, and fill `kern` via the UFO `kerning.plist` and groups, which ufo2ft's `KernFeatureWriter` handles for RTL.
- Do **not** add tracking or letterspacing inside joining forms. Arabic must not be letterspaced (kashida/tatweel U+0640 is the justification mechanism). Make sure the tatweel glyph has a pure baseline stroke from 0 to its advance, with `curs` anchors if `curs` is used.

### Gaps
- Could not fetch googlefonts.github.io/gf-guide/metrics.html to quote the exact current rules. The values above are prior knowledge and should be checked.
- No current, maintained open-source *automatic Arabic kerning* tool was found (the PyPI search for "kerncraft" returned an unrelated HPC package).

---

## Q6. Open-source Arabic font sources as structural templates, and their licenses

### Takeaway
All the candidate fonts are **OFL-1.1**, but several declare **Reserved Font Names (RFNs)**, so derivatives must be renamed. **Amiri** (no Arabic RFN, `.glyphspackage` source plus a hand-written `.fea`, built with glyphsLib, ufo2ft and fontTools) and **Noto Naskh/Sans Arabic** (no RFN, Google CI) are the best structural templates for glyph naming and feature code. SIL's Scheherazade New and Harmattan reserve their names. Use the templates for **naming conventions and feature architecture**, not their outlines, if the output is meant to be independent of them.

### Cited Findings
| Font | Source repo | License / RFN (from OFL.txt in google/fonts or upstream) |
|---|---|---|
| Amiri | [aliftype/amiri](https://github.com/aliftype/amiri) | OFL-1.1, "Copyright 2010-2022 The Amiri Project Authors", **no RFN** — [OFL.txt](https://raw.githubusercontent.com/aliftype/amiri/main/OFL.txt) |
| Noto Naskh Arabic / Noto Sans Arabic | [notofonts/arabic](https://github.com/notofonts/arabic) | OFL-1.1, "Copyright 2022 The Noto Project Authors", **no RFN** — [google/fonts ofl/notonaskharabic](https://raw.githubusercontent.com/google/fonts/main/ofl/notonaskharabic/OFL.txt) |
| Cairo | [Gue3bara/Cairo](https://github.com/Gue3bara/Cairo) | OFL-1.1, "Copyright 2009 The Cairo Project Authors", no RFN — [google/fonts ofl/cairo](https://raw.githubusercontent.com/google/fonts/main/ofl/cairo/OFL.txt) |
| Reem Kufi | [aliftype/reem-kufi](https://github.com/aliftype/reem-kufi) | OFL-1.1, **RFN "Josefin Sans"** (its Latin part comes from Josefin Sans) — [google/fonts ofl/reemkufi](https://raw.githubusercontent.com/google/fonts/main/ofl/reemkufi/OFL.txt) |
| Rakkas | [zeynepakay/Rakkas](https://github.com/zeynepakay/Rakkas) | OFL-1.1, "Copyright 2016 Zeynep Akay", no RFN — [google/fonts ofl/rakkas](https://raw.githubusercontent.com/google/fonts/main/ofl/rakkas/OFL.txt) |
| Lalezar | [BornaIz/Lalezar](https://github.com/BornaIz/Lalezar) | OFL-1.1, no RFN — [google/fonts ofl/lalezar](https://raw.githubusercontent.com/google/fonts/main/ofl/lalezar/OFL.txt) |
| Scheherazade New | [silnrsi/font-scheherazade](https://github.com/silnrsi/font-scheherazade) | OFL-1.1, "Copyright (c) 1994-2026, SIL Global", **RFNs "Scheherazade" and "SIL"** — [OFL.txt](https://raw.githubusercontent.com/silnrsi/font-scheherazade/master/OFL.txt) |
| Harmattan | [silnrsi/font-harmattan](https://github.com/silnrsi/font-harmattan) | OFL-1.1, **RFNs "Fouta", "Andika", "SIL", "Harmattan"** — [OFL.txt](https://raw.githubusercontent.com/silnrsi/font-harmattan/master/OFL.txt) |
| Aref Ruqaa | [aliftype/aref-ruqaa](https://github.com/aliftype/aref-ruqaa) | OFL-1.1, RFN "EURM10" — [google/fonts ofl/arefruqaa](https://raw.githubusercontent.com/google/fonts/main/ofl/arefruqaa/OFL.txt) |
| El Messiri, Tajawal | [Gue3bara/El-Messiri](https://github.com/Gue3bara/El-Messiri), [googlefonts/tajawal](https://github.com/googlefonts/tajawal) | OFL-1.1, no RFN — [google/fonts ofl/elmessiri](https://raw.githubusercontent.com/google/fonts/main/ofl/elmessiri/OFL.txt), [ofl/tajawal](https://raw.githubusercontent.com/google/fonts/main/ofl/tajawal/OFL.txt) |

- Amiri was "actively developed between 2008–2022" and version 1 is considered mature, with no further development planned. Its build is `glyphs2ufo --minimal --generate-GDEF` followed by `scripts/build.py --features=sources/Amiri.fea`. — [Amiri README](https://raw.githubusercontent.com/aliftype/amiri/main/README.md), [Amiri Makefile](https://raw.githubusercontent.com/aliftype/amiri/main/Makefile)
- Reem Kufi ships as a variable font plus static instances, and has `cv01`, `cv02`, … character variants. — [Reem Kufi README](https://raw.githubusercontent.com/aliftype/reem-kufi/main/README.md)
- Cairo's recent sources rely on Glyphs "connecting components", so the font "must be generated by Glyphs". It is less suitable as an open-pipeline template. — [Cairo README](https://raw.githubusercontent.com/Gue3bara/Cairo/master/README.md)
- OFL RFN rule: modified versions may not use a Reserved Font Name. Internal names must change even for small modifications. Format conversion or rebuilding that changes font data normally counts as a modified version. Google Fonts requires new submissions to have **no** RFNs. — [OFL-FAQ (mirror)](https://matojo.unizar.es/ctan/fonts/amsfonts/doc/OFL-FAQ.txt), [openfontlicense.org: How to modify OFL fonts](https://openfontlicense.org/how-to-modify-ofl-fonts/) *(via search summary)*

### Inferences
- For glyph naming, the practical convention in open pipelines (glyphsLib and Noto) is `beh-ar`, `beh-ar.init`, `beh-ar.medi`, `beh-ar.fina`, `lam_alef-ar`, `fatha-ar`, `behDotless-ar`, plus `uniXXXX` production names that glyphsLib can rename. The Glyphs-style names are readable, and the `-ar` suffix separates Arabic from Persian/Urdu variants (`-farsi`, `-urdu`).
- Feature files from OFL fonts are themselves part of the OFL "Font Software". Copying Amiri's `.fea` verbatim makes the output a derivative, so it must be OFL with Amiri's copyright notice. Writing a fresh generator that emits the standard init/medi/fina/rlig/mark pattern from Unicode data does **not** copy protected expression. The patterns are dictated by the OpenType and Unicode specs.
- If the goal is a commercially licensable, non-OFL output, avoid deriving outlines or feature code from any OFL source and generate features from Unicode data. If OFL output is acceptable, Amiri and Noto (no RFN) are the easiest bases, though the font must still be renamed.

### Gaps
- Did not inspect notofonts/arabic's source format (README does not state it; GitHub API blocked) or Amiri's `.fea` glyph-naming in detail.

---

## Q7. Licensing of fonts produced by an AI trained on OFL fonts

### Takeaway
SIL's official OFL FAQ (entries 1.25 and 1.26 in the current version) says that **any font produced by an AI or ML system whose training data or input includes OFL font sources should be considered a derivative work and must remain under the OFL**. Non-font graphical output from such tools is treated like normal use. This is SIL's interpretation, not a court ruling. Separately, US copyright law may give no copyright at all to purely AI-generated material, which could leave such output unlicensable.

### Cited Findings
- OFL FAQ: "Any font produced from such systems whose input or training data contains any source file from a font project licensed under the OFL should be considered a derivative work", and such fonts remain under the OFL "regardless of the way the sources might have been transformed, in whole or in part". Entry 1.26 says OFL fonts may be used by AI design tools when the result is not intended to be a font, which is "equivalent to using an OFL font in the graphic design process". *(Quoted via search-engine extraction; openfontlicense.org was network-blocked.)* — [OFL-FAQ](https://openfontlicense.org/ofl-faq/)
- A community discussion (Chinese) of the FAQ's AI-generated-glyph clause in the context of an AI-assisted font project — [Sumr-Sans issue #1](https://github.com/feiyangjun-1/Sumr-Sans/issues/1)
- TypeDrawers discussion (Oct 2024), opinions only: Dave Crossland argued that training on libre fonts is allowed but outputs derived from OFL fonts must stay OFL. John Hudson noted the tension with marketing AI fonts as proprietary branding assets. — [TypeDrawers: AI for custom type](https://typedrawers.com/discussion/5175/ai-for-custom-type)
- Counterpoint (secondary report of Thomas Phinney's view): under US copyright law, purely AI-generated material has no copyright, so such fonts arguably cannot be licensed under the OFL at all, while hybrid works with substantial human editing may be. — [Fontzz write-up, fontalternatives.com](https://fontalternatives.com/blog/fontzz-ai-generated-fonts/) (not verified against the original post); a TypeDrawers commenter likewise doubted such fonts meet the legal definition of "derivative works" — [TypeDrawers 2026 thread](https://typedrawers.com/discussion/5512/state-of-the-art-in-ai-image-generation-as-we-go-into-2026/p3)
- OFL derivatives must stay OFL and cannot be sold by themselves (they may be bundled with other software). — [SIL Font FAQ](https://software.sil.org/fonts/faq/), [OFL text, Wikipedia overview](https://en.wikipedia.org/wiki/SIL_Open_Font_License)

### Inferences
- Product-design implications: (a) if the model was trained on OFL fonts, the safest default is to **ship every generated font under OFL-1.1**. Include an OFL.txt, list the user as copyright holder for their human contributions, and never use the RFNs of fonts in the training set (Scheherazade, SIL, Harmattan, Andika, Josefin Sans, …). (b) To offer proprietary licensing, train only on licensed or proprietary data or on public-domain fonts, keep data-provenance records, and generate OpenType feature code from Unicode data rather than OFL sources. (c) Because purely AI output may not be copyrightable (US), the user's own style input (their handwriting or sketches) and any human edits are what make the work protectable. Record them.
- The FAQ is guidance from the license steward, not binding law. Enforcement and court treatment are untested, and jurisdictions differ (EU, Saudi Arabia, UAE and Egypt were not researched). Get legal review before commercial launch.

### Gaps
- Could not open openfontlicense.org directly to confirm entry numbering (1.25/1.26), date and exact wording. These came from search-engine extraction.
- No court decisions found on AI-generated fonts or OFL training. Copyright positions outside the US were not researched.
