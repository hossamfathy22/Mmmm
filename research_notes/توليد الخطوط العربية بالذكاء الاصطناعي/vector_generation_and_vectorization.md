# Vector (SVG/Bézier) glyph generation models and image-to-vector conversion tools (as of Nov 2026)

Research method note (read first): this session could reach only `raw.githubusercontent.com` and the web-search tool. The GitHub REST API, github.com HTML, arxiv.org, huggingface.co and project pages were blocked by the session network proxy. As a result:
- **Licenses** below were verified by downloading each repo's actual LICENSE/COPYING file from raw.githubusercontent.com on 2026-11-08 (high confidence).
- **Weights / datasets / scripts** were verified from each repo's README fetched raw on 2026-11-08 (high confidence for what the README says).
- **GitHub star counts could NOT be retrieved** (API and HTML blocked). Any star figures are therefore absent or flagged as unverified estimates. The report writer should not present star counts as verified.
- Paper details (method, curve type, comparisons) come from web-search summaries of arXiv/CVF pages and are marked as such where uncertain.

## (A) Generative models that output vector glyphs directly

### Takeaway
No released vector-glyph generator has been trained on Arabic. All font-specific models (DeepVecFont, DeepVecFont-v2, DualVector, VecFusion, VecFontSDF, VecGlypher, VecFontLLM) target Latin (52 letters or ASCII) and/or Chinese. The strongest open, permissively licensed options in late 2026 are DeepVecFont-v2 (MIT, weights for English and Chinese), DualVector (MIT, weights) and VecGlypher (CVPR 2026, Apache-2.0, 27B weights, Latin-centric). Each would need retraining on Arabic data, plus extra handling for contextual forms, before it could serve the "one word, then the whole Arabic font" use case. Generic SVG LLMs (StarVector, OmniSVG, LLM4SVG) and text-to-SVG optimizers (VectorFusion, SVGDreamer, Word-As-Image) are not font generators. They are useful as vectorizers or for per-glyph stylization, not for producing a consistent style across a whole font.

### Cited Findings

**Font-specific vector generators**

- **DeepVecFont (SIGGRAPH Asia 2021)**, Yizhi Wang & Zhouhui Lian, "Synthesizing High-quality Vector Fonts via Dual-modality Learning". Repo: https://github.com/yizhiwang96/deepvecfont. License: **MIT** (LICENSE.txt, © 2021 Yizhi Wang). It does few-shot generation: "Given a few vector glyphs of a font as reference, our model generates the full vector font". The default image size was raised to 128×128, and Chinese support was moved to v2. — [README](https://raw.githubusercontent.com/yizhiwang96/deepvecfont/master/README.md), [LICENSE](https://raw.githubusercontent.com/yizhiwang96/deepvecfont/master/LICENSE.txt), [paper](https://arxiv.org/pdf/2110.06688)
- **DeepVecFont-v2 (CVPR 2023)**. Repo: https://github.com/yizhiwang96/deepvecfont-v2. License: **MIT** (© 2023 Yizhi Wang). Trained checkpoints for **English and Chinese** are on OneDrive/Baidu (epochs 500/550/600), and the dataset is downloadable. It does few-shot testing with configurable `ref_char_ids`. `MAX_SEQ_LEN` is 50 for English and 70 for Chinese, and a "Customize Dataset" pipeline accepts your own TTF/OTF. **Caveat:** "all the Chinese fonts are collected from Founder, and the fonts CANNOT be used for any commercial uses without permission from Founder." — [README](https://raw.githubusercontent.com/yizhiwang96/deepvecfont-v2/main/README.md), [LICENSE](https://raw.githubusercontent.com/yizhiwang96/deepvecfont-v2/main/LICENSE)
- DeepVecFont-v2 uses a "relaxation representation" with a Transformer backbone to improve "capability and stability of synthesizing long and complex outlines". Its curves are cubic (N_p = 3, "three-order Bézier"), sample points on the curves feed an alignment loss, and a context-based self-refinement module removes artifacts. — [arXiv 2303.14585](https://arxiv.org/abs/2303.14585), [CVF PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_DeepVecFont-v2_Exploiting_Transformers_To_Synthesize_Vector_Fonts_With_Higher_Quality_CVPR_2023_paper.pdf) (via search summary)
- **DualVector (CVPR 2023)**, "Unsupervised Vector Font Synthesis with Dual-Part Representation". Repo: https://github.com/thuliu-yt16/dualvector. License: **MIT** (© 2023 Yingtian Liu). Pretrained models for reconstruction and generation were released on Google Drive (2023-10-11). It depends on diffvg (with DeepVecFont's save_svg.py). Dataset: "the same as that used in DeepVecFont, a subset from SVG-VAE", which is Latin only. — [README](https://raw.githubusercontent.com/thuliu-yt16/dualvector/main/README.md), [LICENSE](https://raw.githubusercontent.com/thuliu-yt16/dualvector/main/LICENSE)
- DualVector represents glyphs as unions of positive and negative closed Bézier paths combined by boolean operations, with a raster-guided contour refinement stage. Reported comparison (from the VecFontLLM paper): "DeepVecFont and DeepVecFont-v2 often collapse into fragmented shapes. DualVector produces denser outlines, but still distorts structure and style." — [VecFontLLM arXiv HTML](https://arxiv.org/html/2607.17251v1) (via search summary). The search did not confirm whether DualVector's primitives are quadratic or cubic.
- **VecFusion (CVPR 2024)**, Thamizharasan et al. (UMass Amherst / Adobe / Google / U. Toronto). A two-stage cascade: a raster diffusion model followed by a vector diffusion model that predicts Bézier control points. It supports few-shot style transfer and is compared against DeepVecFont-v2. **Code release unconfirmed**: the project page has a "Code" link with an unknown target, and an "Any code?" question on the HF paper page went unanswered. — [project page](https://vikastmz.github.io/VecFusion/), [CVF PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Thamizharasan_VecFusion_Vector_Font_Generation_with_Diffusion_CVPR_2024_paper.pdf), [HF papers](https://huggingface.co/papers/2312.10540), [Adobe Research](https://research.adobe.com/publication/vecfusion-vector-font-generation-with-diffusion/)
- **VecFontSDF (CVPR 2023, arXiv rev. Apr 2025)** reconstructs and synthesizes vector fonts via signed distance functions. Output is built from **quadratic Bézier curves** (TrueType-native), and it supports few-shot style transfer. It is not diffusion-based. — [arXiv 2303.12675](https://arxiv.org/html/2303.12675v2) (via search summary; repo/license not verified)
- **VecGlypher (CVPR 2026, arXiv 2602.21461, 2026-02-25)**, Huang et al. A single multimodal LLM emits SVG path tokens for "text-referenced glyph generation (style tags/text + target character)" and "image-referenced glyph generation (reference glyph images + target character)", with "direct one-pass SVG output without raster-to-vector post-processing". Repo: https://github.com/xk-huang/VecGlypher, described as a "re-implementation". License: **Apache-2.0**. **Weights: https://huggingface.co/VecGlypher/VecGlypher-27b-it** (Gemma-3-27B-it base). Configs also exist for Qwen3-4B and Qwen3-0.6B. Data: the Google Fonts dataset is released at https://huggingface.co/datasets/VecGlypher/Google-Fonts-Dataset. The 39K-font Envato set is **not** released "for license considerations", but its processing scripts are. Character sets in the configs are a–z / Latin (a "Diacritics (Latin)" section). — [README](https://raw.githubusercontent.com/xk-huang/VecGlypher/main/README.md), [inference.md](https://raw.githubusercontent.com/xk-huang/VecGlypher/main/docs/public/inference.md), [data.md](https://raw.githubusercontent.com/xk-huang/VecGlypher/main/docs/public/data.md), [CVF PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Huang_VecGlypher_Unified_Vector_Glyph_Generation_with_Language_Models_CVPR_2026_paper.pdf)
- VecGlypher README insight: general SVG-capable LLMs "still struggle to draw valid glyphs and to follow typography style instructions" because "glyph programs are rarely seen in pretraining corpora (fonts are binary TTF/OTF)". In other words, "strong general SVG/code ability does not automatically transfer to high-fidelity typographic glyph generation without targeted data and training." — [README](https://raw.githubusercontent.com/xk-huang/VecGlypher/main/README.md)
- VecGlypher training: Stage 1 uses noisy text–glyph pairs from 39K Envato fonts, and Stage 2 uses 2.5K Google Fonts. Model variants of 4B/27B/70B are reported. A community re-implementation also exists at github.com/dandezign/VecGlypher-model. — [emergentmind summary](https://www.emergentmind.com/topics/vecglypher), [search listing](https://github.com/xk-huang/VecGlypher)
- **VecFontLLM (arXiv 2607.17251, Jul 2026)**, Hao Yuan, Yuxuan Luo, Xing Chen, Zhouhui Lian. It does few-shot Chinese vector fonts: first an anchor scaffold (layout of components and contours), then completion of Bézier control points. The paper says "code and data will be publicly available", and no repo was found. — [arXiv](https://arxiv.org/abs/2607.17251), [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7472390)
- **NIV: Neural Axis Variations (arXiv 2606.05261, 2026)**, Benedek, Shamir, Fried. It converts a static TTF into a variable font (wght, wdth, slnt, opsz axes) and is trained on more than 1M variation tuples from variable Google Fonts. Official code: https://github.com/ndvbd/NIV (`python code/ttf_to_var.py font.ttf --model ... --axes wght,wdth,slnt,opsz`). This is relevant as a post-step (adding weights to a generated Arabic font), but its Arabic coverage is unknown. — [arXiv](https://arxiv.org/abs/2606.05261) (via search summary)
- Other 2024–2026 font work is **raster-only**: HFH-Font (arXiv 2410.06488, Chinese, diffusion), DA-Font (arXiv 2509.16632), "Beyond Patches" (arXiv 2601.01593, Chinese/Korean), and DeepCalliFont (arXiv 2312.10314). "Beyond Patches" notes that vector few-shot font generation still struggles with complex glyphs and with generalizing to unseen glyphs. — [HFH-Font](https://arxiv.org/pdf/2410.06488), [DA-Font](https://arxiv.org/html/2509.16632v1), [Beyond Patches](https://arxiv.org/pdf/2601.01593)
- "Efficient and Scalable Chinese Vector Font Generation via Component Composition" (arXiv 2404.06779) is a further Chinese-only vector method. — [arXiv](https://arxiv.org/pdf/2404.06779)

**Generic SVG generators (icons, illustrations) and SVG LLMs**

- **DeepSVG (NeurIPS 2020)**. Repo: https://github.com/alexandre01/deepsvg. A pretrained **font generation** model and notebook were added in July 2020. The font data follows the SVG-VAE instructions (Latin). **No LICENSE file was found** at main/master. The README shows a GitHub license badge but no file was found, so the license is unverified. — [README](https://raw.githubusercontent.com/alexandre01/deepsvg/master/README.md)
- **SVGformer (CVPR 2023)**, Cao et al. Only the supplemental PDF was found. **No public code confirmed.** — [CVF supplemental](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Cao_SVGformer_Representation_Learning_CVPR_2023_supplemental.pdf)
- **Im2Vec (CVPR 2021)**. Repo: https://github.com/preddy5/Im2Vec. License: **Apache-2.0**. — [LICENSE](https://raw.githubusercontent.com/preddy5/Im2Vec/master/LICENSE.md). Its SVG-Fonts DinoScore in the StarVector benchmark is 0.733, the worst of all methods tested (see B). — [StarVector README](https://raw.githubusercontent.com/joanrod/star-vector/main/README.md)
- **IconShop (SIGGRAPH Asia 2023)**. Repo: https://github.com/kingnobro/IconShop. **No LICENSE file found.** Pretrained models (FIGR-SVG, 100 epochs) are on Google Drive. The README warns that "the pretrained model in this repository produces different results from those shown in the paper" because of dataset differences. It covers icons, not fonts. — [README](https://raw.githubusercontent.com/kingnobro/IconShop/main/README.md)
- **StrokeNUWA (arXiv 2401.17093)** has a stroke tokenizer, an encoder-decoder LLM (Flan-T5 3B backbone) and a post-processing fixer. Repo: https://github.com/ProjectNUWA/StrokeNUWA. No LICENSE file found. — [paper](https://arxiv.org/pdf/2401.17093), [README](https://raw.githubusercontent.com/ProjectNUWA/StrokeNUWA/main/README.md)
- **StarVector (CVPR 2025)**. Repo: https://github.com/joanrod/star-vector. License: **Apache-2.0**. Weights on HF: `starvector/starvector-1b-im2svg` and `starvector/starvector-8b-im2svg`. Datasets: SVG-Stack (2.1M) and **SVG-Fonts (1.8M samples, "Font letter")**. It does image2SVG and text2SVG. "StarVector models will not work for natural images or illustrations… They excel in vectorizing icons, logotypes, technical diagrams…" — [README](https://raw.githubusercontent.com/joanrod/star-vector/main/README.md), [LICENSE](https://raw.githubusercontent.com/joanrod/star-vector/main/LICENSE)
- **OmniSVG (NeurIPS 2025)**. Repo: https://github.com/OmniSVG/OmniSVG. Code under **Apache-2.0**, but the **MMSVG dataset is CC BY-NC-SA 4.0**. Weights: OmniSVG1.1_8B and 1.1_4B (2025-12-02), plus the earlier OmniSVG-3B (2025-07-22). Training code is at OpenVGLab/OmniSVG-train (2025-12-31). The MMSVG-Icon dataset grew 264K→904K and MMSVG-Illustration 66K→255K (2025-12-22). OmniLottie was released 2026-03-02. It targets icons and illustrations, not fonts. — [README](https://raw.githubusercontent.com/OmniSVG/OmniSVG/main/README.md)
- **LLM4SVG (CVPR 2025)**. Repo: https://github.com/ximinng/LLM4SVG. License: **MIT**. Datasets: SVGX-Core-250k and SVGX-SFT-1M on HF. As of the README, weights were still marked "Link for weights pending". — [README](https://raw.githubusercontent.com/ximinng/LLM4SVG/master/README.md)
- **Chat2SVG (CVPR 2025)**. Repo: https://github.com/kingnobro/Chat2SVG. No LICENSE file found. It is an LLM plus diffusion pipeline that uses googlefonts/picosvg for SVG cleaning. — [README](https://raw.githubusercontent.com/kingnobro/Chat2SVG/main/README.md)
- **NeuralSVG (2025)**. Repo: https://github.com/SagiPolaczek/NeuralSVG. License: **MIT**. — [LICENSE](https://raw.githubusercontent.com/SagiPolaczek/NeuralSVG/main/LICENSE)

**Optimization-based text-to-SVG and semantic typography (diffvg plus score distillation)**

- **VectorFusion (CVPR 2023).** No official repo was verified. An implementation is in **PyTorch-SVGRender** (https://github.com/ximinng/PyTorch-SVGRender, **MPL-2.0**), which also implements DiffVG, LIVE, CLIPasso, DiffSketcher, Word-As-Image and SVGDreamer, and is "heavily based on DiffVG". — [README](https://raw.githubusercontent.com/ximinng/PyTorch-SVGRender/main/README.md), [LICENSE](https://raw.githubusercontent.com/ximinng/PyTorch-SVGRender/main/LICENSE)
- **SVGDreamer (CVPR 2024)**. Repo: https://github.com/ximinng/SVGDreamer. License: **MIT** (© 2024–2025 XiMing Xing). — [LICENSE](https://raw.githubusercontent.com/ximinng/SVGDreamer/main/LICENSE)
- **Word-As-Image (SIGGRAPH 2023)**. Repo: https://github.com/Shiriluz/Word-As-Image. License: **CC BY-NC-SA 4.0 (non-commercial)**. It optimizes one letter's outline from an existing TTF (`--font`, `--optimized_letter`) toward a semantic concept, using Stable Diffusion with diffvg. The default experiment uses conformal and tone losses ("conformal_0.5_dist_pixel_100_kernel201"). An HF demo exists. — [README](https://raw.githubusercontent.com/Shiriluz/Word-As-Image/main/README.md), [LICENSE](https://raw.githubusercontent.com/Shiriluz/Word-As-Image/main/LICENSE)
- **DS-Fusion (ICCV 2023)**. Repo: https://github.com/tmaham/DS-Fusion. License: **MIT**. It is a latent-diffusion fine-tune that outputs **raster** glyph images (txt2img 256×256), so vectorization is needed afterward. Only capital letters are supported for the bundled fonts, and `--custom_font` accepts any system font. — [README](https://raw.githubusercontent.com/tmaham/DS-Fusion/main/README.md), [LICENSE](https://raw.githubusercontent.com/tmaham/DS-Fusion/main/LICENSE.txt)
- **Khattat (arXiv 2410.03748, 2024)** is the only Arabic-focused semantic-typography item found. It morphs glyphs for Arabic and English words. Per search summary, the authors note that letter-level approaches "restrict variability and are unsuitable for cursive languages such as Arabic, Persian, and Urdu". — [arXiv](https://arxiv.org/pdf/2410.03748) (via search summary; details not verified)

**Arabic support**

- Searches for 2025–2026 Arabic vector font generation returned **no papers**. Results were Chinese/Latin vector work, DuwatBench (an Arabic calligraphy *understanding* benchmark, arXiv 2601.19898) and older handwriting GANs. — [search results incl. DuwatBench](https://arxiv.org/pdf/2601.19898)

### Inferences
- **The Arabic gap is total for direct vector generators.** Training sets are SVG-VAE Latin (DeepVecFont, DualVector, DeepSVG), Founder Chinese plus English (DeepVecFont-v2), Google Fonts and Envato Latin (VecGlypher), or Chinese (VecFontLLM). Retraining on Arabic Google Fonts (OFL) is feasible for DeepVecFont-v2 (customize-dataset pipeline, tunable MAX_SEQ_LEN) and for VecGlypher (data-build scripts plus Google Fonts processing). Arabic adds positional forms (isolated, initial, medial, final), dots and diacritics as separate components, and kashida/joining constraints that none of these models handle.
- **Sequence-length limits matter.** DeepVecFont-v2 uses MAX_SEQ_LEN of 50/70 commands. Ornate "bubbly" Arabic glyphs with counters and dots may exceed this, while component-based or LLM token approaches (VecGlypher) scale better.
- **One-word input is an extreme few-shot setting.** "مُزن" gives roughly 3–4 letter shapes in specific joining forms (م initial, ز final or isolated, ن final) plus a damma. Few-shot font models expect 4–8 isolated reference glyphs of the *same* script they were trained on. An image-referenced LLM such as VecGlypher is the closest architectural fit, but it would need Arabic fine-tuning.
- **For commercial font output, licensing filters options.** MIT/Apache options: DeepVecFont(-v2) code, DualVector, VecGlypher, StarVector, OmniSVG code, SVGDreamer, LLM4SVG, Im2Vec, LIVE (but see B). Non-commercial: Word-As-Image (CC BY-NC-SA), the OmniSVG MMSVG data (NC), and DeepVecFont-v2's Founder Chinese fonts. Repos with no license file (DeepSVG, IconShop, StrokeNUWA, Chat2SVG) default to "all rights reserved" unless a license is stated elsewhere.
- **Practical recommendation (inference).** A hybrid is likely more robust today than pure vector generation: a raster few-shot or diffusion style model fine-tuned on Arabic, then high-resolution tracing (see B and C). Optimization-based refinement (diffvg, Word-As-Image-style losses) can then nudge outlines. DS-Fusion is also raster-only, which supports this hybrid view.
- **Star counts (unverified, from model prior knowledge, not checked in this session; treat as rough order of magnitude only).** vtracer, StarVector and OmniSVG are in the thousands; diffvg, DeepSVG, Word-As-Image, DeepVecFont and LIVE are high hundreds to low thousands; DualVector, DeepVecFont-v2 and IconShop are low hundreds. Do not cite these as facts.

### Gaps
- Star counts for every repo (GitHub API and HTML blocked in this session).
- VecFusion code and weights status, and its training scripts (project page unreachable).
- VecFontSDF repo URL and license, and SVGformer code.
- Whether VecGlypher's released 27B weights include image-referenced mode and any non-Latin data. The released config names suggest a–z Latin only.
- Whether DualVector outputs quadratic or cubic curves.
- Khattat's code, license and output format, which could not be verified.
- No quantitative contour-cleanliness numbers (node counts per glyph) were found for any generative model.

## (B) Raster-to-vector conversion tools for glyphs

### Takeaway
For single-colour glyph images, the classical tracers (potrace, VTracer, AutoTrace) still match or beat learned methods on fonts. On the StarVector SVG-Fonts benchmark (DinoScore): Potrace 0.967, VTracer 0.964, LIVE 0.956, AutoTrace 0.954, DiffVG 0.821, Im2Vec 0.733. StarVector-8B is best at 0.982. Potrace is the best choice for glyphs because its `alphamax` parameter controls corners explicitly: a low value keeps the angular style, a high value gives the bubbly style. It has a pure-Python port (`potracer`), but it is GPL. VTracer 1.0 (MIT, Rust with Python bindings) is the permissive alternative and adds binary/adaptive thresholding and paper.js-style `--simplify`. All tracers output cubic or line segments, so conversion to TrueType quadratics is done afterward with fontTools cu2qu.

### Cited Findings

**Benchmark on fonts (StarVector SVG-Bench, DinoScore, Image-to-SVG)**

| Method | SVG-Fonts | SVG-Icons |
|---|---|---|
| AutoTrace | 0.954 | 0.946 |
| Potrace | 0.967 | 0.972 |
| VTracer | 0.964 | 0.940 |
| Im2Vec | 0.733 | 0.754 |
| LIVE | 0.956 | 0.959 |
| DiffVG | 0.821 | 0.952 |
| GPT-4-V | 0.842 | 0.848 |
| StarVector-1B | 0.978 | 0.975 |
| StarVector-8B | 0.982 | 0.984 |

— [StarVector README](https://raw.githubusercontent.com/joanrod/star-vector/main/README.md)

**Potrace**

- **Potrace** (Peter Selinger): http://potrace.sourceforge.net/. It is GPL (the pure-Python port is GPL-2.0) and very mature (algorithm paper 2003, port based on 1.16).
  - `turdsize` (despeckle) defaults to 2.
  - `alphamax` (corner threshold) defaults to 1. "The smaller this value, the more sharp corners will be produced". At 0 "no smoothing… output is a polygon", and above 4/3 "all corners are suppressed and the output is completely smooth".
  - `opttolerance` defaults to 0.2. "Larger values allow more consecutive Bezier curve segments to be joined… at the expense of accuracy". Curve optimization can be disabled, which gives larger files.
  - Turn policy defaults to "minority".

  — [potracelib API PDF](https://potrace.sourceforge.net/potracelib.pdf), [man page](https://potrace.sourceforge.net/potrace.1.html), [algorithm paper](https://potrace.sourceforge.net/potrace.pdf), [pypotrace ref](https://pythonhosted.org/pypotrace/ref.html)
- **Python options for potrace.** `potracer` is a pure-Python port of Potrace 1.16 (https://github.com/tatarize/potrace, **GPL-2.0**, `pip install potracer`). It exists because "pypotrace have installation issues and compilation problems". — [README](https://raw.githubusercontent.com/tatarize/potrace/main/README.md), [LICENSE](https://raw.githubusercontent.com/tatarize/potrace/main/LICENSE)

**AutoTrace**

- **AutoTrace**: https://github.com/autotrace/autotrace. License: **GPL-2.0** (COPYING). It supports "Outline and centerline tracing", and installers exist for brew, port, dnf, pkg and Windows. — [README](https://raw.githubusercontent.com/autotrace/autotrace/master/README.md), [COPYING](https://raw.githubusercontent.com/autotrace/autotrace/master/COPYING)
- FontForge context: a FontForge issue thread recommends Potrace over AutoTrace and says AutoTrace "has been dropped from most Linux distributions due to lack of maintenance". (The repo now has active installers, so maintenance appears revived; this conflict is noted.) FontForge's trace needs the AUTOTRACE env var or the "prefer potrace" preference. — [fontforge issue #2652](https://github.com/fontforge/fontforge/issues/2652), [scruss blog](https://scruss.com/blog/2010/05/09/creating-a-truetype-font-from-your-handwriting-with-your-scanner-your-printer-and-fontforge/) (via search summary)

**VTracer**

- **VTracer 1.0** (visioncortex): https://github.com/visioncortex/vtracer. License: **MIT** (© 2024 Tsang Hao Fung). It is Rust and is published as the `vtracer` Python package on PyPI (pyo3; the README shows `pip install vtracer==1.0.0a4`), plus a JS package, a CLI and a desktop app.
  - Modes: `pixel`, `polygon`, `spline`.
  - `--simplify <tolerance>`: "fewest cubics within this tolerance in px (try 1–2.5)", "typically halving file size".
  - Binary `--threshold` and Bradley–Roth `--adaptive` thresholding, `--filter-speckle`, and `--hierarchical cutout` (seam-free).
  - Hidden spline flags: corner-threshold default 60°, segment-length 4, splice-threshold 45°.
  - "VTracer skips Potrace's expensive optimal-polygon search in favor of a fast, linear pipeline", and its output is "much more compact" than Illustrator Image Trace.

  — [README](https://raw.githubusercontent.com/visioncortex/vtracer/master/README.md), [LICENSE](https://raw.githubusercontent.com/visioncortex/vtracer/master/LICENSE)

**FontForge and Inkscape**

- **FontForge autotrace** is a front-end to potrace/autotrace. FontForge is GPL. FontForge provides "Add Extrema" (it may drop a near-coincident point to avoid tiny splines) and validation for "non-integral coordinates, missing points at extrema, self-intersecting contours". Autotraced outlines "can be messy, with too many points". — [FontForge Element menu](https://fontforge.org/archive/elementmenu.html), [FontForge LICENSE](https://raw.githubusercontent.com/fontforge/fontforge/master/LICENSE), [TUGboat FontForge article](https://www.tug.org/tugboat/tb24-3/williams.pdf)
- **Inkscape Trace Bitmap** embeds potrace (the parameters are potrace's), so it is equivalent to potrace for glyphs. — inferred from the potrace site listing; see Gaps.

**Learned and optimization vectorizers**

- **DiffVG (SIGGRAPH 2020)**: https://github.com/BachiLi/diffvg. License: **Apache-2.0**. It is a differentiable rasterizer and the base for LIVE, Word-As-Image, VectorFusion, SVGDreamer and DualVector. It is built from source with CMake. — [README](https://raw.githubusercontent.com/BachiLi/diffvg/master/README.md), [LICENSE](https://raw.githubusercontent.com/BachiLi/diffvg/master/LICENSE)
- **LIVE (CVPR 2022)**: official repo https://github.com/Picsart-AI-Research/LIVE-Layerwise-Image-Vectorization; backup at https://github.com/ma-xu/LIVE. "LIVE is under the Apache-2.0 license. Please contact the authors for commercial use." It reconstructs an image with as few as 5 paths where DiffVG needs 256. It is slow (optimization per image) and needs Python 3.7 with diffvg. — [README](https://raw.githubusercontent.com/ma-xu/LIVE/main/README.md)
- **StarVector** (see A) is the top learned vectorizer on fonts (0.982). It is an LLM emitting SVG code (Apache-2.0, 1B/8B weights).
- **LayerTracer (2025)**: https://github.com/showlab/LayerTracer. License: **MIT**. It uses FLUX LoRA checkpoints for layered icon or emoji generation and vectorization, so it is aimed at icons. — [README](https://raw.githubusercontent.com/showlab/LayerTracer/main/README.md), [LICENSE](https://raw.githubusercontent.com/showlab/LayerTracer/main/LICENSE)
- **SAMVG** appears only as a comparison row in the StarVector paper, and **SuperSVG** (CVPR 2024) returned no repo (`intchous/SuperSVG` has no README or LICENSE at main/master). Neither has verified code. — [StarVector paper](https://arxiv.org/pdf/2312.11556)
- **picosvg** (googlefonts) is used by Chat2SVG for SVG cleanup. It is useful for flattening transforms and strokes before font import. — [Chat2SVG README](https://raw.githubusercontent.com/kingnobro/Chat2SVG/main/README.md)

### Inferences
- **Angular style (e.g. geometric "مزن").** Use potrace with `alphamax` around 0–0.6 (0 gives a pure polygon), or VTracer `polygon` mode. Polygon output maps losslessly to TrueType (line segments only).
- **Bubbly style.** Use potrace with `alphamax` around 1.0–1.33 and `opttolerance` 0.2–0.5, or VTracer `spline` with `--simplify` 1–2.5 px. Then convert cubic to quadratic.
- **Mixed styles.** A per-glyph or per-region corner decision (from the learned style) can be passed to potrace by picking `alphamax`.
- **Learned vectorizers are not worth it for glyph tracing here.** Their gains over potrace on SVG-Fonts are small (0.982 vs 0.967), they are trained on Latin font data, they are heavy (8B), and as LLMs they can hallucinate topology. Classical tracers are deterministic and topology-faithful at high resolution.
- **DiffVG/LIVE-style optimization is better suited to *refinement*.** For example, fit a fixed, small number of quadratic segments to a traced contour against the raster, which minimizes nodes while preserving shape.
- **License choice for a commercial product.** Prefer VTracer (MIT) or a subprocess call to the potrace binary to avoid linking GPL code. `potracer` and AutoTrace are GPL-2.0.

### Gaps
- Inkscape trace and SAMVG/SuperSVG licenses and code were not verified (Inkscape site not fetched).
- No independent measurements of node counts per glyph for potrace vs VTracer vs StarVector were found.
- No type-design forum (TypeDrawers) thread directly comparing tracer quality was found. The search returned only FontForge docs and blogs.
- Whether any tracer outputs native quadratic curves: none found. Potrace and VTracer emit cubic Béziers and lines.

## (C) Best practice for raster-to-font pipelines

### Takeaway
Generate or trace at high resolution (at least 1000 px per em; see the inference below). Threshold to binary, despeckle, and trace with a corner-aware tracer. Simplify, add extrema, and round to integer font units. Convert cubic to quadratic with fontTools cu2qu at max_err ≈ UPM/1000, and set contour direction (clockwise outer for TrueType, counter-clockwise for CFF). Build with fontmake/ufo2ft, then validate.

### Cited Findings
- **cu2qu (in fontTools; MIT).** `Cu2QuPen` `max_err` is "the maximum approximation error in font units", and the docs suggest "a value near units-per-em divided by 1000". The standalone cu2qu default is **0.001 em**. Converting several masters together preserves interpolation compatibility. The `reverse_direction` option flips contour direction. — [cu2quPen docs](https://fonttools.readthedocs.io/en/latest/pens/cu2quPen.html), [cu2qu docs](https://fonttools.readthedocs.io/en/stable/cu2qu/index.html), [fontTools LICENSE](https://raw.githubusercontent.com/fonttools/fonttools/main/LICENSE)
- **ufo2ft.** The `cubicToQuadratic` filter (options `conversionError`, `reverseDirection`, `rememberCurveType`) runs by default when compiling TrueType-flavored OpenType. "Cubic (PostScript-style) outlines are usually counter-clockwise, quadratic (TrueType-style) outlines are usually clockwise." The UFO lib key `com.github.googlei18n.cu2qu.curve_type = quadratic` skips conversion for already-quadratic outlines. — [ufo2ft README](https://github.com/googlefonts/ufo2ft/blob/main/README.rst) (via search summary)
- **Extrema.** Both TrueType and Type 1 expect on-curve points at horizontal and vertical extremes, and FontForge's "Add Extrema" adds them. — [FontForge Element menu](https://fontforge.org/archive/elementmenu.html)
- **Validation targets.** Check for non-integral coordinates, missing extrema and self-intersecting contours (FontForge validation). — [FontForge docs](https://fontforge.org/archive/elementmenu.html)
- **Potrace knobs to tune.** turdsize (2), alphamax (1.0), opttolerance (0.2) and turnpolicy (minority). — [potracelib](https://potrace.sourceforge.net/potracelib.pdf)
- **VTracer knobs to tune.** `--simplify` 1–2.5 px, `--threshold` / `--adaptive` for uneven scans, and `--filter-speckle`. — [VTracer README](https://raw.githubusercontent.com/visioncortex/vtracer/master/README.md)
- **DeepVecFont training resolution.** It moved from 64 to 128 px ("should give better results"). DS-Fusion generates at 256 px. Both are far below the resolution needed for tracing, so a super-resolution or upscaling step is needed before tracing. — [DeepVecFont README](https://raw.githubusercontent.com/yizhiwang96/deepvecfont/master/README.md), [DS-Fusion README](https://raw.githubusercontent.com/tmaham/DS-Fusion/main/README.md)

### Inferences
These are recommended pipeline steps, derived from the sources above plus standard type-production practice. Items not directly sourced are marked.

1. **Resolution.** Rasterize or generate each glyph at 1024–2048 px per em (e.g. UPM 1000 at 1–2 px per unit). Potrace error tolerances are in pixels (opttolerance 0.2 px), so at 1 px/unit the tracing error is about 0.2 units, well under UPM/1000 = 1 unit. If a generator outputs 128–256 px, upscale it (Real-ESRGAN-type, or diffusion upscaling) before thresholding. Tracing 128 px directly gives wobbly outlines with too many nodes. *(Inference; no single source.)*
2. **Pre-processing.** Apply a light Gaussian blur (σ ≈ 0.5–1 px at 1k resolution), then a threshold at 50% (or VTracer adaptive), then morphological open/close to remove specks. Use turdsize or filter-speckle to drop islands, but protect Arabic dots (nuqat) and diacritics: set the speckle threshold well below the dot area. *(Inference; the dot-protection point is Arabic-specific.)*
3. **Tracing per style.**
   - Angular: potrace `alphamax` 0–0.5, or VTracer polygon.
   - Rounded: potrace `alphamax` 1.0–1.33, `opttolerance` 0.2–0.4.
4. **Post-processing.** Simplify (paper.js-style or Ramer–Douglas–Peucker on near-collinear points). Add extrema (FontForge `addExtrema` or the booleanOperations/skia-pathops pipeline). Remove overlaps (skia-pathops). Round to integers. Make tangents at smooth points collinear, and snap near-horizontal or near-vertical handles. *(Inference; standard practice.)*
5. **Quadratic conversion and direction.** Use cu2qu with max_err ≈ UPM/1000, reverse direction to clockwise outer for glyf, and keep cubic outlines for CFF/OTF if you are shipping CFF. Let ufo2ft/fontmake handle both.
6. **Topology checks.** Ensure counters have the opposite direction to outer contours, there are no self-intersections, and start points are consistent (matters for variable or interpolated masters, e.g. a NIV post-step).
7. **Arabic-specific.** Trace dots and diacritics as separate contours or components so they can be reused as OpenType components and anchored (mark/mkmk). Keep baseline-joining stroke ends flat and exactly at the join height, so connections are seamless across positional forms. Snap the join-stroke y-coordinates after tracing. *(Inference.)*

### Gaps
- No published, sourced numeric recommendation (e.g. "trace at N px/em") from type-design community discussions was found. TypeDrawers and Glyphs forum threads could not be fetched in this session.
- Google Fonts / fontbakery outline-check specifics (e.g. smooth-point, semi-vertical and colinear-vector checks) were not retrieved.
- No source evaluated tracers on Arabic glyphs specifically.
