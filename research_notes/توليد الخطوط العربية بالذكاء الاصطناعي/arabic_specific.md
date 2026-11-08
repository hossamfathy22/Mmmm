# Arabic-Specific Work on AI Font/Calligraphy Generation, Glyph Segmentation, and Datasets (as of Nov 2026)

Scope note: research done on 2026-11-08. In this session arxiv.org, kraken.re, learn.microsoft.com and fonts.google.com could not be fetched directly (network policy). Search-engine snippets of arXiv papers were used, and GitHub pages and the GitHub code-search API were fetched directly. Claims taken only from snippets are marked as such. Facts about the Unicode/OpenType standards come from the standard reference documents, which I cite by URL but could not re-fetch this session.

Running example used in the inferences: the target word مُزن = م (initial form) + ز (final form; ز does not join to the left) + ن (**isolated** form, because ز breaks the join) + damma over م + one dot on ز + one dot on ن.

---

## Q1. Arabic font generation, calligraphy generation, and handwriting synthesis: papers and repos (link, license, weights, quality, inputs)

### Takeaway
No public, Arabic-specific few-shot or one-shot **font** (full glyph set → OpenType) generator with released weights turned up. Arabic generative work falls into three groups: (a) small GAN calligraphy/style-transfer projects (Pix2Pix, CycleGAN, DCGAN, StyleGAN2-ada) with tiny datasets and mostly no weights; (b) one Arabic port of a one-shot handwriting **diffusion** model (Arabic-One-DM), which works at word level and handles contextual forms through a Unifont-rendered content encoder; (c) multilingual text-image work where Arabic is one script among several. The full-font, Arabic, one-word-reference problem looks open.

### Cited Findings

**Arabic calligraphy GANs**
- *Arabic Calligraphy Generation Through Image-to-Image Translation Using GANs* (Hadj Azzem, Moussaoui, Berrimi; 2023 EICEEAI, IEEE, DOI 10.1109/EICEEAI.2023.10590292). It uses Pix2Pix and CycleGAN to translate conventional-font images (e.g., Arial) into **Reqaa (Ruq'ah), Diwani and Farsi** calligraphy. The authors release an **aligned** font→calligraphy paired dataset. Repo license: **MIT**. No pretrained weights are listed, results are called "promising" with sample images only, and the README gives no quantitative scores (FID appears only as a repo topic). — [GitHub repo](https://github.com/Yousra-Chahinez/generate_Arabic_calligraphy_GANs); [ResearchGate](https://www.researchgate.net/publication/382379955_Arabic_Calligraphy_Generation_Through_Image-to-Image_Translation_Using_Generative_Adversarial_Networks_GANs)
- *Unveiling New Artistic Dimensions in Calligraphic Arabic Script with GANs* (ACM, 2024, DOI 10.1145/3664210). It applies StyleGAN2-ada to **Nastaliq**, introduces custom datasets **Nas4-60k** and **Nas4-60k-aug**, and explores latent-space interpolation. This is unconditional style generation, not text-conditioned glyph generation. — [ACM DL](https://dl.acm.org/doi/10.1145/3664210)
- A comparative study of **CycleGAN, Pix2pix and DCGAN** for Arabic calligraphy generation and style transfer (2021 3ICT conference, DOI 10.1109/3ICT53449.2021.9581388). — [search snippet via ResearchGate/Semantic Scholar results](https://www.semanticscholar.org/paper/Generative-Adversarial-Networks-(GAN)-for-Arabic-Ahmed-Ali/6c2e50249eefc5557c1be7ed048c8fd60956e507)
- *Generative Adversarial Networks (GAN) for Arabic Calligraphy* (Ahmed, Ali et al.). It uses DCGAN to generate Arabic calligraphy **letters**, and Chebouat did similar DCGAN letter generation. — [Semantic Scholar](https://www.semanticscholar.org/paper/Generative-Adversarial-Networks-(GAN)-for-Arabic-Ahmed-Ali/6c2e50249eefc5557c1be7ed048c8fd60956e507); cited in [IJACSA 2025 Vol.16 No.3 Paper 81](https://thesai.org/Downloads/Volume16No3/Paper_81-Handwritten_Arabic_Calligraphy_Generation.pdf)
- A university thesis, "Generating Arabic Letters using Generative Adversarial ..." (Univ. Ouargla, Algeria), does GAN generation of isolated Arabic letters. — [PDF](https://dspace.univ-ouargla.dz/jspui/bitstream/123456789/21930/1/Generating%20Arabic%20Letters%20using%20Generative.pdf)
- *Handwritten Arabic Calligraphy Generation* (IJACSA 2025, Vol.16 No.3). It notes that **KHATT "primarily focuses on non-artistic Arabic text," which limits it for calligraphy generation** that needs stylized features. — [IJACSA PDF](https://thesai.org/Downloads/Volume16No3/Paper_81-Handwritten_Arabic_Calligraphy_Generation.pdf)

**Arabic handwriting synthesis (diffusion)**
- **Arabic-One-DM**: *"One Stroke, One Shot: Diffusing a New Era in Arabic Handwriting Generation"* (Abushahla, Panopio, Elfattal, Zualkernan; IEEE, BibTeX key says 2025). It adapts One-DM (a one-shot diffusion mimicker built for English, Chinese and Japanese) to Arabic:
  - Architecture: ResNet-18 style/content feature extractors, an OCR/HTR model, and a Stable Diffusion v1-5 VAE.
  - Contextual forms: a **GNU Unifont module renders the isol/init/medi/fina forms of each letter**, with a dummy letter to force joining, a joining heuristic for non-joining letters, `arabic_reshaper` + `bidi`, and a custom `shape_arabic_text`. The module can also render letters with harakat.
  - Pretraining data: **Khat²**, synthetic word images rendered in many fonts and inspired by Font², with generation scripts in a separate Khat_Squared repo.
  - Real data: IFN/ENIT + AlexU-Word + AHAWP (words) + IESK-ArDB, about **62,437 words from about 1,417 writers**. KHATT is not used.
  - Input: one reference word image plus the target text.
  - Release status: the README states **no license**, gives **no download links for the trained weights**, and reports no quantitative results (evaluation scripts only).
  — [GitHub](https://github.com/7abushahla/Arabic-One-DM)
- A US patent, "Method for synthesizing Arabic handwritten text" (US 10185882), describes concatenation-based synthesis as a route to open-vocabulary Arabic handwriting. — [USPTO PDF](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10185882)
- A 2025 systematic review of HTR data augmentation and generation notes that Arabic generation must handle **diacritics**, and lists IFN/ENIT among the Arabic datasets. — [arXiv 2507.06275](https://arxiv.org/html/2507.06275v1)
- Recent general handwriting generators either don't cover Arabic or I could not confirm Arabic coverage:
  - DiffBrush, text-line diffusion, Aug 2025 — [arXiv 2508.03256](https://arxiv.org/abs/2508.03256)
  - Layout Stroke Imitation, stroke-level diffusion for calligraphic style imitation, 2025 — [arXiv 2509.15678](https://arxiv.org/abs/2509.15678)
  - DiffInk, online handwriting latent-diffusion transformer, Sep 2025 — [arXiv 2509.23624](https://arxiv.org/abs/2509.23624)
  - Autoregressive styled text image generation (Emuru line), Oct 2025 — [arXiv 2510.23240](https://arxiv.org/pdf/2510.23240)
  - HandwritingAgent, SVG handwriting synthesis, 2026 — [arXiv 2606.18788](https://arxiv.org/pdf/2606.18788)

**Arabic in multilingual / other generation work**
- *Text Image Generation for Low-Resource Languages with Dual Translation Learning* (arXiv 2409.17747, 2024). It is a diffusion model conditioned on "synthetic" vs "real" states, and its authors **collected 64 Arabic font files** for synthetic rendering. — [arXiv HTML](https://arxiv.org/html/2409.17747v1)
- *Application of deep learning on font design for ethnic minority writing systems* (PeerJ CS, 2025). It uses MF-net for one-time style transfer of a full script's glyphs, with a diffusion model (DM) refining style and clarity. Arabic is mentioned as one of the dataset's scripts, but the excerpt gives no detail. — [PeerJ](https://peerj.com/articles/cs-3340/)
- *Style Transfer as Bias Mitigation: Diffusion Models for Synthetic Mental Health Text for Arabic* (arXiv 2601.14124) is NLP text style transfer, not visual. Listed only to prevent confusion. — [arXiv](https://arxiv.org/pdf/2601.14124)
- DRG-Font (arXiv 2604.13797, 2026), a few-shot font generation method, cites an **Arabic character skeletonization** paper in its references. Arabic results are not visible in the snippet. — [arXiv](https://arxiv.org/pdf/2604.13797)

**Arabic calligraphy recognition (relevant as an evaluator or segmenter)**
- **DuwatBench** (MBZUAI, EACL 2026 main; released 22 Jan 2026; on HF 23 Jan 2026; arXiv 2601.19898):
  - Contents: 1,272 curated calligraphy samples with more than 9.5k word instances (about 1,475 unique words).
  - Styles: Thuluth, Diwani, Naskh, Kufic, Ruq'ah, Nasta'liq. Per a search snippet, Kufic has 83 samples and Diwani 230.
  - Annotations: **bounding boxes**, transcriptions, style and theme labels.
  - License: code Apache-2.0; images are "under their respective licenses" (no single data license).
  - Results: the best model, Gemini-2.5-flash, reached only **0.4167 exact match**. The open models evaluated include EasyOCR, TrOCR-Arabic, Qwen2.5-VL and AIN.
  — [GitHub](https://github.com/mbzuai-oryx/DuwatBench); [project page](https://mbzuai-oryx.github.io/DuwatBench); [arXiv](https://arxiv.org/pdf/2601.19898)
- **Nassq OCR v3**, an Arabic calligraphy transcription model on HF. — [HF model](https://huggingface.co/AyaEhab258/NASAQ4.1)

### Inferences
- The closest reusable Arabic component is Arabic-One-DM's **content-encoding trick**: render each target contextual form with a plain reference font (Unifont) and feed that as the content condition. This maps directly onto font generation: render each target glyph form with a neutral Arabic font (e.g., Noto Naskh or Noto Kufi) as the content image, then style it with the user's reference. The same approach (content glyph image + style reference) is used by FontDiffuser, Diff-Font and the like, and those would need Arabic training data.
- The existing GAN calligraphy work targets traditional scripts (Ruq'ah, Diwani, Nastaliq, Farsi), not **display/logo lettering** (bubbly, geometric, fragmented). None of it is trained on modern Arabic display fonts, so a new model would need training on display-style Arabic fonts (see Q5).
- Since the reference word مزن decomposes into م-init + ز-fina + ن-isol, a one-word reference gives the model only **3 of roughly 100+ contextual letter shapes**, plus 2 dots and 1 damma. This is far below even "few-shot" Chinese settings, which typically use 4–16 reference glyphs. The style representation therefore has to be learned from stroke-level traits (terminal shape, stroke contrast, corner rounding, fragmentation pattern), not copied from matching glyphs.
- DuwatBench's 0.42 exact-match ceiling for frontier VLMs on calligraphy suggests that even reading stylized Arabic is unreliable. Auto-captioning or OCR-checking generated stylized glyphs (e.g., for an OCR-consistency loss) will be noisy for heavy display styles.

### Gaps
- I found no Arabic paper that generates a complete OpenType font, with all positional forms and GSUB, from one or a few references. Nor did I find "AraGAN"-named font work. "AraGAN" search hits point to unrelated Arabic NLP models; not further verified.
- No license or weights were found for Arabic-One-DM, and the venue and date are unconfirmed.
- I found no Arabic-specific diffusion calligraphy generator with released weights.
- Muharaf- or KHATT-based **synthesis** papers did not surface; KHATT and Muharaf appear to be used almost only for recognition.

---

## Q2. Multilingual few-shot font models that include Arabic in training

### Takeaway
The mainstream few-shot and one-shot font generators are all trained and evaluated on CJK and/or Latin, and none was confirmed to include Arabic:
- MX-Font and MX-Font++
- FTransGAN
- MA-Font
- Diff-Font
- FontDiffuser
- the ViT one-shot multilingual model
- VecGlypher (CVPR 2026)

Cross-lingual transfer, such as a Latin reference styling another script, has been shown for CJK↔Latin only. Arabic would need fine-tuning or retraining on Arabic font data.

### Cited Findings
- **FTransGAN** (WACV 2021), *Few-shot Font Style Transfer between Different Languages*: end-to-end cross-language few-shot style transfer, with a large multi-language dataset built by the authors (Chinese↔English). — [CVF PDF](https://openaccess.thecvf.com/content/WACV2021/papers/Li_Few-Shot_Font_Style_Transfer_Between_Different_Languages_WACV_2021_paper.pdf)
- **MX-Font** (ICCV 2021): a multi-headed encoder with weak local-component supervision, giving large gains in zero-shot **cross-lingual** transfer. — [arXiv 2104.00887](https://arxiv.org/pdf/2104.00887). **MX-Font++** (2025) reports strong results on low-resource languages. — [arXiv 2503.02799](https://arxiv.org/pdf/2503.02799)
- **MA-Font** (IEEE 2024): cross-language transfer; one experiment generates four languages in one style using **Latin letters as the style reference**. — [IEEE](https://ieeexplore.ieee.org/abstract/document/10509670/)
- **One-Shot Multilingual Font Generation via ViT** (arXiv 2412.11342): tested with Chinese content and Japanese-hiragana style. — [arXiv](https://arxiv.org/pdf/2412.11342)
- **GC-Font** (BMVC 2025): Chinese few-shot generation; multilingual extension is listed as future work. — [BMVC PDF](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_38/paper.pdf)
- **Diff-Font** (arXiv 2212.05895): one-shot diffusion font generation evaluated on Chinese and Korean. — [arXiv](https://arxiv.org/abs/2212.05895). **FontDiffuser** (AAAI 2024): one-shot diffusion with multi-scale content aggregation and style contrastive learning, with no Arabic evaluation found. — [arXiv 2312.12142](https://arxiv.org/pdf/2312.12142)
- **VecGlypher** (Meta AI + UCSC; arXiv 2602.21461, Feb 2026; CVPR 2026): a multimodal LLM that emits **SVG glyph outlines** from a text description or reference glyph images.
  - Training stage 1: about 39,497 noisy Envato fonts.
  - Training stage 2: post-training on about **2,497 Google Fonts (157,899 glyph instances)** with tags and exemplars.
  - Arabic coverage: not stated in the snippets. A secondary summary lists "broader Unicode support and underrepresented script blocks" as future work (unverified).
  — [project page](https://xk-huang.github.io/VecGlypher); [arXiv](https://arxiv.org/abs/2602.21461); [CVPR 2026 poster](https://cvpr.thecvf.com/virtual/2026/poster/38733)
- **VecFusion** (vector font diffusion) evaluated on a dataset of 577 distinct Unicode glyphs across "various languages"; no Arabic-specific results were found. — [arXiv 2312.10540](https://arxiv.org/pdf/2312.10540)
- Other recent artistic-glyph work found (Arabic coverage not shown in the snippets):
  - FontCrafter, element-driven artistic fonts, 2026 — [arXiv 2603.22054](https://arxiv.org/pdf/2603.22054)
  - ArtGlyphDiffuser, Pattern Recognition 2025 — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0031320325008337)
  - QT-Font, quadtree diffusion, SIGGRAPH 2024 — [ACM](https://dl.acm.org/doi/10.1145/3641519.3657451)
  - GRIF-DM, impression fonts — [arXiv 2408.07259](https://arxiv.org/pdf/2408.07259)

### Inferences
- Component-based models such as MX-Font and LF-Font rely on component or radical decomposition designed for Hangul and Hanzi. Arabic has a natural analogue: rasm skeleton + dots + marks (see Q4). That analogue could supply the "weak component labels" that MX-Font-style training needs.
- VecGlypher's Google Fonts stage almost certainly includes some families with an Arabic subset, since roughly 62 Google Fonts families declare one (Q5). Whether Arabic glyphs were actually used is unverified. If they were, it would be the only large model with any Arabic vector-outline exposure.
- Cross-script transfer is weak evidence for usability: the reference here is **Arabic**, so cross-lingual transfer is not needed. The real need is Arabic→Arabic transfer from extremely few, non-matching characters.

### Gaps
- I could not open the VecGlypher, FontDiffuser or MX-Font++ full texts (arxiv.org blocked) to check for Arabic in their training sets.
- I found no evidence that commercial or closed systems with Arabic font generation have public methods or weights.

---

## Q3. Segmenting a connected Arabic word image into letter shapes: classical methods, learning methods, and OCR engines with character boxes

### Takeaway
Arabic OCR has moved to **segmentation-free** line or word recognition (CTC, seq2seq, VLMs) because explicit character segmentation of cursive Arabic is error-prone: overlapping ligatures, vertical stacking and ambiguous join points cause over- and under-segmentation. Of the open engines:
- **Kraken** explicitly claims character-level bounding boxes, derived from CTC alignment.
- Tesseract and Surya are evaluated mostly at line level.
- VLM-based Arabic OCR (Qari-OCR, Baseer) returns text only.

For stylized display lettering, the practical route is to exploit the PAW (piece-of-Arabic-word) structure: connected components plus dots and marks as separate components. Join points inside each PAW then need cutting, done semi-automatically or with a human in the loop.

### Cited Findings

**Classical methods and why they struggle**
- Vertical overlap caused by ligatures means **segmenting a word along a single line cannot achieve perfect segmentation** (Naskh OCR system, 2020). — [arXiv 2009.09115](https://ar5iv.labs.arxiv.org/html/2009.09115)
- Threshold- or projection-based methods **over-segment** characters made of several strokes and **under-segment** where characters overlap. A 2016 approach instead scores sliding-window positions on a sub-word image with a multichannel NN to find cut points, then classifies the segments with a CNN. — [Springer chapter](https://rd.springer.com/chapter/10.1007/978-3-319-46182-3_20); [Academia.edu](https://www.academia.edu/28881429/Predictive_Segmentation_Using_Multichannel_Neural_Networks_in_Arabic_OCR_System)
- A 2024 CMC paper locates cut points in handwritten words with directional pixel-density analysis (results not visible in the snippet). — [SciOpen](https://www.sciopen.com/article/10.32604/cmc.2024.048527)
- A hybrid method for character segmentation from Arabic handwritten document images. — [ResearchGate PDF](https://researchgate.net/profile/Mufeed-Saif/publication/360343135_Characters_Segmentation_from_Arabic_Handwritten_Document_Images_Hybrid_Approach/links/6271416a973bbb29cc5e7661/Characters-Segmentation-from-Arabic-Handwritten-Document-Images-Hybrid-Approach.pdf)
- Explicit character segmentation via semantic segmentation (**SegNet**) followed by a CNN+LSTM recognizer, on news-ticker text. — [TechScience CMC](https://www.techscience.com/cmc/v79n2/56418/html)
- A 2025 review: the field has shifted to segmentation-free end-to-end pipelines; overlapping strokes and unclear boundaries remain hard; **segmentation errors propagate** into recognition. — [IASJ 2025 PDF](https://iasj.rdd.edu.iq/journals/uploads/2025/12/30/01173bf796342448614db3bc267e9e3f.pdf)
- A US patent, "Arabic script analysis with connection points" (US 10242276), covers detecting the connection (join) points between letters. — [USPTO PDF](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10242276)
- A comprehensive Arabic OCR survey (2023) covers segmentation approaches and challenges. — [arXiv 2312.11812](https://arxiv.org/html/2312.11812v1); [AUN PDF](https://www.aun.edu.eg/fci/sites/default/files/researches/Advancements%20and%20Challenges%20in%20Arabic%20Optical%20Character%20Recognition%20A%20Comprehensive%20Survey.pdf)

**OCR engines and character boxes**
- **Kraken** (open source, used by eScriptorium) recognizes whole lines and **skips character segmentation** by design, which avoids the step that "persistently fails on connected scripts". Its abstract claims output with **"fine-grained bounding boxes down to the character level"**. No accuracy figure for Arabic character boxes was found. — [DH abstract](https://dh-abstracts.library.virginia.edu/works/9912); [Open-source Arabic-script OCR case study, arXiv 2402.10943](https://arxiv.org/pdf/2402.10943); [Romanov blog on Arabographic OCR](https://maximromanov.github.io/2016/10-05.html)
- **Surya** vs **Tesseract** (2026 evaluation of cloud and self-hosted OCR on Arabic, English and French documents):
  - Google Document AI produced good **line** boxes on 47/48 pages.
  - Surya produced good line boxes on **46/48**.
  - Tesseract produced good line boxes on **24/48 (50%)**.
  - The evaluation is line-level, not character-level.
  — [arXiv 2610.25235](https://arxiv.org/html/2610.25235)
- **QARI-OCR** (arXiv 2506.02295, 2025): fine-tuned from **Qwen2-VL-2B-Instruct**, with model cards on HF (NAMAA-Space). The license metadata says **apache-2.0**, but the card text says it "follows the licensing terms of the original Qwen2 VL model". It is a text-output VLM with no character boxes. **Baseer OCR** is another Qwen2-VL-based Arabic document OCR, stated as Apache-2.0. — [HF Qari-OCR-0.1](https://huggingface.co/NAMAA-Space/Qari-OCR-0.1-VL-2B-Instruct); [HF paper page](https://huggingface.co/papers/2506.02295); [Baseer OCR](https://huggingface.co/AbdoTarek/Baseer-OCR-V1.0)
- **EasyOCR** and **TrOCR-Arabic** were among the models evaluated on DuwatBench (calligraphy), where all models scored low; the best overall was 0.4167 exact match, by Gemini-2.5-flash. — [DuwatBench GitHub](https://github.com/mbzuai-oryx/DuwatBench)
- **OCR4all**: an open-source semi-automatic OCR workflow for historical printings, which builds on Calamari and Kraken components. — [arXiv 1909.04032](https://arxiv.org/pdf/1909.04032)

### Inferences
- **The structure of مزن makes segmentation easier than average.** ز is a non-joining letter, so the word image has two main ink bodies, "مز" (م-init joined to ز-fina) and "ن" (isolated), plus three small components: the dot of ز, the dot of ن and the damma. Connected-component labeling with size and position heuristics would separate ن, the dots and the damma directly. Only one cut is needed, at the م→ز join on the baseline.
- **Fragmented or geometric styles break the connected-component assumption.** A stencil-like or fragmented Kufi may split one letter into several components, or merge a dot into the body. Bubbly styles can make the damma or dots touch or overlap the body. So component-to-letter assignment has to be resolved by alignment with the known text, which is always available here because the user's word is known. A simple approach: render the known word in a few reference fonts, then align (e.g., chamfer or DTW over skeleton or contour) to propose cut points.
- Because the text is known, a **forced-alignment** approach fits better than open OCR: use a CTC recognizer (Kraken, or a CRNN trained on synthetic Arabic) and take the CTC alignment of the known label sequence as character spans. Expect these spans to be approximate, since CTC peaks are not glyph boundaries. They are acceptable for cropping, but not as precise outlines.
- For a product, a **human-in-the-loop correction UI** (drag the cut lines, assign components to letters) is likely cheaper and more reliable than a fully automatic segmenter, given 3 letters per word.
- Skeletonization and baseline detection (horizontal projection peak) remain useful for finding the baseline and the join strokes (kashida region) of م→ز. They are unreliable for display styles with strong baseline play, bouncing letters or stacked ligatures.

### Gaps
- No benchmark of character-box accuracy on Arabic was found for Kraken, Tesseract (symbol-level iterator), Surya, PaddleOCR (Arabic model) or EasyOCR. Whether current PaddleOCR or Surya versions expose per-character Arabic boxes was not verified (their docs were not fetched).
- I found no work segmenting **stylized display or logo** Arabic lettering. All the segmentation literature is on handwriting or print.
- kraken.re docs (API details of character cuts) could not be fetched in this session.

---

## Q4. Decomposing an Arabic glyph set for generation (rasm, dots, marks) and the minimal glyph set for a functional font

### Takeaway
An Arabic font can be generated as **shared skeleton bodies (rasm) × contextual form + separately designed dots (i'jam) + marks (harakat) + a few special ligatures**, and then assembled with OpenType GSUB (positional forms, lam-alef) and GPOS (mark anchors). This cuts the drawing problem from roughly 130–150 complete glyphs to about 18 base skeleton families (roughly 50–60 distinct body shapes across positions), 1–3 dot shapes, about 9 mark shapes and a few hamza and madda shapes. Modern fonts should encode letters in the Arabic block U+0600–U+06FF and use **unencoded positional glyphs plus GSUB** `init/medi/fina/isol` + `rlig` (lam-alef) + `mark/mkmk`. The legacy Presentation Forms-B code points are optional, for compatibility only.

### Cited Findings
- The Unicode Arabic block is **U+0600–U+06FF**. It holds the basic letters U+0621–U+064A, the tashkil/harakat **U+064B–U+0652** (fathatan, dammatan, kasratan, fatha, damma, kasra, shadda, sukun) plus U+0670 superscript alef, Arabic-Indic digits U+0660–U+0669, the punctuation ، ؛ ؟ (U+060C, U+061B, U+061F), tatweel U+0640, and dotless archigraphemes **U+066E (dotless beh)** and **U+066F (dotless qaf)**. U+0649 alef maksura is the dotless yeh and U+06BA the dotless noon. — [Unicode Arabic chart U0600.pdf](https://www.unicode.org/charts/PDF/U0600.pdf) (standard reference; not re-fetched this session)
- **Arabic Presentation Forms-B (U+FE70–U+FEFF)** encodes the positional forms of the basic letters and the **lam-alef ligatures U+FEF5–U+FEFC**. **Presentation Forms-A (U+FB50–U+FDFF)** has extended-letter forms and many ligatures. These are compatibility characters, and modern shaping uses GSUB on unencoded glyphs. — [Unicode chart UFE70.pdf](https://www.unicode.org/charts/PDF/UFE70.pdf); [Unicode chart UFB50.pdf](https://www.unicode.org/charts/PDF/UFB50.pdf) (standard references; not re-fetched)
- The Microsoft OpenType spec "Developing OpenType Fonts for Arabic Script" defines the shaping features applied by shaping engines (e.g., HarfBuzz, DirectWrite):
  - `isol`, `fina`, `medi`, `init` (positional forms chosen from Unicode joining type)
  - `rlig` (required ligatures, notably lam-alef)
  - `calt`, `liga`, `dlig`
  - `curs` (cursive attachment)
  - `kern`, `mark`, `mkmk`
  — [Microsoft Typography: Arabic script development](https://learn.microsoft.com/en-us/typography/script-development/arabic) (standard reference; could not fetch this session)
- Arabic-One-DM's content module renders **isol/init/medi/fina** per letter and uses a joining heuristic for non-joining letters, with `arabic_reshaper` + `bidi` for shaping. This shows the positional-form decomposition used in practice for Arabic generation. — [GitHub Arabic-One-DM](https://github.com/7abushahla/Arabic-One-DM)
- KHATT, a recognition dataset, was noted as non-artistic, and display styles are absent from recognition datasets. That supports designing the generator around font data rather than handwriting data. — [IJACSA 2025](https://thesai.org/Downloads/Volume16No3/Paper_81-Handwritten_Arabic_Calligraphy_Generation.pdf)

### Inferences (decomposition design; my analysis, based on standard Arabic letter structure)

**Joining classes**
- Dual-joining letters take 4 forms: ب ت ث ج ح خ س ش ص ض ط ظ ع غ ف ق ك ل م ن ه ي ئ.
- Right-joining (non-joining-left) letters take 2 forms (isol, fina): ا آ أ إ د ذ ر ز و ؤ ة ى.
- Non-joining: ء.
- For the 28 basic letters this gives 22×4 + 6×2 = **100 positional forms**. Adding ء, ة (2), ى (2), the hamza and madda alef variants (آ أ إ, each 2), ؤ (2) and ئ (4) brings it to about **120 letter glyphs**.
- Lam-alef needs **8 ligature glyphs**: لا لأ لإ لآ, each in isol and fina.

**Shared rasm skeleton families**, where dots are the only difference:
- ب ت ث + ن ي ئ in **init/medi** (the "tooth"). Note that ن-isol/fina and ي-isol/fina have distinct bodies.
- ج ح خ
- د ذ
- ر ز
- س ش
- ص ض
- ط ظ
- ع غ
- ف ق in init/medi. Their fina/isol bowls differ.
- ك
- ل
- م
- ه / ة. ة is the ه-fina/isol body plus two dots.
- و / ؤ
- ى / ي fina/isol
- ا and its hamza variants

That is about **18 skeleton families**. Counted with their positional variants, it comes to about 50–60 distinct body outlines, which is roughly half the number of full letter glyphs.

**Dots (i'jam)**
- Shapes: 1, 2 and 3 dots, in above or below positions.
- The base shape is a single dot design. The 2- and 3-dot arrangements are compositions, with spacing and stacking adjusted per style (some styles fuse two dots into a dash).
- Generating one dot shape and placing it via anchors (GPOS `mark`) or as components keeps dots consistent.

**Marks**
- The harakat set is fatha, damma, kasra, sukun, shadda, plus the tanwin variants (fathatan, dammatan, kasratan) and superscript alef: about 9 mark shapes. The tanwin marks are doubles of the base marks.
- Hamza (ء, above and below forms) and madda are separate small shapes.
- All are positioned with `mark`/`mkmk` anchors on bases.

**Reference coverage from مُزن**
- It supplies the م family (init only), the ر/ز family (fina only), the ن isolated bowl, the dot shape and the damma.
- The damma gives one mark shape. Related marks (the waw-like damma relates loosely to و's head) offer weak cues only.
- Everything else must be extrapolated, notably ا, ل, ك, the ع head, the ه loops, the ص/ط bowls and the teeth of س/ب.

**Minimal functional font**
- U+0621–U+064A letters
- U+064B–U+0652 + U+0670 marks
- U+0660–U+0669 digits (optional; Latin digits can be used)
- U+060C, U+061B, U+061F, U+0640
- Positional forms as unencoded glyphs, wired through GSUB `init/medi/fina/isol` + `rlig` for lam-alef
- GPOS `mark`/`mkmk` anchors (plus optional `curs` for joins)
- A Latin space and basic punctuation
- Persian/Urdu letters (پ چ ژ گ ک ی from U+067E, U+0686, U+0698, U+06AF, U+06A9, U+06CC) are optional extensions. They reuse the same skeletons with different dots.

**Pipeline**
- Generating outlines per skeleton family and composing dots and marks programmatically, e.g., with fontTools/feaLib or ufo2ft (tooling choices are not sourced here), keeps the glyph-to-glyph consistency that per-glyph neural generation often loses.
- The joining stroke height and thickness (where the kashida meets the baseline) must match across all init, medi and fina forms, or joined text will show breaks. This is the main Arabic-specific consistency constraint: an extra objective, or a post-process that snaps join endpoints to a shared baseline and stroke height.

### Gaps
- The exact Microsoft feature order and text could not be quoted (learn.microsoft.com blocked), so the feature list above relies on standard-reference knowledge.
- I found no published study measuring how much rasm or dot decomposition improves Arabic glyph-generation quality.
- Kashida/tatweel justification and stylistic `calt` (e.g., bouncing baseline in display styles) were not researched.

---

## Q5. Datasets: Google Fonts OFL Arabic families, other open Arabic fonts, KHATT, IFN/ENIT, Muharaf, APTI, KAFD, calligraphy image datasets

### Takeaway
For training an Arabic glyph style-transfer model, **open-licensed fonts are by far the best data**: aligned glyph sets in many styles, with vector outlines and permissive OFL licenses. Google Fonts has **about 62 families** that declare an `arabic` subset (GitHub code search, Nov 2026), including several display styles close to the two targets (bubbly: Lemonada, Marhey, Baloo Bhaijaan 2, Lalezar, Jomhuria, Rakkas, Cairo Play, Reem Kufi Fun; geometric/angular: Reem Kufi, Kufam, Blaka, Handjet, Noto Kufi Arabic). Handwriting datasets (KHATT, IFN/ENIT, Muharaf) and printed-text datasets (APTI, KAFD) are recognition-oriented, mostly non-commercial or LDC-licensed, and not stylistically suited to display-font generation. Calligraphy image sets (DuwatBench, HICMA, Nas4-60k, Calliar) are small and have mixed or NC licenses.

### Cited Findings

**Google Fonts (OFL)**
- A GitHub code search of `google/fonts` for METADATA.pb files containing `subsets: "arabic"` returned **62 families** (all under `ofl/`), on 2026-11-08 via the GitHub API. GitHub code search can under-report, so treat 62 as a lower bound. — [google/fonts repo](https://github.com/google/fonts)
- The families, by path name:
  - oi, mada, kufam, vibes, blaka, blakaink, blakahollow, rubik
  - amiri, amiriquran, mirza, cairo, cairoplay, ruwudu, firjar, thabit
  - marhey, changa, gulzar, fustat, rakkas, estedad, tajawal, katibeh
  - beiruti, handjet, almarai, lalezar, lemonada, alyamama, alansans, jomhuria
  - alkalami, parastoo, reemkufi, reemkufiink, reemkufifun, arefruqaa, arefruqaaink, vazirmatn
  - elmessiri, readexpro, alexandria, googlesans, markazitext, cascadiacode, cascadiamono, badeendisplay
  - baloobhaijaan2, notonastaliqurdu, playpensansarabic, lateef, zain, scheherazadenew, harmattan, qahiri
  - notosansarabicui, notonaskharabicui, ibmplexsansarabic, notokufiarabic, notosansarabic, notonaskharabic
  
  — [google/fonts ofl/ directory](https://github.com/google/fonts/tree/main/ofl)
- Google Fonts is the second-stage training set of VecGlypher (about 2,497 families, 157,899 glyph instances, all scripts), which shows it is the standard curated source for glyph-generation training. — [VecGlypher project page](https://xk-huang.github.io/VecGlypher)
- The low-resource text-image paper collected **64 Arabic font files** for synthesis. That is a comparable scale for a typical research-collected Arabic font set. — [arXiv 2409.17747](https://arxiv.org/html/2409.17747v1)

**Printed-text datasets**
- **KAFD** (King Fahd University of Petroleum & Minerals Arabic Font Database; Pattern Recognition 2014). Note: the database is from KFUPM, not "King Abdulaziz".
  - Contents: **40 fonts, 10 sizes (8–24 pt), 4 styles (normal, bold, italic, bold-italic), 4 resolutions (100, 200, 300, 600 dpi)**. The LDC describes it as about **2.5 million** scanned printed pages with transcripts.
  - License: the authors say it is "freely available to researchers", while the LDC distribution **LDC2016T21** uses the LDC User Agreement for Non-Members.
  — [KFUPM Pure](https://pure.kfupm.edu.sa/en/publications/kafd-arabic-font-database/); [LDC2016T21](https://catalog.ldc.upenn.edu/LDC2016T21); [paper PDF](https://Catalog.ldc.upenn.edu/docs/LDC2016T21/2014-J-KAFD%20Arabic%20Font%20Database_1-s2.0-S0031320313005463-main.pdf)
- **APTI** (Arabic Printed Text Image database, Univ. Fribourg / HES-SO; ICDAR 2009): a synthetic word-level set of **45,313,600 word images** (more than 250M characters) rendered in multiple fonts, sizes and styles. It is "freely available for the scientific community"; no specific license name was found. — [DIVA APTI page](https://diuf.unifr.ch/main/diva/APTI); [HES-SO publication](https://publications.hevs.ch/index.php/publications/show/698)
- **Khat²**: synthetic Arabic word images in many fonts, including handwriting-style fonts (Arabic-One-DM's pretraining set; generation scripts in the Khat_Squared repo). — [Arabic-One-DM GitHub](https://github.com/7abushahla/Arabic-One-DM)
- A "database containing 9 categories of computer printed Arabic text snippets" (HKUST). — [HKUST project page](https://smedia.hkust.edu.hk/james/projects/ict.html)

**Handwriting datasets**
- **KHATT**: scanned Arabic handwriting from **1,000 writers** of diverse countries, ages, handedness and education. Distributed by the LDC as **LDC2015T23** under an LDC license for non-members. — [LDC catalog](https://catalog.ldc.upenn.edu/LDC2015T23). It is non-artistic, so it has limited value for stylized generation. — [IJACSA 2025](https://thesai.org/Downloads/Volume16No3/Paper_81-Handwritten_Arabic_Calligraphy_Generation.pdf)
- **Muharaf** (NeurIPS 2024 Datasets & Benchmarks; arXiv 2406.09630): historical handwritten Arabic manuscripts.
  - Public part: **1,216 images**, under **CC BY-NC-SA** (one aggregator lists both 4.0 and 3.0; verify on HF).
  - Restricted part: 428 images under a proprietary license.
  - Open question: an HF discussion asks whether share-alike applies to trained weights and whether commercial licensing exists, and the answer was not visible.
  — [NeurIPS page](https://papers.nips.cc/paper_files/paper/2024/hash/6b8cb6b291045e217c3ff3f854f2fd0f-Abstract-Datasets_and_Benchmarks_Track.html); [arXiv](https://arxiv.org/html/2406.09630v1); [hyper.ai listing](https://hyper.ai/en/datasets/35487); [HF discussion](https://huggingface.co/datasets/aamijar/muharaf-public/discussions/2)
- **IFN/ENIT, AlexU-Word, AHAWP, IESK-ArDB** were combined in Arabic-One-DM into about 62,437 words from about 1,417 writers. Their individual licenses were not checked. — [Arabic-One-DM GitHub](https://github.com/7abushahla/Arabic-One-DM)

**Calligraphy image datasets**
- **DuwatBench**: 1,272 samples, 6 styles (Kufic 83, Diwani 230 per a snippet), bounding boxes and transcriptions. Code is Apache-2.0; images are "under their respective licenses". HF: MBZUAI/DuwatBench. — [GitHub](https://github.com/mbzuai-oryx/DuwatBench); [project page](https://mbzuai-oryx.github.io/DuwatBench)
- **HICMA**: more than 5,000 images in 5 styles (Naskh, Thuluth, Diwani, Kufic, Muhaqqaq), **CC BY-NC 4.0**. This is per a search snippet from a model card trained on it, and the primary page was not fetched. — [search result context: ACL ArabicNLP 2023](https://aclanthology.org/2023.arabicnlp-1.3.pdf)
- **Roboflow "ARABIC STYLE"**: a small classification set (Dewani, Naskh, Riqa, Thuluth), CC BY 4.0. — [Roboflow Universe](https://universe.roboflow.com/demiana/arabic-style)
- **Calliar** (arXiv 2106.10745): an **online** (stroke-sequence) handwritten Arabic calligraphy dataset. No license was found in the results. — [arXiv](https://arxiv.org/pdf/2106.10745); [project page](https://arbml.github.io/Calliar/page)
- **Nas4-60k / Nas4-60k-aug**: Nastaliq calligraphy sets made for the StyleGAN2-ada paper. Availability and license were not found. — [ACM DL](https://dl.acm.org/doi/10.1145/3664210)
- **Hadj Azzem et al. aligned font→calligraphy pairs** (Reqaa, Diwani, Farsi), MIT-licensed repo; the size is not stated. — [GitHub](https://github.com/Yousra-Chahinez/generate_Arabic_calligraphy_GANs)

### Inferences

**Style grouping of the Google Fonts Arabic families** (my classification by design style; verify visually). This is useful for choosing style-transfer training pairs.

- *Rounded or bubbly display*, close to the "bubbly" target:
  - Lemonada
  - Marhey
  - Baloo Bhaijaan 2
  - Lalezar
  - Jomhuria
  - Rakkas
  - Cairo Play
  - Reem Kufi Fun
  - Oi
  - Badeen Display
  - Playpen Sans Arabic
- *Geometric, angular or Kufic*, close to the "angular/fragmented" target:
  - Reem Kufi / Reem Kufi Ink
  - Kufam
  - Blaka / Blaka Ink / Blaka Hollow
  - Handjet (modular)
  - Noto Kufi Arabic
  - Changa
  - Katibeh
- *Sans and UI*:
  - Cairo, Tajawal, Almarai, Readex Pro, Alexandria, IBM Plex Sans Arabic
  - Noto Sans Arabic (UI), Mada, Zain, Beiruti, Rubik, Alan Sans, Google Sans
  - El Messiri, Harmattan, Firjar, Alyamama
- *Naskh, text and calligraphic*:
  - Amiri / Amiri Quran, Noto Naskh Arabic (UI), Scheherazade New, Lateef, Markazi Text, Mirza
  - Aref Ruqaa / Aref Ruqaa Ink, Ruwudu, Qahiri, Fustat
  - Gulzar / Noto Nastaliq Urdu (Nastaliq)
  - Alkalami
- *Persian-oriented*: Vazirmatn, Estedad, Parastoo.
- *Monospace*: Cascadia Code/Mono, Thabit.

**Data volume and augmentation**
- About 62 families yield perhaps about 150–250 font instances once weights are counted (e.g., Cairo, Tajawal, Readex Pro, Noto and Rubik are multi-weight or variable). Each instance has roughly 100–300 Arabic glyphs, so this is on the order of 10⁴–10⁵ aligned Arabic glyph images. That is small next to CJK font datasets.
- The total open supply could be expanded with other OFL Arabic fonts outside Google Fonts (e.g., SIL, Arabic Typesetting-alikes, and the many OFL Arabic fonts on GitHub). Augmentation (stroke-width morphology, rounding/erosion, fragmentation masks) could synthesize "bubbly" and "fragmented" variants of existing families.
- Variable Arabic fonts (axes such as wght) give continuous style-interpolation data cheaply.

**Licensing for a commercial product**
- OFL fonts are safe to train on and to derive outputs from. OFL's Reserved Font Name and derivative clauses mainly matter if the original font is redistributed or modified; whether model outputs count as derivatives is a legal question not researched here.
- KHATT and KAFD (LDC), Muharaf (NC-SA) and HICMA (NC) are **risky** for commercial training, and they are stylistically irrelevant to display lettering anyway.
- APTI and KAFD rendered images could serve for training a **segmenter or recognizer** (forced alignment) rather than the generator.

### Gaps
- The exact fonts.google.com count of Arabic-script families could not be retrieved (fonts.google.com blocked), so the 62 figure comes from GitHub code search and may be slightly low (e.g., families whose METADATA formats subsets differently).
- Per-family licenses were not individually verified; all sit under `ofl/`, which in the google/fonts repo means SIL OFL.
- No sizes or licenses were verified for IFN/ENIT, AlexU-Word, AHAWP or IESK-ArDB.
- The HICMA primary source and Calliar's license and size were not verified.
- No dataset of **Arabic logo or display lettering** images (e.g., Behance/Dribbble-style Arabic logotypes) with labels was found. This is the closest match to the user input domain and appears to be absent from the open literature.
