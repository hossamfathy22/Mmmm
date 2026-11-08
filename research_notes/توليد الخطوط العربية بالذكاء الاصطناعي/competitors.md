# Competitive Landscape: Handwriting/Lettering-to-Font and AI Font Generators (Arabic Focus, as of Nov 2026)

> Method note: these notes rest on WebSearch result summaries (extended mode), run on 2026-11-08. Direct page fetches (WebFetch) were blocked by the session network policy. yofont.com, fontmaker.co and forum.glyphsapp.com did not resolve. Vendor claims were therefore not read in full and may be paraphrased by the search tool. Most product facts come from vendors' own marketing pages, and they are labelled as vendor claims where relevant.

## 1. Template/handwriting-to-font services (Calligraphr, MyScriptFont, Fontself, YourFonts, PaintFont, Glyphr Studio, FontStruct, BirdFont, Fontark, Prototypo): Arabic support, input, pricing

### Takeaway
The classic template-based tools all need the user to draw every glyph themselves: a full template, not a few letters. None of them offers verified, first-class Arabic support with automatic contextual (initial/medial/final/isolated) shaping. Fontself and FontStruct explicitly say Arabic is unsupported or only partly supported. Calligraphr has the clearest pricing ($6–10/mo Pro), but there is no confirmed Arabic template.

### Cited Findings
- **Calligraphr pricing:** Free tier allows 1 font at a time and 75 characters per font. Pro costs $10 for one month or $6/month on a 6-month upgrade, and allows 20 concurrent fonts, 600 characters per font, ligatures and letter-spacing control. A third-party comparison checked these prices against the vendor site on 1 Nov 2026. — [Calligraphr Pricing](https://www.calligraphr.com/en/pricing/); [Coki Fonts: Calligraphr Alternatives (Nov 2026)](https://cokifonts.com/blog/calligraphr-alternatives/)
- **Calligraphr language templates:** a 2022 review listed template languages as English, German, French, Spanish and Japanese, with no Arabic. This may be outdated. — [Podfeet Podcasts (2022)](https://www.podfeet.com/blog/2022/06/calligrapher/)
- **Calligraphr and Arabic (weak source):** an Alibaba "product insights" article says that for complex scripts with contextual shaping such as Arabic, "manual glyph linking is required and success varies." It also says Calligraphr covers Unicode ranges beyond Latin-1, such as Cyrillic, Greek and extended Latin. This is low reliability. — [Alibaba product insights](https://www.alibaba.com/product-insights/ai-powered-calligraphy-font-generators-calligraphr-vs-fontself-which-converts-your-handwriting-into-usable-fonts.html)
- **Fontself:** Fontself staff said on the community forum that Fontself "does not yet support Arabic. We hope to do so at some point but there's still work ahead." The post is about 7 years old, and no newer statement was found. — [Fontself Community: Arabic support help](https://community.fontself.com/t/arabic-support-help/1422)
- **Fontself plus Firefly as a hobbyist workaround:** an artist used Adobe Firefly Text Effects output as glyph art and built a color font in Photoshop with the Fontself plug-in. It covered capital letters only. — [Kris Kashtanova on X](https://x.com/icreatelife/status/1689300059161083904)
- **FontStruct:** the official FAQ says FontStruct has "only partial support for non-latin fonts." You can draw individual Arabic letters, but initial/medial/final glyphs will not appear as you type in desktop software such as Word. Users are told to export and finish the font in FontForge, which "is not an easy task." Community-made Arabic FontStructions, such as pixel fonts, do exist. — [FontStruct FAQ](https://fontstruct.com/faq/59/technical_questions/what_about_non_latin_fonts_arabic_hebrew_etc); [Arabic pixel font on FontStruct](https://fontstruct.com/fontstructions/show/2618076/arabic-pixel-font-3)
- **Glyphr Studio, BirdFont, Fontark, Prototypo:** these are general-purpose editors. BirdFont exports TTF, OTF and SVG fonts from vector drawings. Fontark offers real-time multi-glyph editing and an automatic outline generator. Prototypo is parametric, with 30+ parameters such as thickness, aperture and roundness. A roundup claims these tools "can also be used to create complex none english scripts such as Cyrillic, Hebrew, Arabic," with no detail on OpenType shaping. — [Birdfont](https://birdfont.org/); [Gecko&Fly: 8 Free Font Creators](https://www.geckoandfly.com/39029/free-font-creators-editors/); [localfonts.eu font creators](https://localfonts.eu/typography-basics/typographic-utilities/font-creators/)
- **Newer browser handwriting makers:** FontMaker (fontmaker.co) lets you photograph a handwritten page or draw letters, then download TTF/OTF. Processing happens in the browser. FontCrafter (Pirillo) also runs locally in the browser. Neither page mentions Arabic. — [FontMaker app](https://www.fontmaker.co/app); [FontCrafter](https://arcade.pirillo.com/fontcrafter.html)
- **Mobile apps:** the iOS "Font Maker" app claims 40+ languages and alphabets but does not name Arabic. 10015.io's handwriting builder covers Latin, Cyrillic, Greek and Hebrew, not Arabic. — [App Store: Font Maker](https://apps.apple.com/qa/app/font-maker-create-your-font/id6443977931?l=ar); [10015.io](https://10015.io/tools/text-to-handwriting-converter)
- **What proper Arabic support requires:** Arabic is cursive and right-to-left, and letters have at least 4 basic forms: initial, medial, final and isolated. Fonts need RTL behaviour, positional forms, dynamically positioned diacritics, ligatures and alternates. These are usually handled through OpenType contextual substitution. — [SIL Arabic Fonts: Application Support](https://software.sil.org/arabicfonts/support/application-support/); [Arabeyes: Developing Arabic fonts](https://www.arabeyes.org/Developing_Arabic_fonts)

### Inferences
- Template tools treat each glyph as an independent drawing, so an Arabic user would have to draw roughly 100–150+ positional glyphs by hand and then fix the OpenType shaping (init/medi/fina/isol, the lam-alef ligature) in another editor. This is the main pain point the proposed product would remove.
- Calligraphr's 600-character Pro limit is enough for a basic Arabic positional set, but there is no evidence that it generates the GSUB shaping tables automatically.

### Gaps
- MyScriptFont, YourFonts and PaintFont: no search results described their Arabic support or current pricing. Their status (live or defunct) is unverified.
- Calligraphr's own help pages on Arabic or custom Unicode templates could not be retrieved, because fetching was blocked.
- Prototypo's current operating status (live or shut down) could not be confirmed.
- Glyphr Studio's and BirdFont's Arabic OpenType shaping support could not be confirmed from primary docs.

## 2. AI font generators (Adobe, Monotype, Recraft, Kittl, Ideogram, startups, Chinese commercial products): what can they do from few samples?

### Takeaway
By late 2026 a wave of small AI startups (YoFont, Lipi.ai/GLIPH, Mixfont, Coki Fonts, Måns Grebäck's AI Font Creator) generate installable fonts from an image or a few letters. Almost all are Latin-only, and the big incumbents (Adobe, Monotype, Recraft) still do not ship few-shot font generation. Adobe's 3-letter "Project Glyph Ease" stayed a 2023 sneak, and Monotype's 2026 AI work is search, not generation. China has the most mature commercial few-shot products, such as Founder Shouji (100 characters to a full GB2312 font) and Zitijia AI (8 characters), which shows the business model works at scale.

### Cited Findings
**Adobe**
- **Project Glyph Ease** (Adobe MAX Sneaks, Oct 2023): a designer creates three reference letters, from existing vectors or hand-drawn on paper, and the tool "automatically create[s] the remaining letters in a consistent style." Output is vector letters that can be edited as live text. Adobe gave no release date. — [Adobe Blog: MAX Sneaks 2023](https://blog.adobe.com/en/publish/2023/10/11/max-sneaks-2023); [John Nack: Sneak peek Project Glyph Ease](https://jnack.com/blog/2023/10/29/sneak-peek-project-glyph-ease/)
- One outlet speculated a release in H1 2024. No evidence was found that it shipped in Illustrator or Firefly as of 2026. — [Nerdschalk: Adobe MAX Sneaks 2023](https://nerdschalk.com/adobe-max-sneaks-2023-list-of-new-tools-and-features/)
- **Firefly Text Effects** applies generated textures and styles to typed text. It produces images, not a font file. — [MakeUseOf: Firefly font prompt ideas](https://www.makeuseof.com/adobe-firefly-ai-font-generation-ideas/)
- Earlier Adobe research sneak: "Fontphoria" (MAX 2018). — [Dezignark: Fontphoria](https://dezignark.com/blog/fontphoria-adobe-max-2018-sneak-peek-adobe-creative-cloud/)

**Monotype**
- In Feb 2026 Monotype launched AI Search, a natural-language font discovery tool over 250,000+ fonts. In May 2026 it launched a MyFonts app for ChatGPT. Both are discovery tools, not generators. — [Monotype press release: AI Search](https://www.monotype.com/company/press-release/monotype-launches-ai-search-transforming-font-search-and-discovery-conversational-ai); [Monotype press release: MyFonts app for ChatGPT](https://www.monotype.com/company/press-release/monotype-launches-myfonts-app-chatgpt-bringing-real-font-discovery-ai)
- A Monotype executive said "the quality isn't there yet to reliably take something generated by GenAI and scale it up for use across commercial touchpoints." — [Transform magazine: Can AI design a typeface? (2025)](https://www.transformmagazine.net/articles/2025/can-ai-design-a-typeface/)

**Recraft / Kittl / Ideogram**
- Recraft supports uploading fonts (.woff, .woff2, .ttf, .otf) to use in its Studio, and generates vectors as SVG. No font generation or export feature was found. — [Recraft docs: Using custom fonts](https://www.recraft.ai/docs/recraft-studio/image-generation/working-with-text-and-prompts/using-custom-fonts)
- The open-source project "Fontify" can use Recraft Vector as a glyph source and compile fonts offline. Its example covers A–Z and a–z, and it exports TTF only (no CFF/OTF). — [GitHub: ferrerluis/fontify](https://github.com/ferrerluis/fontify)
- Kittl is listed as an AI design platform for text effects on logos and merch, not a font-file generator. — [Lilach Bullock: AI Font Generators 2026](https://www.lilachbullock.com/best-ai-font-generators/); [Coki Fonts: Best AI Font Generators (Nov 2026)](https://cokifonts.com/blog/best-ai-font-generators/)
- A Nov 2026 roundup found that among the tools it reviewed, only Coki Fonts (any image), Mixfont (prompt or image) and GLIPH (image) output an installable font file. FontVibe and Picsart produce styled images only, and Calligraphr and Fontself make real fonts but are not AI. (The roundup is published by Coki Fonts, a competitor, so it is a biased source.) — [Coki Fonts: Best AI Font Generators (Nov 2026)](https://cokifonts.com/blog/best-ai-font-generators/)

**AI startups that produce real font files**
- **YoFont:** generates a font from a one-sentence prompt or a reference image (logo, poster, handwriting photo). It has a dedicated "Logo to Font" page. Its guidance: "four characters define a style loosely; ten define it well." Missing letters, for example lowercase from an all-caps logo, are steered by an extra text prompt. It exports TTF, OTF, WOFF and WOFF2, has a per-glyph editor, and glyphs can be refined by prompt ("make strokes thicker"). — [YoFont: Font From Image](https://yofont.com/font-from-image); [YoFont: Logo to Font](https://yofont.com/logo-font-generator); [Futuretools: YoFont](https://futuretools.io/tools/yofont-bsvdws)
  - Pricing: monthly glyph balance, where 1 glyph = 1 character. Tiers are Free, Starter $9, Pro $19 and Studio $49. The first font is free, or 100 free glyphs on Google sign-in; sources conflict. — [YoFont plans](https://yofont.com/plans); [Futuretools: YoFont](https://futuretools.io/tools/yofont-bsvdws)
  - Arabic: see section 3. Its Product Hunt page says it works with "Latin, Cyrillic, Arabic, Hebrew, CJK and more," and the founder framed the launch as "why can we generate video in 2025 but not real fonts." — [Product Hunt: YoFont](https://www.producthunt.com/p/yofont/why-can-we-generate-video-in-2025-but-not-real-fonts-so-i-built-yofont)
- **Måns Grebäck's AI Font Creator** (Glyphs Forum showcase, 2026): makes a font from a single image with a few letters. It is free, with 10 credits/day when logged in. The author says quality "is not perfect and the results can vary a lot," and the web beta has no download option. A Glyphs 3 plugin (AIFontGenerator) uses selected glyphs as reference and creates a new master. Related plugins (AI Kerning, AI Spacing, AI Anchors, AI Variable Font) are "free on A–Z a–z, full version by license," with models running on the author's server. No Arabic support is documented. — [Glyphs Forum: AI Font Creator](https://forum.glyphsapp.com/t/my-new-ai-font-creator-makes-a-font-from-an-image/35870); [GitHub PR: Måns AI Tools](https://github.com/schriftgestalt/glyphs-packages/pull/236); [GitHub: mansgreback](https://github.com/mansgreback)
- **Lipi.ai (and its sibling GLIPH):** image or handwriting to font. Glyphs are extracted, vectorized and built into a font, with export to OTF, TTF, WOFF and WOFF2. Font Studio editing is free and you pay at export: $4.99 non-exclusive commercial or $9.99 exclusive "full ownership" (Slashdot lists $7.99/font). Language coverage is Latin-only: Standard tier ~36–37 languages for 1 credit, Extended 81–82 languages for 3 credits. Arabic is on the "longer-term roadmap" with Greek, Cyrillic, Devanagari and CJK. — [Lipi.ai Handwriting](https://www.lipi.ai/handwriting); [Lipi.ai Multilingual](https://www.lipi.ai/multilingual); [Lipi.ai Font Studio](https://www.lipi.ai/font-studio); [Slashdot: Lipi.AI](https://slashdot.org/software/p/Lipi.AI/)
- **GLIPH:** AI font from any image. Base fonts cover 36 Latin-script languages. Credit-based, with a 10-credit free trial; a third-party blog says each sheet upload costs 5 credits. — [GLIPH](https://gliph.us/); [GeekExtreme: Create font from image](https://www.geekextreme.com/create-font-from-image-free/)
- **Mixfont:** AI font from a text prompt or image. Free plan has 50 starter credits; Pro is $20/mo; Max is $200/mo. There is also an API with its own pricing. Its extended glyph set covers Latin-script languages, so no Arabic. — [Mixfont pricing](https://www.mixfont.com/pricing); [Mixfont API pricing](https://www.mixfont.com/docs/pricing)
- **Coki Fonts:** AI font from any image. Five free generations, a $5 pack adds 10 more, and generation is unlimited if you bring your own OpenAI key. Language coverage is not stated. — [Coki Fonts guide](https://cokifonts.com/guide/)

**Chinese commercial products (the most mature few-shot market)**
- **Founder Shouji (方正手迹造字):** built on AI font-generation technology from Peking University's Wangxuan Institute. Writing a minimum of 100 characters produces a basic font, and 898 are recommended for accuracy. Input is on-screen handwriting or a photo of paper. Output is the full GB2312 set (6,763 hanzi plus symbols). The app is free to download, but commercial licensing terms are unclear. — [Baidu Baike: 手迹造字](https://baike.baidu.com/item/%E6%89%8B%E8%BF%B9%E9%80%A0%E5%AD%97/54125844); [方正手迹官网](https://www.fzshouji.com/m/make); [App Store: 手迹造字](https://apps.apple.com/us/app/%E6%89%8B%E8%BF%B9%E9%80%A0%E5%AD%97-%E4%B8%AA%E4%BA%BA%E5%AD%97%E5%BA%93%E7%B2%BE%E5%93%81%E5%AD%97%E4%BD%93%E5%AE%89%E8%A3%85%E5%BA%94%E7%94%A8/id1190427931)
- **Zitijia AI Shenbi (字体家AI神笔):** eight handwritten characters produce a style model that outputs a font of 6,000+ characters. Priced in packages, with the standard package at GB2312 (6,763 hanzi). First-set discount for new customers. — [字体家AI神笔 FAQ](https://www.ai.zitijia.com/faq); [猫啃网: 字体家AI造字夏日手写](https://www.maoken.com/freefonts/25020.html)
- **Hanyi and Alibaba:** in 2018 Hanyi and Alibaba's Natural Human-Computer Interaction Lab released "Alibaba Hanyi Smart Hei," a font generated with computational methods. Hanyi has been listed on the SZSE ChiNext since Aug 2022. — [Baidu Baike: HANYI FONTS](https://baike.baidu.com/en/item/HANYI%20FONTS/43605)
- **ByteDance Jimeng / Doubao:** renders Chinese text inside generated images. It does not build font files from samples. — [Medium: Jimeng AI Chinese text generation](https://medium.com/@xiongli.karl/incredible-breakthrough-jimeng-ai-now-supports-direct-chinese-text-generation-in-images-216fc534a416)
- **Research:** few-shot Chinese font generation remains an active arXiv topic (2018 CycleGAN; 2026 few-shot papers). — [arXiv 1801.08624](https://arxiv.org/pdf/1801.08624); [arXiv 2602.18874](https://arxiv.org/html/2602.18874v1)

### Inferences
- The "few letters to full font" capability is now commoditized for Latin in Western startups, and for CJK in China, where it is driven by the need for thousands of hanzi. Arabic sits between them. It needs far fewer base letters (about 28) than CJK, but it needs positional forms, connection strokes (kashida/joins) and OpenType shaping that Latin-focused pipelines ignore.
- Incumbents (Adobe, Monotype) are cautious or absent, so a startup window remains, but larger players could enter quickly. Adobe already demoed the 3-letter workflow in 2023.
- The Chinese benchmarks (100 characters at Founder, 8 at Zitijia) suggest few-shot quality is good enough for consumer and personal use. Professional and commercial quality is still contested, per Monotype's comment.

### Gaps
- Fontjoy (a pairing tool), Fontshare tools, Ideogram font features and Google/DeepMind font demos were not covered by results, so their font-generation capabilities are unverified. Fontjoy is known as a pairing tool, not a generator, but no source was retrieved.
- Output quality of YoFont, Mixfont, GLIPH and Coki has no independent benchmark. The only "ranking" found was published by a competitor (Coki).
- No pricing was found for Hanyi's AI custom fonts.

## 3. Arabic-specific services and foundries (AI or handwriting-to-Arabic-font; Arabic logo lettering to font)

### Takeaway
The only product found that claims to generate a real Arabic font (with positional forms) from an image or logo is YoFont, and that is a vendor claim. Every other AI font startup is explicitly Latin-only, with Arabic on the roadmap at best (Lipi.ai). Arabic foundries (29LT, TPTQ Arabic, Khatt Foundation) show no public generative-AI products. Arabic-language "AI calligraphy" sites only set text in existing fonts or output SVG/PNG artwork, not fonts.

### Cited Findings
- **YoFont Arabic page (vendor claim):** it builds "a real Arabic font" from a sentence, in up to three weights (Thin, Regular, Bold), exported as TTF, OTF, WOFF and WOFF2. It claims full coverage includes all four positional forms of every letter, the lam-alef ligature, Arabic-Indic digits and harakat. Language counts conflict (19 languages/9 writing systems on one page vs 60 languages on another). The plans page marks font downloads unavailable on one tier, which conflicts with the "first font free" claim. — [YoFont: Arabic Font Generator](https://yofont.com/arabic-font-generator); [YoFont plans](https://yofont.com/plans); [Futuretools: YoFont](https://futuretools.io/tools/yofont-bsvdws)
- YoFont also offers "Logo to Font" with image input. No source confirmed that Arabic logo lettering specifically works with it. — [YoFont: Logo to Font](https://yofont.com/logo-font-generator)
- **Lipi.ai:** Arabic is on the "longer-term roadmap," not available. — [Lipi.ai Multilingual](https://www.lipi.ai/multilingual)
- **Mixfont and GLIPH:** Latin-script glyph sets only. — [Mixfont API pricing](https://www.mixfont.com/docs/pricing); [GLIPH](https://gliph.us/)
- **Arabic "font generators" are mostly restylers:** most such sites restyle text with Unicode symbols or set text in pre-made typefaces. Examples include thearabickeyboard.com (22 fonts, copy and paste), arabiccalligraphyfonts.com (291 styles), font-generator.com/arabic and quickfontgen.com. — [The Arabic Name Generator: Best AI Arabic Calligraphy Generator 2026](https://thearabicnamegenerator.com/best-ai-arabic-calligraphy-generator/); [The Arabic Keyboard](https://www.thearabickeyboard.com/arabic-calligraphy-generator/); [Arabic Calligraphy Fonts](https://arabiccalligraphyfonts.com/); [Lucky11 Studios: Arabic Font Generator (Feb 2026)](https://lucky11studios.com/2026/02/09/arabic-font-generator-and-calligraphy-fonts-how-to-find-and-use-them/)
- **gdesign.ai and Kaleam:** Arabic AI calligraphy platforms that take typed text, apply styles and export SVG/PNG. They produce artwork, not fonts. The source is a personal Arabic blog from Jul 2026. — [bbkiwi2011: gdesign Arabic calligraphy AI (Jul 2026)](https://www.bbkiwi2011.com/2026/07/arabic-calligraphy-ai-gdesign.html)
- **Arabic-language search:** a search for an Arabic platform that turns a handwriting image into a font file returned only OCR tools (Baseer, Arabic OCR, Zinki) and calligraphy generators. No Arabic-native image-to-font product was found. — [Baseer OCR](https://baseerocr.com/); [Arabic OCR](https://arabic-ocr.com/); [Zinki](https://zinki.ai/home/)
- **Arabic Calligrapher** (desktop software): a 2D vector editor for Islamic art and Arabic calligraphy. It is not a font generator. — [Arabic Calligrapher (informer)](https://arabic-calligrapher.informer.com/)
- **Foundries:** 29LT (Pascal Zoghbi, Madrid) makes multiscript Arabic/Latin type and released a 2026 Ramadan font listing. TPTQ Arabic (Kristyan Sarkis, The Hague, founded 2015). Khatt Foundation (Huda AbiFarès) runs "Typographic Matchmaking" collaborations. No AI or generative products from them were found. These foundries are the likely human-led custom-Arabic-type alternative. — [Fonts In Use: 29LT](https://fontsinuse.com/foundry/1705/29lt); [Luc Devroye: TPTQ Arabic](https://luc.devroye.org/fonts-55334.html); [Luc Devroye: Arabic type design](https://luc.devroye.org/arab.html); [Wikipedia: Huda AbiFarès](https://en.wikipedia.org/wiki/Huda_Smitshuijzen_AbiFar%C3%A8s)
- **Legacy production knowledge:** Khatt's Arabic Font Production tutorial (Stéphane Boeuf, 2017) documents how hard calligraphic Arabic font engineering is. — [Khatt: Arabic Font Production Tutorial Part II](https://www.khtt.net/image/2017/4/11/aft_booklet02_en.pdf)

### Inferences
- YoFont is the closest direct competitor to "upload Arabic lettering, get a full Arabic font." Its Arabic claims are unverified marketing, with internal inconsistencies, and it is a generalist across 9+ scripts. It is not specialised for Arabic display styles (bubbly or geometric) or for learning from a single connected word.
- The proposed input, one connected Arabic word, is harder than what any competitor documents. YoFont recommends 4–10 separate characters. Grebäck and Glyph Ease use a few isolated letters. A connected word needs letter segmentation, identification of positional forms and inference of the join and baseline logic. Nobody advertises this.
- A likely moat is Arabic-specific OpenType engineering: automatic GSUB init/medi/fina/isol tables, rlig lam-alef, mark positioning for harakat, and kashida. Template tools (FontStruct, Fontself) cannot do this, and Latin AI tools skip it.

### Gaps
- No independent test of YoFont's Arabic output quality or shaping correctness was found. The page could not be fetched.
- No Saudi, Egyptian or Gulf startup offering AI Arabic font generation was found in English or Arabic searches. Absence of evidence is not proof of absence; LinkedIn, Monsha'at or Wamda databases were not searched.
- The Forbes Middle East piece "Traditional Arabic Calligraphy Is Getting A Technological Makeover" was referenced on Wikipedia but its content was not retrieved.

## 4. Remaining gap, pricing benchmarks and business models

### Takeaway
No verified product turns one word of stylized Arabic display lettering into a complete, correctly shaped Arabic OpenType font. The nearest is YoFont, which takes multiple characters plus a prompt, with unverified Arabic claims. Consumer pricing has settled around $5–10 per font export, or $9–20/month subscriptions with credits or glyph balances. Exclusive and commercial licences cost more, and B2B/API tiers reach about $200/month.

### Cited Findings
- **Pricing benchmarks:**
  - Calligraphr Pro: $6–10/mo. — [Calligraphr Pricing](https://www.calligraphr.com/en/pricing/)
  - YoFont: Free, then $9, $19 or $49/mo on a glyph balance. — [YoFont plans](https://yofont.com/plans)
  - Mixfont: Free (50 credits), Pro $20/mo, Max $200/mo, plus an API. — [Mixfont pricing](https://www.mixfont.com/pricing)
  - Lipi.ai: pay per export, $4.99 non-exclusive or $9.99 exclusive (Slashdot says $7.99/font). — [Lipi.ai Handwriting](https://www.lipi.ai/handwriting); [Slashdot](https://slashdot.org/software/p/Lipi.AI/)
  - GLIPH: credits, with a 10-credit trial. — [GLIPH](https://gliph.us/)
  - Coki: 5 free, then $5 for 10 more, or bring your own OpenAI key. — [Coki Fonts guide](https://cokifonts.com/guide/)
  - Grebäck: free, 10 credits/day; plugins licensed beyond A–Z. — [GitHub PR](https://github.com/schriftgestalt/glyphs-packages/pull/236)
  - Zitijia: packages with a first-set discount. — [字体家 FAQ](https://www.ai.zitijia.com/faq)
- **Business-model patterns seen:**
  - Freemium with credits or glyph balances (YoFont, Mixfont, GLIPH, Grebäck).
  - Free editing with payment at export, and separate exclusive and non-exclusive licences (Lipi.ai).
  - Bring-your-own-API-key (Coki).
  - API for developers (Mixfont).
  - Plugin licensing inside pro editors (Grebäck for Glyphs).
  - Free consumer app with an ecosystem or licensing upsell (Founder Shouji).
  - Font hosting and CDN embedding for websites (YoFont).
  — sources as above; [YoFont: font for website](https://yofont.com/font-for-website)
- **Input-requirement benchmarks:**
  - Adobe Glyph Ease demo: 3 letters. — [Adobe Blog](https://blog.adobe.com/en/publish/2023/10/11/max-sneaks-2023)
  - YoFont: 4 characters loose, 10 good. — [YoFont: Font From Image](https://yofont.com/font-from-image)
  - Zitijia: 8 hanzi. — [字体家 FAQ](https://www.ai.zitijia.com/faq)
  - Founder: 100–898 hanzi. — [Baidu Baike](https://baike.baidu.com/item/%E6%89%8B%E8%BF%B9%E9%80%A0%E5%AD%97/54125844)
  - Calligraphr: full template, up to 75 or 600 characters. — [Calligraphr Pricing](https://www.calligraphr.com/en/pricing/)
- **Quality caveats from the field:**
  - Grebäck: results "not perfect and can vary a lot." — [Glyphs Forum](https://forum.glyphsapp.com/t/my-new-ai-font-creator-makes-a-font-from-an-image/35870)
  - Monotype: GenAI quality "isn't there yet" for commercial scale. — [Transform magazine](https://www.transformmagazine.net/articles/2025/can-ai-design-a-typeface/)
  - FontStruct: Arabic shaping is not handled. — [FontStruct FAQ](https://fontstruct.com/faq/59/technical_questions/what_about_non_latin_fonts_arabic_hebrew_etc)

### Inferences
- **Gap 1, input:** no tool documents learning from a single connected Arabic word, which requires segmenting ligated or joined letters.
- **Gap 2, Arabic shaping:** no verified tool auto-generates correct OpenType Arabic shaping (positional forms, lam-alef, harakat anchors). Only YoFont claims it.
- **Gap 3, display styles:** stylized Arabic display lettering (bubbly, geometric/Kufic-like) is unserved. Arabic tools offer only preset calligraphy styles.
- **Gap 4, Arabic-native UX and market:** no Arabic-language product or MENA startup was found, while MENA brands commission custom Arabic logotypes from human foundries (29LT, TPTQ) that offer no AI service.
- **Pricing:** a plausible positioning is $5–15 per font for consumer and creator use, with higher-priced exclusive or commercial licences, plus subscriptions or an API for agencies. This is an inference from the benchmarks above, not a sourced figure.
- **Competitive threat:** YoFont, a generalist that already markets Arabic and logo-to-font, and Adobe, which could ship Glyph Ease-like features at any time, are the main threats. Lipi.ai has Arabic on its roadmap.

### Gaps
- No market-size data for Arabic custom fonts or Arabic AI font tools was found in this pass.
- No user reviews or independent tests of any Arabic output from AI font generators were found.
- The live status of several older tools (MyScriptFont, YourFonts, PaintFont, Prototypo) was not verified.
