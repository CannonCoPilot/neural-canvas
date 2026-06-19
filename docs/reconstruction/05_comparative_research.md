# neural-canvas — Comparative Research (Reconstruction)

**Task:** A6 — comparative research on tools/software adjacent to neural-canvas's pipeline.
**Method:** Web search (June 2026), six target areas mapped to neural-canvas's pipeline stages.
**Scope:** Approach, maturity, and "what neural-canvas can borrow" per area — no code-level prescriptions.

neural-canvas's pipeline is: **identify artwork → vision analysis → crop to TV/frame aspect (preserving artistic intent) → upscale → overlay museum placard → push to Google Photos.** Each section below maps to one of those stages.

---

## 1. Artwork Identification

### Landscape
- **Consumer apps:** *ArtScan* (dedicated AI painting recognition + art-history chat), *Google Lens* (free, ubiquitous, but returns generic web results rather than curated art metadata), *Smartify* (museum-partnered recognition across 700+ venues with curated/audio content), *Google Arts & Culture* (virtual-museum exploration), *Magnus* (contemporary-market pricing), *TinEye* (reverse-image provenance — finds exact and modified copies, builds a usage trail).
- **Developer building blocks:** There is **no official Google Lens API**. Two distinct "reverse image search API" categories exist:
  1. *Public-web visual search* — TinEye, Bing Visual Search, and unofficial Google Lens wrappers (SerpAPI ~100 searches/mo free then ~$50/mo; Apify actors with ~$5/mo free credit).
  2. *Your-own-catalog vector search* — embed the query image with a **CLIP- or SigLIP-family encoder**, index against a known-artwork catalog in a vector DB (Pinecone / Weaviate / Qdrant), retrieve via ANN (HNSW/IVF) ranked by cosine similarity. Maximum control, maximum ops work.

### Maturity
General visual search is mature and commoditized. **Art-specific** precision (correct title/artist/date) requires either a museum-partnered service (Smartify) or your own curated embedding catalog; raw web search is noisy for attribution.

### What neural-canvas can borrow
neural-canvas currently identifies via **VLM vision calls** (Grok/GPT vision through OpenRouter). That is flexible but can hallucinate attributions. Pragmatic upgrades:
- Add a **CLIP + vector-DB catalog** (e.g., embeddings of WikiArt / public-domain museum collections) as a deterministic cross-check on the VLM's title/artist guess — agree-or-flag, raising confidence.
- Use a **reverse-image API (SerpAPI/Apify Google Lens or TinEye)** as a secondary confirmation signal and for provenance.
- Treat VLM identification as a *hypothesis* and reconcile against retrieval — this is the single highest-leverage reliability improvement for the identification stage.

---

## 2. AI Upscaling

### Landscape
| Tool | Type | Notes |
|---|---|---|
| **Topaz Gigapixel** | Paid, local | Gold standard; best 4x fine-detail (hair/text). Now subscription (~$199/yr in Photo AI bundle; formerly $99 perpetual). Lightroom/PS plugins. |
| **Upscayl** | Free, open source (AGPL-3.0) | Best free option; wraps **Real-ESRGAN**; runs on Apple Silicon/AMD; ~90% of Topaz quality at 2x; commercial use OK (only forks of the app must open-source). |
| **Real-ESRGAN (standalone)** | Open source | Developer favorite; CLI / chaiNNer for batch pipelines; custom-trainable; built-in face enhancement; slower than Topaz. |
| **GFPGAN** | Open source | **Face-restoration** model specifically — not a general upscaler. |
| **SeedVR2** | Open source (2026) | New diffusion upscaler; beats Real-ESRGAN in blind tests (≈4.1 vs 3.5/5), near Topaz; needs **12–16 GB VRAM**. |

### Maturity
Very mature. Open-source GAN upscalers (Real-ESRGAN) are production-ready and scriptable; diffusion upscalers (SeedVR2) are the emerging quality leader but hardware-hungry.

### What neural-canvas can borrow
For a **silent, automated, local** pipeline, **Real-ESRGAN via CLI/API is the natural fit** — open, headless, batchable, Apple-Silicon-friendly, no subscription or upload. Reserve **SeedVR2** as a higher-quality option where GPU VRAM allows and the target is large prints/4K frames. Avoid Topaz for an unattended pipeline (GUI/subscription/licensing friction). Add **GFPGAN/face-enhance** only when a work contains faces that the base upscaler softens. Mind Upscayl's AGPL terms if ever redistributing.

---

## 3. Content-Aware / Saliency Cropping

### Landscape
- **Seam carving** (Avidan & Shamir, 2007): foundational; removes/inserts low-energy "seams." Weakness: no semantics → distorts key objects under aggressive resizing.
- **Saliency-based smart crop:** *top-down* (face/object detectors) vs *bottom-up* (visual saliency maps). Production libraries (smartcrop.js, Filestack-style) center the crop on the most salient region and work regardless of content.
- **2024–25 state of the art — hybrid pipelines:** *PruneRepaint* (Shen et al., 2024) fuses seam-carving energy with a spatially-weighted **saliency prior** and **diffusion repainting** to retarget to arbitrary aspect ratios without object distortion. *Content-aware seam carving (CSC)* adds semantic + spatial priors. Multi-crop partitioning selects several non-overlapping candidate crops.

### Maturity
Saliency smart-crop libraries are production-ready today. Diffusion-repaint retargeting (PruneRepaint) is research-grade but directly addresses the "don't mangle the art" requirement.

### What neural-canvas can borrow
The requirement is "crop to TV/frame aspect **preserving artistic intent**":
- **Default:** a **deep saliency-map smart crop** — reframe to 16:9 while keeping the focal subject, far safer than naive seam carving (which distorts).
- **For large aspect changes:** prefer **generative outpainting / repainting** to *extend* the canvas (matting the work, or diffusion-extending the background) rather than cutting art away — arguably the most faithful interpretation of "preserve artistic intent."
- neural-canvas's **VisionAgent** already produces an analysis; feed a **VLM-derived subject bounding box / focal description** into a deterministic crop step (best of both: semantic understanding + reproducible geometry).

---

## 4. Placard / Caption Overlay Generation

### Landscape
- **Pillow (PIL)** is the standard: `ImageDraw.text()` + `ImageFont.truetype()`. Pillow has **no built-in text wrapping** — combine with the stdlib `textwrap` module; use **`textbbox`** for precise centering/dynamic sizing, and `draw.rectangle()` to lay a solid/semi-transparent label panel behind text for legibility.
- **pynter** is a thin caption-banner layer over PIL (`generate_captioned(...)`, with a `filter_color` semi-transparent darkening overlay for legibility over busy imagery).

### Maturity
Extremely mature; trivial, pure-Python dependency. The lowest-risk component in the entire pipeline.

### What neural-canvas can borrow
neural-canvas's **PlacardAgent** almost certainly already uses PIL. Recommended conventions for a museum-label aesthetic:
- Render the placard on a **dedicated solid band/matte** (title / artist / date / medium) rather than overlaying directly on the artwork — preserves the art and reads like a gallery wall label. A translucent lower-third is the alternative when band space is unavailable.
- Use **`textbbox`-driven dynamic sizing** + `textwrap` for variable-length metadata, and a **custom TrueType font** for typographic fidelity.

---

## 5. Google Photos API Integration

### Landscape — a breaking change neural-canvas must heed
Google restructured the Photos APIs effective **April 1, 2025**:
- **Scopes removed:** `photoslibrary`, `photoslibrary.readonly`, `photoslibrary.sharing` → calls relying on them now return **403 PERMISSION_DENIED**.
- **Uploads still work** via the **`photoslibrary.appendonly`** scope. The two-step flow is unchanged: upload bytes → receive `uploadToken` (valid 1 day) → `mediaItems.batchCreate`.
- **Library API is now app-created-content-only:** items can only be added to **albums your app created**; reading user library at large moved to the **Picker API**; `photoslibrary.readonly.appcreateddata` / `photoslibrary.edit.appcreateddata` for app-created data.
- **Constraints:** `batchCreate` ≤ **50 items/call**; don't run parallel `batchCreate` per user; 200 OK or 207 MULTI-STATUS on partial success.

### Maturity / risk
Stable for the *upload* use case, but the 2025 scope purge silently breaks older integrations.

### What neural-canvas can borrow / must fix
neural-canvas only **pushes finished art** to Google Photos — squarely in the still-supported path. **Action items:**
- Ensure delivery uses the **`photoslibrary.appendonly`** scope; any legacy code requesting the broad `photoslibrary` scope is **now broken** and must migrate (flag this for the A7 credential audit and A4 status doc).
- Create and target a **dedicated app-owned album** ("neural-canvas") since cross-album writes are restricted to app-created albums.
- Batch deliveries in ≤50-item calls, serially per account.

---

## 6. End-to-End Digital-Frame Products

### Landscape
| Product | What it is | 2026 notes |
|---|---|---|
| **Samsung Frame / Frame Pro** | Art-mode 4K TV | Frame Pro (LS03HW) adds Neo QLED Mini-LED; matte anti-glare; 5,000+ works (Art Store subscription); custom bezels cost extra; displays your own photos; One Connect box now wireless. |
| **Meural Canvas (II)** | Dedicated art frame | 40,000-work library; upload art/photos/video/animated via app; the strongest *pure* art frame. |
| **Hisense CanvasTV** | Art TV, value pick | Wood frame **included**; subscription-free ~1,000+ works; ~$687 (≈$260 cheaper than an equivalently-bezeled Frame). |
| **Amazon Ember Artline** | Art TV | Free art library, motion sensor, included frame; but photography-heavy catalog and not flush-mount. |

### Maturity
Mature consumer category (Samsung Frame dates to 2017 and still leads on overall art experience).

### What neural-canvas can borrow / strategic positioning
These products **display** curated catalogs; none **enrich the user's own arbitrary artwork** with identification + intelligent crop + upscale + museum placard. That is neural-canvas's differentiator. Borrowables:
- **Presentation goals:** matte/anti-glare look, art-mode framing/matting conventions, and **motion-sensor power-saving** behavior.
- **Delivery target:** the **Samsung Frame can display your own photos**, and most frames ingest from cloud photo libraries — so **Google Photos is a sound universal delivery channel** to feed a Frame/Meural/Hisense or a generic cast/screensaver display. neural-canvas is effectively the *content-production* front end these closed ecosystems lack for personal artwork.

---

## Cross-Cutting Recommendations

| Pipeline stage | Borrow / adopt | Confidence |
|---|---|---|
| Identification | VLM hypothesis **+ CLIP/vector-DB catalog cross-check**; reverse-image (SerpAPI/TinEye) for provenance | High value, medium effort |
| Upscaling | **Real-ESRGAN** (headless/batch); SeedVR2 if VRAM ≥12 GB; GFPGAN for faces | High value, low effort |
| Cropping | **Deep saliency smart-crop**; generative **outpainting** for large aspect changes; VLM bbox → deterministic crop | High value, medium effort |
| Placard | PIL + `textwrap`/`textbbox`, dedicated label band, custom TTF | Low risk, low effort |
| Delivery | **`photoslibrary.appendonly` scope**, app-owned album, ≤50/batch — *migrate any legacy `photoslibrary` scope* | Critical fix |
| Display | Target Google Photos → Samsung Frame / Meural / generic frame; adopt matte presentation + motion sensing | Strategic |

**Top three actions:** (1) harden identification with a retrieval cross-check; (2) standardize on Real-ESRGAN for unattended upscaling; (3) verify/migrate the Google Photos scope to `appendonly` before it silently fails.

---

## Sources

**Identification:** [Best Painting Recognition Apps 2026 (ArtScan)](https://paintingrecognition.com/blog/best-painting-recognition-apps-compared.html) · [Art Identifier Apps 2026](https://www.aiimagedetector.com/blog/art-identifier-app) · [Reverse Image Search API Guide 2026 (Vecstore)](https://vecstore.app/blog/reverse-image-search-api) · [Google Lens API (Apify, unofficial)](https://apify.com/api/google-lens-api)

**Upscaling:** [Best AI Image Upscalers 2026 (LetsEnhance)](https://letsenhance.io/blog/all/best-ai-image-upscalers/) · [Topaz Gigapixel Pricing + Free Alternatives 2026](https://www.sammapix.com/blog/best-free-topaz-gigapixel-alternatives-2026) · [Best Free Topaz Alternatives Tested (DEV)](https://dev.to/samma1997/best-free-topaz-gigapixel-alternatives-2026-7-tools-tested-26lc) · [Best AI Upscaler 2026 (Apatero)](https://apatero.com/blog/best-ai-image-upscaler-2026-topaz-magnific-upscayl)

**Cropping:** [Content-Aware Adaptive Cropping (EmergentMind)](https://www.emergentmind.com/topics/content-aware-adaptive-cropping) · [Prune and Repaint: Content-Aware Retargeting (arXiv 2410.22865)](https://arxiv.org/pdf/2410.22865) · [Smart Image Cropping Using Saliency (Filestack)](https://blog.filestack.com/smart-image-cropping-using-saliency/) · [Seam Carving Design Techniques (arXiv 2410.21207)](https://arxiv.org/html/2410.21207v1)

**Placard overlay:** [Wrap & Render Multiline Text with Pillow (DEV)](https://dev.to/emiloju/wrap-and-render-multiline-text-on-images-using-pythons-pillow-library-2ppp) · [Adding Text to Images with PIL (Cloudinary)](https://cloudinary.com/guides/image-effects/a-guide-to-adding-text-to-images-with-python) · [pynter caption layer on PIL (GitHub gist)](https://gist.github.com/turicas/1455973)

**Google Photos API:** [Updates to the Google Photos APIs](https://developers.google.com/photos/support/updates) · [mediaItems.batchCreate reference](https://developers.google.com/photos/library/reference/rest/v1/mediaItems/batchCreate) · [Google Photos API Deprecation (memoryKPR)](https://memorykpr.com/blog/google-photos-api-deprecation-what-it-means-for-third-party-apps-and-how-to-prepare/)

**Digital frames:** [Meural Canvas vs Samsung Frame (oneSmartcrib)](https://www.onesmartcrib.com/meural-canvas-vs-samsung-frame/) · [Best Art TVs 2026 (Yahoo Tech)](https://tech.yahoo.com/home-entertainment/tvs/article/best-art-tvs-140642933.html) · [Samsung Frame Pro vs Frame 2026 (Big Screen Store)](https://thebigscreenstore.com/blogs/news/samsung-frame-pro-vs-frame-2026)

---

*End of `05_comparative_research.md`.*
