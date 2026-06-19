# Neural Canvas — Product Requirements & Vision (Reconstructed)

> **Reconstruction task A2** (Pulse `AION-2c16c139`). Synthesized from `README.md`,
> `Art_Agent_Team_Executive_Summary.md`, `Art_Agent_Team_Product_Design_Plan.md`,
> `Art_Agent_Team_Plan.md`, `Art_Agent_Team_Prompt_Design.md`,
> `Art_Agent_Team_Assessment_and_Next_Steps.md`, `memory.md`, and code intent
> (`run_docent_mvp.py`, `art_agent_team/docent_agent.py`, `agents/`).
>
> **Naming:** the repository README brands the product **"Neural Canvas"**; the
> planning corpus calls it the **"AI Art Team" / "AI Art Docent."** They are the
> same project. This document uses *Neural Canvas* for the product and the
> documented agent names for components.
>
> **Fidelity note:** where a requirement is *designed but not implemented*, or is
> *inferred* rather than stated, this document says so explicitly. Claims the
> source corpus does **not** support are listed under §10 rather than asserted.

---

## 1. Product Vision

> *"Teaching machines to see art the way curators do."* — `README.md`

Neural Canvas is a multi-agent vision-AI system that prepares fine-art images for
display on modern 16:9 screens **without sacrificing the artist's compositional
intent.** Where a naïve center-crop bisects a focal subject or amputates a
horizon line, Neural Canvas first *understands what it is looking at* — artist,
title, period, movement, genre, and the location of every significant object —
and only then reframes, enhances, and labels the work to a museum standard.

The guiding value is **artistic integrity**: every transformation must respect
the original work's composition, color balance, historical context, and
emotional impact (`Art_Agent_Team_Product_Design_Plan.md` §"Artistic
Sensibilities"). The tagline captures the ethos: *"Treating every pixel with the
respect the artist intended."*

---

## 2. Core Goal & Problem Statement

**Problem.** Resizing a painting is not the same as resizing a photograph.
Generic crop/resize tools have no model of *what matters* in an image, so they
routinely destroy the very composition that makes a work worth displaying
(`README.md`, "Teaching Machines to See Art").

**Goal.** Given a source artwork image, produce a display-ready rendition through
an end-to-end pipeline:

1. **Identify** the work — artist, title, date, nationality, style, movement,
   primary/secondary subjects — via multi-model research with confidence scoring
   and art-historical date-range validation.
2. **Analyze & crop** to a **16:9** aspect ratio using *genre-specialized* logic
   that preserves the focal composition (target: retain ≥90% of high-importance
   objects).
3. **Upscale** the crop to **4K (3840×2160)**, preserving brush strokes, paint
   layering, and medium texture while minimizing artifacts.
4. **Overlay a museum-style placard** (title, artist, nationality, date) in a
   textured white-cardstock aesthetic, positioned to avoid occluding key content.
5. **Output** a final image suitable for **modern digital displays, large
   screens, or prints** (`Art_Agent_Team_Product_Design_Plan.md` §2).

The differentiator is step 2's **genre awareness**: a Monet landscape and a
Caravaggio religious scene follow entirely different cropping rules.

---

## 3. The Two Paradigms

The corpus describes — and the codebase partially realizes — **two distinct ways
of operating** the same underlying agent team. They share the Research → Vision →
Upscale → Placard component stack; they differ in how work is initiated and how
the user interacts.

### Paradigm A — Staged batch pipeline *(implemented)*

The shipped `DocentAgent.start_workflow()` is a **non-conversational,
folder-driven batch processor.** It discovers all supported images in a
configured **input folder**, then presents a numbered **stage-selection menu**
and processes each image through the chosen stages:

```
1. Research  (identify / extract metadata)
2. Vision    (intelligent 16:9 crop)
3. Upscale   (enhance to 4K)
4. Placard   (museum-style label)
5. Full workflow (all stages)
```

Source: `art_agent_team/docent_agent.py` (the code comment is explicit —
*"Display workflow options as a numbered list (reverted to original,
non-conversational). No persona prompt or conversational placeholder."*) and
`run_docent_mvp.py`. This is the "silent"/automated paradigm referenced in the
reconstruction brief; note it is an **input-folder batch**, not a live
filesystem *watcher* — no watch-folder daemon exists in the code.

### Paradigm B — Conversational art docent *(designed, not implemented)*

`Art_Agent_Team_Prompt_Design.md` defines a `docent_persona_prompt` casting the
DocentAgent as *"an expert art historian … whose core mission is to assist users
by analyzing, discussing, restoring, and presenting artworks they share."* Its
documented capabilities map directly onto the brief's *identify / explain /
search / trigger* model:

| Brief verb | Documented capability (`Art_Agent_Team_Prompt_Design.md` §1) |
|---|---|
| **Identify** | (1) Identification & Analysis — artwork origin, style, key features |
| **Explain** | (2) Insightful Description + (3) Engaging Discussion of context & significance |
| **Search** | (1)/(3) contextual analysis; (Assessment §7) ArtVee favorites scraping |
| **Trigger** | (8) **Adaptive Workflow Execution** — interpret free-form requests to autonomously select/sequence Research, Vision, Upscale, Placard, Cropping |
| (cross-cutting) | (7) **Conversational Interaction** — natural bidirectional dialogue |

This paradigm is a **design intent only.** The implemented agent deliberately
reverted away from the conversational persona to the numbered menu (see Paradigm
A). Reconciling the two is an open product decision (see §10).

---

## 4. Capability Requirements (end-to-end)

| # | Capability | Requirement summary | Source |
|---|---|---|---|
| R1 | **Artwork identification** | Multi-model "committee" research: parallel fan-out to ~7 vision models, consolidated by a reasoning model with confidence scoring and art-historical **date-range cross-validation** against a movement catalog. Extract artist, title, date, nationality, style, movement, primary/secondary subjects, confidence, grounding, limitations. | README; Exec Summary; Prompt_Design §2 |
| R2 | **Genre classification** | Assign exactly one of 9 styles: abstract, landscape, still life, surrealist, genre-scene, animal-scene, portrait, figurative, religious/historical (+ Default fallback). Drives downstream cropping logic. | Plan §6a; README |
| R3 | **Intelligent 16:9 crop** | Genre-specialized `VisionAgent` selects the crop that preserves focal composition using detected bounding boxes / segmentation masks and importance scores; primary subject boosted (2× next-highest or raw, whichever greater). | Plan §10,17; Product Design §1 |
| R4 | **4K upscaling** | Enhance crop to 3840×2160, preserving brush stroke / texture / edge detail; reduce artifacts. ESRGAN-class model preferred; dual-backend (Stability AI primary, image-upscaling.net fallback). | Exec Summary; Product Design §2; README |
| R5 | **Museum placard overlay** | Textured white-cardstock label, black sans-serif text (Title bold / Artist italic / Nationality, Date), lower-right, ~20–30% width, legible ≥12pt at 4K, semi-transparent, positioned to avoid key content. | Product Design §3 |
| R6 | **Artistic-integrity validation** | Quantitative fidelity gates: composition (IoU ≥0.9 vs. original object positions), color (histogram correlation ≥0.95), detail (Laplacian), emotional-impact (contrast) — thresholds adjusted per genre. Flag low-fidelity outputs for review. | Product Design §4; README (FidelityMetrics) |
| R7 | **Output delivery** | Write cropped / upscaled / final + fidelity report as discrete files under a structured output directory with clear naming (`<basename>_cropped/_upscaled/_final`). Final renditions target modern 16:9 displays, large screens, or prints. | Product Design §2,§"User Experience" |
| R8 | **Parallel/async orchestration** | DocentAgent manages an image "stack"/queue and a pool of ResearchAgent workers via thread-safe queues; routes each image to the correct genre VisionAgent, then to Upscale and Placard. | Plan §18,19; README |

---

## 5. Genre-Aware Processing (the core differentiator)

Nine `VisionAgent` subclasses inherit a shared abstract base (`BoundingBoxRegion`,
`SegmentationMask`) and each implement distinct cropping strategy
(`Art_Agent_Team_Plan.md` §17):

| Genre | Cropping intent |
|---|---|
| Landscape | Preserve horizon line & wide vistas; truncate primary subject bottom-up; warn if primary subject truncated |
| Portrait | Prioritize facial features; discount background; keep secondary faces if possible |
| Religious/Historical | Preserve central narrative action & primary subject; discount crowds/background |
| Surrealist | Context-based scoring; otherwise follow Landscape logic |
| Still Life | Preserve centrality of the central object cluster |
| Animal-scene | Animal-face detection (beak, snout, muzzle, mane); discount distant background |
| Figurative | Faces first, then figures, then original centrality |
| Genre-scene | Discount crowds/background; emphasize foreground primary/secondary subjects |
| Abstract | Weighted centroid of significant shapes; center crop on centroid |
| Default | Center-weighted 16:9 fallback |

This per-genre polymorphism — not a single universal cropper — is the explicit
"main aim of the project" (`Art_Agent_Team_Plan.md`, "Focusing and Future Steps").

---

## 6. Artistic-Integrity Principles (product values)

All processing is constrained by four defined **"artistic sensibilities"**
(`Art_Agent_Team_Product_Design_Plan.md` §"Artistic Sensibilities Definition"):

- **Composition** — preserve arrangement, balance, focal points; guide the eye as
  the artist intended.
- **Color balance** — avoid tonal/contrast shifts that change mood or style.
- **Historical context** — reflect the work's era/movement in metadata and
  processing choices (e.g., no modern color profiles on historical works).
- **Emotional impact** — never crop out or wash out the elements carrying the
  work's emotional resonance.

These are not aspirational prose only: they are operationalized as the §R6
fidelity gates.

---

## 7. Target Users & Use Cases

The corpus does not define formal personas; the following are **inferred** from
stated objectives and features:

- **Art enthusiasts / collectors** who want to display fine art on 16:9 screens,
  smart-TV art modes, or digital frames at 4K with museum-quality framing and
  labels — the "modern digital displays / large displays or prints" target
  (`Product Design Plan` §2).
- **Curators / educators** who would benefit from the conversational docent's
  identify-and-explain capabilities (Paradigm B) for learning and discussion.
- **Users with existing art-image libraries** processed in bulk via the
  input-folder batch (Paradigm A).
- **(Aspirational) ArtVee.com users** — a Medium-Low priority item proposes
  scraping a user's ArtVee "Favorites" (with their login) to source
  high-resolution images for processing (`Assessment` §7).

Primary use case: *"point the system at a folder of artwork (or hand it one
piece in conversation) and receive display-ready, intelligently-cropped,
upscaled, placarded renditions that honor each work's composition."*

---

## 8. Success Criteria & Acceptance Metrics

Concrete, source-stated benchmarks (`Art_Agent_Team_Assessment_and_Next_Steps.md`
§"Success Metrics"; `Art_Agent_Team_Product_Design_Plan.md` §1,§4):

| Dimension | Target | Source |
|---|---|---|
| Upscaling quality | PSNR ≥ 30 dB | Assessment; Product Design §2 |
| Upscaling speed | < 30 s / image (e.g. 8 GB-VRAM GPU) | Assessment; Product Design §2 |
| Crop fidelity | IoU ≥ 0.9 for high-importance objects; ≥90% of objects with importance >0.8 retained | Assessment; Product Design §1,§4 |
| Color fidelity | Histogram correlation ≥ 0.95 vs. original | Product Design §4 |
| Overall artistic fidelity | Composite fidelity score ≥ 0.85 (flag below) | Product Design §4 |
| Placard legibility | Text legible at 4K (≥12 pt) | Assessment; Product Design §3 |
| Conversational UI | User task completion < 2 min | Assessment |
| Workflow stability | 100% completion without errors — *achieved for 9 test images* in live testing | Assessment |
| (Aspirational) ArtVee scraping | 95% successful downloads under test | Assessment §7 |

**Strategic goal:** a functional **beta** incorporating upscaling and placard
features, with iterative testing for fidelity across genres
(`Art_Agent_Team_Executive_Summary.md` §"Strategic Goals"; implementation plan
targeted a beta by mid-2025).

---

## 9. Implementation Status at a Glance *(context for requirements; full detail is A4's scope)*

- **Implemented:** intelligent 16:9 cropping with genre-specific VisionAgents;
  multi-model research/identification; staged batch DocentAgent (Paradigm A).
- **Placeholder / partial:** `UpscaleAgent` (R4) and `PlacardAgent` (R5) are
  repeatedly described as placeholders "awaiting full implementation"
  (`Executive Summary` §"Project Status"; `Assessment` §"Gaps").
- **Designed, not built:** conversational docent (Paradigm B); CNN-based
  fidelity scorer; ArtVee scraping.

This is included only so the requirements are read against reality; the
authoritative status/roadmap reconstruction is task A4 (`03_status_and_roadmap.md`).

---

## 10. Scope Boundaries, Open Questions & Corpus Divergences

**Explicitly NOT evidenced in the corpus (do not assume as requirements):**

- **Google Photos delivery.** The reconstruction brief mentioned "send to Google
  Photos," but **no Google Photos / photo-library integration exists anywhere**
  in the docs or code. The only "upload" calls are to the **image-upscaling.net**
  API (`upscale_agent.py`) and a Gemini **files** upload helper
  (`Last_instructions.txt`). The stated delivery target is local output files for
  *digital displays / prints*. Treat Google Photos as an unconfirmed
  aspiration pending owner confirmation, **not** a current requirement.
- **Live watched-folder daemon.** Paradigm A is input-folder batch processing on
  demand, not a background filesystem watcher.

**Open product questions:**

1. **Paradigm reconciliation** — ship the conversational docent (Paradigm B), the
   batch menu (Paradigm A), or both behind a shared core? The code chose A and
   shelved B; the design corpus still centers B.
2. **Delivery target** — confirm the intended end destination (digital frame /
   smart-TV art mode / Google Photos / local export) since the brief and corpus
   disagree.
3. **Upscaling backend** — ESRGAN (Product Design) vs. Stability AI /
   image-upscaling.net (README/code) as the canonical engine.

---

## 11. Source Traceability

| Source | Primary contribution to this document |
|---|---|
| `README.md` | Vision, architecture, genre table, tech decisions, output target |
| `Art_Agent_Team_Executive_Summary.md` | Objectives, component roles, planned Upscale/Placard, strategic goals, LLM constraints |
| `Art_Agent_Team_Product_Design_Plan.md` | Feature specs (crop/upscale/placard), artistic sensibilities, fidelity metrics, UX considerations |
| `Art_Agent_Team_Plan.md` | Two-paradigm workflow detail, genre cropping rules, async orchestration, primary-subject scoring |
| `Art_Agent_Team_Prompt_Design.md` | Conversational docent persona & capabilities (Paradigm B); research field schema |
| `Art_Agent_Team_Assessment_and_Next_Steps.md` | Gaps, success metrics/benchmarks, ArtVee scraping, beta priorities |
| `memory.md` | (empty stub — no content) |
| `run_docent_mvp.py`, `docent_agent.py`, `agents/` | Code intent: implemented batch/menu paradigm; placeholder Upscale/Placard; no Google Photos/watcher |

---

*Reconstructed for Phase A (neural-canvas). Requirements are derived strictly
from the source corpus; aspirational and unevidenced items are flagged, not
asserted.*
