# neural-canvas — Revised Master Plan (Reconstruction Synthesis)

*Reconstruction doc 07 (Phase A, ticket A8). Synthesizes the six prior reconstruction
documents into one authoritative, forward-looking plan:*

- `00_corpus_inventory.md` — what exists (47 modules, agent catalog)
- `01_product_requirements.md` — validated vision & requirements
- `02_architecture.md` — actual system architecture & 8 divergences
- `03_status_and_roadmap.md` — implementation state & gaps (code-authoritative)
- `04_adversarial_review.md` — risks in the planning corpus
- `05_comparative_research.md` — adjacent tools & 2026 best practice
- `06_credential_audit.md` — needed-vs-present credentials

**Authority rule:** where sources conflict, **running code (00/03) overrides the planning
corpus (the subject of 04)**. Several "critical" planning-doc risks have already been
partially retired by the code's evolution; this plan states which still stand.

---

## 1. Executive Synthesis — where neural-canvas actually is

neural-canvas (a.k.a. "AI Art Team" / "AI Art Docent") is a multi-agent vision-AI system that
prepares fine-art images for 16:9 / 4K display **without destroying the artist's compositional
intent**. The intended pipeline is: **identify → genre-aware 16:9 crop → 4K upscale → museum
placard → deliver**.

**The honest one-line status (from 03):** the *middle* of the pipeline (identify → crop → upscale →
placard) is substantially implemented — **more advanced than the planning docs claim** — but the
product is **not end-to-end**. It has **no delivery stage**, upscaling is **cloud-API-only with no
fallback**, one genre analyzer is a **stub**, and a **research-output key mismatch** silently
degrades identification in the orchestrated path.

**The strategic position (from 05):** Samsung Frame, Meural, and Hisense Canvas all *display*
curated catalogs; **none enrich a user's own arbitrary artwork** with identification + intelligent
crop + upscale + museum placard. That enrichment is neural-canvas's genuine differentiator — it is
the *content-production front end* those closed display ecosystems lack.

**The validated core asset (from 04 §6):** the **genre-specific 16:9 cropping subsystem** is the
one real, implemented, tested asset and the project's intellectual core. Everything downstream of
the crop should be treated as near-greenfield; the identification step upstream is a research risk,
not a solved problem.

---

## 2. Validated Product Vision

> *"Teaching machines to see art the way curators do."* — `README.md`

A system that, given a source artwork image, produces a **display-ready rendition that honors the
work's composition, color, historical context, and emotional impact** (the four "artistic
sensibilities" of `01` §6), then delivers it to where the user actually views art.

**Guiding value — artistic integrity:** every transformation must respect the original. This is not
prose; it is operationalized as fidelity gates (§8).

**Differentiator — genre awareness:** a Monet landscape and a Caravaggio religious scene follow
different cropping rules (nine genre `VisionAgent` strategies, `01` §5). This per-genre polymorphism
is the "main aim of the project."

**Paradigm decision (resolving `01` §10 open question #1):** ship **Paradigm A (staged batch
pipeline)** as the product spine — it is implemented and matches the "point it at a folder, get
display-ready art" use case. Treat **Paradigm B (conversational docent)** as a *post-v1* interface
layer over the same agent core, **not** a v1 requirement (`04` H3: it is currently a persona string,
not a design). One core, two front ends — but only one front end ships first.

**Delivery-target decision (resolving `01` §10 open question #2):** adopt **Google Photos as the
universal delivery channel**. Rationale (from `05` §5–6): Google Photos feeds Samsung Frame, Meural,
and generic cast/screensaver displays, so a single upload path reaches every target display without
per-device integration. This is the missing terminal stage from `03` and `04` C4.

---

## 3. Target Architecture

### 3.1 Current architecture (from `02`)

One **orchestrator** (`DocentAgent`, ~1,130 lines) coordinating four worker-agent families,
**sequential per-image** (the threaded/queued design in the Implementation Plan was abandoned —
`02` §3.2):

| Stage | Agent | State |
|---|---|---|
| 1. Identify | `ResearchAgent` (per-image; internal thread-pool vision ensemble) | ✅ implemented |
| 2. Crop | `VisionAgent<genre>` (9 genre subclasses + Default) | ✅ implemented (the core asset) |
| 3. Upscale | `UpscaleAgent` (image-upscaling.net + Stability AI) | 🟡 partial — cloud-API only |
| 4. Placard | `PlacardAgent` (PIL overlay) | 🟡 partial — smart positioning stubbed |
| 5. Deliver | *(none)* | ❌ absent |

### 3.2 Target architecture (the deltas)

```
input/ ─▶ DocentAgent (orchestrator)
            │  per image
            ▼
        ResearchAgent ──┐  identify + genre   [FIX key-mismatch contract]
            ▼           │
        VisionAgent<genre> ─ 16:9 crop         [the validated core]
            ▼           │
        UpscaleAgent ───┤  4K                  [ADD local Real-ESRGAN fallback]
            ▼           │
        PlacardAgent ───┤  museum label        [ADD saliency-aware placement]
            ▼           │
        DeliveryAgent ──┘  Google Photos       [NEW terminal stage — appendonly scope]
            ▼
        output/<base>_final.jpg  +  Google Photos album "neural-canvas"
```

**Five structural changes** (each maps to a gap below):

1. **New `DeliveryAgent`** as the terminal stage — closes the pipeline (`03` gap #1, `04` C4).
2. **`ResearchAgent` ↔ `DocentAgent` contract fix** — orchestrator reads `individual_results` /
   `consolidated_results`; `unified_research` now returns `research_attempts` /
   `consolidated_findings`, so the orchestrator falls back to empty metadata + `Default` genre
   (`02` §5.1 / finding #4). **This is the highest-leverage one-line fix in the codebase** — it
   silently defeats genre awareness, the product's differentiator.
3. **Local upscaling fallback** (Real-ESRGAN) so "4K" does not depend on a third-party API (`03`
   gap #2).
4. **Real `VisionAgentReligiousHistorical`** to replace the 30-line templated stub that fabricates
   analysis (`03` gap #3, `00` §7).
5. **Identification hardening** — VLM hypothesis cross-checked against a retrieval catalog (`04` C2,
   `05` §1).

Defer (not v1): rewiring to the threaded/queue model (sequential is fine at current volume), the
dormant `_refine_agent_input_with_grok` loop, and the drifted `VisionAgentAbstract` ABC subsystem
(`02` §3.5, §5.2) — consolidate or delete during hardening, not before.

---

## 4. Reconciled Findings — gaps that still block, risks that still stand

The adversarial review (`04`) critiqued the **planning corpus**; the status doc (`03`) inspected the
**code**. Reconciling them prevents fixing already-solved problems:

| Planning-doc risk (`04`) | Current code reality (`00`/`03`) | Standing severity |
|---|---|---|
| **C1** 4K upscaling is mock (ESRGAN file missing) | **Re-architected** to real cloud APIs (image-upscaling.net + Stability AI) | ⬇️ Downgraded — *but* API-only with no fallback (`03` gap #2) |
| **C2** identification contradictory & infeasible | Implemented as VLM vision guess; **no** web/reverse-image search; hallucination-prone | 🔴 **Stands** — top risk |
| **C3** "artistic-fidelity CNN" is research-as-sprint | `fidelity_metrics.py` exists as **numpy heuristics**, not a CNN | ⬇️ De-scoped in practice — keep as deterministic gate, drop the CNN ambition |
| **C4** no delivery stage | Confirmed absent | 🔴 **Stands** — closes in Phase 1 |
| **H1/L2** ephemeral `:free`/`-exp`/`-beta` model IDs pinned | `ModelRegistry` exists but IDs are pinned "DO NOT CHANGE" | 🟠 Stands — needs stable IDs + fallback |
| **H2** high-polygon masks vs bbox-only tools | Code uses bounding boxes | 🟢 Resolved by de-scoping masks (use bbox) |
| **H4** dual-model aggregation unspecified | `_consolidate_results` exists (Grok consolidation) | 🟠 Partially addressed — validate quality |
| **M3** fidelity metrics at wrong stage/geometry | `fidelity_metrics` present but passive | 🟠 Stands — fix stage/geometry when promoting to a gate |

**Net:** the two genuinely product-blocking gaps are **delivery (C4)** and **identification
reliability (C2)**; the genre-aware crop is sound; upscaling/placard/fidelity need hardening, not
invention.

---

## 5. Recommended Technology Choices (from `05`)

| Stage | Recommendation | Why | Effort |
|---|---|---|---|
| **Identification** | Keep VLM as a *hypothesis*; add a **CLIP/SigLIP + vector-DB catalog** (WikiArt / public-domain museum sets in Qdrant/Weaviate) as an agree-or-flag cross-check; optional reverse-image (SerpAPI/TinEye) for provenance | Single highest-leverage reliability gain; directly retires `04` C2 | Med |
| **Upscaling** | Standardize on **Real-ESRGAN** (headless, batchable, Apple-Silicon-friendly, no subscription) as the **local fallback**; keep cloud APIs as primary; **SeedVR2** only where VRAM ≥12 GB; GFPGAN only for faces | Removes single-point API dependency (`03` gap #2); honest perceptual quality | Low–Med |
| **Cropping** | Keep the genre `VisionAgent` core; feed **VLM bbox → deterministic saliency smart-crop**; for large aspect changes prefer **generative outpainting** over cutting art away | "Preserve artistic intent" literally; reproducible geometry | Med |
| **Placard** | **PIL + `textwrap` + `textbbox`** dynamic sizing on a **dedicated label band/matte**, custom TTF; widen to planned ~20–30% (code currently ~5%, `02` §5.4) | Lowest-risk component; gallery-label aesthetic | Low |
| **Delivery** | **Google Photos `photoslibrary.appendonly`** scope; app-owned album "neural-canvas"; batch ≤50 items/call, serial per account | **Critical:** the broad `photoslibrary` scope returns 403 since 2025-04-01 (`05` §5) | Med |
| **Display target** | Google Photos → Samsung Frame / Meural / generic frame; adopt matte/anti-glare presentation conventions | Reaches every display via one upload path | Strategic |

**Metric corrections (retiring `04` M1–M3):** replace **PSNR** (undefined for a no-reference GAN
upscale) with a **perceptual/SSIM or human-rating** gate; apply **composition (IoU)** checks
**post-crop** (not post-upscale) and **color-histogram** checks where color actually changes.

---

## 6. Credential & Setup Prerequisites (from `06`)

**Vault** (correcting the original brief): the live vault is at the monorepo root —
`Project_Aion/.claude/secrets/credentials.yaml` (+ `config/credentials.local.yaml`), **not**
`alfred/.claude/secrets/`. *(Presence only was audited — no secret values were read or emitted.)*

**Provisioned and ready** (no gaps for the primary identify→analyze→upscale→placard path): Grok,
OpenAI, OpenRouter, Gemini/`GOOGLE_API_KEY`, Google Custom Search (+engine id), Google Vision,
Stability AI, image-upscaling engine id.

**Setup actions required before / during the roadmap:**

1. **Create `art_agent_team/config/config.yaml`** — the code defaults to it but the directory does
   **not exist in-tree** (`00` §8 finding #2); the product cannot run without it. Document the full
   key schema (`04` L3 — schema was never defined).
2. **Google Photos OAuth client (NEW)** — required for delivery; **none exists** (`06`). Needs an
   **OAuth 2.0 client** (not a simple key) with the **`photoslibrary.appendonly`** scope. Blocks
   Phase 1.
3. **Vertex AI (only hard credential gap)** — code offers a `vertexai` path needing GCP project +
   region + **service-account JSON / ADC**, absent from the vault. **Decide:** provision a GCP SA, or
   disable/guard the Vertex branches and rely on the API-key paths (Grok/Gemini) which appear
   primary. Recommended: **guard Vertex as optional** for v1.
4. **Wiring note (`06` §4):** agents read both lowercase config keys (`grok_api_key`) and uppercase
   env vars (`GROK_API_KEY`); `main.py` bridges config → `os.environ`. Populate whichever the chosen
   entry point expects.
5. **Bonus art-domain creds available, unused** (`06` §5): `research.rijksmuseum`,
   `image.hive_reverse_image`, `image.clarifai`, `image.ideogram`, `image.fal_ai` — candidates for
   the identification retrieval cross-check (§5) and stronger reverse-image lookup.

---

## 7. Prioritized Roadmap & Phased Delivery

Re-sequenced for the **current** code state (not the stale 4/30/2025 assessment). Each phase has an
exit milestone.

### Phase 0 — Make it run & stop the silent failures *(days)*
- **P0.1** Create `config/config.yaml` + documented schema (§6 action 1).
- **P0.2** **Fix the `ResearchAgent`→`DocentAgent` key mismatch** (`02` finding #4) — restores genre
  awareness end-to-end. *Highest value-to-effort item in the plan.*
- **P0.3** Guard/disable Vertex branches so a missing SA can't crash a run (§6 action 3).
- **M0 (exit):** a single image runs identify→crop→upscale→placard locally with **correct genre
  routing** and produces `<base>_final.jpg`.

### Phase 1 — Close the pipeline (end-to-end) *(1–2 weeks)*
- **P1.1** Build **`DeliveryAgent`** (Google Photos: OAuth `appendonly`, app-owned "neural-canvas"
  album, ≤50/batch, serial). Provision the OAuth client (§6 action 2).
- **P1.2** Wire `DeliveryAgent` into `DocentAgent` as the terminal stage; idempotent re-runs.
- **M1 (exit):** **artwork in → finished image in a Google Photos album** — the headline workflow
  completes for the first time (retires `03` gap #1 / `04` C4).

### Phase 2 — Harden the existing stages *(2–3 weeks)*
- **P2.1** **Local Real-ESRGAN fallback** in `UpscaleAgent`; verify real 4K output, never a silent
  skip (`03` gap #2).
- **P2.2** **Identification reliability:** add CLIP/vector-DB catalog cross-check (agree-or-flag);
  treat VLM output as hypothesis (`04` C2, `05` §1). Use available art creds (§6 action 5).
- **P2.3** Implement **real `VisionAgentReligiousHistorical`** (replace templated stub).
- **P2.4** **Smart placard placement** (saliency / low-variance region) + widen to ~20–30%.
- **P2.5** Introduce a **model registry with stable IDs + fallbacks** (`04` H1/L2).
- **M2 (exit):** no stub analyzers; upscaling works offline; identification has a deterministic
  cross-check; no ephemeral-model single points of failure.

### Phase 3 — Verify & qualify *(1–2 weeks)*
- **P3.1** **Integration tests** for Research→Vision→Upscale→Placard→Delivery handoff (`03` gap #5).
- **P3.2** Promote `fidelity_metrics` from passive to a **gating check** with **corrected metrics**
  (SSIM/perceptual upscale gate; IoU post-crop; color post-color-change — retires `04` M1–M3).
- **P3.3** Doc/comment cleanup: remove stale "ESRGAN" claims in `docent_agent.py`; reconcile docs to
  the cloud-API reality.
- **M3 (exit):** green end-to-end integration suite + fidelity gate enforced across all nine genres.

### Phase 4 — UX & sourcing (post-v1) *(optional)*
- **P4.1** Decision on **Paradigm B** (conversational docent) as an interface over the v1 core —
  design it properly first (`04` H3), don't ship a persona string.
- **P4.2** Optional ArtVee batch sourcing — only with ToS/credential-security design (`04` L5).
- **M4 (exit):** a second front end (chat) or a sourcing integration, on the proven v1 core.

**Critical path to a usable v1: Phase 0 → Phase 1 (M1).** Everything before M1 is plumbing the
existing parts together; M1 is the first time the product does what it claims.

---

## 8. Revised Success Criteria (corrected from `01` §8 / `04` M1–M3)

| Dimension | Revised target | Change from corpus |
|---|---|---|
| End-to-end completion | 100% of test images reach a Google Photos album | **New** — was undefined (no delivery) |
| Genre routing correctness | Correct genre agent selected (not silent `Default`) on a labeled set | **New** — exposes the key-mismatch regression |
| Crop fidelity | ≥90% retention of importance>0.8 objects; IoU measured **post-crop** | Stage/geometry corrected |
| Upscale quality | **SSIM / perceptual** gate (+ human spot-rating); local fallback verified | **PSNR retired** (wrong metric for GAN, no reference) |
| Color fidelity | Histogram corr ≥0.95 **where color changes** | Stage corrected |
| Placard legibility | Legible ≥12 pt at 4K, no occlusion of salient content | Add occlusion check |
| Identification reliability | VLM/retrieval agreement rate; flag-on-disagree | **New** — addresses C2 |
| Robustness | No silent skips; offline upscaling path proven | **New** — addresses API-only risk |

---

## 9. Risk Register (top risks to retire first)

1. **Identification reliability (`04` C2)** — VLM-only attribution hallucinates. *Mitigate:* retrieval
   cross-check (P2.2). *Owner-decision:* accept flag-on-disagree vs require human confirm for low
   confidence.
2. **Delivery scope breakage (`05` §5)** — any legacy `photoslibrary` scope is **already 403**. *Mitigate:*
   build on `appendonly` from day one (P1.1).
3. **Upscaling API single-point dependency (`03` gap #2)** — *Mitigate:* Real-ESRGAN fallback (P2.1).
4. **Silent genre-routing failure (`02` #4)** — *Mitigate:* P0.2 + the genre-routing success metric.
5. **Ephemeral model IDs (`04` H1)** — *Mitigate:* registry + fallbacks (P2.5).
6. **Vertex credential gap (`06`)** — *Mitigate:* guard as optional (P0.3); provision SA only if a
   Vertex-specific capability is required.
7. **Provenance of past "achievements" (`04` L6)** — historical "100% / 9 images" claims are
   AI-authored and unverified; **re-establish a real test baseline in Phase 3**, don't trust the logs.

---

## 10. Bottom Line

neural-canvas is **architecturally complete in skeleton** with a genuinely strong, differentiated
core (genre-aware cropping) and a credential posture that is **ready for everything except Google
Photos delivery and optional Vertex**. It is **two focused phases from a working v1**: fix the silent
genre-routing regression and stand up the config (Phase 0), then build the Google Photos delivery
stage (Phase 1, milestone M1 — "artwork in → finished image in Google Photos"). Hardening
(identification cross-check, local upscaling fallback, the religious/historical analyzer, smart
placard, honest metrics) follows in Phases 2–3. The conversational docent and ArtVee sourcing are
real but **post-v1** — design them properly rather than shipping the persona string the planning
corpus left behind.

The corpus oversold the flashy parts (upscaling, a fidelity CNN, a chat docent) and undersold the
real one (the crop). This plan inverts that: **ship the validated core end-to-end first, harden
second, expand interfaces last.**

---

*End of `07_revised_master_plan.md`. Synthesized from reconstruction docs 00–06. No secret values
were read or emitted; credentials are referenced by presence and key-path only.*
