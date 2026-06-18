# neural-canvas — Pipeline Stages (identification → vision → upscale → placard)

**Task:** A3.2.1 · **Scope:** documentation only (no code changes). Reconstructed from the
current implementation under `art_agent_team/` on 2026-06-17. Line references are to files as
read during reconstruction and are indicative, not contractual.

> **Accuracy note up front.** The master specification lists the target pipeline as
> *identification → vision analysis → cropping → upscaling → placard overlay → Google Photos
> delivery*. The **implemented** pipeline is *identification → vision (analyze + crop) →
> upscale → placard → local file output*. **Google Photos delivery is not implemented**, and
> **smart placard positioning is a stub**. Both are documented below under their stages with an
> explicit status, because the task requires alignment with the *current* implementation.

---

## 1. Orchestration layer

All stages are driven by **`DocentAgent`** (`art_agent_team/docent_agent.py`). It owns
configuration, instantiates the stage agents, and threads a single image path through the
stages. There are **two entry points**:

| Entry point | Signature | Use | Stage selection |
|---|---|---|---|
| `start_workflow()` | `() -> None` | Interactive CLI | User picks stages via a numbered menu (`1` Research, `2` Vision, `3` Upscale, `4` Placard, `5` Full) |
| `handle_request()` | `(task_description: str, image_path: str) -> Dict` | Programmatic / single image | Forces all four stages; returns `{"final_image_path", "metadata"}` (or `{"error", ...}`) |

`handle_request()` is the **chat-docent integration point** — the non-interactive hook an
interactive/conversational docent layer calls to run one image end-to-end. `task_description`
is currently used for logging only; the conversational front-end itself is not present in
`docent_agent.py` (see §8).

### Construction & shared state
- `__init__(config_path='art_agent_team/config/config.yaml')` loads YAML config, exports API
  keys to environment variables, and sets `input_folder`, `workspace_folder`, `output_folder`,
  `log_dir`.
- Stage agents: `ResearchAgent` (instantiated **per image**), the genre-mapped vision agents
  (`self.vision_agent_classes`), and shared `self.agents['upscale']` / `self.agents['placard']`.
- Per-image state threaded through the stages:
  - `current_image_path` — the "live" image; each successful stage advances it.
  - `metadata` — consolidated research findings (drives placard text + genre).
  - `analysis_results` — optional vision analysis output, passed to the crop step.
  - `genre` — defaults to `'Default'`; set by Research; selects the vision agent.
- Execution is **strictly sequential, single image at a time**. Stages are independent `if`
  blocks; a later stage consumes whatever `current_image_path` points at.

### Textual architecture / control flow

```
                         ┌───────────────────────────────────────────┐
   input_folder/*.img ──▶│            DocentAgent (orchestrator)        │
                         │  config.yaml ▸ agents ▸ per-image state loop │
                         └───────────────────────────────────────────┘
                                            │  current_image_path, metadata, genre
        ┌───────────────────────────────────┼───────────────────────────────────┐
        ▼                                    ▼                                    ▼
┌───────────────┐   metadata,genre  ┌───────────────┐   cropped img    ┌───────────────┐
│ 1. IDENTIFY   │ ────────────────▶ │ 2. VISION      │ ───────────────▶ │ 3. UPSCALE    │
│ ResearchAgent │                   │ <Genre>Agent   │                  │ UpscaleAgent  │
│ research_and_ │                   │ analyze_image +│                  │ upscale_image │
│ process()     │                   │ copy_and_crop_ │                  │ →3840×2160    │
└───────────────┘                   │ image()        │                  └───────────────┘
   parallel multi-model             └───────────────┘   net▸stability         │ upscaled img
   vision search + consolidate          (crop binding                          ▼
                                          caveat, §5)                  ┌───────────────┐
                                                                       │ 4. PLACARD     │
                                                                       │ PlacardAgent   │
                                                                       │ add_plaque()   │
                                                                       └───────────────┘
                                                                               │ <name>_final.<ext>
                                                                               ▼
                                                                       output_folder/
                                                                       (5. Google Photos
                                                                        delivery — NOT
                                                                        IMPLEMENTED, §6)
```

Intermediate artifacts are written to `workspace_folder/` as `<name>_cropped`, `<name>_upscaled`;
the final composite is `<name>_final` in `output_folder/`. With `cleanup_workspace` (default
`True`), intermediates are deleted after a successful placard stage.

---

## 2. Stage 1 — Identification (ResearchAgent)

**Agent:** `art_agent_team/agents/research_agent.py` · **Entry:** `research_and_process(image_path)`
→ delegates to `unified_research(image_path)`.

**Flow:** preprocess input → **parallel multi-model vision search** → consolidate.
- `ModelRegistry` initialises and registers model clients by capability; `PromptTemplate`
  builds the vision and consolidation prompts (and supplies the art-movement list).
- `_execute_parallel_searches(...)` queries multiple model sources concurrently:
  `grok_text`, `gemini_text`, `grok_image`, `gemini_image`, `openrouter_text`
  (Llama Maverick + Llama Scout), `openrouter_qwen_image`, `vertex_vision_subjects`.
- `_make_llm_call_with_retry(...)` wraps each call with retry; results normalise to dicts with
  `{success, result|error}`.
- `_consolidate_results(...)` merges the per-model results into consolidated findings.
- `categorize_artwork(...)` reconciles `movement` against `date` using Grok and the recognised
  movement list, correcting the movement when the date is inconsistent.

**Output (consolidated findings):** `style`, `movement`, `date`, `genre`, `confidence_score`,
`grounding_used`, `primary_subjects`, `secondary_subjects`, plus identity fields (`title`,
`author`, `nationality`) used downstream by the placard. **`genre` selects the vision agent.**

**Error handling / fallback:** consolidation errors or exceptions leave `metadata` empty or with
an `error` key; `genre` remains `'Default'` so the pipeline can still proceed (degrading to the
default vision agent and an unlabelled-but-present placard).

---

## 3. Stage 2 — Vision analysis & cropping (genre vision agents)

**Agents:** `art_agent_team/agents/vision_agent_*.py`, all subclassing
`VisionAgentAbstract` (`vision_agent_abstract.py`). Genre→class map lives in
`agents/__init__.py`:

| Genre key | Class |
|---|---|
| `Default` | `DefaultVisionAgent` |
| `Landscape` | `VisionAgentLandscape` |
| `Animal` | `VisionAgentAnimal` |
| `Portrait` | `VisionAgentPortrait` |
| `Still Life` | `VisionAgentStillLife` |
| `Figurative` | `VisionAgentFigurative` |
| `Genre Scene` | `VisionAgentGenre` |
| `Religious/Historical` | `VisionAgentReligiousHistorical` |
| `Surrealist` | `VisionAgentSurrealist` |

**Control flow in the orchestrator:**
1. Select `vision_agent_classes[genre]`; if absent, fall back to `Default`; if `Default` is also
   missing, **skip** the stage.
2. If the agent has `analyze_image`, call `analyze_image(current_image_path, metadata)` →
   `analysis_results` (failures are caught; `analysis_results = None`).
3. If the agent has `copy_and_crop_image`, call
   `copy_and_crop_image(current_image_path, cropped_output_path, analysis_results)` →
   cropped image in `workspace_folder/`.
4. On a valid new path, advance `current_image_path`; otherwise keep the previous image.

**Shared base capabilities** (`VisionAgentAbstract`): `BoundingBoxRegion`, `SegmentationMask`,
`_create_bounding_box_image`, `_create_segmentation_mask_image`, `_create_cropped_images`,
`generate_outputs`, and the abstract `_perform_specific_analysis` / `process` contract. The
intent is genre-aware, saliency-preserving cropping toward a display (art-frame / TV) aspect.

> **⚠ Crop-binding caveat (verified).** `copy_and_crop_image` is defined **only on
> `DefaultVisionAgent`**. Every genre agent inherits from `VisionAgentAbstract` (which does
> **not** define `copy_and_crop_image`) and does **not** subclass `DefaultVisionAgent`. Because
> the orchestrator guards the crop call with `hasattr(vision_agent, 'copy_and_crop_image')`,
> **cropping only fires for the `Default` genre.** For all other genres the agent contributes
> `analyze_image` results, but the crop step is silently skipped ("No suitable method") and the
> image passes through uncropped. The genre-aware crop is therefore **partially wired**: analysis
> is genre-specific, the crop call path is not.

**Error handling / fallback:** analysis and crop are independently `try`/`except`-wrapped; any
failure logs and **continues with the pre-stage image** (no hard stop).

---

## 4. Stage 3 — Upscaling (UpscaleAgent)

**Agent:** `art_agent_team/agents/upscale_agent.py` · **Entry:**
`upscale_image(input_path, output_path, target_width=3840, target_height=2160)` (4K UHD target).

**Backends & preference:**
- `_upscale_with_image_upscaling_net(...)` — uploads to the image-upscaling.net API
  (`client_id`/`image_upscaling_id`) and polls for completion.
- `_upscale_with_stability_ai(...)` — Stability AI upscaling.
- Selected by `self.upscaler_preference`:
  - `"image_upscaling_net"` — net only (no fallback).
  - `"stability_ai"` — Stability only (no fallback).
  - `"fallback"` — try **image-upscaling.net first, then Stability AI**; return on first success.
- Readiness flags `image_upscaling_net_ready` / `stability_ai_ready` gate each backend.

**Error handling / fallback:**
- Missing input file → return `None` (orchestrator keeps previous image).
- Neither backend ready → log error, abort (`None`).
- In `fallback` mode, a failed first backend cascades to the second; if both fail, `None`.
- Orchestrator: on `None`/invalid output, **keep the previous image** and continue to placard.

---

## 5. Stage 4 — Placard overlay (PlacardAgent)

**Agent:** `art_agent_team/agents/placard_agent.py` · **Entry:**
`add_plaque(input_path, output_path, metadata)` → returns final path or `None`.

**Flow:**
1. `_think_with_grok(...)` — advisory layout "reasoning" prompt (Grok `grok-3-mini-fast-high-beta`);
   narration only, does not compute placement.
2. Load artwork as RGBA (missing/failed load → `None`).
3. Load placard background from `background_image_path`; if missing/unreadable, **fall back to a
   white background** of `fallback_bg_size`.
4. Resize placard to ~**5% of artwork width** (floor `min_plaque_width = 50px`), LANCZOS;
   apply `plaque_opacity_int`.
5. Draw dynamic text from `metadata`: `title`, `author`, `date`, `nationality`, `style`, `genre`.
6. Position via `_find_optimal_plaque_position(...)` and paste; save to `output_path`
   (`<name>_final.<ext>` in `output_folder`).

> **⚠ "Smart placard positioning" status (verified).** `_find_optimal_plaque_position` is a
> **placeholder that always returns the lower-right corner** (with configured margins). Its
> docstring/`TODO` notes the intended low-variance/saliency search (e.g. OpenCV) is **not yet
> implemented**. Placement is currently fixed, not content-aware.

**Error handling / fallback:** artwork load failure or resize failure → `None`; background load
failure → white fallback (non-fatal). On orchestrator success, intermediates are cleaned up when
`cleanup_workspace` is enabled (default `True`); on failure the pipeline keeps the pre-placard
image.

---

## 6. Stage 5 — Google Photos delivery (NOT IMPLEMENTED)

No Google Photos / album / external-delivery code exists in `docent_agent.py` or the agents.
The only network "upload" in the codebase is the **image-upscaling.net** upload inside the
upscale backend (a processing call, not delivery). The implemented pipeline **terminates at the
placard stage**, writing `<name>_final.<ext>` to the local `output_folder`. Delivery to Google
Photos is an aspirational spec item and a known gap.

---

## 7. Cross-cutting: error handling, fallbacks, configuration

**Graceful-degradation pattern (uniform across stages):** each stage is wrapped in
`try`/`except`; a per-stage missing-input guard skips the stage if `current_image_path` no longer
exists; failures are written to the run's Markdown log via `_log_to_markdown` /
`_format_log_entry`; and on any stage failure the pipeline **continues with the last good
image** rather than aborting the whole run. This means a run can complete with some stages
skipped (e.g. upscale failed → placard applied to the cropped/original image).

**Configuration (`art_agent_team/config/config.yaml`) — keys observed:**
`input_folder`, `workspace_folder`, `output_folder`, `log_dir`, `cleanup_workspace`,
`upscaler_preference`, `image_upscaling_id` (image-upscaling.net client id),
`background_image_path`, `plaque_opacity`, plus API keys exported to environment
(e.g. `GROK_API_KEY`, Gemini/Vertex, OpenRouter, Stability — consumed by the respective agents).

---

## 8. Feature integration status (per master spec)

| Spec feature | Where it lives | Status |
|---|---|---|
| identification | `ResearchAgent.unified_research` | ✅ Implemented (multi-model + consolidation) |
| vision analysis | genre `analyze_image` | ✅ Implemented (genre-specific) |
| cropping | `DefaultVisionAgent.copy_and_crop_image` | ⚠ Partial — only the `Default` genre crops (binding caveat, §3) |
| upscaling | `UpscaleAgent.upscale_image` | ✅ Implemented (net / Stability, fallback) |
| placard overlay | `PlacardAgent.add_plaque` | ✅ Implemented (text + opacity + sizing) |
| smart placard positioning | `_find_optimal_plaque_position` | ❌ Stub — fixed lower-right corner |
| Google Photos delivery | — | ❌ Not implemented |
| chat-docent | `DocentAgent.handle_request` | ⚠ Programmatic hook present; conversational front-end absent (CLI is a numbered menu) |

---

*End of `pipeline_stages.md`.*
