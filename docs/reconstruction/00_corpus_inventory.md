# neural-canvas — Corpus Inventory & Categorization (Reconstruction)

*Reconstruction doc 00. Structured inventory of the neural-canvas / "AI Art Team" codebase at `/Users/nathanielcannon/Claude/Projects/neural-canvas`. Read-only analysis; nothing in the project was modified.*

## 0. Method & basis

- Enumerated every `*.py` (47 modules) and `*.md` (60 docs) outside `.git`/`.venv`.
- Class/method signatures extracted via `grep`/`ast`; purposes taken from docstrings, top-of-file comments, and the agent registry (`art_agent_team/agents/__init__.py`).
- Implementation status inferred from code presence, placeholder/TODO markers, and module size. "Stub" = present but non-functional; "partial" = works but a sub-feature is a placeholder.

## 1. Top-level layout

```
neural-canvas/
├── art_agent_team/          # the product package
│   ├── docent_agent.py      # orchestrator
│   ├── main.py              # env/logging bootstrap
│   ├── fidelity_metrics.py  # output-quality scoring
│   ├── #Segmentation Utils.py   # segmentation helpers (malformed filename)
│   ├── agents/              # the agent team (+ registry __init__.py)
│   ├── tests/               # pytest suite (unit + integration + live monitor)
│   ├── docs/                # refactor & testing-framework design docs
│   └── debug/               # error investigation log
├── run_docent_mvp.py        # entry point (MVP)
├── run_live_test.py         # entry point (live test harness)
├── demo_OpenAI.py, grok_api_test.py, grok_vision_api.py,
│   openrouter_test.py, download_demo_image.py   # API connectivity probes (scratch)
├── Art_Agent_Team_*.md (7)  # original planning corpus
├── README.md, memory.md     # overview + dev memory
├── docs/reconstruction/     # THIS reconstruction batch's outputs (00–07)
├── logs/                    # ~30 historical dev/debug/roadmap logs
├── input/ output/ workspace/ test_images/   # runtime data dirs
└── requirements.txt
```

## 2. Python modules by role

### 2.1 Orchestration
| Module | Lines | Contents | Status |
|---|---|---|---|
| `art_agent_team/docent_agent.py` | 1130 | `DocentAgent` — orchestrates the full workflow | Implemented |
| `art_agent_team/main.py` | 180 | `load_and_set_env_from_config`, `TruncateImageDataFilter` (log sanitizer) — bootstrap | Implemented |

### 2.2 Entry points
| File | Lines | Purpose | Status |
|---|---|---|---|
| `run_docent_mvp.py` | 78 | MVP runner — instantiates `DocentAgent`, runs the MVP path | Implemented |
| `run_live_test.py` | 222 | Live end-to-end test harness against real images/APIs | Implemented |
| `art_agent_team/tests/run_live_test_monitor.py` | 258 | Monitors/validates a live test run | Implemented |

### 2.3 Agents (`art_agent_team/agents/`)
| Module | Lines | Class(es) | Status |
|---|---|---|---|
| `research_agent.py` | 1400 | `ResearchAgent`, `ModelRegistry`, `PromptTemplate` | Implemented |
| `vision_agent_abstract.py` | 505 | `VisionAgentAbstract` (ABC), `BoundingBoxRegion`, `SegmentationMask`, image exceptions, `DummyVisionAgent` | Implemented (base) |
| `vision_agent_default.py` | 143 | `DefaultVisionAgent` (fallback) | Implemented |
| `vision_agent_landscape.py` | 253 | `VisionAgentLandscape` | Implemented |
| `vision_agent_animal.py` | 441 | `VisionAgentAnimal` | Implemented |
| `vision_agent_portrait.py` | 259 | `VisionAgentPortrait` | Implemented |
| `vision_agent_still_life.py` | 415 | `VisionAgentStillLife` | Implemented |
| `vision_agent_figurative.py` | 267 | `VisionAgentFigurative` | Implemented |
| `vision_agent_genre.py` | 264 | `VisionAgentGenre` | Implemented |
| `vision_agent_surrealist.py` | 251 | `VisionAgentSurrealist` | Implemented |
| `vision_agent_religious_historical.py` | 30 | `VisionAgentReligiousHistorical` | **Minimal stub** |
| `upscale_agent.py` | 374 | `UpscaleAgent` | Implemented (API-based); local-model loading stubbed |
| `placard_agent.py` | 416 | `PlacardAgent` | Implemented; smart positioning is a placeholder |
| `agents/__init__.py` | 47 | Registry: `vision_agent_classes` category→class map; re-exports | Implemented |

### 2.4 Pipeline / support
| Module | Lines | Contents | Status |
|---|---|---|---|
| `fidelity_metrics.py` | 161 | `FidelityMetrics` — composition/color/detail/emotional scoring (numpy) | Implemented |
| `#Segmentation Utils.py` | 219 | Segmentation utility helpers | **Likely orphaned** — leading `#` in filename makes it non-importable as a normal module; no import references found |

### 2.5 API connectivity probes (root, scratch — not part of product pipeline)
| File | Lines | Service probed |
|---|---|---|
| `demo_OpenAI.py` | 32 | OpenAI client smoke test |
| `grok_api_test.py` | 13 | xAI/Grok via OpenAI-compatible client |
| `grok_vision_api.py` | 46 | Grok vision (base64 image) |
| `openrouter_test.py` | 80 | OpenRouter key/connectivity check |
| `download_demo_image.py` | 20 | Fetch demo image for upscaling notebook |
| `logs/demo image.py` | 56 | Demo image script (misfiled under logs/) |
| `image_upscaling_api_demo.ipynb` | — | Upscaling-API demo notebook |

### 2.6 Tests (`art_agent_team/tests/` + root)
- **Package suite:** `test_docent_agent.py`, `test_research_agent.py`, `test_vision_agent.py`, `test_upscale_agent.py` (+ `_mvp`), `test_placard_agent.py` (+ `_mvp`), `test_artistic_integrity.py`, `test_error_detection.py`, `test_full_workflow.py`, `test_workflow_validation.py`, `test_logging_handler.py`, `integration/test_vision_workflow.py`, `conftest.py`.
- **Root (ad-hoc):** `test_docent_agent_workflow_parsing.py`, `test_research_agent_blob_typeerror.py`, `test_research_agent_logging.py` — regression repros, not in the package suite.

## 3. Agent catalog — class · purpose · key methods

### DocentAgent (`docent_agent.py`) — orchestrator
"Orchestrates the art image processing workflow for the AI Art Team project."
Key methods: `start_workflow()` (interactive/watched paradigm), `handle_request(task_description, image_path)` (programmatic paradigm), `_load_config()`, `_get_image_paths()`, `_think_with_grok()`, `_refine_agent_input_with_grok()`, `_log_to_markdown()`. Defaults to config at `art_agent_team/config/config.yaml`.

### ResearchAgent (`research_agent.py`) — identification & metadata
"Researches artwork using multiple vision LLMs concurrently and consolidates findings."
Key methods: `research_and_process()`, `categorize_artwork()` (→ genre key for vision-agent routing), `unified_research()`, `_execute_parallel_searches()`, `_consolidate_results()`, `_make_llm_call_with_retry()` / `_make_llm_call()`, `_preprocess_input()`, `_censor_response_text()`.
Helper classes: `ModelRegistry` ("Manages LLM models, API clients, and configurations" — `get_client_for_model()`, `get_models_by_capability()`); `PromptTemplate` ("Manages master and model-specific prompt generation" — `get_vision_prompt()`, `get_consolidation_prompt()`, loads art-movement data).

### VisionAgentAbstract (`vision_agent_abstract.py`) — analysis & cropping base (ABC)
Abstract base for all genre vision agents. Concrete subclasses implement `_perform_specific_analysis()`. Shared machinery: `process()`, `_create_bounding_box_image()`, `_create_segmentation_mask_image()`, `_create_cropped_images()`, `generate_outputs()`. Data classes `BoundingBoxRegion` (normalized vertices → absolute) and `SegmentationMask`; custom exceptions `CorruptedImageError`, `UnsupportedImageFormatError`, `InvalidMaskError`; `DummyVisionAgent` for tests.

### Genre VisionAgents (Landscape / Animal / Portrait / Still Life / Figurative / Genre / Surrealist / ReligiousHistorical / Default)
Each subclasses `VisionAgentAbstract` and provides genre-specific saliency/cropping using Google Vertex AI / Cloud Vision. Recurring method set: `analyze_image(image_path, research_data)` or `_perform_specific_analysis(...)`, `_calculate_object_importance()`, `_threshold_important_objects()`, `_save_labeled_version()`, `_save_masked_version()`. `VisionAgentAnimal`/`StillLife` carry richer object-importance + mock-GCP scaffolding. `DefaultVisionAgent` is the fallback. `VisionAgentReligiousHistorical` (30 lines) is a minimal stub (only a `process()` override).

### UpscaleAgent (`upscale_agent.py`) — resolution enhancement
Key methods: `upscale_image(input, output, target_width=3840, target_height=2160)` (4K/art-frame target), `_upscale_with_image_upscaling_net()`, `_upscale_with_stability_ai()`. Implemented against external upscaling APIs; local ESRGAN model loading is commented-out placeholder code.

### PlacardAgent (`placard_agent.py`) — museum placard overlay
Key methods: `add_plaque(input, output, metadata)`, `_find_optimal_plaque_position()` (**placeholder** — defaults to lower-right; `TODO` smart positioning via OpenCV low-variance detection), `_think_with_grok()`.

### FidelityMetrics (`fidelity_metrics.py`) — output QA
`validate_fidelity(original, processed, genre)` plus `_compute_composition_balance()`, `_compute_color_fidelity()`, `_compute_detail_preservation()`, `_compute_emotional_impact()`.

## 4. Vision-agent registry (`agents/__init__.py`)
`vision_agent_classes` maps category → class: `Default`→DefaultVisionAgent, `Landscape`, `Animal`, `Portrait`, `Still Life`, `Figurative`, `Genre Scene`→VisionAgentGenre, `Religious/Historical`→VisionAgentReligiousHistorical, `Surrealist`. `VisionAgentAbstract` is the (unregistered) ABC base; an `Abstract` mapping is present but commented out. `ResearchAgent`, `UpscaleAgent`, `PlacardAgent` are re-exported.

## 5. Markdown documentation index

### 5.1 Original planning corpus (root)
| File | Purpose (one line) |
|---|---|
| `README.md` | "Neural Canvas" — project overview |
| `memory.md` | Dev memory/notes file |
| `Art_Agent_Team_Plan.md` | Overall Art Agent Team plan |
| `Art_Agent_Team_Executive_Summary.md` | Executive summary of the project |
| `Art_Agent_Team_Product_Design_Plan.md` | Product design plan |
| `Art_Agent_Team_Implementation_Plan.md` | Implementation plan |
| `Art_Agent_Team_Prompt_Design.md` | Prompt design documentation |
| `Art_Agent_Team_Research_Findings.md` | Research findings |
| `Art_Agent_Team_Assessment_and_Next_Steps.md` | Analytical assessment & next steps |

### 5.2 Package docs (`art_agent_team/`)
| File | Purpose |
|---|---|
| `docs/Prompt_Design_Specifications.md` | Prompt design specifications |
| `docs/ResearchAgent_Refactor_{Executive_Summary,Product_Design_Plan,Detailed_Plan}.md` | ResearchAgent refactor design set |
| `docs/Testing_Framework_{Executive_Summary,Product_Design_Plan,Task_Accomplishment_Plan}.md` | Real-world testing framework design set |
| `debug/error_investigation_log.md` | Roo Debugger error investigation log |
| `tests/logs/docent_workflow_log.md` | Docent workflow run log |

### 5.3 `logs/` (~30 historical dev artifacts)
Roadmap planning copies (`art_ai_roadmap_*`), Perplexity-MCP design set, agent run logs (`orchestrator_log`, `coder_log`, `roo_coder_log`, `docent_workflow_log`), and ~15 `debug_*` logs (vision agent, upscale agent, placard agent, research agent verification runs). Historical only — not current spec.

### 5.4 `docs/reconstruction/` (this batch)
`00_corpus_inventory.md` (this file) plus sibling outputs `02_architecture.md`, `03_status_and_roadmap.md`, `04_adversarial_review.md`, `05_comparative_research.md` (and pending `01`, `06`, `07`). Generated by the current reconstruction run — not part of the original corpus.

## 6. Entry points (`run_*.py`)
| Entry point | Role | Status |
|---|---|---|
| `run_docent_mvp.py` | Primary MVP runner (constructs `DocentAgent`) | Implemented |
| `run_live_test.py` | Live end-to-end test against real images/APIs | Implemented |
| `art_agent_team/tests/run_live_test_monitor.py` | Live-run monitor/validator | Implemented |

## 7. Implementation status summary
- **Implemented (functional):** DocentAgent orchestration; ResearchAgent multi-LLM identification (ModelRegistry, PromptTemplate); VisionAgentAbstract + 8 genre agents (Vertex AI saliency/crop); UpscaleAgent (image-upscaling.net + Stability AI APIs); PlacardAgent overlay; FidelityMetrics; main bootstrap; both entry points.
- **Partial / placeholder:** PlacardAgent `_find_optimal_plaque_position` (lower-right default; TODO OpenCV); UpscaleAgent local ESRGAN loading (commented-out — API path used instead).
- **Minimal stub:** `VisionAgentReligiousHistorical` (30 lines).
- **Likely orphaned:** `#Segmentation Utils.py` (malformed filename, no imports).

## 8. Notable findings / gotchas
1. **Google Photos delivery is absent.** No agent or module implements the "push to Google Photos" stage named in the product goal — there is no PhotosAgent/delivery module anywhere in the corpus. Major pipeline gap (delivery endpoint missing).
2. **Config dir missing in-tree.** `DocentAgent`/agents default to `art_agent_team/config/config.yaml`, but no `art_agent_team/config/` directory exists (gitignored or expected to be created at setup). The product cannot run without it.
3. **`#Segmentation Utils.py`** — leading `#` and space make it non-importable as a standard module; treat as dead/needs-rename.
4. **Scratch probes at repo root** (`demo_OpenAI.py`, `grok_*`, `openrouter_test.py`) mix API connectivity tests into the top level; multiple LLM providers in play (OpenAI, xAI/Grok, OpenRouter, Google Gemini/Vertex).
5. **Dependencies (`requirements.txt`, 20 pkgs):** heavy Google stack (`google-generativeai`, `google-cloud-vision`, `google-cloud-aiplatform`/`vertexai`, `google-api-python-client`/`-auth`), `openai`, `Pillow`, `numpy`, `opencv-python`, `PyYAML`, `python-dotenv`, `fonttools`, `pytest`/`pytest-cov`. No Google Photos library present — corroborates finding #1.
6. **Two execution paradigms** are visible in `DocentAgent`: `start_workflow()` (interactive/watched) and `handle_request()` (programmatic) — detailed in `02_architecture.md`.

---

*End of `00_corpus_inventory.md`.*
