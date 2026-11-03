# Neural Canvas — full pipeline code review (IN PROGRESS, started 2026-11-02 by W0:Jarvis)

Sir's order: review → redesign as Alfred Nexus/Pulse task (watched folder → kanban task → run) → rebuild on local
Aion models → fix E2E → iterate → show a successful run on all 12 test artworks. Full scope: Project_Aion scratchpad.

## Inventory (verified 2026-11-02)
- `art_agent_team/` ≈ 10,742 lines of Python incl. 17 test files. Largest: `agents/research_agent.py` 1,433,
  `docent_agent.py` 1,185 (orchestrator), `placard_agent.py` 416, `upscale_agent.py` 374, `delivery_agent.py` 156,
  `identification/artwork_catalog.py` 189 (CLIP+Qdrant), `fidelity_metrics.py` 161, `main.py` 180.
- **Nine genre VisionAgent classes** (abstract 505, animal 441, still_life 415, figurative 273, genre 270,
  portrait 265, surrealist 257, landscape 253, default 143) ≈ 2,800 lines.
- Uncommitted at start: `docent_agent.py` — **docstring-only** rewrite (describes the current sequential design;
  says the threaded/queue design was removed). Accurate; safe to commit.

## Findings — full review (2026-11-02, static analysis + local pytest, no cloud calls)

### Severity summary

| Severity | Count | Blocks a rebuild? |
|---|---|---|
| Critical | 5 | Yes — must fix before any E2E run counts as real |
| High | 7 | Yes — breaks the stated design even when the code "runs" |
| Medium | 9 | Should fix during rebuild |
| Low | 6 | Cleanup, not blocking |

### Critical

**C1. Live xAI API key committed in plaintext, already pushed to the public repo.**
`grok_api_test.py:4` (`api_key="xai-...`) and `grok_vision_api.py:6` (`XAI_API_KEY = "xai-...`) hardcode a real
Grok key (not a `YOUR_...` placeholder). Both files are tracked (`git ls-files` confirms) and the key is present in
commit `ab90f4a` ("Initial import..."), which is on `origin/main` of the **public** `CannonCoPilot/neural-canvas`
repo — i.e. this key has been exposed publicly since the initial import.
*Fix*: rotate the key at the xAI console now (treat as already-leaked, not "at risk"); then scrub it from history
(`git filter-repo` or BFG) before any further push; add a pre-commit secret scan (gitleaks or similar); never
hardcode a key in a demo script — read from env/config like every other agent does.

**C2. Identification is cheating off the filename, through two independent paths.**
(a) `research_agent.py`'s `PromptTemplate._load_master_template()` embeds `"Filename tokens: {filename_tokens}\n"`
(research_agent.py:371) directly into every vision-model prompt; `_preprocess_input()` (research_agent.py:735)
extracts those tokens straight from `os.path.basename()`. Since the 12 test images are named with their exact
title/artist/date, the VLM is handed the answer before it looks at a single pixel.
(b) `identification/artwork_catalog.py`'s `parse_metadata_from_filename()` (lines 41-62) is the *only* metadata
source `build_from_dir()` (lines 122-130) uses to seed the "local-first" CLIP+Qdrant catalog. `cross_check()`
(lines 146-169) then adopts the catalog's title/artist/date as ground truth whenever cosine similarity ≥ 0.90
(line 156). If the catalog is ever built from the same 12 test images it is later asked to identify, each image
matches itself at ~1.0 similarity and "confirms" its own filename-derived answer — a closed loop, not a
cross-check. Neither path can be trusted as evidence the pipeline identifies art from pixels.
*Fix*: strip filename tokens from every prompt; seed the catalog only from a verified, independent metadata
source (e.g. a curated CSV checked against a museum API), never from the filenames of the images under test;
hold out the 12 test images from catalog construction entirely.

**C3. The "intelligent, genre-aware crop" does not exist — it always falls through to a blind center-crop.**
Three independent breaks, each sufficient alone:
  1. The vision-model prompt schema never asks for a `genre` field (only `style`/`movement`/etc.,
     research_agent.py:344-372); `_ensure_standard_fields()` defaults `genre` to `None`
     (research_agent.py:1317, 1341, 1420) and it is never populated.
  2. Even if it were populated, `PREDEFINED_STYLES` (research_agent.py:33-38, lowercase/hyphenated, e.g.
     `"genre-scene"`) doesn't match the `vision_agent_classes` dict's keys (agents/__init__.py:6-24, title-cased/
     spaced, e.g. `"Genre Scene"`, `"Religious/Historical"` as one combined key). `"Abstract"` has no entry at
     all — commented out at agents/__init__.py:25 — so `VisionAgentAbstract` is unreachable even in principle.
  3. Even if dispatch somehow matched, 8 of the 9 genre `VisionAgent*` classes never implement
     `copy_and_crop_image` (`grep -rn "def copy_and_crop_image" agents/*.py` returns exactly one hit:
     `vision_agent_default.py:71`), so `docent_agent.py`'s `hasattr(vision_agent, 'copy_and_crop_image')` duck-type
     check (docent_agent.py:548) silently skips the crop step for every genre but the default.
  Net effect: regardless of what genre is detected, every image is cropped by
  `vision_agent_default.py`'s pure-geometry center crop, which doesn't even read its own `analysis_results`
  parameter (accepted at vision_agent_default.py:95 signature, never referenced in the body). The "scene
  interpretation → subject-aware crop" step the user described does not run, for any image, ever.
*Fix*: either (a) delete the 9-way genre dispatch and build one cropping function driven directly by the VLM's
subject/bounding-box output, or (b) if genre-specific cropping is wanted, add `genre` to the schema with a
vocabulary that is the single source of truth for both the prompt and the dispatch dict, and implement
`copy_and_crop_image` once (parametrised by a per-genre weighting), not nine times.

**C4. No stage ever enforces the literal 3840x2160 target resolution.**
`UpscaleAgent.upscale_image(..., target_width: int = 3840, target_height: int = 2160)`
(upscale_agent.py:259-265) accepts these parameters but its own docstring admits
"target_width and target_height are for interface consistency, actual scaling depends on API" (lines 262-263).
`_upscale_with_image_upscaling_net()` hardcodes `scale=4` (no target-size math); `_upscale_with_stability_ai()`
only ever downsizes an oversized input to fit Stability's 1,048,576-pixel ceiling
(upscale_agent.py:174-209) and never scales *up* to a fixed target. There is no post-upscale resize/crop-to-exact
step anywhere in `docent_agent.py`. The crop stage (vision_agent_default.py:95-131) also never resizes to
3840x2160 — it only normalizes the aspect ratio at native resolution. So "16:9 3840x2160 target" is aspirational
in the docstrings and enforced nowhere in the code.
*Fix*: add one explicit final step — resize/pad to exactly 3840x2160 after upscaling — and assert on it so a
short-of-target output fails loudly instead of shipping at whatever resolution the 4x multiplier happened to
produce.

**C5. `PlacardAgent.add_plaque()` crashes on every call: `genre` is referenced but never defined.**
`placard_agent.py:139` — `if genre in ['surrealist', 'abstract']:` — `genre` is not a parameter, not assigned
from `metadata` (the method only pulls `title`/`author`/`date`/`nationality`/`style` via `metadata.get(...)`,
lines 144-148; there is no `metadata.get('genre', ...)` anywhere), and not a local anywhere above line 139. This
line sits outside every enclosing `try` block (the nearest one closes at line 133), so it raises an uncaught
`NameError` and aborts the method before any text is drawn, every single time `add_plaque` runs. The placard
stage, as written, cannot currently add text to an artwork at all — only the blank plaque background would ever
get pasted, and even that depends on whether the orchestrator's own try/except around this call (it exists;
see docent_agent.py's placard stage) masks the crash as a non-fatal warning and falls back to the un-placarded
image.
*Fix*: either read `genre = metadata.get('genre', '')` before line 139, or remove the dead genre-based color
branch — either way, this is a one-line crash that nobody has apparently run and caught end-to-end.

### High

**H1. `docent_agent.py` is a ~1,200-line god object with a ~250-line duplicated pipeline inside itself.**
`handle_request()` (docent_agent.py:934-1183) re-implements almost the entire Research→Vision→Upscale→Placard
sequence already present in `start_workflow()` (docent_agent.py:227-742). Two independently-maintained copies of
the same five-stage pipeline is itself a standing bug generator (fix one, forget the other).
*Fix*: extract one `_run_pipeline(image_path)` method; have both entry points call it.

**H2. `start_workflow()` is interactive and therefore unautomatable.**
`input("Your selection: ")` at docent_agent.py:261 blocks on a human at the terminal. There is no headless/
batch/automation entry point into the main workflow — which directly blocks the user's stated goal of running
this from a watched-folder trigger with zero human-in-the-loop.
*Fix*: split the interactive menu out of `start_workflow()`; give the pipeline itself a pure
`run(image_paths: list[str]) -> list[Result]` signature with no `input()` calls, callable from a Pulse task.

**H3. Config loading actively deletes credentials from the environment.**
`docent_agent.py._load_config()` (139-205) deletes `GOOGLE_API_KEY`/`GROK_API_KEY`/`OPENAI_API_KEY` from
`os.environ` if they're absent from `config.yaml` (lines 183, 190, 199-203) — so if `main.py` already set them
from a different source (e.g. shell env, a secrets manager) and `config.yaml` simply omits the key,
`docent_agent.py` actively unsets a working credential rather than leaving it alone. Combined with `main.py`'s own,
separately-written env-promotion logic (`load_and_set_env_from_config`, main.py:107-164), there are two divergent,
redundant copies of "config → env" logic that can silently fight each other.
*Fix*: one config-loading function, in one file, with "absent in config" meaning "leave existing env alone," not
"delete it."

**H4. Three separate, overlapping Google AI SDKs/products coexist in the same codebase.**
`research_agent.py` imports the deprecated `google.generativeai` SDK (lines 21-25, guarded try/except) *and*
`vertexai`/`vertexai.preview.generative_models` (lines 16-18) for the same job (Gemini calls); `agents/__init__.py`
eagerly imports `vision_agent_landscape.py`, which separately imports `google.cloud.vision` (a fourth, distinct
Google product, vision_agent_landscape.py:30) for Cloud Vision label/landmark detection. None of these three
share a client, a credential path, or an error-handling convention.
*Fix*: pick one Google access path (or none, per the user's stated local-model-first direction) and delete the
other two; this alone would remove a large share of the "which key did I forget to set" failure surface.

**H5. The eager import chain in `agents/__init__.py` means the whole `art_agent_team.agents` package requires
`google-cloud-vision` to even import — and this is why most of the test suite cannot run.**
`agents/__init__.py:4` unconditionally does `from .vision_agent_landscape import VisionAgentLandscape` to build
the `vision_agent_classes` registry dict; `vision_agent_landscape.py:30` does `from google.cloud import vision`
at module scope. Verified live: in both the system Python 3.14 and the Project_Aion venv
(`/Users/nathanielcannon/Claude/Project_Aion/infrastructure/.venv/bin/python`, which has most other deps),
`ModuleNotFoundError: No module named 'google.cloud'` is raised the instant anything imports
`art_agent_team.agents` — including the 8 other genre agents that have nothing to do with Google Cloud Vision.
*Fix*: lazy-import each genre agent inside the registry dict's construction (or on first dispatch), so importing
the package doesn't require every provider SDK to be installed.

**H6. `FidelityMetrics` (161 lines) is fully dead code — never imported or called anywhere in the pipeline.**
`grep -rn "FidelityMetrics\|fidelity_metrics" docent_agent.py main.py agents/*.py` returns zero hits. It is also
built on an obsolete `skimage` call: `ssim(orig_img, proc_img, multichannel=True, full=True)`
(fidelity_metrics.py:102) — `multichannel` was removed in favor of `channel_axis` in scikit-image ≥0.19, so even
if someone wired this in today it would likely raise a `TypeError` on a current install. `skimage` itself isn't
even installed in the available Python environments (verified: `ModuleNotFoundError: No module named 'skimage'`).
*Fix*: either wire `validate_fidelity()` into the placard/delivery handoff as the quality gate its docstring
implies it should be, fixing the `channel_axis` API first, or delete the file — right now it is 161 lines of
inert, already-bit-rotted QA logic that gives false confidence it exists.

**H7. `_think_with_grok()` / `_refine_agent_input_with_grok()` are duplicated, dead(-ish), narration-only code
with hardcoded model IDs in three different files.**
`docent_agent.py:803-901` (~100 lines, never called from `start_workflow()` or `handle_request()`),
`placard_agent.py:350-376`, and the same pattern again inline elsewhere all hardcode
`model="grok-3-mini-fast-high-beta"` and `base_url="https://api.x.ai/v1"`, and in every call site the return
value is either discarded or only logged — these are cloud calls that produce no artifact the pipeline actually
uses. `placard_agent.py:72-75` calls one of these unconditionally on every `add_plaque()` invocation, paying
latency/cost for narration nobody reads.
*Fix*: delete the narration-only calls; if "agent reasoning" logging is wanted, generate it locally (it's not
decision-relevant) rather than paying for a cloud round-trip per stage per image.

### Medium

**M1. Nine genre `VisionAgent*` classes are not literal duplicates of each other's analysis, but share large,
measured amounts of boilerplate.** `diff -y --suppress-common-lines` shows `vision_agent_portrait.py` (265
lines) differs from `vision_agent_figurative.py` (273 lines) in only 52 lines, from `vision_agent_genre.py` in
only 52 lines, and from `vision_agent_surrealist.py` in only 55 lines — i.e. these four files are roughly
**80% line-identical**, with only genre-specific prompt text/labels differing. `vision_agent_animal.py` (441
lines) vs `vision_agent_still_life.py` (415 lines) diverge more (318 differing lines) but still share the same
structural skeleton (credential loading, Google Vision API calls, `__main__` self-test blocks).
*Fix*: collapse to one `VisionAgent` class parametrised by a per-genre config dict (labels, prompt fragments,
weighting), not nine files with ~80% shared scaffolding.

**M2. `vision_agent_religious_historical.py` is a 30-line stub, not an implementation.** The entire file
(religious_historical.py:1-30) is a hardcoded placeholder: fixed `"genre_confidence": 0.90, # Example confidence`
(line 20), hardcoded `identified_elements`/`color_palette`/`composition_style` lists (lines 21-23), and no
`analyze_image`/`copy_and_crop_image` at all — it only implements the never-invoked abstract `process()` method.
*Fix*: either implement it for real or remove it from the registry rather than let it silently return canned
fake data if ever reached.

**M3. Two of the genre agents embed `unittest.mock`-based self-tests directly in production modules.**
`vision_agent_animal.py` and `vision_agent_still_life.py` each carry ~100+ lines of mock-based `__main__` test
scaffolding inline (confirmed via grep for `unittest.mock`/`Placeholder`/`TODO` markers: 7 and 6 hits
respectively) rather than living in `tests/`. This inflates "production" line counts and makes the real
implementation harder to find inside each file.
*Fix*: move all self-test blocks into `tests/`.

**M4. Obsolete/preview model IDs hardcoded with an explicit "do not touch" comment blocking any update.**
`research_agent.py:75-80`: `GEMINI_IMAGE_SEARCH_MODEL = "gemini-2.5-pro-exp-03-25"`,
`FLASH_IMAGE_SEARCH_MODEL = "gemini-2.0-flash-exp"`, `GROK_IMAGE_SEARCH_MODEL = "grok-2-vision-1212"`,
`CONSOLIDATION_MODEL = "grok-3-mini-fast-high-beta"` — all four are preview/dated snapshot IDs, several already
past their vendor deprecation window as of this review. Line 65's comment
`# USER REQUIREMENT: DO NOT CHANGE THESE MODEL DEFINITIONS.` froze the design against exactly the kind of model
rot that has since occurred.
*Fix*: in the rebuild, treat this constraint as expired by the user's new "move to local Aion models" direction;
replace all four with the local Ollama/MLX model registry already running in Project_Aion.

**M5. TLS verification explicitly disabled on an image download.** `upscale_agent.py:125`:
`requests.get(..., verify=False)` with an in-line comment "due to known SSL issues" — this disables certificate
validation on a network call that downloads the upscaled artwork back from a third-party service.
*Fix*: fix the actual cert issue (likely a missing intermediate CA bundle) rather than disabling verification;
if the vendor's cert chain is genuinely broken, pin a specific CA bundle rather than disabling TLS entirely.

**M6. Placard sizing and positioning ignore their own config keys.** `config.example.yaml`'s
`placard_agent.width_fraction: 0.25` and `placard_agent.position: "lower-right"` are documented, but
`placard_agent.py` never reads either: `add_plaque()` hardcodes `target_plaque_width_percent = 5`
(placard_agent.py:108, i.e. 5%, not the configured 25%), and `_find_optimal_plaque_position()`
(placard_agent.py:336-348) unconditionally returns the lower-right corner regardless of `position` — its own
docstring admits "Currently a placeholder... TODO: Implement smart positioning" (lines 339, 346), contradicting
`add_plaque()`'s docstring claim of "smart positioning to avoid obscuring key artistic elements" (lines 59-61).
*Fix*: wire `width_fraction`/`position` from config; either implement real low-detail-area placement (e.g. simple
variance/edge-density scan, which the TODO already names) or stop claiming it exists.

**M7. Placard metadata field name drifted out of sync with its own self-test.** `add_plaque()` reads
`metadata.get('author', 'Unknown Artist')` (placard_agent.py:145), but the file's own `__main__` example builds
`metadata = {..., 'artist': 'Jane Doe', ...}` (placard_agent.py:404) — `'artist'` is never read, so the bundled
self-test would always print "Unknown Artist" even though it supplies a name. This is the same `author`/`artist`
key drift that recurs across the pipeline's metadata dicts.
*Fix*: pick one key name (`artist`, to match the rest of the pipeline's vocabulary in research_agent.py) and
grep-enforce it everywhere metadata dicts are built or read.

**M8. Fragile, substring-based JSON/error handling in the LLM response path.** `research_agent.py`'s
`_make_llm_call` strips markdown fences before `json.loads()` (lines 1236-1242) and includes a regex-based
"binary data censoring" heuristic — `binary_pattern = r'(?:\\.{1,10}){10,}'` (lines 1254-1266, repeated at 1300)
— intended to catch base64/binary noise in model output, but a regex this loose can false-positive on
legitimate repeated-escape text in a real description. Retry classification
(`_make_llm_call_with_retry`, lines 961-1014) also decides "retryable vs not" by substring match on the
exception message (line 985), which silently breaks the moment a provider rewords an error string.
*Fix*: validate against a JSON schema instead of regex-sniffing for "looks binary"; classify retryable errors by
HTTP status code / exception type, not message text.

**M9. `UpscaleAgent` documents a `local_esrgan` option it doesn't implement, and carries a half-finished,
commented-out local path.** `config.example.yaml:19` lists `local_esrgan` as a valid `upscaler_preference`
value, but `UpscaleAgent.__init__` (upscale_agent.py:50) only validates
`['image_upscaling_net', 'stability_ai', 'fallback']` — `local_esrgan` is a dangling, unimplemented config value.
A commented-out `_load_esrgan_model` stub (lines 60-72) and a commented-out `__main__` self-test (325-375) show
a local-model path was started and abandoned.
*Fix*: this is exactly the direction the user now wants (local-first upscaling) — finish this path with a real
local Real-ESRGAN (or equivalent Aion-hosted model) rather than re-discovering the same need from scratch.

### Low

**L1. `art_agent_team/#Segmentation Utils.py` is tracked in git** (`git ls-files` confirms) — a `#` and a space
in a filename breaks naive `xargs`/shell globbing over the source tree.
**L2. Nine run artifacts are tracked in git under `art_agent_team/output/` and `art_agent_team/tests/test_data/output/`**
(e.g. `art_agent_team/output/Emile Clause - Summer morning 1891_cropped.jpg` and 8 siblings, plus
`tests/test_data/output/live_test_run/.../*_placarded.jpg`) — generated output, including files named with the
exact title/artist/date of the test artwork (reinforcing C2's filename-leakage concern at the test-fixture
level), should not be committed.
**L3. `art_agent_team/test_log.txt` is tracked and 0 bytes** — a placeholder that should not be in version
control at all.
**L4. `image_upscaling_api_demo.ipynb` is tracked at the repo root** — a notebook has no place being
version-controlled alongside the pipeline source; also not covered by any `.gitignore` notebook-output rule.
**L5. `terminal_output.log` (539,530 bytes) is correctly `.gitignore`'d** (verified via
`git check-ignore -v`) so it is not a git-hygiene problem, but `main.py:75` hardcodes the path as a bare
relative filename (`'terminal_output.log'`, mode `'a'`, i.e. it grows forever) — it should live under a `logs/`
directory with rotation, not accumulate unbounded in the CWD.
**L6. `main.py`'s `TruncateImageDataFilter` (lines 13-68) is a ~55-line regex-based logging filter built to solve
one specific symptom (huge base64 blobs in logs) that would be better solved by not logging raw image payloads
in the first place at the call site.**

## What is worth keeping (carry forward into the rebuild)

- **The 16:9 center-crop math in `vision_agent_default.py:95-131`.** The aspect-ratio branch
  (`current_ratio > target_ratio` → crop width; else → crop height) is correct for both landscape and portrait
  sources and is the one piece of crop logic that actually runs today. Keep the math; replace the "accept
  `analysis_results` and ignore it" behavior with a real subject-aware crop that uses the VLM's bounding boxes to
  bias `left`/`top` instead of a pure center-crop.
- **`identification/artwork_catalog.py`'s CLIP+Qdrant mechanism itself** (`embed()`, `query()`, `cross_check()`,
  lines 105-169) — local, free, and the agree-or-flag reconciliation pattern (lines 146-169: adopt catalog
  metadata above a 0.90 cosine threshold, otherwise flag the VLM hypothesis as low-confidence) is sound *design*.
  Just never seed it (`build_from_dir`/`parse_metadata_from_filename`) from the same images it will later be
  asked to identify.
- **`delivery_agent.py` end to end** — the best-engineered file in the codebase. Correct, narrow OAuth scope
  (`photoslibrary.appendonly`, line 20, with a comment explaining *why* the broader scope now 403s), graceful
  `available()` degradation when OAuth isn't configured (lines 39-41, 119-121) rather than crashing, and correct
  ≤50-item serial batching per the Google Photos API's hard limit (`MAX_BATCH = 50`, line 24; loop at 137-152).
  Keep this file close to as-is.
- **`PromptTemplate`'s field schema** in `research_agent.py:344-372` (minus the filename-token leak at line 371)
  — author/title/date/nationality/style/movement/primary_subjects/secondary_subjects/brief_description/
  confidence_score/grounding_used/limitations is a reasonable, complete identification schema. Add a real `genre`
  field to it with a vocabulary shared verbatim with whatever dispatch table consumes it, and this schema is
  otherwise worth keeping.
- **`UpscaleAgent._upscale_with_stability_ai()`'s oversized-input downsizing math** (upscale_agent.py:174-209,
  `math.sqrt(max_pixels / current_pixels)`) — correct, reusable logic for fitting an image under a fixed
  pixel-count API ceiling; keep the formula when wiring a local upscaler with its own VRAM/pixel ceiling.
- **The config schema in `config.example.yaml`** — the key names and structure (identification engine modes,
  placard knobs, Google Photos album/scope) are sensible; the problem is that `placard_agent.py` and
  `upscale_agent.py` don't actually read several of the keys this schema promises (see M6, M9). Keep the schema,
  make the code honor it.

## Test suite: does not currently run, in any available environment

`cd art_agent_team && python3 -m pytest -q -x --co` (system Python 3.14) fails at the first collected file with
`ModuleNotFoundError: No module named 'google'` (via the `agents/__init__.py` → `vision_agent_landscape.py` eager
import chain, see H5). Re-running collection-only (no `-x`) with
`/Users/nathanielcannon/Claude/Project_Aion/infrastructure/.venv/bin/python` (which has most other project deps
already installed) against all 17 test files plus the 3 top-level `test_*.py` scripts produces **11 collection
errors out of 14 importable test modules**, all for the identical reason: `ModuleNotFoundError: No module named
'google.cloud'`, raised the instant anything imports `art_agent_team.agents`. The remaining 3 files that *do*
import cleanly (`tests/test_error_detection.py`, `tests/test_logging_handler.py`,
`tests/test_workflow_validation.py`) collect **zero actual pytest test functions** — on inspection they contain
only helper classes (`TestLogHandler`, `TestResultType`, `TestResult`), not tests pytest can run.
**Honest result: 0 of the ~17 test files in this repository currently execute a single test, in either Python
environment available on this machine, without first installing `google-cloud-vision` (a cloud SDK the task
explicitly said not to add/call) — and even then, no file was observed to contain runnable assertions that don't
also require a live network call to a vision/LLM provider.** The "17 test files" in the Inventory count exist on
disk; none are currently a working regression safety net. This must be treated as a rebuild requirement, not a
footnote: no stage of the new pipeline should ship without at least one test that runs offline, against a fixture
image, with all three cheat paths (filename tokens, catalog self-seeding, live network calls) blocked.

## Next steps (resume here)

1. **Security first, before anything else touches this repo again:** rotate the leaked xAI key (C1), then decide
   with Sir whether to scrub repo history now or accept the leak as sunk and rotate-only.
2. Redesign as an Alfred Nexus/Pulse task per Sir's order: watched-folder trigger → Pulse kanban task → pipeline
   run — this requires H2 (kill the `input()` blocking call) and H5 (fix the eager-import chain) before a
   headless task runner can even import the package reliably.
3. Rebuild identification so it cannot cheat: remove filename tokens from every prompt (C2a); rebuild the catalog
   from a source independent of the 12 test images, or hold the test images out of catalog construction entirely
   (C2b).
4. Replace the 9-genre dispatch with one parametrised cropping agent driven by real VLM subject output (C3), and
   add an explicit "resize to literal 3840x2160" step with a hard assertion (C4), before claiming any run is
   end-to-end.
5. Move all LLM/vision calls to local Project_Aion models (Ollama VLM already named in
   `identification.vision_model: qwen3-vl:8b`, MLX embeddings, local Real-ESRGAN per M9) — this also resolves H4
   (three competing Google SDKs) and M4 (frozen, rotting model IDs) by removing the cloud providers outright.
6. Only after 1-5: re-run the 12 test artworks and show a real end-to-end pass, with at least one offline,
   network-free regression test per stage checked into `tests/` and actually collected by pytest.
