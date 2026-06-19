# neural-canvas — Implementation Status & Roadmap (Reconstruction)

*Reconstructed by code inspection (A4) of `art_agent_team/` and cross-referenced against
`Art_Agent_Team_Assessment_and_Next_Steps.md` (dated 4/30/2025). Where code and the
assessment disagree, **the code is authoritative** — the assessment is stale and the
codebase has advanced past it.*

---

## 1. Method & evidence basis

- Static inspection of every module under `art_agent_team/` (line counts, class/method
  surface, stub/TODO markers, real-vs-mock API calls).
- Stub markers via `grep` for `TODO/NotImplementedError/placeholder/mock/pass`.
- Orchestration wiring read from `docent_agent.py` (which stages it actually invokes).
- Cross-reference: `Art_Agent_Team_Assessment_and_Next_Steps.md`.
- Evidence is cited as `file:line` or line counts. "Mock" usage in `__main__` self-test
  blocks is **not** counted as a production stub.

---

## 2. Status table (component → state → evidence)

| Component | File (lines) | State | Evidence |
|---|---|---|---|
| **DocentAgent** (orchestrator) | `docent_agent.py` (1130) | ✅ Implemented | Wires Research→Vision→Upscale→Placard; threaded/queued batch loop; instantiates `UpscaleAgent`/`PlacardAgent` at `:63-68`, invokes stages at `:273/470/569`. |
| **ResearchAgent** (identification/metadata) | `research_agent.py` (1400) | ✅ Implemented | Largest module; real vision/LLM API usage; instantiated per-image at `docent_agent.py:278`. |
| **VisionAgent base** | `vision_agent_abstract.py` (505) | ✅ Implemented | Shared crop/analysis machinery for genre subclasses. |
| Genre VisionAgents (animal, landscape, still_life, portrait, genre, figurative, surrealist) | 251–441 each | ✅ Implemented | Real Google Vision client + service-account creds in animal/landscape/still_life; others substantial (no stub markers). |
| **VisionAgentDefault** (fallback) | `vision_agent_default.py` (143) | 🟡 Partial | Fallback path used when no genre match (`docent_agent.py:473`). |
| **VisionAgentReligiousHistorical** | `vision_agent_religious_historical.py` (30) | ❌ **Stub** | Returns **hardcoded** description + example confidence/palette; no real API call. Explicit "Placeholder implementation". |
| **UpscaleAgent** | `upscale_agent.py` (374) | 🟡 Partial (impl, API-only) | Real cloud upscaling: `_upscale_with_image_upscaling_net` + `_upscale_with_stability_ai`. Local ESRGAN path **abandoned** (commented out `:60-72`). No local fallback; skips if no API key (`:57`). |
| **PlacardAgent** | `placard_agent.py` (416) | 🟡 Partial | Overlay implemented; **smart positioning is a TODO** (`:346`), defaults to lower-right (`:339`). |
| **FidelityMetrics** | `fidelity_metrics.py` (161) | 🟡 Partial | Output QA module exists (the assessment asked for this; now present). |
| **Config/env loader** | `main.py` (180) | ✅ Implemented | Loads YAML config, validates keys, detects placeholder values (`:137-159`). |
| **Entry points** | `run_docent_mvp.py`, `run_live_test.py` | ✅ Present | MVP runner + live test harness. |
| **Google Photos / delivery** | — | ❌ **Absent** | No upload/album/delivery module. The only `google.*` imports are `oauth2.service_account` for Google **Vision** creds, not Photos. Pipeline ends at local placard output. |

Legend: ✅ implemented · 🟡 partial · ❌ stub/absent.

---

## 3. Reconciliation with `Art_Agent_Team_Assessment_and_Next_Steps.md`

The assessment (4/30/2025) is **stale**; the code has moved on. Divergences:

| Assessment claim (Apr 2025) | Current code reality |
|---|---|
| "Upscaling… remain unimplemented, placeholder for `UpscaleAgent`; ESRGAN model missing → mock upscaling" | **Re-architected**: local ESRGAN dropped; real **cloud-API** upscaling (image-upscaling.net + Stability AI) now implemented. Still key-dependent. |
| "Unit tests confirm `PlacardAgent` functionality is not yet implemented" | **Now implemented** (basic overlay, 416 lines); only smart-positioning remains a TODO. |
| "No explicit mechanism for testing artistic fidelity" | `fidelity_metrics.py` (161 lines) now exists. |
| Google Photos delivery not flagged as a gap | **Still absent** — and it is the product's defined final step, so it is now the largest gap. |

The assessment's stage ordering (testing → UpscaleAgent → PlacardAgent → fidelity → UI →
scraping) has been partially overtaken: stages 3–5 are substantially done; the unaddressed
items are integration testing, the delivery stage, and the religious/historical stub.

---

## 4. Critical end-to-end gaps (what blocks a working product)

1. **No delivery stage (Google Photos).** *Critical.* The stated goal — push the finished
   image to Google Photos — has no code. The pipeline terminates at a local placarded file.
2. **Upscaling is entirely API-dependent.** *High.* No offline/local fallback; if keys are
   absent or the third-party API is down, upscaling silently skips and the "4K" promise fails.
3. **One genre agent is a stub.** *Medium.* `VisionAgentReligiousHistorical` fabricates a
   templated description — any religious/historical artwork gets fake analysis, not real vision.
4. **Smart placard positioning unimplemented.** *Medium.* Always lower-right; can occlude
   salient content on some compositions.
5. **No integration tests for agent handoff.** *Medium.* Threading/queue handoff between
   stages is untested end-to-end (assessment stage 1, still open).

### Secondary risks / smells
- `DocentAgent` docstring still claims "upscaling with ESRGAN" — documentation drift vs the
  actual cloud-API implementation.
- Dependence on multiple external vision/LLM APIs (Google Vision, Gemini, Grok, OpenRouter,
  Stability) with no unified fallback policy.

---

## 5. Implied roadmap (re-prioritized for the *current* state)

**Phase 1 — Close the pipeline (make it end-to-end).**
- Build the **delivery agent** (Google Photos API: OAuth, album selection, upload).
- Wire it into `DocentAgent` as the terminal stage after Placard.

**Phase 2 — Harden the existing stages.**
- Add a local/offline upscaling fallback (Real-ESRGAN) so "4K" doesn't depend on a 3rd-party API.
- Implement `VisionAgentReligiousHistorical` for real (replace the templated stub).
- Implement smart placard positioning (saliency/low-variance region detection).

**Phase 3 — Verify & qualify.**
- Integration tests for Research→Vision→Upscale→Placard→Delivery handoff.
- Exercise `fidelity_metrics` as a gating check (color/detail/composition thresholds).
- Reconcile docstrings/docs with the cloud-API reality.

**Phase 4 — UX & sourcing (assessment stages 6–7).**
- CLI/GUI for input selection and parameter overrides.
- Optional ArtVee scraping for batch input.

---

## 6. Remaining work items (checklist to a functioning end-to-end product)

- [ ] **Delivery: Google Photos upload module + DocentAgent wiring** (blocks end-to-end).
- [ ] UpscaleAgent: local fallback model; verify real 4K output, not silent skip.
- [ ] VisionAgentReligiousHistorical: real implementation (currently a stub).
- [ ] PlacardAgent: smart positioning (`placard_agent.py:346` TODO).
- [ ] Integration tests for multi-stage threaded handoff.
- [ ] FidelityMetrics: promote from present-but-passive to an actual quality gate.
- [ ] Doc/comment cleanup: remove stale "ESRGAN" claims in `docent_agent.py`.
- [ ] Config: document required API keys (Google Vision, Gemini, Grok/OpenRouter, Stability, Google Photos).

---

## 7. One-line bottom line

The middle of the pipeline (identify → analyze → crop → upscale → placard) is **substantially
implemented and more advanced than the planning docs claim**, but the product is **not
end-to-end**: it has **no delivery stage**, upscaling is **API-only with no fallback**, and one
genre analyzer is a **stub** — so the headline "artwork in → finished image in Google Photos"
workflow cannot complete today.

---

*End of `03_status_and_roadmap.md`.*
