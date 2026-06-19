# neural-canvas — Credential / API Audit (Reconstruction)

**Scope:** External API/service credentials required by the neural-canvas code, audited against the Aion credential vault. **Presence only** — no secret values were read, printed, or stored. Vault entries are referenced by key *path* (name), never value.

**Audit date basis:** static inspection of `art_agent_team/` agents, root scripts (`demo_OpenAI.py`, `grok_api_test.py`, `grok_vision_api.py`, `openrouter_test.py`, `run_*.py`), `requirements.txt`, and `art_agent_team/tests/test_config.yaml`.

---

## 1. Vault location (correction)

The task brief named the vault at `alfred/.claude/secrets/credentials.yaml` — **that path does not exist**. The live vault is at the **monorepo root**:

- `Project_Aion/.claude/secrets/credentials.yaml` — primary (2-document YAML)
- `Project_Aion/.claude/config/credentials.local.yaml` — local override layer (same services, dev keys)

Both were audited. Presence below is satisfied if a key exists in **either** file.

---

## 2. External services referenced by the code

| Service | Code env var(s) / config key | Where used | Purpose |
|---|---|---|---|
| xAI **Grok** | `GROK_API_KEY` / `grok_api_key` | research_agent, placard_agent, docent_agent, main.py | Core LLM/vision: artwork identification, vision analysis, placard text (OpenAI-compatible endpoint) |
| **OpenAI** | `OPENAI_API_KEY` / `openai_api_key` | docent_agent, demo_OpenAI.py | LLM calls / demos |
| **OpenRouter** | `OPENROUTER_API_KEY` / `openrouter_api_key` | research_agent, main.py | LLM aggregator (research fallback) |
| Google **Gemini** (Generative AI) | `GOOGLE_API_KEY` / `google_api_key` | research_agent, docent_agent, main.py | `google-generativeai` Gemini models |
| Google **Custom Search** | `google_search_api_key` + CSE engine id | research_agent (`google-api-python-client`), test_config | Web/reverse search for artwork identification |
| Google **Cloud Vision** | Google creds → `google-cloud-vision` | research/vision agents (fallback) | Label/vision fallback tasks |
| **Stability AI** | `STABLEDIFFUSION_API_KEY` / `stablediffusion_api_key` | upscale_agent | Cloud image upscaling |
| Image-upscaling engine id | `IMAGE_UPSCALING_ID` / `image_upscaling_id` | upscale_agent | Upscale model/engine identifier (paired with Stability) |
| Google **Vertex AI** | `VERTEX_PROJECT_ID`, `VERTEX_LOCATION`, `VERTEX_MODEL_ID`, `GCP_PROJECT_ID`, `GOOGLE_APPLICATION_CREDENTIALS` | research_agent, vision agents (`vertexai`, `google-cloud-aiplatform`) | Alt generative/vision path via GCP project + service account |
| **Real-ESRGAN / ESRGAN** | — (local model) | upscale_agent, docent_agent | Local upscaling — **no credential** |
| **Google Photos** | — (no code reference) | product vision only | Final delivery target (not yet in code) |

`dummy_key` (vision_agent_abstract.py) and `mock_llm_endpoint` (test_config.yaml) are test placeholders — excluded.

---

## 3. Needed-vs-present gap table

| Service / credential | Needed by code | Vault key path(s) (present) | Status |
|---|---|---|---|
| Grok API key | Yes (core) | `llm.grok`, `llm.grok_personal`; `grok.personal_key`, `grok.vscode_key` | ✅ Present |
| OpenAI API key | Yes | `llm.openai`, `llm.jarvis_openai`; `openai.jarvis_key`, `openai.vscode_key` | ✅ Present |
| OpenRouter API key | Yes | `openrouter.primary`, `openrouter.code`, `openrouter.test` | ✅ Present |
| Google Gemini / `GOOGLE_API_KEY` | Yes | `llm.gemini`, `llm.google_api_all`; `gemini_studio.key` | ✅ Present |
| Google Custom Search — API key | Yes | `google_search.api_key`; `google_custom_search.api_key` | ✅ Present |
| Google Custom Search — engine/CSE id | Yes | `google_search.thermoseek_id`; `google_custom_search.engine_id` | ✅ Present |
| Google Cloud Vision key | Yes (fallback) | `image.google_vision`; `google_vision.key` | ✅ Present |
| Stability AI key | Yes | `image.stability`; `stability_ai.key` | ✅ Present |
| Image-upscaling engine id | Yes | `image.upscaling_client_id`; `image_upscaling.client_id` | ✅ Present |
| Vertex — GCP project id / location | Yes (alt path) | — none | ❌ Missing |
| Vertex — service account / `GOOGLE_APPLICATION_CREDENTIALS` | Yes (alt path) | — none (vault holds API keys, not a GCP SA JSON / ADC) | ❌ Missing |
| Real-ESRGAN (local) | Local model | — | ⚪ N/A (no credential) |
| Google Photos OAuth client | Product vision (no code yet) | — none | ⚠️ Future — not in code |

---

## 4. Findings & recommendations

1. **Every API-key-based service the code references is provisioned.** Grok, OpenAI, OpenRouter, Gemini, Google Custom Search (+engine id), Google Vision, Stability AI, and the upscaling engine id are all present in the vault. No API-key gaps for the primary identify → analyse → upscale → placard path.

2. **Vertex AI is the only hard credential gap.** The code offers a `vertexai` / `google-cloud-aiplatform` path requiring a GCP **project id + region + service-account JSON** (Application Default Credentials), none of which are in the vault. Vertex is an *alternative* generative/vision backend; the API-key paths (Grok/Gemini) appear primary, so this likely does **not** block an end-to-end run — but any code branch that selects Vertex will fail at auth. **Action:** either provision a GCP service account + ADC, or confirm Vertex branches are disabled/optional and guard them.

3. **Google Photos delivery is unimplemented.** The product vision ends with "push to Google Photos," but no code references the Photos API and no credential exists. The Google Photos Library API needs an **OAuth 2.0 client** (not a simple key) and, per recent API changes, app-created-media scoping. **Action:** treat as a net-new integration in the roadmap, not a missing key.

4. **Config-key vs env-var dual lookup.** Agents read both lowercase config keys (`grok_api_key`) and uppercase env vars (`GROK_API_KEY`); `main.py` bridges config → `os.environ`. Vault-to-runtime wiring must populate whichever the chosen entry point expects.

5. **Bonus art-domain credentials available (unused by current code):** `research.rijksmuseum`, `image.hive_reverse_image`, `image.clarifai`, `image.ideogram`, `image.fal_ai`. Candidates for stronger artwork identification / reverse-image lookup in a future iteration.

---

*End of `06_credential_audit.md`. No secret values were read or emitted; only key presence was checked.*
