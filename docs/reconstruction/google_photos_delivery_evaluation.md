# neural-canvas — Google Photos Delivery: Evaluation & Future-Task Proposal

**Task:** A3.2.2 · **Scope:** evaluation + documentation only (no code changes, no implementation).
Reconstructed from the codebase on 2026-06-17.

## 1. Evaluation result — **NOT IMPLEMENTED**

The Google Photos delivery stage does **not exist** in the codebase. The implemented pipeline
terminates at the **placard** stage, writing the final composite (`<name>_final.<ext>`) to the
local `output_folder`. There is no upload-to-Google-Photos, album, or external-delivery code.

### Evidence

| Check | Result |
|---|---|
| Delivery/upload code in `art_agent_team/` | **None.** No `photoslibrary`, `mediaItems`, album, or delivery calls. The only network "upload" is to **image-upscaling.net** inside `UpscaleAgent` (a processing call, not delivery). |
| `google.auth` / `google.oauth2` usage | Present only in **vision agents** (`vision_agent_animal/landscape/still_life.py`) for **Vertex AI / Cloud Vision** credentials — unrelated to Photos. |
| `requirements.txt` | Has generic Google libs (`google-generativeai`, `google-api-python-client`, `google-cloud-vision`, `google-auth*`, `google-cloud-aiplatform`). **No Google Photos package** and **no `google-auth-oauthlib`** (needed for the OAuth2 user-consent flow Photos requires). |
| `config.yaml` | **No** delivery/Photos/album/OAuth keys. |
| Orchestrator (`DocentAgent`) | `start_workflow` exposes only stages 1–4 (Research, Vision, Upscale, Placard); `handle_request` runs the same four. No delivery stage. |
| Planning corpus | Terminal stage is **"Save & Display Output"** (`Art_Agent_Team_Implementation_Plan.md:130`) targeting 16:9 / 4K **digital displays** — local save, not Google Photos. |

### Provenance note
"Google Photos delivery" appears in the **master/parent specification** and in the reconstructed
**product vision** (`01_product_requirements.md`, A2) as the mechanism for getting finished
artwork onto a display (the digital art-frame / TV use case). It is **not** a designed stage in
the original planning documents, which stop at local "Save & Display Output." It is therefore an
**aspirational delivery target**, not a descoped-but-specified one.

---

## 2. Future-task proposal — `DeliveryAgent` (Google Photos backend)

### 2.1 Expected interface
Add a new agent under `art_agent_team/agents/`, consistent with existing agent conventions
(`__init__(self, config)`, single public entry, returns result-or-`None`):

```python
class DeliveryAgent:               # or GooglePhotosAgent
    def __init__(self, config: Optional[Dict] = None): ...
    def deliver(self, image_path: str, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Upload the final placarded image to Google Photos (and optionally add to an album).
        Returns {"media_item_id", "album_id", "product_url"} on success, None on failure."""
```

### 2.2 Orchestration integration (Stage 5)
- Register as `self.agents['delivery']` in `DocentAgent.__init__`.
- Add a **Stage 5 block** after Placard in both `start_workflow()` (new menu option `6`/extend
  "Full Workflow") and `handle_request()`.
- **Trigger condition:** only when the placard stage produced a valid final image
  (`current_image_path` exists and points at `output_folder/<name>_final`).
- **Follow the existing graceful-degradation pattern:** `try/except`, missing-input guard,
  log via `_log_to_markdown` / `_format_log_entry`, and on failure continue without aborting the
  run (delivery failure must not lose the locally-saved final image).
- **Input:** final placarded image + `metadata` (for caption/description and album routing).

### 2.3 Authentication & dependencies
- **API:** Google Photos Library API, accessed via `google-api-python-client` (already present)
  using the discovery service `photoslibrary` (v1).
- **Auth:** Google Photos requires **OAuth2 user credentials** (a service account cannot post to
  a user's library). Add **`google-auth-oauthlib`** (currently missing) for the consent flow;
  reuse `google-auth` for token refresh. Persist a refresh token.
- **Scopes:** use the **app-created** scopes —
  `https://www.googleapis.com/auth/photoslibrary.appendonly` for upload, and
  `…/photoslibrary.edit.appcreateddata` for managing app-created albums.

> **⚠ Critical dependency/risk (from comparative research, `05_comparative_research.md`, A6).**
> The Google Photos Library API had a **breaking change (effective ~March 31, 2025)**: third-party
> apps may only **read/manage media they themselves created**, and several broad read scopes were
> removed. For neural-canvas this is **acceptable** — it uploads its *own* images — but the design
> **must not** rely on listing or modifying the user's pre-existing library/albums. Album
> creation/append is limited to app-created albums.

### 2.4 Configuration keys to add (`config.yaml`)
`google_photos_enabled` (bool), `google_photos_album_name` / `google_photos_album_id`,
`google_oauth_client_secret_path`, `google_oauth_token_path`.

### 2.5 Assumptions
- One-time interactive OAuth consent is acceptable (headless runs reuse a stored refresh token).
- Album is app-created (created on first run if absent).
- Network availability; Photos API quota/rate limits and per-image size limits are respected.
- Upload is "best-effort": the canonical artifact remains the local `_final` file.

### 2.6 Alternatives (per A6)
Abstract `DeliveryAgent` behind a backend interface so delivery targets are pluggable —
Google Photos (shared album the frame subscribes to), **Samsung Frame / Meural** native channels,
or a **local/network folder sync** the display reads. This decouples "produce final image" from
"get it onto the display device," which is the actual product goal.

---

## 3. Summary
Google Photos delivery is **unimplemented**; the pipeline ends at a locally-saved placarded image.
Implementing it is a self-contained future task: a `DeliveryAgent` wired as an optional, failure-
tolerant Stage 5, using the Google Photos Library API under OAuth2 app-created scopes, with
`google-auth-oauthlib` added as the one missing dependency. The March-2025 API restriction shapes
the design (app-created media only) but does not block the core upload use case.

---

*End of `google_photos_delivery_evaluation.md`.*
