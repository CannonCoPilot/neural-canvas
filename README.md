# Neural Canvas — an AI art docent for your TV

Drop an image of an artwork into a folder — no title, no artist, nothing but pixels. Neural Canvas
identifies the work, reads the scene, composes it for a 4K (3840×2160) screen without cutting through
its subjects, writes a museum wall label, and delivers the finished frame to the Google Photos album
your TV's ambient mode shows. Every step runs on local models; each run is a task on the Aion Pulse board.

```
watch folder ─▶ Pulse task ─▶ identify ─▶ interpret ─▶ compose ─▶ upscale ─▶ label ─▶ deliver
 (var/inbox)   (kanban card)   who/what    subjects,     16:9 crop   Real-     museum   Google Photos
                               it is       intent        or mat      ESRGAN    card     "Neural Canvas"
```

## How each stage works

| Stage | Method (all local unless noted) | Gate — the run blocks if it fails |
|---|---|---|
| **identify** | Qwen3-VL 32B proposes candidates and reads signatures. Reference photos come from Wikimedia Commons (title search), then the artist's whole **Wikidata oeuvre** (every catalogued painting, embeddings cached), then web image search for in-copyright works. | A reference must match on **two independent instruments**: CLIP image similarity ≥ 0.86 **and** ≥ 25 ORB/RANSAC geometric inliers. Facts come from the matched record (Wikidata), never from the model's guess. Web-caption facts need the artist corroborated by the model's own reading. |
| **interpret** | Qwen3-VL 32B: genre, subjects with boxes, focal point, intent, 40–65-word label paragraph constrained to verified facts. | At least one essential subject; label paragraph present. |
| **compose** | One subject-preserving algorithm for every genre (genre only tunes weights): best 16:9 window by subject importance, spectral-residual saliency and thirds; if no 16:9 window keeps every essential subject, trim only as far as they allow and mat the rest (museum wall). | Every essential subject ≥ 97 % retained. |
| **upscale** | Real-ESRGAN x4plus on Apple MPS (tiled), then Lanczos to the exact display size. | Exact size; ≤ 4× needed; colour drift ΔE ≤ 4 vs a plain resize. |
| **label** | Georgia-set museum card: artist, nationality and life dates, title and date, medium, collection, paragraph. Placed on the mat, else the least-salient corner. | No overlap with essential subjects; contrast ≥ 7:1; text fits. |
| **deliver** | Google Photos Library API, app-created album "Neural Canvas". | Media item created. |

A failed gate never degrades into a "success": the Pulse card goes to **Blocked** with the reason, stays
open, and an alert goes to the Nexus message bus. The original file name and descriptive EXIF are never
seen by any stage — ingest copies the image to a content-hashed path.

## Running it

```bash
uv venv .venv && VIRTUAL_ENV=.venv uv pip install -e '.[test]'
ollama pull qwen3-vl:32b-instruct
curl -L -o RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth   # sha256 4fa0d389…
```

| Command | What it does |
|---|---|
| `neural-canvas serve` | the service: watch folder → Pulse tasks → runs (installed as launchd `com.aion.neural-canvas`, see `deploy/`) |
| `neural-canvas submit IMG…` | drop images into the watch folder |
| `neural-canvas status` | the neural-canvas cards on the Pulse board |
| `neural-canvas retry [--stage deliver]` | re-queue blocked cards (finished stages are reused) |
| `neural-canvas run IMG… [--no-deliver]` | run directly without Pulse (development) |
| `neural-canvas auth-photos` | one-time Google Photos consent |
| `neural-canvas eval tests/fixtures/gold.yaml` | score finished runs against the 12-artwork gold set |

Configuration: copy keys from `neural_canvas/config.py` `DEFAULTS` into a `config.yaml` at the repo root
(gitignored) to override paths, models or thresholds.

### Google Photos (one-time)
1. Google Cloud Console → enable **Photos Library API** → Credentials → *Create OAuth client ID* → *Desktop app*.
2. Save the JSON as `~/Claude/Project_Aion/.claude/secrets/neural-canvas-gphotos-client.json`.
3. `neural-canvas auth-photos`, then `neural-canvas retry --stage deliver`.
4. On the TV: Ambient mode → Google Photos → select the **Neural Canvas** album.

## Layout
`neural_canvas/` the package · `tests/` unit tests + gold set · `test_images/` 12 test artworks ·
`deploy/` launchd plist · `docs/` code review, reconstruction notes, original planning docs (`docs/history/`) ·
`var/` runtime (inbox, work, output, state, logs — gitignored).
