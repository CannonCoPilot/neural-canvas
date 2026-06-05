<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Google_Gemini-8E75B2?style=for-the-badge&logo=google&logoColor=white" alt="Gemini">
  <img src="https://img.shields.io/badge/Vertex_AI-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white" alt="Vertex AI">
  <img src="https://img.shields.io/badge/Grok_Vision-000000?style=for-the-badge&logo=x&logoColor=white" alt="Grok">
  <img src="https://img.shields.io/badge/OpenRouter-6366F1?style=for-the-badge&logo=openai&logoColor=white" alt="OpenRouter">
  <img src="https://img.shields.io/badge/Pillow-FFD43B?style=for-the-badge&logo=python&logoColor=black" alt="Pillow">
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
</p>

# Neural Canvas

**A multi-agent vision AI pipeline that analyzes, interprets, and intelligently resizes artwork using coordinated AI specialists.**

---

## What Makes This Interesting

Most image processing pipelines treat artwork like any other photo. Neural Canvas treats it like a museum curator would -- understanding what is in the image, why it matters compositionally, and how to modify it without destroying what makes it art.

The system orchestrates **10+ vision and language models** across four AI providers in a concurrent pipeline. A Research Agent fans out parallel calls to Gemini, Grok Vision, Llama 4, Qwen VL, and InternVL, then consolidates their findings through a dedicated reasoning model before any pixel is touched. Genre-specialized Vision Agents (landscape, portrait, religious/historical, surrealist, and five others) apply domain-specific cropping strategies informed by that research. The result: a 16:9 crop that preserves focal points a naive center-crop would destroy.

Three engineering decisions set this apart:

- **Model-as-committee research.** Rather than trusting a single model's art identification, the Research Agent queries 7+ vision models concurrently with `ThreadPoolExecutor`, cross-validates their metadata (artist, title, date, movement, style), and consolidates through Grok's reasoning model with explicit confidence scoring and art-historical date-range validation against a curated movement catalog.

- **Genre-polymorphic vision pipeline.** Nine concrete `VisionAgent` subclasses inherit from a shared abstract base with bounding-box and segmentation-mask primitives. The Docent orchestrator dynamically selects the appropriate subclass based on research output, so a Monet landscape and a Caravaggio religious scene follow entirely different cropping logic.

- **Fidelity-aware quality gates.** A `FidelityMetrics` module computes structural similarity (SSIM), color histogram divergence, and composition scores between original and processed images, with genre-adjusted thresholds to ensure modifications stay within acceptable artistic tolerances.

---

## Architecture

```
                          input/
                            |
                     +------+------+
                     | DocentAgent |  (Orchestrator)
                     +------+------+
                            |
              +-------------+-------------+
              |                           |
     +--------v--------+       +---------v---------+
     |  ResearchAgent   |       |    VisionAgent     |
     |  (Multi-model    |       |  (Genre-specific   |
     |   parallel fan)  |       |   subclass)        |
     +--------+---------+       +---------+----------+
              |                           |
   +----------+----------+               |
   |   |   |   |   |   | |               |
  Gem Grok Llm Qwn Int Mst              |
   |   |   |   |   |   | |               |
   +----------+----------+               |
              |                           |
     +--------v--------+                 |
     |  Consolidation   |                 |
     |  (Grok reasoning)|                 |
     +--------+---------+                 |
              |                           |
              +-------------+-------------+
                            |
                  +---------v----------+
                  |   UpscaleAgent     |
                  | (Stability AI /    |
                  |  image-upscaling)  |
                  +---------+----------+
                            |
                  +---------v----------+
                  |   PlacardAgent     |
                  | (Museum-style      |
                  |  label overlay)    |
                  +---------+----------+
                            |
                        output/
```

---

## Key Technical Features

| Feature | Detail |
|---|---|
| **Concurrent multi-model research** | 7 vision models queried in parallel via `ThreadPoolExecutor` with configurable concurrency limits, exponential backoff retry, and per-call timeouts |
| **Model registry pattern** | `ModelRegistry` class manages client initialization, model registration, capability tagging (vision vs. text), and lazy Vertex AI model instantiation with thread-safe locking |
| **Prompt template system** | `PromptTemplate` class generates model-specific prompts from a master template, with art movement catalog loaded from external data for date-range cross-validation |
| **Genre-polymorphic agents** | 9 Vision Agent subclasses with shared abstract base providing bounding-box visualization, segmentation mask overlay, and standardized output generation |
| **Google Cloud Vision integration** | Landscape and other genre agents use Cloud Vision API for object localization, producing `BoundingBoxRegion` data classes with normalized-to-absolute coordinate conversion |
| **Artistic fidelity scoring** | SSIM-based structural comparison, color histogram analysis, and genre-adjusted pass/fail thresholds via `FidelityMetrics` |
| **Intelligent cropping to 16:9** | Aspect-ratio conversion that uses detected bounding boxes and compositional analysis rather than naive center-crop |
| **Museum placard generation** | Automated museum-style label overlay with card-stock texture background, metadata-driven text layout, and smart positioning to avoid occluding key elements |
| **Multi-provider upscaling** | Dual-backend upscaler (Stability AI + image-upscaling.net) with configurable preference and automatic fallback |
| **Structured consolidation** | Cross-model result merging with JSON schema validation, confidence score normalization, binary data censoring, and standardized field enforcement |

---

## Tech Stack

| Layer | Technologies |
|---|---|
| **Orchestration** | Python, YAML config, argparse CLI |
| **Vision Models** | Gemini 2.5 Pro, Gemini 2.0 Flash, Grok-2 Vision, Llama 4 Maverick/Scout, Qwen 2.5 VL 72B, InternVL3 14B |
| **Reasoning Model** | Grok-3-mini (consolidation, inter-agent thinking steps) |
| **APIs** | Google Generative AI, Vertex AI, OpenAI-compatible (xAI, OpenRouter), Stability AI, Google Cloud Vision |
| **Image Processing** | Pillow (PIL), OpenCV, NumPy, scikit-image (SSIM) |
| **Testing** | pytest, pytest-cov, 17 test modules |

---

## Project Metrics

| Metric | Value |
|---|---|
| Python source files | 47 |
| Test modules | 17 |
| Vision agent specializations | 9 (Landscape, Portrait, Still Life, Animal, Figurative, Genre Scene, Religious/Historical, Surrealist, Default) |
| Vision models orchestrated | 7 concurrent + 1 consolidation |
| API providers integrated | 4 (Google, xAI, OpenRouter, Stability AI) |
| Art movements in validation catalog | 50+ (with date ranges for cross-validation) |
| Planning/design documents | 7 |
| Test artwork corpus | 12 images spanning 1862--2025 |
| Jupyter notebooks | 1 (upscaling API exploration) |

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/your-username/neural-canvas.git
cd neural-canvas

# Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure API keys
cp art_agent_team/config/config.yaml.example art_agent_team/config/config.yaml
# Edit config.yaml with your API keys for Google, xAI/Grok, OpenRouter, and Stability AI

# Place artwork images in the input/ directory, then run
python -m art_agent_team.main --input_folder input/

# Run the test suite
pytest art_agent_team/tests/ -v
```

The pipeline supports selective stage execution -- run research-only for metadata extraction, vision-only for cropping analysis, or the full four-stage workflow.

---

## License

MIT

---

<p align="center"><i>Treating every pixel with the respect the artist intended.</i></p>
