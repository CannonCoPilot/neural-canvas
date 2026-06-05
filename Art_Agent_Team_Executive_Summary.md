# AI Art Team Project: Executive Summary

## Project Overview
The AI Art Team project aims to develop an interoperable system of AI agents to process artwork images, ensuring the preservation of artistic integrity while applying modifications such as aspect ratio adjustments, upscaling to 4K resolution, and adding informative plaques. The system is designed to handle diverse genres of artwork, applying genre-specific processing to maintain or enhance the aesthetic and emotional impact of each piece.

## Objectives
- **Aspect Ratio Adjustment**: Adjust images to a 16:9 aspect ratio by intelligently cropping to prioritize key focal points and compositional elements, minimizing loss of critical content.
- **Upscaling to 4K Resolution**: Enhance image resolution to 4K using advanced techniques like ESRGAN to preserve fine details such as brush strokes, paint layering, and medium texture, while reducing artifacts.
- **Plaque Integration**: Overlay aesthetically integrated plaques with key information (title, artist, nationality, date) using a textured white cardstock aesthetic, black sans-serif lettering, positioned in the lower right corner for minimal visual disruption.
- **Artistic Integrity**: Ensure all modifications respect the composition, color balance, historical context, and emotional impact of the original artwork through genre-specific processing and intelligent decision-making.

## Key Components
- **DocentAgent**: Orchestrates the workflow, managing image input, agent coordination, and output generation.
- **ResearchAgent**: Analyzes artwork to extract metadata and contextual information, guiding subsequent processing steps.
- **VisionAgent Classes**: Genre-specific agents (e.g., Landscape, Portrait) that apply tailored image analysis and cropping strategies to preserve artistic elements.
- **UpscaleAgent (Planned)**: A dedicated agent for upscaling images to 4K resolution, to be integrated into the existing workflow.
- **PlacardAgent (Planned)**: A new agent for designing and overlaying plaques with specified aesthetic preferences.

## Project Status
The current system implements aspect ratio adjustments through intelligent cropping and supports genre-specific processing via specialized VisionAgents. Upscaling to 4K and plaque integration are identified as areas for enhancement, with placeholders in the codebase awaiting full implementation.

## Strategic Goals
- Develop a functional beta version within the next 3 months, incorporating upscaling and plaque features.
- Conduct iterative testing to ensure artistic fidelity across diverse artwork genres.
- Establish a scalable architecture that can accommodate additional agents or processing modules as needed.

## Conclusion
The AI Art Team project represents a pioneering effort to blend AI technology with artistic preservation, delivering high-quality processed images that respect the original intent of the artwork. This executive summary outlines the vision and key priorities to guide the team towards a successful beta release.
## LLM Constraints

### ResearchAgent
- Vision Models: 
 - Vision Models to be called through the OpenRouter API using OpenRouter API Key
OPENROUTER_LLAMA_MAVERICK = "meta-llama/llama-4-maverick:free"
OPENROUTER_LLAMA_SCOUT = "meta-llama/llama-4-scout:free"
OPENROUTER_QWEN_VL = "qwen/qwen2.5-vl-72b-instruct:free"
OPENROUTER_INTERNVL = "opengvlab/internvl3-14b:free"
OPENROUTER_MISTRAL3 = "mistralai/mistral-small-3.1-24b-instruct:free"
OPENROUTER_GEMMA3 = "google/gemma-3-27b-it:free"
 - Vision Models to be called through the Google Gemini API using Gemini API Key, or Google Vertex AI API using Vertex AI project credentials
GEMINI_IMAGE_SEARCH_MODEL = "gemini-2.5-pro-exp-03-25"
FLASH_IMAGE_SEARCH_MODEL = "gemini-2.0-flash-exp"
 - Vision Models to be called through the Grok OpenAI-compatible API using Grok API Key
GROK_IMAGE_SEARCH_MODEL = "grok-2-vision-1212"
 - Consolidation Model to be called through the Grok OpenAI-compatible API using Grok API Key
CONSOLIDATION_MODEL = "grok-3-mini-fast-high-beta"
- Consolidation/Communication Model: `grok-3-mini-fast-high-beta`

### VisionAgents
- Vision Analysis Model: `gemini-2.5-pro-exp-03-25`
- Communication/Reasoning Model: `grok-3-mini-fast-high-beta`

### PlacardAgent
- Communication/Reasoning Model: `grok-3-mini-fast-high-beta`

### DocentAgent
- Communication/Reasoning Model: `grok-3-mini-fast-high-beta`

### General
- 'Thinking' functionality uses `grok-3-mini-fast-high-beta` for internal reasoning/validation across relevant agents.