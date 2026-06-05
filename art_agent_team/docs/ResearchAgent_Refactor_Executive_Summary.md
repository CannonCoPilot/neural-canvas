# Executive Summary: ResearchAgent Refactoring

## Overview
This document summarizes the architectural design for refactoring the ResearchAgent in the Art_AI project. The goal is to create a unified workflow for text and image processing, eliminating separate text-only functions and integrating new vision-capable LLM models. This refactoring enhances efficiency, consistency, and scalability by consolidating LLM interactions into a single entry point.

## Key Changes
- **Unified Workflow**: A single method handles both text and image inputs, integrating searches and processing in parallel to reduce redundancy.
- **Model Integration**: New models (e.g., OPENROUTER_INTERNVL, OPENROUTER_MISTRAL3) are added to a dynamic registry, allowing parallel calls for multi-model processing.
- **Data Flow**: Input handling includes filename tokens and image data, with consolidation ensuring standardized JSON output.
- **Structures**: A master prompt template with model-specific customizations is defined, along with enhanced error handling and fallback strategies.

## Benefits
- Improved performance through parallel processing and reduced API calls.
- Enhanced robustness with better error handling and dynamic model selection.
- Simplified codebase, making maintenance and extensions easier.

## Next Steps
Review this design with stakeholders to incorporate feedback before finalizing the implementation plan.
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