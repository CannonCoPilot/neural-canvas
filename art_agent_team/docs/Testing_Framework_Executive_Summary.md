# Executive Summary: Real-World Testing Framework for Art_AI Project

## Overview
The Art_AI project is transitioning from simulated testing, which relies on mocks, to a comprehensive end-to-end real-world testing framework. This design focuses initially on refactoring `test_research_agent.py` to incorporate actual API calls to Grok and Gemini services, real image processing, and genuine categorization. The framework ensures test reliability, repeatability, and maintainability while addressing key concerns such as secure API key handling, error management, and cost control.

## Key Objectives
- **Real API Calls:** Shift from mocked responses to authentic interactions with external APIs, enabling accurate validation of vision and LLM functionalities.
- **Test Data Management:** Utilize actual test images stored in a dedicated directory, with ground truth derived from image filenames or metadata for consistent validation.
- **Error Handling and Logging:** Implement robust mechanisms to handle API errors, rate limiting, and failures, with detailed logging for debugging and traceability.
- **Test Flexibility:** Support both individual test cases and full workflow simulations to cover various scenarios, including edge cases like corrupted files or unsupported formats.
- **Security and Cost Management:** Ensure API keys are handled securely, and introduce strategies to mitigate rate limiting and associated costs during testing.

## Architectural Highlights
- The framework builds on Python's unittest module, with potential migration to pytest for enhanced features like fixtures and parameterization.
- Integration with existing agents (e.g., ResearchAgent, VisionAgents) will be refactored to use real API clients, with added layers for retry logic and caching to handle external dependencies.
- Mermaid diagrams in subsequent documents will illustrate the testing workflow and component interactions.

## Benefits and Challenges
- **Benefits:** Enhanced test accuracy, reduced risk of production failures, and improved developer productivity through better debugging tools.
- **Challenges:** Dependency on external APIs may introduce flakiness; strategies include using test accounts, implementing delays, and running tests in controlled environments to manage costs and reliability.

This executive summary provides a high-level view, with detailed documents to follow for in-depth implementation plans.
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