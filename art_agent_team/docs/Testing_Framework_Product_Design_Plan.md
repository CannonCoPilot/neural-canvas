# Product Design Plan: Real-World Testing Framework for Art_AI

## Overview
This document outlines the design of a producible real-world testing framework for the Art_AI project, specifically targeting the transition of `test_research_agent.py` from mocked to end-to-end testing. The framework will be implemented in Python, leveraging libraries like unittest or pytest, and will integrate with existing agents to ensure seamless operation with actual API calls. The design emphasizes modularity, scalability, and adherence to best practices for testing AI-driven applications, with no use of mocks or simulations.

## Design Components
### 1. Testing Architecture
- **Framework Structure:** 
  - Use a layered architecture: 
    - **Test Harness Layer:** Manages test execution, API key injection, and environment setup using real API interactions.
    - **Agent Interaction Layer:** Interfaces with ResearchAgent and VisionAgents to perform genuine API calls and image processing.
    - **Validation Layer:** Compares API responses against ground truth data for accuracy, ensuring all validations use real data sources.
  - Mermaid Diagram:
    ```
    graph TD
        A[Test Harness] --> B[Agent Interaction]
        B --> C[Real API Calls (Grok, Gemini)]
        B --> D[Real Image Processing]
        C --> E[Validation Layer]
        D --> E
        E --> F[Ground Truth Data]
        F --> G[Assertion and Logging]
    ```
- **Tooling:** 
  - Primary testing framework: pytest for its rich feature set, including fixtures, parameterization, and plugins for API testing.
  - Error Handling: Use Python's logging module enhanced with structured logging (e.g., via loguru) to capture detailed error traces and API responses.

### 2. Test Data Management Strategy
- **Image Assets:** Store test images in a dedicated directory (e.g., `test_images/`) with metadata files (JSON or YAML) containing ground truth categorizations. This allows for easy updates and expansion using the provided art assets.
- **Ground Truth Derivation:** Transition from filename-based ground truth to a more robust system, such as a JSON file mapping image paths to expected results (e.g., {"image_path": "path", "category": "landscape", "artist": "Expected Artist"}).
- **API Response Handling:** Implement strategies to handle real API responses, such as storing results in a database for reference during multiple test runs, to ensure repeatability without artificial caching that could simulate responses.

### 3. Error Handling and Logging Approach
- **Error Handling:** 
  - Wrap API calls in try-except blocks with retry logic (e.g., using tenacity library for exponential backoff on rate limit errors) to handle real-world failures authentically.
  - Define custom exceptions for API-specific errors (e.g., RateLimitError, APIKeyInvalidError) to improve debuggability and ensure no fallback to mocks.
- **Logging:** 
  - Log all API requests and responses at DEBUG level, errors at ERROR level, with timestamps and agent IDs for traceability.
  - Integrate with the existing logging setup in ResearchAgent to maintain consistency and provide real-time insights during tests.
- **Mermaid Diagram for Error Flow:**
    ```
    graph TD
        A[API Call] -->|Success| B[Validate Response]
        A -->|Failure| C[Handle Error]
        C --> D[Retry Logic]
        C --> E[Log Error]
        D -->|Max Retries Exceeded| E
        B --> F[Assert Against Ground Truth]
        F --> G[Pass/Fail]
    ```

### 4. Guidelines for Writing Real-World Tests
- **Test Types:** 
  - Unit Tests: Focus on individual methods with real dependencies where possible, ensuring that even unit tests use actual API calls if they interact with external services.
  - Integration Tests: Execute full workflows with real API calls, parameterized for different images and scenarios to cover all real-world cases.
  - Edge Case Tests: Cover rate limiting, invalid keys, corrupted images, and unsupported formats using genuine conditions, with no simulations.
- **Security Guidelines:** 
  - API keys should be loaded from environment variables in all tests to avoid exposure. Use libraries like python-dotenv for management.
  - In CI/CD environments, use secure vaults (e.g., GitHub Secrets) to inject keys, ensuring all tests run with real credentials.
- **Cost and Rate Limiting Management:** 
  - Limit concurrent tests and add delays between API calls to handle rate limiting in real scenarios.
  - Monitor API usage with counters and abort tests if quotas are approached, using actual API responses to inform decisions.
- **Maintainability:** 
  - Use pytest fixtures for setup/teardown of API clients and test data with real integrations.
  - Parameterize tests with @pytest.mark.parametrize to handle multiple image inputs efficiently.
  - Structure tests in modules by agent or functionality for scalability, with clear documentation on real-world dependencies.

## Potential Issues and Mitigations
- **API Flakiness:** Mitigate with retry mechanisms and ensure all tests are designed to handle real-world variability, without any fallback options.
- **Cost Overruns:** Implement a budget tracker and use strategies like scheduling tests during off-peak hours or limiting test runs, always with real API interactions.
- **Data Privacy:** Use the provided art assets directly, as per user permission, ensuring no additional safeguards are needed beyond standard practices.
- **Dependency Management:** Use virtual environments and requirements.txt to handle library versions for API clients, ensuring all dependencies are real and tested.

This design plan has been revised to strictly adhere to real-world implementations, removing all references to mocks or simulations as per user feedback.
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