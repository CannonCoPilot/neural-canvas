# Detailed Plan for Accomplishing the Task: Real-World Testing Framework for Art_AI

## Overview
This document provides a step-by-step plan for implementing the real-world testing framework for the Art_AI project, focusing on refactoring `test_research_agent.py`. The plan outlines tasks, responsibilities, timelines, and technical steps to be executed by the team, ensuring a smooth transition to end-to-end testing with actual API calls and no use of mocks or simulations. The implementation will be carried out by the Coder mode, with coordination from the Orchestrator, adhering strictly to genuine integrations.

## Plan Structure
- **Phases:** Broken into preparation, implementation, testing, and deployment phases, all based on real-world interactions.
- **Mermaid Diagram:** Illustrates the workflow and dependencies.
- **Roles:** Architect (planning), Coder (implementation), Orchestrator (coordination).
- **Timeline:** Estimated in days, assuming dedicated resources.

## Mermaid Diagram of Implementation Workflow
```
graph TD
    A[Preparation Phase] --> B[Implementation Phase]
    B --> C[Testing Phase]
    C --> D[Deployment Phase]
    A -->|Gather Requirements| E[Refactor test_research_agent.py]
    B -->|Code Changes| F[Integrate Real API Calls]
    B -->|Add Error Handling| G[Implement Logging]
    C -->|Run Real-World Tests| H[Validate with Actual Data]
    D -->|Merge to Main| I[CI/CD Integration]
```

## Step-by-Step Plan
### 1. Preparation Phase (1-2 days)
- **Objective:** Set up the environment and gather all necessary resources with real components.
- **Tasks:**
  - Review existing code and documents (e.g., `research_agent.py`, `test_research_agent.py`, config files) to ensure all setups use actual API keys and data.
  - Create or update ground truth data in a JSON file (e.g., `test_ground_truth.json`) based on image filenames and user input, using provided art assets directly.
  - Ensure secure API key handling by loading from environment variables in all contexts, with no dummy values.
- **Responsibilities:** Architect to finalize design; Coder to prepare test data with real integrations.
- **Deliverables:** Updated test data files and environment setup scripts that use genuine configurations.

### 2. Implementation Phase (3-5 days)
- **Objective:** Refactor code to use real API calls and implement framework components without any fallbacks.
- **Tasks:**
  - Refactor `test_research_agent.py` to remove all mocks and integrate authentic API clients, with retry logic and real response handling.
  - Implement test fixtures for API key injection and image loading in pytest, ensuring all calls are to live APIs.
  - Add parameterization for testing multiple images and scenarios with genuine data flows.
  - Integrate logging and error handling as per the product design plan, focusing on real-world failure modes.
- **Responsibilities:** Coder to handle code changes; Architect to review for architectural fit and ensure no simulated elements.
- **Potential Challenges:** API rate limiting—mitigate with delays and monitoring, but always using real calls; no artificial workarounds.
- **Deliverables:** Refactored test file and new utility modules for authentic error handling and logging.

### 3. Testing Phase (2-3 days)
- **Objective:** Validate the new framework with real-world data and actual API interactions.
- **Tasks:**
  - Run individual tests for standard, hybrid, and edge cases using live API calls and real images.
  - Perform full workflow tests to ensure end-to-end functionality with genuine responses.
  - Validate against ground truth and log discrepancies for debugging, using real data sources only.
  - Address any errors iteratively with real scenarios, ensuring no use of mocks for any part of the process.
- **Responsibilities:** Coder to execute tests; Debug mode (if needed) for issue resolution with actual system interactions.
- **Deliverables:** Test reports and coverage metrics based on real executions.

### 4. Deployment Phase (1 day)
- **Objective:** Integrate the framework into the CI/CD pipeline and document usage with real-world considerations.
- **Tasks:**
  - Merge changes into the main branch after thorough real-world testing.
  - Set up CI/CD to run tests with live APIs in a controlled environment (e.g., using GitHub Actions with secrets for API keys), ensuring all tests use actual services.
  - Update project documentation with guidelines for running and maintaining the tests, emphasizing real integrations.
- **Responsibilities:** Orchestrator to coordinate deployment; Coder to implement CI/CD scripts with genuine dependencies.
- **Deliverables:** Updated README and CI/CD configuration files that reflect authentic testing practices.

## Timeline and Milestones
- **Day 1-2:** Preparation complete with real data setups.
- **Day 3-7:** Implementation and initial real-world testing.
- **Day 8-10:** Full testing with live APIs and refinements.
- **Day 11:** Deployment and final review.
- **Milestones:** Code refactor complete, tests passing with real data, framework integrated without any compromises.

## Potential Issues and Mitigations
- **API Flakiness:** Mitigate with retry mechanisms and design tests to handle real-world variability authentically, with no fallback options.
- **Cost Management:** Implement a budget tracker and use strategies like scheduling tests during off-peak hours, always with real API interactions.
- **Data Privacy:** Use the provided art assets directly as per user permission, with no additional checks, focusing on real integration.
- **Dependency Management:** Use virtual environments and requirements.txt to handle library versions for API clients, ensuring all dependencies are tested with live services.

This plan has been revised to strictly adhere to real-world implementations, removing all references to mocks or simulations as per user feedback.
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