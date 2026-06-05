# Art Agent Team Prompt Design Documentation

## Overview

This document provides a comprehensive overview of the LLM prompts used across the Art Agent Team codebase. It includes details about the prompts, the scripts/agents that use them, and their purposes.

## Prompt Documentation

### 1. DocentAgent

*   **Function:** `_think_with_grok` (Original internal reasoning)
    *   **Prompt Text:** `prompt` parameter passed to the function (e.g., "Preparing to orchestrate...")
    *   **LLM Model:** `grok-3-mini-fast-high-beta`
    *   **Purpose:** High-level internal reasoning/planning for the agent. Currently logged, not directly altering workflow logic based on user choice.
    *   **Input:** String prompt.
    *   **Output:** Text response (logged).

*   **Function:** `_refine_agent_input_with_grok` (New inter-agent refinement)
    *   **Prompt Structure:**
        ```
        Generate refined parameters/instructions for the upcoming {target_agent_name} stage.
        Context:
        - Target Image: {current_image_path}
        - Previous Stage Results: {previous_stage_results_summary}
        - User Preference Input: {user_context_input}

        Based on this context, provide parameters tailored for the {target_agent_name} agent.
        Respond ONLY with a JSON dictionary containing relevant keys for the {target_agent_name} agent.
        Expected keys for Vision: [e.g., 'focus_areas', 'crop_preference']
        Expected keys for Upscale: [e.g., 'upscale_model', 'preserve_texture_level']
        Expected keys for Placard: [e.g., 'placard_style', 'additional_notes']
        If no specific refinement is needed based on the context, return an empty JSON object {{}}.
        ```
    *   **LLM Model:** `grok-3-mini-fast-high-beta`
    *   **Purpose:** To analyze context (previous results, user input) and generate specific, structured parameters/instructions to enhance the input for the *next* agent in the sequence.
    *   **Input:** `target_agent_name` (str), `current_image_path` (str), `previous_stage_results` (dict/None), `user_context_input` (str).
    *   **Output:** JSON dictionary containing refined parameters (e.g., `{'upscale_model': 'RealESRGAN_x4plus'}`) or an empty dictionary `{}`. Structure is validated post-call.

            docent_persona_prompt = """
You are DocentAgent, an expert art historian with specialized experience leading collaborative teams of researchers and art restorationists. Your deep knowledge spans art history, artistic techniques, cultural contexts, and digital restoration. Your core mission is to assist users by analyzing, discussing, restoring, and presenting artworks they share.

Your capabilities include:
1.  **Identification & Analysis:** Accurately identify artwork origin, artistic style, and key features through detailed visual and contextual analysis.
2.  **Insightful Description:** Provide thorough descriptions highlighting artistic techniques, symbolism, and cultural relevance.
3.  **Engaging Discussion:** Facilitate discussions on historical context, artistic significance, and critical interpretations to foster user learning.
4.  **Digital Restoration (Upscaling):** Enhance artwork quality by upscaling images to reveal finer details while preserving the original artistic intent.
5.  **Informative Labeling (Placard):** Create museum-style placards with comprehensive metadata (title, artist, date, style, cultural notes, provenance).
6.  **Optimal Presentation (Cropping):** Intelligently crop artworks to a 16:9 aspect ratio suitable for modern digital displays, ensuring key compositional elements are preserved.
7.  **Conversational Interaction:** Engage in natural, bidirectional dialogue, responding to queries, providing clarifications, and building rapport.
8.  **Adaptive Workflow Execution:** Interpret user requests, even when described in free-form language, to autonomously select and sequence the appropriate workflow stages (Research, Vision, Upscale, Placard, Cropping).

When preparing to orchestrate a workflow, consider the user's request and determine the optimal sequence of actions based on these capabilities.
"""

### 2. ResearchAgent

*   **Function:** `get_vision_prompt` (within `PromptTemplate` class)
    *   **Prompt Structure:**
        ```
        Analyze the image and filename tokens provided. Extract the following fields:
        1. author
        2. title
        3. date
        4. nationality
        5. style (from the predefined list: {style_list})
        6. movement (from the predefined list:
        {movement_list}
        )
        7. primary_subjects
        8. secondary_subjects
        9. brief_description
        10. confidence_score
        11. grounding_used
        12. limitations

        Validate the identified 'date' against the time range associated with the identified 'movement' from the provided list. Ensure the date falls within the movement's time range.

        Output the results in JSON format only.

        Additional instructions for the model: {model_specific_instructions}

        Filename tokens: {filename_tokens}
        ```
    *   **LLM Models:** Various vision models via ModelRegistry (e.g., Gemini, Grok Vision, OpenRouter models).
    *   **Purpose:** To analyze an image and filename, extract metadata, validate date/movement, and return structured JSON.
    *   **Input:** Image bytes, filename tokens, model name (for specific instructions).
    *   **Output:** JSON dictionary containing artwork metadata.

*   **Function:** `get_consolidation_prompt` (within `PromptTemplate` class)
    *   **Prompt Structure:**
        ```
        Consolidate the results from multiple vision models provided as a JSON string. Reference the 'movement' field and use the provided movement list (
        {movement_list}
        ) to validate the consolidated 'date' against the consolidated 'movement''s time range. If there are conflicts or low confidence, choose the most plausible movement.

        Output the consolidated results in JSON format only.

        Input Data (JSON string containing results from various models):
        {results_json_string}
        ```
    *   **LLM Model:** `CONSOLIDATION_MODEL` (currently `grok-3-mini-fast-high-beta`).
    *   **Purpose:** To synthesize results from multiple vision models into a single, consistent metadata record, validating date/movement.
    *   **Input:** JSON string containing results from individual vision model calls.
    *   **Output:** JSON dictionary containing consolidated artwork metadata.

*   **Function:** `categorize_artwork` (calls `_think_with_grok`)
    *   **Prompt Structure:**
        ```
        Given the following artwork metadata:
        - Date: {date}
        - Movement: {movement}
        Here is a list of recognized art movements and their time ranges:
        {movement_list_formatted_as_bullet_points}

        Analyze whether the provided date fits the movement's time range.
        If not, suggest the most appropriate movement from the list above based on the date.
        Return only the final movement string (from the list above) that best fits the date, or the original movement if it is consistent.
        ```
    *   **LLM Model:** `grok-3-mini-fast-high-beta`
    *   **Purpose:** Internal reasoning step within `categorize_artwork` to validate/refine the artwork's movement based on its date and a predefined list of movements.
    *   **Input:** Artwork date (str), artwork movement (str), formatted movement list (str).
    *   **Output:** Text response containing the suggested final movement string.

### 3. VisionAgentPortrait (Example Vision Agent)

*   **Function:** `analyze_image`
    *   **Prompt Text:**
        ```markdown
        Analyze this portrait painting. Focus intensely on identifying and localizing the face and head of the primary subject, and all individual facial features (eyes, nose, mouth, ears, hair, facial hair). Also identify any secondary subjects and important objects.

        Research Information (if available):
        {metadata_summary}

        User Preferences (if provided):
        {user_context_input}

        Identify objects and their bounding boxes [y0, x0, y1, x1] (normalized 0-1000). Prioritize based on context and standard portrait analysis:
        - The primary subject's face and head
        - Individual facial features of the primary subject (eyes, nose, mouth, ears, hairline, facial hair, neck)
        - Secondary subjects (people or animals)
        - Important objects (jewelry, clothing details, props mentioned in research or user input)
        - Background elements (less important unless specified)

        For each identified object, provide:
        - label: Concise description (e.g., "primary subject's face", "left eye", "pearl necklace", "background curtain")
        - box_2d: [y0, x0, y1, x1]
        - confidence: 0.0-1.0
        - type: Category (e.g., "human_face", "human_figure", "animal_face", "animal_figure", "facial_feature", "object", "background")
        - importance: Initial estimate 0.0-1.0 (will be refined by agent)
        - features: List of detected facial features if applicable (for face/head objects)

        Format as JSON:
        {{
            "objects": [
                {{
                    "label": "...",
                    "box_2d": [y0, x0, y1, x1],
                    "confidence": ...,
                    "type": "...",
                    "importance": ...,
                    "features": [...]
                }}
            ]
        }}
        ```
    *   **LLM Models:** Specific models configured for the VisionAgent (e.g., Gemini Pro, Grok Vision).
    *   **Purpose:** Analyze portrait paintings focusing on faces, heads, features, and objects, incorporating research metadata and user context if available.
    *   **Input:** Image path, `metadata` (dict/None), `refined_params` (dict/None - potentially containing user context).
    *   **Output:** JSON dictionary containing identified objects and their details.

*(Note: Prompts for other specific VisionAgents like Landscape, Genre, etc., would follow a similar pattern but tailored to their specific analysis goals.)*

## Updated Optimization Analysis

1.  **Clarity and Specificity:** Prompts generally maintain good clarity. The dynamic prompts in `DocentAgent._refine_agent_input_with_grok` explicitly request JSON, which aids predictability. Including summaries of previous results and user context enhances specificity for inter-agent refinement.
2.  **Model Customization:** The system uses different models (Grok, Gemini, OpenRouter models). The `ResearchAgent` includes basic model-specific instructions. Further customization within `_refine_agent_input_with_grok` or individual agent prompts based on the target LLM could yield better results.
3.  **Few-Shot Learning:** No explicit few-shot examples are currently used. Adding 1-2 examples within the prompts (especially for JSON generation or complex reasoning tasks like refinement) could significantly improve output consistency and quality.
4.  **Context Handling:** The new `_refine_agent_input_with_grok` explicitly incorporates context from previous stages and user input, which is a good practice. Ensuring the summaries passed are concise yet informative is key.
5.  **Error Handling:** Agents generally handle API errors and parsing errors. The refinement step includes validation of the LLM's JSON output structure.

## Updated Summary Report

*   **Total Prompts Documented:** 6+ distinct prompt structures identified (including dynamic variations).
*   **Patterns:** Increased use of dynamic prompt construction (`DocentAgent`, `ResearchAgent`). Continued mix of JSON and text outputs. Explicit context passing between stages via the refinement step.
*   **Areas for Improvement:**
    1.  Implement few-shot examples, particularly for JSON generation and the refinement task in `_refine_agent_input_with_grok`.
    2.  Enhance model-specific instructions/tuning, especially for the different vision models used by `ResearchAgent` and `VisionAgent` subclasses.
    3.  Refine the summarization logic for `previous_stage_results` passed to the refinement step to ensure optimal context without excessive length.
    4.  Consider adding more robust validation for the *content* (not just structure) of LLM responses where critical (e.g., ensuring bounding box coordinates are valid).

## Updated Next Steps

1.  Prioritize adding few-shot examples to key prompts (e.g., `_refine_agent_input_with_grok`, `ResearchAgent.get_vision_prompt`).
2.  Investigate and implement more model-specific tuning within prompts.
3.  Refine context summarization passed between agents.
4.  Continuously monitor LLM outputs and adjust prompts based on performance, focusing on consistency and accuracy.