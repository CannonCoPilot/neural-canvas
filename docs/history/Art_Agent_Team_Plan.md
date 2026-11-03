# Art Agent Team Plan

This document outlines the plan for developing a team of AI agents to process and enhance art images.

## Phase 1: Initial Setup & Vision Agent (Completed)

*   **Goal:** Create a basic workflow where a single agent analyzes an image, identifies key features, and performs an intelligent crop to a 16:9 aspect ratio.
*   **Agents:**
    *   `VisionAgent`: Analyzes image using Google Vision API, identifies objects/landmarks/labels, calculates feature importance, performs intelligent cropping, saves labeled and cropped outputs.
*   **Status:** Completed. Basic analysis and cropping implemented and tested. Enhanced with Gemini API integration, improved scoring, and detailed facial feature detection.

## Phase 2: Workflow Redesign & Specialization (In Progress)

*   **Goal:** Redesign the workflow for a multi-agent system with specialized roles for research, style-specific analysis, and upscaling. Implement asynchronous processing and dual-model analysis.
*   **Agents & Workflow:**
    1.  **`DocentAgent` (Orchestrator):**
        *   Interacts with the user: Prompts for image input path, explains processing steps.
        *   Manages Image Stack: Fetches images from the input folder (`/Users/nathaniel.cannon/Documents/VScodeWork/Art_AI/art_agent_team/tests/test_data/input`) and places them in a processing queue (`image_queue`).
        *   Manages Research Agents: Initializes and manages a pool of 8 `ResearchAgent` worker threads.
        *   Manages Vision Handoff: Receives results (including style) from `ResearchAgents` via `vision_input_queue`. Routes tasks to the appropriate specialized `VisionAgent` instance (implementation TBD - could be direct call or another queue/worker pool).
        *   Manages Upscale Handoff: Receives cropped image paths from `VisionAgents` via `upscale_queue`. Routes tasks to `UpscaleAgent` (implementation TBD).
        *   Uses LLM (`grok-3-mini-fast-beta`) for coordination if needed.
    2.  **`ResearchAgent` (8 Instances):**
        *   Pulls image path from `DocentAgent`'s `image_queue`.
        *   Performs Text Search: Uses image filename tokens for web search (using Google Search API or similar), summarizes findings using LLM (`grok-3-mini-fast-beta`).
        *   Performs Image Search: Uses the image for reverse image search (using appropriate API/tool), summarizes findings using LLM.
        *   Confirms Artwork Identity: Compares text and image search summaries using LLM.
        *   Generates Descriptions (using LLM):
            *   Creates a paragraph-length summary description.
            *   Creates a structured one-sentence description: "This is a(n) {style} painting of a(n) {primary_subject} {description_of_action} set in a {description_of_scene}, together with {secondary_subjects}."
        *   Determines Style (using LLM): Assigns one style from {abstract, landscape, still life, surrealist, genre-scene, animal-scene, portrait, figurative, religious/historical}.
        *   Identifies Subjects (using LLM): Determines primary and secondary subjects based on style rules.
        *   Handoff to Docent: Puts results (image path, descriptions, style, subjects) onto `DocentAgent`'s `vision_input_queue`.
    3.  **`VisionAgent` (8 Style-Specific Scripts):**
        *   Receives data (image path, descriptions, style, subjects) from `DocentAgent` (or router).
        *   Fetches the image file.
        *   Dual-Model Analysis:
            *   Sends customized prompts (incorporating `ResearchAgent` descriptions and tailored to the specific artwork style) to `grok-2-vision-1212` and `gemini-2.5-pro-preview-03-25`.
        *   Aggregates Results: Combines object identification results from both models into a unified set (handling potential conflicts/duplicates).
        *   Calculates Importance Scores (Style-Specific): Uses the refined scoring system with style-specific adjustments. Boosts primary subject score.
        *   Identifies Important Objects: Uses percentile thresholding.
        *   Generates Segmentation Masks: Creates masks only for the most important objects.
        *   Performs Style-Specific Cropping: Implements cropping logic tailored to the artwork style (Abstract, Landscape, Still Life, etc.) as defined in the plan.
        *   Saves Outputs: Generates labeled (boxes), masked (important objects), and cropped versions.
        *   Handoff to Docent: Places the cropped image path onto `DocentAgent`'s `upscale_queue`.
    4.  **`UpscaleAgent`:**
        *   Receives cropped image path from `DocentAgent`'s `upscale_queue`.
        *   Performs image upscaling (specific method TBD).
        *   Saves final upscaled image.
*   **Status:** Refactoring in progress. Specialized VisionAgent files created. ResearchAgent and DocentAgent refactoring pending. Style-specific logic implementation pending.

## Phase 3: Placard Generation & Final Output

*   **Goal:** Add placard generation and finalize the output format.
*   **Agents:**
    *   `PlacardAgent`: Receives analysis data, generates a museum-style placard.
*   **Status:** Not started.

## Functional Review (Pre-Refactor)

*   **`DocentAgent`:** Needs significant refactoring for async worker management (Research & Vision agents), queue-based handoffs, removal of outdated logic (LLM parsing, direct agent calls), and integration of 8 ResearchAgents. Needs model update to `grok-3-mini-fast-beta`.
*   **`ResearchAgent`:** Requires complete implementation of core logic: web/image search integration (API placeholders needed), LLM integration (`grok-3-mini-fast-beta`) for summarization/comparison/generation/classification, style/subject identification logic, and proper handoff via queue.
*   **`VisionAgent` (Template):** Needs Grok Vision API integration (replacing Vision API client), dual-model analysis (Gemini Pro + Grok Vision), result aggregation, style-based routing logic (or be replaced by specialized agents), primary subject score boosting, and handoff mechanism. Input method needs changing.
*   **Specialized `VisionAgents`:** Created but contain template code. Require customization of prompts, scoring, and cropping logic per style.

## Implementation Roadmap

1.  **Update Plan:** Add functional review and roadmap (Completed).
2.  **Dependencies:** Update `requirements.txt` (add `google-api-python-client`, `openai` for xAI Grok). Run `pip install`.
3.  **`ResearchAgent` Refactor:**
    *   Implement core structure (init with ID, worker loop).
    *   Add LLM (`grok-3-mini-fast-beta`) init.
    *   Implement placeholder search functions.
    *   Implement LLM calls for all generation/classification tasks.
    *   Implement style determination logic.
    *   Implement handoff to `vision_input_queue`.
4.  **`DocentAgent` Refactor:**
    *   Update `__init__` for 8 `ResearchAgent` instances and new queues.
    *   Refine worker management loop.
    *   Implement basic routing logic (log target VisionAgent type).
    *   Update model to `grok-3-mini-fast-beta`.
5.  **`VisionAgent` Template Refactor:**
    *   Update `__init__` for Grok Vision (`grok-2-vision-1212`) & Gemini Pro (`gemini-2.5-pro-preview-03-25`).
    *   Modify `analyze_image` signature.
    *   Implement dual-model calling & basic aggregation.
    *   Implement primary subject score boosting.
    *   Add handoff logic (`upscale_queue`).
    *   Remove hardcoded input folder.
6.  **Update Specialized `VisionAgent` Files:** Copy refactored template code into the 8 specialized files, ensuring class names match filenames.
7.  **Customize Specialized `VisionAgents` (Iterative - One style at a time):**
    *   **Landscape:** Customize prompts, scoring (discount small objects), cropping (mask avoidance, bottom-up crop).
    *   **Abstract:** Customize prompts, scoring (shape-based), cropping (centroid-based).
    *   **Still Life:** Customize prompts, scoring (central cluster focus), cropping (preserve cluster centrality).
    *   **Surrealist:** Customize prompts, scoring (context focus), cropping (follow Landscape).
    *   **Portrait:** Customize prompts (facial features), scoring (discount background), cropping (prioritize face).
    *   **Figurative:** Customize prompts (faces, figures), scoring, cropping (prioritize faces then figures).
    *   **Genre/Religious/Historical:** Customize prompts, scoring (discount crowds/background), cropping (preserve action centrality).
    *   **Animal-Scene:** Customize prompts (animal features), scoring (discount background), cropping (prioritize subjects).
8.  **Implement `VisionAgent` Routing:** Update `DocentAgent` to instantiate/call the correct specialized `VisionAgent`.
9.  **Implement `UpscaleAgent`:** Basic upscale logic, pull from `upscale_queue`.
10. **Update `main.py`:** Orchestrate the full flow.
11. **Testing:** Write/update tests.

## Key Technologies

*   Python 3.12+
*   Google Gemini API (`google-generativeai`) - Models: `gemini-2.5-flash-preview-04-17`, `gemini-2.5-pro-preview-03-25`
*   Grok API (`openai` for xAI Grok) - Models: `grok-2-vision-1212`, `grok-3-mini-fast-beta` (Requires integration)
*   Pillow (PIL Fork) for image manipulation
*   NumPy for numerical operations
*   PyYAML for configuration
*   Web search library (e.g., `google-api-python-client`, `requests`, `BeautifulSoup`, or `serpapi`)
*   Asynchronous programming library (e.g., `threading`, `queue`, potentially `asyncio`)


# To add: Focusing and Future Steps for the Roadmap
* A main aim of the project is to customize and optimize each VisionAgent variant to the nuanced differences between the genres of art that they analyze.  Focus on customizing, optimizing, and troubleshooting the object detection, object localization, object scoring, object bounding and masking, and the cropping logic.
* Focus on designing the VisionAgents' functions, prompts, and workflow algorithm to identify sets of the major high-ranking features of an image, and then decide how to minimize loss when cropping to the 16:9 aspect ratio.
* Focus on customizing the algorithms to identify the set of high ranking objects in an image, and then decide how to best preserve them when cropping to the 16:9 aspect ratio cropping boundary.
* We need to ensure VisionAgents use the newest and best LLMs for image analysis.  For now, always use 'gemini-2.5-pro-preview-03-25' and 'grok-2-vision-1212' for image analysis and artistic reasoning.
* Allow the "thinking" steps done by the VisionAgents, DocentAgent, ResearchAgent and other agents to be done by 'grok-3-mini-fast-beta' and should include 'reasoning: { effort: "high" }' in the prompt.

* We need to ensure that we are calling the correct tools for Gemini to identify objects in the image and retun bounded boxes on those objects. Import the necessary generative, reasoning, and image processing modules.
* The vision agents explicitly instruct the LLMs through prompting in order to identify things like animals, people, major landscape features, furniture, and so on.  
* Explicitly instruct the models to get bounding boxes, and then the agent calls functions to plot them onto the original image, label them
* VisionAgents will verify that all detected objects are given their own scores, that high-importance objects have bounding boxes, that high-importance objects have sophisticated high-polygon segmental masks
* VisionAgents will implement a cropping algorithm that leverages all of that object metadata (boxes, masks, scores, and flags like "primary_subject" or "prominent_object" to calculate the a cropping boundary that avoids cropping important features, maximizes the cropping boundary, and avoids leaving partially cropped mid-to-high scoring (aka "important") objects.
* When a single object takes up the majority of an image and needs to be bisected by the cropping boundary, the algorithm should identify faces and attempt to retain as many faces of high ranking objects within the cropping boundary.
* For optimizing the cropping logic, first implement a region-based algorithm that evaluates minimal bounding regions around the top N features (expanding them to the 16:9 frame)
* Then modify the cropping logic by integrating face-detection so that when a single or few large promary or secondary subjects dominate the image, the cropping boundary will include as many detected faces as possible.
* Create logic that can identify faces as their own scored objects.  These should be the highest scoring features.  The portrait and figurative vision agents should identify human faces and the animal-scene agent should identify animal faces.  When possible the cropping algorithm should attempt to keep face objects within the cropping boundary.
* Create tests that will output the number and types of objects identified by the vision model.  Have the test script present all of the information from the returned output of the model in a table format, and also save as a csv file, to compare those results with a visual inspection of the image.
* Explore and adopt sophisticated high-polygon segmental masking algorithms and techniques to minimize unnecessary penalties when cropping around objects.
* API calls sent to grok and gemini models by the vision agents will be essentially and functionally equivalent, allowing for the minor structural differences in the APIs.  This is so that 
* API keys should be properly parsed from the config file and stored as environmental variables.

# To add details to the agents and workflow
The overall purpose of this redesign is to have the DocentAgent do a preliminary search on the artwok in question.  The DocentAgent will use text from the image file name to to a text-based web search, and use the image to do an image search.  It will then create a custom prompt that will be passed as a texts string for the VisionAgent to add to it's prompts.  The Vision agent will use the custom prompt text from the DocentAgent to decide among several object detection and scoring algorithms.  There will be a different algorithm with different prompt text for each of several different categories of artwork.  Here is the workflow in more detail:

1) DocentAgent interacts with the user. It prompts the user to provide a path to image files for processing.  It explains to the user the steps of processing the images. DocentAgent then fetches the image files from the input folder.  It then hands each image to a ResearchAgent.
2) There are many ResearchAgents that interact asyncronously with DocentAgent and with the various customized VisionAgents. ResearchAgents perform a text web search using the tokens from the image filename and returns a brief summary description of the artwork.
3) ResearchAgents perform an image web search using the image and returns a brief summary description of the artwork.
4) ResearchAgents compare these two summary decriptions to confirm that it has correctly identified the artwork.
5) ResearchAgents combine the information from the two summaries and rewrites a paragraph-length description of the artwork.
6) ResearchAgents use their knowledge of the artwork to write a one sentence description of the artwork in the format: "This is a(n) {style} painting of a(n) {primary_subject} {description_of_action} set in a {description_of_scene}, together with {secondary_subjects}."  
   6a) For this key description the ResearchAgents will choose from the following values to assign to the following parameters, or will use the following rules:
style = {abstract, landscape, still life, surrealist, genre-scene, animal-scene, portrait, figurative, religious/historical}
   6b) primary_subject = the ResearchAgents should provide a simple phrase describing what it believes to be the main subject (ie primary focus of the painting).  
   6c) If the painting style is 'landscape' then the primary subject should be defined as the main landscape feature (e.g. valley, mountain, river). 
   6d) If the painting is 'abstract' then the primary subject should be defined with a short descriptive phrase of the shapes that appear.
   6e) If the painting is surrealist then the primary subject should be defined with a short descriptive phrase about the scene.
   6f) description_of_action = the ResearchAgents should provide a short phrase describing the action, or attitude, or position, of the primary subject, or of the spatial relationship of the main subject in relation to the scene of the painting. 
   6g) description_of_scene = the ResearchAgents should provide a short phrase describing the scene in which the primary subject and secondary subjects are set.
   6h) secondary_subjects = the ResearchAgents will provide a list of other important subjects in the painting.  To do this it will evaluate up to 10 objects as candidates for secondary subjects, it should choose 1-3 objects as secondary subjects based on their visual prominence or contextual importance as depends on the nature of the scene.
7) Each ResearchAgent will then call VisualAgent and pass to it a list containing the image filename, paragraph-length summary description, and the standard-format simple description for the image file it has processed.
8) VisualAgent will fetch the image file found in the dictionary passed to it by ResearchAgent.
9) VisualAgent will parse the simple description to get the painting style.  Depending on the style the VisualAgent will select one of several function-prompt-constructs (see e.g. 'analyze_image' function) which are customized to visually analyze art of that style.  It will identify and score objects, create box-boundaries around the objects, and create hi-poly segmentation masks for the highest scoring objects.
10) The primary subject's score will always be increased to 2x the highest score from among all other objects, or keep its own raw score, whichever is greater.
11) The VisualAgent's function-prompt-constructs will each follow the same general process (see analyze_image, _calculate_object_importance, _threshold_important_objects and other functions) but will be customized to best handle the kinds of primary and secondary subjects typical of each genre of painting.
12) For coding simplicity and cleanliness, it may be best to create 'clones' of the VisionAgent and then further customize each one independently.  This will avoid unneccessarily re-writing and potentially breaking a single very large Agent script, and will allow for independent testing of each VisionAgent, and for individual modification of each VisionAgent.
13) The VisionAgent(s) should take the customized description and summary texts from the ResearchAgent and incorporate them into their  pre-written prompt(s) to send to vision-capable models.
14) The current 'template' VisionAgent workflow should be modified to utilize TWO image-input able AI models.  The VisionAgent(s) will send their customized prompts to 'grok-2-vision-1212' and to 'gemini-2.5-pro-preview-03-25'.
15) The VisionAgent will aggregate the object identification results of both AI models into a unified set of object localizations with category labels (or labels to be used for categorization) to be used for calculating object importance scores.
16) The VisionAgent(s) will then proceed through the rest of the workflow, basically as currently defined, to create an optimized 16:9 cropping of the image. It will continue to output a labelled, masked, and cropped version of each image it processes.
17) The cropping algorithm for each VisionAgent variation will be customized, following these general rules:
   Abstract - Bounding boxes should localize the abstract shapes of significant size in the image. Then the VisionAgentAbstract should calculate a weighted centroid of the identified shapes in the image. Then it adjusts the cropping boundary to be as centered on the centroid as possible while maintaining 16:9 ratio and maximizing its size.
   Landscape - Bounding boxes should localize important landscape features (e.g. prominent distant mountains, waterfalls, rivers, foreground plants or rocks, and so on).  Scoring shuold effectively discount small objects, unless they are very contextually important.  Then it should create hi-poly segmentation masks of the three most prominent/important high-scoring objects.  Then it should adjust the 16:9 cropping boundary to try to avoid bisecting the segmentation masks.  If it must bisect masks to crop to 16:9 it should bisect the masks beginning with the lowest scoring of the three high-prominence objects, finally only truncating the mask fo the principle_subject if necessary to enforce the 16:9 aspect ratio.  It should generally truncate the primary subject from bottom up when cropping in the y-axis direction.  It should maintain the centrality or offset nature of the primary subject if cropping it in the x-azis direction. It should give a warning if it truncates the primary subject.
   Still life - Bounding boxes should localize the fruit, plants, flowers, food, and everyday objects grouped near the center of the image.  Scoring should effectively discount identified objects outside of this central cluster.  The cropping boundary should preserve the approximate x,y centrality or offset of the centroid of the central cluster of objects.
   Surrealist - Bounding boxes are likely to localize objects which defy conventional categorization.  Object scoring should be based on perceived context and importance.  Cropping should otherwise generally follow the logic laid out in the 'Landscape' algorithm.
   Portrait - Bounding boxes should principally localize the facial features of the face and head of the individual in the portrait.  Bounding boxes may also localize other objects of contextual significance.  Scoring of objects should effectively discount background features.  The cropping boundary should attempt to maintain the centrality or offset of the cluster of bounding boxes that define the facial features of the primary subject, and should aim to avoid cropping the facial features of any secondary subjects if possible.
   Figurative - Bounding boxes should localize the facial features of primary and secondary subjects, and shuold localize the entirety of the figures of teh primary and secondary subjects.  The cropping boundary should prioritize keeping the faces of the subjects, then the figures of the subjects, and lastly preserve the original centrality or offset of the primary subject.
   Genre-scene - Bounding boxes might identify a large number of individual people or animals. Scoring should be optimized to discard background objects, and people in crowds, and should indicate the higher importance of primary and secondary subjects and foreground objects.
   Religious/Historical - Bounding boxes might identify a large number of individual people or animals. Scoring should be optimized to discard background objects, and people in crowds, and should indicate the higher importance of primary and secondary subjects and foreground objects.  The cropping boundary should seek to preserve the centrality of contextion action within the scene, and keep the primary subject in frame. 
   Animal-scene - Bounding boxes might identify a large number of individual animals. Scoring should be optimized to discard animals and objects in the distant background and should indicate the higher importance of primary and secondary subjects and foreground objects.  Idenfification of animal faces should be done using a modified key-word set of features (e.g. including animal face terms like beak, snout, muzzle, mane, tongue among the other facial descriptors used in 'portrait' and 'figurative' feature identification.
18) The DocentAgent should be able to create a 'stack' of images to be handed off to the various ResearchAgents.  It will asynchronously communicate with those ResearchAgents from that 'stack' whenever one ResearchAgent has completed its work on the previous image and handed it off to the correct VisionAgent. 
19) VisionAgents should have a hand-off function to put cropped images into another 'stack' for the UpscaleAgent to pull from.

Parallel Instances of ResearchAgents and of VisionAgents
You do not need multiple copies of the ResearchAgent script to achieve parallel, asynchronous research. In fact, having multiple copies of the same script is unnecessary and can make maintenance harder.

How it works (and should work):

You can run multiple instances of the same ResearchAgent class, each in its own thread or process.
Each instance can independently perform web searches and communicate asynchronously with a single DocentAgent instance, using thread-safe queues (as your current architecture does).
The DocentAgent can coordinate these agents, distribute work, and collect results, then route them to the appropriate VisionAgent specialists.
Why this works
Python classes are reusable: You can instantiate the same class as many times as you want, each with its own state.
Thread/process safety: As long as shared data is managed via thread-safe mechanisms (like Queue), you can safely run many agents in parallel.
Simpler maintenance: You only need to update/fix one ResearchAgent script, not eight.

Plan:

Update Art_Agent_Team_Plan.md: Reflect the new workflow described by the user.
Refactor DocentAgent:
Implement user interaction for image path input.
Explain processing steps to the user.
Fetch images from the input folder.
Create an image processing "stack" (e.g., a queue or list).
Implement asynchronous communication logic to hand off images to ResearchAgent instances.
Manage the stack and track progress.
Create/Refactor ResearchAgent:
Implement text-based web search using image filename tokens.
Implement image-based web search.
Compare search results for confirmation.
Generate paragraph-length description.
Generate the structured one-sentence description based on style rules.
Implement logic for selecting primary/secondary subjects based on prominence/context.
Pass image filename, summary, and structured description to the appropriate VisionAgent.
Refactor VisionAgent (Template):
Remove hardcoded input folder logic (will receive path from ResearchAgent).
Accept image path, summary, and structured description.
Parse the style from the structured description.
Implement logic to select/route to a style-specific VisionAgent clone (or use conditional logic within a single agent initially).
Modify analyze_image to incorporate the summary/description into prompts for grok-2-vision-1212 and gemini-2.5-pro-preview-03-25.
Implement aggregation logic for results from both models.
Keep the core analysis, scoring, masking, and cropping logic as a base.
Implement hand-off function/mechanism for UpscaleAgent.
Create VisionAgent Clones (or implement conditional logic):
Create specialized versions (or conditional blocks) for each style: abstract, landscape, still life, surrealist, genre-scene, animal-scene, portrait, figurative, religious/historical.
Customize the analyze_image prompt, object scoring (_calculate_object_importance), and cropping logic (_find_optimal_crop, _evaluate_crop) for each style according to the rules provided.
Ensure primary subject scoring rule (2x highest other score or raw score) is implemented.
Refactor UpscaleAgent:
Implement logic to pull cropped images from the VisionAgent hand-off stack/queue.
Perform upscaling (details not specified, assume basic upscale for now).
Update main.py: Orchestrate the new workflow, initializing agents and managing the flow.
Update requirements.txt: Add any new dependencies (e.g., web search libraries, async libraries).
Update Tests: Adapt existing tests and create new ones for the refactored agents and workflow.
## LLM Constraints

### ResearchAgent
- Vision Models: Fixed as specified (OpenRouter, Gemini, Grok vision models)
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