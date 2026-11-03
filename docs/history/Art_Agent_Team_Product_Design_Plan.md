# AI Art Team Project: Product Design Plan

## Introduction
This document outlines the product design specifications for the AI Art Team project, focusing on processing artwork images to meet specific aesthetic and functional requirements. The design plan addresses aspect ratio adjustments, upscaling to 4K resolution, and plaque integration while preserving artistic integrity through careful consideration of composition, color balance, historical context, and emotional impact.

## Design Objectives
- **Aspect Ratio Adjustment**: Transform images to a 16:9 aspect ratio using intelligent cropping that prioritizes key focal points and compositional elements.
- **Upscaling to 4K Resolution**: Enhance images to 4K resolution (3840x2160 pixels) using advanced algorithms to maintain fine details and minimize artifacts.
- **Plaque Integration**: Overlay informative plaques with artwork metadata (title, artist, nationality, date) in an aesthetically pleasing manner that minimizes visual disruption.
- **Artistic Integrity**: Ensure all modifications respect and enhance the original artwork's artistic sensibilities across diverse genres.

## Artistic Sensibilities Definition
In the context of image processing for the AI Art Team project, "artistic sensibilities" are defined as follows:
- **Composition**: The arrangement of visual elements within the artwork, including balance, symmetry, and focal points. Modifications must preserve the intended structure and guide the viewer's eye as the artist intended.
- **Color Balance**: The harmony and distribution of colors within the artwork. Processing should avoid altering color tones or contrasts in ways that detract from the original mood or style.
- **Historical Context**: The cultural and temporal background of the artwork, which influences its interpretation. Metadata and modifications should reflect an understanding of the artwork's era and artistic movement.
- **Emotional Impact**: The intended emotional response elicited by the artwork. Processing must avoid changes that diminish the emotional resonance, such as cropping out key emotive elements or altering color schemes that contribute to the mood.

These sensibilities will guide all design decisions to ensure that processed images maintain or enhance the original artistic intent.

## Feature Specifications

### 1. Aspect Ratio Adjustment (16:9)
- **Objective**: Adjust artwork images to a 16:9 aspect ratio suitable for modern display formats while preserving critical compositional elements.
- **Methodology**: Utilize intelligent cropping algorithms within genre-specific `VisionAgent` classes to identify and prioritize key focal points.
  - **Focal Point Identification**: Use dual-model analysis (e.g., Gemini Pro and Grok Vision) to detect primary and secondary subjects, assigning importance scores based on genre-specific modifiers (e.g., terrain in landscapes, faces in portraits).
  - **Crop Optimization**: Evaluate multiple crop options to maximize inclusion of high-importance elements, adjusting crop boundaries to maintain balance and symmetry. For example, in landscapes, prioritize wide vistas and horizon lines; in portraits, ensure faces are centered or follow the rule of thirds.
  - **Constraints**: Minimize loss of critical content by ensuring at least 90% of high-importance objects (importance score > 0.8) are retained within the cropped area.
- **Output**: Cropped image at 16:9 aspect ratio saved as a separate file (e.g., `<basename>_cropped.jpg`).

### 2. Upscaling to 4K Resolution
- **Objective**: Enhance cropped images to 4K resolution (3840x2160 pixels) to provide high-quality outputs suitable for large displays or prints.
- **Methodology**: Implement an `UpscaleAgent` using advanced deep learning-based upscaling techniques, with a preference for ESRGAN (Enhanced Super-Resolution Generative Adversarial Network) as specified by the user.
  - **Detail Preservation**: Focus on preserving fine details such as brush strokes, paint layering, and texture of the canvas or other mediums. ESRGAN should be tuned to prioritize texture and edge sharpness over noise reduction to avoid smoothing out artistic details.
  - **Artifact Reduction**: Apply post-processing filters to minimize upscaling artifacts, ensuring that generated pixels blend seamlessly with original content. Target a peak signal-to-noise ratio (PSNR) of at least 30 dB compared to a reference upscaled image.
  - **Resolution Target**: Ensure output dimensions match 4K (3840x2160) while maintaining the 16:9 aspect ratio. If the cropped image does not match exactly, pad with a neutral border (e.g., black or a color derived from the image's edge pixels) to fit.
- **Performance**: Process images within 30 seconds per image on standard hardware (e.g., GPU with 8GB VRAM) to maintain workflow efficiency.
- **Output**: Upscaled image saved as a separate file (e.g., `<basename>_upscaled.jpg`).

### 3. Plaque Integration
- **Objective**: Overlay an informative plaque on the processed image containing metadata (title, artist, nationality, date) in a visually appealing and non-intrusive manner.
- **Design Specifications**:
  - **Aesthetic**: Textured white cardstock appearance as the background, with black sans-serif lettering (e.g., Helvetica or Arial) for readability. No border to maintain a clean, minimalist look.
  - **Positioning**: Place the plaque in the lower right corner of the image, slightly offset from the edges (e.g., 2% of image width/height as margin) to avoid covering key content.
  - **Size**: Size the plaque to be inconspicuous yet readable, fitting the entire text content. Target a width of 20-30% of the image width and adjust height dynamically based on text length, with a minimum font size of 12pt at 4K resolution for legibility.
  - **Content Layout**: Format text as follows:
    - Line 1: **Title** (bold)
    - Line 2: Artist (italic)
    - Line 3: Nationality, Date (plain text)
    - Use line spacing of 1.2 for clarity.
  - **Opacity**: Apply a semi-transparent background (e.g., 80% opacity for the cardstock) to ensure the plaque blends with the image while maintaining text readability.
- **Implementation**: Develop a `PlacardAgent` to handle plaque design and integration, using image processing libraries like PIL (Python Imaging Library) to render text and overlay the plaque on the upscaled image.
- **Output**: Final image with plaque saved as a separate file (e.g., `<basename>_final.jpg`).

### 4. Artistic Integrity Assurance
- **Objective**: Ensure all modifications respect the defined artistic sensibilities across genres.
- **Methodology**:
  - **Composition Check**: Post-cropping, validate that key focal points (importance score > 0.8) are retained within the frame using bounding box overlap metrics (IoU > 0.9 with original positions).
  - **Color Fidelity**: Post-upscaling, compare color histograms of the processed image against the original to ensure no significant shifts (target histogram correlation > 0.95). If deviations occur, apply color correction to align with original tones.
  - **Historical Context**: Use metadata from `ResearchAgent` to inform processing decisions, such as avoiding modern color profiles for historical artworks or ensuring plaque text reflects accurate historical data.
  - **Emotional Impact**: Leverage genre-specific importance scoring to prioritize elements tied to emotional resonance (e.g., faces in portraits, dramatic skies in landscapes) during cropping and upscaling.
- **Validation**: Implement a feedback loop where processed images are analyzed by a secondary model (e.g., a lightweight CNN) trained on art critique metrics to score artistic fidelity, flagging images for manual review if scores fall below a threshold (e.g., 0.85 on a 0-1 scale).
- **Output**: Artistic fidelity report saved alongside final images for transparency (e.g., `<basename>_fidelity_report.txt`).

## Integration with Existing System
- **Workflow Integration**: Extend the `DocentAgent` workflow to include sequential processing by `UpscaleAgent` and `PlacardAgent` after `VisionAgent` cropping. Use queues to manage handoff between agents, ensuring thread-safe operations.
- **Genre-Specific Processing**: Maintain genre-specific logic in `VisionAgent` classes for cropping, passing genre data to `UpscaleAgent` and `PlacardAgent` to adjust processing (e.g., different upscaling sharpness for surrealist vs. realist artworks).
- **Configuration**: Store design parameters (e.g., plaque aesthetics, upscaling algorithm settings) in the central configuration file (`config.yaml`) for easy adjustment and consistency across agents.

## User Experience Considerations
- **Input Flexibility**: Allow users to specify input folders and override default design parameters (e.g., plaque position or font) via command-line arguments or a GUI interface in future iterations.
- **Output Organization**: Save all intermediate and final outputs in a structured output directory with clear naming conventions to facilitate review and debugging.
- **Progress Feedback**: Provide real-time feedback during processing via logging or a progress bar to inform users of workflow status, especially for batch operations.

## Conclusion
This product design plan provides detailed specifications for enhancing the AI Art Team project to meet user requirements while preserving artistic integrity. By focusing on intelligent cropping, advanced upscaling with ESRGAN, and aesthetically integrated plaques, the system will deliver high-quality processed artwork images. The next step is to translate these specifications into a technical implementation plan for the development team.
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