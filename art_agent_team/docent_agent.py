import logging
import os
import yaml
import json
import time
import datetime # Added for markdown logging
import importlib
import requests  # Required for test mocking
from typing import Dict, Any, Optional, List, Tuple

from art_agent_team.agents.research_agent import ResearchAgent
from art_agent_team.agents import vision_agent_classes # Import the centrally defined dictionary
from art_agent_team.agents.vision_agent_abstract import CorruptedImageError # Import CorruptedImageError

# Ensure ResearchAgent is available (already imported, but good to keep the check explicit if desired, or remove)
# try:
#     from art_agent_team.agents.research_agent import ResearchAgent # Already imported
# except ImportError:
#     logging.error("ResearchAgent class not found. Ensure the module is correctly imported and available.")
#     raise

# Configure basic logging
# Use a more detailed format including thread name
log_format = '%(asctime)s - %(levelname)s - [%(threadName)s] - %(name)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
# Get a logger specific to this module
logger = logging.getLogger(__name__)


class DocentAgent:
    """
    Orchestrates the art image processing workflow for the AI Art Team project.
    This class coordinates a series of AI agents to process artwork images, ensuring modifications respect artistic sensibilities such as composition, color balance, historical context, and emotional impact. The workflow includes intelligent cropping, upscaling with ESRGAN for detail preservation, and plaque overlay for metadata addition. Threading is used for concurrent processing to handle batches efficiently, with queues facilitating safe data handoff between stages. Configuration is loaded from YAML to allow easy parameter tuning, and logging is implemented for traceability and debugging.
    """

    def __init__(self, config_path='art_agent_team/config/config.yaml'):
        """
        Initialize the DocentAgent, setting up all necessary components for the AI Art Team's image processing pipeline.
        This constructor ensures that the agent is configured for secure and efficient operation, loading settings from a YAML file and preparing the environment for multi-threaded processing.
        - Config handling: The method loads configuration and sets API keys as environment variables, enhancing security by avoiding hard-coded secrets.
        - Directory creation: Automatically creates required directories to organize files, reducing the risk of runtime errors and improving code reliability.
        - Queue initialization: Establishes communication channels between workflow stages, supporting concurrent execution and preventing bottlenecks in the processing chain.
        - Agent setup: Instantiates or references agents (Research, Upscale, Placard) based on best practices for object-oriented design, allowing for easy extension or modification of the system.
        Design decision: Agents are instantiated here for simplicity, but research agents are created per image to handle potential state changes, balancing performance and memory usage.
        Artistic integrity: Configuration allows for tuning parameters that affect how images are processed, such as upscaling models or placard styles, to accommodate different artistic requirements and genres.
        Best practice adherence: Follows SOLID principles with single responsibility for each agent, and uses type hints and logging for better code maintainability and debugging.
        """
        # Load configuration (empty dict is valid for testing)
        self.config = self._load_config(config_path) or {}

        # Set up file system directories to maintain a clean and organized workflow structure
        self.input_folder = self.config.get('input_folder', 'input')  # Directory for user-provided input images
        self.workspace_folder = self.config.get('workspace_folder', 'workspace')  # Area for temporary or intermediate files during processing
        self.output_folder = self.config.get('output_folder', 'output')  # Location to save final processed images and reports
        os.makedirs(self.input_folder, exist_ok=True)  # Ensure directories exist to handle file I/O gracefully
        os.makedirs(self.workspace_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)

        # Queues removed for sequential processing

        # Set up agents with configuration-driven parameters for modularity and adaptability
        # Import agents needed for the workflow stages
        from art_agent_team.agents.upscale_agent import UpscaleAgent
        from art_agent_team.agents.placard_agent import PlacardAgent
        self.agents = {
            # ResearchAgent is instantiated per image in the processing thread
            'upscale': UpscaleAgent(config=self.config),
            'placard': PlacardAgent(config=self.config),
        }
        # Store vision agent classes loaded at the start
        self.vision_agent_classes = vision_agent_classes
        logger.info(f"Loaded Vision Agent classes: {list(self.vision_agent_classes.keys())}")

        # LLM client initialization removed - DocentAgent no longer interprets prompts directly
        # self.llm_client = None
        # self.llm_model_name = "grok-3-mini-fast-beta"
        # self._init_llm_client() # Removed

        # Initialize markdown log file
        # Determine log directory: Check config for 'log_dir', otherwise derive from config_path directory
        log_dir = self.config.get('log_dir')
        if not log_dir:
            if config_path and os.path.isabs(config_path):
                log_dir = os.path.join(os.path.dirname(config_path), 'logs')
            else:
                log_dir = 'logs'  # Fallback to default relative path
        self.markdown_log_file = os.path.join(log_dir, 'docent_workflow_log.md')
        # Ensure the 'logs' directory exists
        os.makedirs(os.path.dirname(self.markdown_log_file), exist_ok=True)
        # Clear the log file at the start of a new DocentAgent instance for a fresh log per session/run
        # Or, if appending across sessions is desired, this line can be removed or made conditional.
        # For this MVP, let's start fresh each time DocentAgent is initialized.
        with open(self.markdown_log_file, 'w', encoding='utf-8') as f:
            f.write(f"# Docent Workflow Log - Initialized: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        logger.info("DocentAgent initialized. Ready to orchestrate user-selected workflow stages.")

    def _log_to_markdown(self, content: str):
        """Appends content to the markdown workflow log."""
        try:
            with open(self.markdown_log_file, 'a', encoding='utf-8') as f:
                f.write(content + "\n")
        except Exception as e:
            # Use the class logger instance if available, otherwise the module-level logger
            log_instance = getattr(self, 'logger', logger)
            log_instance.error(f"Failed to write to markdown log {self.markdown_log_file}: {e}")

    def _format_log_entry(self, agent_name: str, action: str, details: Optional[List[str]] = None, summary: Optional[str] = None, error: Optional[str] = None, image_context: Optional[str] = None) -> str:
        """Formats a standard entry for the markdown log."""
        timestamp_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_lines = [
            "---"
        ]
        title = f"### {agent_name}: {action}"
        if image_context:
            title += f" (Image: `{image_context}`)"
        log_lines.append(title)
        log_lines.append(f"- **Timestamp:** {timestamp_str}")

        if details:
            for i, detail in enumerate(details):
                prefix = "**Details:**" if i == 0 else "           " # Align subsequent detail lines
                log_lines.append(f"- {prefix} {detail}")
        if summary:
            log_lines.append(f"- **Summary:** {summary}")
        if error:
            log_lines.append(f"- **Error:** {error}")
        return "\n".join(log_lines) + "\n"

    def _load_config(self, config_path='art_agent_team/config/config.yaml'): # Provide default here
        """Loads configuration from YAML, extracts API keys, and sets them as environment variables."""
        config = {}
        
        if config_path is None:
            logger.warning("No config path provided. Using empty configuration.")
            return config

        # Assume config_path is relative to project root if not absolute
        if not os.path.isabs(config_path):
             project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
             full_config_path = os.path.join(project_root, config_path)
        else:
             full_config_path = config_path


        # --- Load Config from YAML ---
        try:
            with open(full_config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                if loaded_config: # Ensure loaded config is not None
                    config.update(loaded_config)
            logger.info(f"Configuration loaded from {full_config_path}")
        except FileNotFoundError:
            logger.error(f"YAML configuration file not found at {full_config_path}. Cannot proceed without configuration.")
            raise # Re-raise as this is critical
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration file {full_config_path}: {e}")
            raise # Re-raise as this is critical
        except Exception as e:
            logger.error(f"Unexpected error loading YAML config file {full_config_path}: {e}")
            raise # Re-raise as this is critical

        # --- Extract API Keys from Config and Set Environment Variables ---
        # This ensures agents can access keys if needed via os.environ
        google_api_key = config.get('google_api_key')
        grok_api_key   = config.get('grok_api_key')
        openai_key = config.get('openai_api_key') # Explicit OpenAI key

        if google_api_key:
            os.environ['GOOGLE_API_KEY'] = google_api_key
            logger.info("Google API Key set in environment.")
        else:
            logger.warning("google_api_key not found in config.yaml.")
            if 'GOOGLE_API_KEY' in os.environ: del os.environ['GOOGLE_API_KEY']

        if grok_api_key:
            os.environ['GROK_API_KEY'] = grok_api_key
            logger.info("Grok API Key set in environment.")
        else:
            logger.warning("grok_api_key not found in config.yaml.")
            if 'GROK_API_KEY' in os.environ: del os.environ['GROK_API_KEY']

        # Set OPENAI_API_KEY - only if explicitly provided in config
        openai_api_key_from_config = config.get('openai_api_key') # openai_key was defined from this above
        if openai_api_key_from_config:
            os.environ['OPENAI_API_KEY'] = openai_api_key_from_config
            logger.info("OpenAI API Key set in environment from 'openai_api_key' in config.")
        else:
            # If not in config, ensure it's not set or remove it if it was set by other means
            if 'OPENAI_API_KEY' in os.environ:
                del os.environ['OPENAI_API_KEY']
                logger.info("'openai_api_key' not found in config. Removed pre-existing OPENAI_API_KEY from environment.")
            else:
                logger.info("'openai_api_key' not found in config. OPENAI_API_KEY is not set.")

        return config

    # Removed _init_llm_client and _call_llm as DocentAgent no longer directly uses LLM for prompts

    def start_workflow(self):
        """
        Orchestrates the image processing workflow based on user selection.
        Allows users to choose which stages (Research, Vision, Upscale, Placard)
        to run sequentially on images found in the input folder.
        """
        logger.info("Starting user-selectable workflow.")
        self._log_to_markdown(f"\n## Workflow Run Started (User Selectable)\n- **Timestamp:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
 
        # Step 0: Display workflow options as a numbered list (reverted to original, non-conversational)
        # No persona prompt or conversational placeholder.

        # Step 1: Get list of images to process
        image_paths = self._get_image_paths(self.input_folder) or []
        num_images = len(image_paths)

        if num_images == 0:
            logger.warning(f"No supported image files found in {self.input_folder}. Workflow ending.")
            print("No images found to process.")
            self._log_to_markdown(self._format_log_entry(agent_name="DocentAgent", action="Workflow Initialization", summary=f"No supported image files found in `{self.input_folder}`. Workflow ending."))
            return
 
        logger.info(f"Found {num_images} images for potential processing.")
        print(f"Found {num_images} images in '{self.input_folder}'.")
        self._log_to_markdown(self._format_log_entry(agent_name="DocentAgent", action="Image Discovery", summary=f"Found {num_images} images in `{self.input_folder}` for potential processing: {', '.join([os.path.basename(p) for p in image_paths])}"))

        # Step 2: Present numbered workflow options to user
        print("\nSelect the workflow stages to run on the image(s):")
        print("  1. Research (analyze and extract metadata)")
        print("  2. Vision (intelligent cropping)")
        print("  3. Upscale (enhance image quality)")
        print("  4. Placard (add museum-style label)")
        print("  5. Full Workflow (all stages in order)")
        print("Enter the numbers of the stages you want to run, separated by commas (e.g., 1,3,4 or 5 for full workflow):")
        user_input = input("Your selection: ").strip()
        selected = set(s.strip() for s in user_input.split(",") if s.strip())

        run_research = "1" in selected or "5" in selected
        run_vision = "2" in selected or "5" in selected
        run_upscale = "3" in selected or "5" in selected
        run_placard = "4" in selected or "5" in selected

        selected_stages = []
        if run_research: selected_stages.append("Research")
        if run_vision: selected_stages.append("Vision")
        if run_upscale: selected_stages.append("Upscale")
        if run_placard: selected_stages.append("Placard")
        logger.info(f"User selected workflow: {' -> '.join(selected_stages)}")
        print(f"Executing workflow: {' -> '.join(selected_stages)}")
        print(f"Processing {num_images} image(s)...")
        self._log_to_markdown(self._format_log_entry(agent_name="DocentAgent", action="Workflow Configuration", details=[f"User selected stages: `{' -> '.join(selected_stages)}`", f"Number of images to process: {num_images}"]))
 
        # Step 4: Process each image sequentially based on selected stages
        processed_count = 0
        skipped_count = 0
        for initial_image_path in image_paths:
            base_filename = os.path.basename(initial_image_path)
            logger.info(f"--- Processing image: {base_filename} ---")
            print(f"\nProcessing: {base_filename}")
            self._log_to_markdown(f"\n---\n### Workflow Started for Image: `{base_filename}`\n- **Timestamp:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n- **Input Image:** `{initial_image_path}`")
 
            # Initialize state for this image
            current_image_path = initial_image_path # Start with the original image
            metadata: Dict[str, Any] = {}
            analysis_results: Optional[Dict[str, Any]] = None # Store vision analysis if run
            genre = 'Default' # Default genre if research is skipped or fails
            image_processed_successfully = True # Flag to track if processing should continue
            cropped_intermediate_path = None # Track cropped path for potential cleanup
            upscaled_intermediate_path = None # Track upscaled path for potential cleanup

            # --- Research Stage ---
            if run_research:
                logger.info(f"[{base_filename}] Running Research stage...")
                self._log_to_markdown(self._format_log_entry(agent_name="DocentAgent", action="Initializing Research", details=[f"Delegating metadata extraction to `ResearchAgent`.", f"Passing image `{os.path.basename(current_image_path)}` for analysis."], image_context=base_filename))
                research_stage_error = None
                research_summary_log = "Research stage skipped or output invalid."
                try:
                    # Instantiate ResearchAgent for each image
                    research_agent = ResearchAgent(self.config, self.vision_agent_classes)
                    research_output = research_agent.research_and_process(current_image_path)
 
                    # --- Display Individual Research Attempts ---
                    print(f"  --- Individual Research Attempts for {base_filename} ---")
                    if research_output and isinstance(research_output, dict) and "individual_results" in research_output:
                        individual_results = research_output.get("individual_results", {})
                        display_order = ["grok_text", "gemini_text", "grok_image", "gemini_image", "openrouter_text", "openrouter_qwen_image", "vertex_vision_subjects"]

                        for source in display_order:
                            if source not in individual_results:
                                print(f"    - {source.replace('_', ' ').title()}: Not Run/Found")
                                continue

                            attempt = individual_results[source]
                            container_success = attempt.get("success", False)
                            result_data = attempt.get("result", {}) # Should always be a dict now

                            # Helper function to display common fields
                            def display_common_fields(res_dict):
                                if not isinstance(res_dict, dict): return
                                conf = DocentAgent._format_confidence_score(res_dict.get('confidence_score'))
                                ground = res_dict.get('grounding_used', 'N/A')
                                primary_subj = res_dict.get('primary_subjects', [])
                                secondary_subj = res_dict.get('secondary_subjects', [])
                                print(f"        Confidence: {conf}, Grounding: {ground}")
                                if primary_subj: print(f"        Primary Subjects: {primary_subj}")
                                if secondary_subj: print(f"        Secondary Subjects: {secondary_subj}")

                            # Special handling for openrouter_text nested results
                            if source == "openrouter_text":
                                print(f"    - OpenRouter Text Search:")
                                if not container_success:
                                     print(f"      Status: Failure (Container)")
                                     if isinstance(result_data, dict) and "error" in result_data:
                                          print(f"      Error: {result_data.get('error', 'Unknown container error')}")
                                     continue

                                llama_mav = result_data.get("llama_maverick", {"error": "Not found"})
                                llama_sco = result_data.get("llama_scout", {"error": "Not found"})

                                for llama_key, llama_res in [("Llama Maverick", llama_mav), ("Llama Scout", llama_sco)]:
                                    if isinstance(llama_res, dict):
                                        llama_success = "error" not in llama_res
                                        llama_success_str = "Success" if llama_success else "Failure"
                                        print(f"      - {llama_key}: {llama_success_str}")
                                        if not llama_success:
                                            print(f"        Error: {llama_res.get('error', 'Unknown error')}")
                                        # Display common fields even on failure (might have defaults)
                                        display_common_fields(llama_res)
                                        if llama_success:
                                            # Display other details only on success
                                            # Safely print other details, truncating long strings
                                            details_list = []
                                            max_detail_len = 200 # Max length for printing unknown values
                                            for k, v in llama_res.items():
                                                if k not in ['error', 'confidence_score', 'grounding_used', 'primary_subjects', 'secondary_subjects']:
                                                    v_str = str(v) # Convert to string
                                                    if len(v_str) > max_detail_len:
                                                        v_display = v_str[:max_detail_len] + "... <truncated>"
                                                    else:
                                                        v_display = v_str
                                                    details_list.append(f"{k}: {v_display}")
                                            if details_list: print(f"        Other Details: {{{', '.join(details_list)}}}")
                                    else:
                                         print(f"      - {llama_key}: Invalid Format ({type(llama_res)})")

                            else: # Handle other sources (grok, gemini, qwen, subjects)
                                success_str = "Success" if container_success else "Failure"
                                print(f"    - {source.replace('_', ' ').title()}: {success_str}")

                                # result_data should always be a dict due to _make_llm_call changes
                                if not isinstance(result_data, dict):
                                     print(f"      Error: Unexpected result format ({type(result_data)})")
                                     continue # Skip if format is wrong despite safeguards

                                error_msg = result_data.get("error")

                                if error_msg: # Check for error key first
                                     print(f"      Error: {error_msg}")
                                     # Still display common fields if they exist (might have defaults)
                                     display_common_fields(result_data)
                                elif not container_success:
                                     # Should ideally have an error message, but handle just in case
                                     print(f"      Error: Unknown container failure (no error message in result)")
                                     display_common_fields(result_data)
                                else: # Successful container and no error key in result_data
                                    # Display common fields
                                    display_common_fields(result_data)
                                    # Display other details
                                    if source == "vertex_vision_subjects":
                                        # Subjects already printed by display_common_fields
                                        pass
                                    else: # Grok/Gemini/Qwen results
                                        # Safely print other details, truncating long strings
                                        details_list = []
                                        max_detail_len = 200 # Max length for printing unknown values
                                        for k, v in result_data.items():
                                            if k not in ['error', 'confidence_score', 'grounding_used', 'primary_subjects', 'secondary_subjects']:
                                                v_str = str(v) # Convert to string
                                                if len(v_str) > max_detail_len:
                                                    v_display = v_str[:max_detail_len] + "... <truncated>"
                                                else:
                                                    v_display = v_str
                                                details_list.append(f"{k}: {v_display}")
                                        if details_list: print(f"      Other Details: {{{', '.join(details_list)}}}")

                    else:
                        print("    No individual results data found.")
                    print(f"  -------------------------------------------------")

                    # --- Process and Display Consolidated Research Results ---
                    metadata = {} # Initialize metadata as empty
                    genre = 'Default' # Initialize genre as default

                    if research_output and isinstance(research_output, dict) and "consolidated_results" in research_output:
                        consolidated_metadata = research_output.get("consolidated_results")

                        # consolidated_metadata should always be a dict now
                        if consolidated_metadata and isinstance(consolidated_metadata, dict):
                            if 'error' not in consolidated_metadata:
                                metadata = consolidated_metadata # Assign valid consolidated data
                                print(f"  --- Consolidated Research Results for {base_filename} ---")
                                # Display standard fields
                                keys_to_display = ['author', 'title', 'date', 'nationality', 'style', 'genre', 'brief_description']
                                for key in keys_to_display:
                                    display_key = key.replace('_', ' ').capitalize()
                                    value = metadata.get(key) # Use .get() which returns None if missing
                                    print(f"    {display_key}: {value if value is not None else 'N/A'}") # Display N/A for None

                                # Display new fields
                                conf = DocentAgent._format_confidence_score(metadata.get('confidence_score'))
                                ground = metadata.get('grounding_used', 'N/A')
                                primary_subj = metadata.get('primary_subjects', [])
                                secondary_subj = metadata.get('secondary_subjects', [])
                                print(f"    Confidence score: {conf}")
                                print(f"    Grounding used: {ground}")
                                print(f"    Primary subjects: {primary_subj if primary_subj else 'N/A'}")
                                print(f"    Secondary subjects: {secondary_subj if secondary_subj else 'N/A'}")

                                print(f"  -------------------------------------------------")
                                genre = metadata.get('genre') or 'Default' # Update genre
                                logger.info(f"[{base_filename}] Research consolidation complete. Genre: {genre}. Confidence: {conf}. Grounding: {ground}. Metadata keys: {list(metadata.keys())}")
                                print(f"  - Research: OK (Consolidated Genre: {genre}, Confidence: {conf})")
                            else: # Error key exists in consolidated_metadata
                                error_msg = consolidated_metadata.get('error', 'Unknown consolidation error')
                                logger.warning(f"[{base_filename}] Research consolidation completed with an error message: {error_msg}")
                                print(f"  - Research: CONSOLIDATION ERROR ({error_msg})")
                                print("  - Consolidated Research Results: Not displayed due to error.")
                                # Keep metadata empty and genre default
                        else: # Handle None or unexpected type (less likely now)
                            logger.warning(f"[{base_filename}] Research consolidation did not return a valid dictionary.")
                            print("  - Research: FAILED (Invalid consolidation output type)")
                            print("  - Consolidated Research Results: No data to display.")
                            # Keep metadata empty and genre default
                    else: # Handle case where research_output structure is missing keys
                         # Log keys instead of the full potentially large/problematic dictionary
                         output_keys = research_output.keys() if isinstance(research_output, dict) else type(research_output)
                         logger.warning(f"[{base_filename}] Research output structure invalid or missing keys. Output keys/type: {output_keys}")
                         print("  - Research: FAILED (Invalid output structure)")
                         # Keep metadata empty and genre default
                         research_summary_log = "Research FAILED (Invalid output structure)."
                         research_stage_error = "Invalid research output structure."

                except Exception as research_err:
                    logger.error(f"[{base_filename}] Research stage failed unexpectedly: {research_err}", exc_info=True)
                    print(f"  - Research: FAILED ({research_err})")
                    research_stage_error = str(research_err)
                    research_summary_log = f"Research FAILED: {research_err}"
                    # Ensure metadata is empty and genre is default on unexpected failure
                    metadata = {}
                    genre = 'Default'
                finally:
                    if 'error' not in metadata and metadata: # Log success if no error key and metadata exists
                         research_summary_log = f"Completed. Genre: {genre}, Confidence: {DocentAgent._format_confidence_score(metadata.get('confidence_score'))}, Grounding: {metadata.get('grounding_used', 'N/A')}, Metadata keys: {list(metadata.keys())}"
                    elif 'error' in metadata: # Log error if error key exists
                         research_summary_log = f"Consolidation Error: {metadata.get('error')}"
                    # If metadata is empty and no specific error logged above, it means a more general failure or skipped.
                    
                    self._log_to_markdown(self._format_log_entry(agent_name="ResearchAgent", action="Findings", summary=research_summary_log, error=research_stage_error, image_context=base_filename))

            # --- Vision Stage ---
            if run_vision:
                logger.info(f"[{base_filename}] Running Vision stage (Genre: {genre})...")
                vision_agent_name_log = "VisionAgent (Unknown)"
                vision_details_log = [f"Input image: `{os.path.basename(current_image_path)}`", f"Determined Genre for Vision: `{genre}`"]
                if metadata:
                    vision_details_log.append(f"Metadata (keys): {list(metadata.keys())}")
                
                vision_summary_log = "Vision stage skipped."
                vision_error_log = None

                # Determine VisionAgent class
                vision_agent_class = self.vision_agent_classes.get(genre)
                if not vision_agent_class:
                    logger.warning(f"[{base_filename}] No specific VisionAgent for genre '{genre}'. Using Default.")
                    vision_details_log.append(f"Decision: No specific VisionAgent for genre `{genre}`. Using Default.")
                    vision_agent_class = self.vision_agent_classes.get('Default')
 
                if not vision_agent_class:
                    logger.error(f"[{base_filename}] Default VisionAgent not found. Skipping Vision stage.")
                    print(f"  - Vision: SKIPPED (Default agent not found)")
                    vision_summary_log = "SKIPPED (Default VisionAgent not found)"
                    vision_error_log = "Default VisionAgent not found."
                    self._log_to_markdown(self._format_log_entry(agent_name="DocentAgent", action="Initializing Vision", details=vision_details_log, summary=vision_summary_log, error=vision_error_log, image_context=base_filename))
                else:
                    vision_agent_name_log = vision_agent_class.__name__
                    vision_details_log.append(f"Decision: Selected `VisionAgent`: `{vision_agent_name_log}`.")
                    self._log_to_markdown(self._format_log_entry(agent_name="DocentAgent", action="Initializing Vision", details=vision_details_log, image_context=base_filename))
                    
                    try:
                        vision_agent = vision_agent_class(self.config)
                        logger.debug(f"[{base_filename}] Instantiated Vision Agent: {vision_agent.__class__.__name__}")
                        
                        analysis_log_summary = "Not performed or not applicable."
                        # Optional: Analyze image first
                        if hasattr(vision_agent, 'analyze_image'):
                            try:
                                # Pass metadata if available from research stage
                                analysis_results = vision_agent.analyze_image(current_image_path, metadata) # This can be a dict or None
                                logger.info(f"[{base_filename}] Vision analysis complete.")
                                print(f"  - Vision Analysis: OK")
                                analysis_log_summary = f"Analysis complete. Result keys: {list(analysis_results.keys()) if analysis_results else 'None'}"
                            except Exception as analyze_err:
                                logger.error(f"[{base_filename}] Vision analysis failed: {analyze_err}", exc_info=True)
                                print(f"  - Vision Analysis: FAILED ({analyze_err})")
                                analysis_results = None # Ensure it's None on failure
                                analysis_log_summary = f"Analysis FAILED: {analyze_err}"
                                vision_error_log = vision_error_log + f"; Analysis Error: {analyze_err}" if vision_error_log else f"Analysis Error: {analyze_err}"
                        
                        vision_details_log_for_action = [f"Analysis summary: {analysis_log_summary}"]
                        if analysis_results:
                             vision_details_log_for_action.append(f"Analysis results passed to crop: {analysis_results}")
                        else:
                             vision_details_log_for_action.append("No analysis results passed to crop.")

                        # Define cropped output path
                        cropped_filename = f"{os.path.splitext(base_filename)[0]}_cropped{os.path.splitext(base_filename)[1]}"
                        cropped_output_path = os.path.join(self.workspace_folder, cropped_filename)
                        vision_details_log_for_action.append(f"Target crop output path: `{cropped_output_path}`")
 
                        # Attempt cropping
                        cropped_path = None
                        if hasattr(vision_agent, 'copy_and_crop_image'):
                            logger.debug(f"[{base_filename}] Attempting crop with copy_and_crop_image...")
                            cropped_path = vision_agent.copy_and_crop_image(
                                current_image_path,
                                cropped_output_path,
                                analysis_results # Pass analysis results (can be None)
                            )
                        # Add fallback if needed, e.g., crop_to_aspect_ratio
                        # elif hasattr(vision_agent, 'crop_to_aspect_ratio'): ...
                        else:
                            logger.warning(f"[{base_filename}] Vision agent {vision_agent.__class__.__name__} has no 'copy_and_crop_image' method. Skipping crop.")
                            print(f"  - Vision Crop: SKIPPED (No suitable method)")

                        # Update current_image_path if cropping was successful
                        if cropped_path and os.path.exists(cropped_path):
                            if cropped_path != current_image_path:
                                logger.info(f"[{base_filename}] Vision crop successful. New path: {cropped_path}")
                                print(f"  - Vision Crop: OK -> {os.path.basename(cropped_path)}")
                                current_image_path = cropped_path
                                cropped_intermediate_path = cropped_path # Store for potential cleanup
                            else:
                                logger.info(f"[{base_filename}] Vision agent did not perform crop. Using previous path.")
                                print(f"  - Vision Crop: OK (No change)")
                        elif cropped_path is None: # Case where cropping method was skipped
                            pass # Path remains unchanged, already logged/printed
                        else: # Case where cropping method ran but failed (returned invalid path)
                            logger.error(f"[{base_filename}] Vision crop failed or output path invalid: {cropped_path}. Using previous path for subsequent stages.")
                            print(f"  - Vision Crop: FAILED (Output invalid). Continuing with previous image.")
                            # Keep current_image_path as it was before this stage
                            vision_summary_log = f"Crop FAILED (Output invalid). Path: {cropped_path}. Using previous image: `{os.path.basename(current_image_path)}`"
                            vision_error_log = vision_error_log + "; Crop output invalid" if vision_error_log else "Crop output invalid"
                        
                        self._log_to_markdown(self._format_log_entry(agent_name=vision_agent_name_log, action="Image Cropping", details=vision_details_log_for_action, summary=vision_summary_log, error=vision_error_log, image_context=base_filename))

                    except Exception as vision_err:
                        logger.error(f"[{base_filename}] Vision stage failed: {vision_err}", exc_info=True)
                        print(f"  - Vision: FAILED ({vision_err})")
                        vision_summary_log = f"Vision stage FAILED: {vision_err}. Using previous image: `{os.path.basename(current_image_path)}`"
                        vision_error_log = str(vision_err)
                        self._log_to_markdown(self._format_log_entry(agent_name=vision_agent_name_log, action="Image Cropping", summary=vision_summary_log, error=vision_error_log, image_context=base_filename))
                        # Keep current_image_path as it was before this stage
 
            # --- Upscale Stage ---
            if run_upscale:
                logger.info(f"[{base_filename}] Running Upscale stage...")
                upscale_summary_log = "Upscale stage skipped."
                upscale_error_log = None
                upscale_details_log = [f"Input image for upscale: `{os.path.basename(current_image_path)}`"]
                self._log_to_markdown(self._format_log_entry(agent_name="DocentAgent", action="Initializing Upscale", details=[f"Delegating to `UpscaleAgent`.", f"Passing image `{os.path.basename(current_image_path)}`."], image_context=base_filename))

                # Check if input path exists before attempting upscale
                if not os.path.exists(current_image_path):
                     logger.error(f"[{base_filename}] Input image for Upscale not found: {current_image_path}. Skipping Upscale.")
                     print(f"  - Upscale: SKIPPED (Input not found)")
                     upscale_summary_log = f"SKIPPED (Input not found: {current_image_path})"
                     upscale_error_log = "Input image for upscale not found."
                     self._log_to_markdown(self._format_log_entry(agent_name="UpscaleAgent", action="Image Upscaling", details=upscale_details_log, summary=upscale_summary_log, error=upscale_error_log, image_context=base_filename))
                else:
                    try:
                        upscale_agent = self.agents['upscale']
                        # Define upscale output path
                        name, ext = os.path.splitext(os.path.basename(current_image_path))
                        # Avoid double suffixes if input was already _cropped
                        base_name_for_upscale = name.replace('_cropped', '')
                        upscaled_filename = f"{base_name_for_upscale}_upscaled{ext}"
                        upscale_output_path = os.path.join(self.workspace_folder, upscaled_filename)
                        upscale_details_log.append(f"Target upscale output path: `{upscale_output_path}`")
 
                        upscaled_path = upscale_agent.upscale_image(current_image_path, upscale_output_path)
 
                        if upscaled_path and os.path.exists(upscaled_path):
                            logger.info(f"[{base_filename}] Upscaling successful. New path: {upscaled_path}")
                            print(f"  - Upscale: OK -> {os.path.basename(upscaled_path)}")
                            current_image_path = upscaled_path
                            upscaled_intermediate_path = upscaled_path # Store for potential cleanup
                            upscale_summary_log = f"Upscaling successful. Output: `{os.path.basename(upscaled_path)}`"
                        else:
                            logger.error(f"[{base_filename}] Upscaling failed or output path invalid. Using previous path for subsequent stages.")
                            print(f"  - Upscale: FAILED (Output invalid). Continuing with previous image.")
                            upscale_summary_log = f"Upscaling FAILED (Output invalid: {upscaled_path}). Using previous image: `{os.path.basename(current_image_path)}`"
                            upscale_error_log = "Upscale output invalid or not found."
                            # Keep current_image_path as it was
                        self._log_to_markdown(self._format_log_entry(agent_name="UpscaleAgent", action="Image Upscaling", details=upscale_details_log, summary=upscale_summary_log, error=upscale_error_log, image_context=base_filename))
 
                    except Exception as upscale_err:
                        logger.error(f"[{base_filename}] Upscale stage failed: {upscale_err}", exc_info=True)
                        print(f"  - Upscale: FAILED ({upscale_err})")
                        upscale_summary_log = f"Upscale stage FAILED: {upscale_err}. Using previous image: `{os.path.basename(current_image_path)}`"
                        upscale_error_log = str(upscale_err)
                        self._log_to_markdown(self._format_log_entry(agent_name="UpscaleAgent", action="Image Upscaling", details=upscale_details_log, summary=upscale_summary_log, error=upscale_error_log, image_context=base_filename))
                        # Keep current_image_path as it was
 
            # --- Placard Stage ---
            if run_placard:
                logger.info(f"[{base_filename}] Running Placard stage...")
                placard_summary_log = "Placard stage skipped."
                placard_error_log = None
                placard_details_log = [f"Input image for placard: `{os.path.basename(current_image_path)}`"]
                if metadata:
                    placard_details_log.append(f"Metadata provided (keys): {list(metadata.keys())}")
                else:
                    placard_details_log.append("No metadata provided (Research skipped or failed).")
                self._log_to_markdown(self._format_log_entry(agent_name="DocentAgent", action="Initializing Placard", details=[f"Delegating to `PlacardAgent`."] + placard_details_log, image_context=base_filename))

                # Check if input path exists before attempting placard
                if not os.path.exists(current_image_path):
                     logger.error(f"[{base_filename}] Input image for Placard not found: {current_image_path}. Skipping Placard.")
                     print(f"  - Placard: SKIPPED (Input not found)")
                     placard_summary_log = f"SKIPPED (Input not found: {current_image_path})"
                     placard_error_log = "Input image for placard not found."
                     self._log_to_markdown(self._format_log_entry(agent_name="PlacardAgent", action="Placard Creation", details=placard_details_log, summary=placard_summary_log, error=placard_error_log, image_context=base_filename))
                else:
                    try:
                        placard_agent = self.agents['placard']
                        # Define final output path
                        name, ext = os.path.splitext(os.path.basename(current_image_path))
                        # Clean up intermediate suffixes for final name
                        original_base_name = name.replace('_upscaled', '').replace('_cropped', '')
                        final_filename = f"{original_base_name}_final{ext}"
                        final_output_path = os.path.join(self.output_folder, final_filename)
                        placard_details_log.append(f"Target final output path: `{final_output_path}`")
 
                        # Pass metadata (might be empty if research failed or was skipped)
                        placarded_path = placard_agent.add_plaque(current_image_path, final_output_path, metadata)
 
                        if placarded_path and os.path.exists(placarded_path):
                            logger.info(f"[{base_filename}] Placard addition successful. Final output: {placarded_path}")
                            print(f"  - Placard: OK -> {os.path.basename(placarded_path)}")
                            current_image_path = placarded_path # Update path to final output
                            placard_summary_log = f"Placard addition successful. Final output: `{os.path.basename(placarded_path)}`"
 
                            # --- Optional Cleanup ---
                            cleanup_enabled = self.config.get('cleanup_workspace', True)
                            if cleanup_enabled:
                                files_to_remove = []
                                # Add intermediate files if they exist and are different from final output
                                if cropped_intermediate_path and os.path.exists(cropped_intermediate_path) and cropped_intermediate_path != final_output_path:
                                    files_to_remove.append(cropped_intermediate_path)
                                if upscaled_intermediate_path and os.path.exists(upscaled_intermediate_path) and upscaled_intermediate_path != final_output_path:
                                     files_to_remove.append(upscaled_intermediate_path)
                                # Also remove the direct input to placard if it wasn't the final output (e.g. if upscale ran but placard saved elsewhere)
                                input_to_placard = upscaled_intermediate_path or cropped_intermediate_path or initial_image_path
                                if input_to_placard != final_output_path and input_to_placard not in files_to_remove and os.path.exists(input_to_placard) and self.workspace_folder in input_to_placard:
                                     files_to_remove.append(input_to_placard)


                                for file_to_remove in set(files_to_remove): # Use set to avoid duplicates
                                    try:
                                        os.remove(file_to_remove)
                                        logger.debug(f"[{base_filename}] Removed intermediate file: {file_to_remove}")
                                    except OSError as remove_err:
                                        logger.warning(f"[{base_filename}] Failed to remove intermediate file '{file_to_remove}': {remove_err}")
                            else:
                                logger.debug(f"[{base_filename}] Workspace cleanup disabled.")

                        else:
                            logger.error(f"[{base_filename}] Placard addition failed or output path invalid.")
                            print(f"  - Placard: FAILED (Output invalid)")
                            placard_summary_log = f"Placard addition FAILED (Output invalid: {placarded_path})."
                            placard_error_log = "Placard output invalid or not found."
                            # current_image_path remains the input to this failed stage
                        self._log_to_markdown(self._format_log_entry(agent_name="PlacardAgent", action="Placard Creation", details=placard_details_log, summary=placard_summary_log, error=placard_error_log, image_context=base_filename))
 
                    except Exception as placard_err:
                        logger.error(f"[{base_filename}] Placard stage failed: {placard_err}", exc_info=True)
                        print(f"  - Placard: FAILED ({placard_err})")
                        placard_summary_log = f"Placard stage FAILED: {placard_err}."
                        placard_error_log = str(placard_err)
                        self._log_to_markdown(self._format_log_entry(agent_name="PlacardAgent", action="Placard Creation", details=placard_details_log, summary=placard_summary_log, error=placard_error_log, image_context=base_filename))
                        # current_image_path remains the input to this failed stage
 
            # --- Image Processing Summary ---
            if image_processed_successfully: # Check overall success flag if implemented, otherwise assume processed
                processed_count += 1
                logger.info(f"--- Finished processing image: {base_filename} ---")
                self._log_to_markdown(self._format_log_entry(agent_name="DocentAgent", action="Image Processing Complete", summary=f"Successfully processed. Final/current path: `{current_image_path}`", image_context=base_filename))
            else:
                skipped_count += 1
                logger.warning(f"--- Skipped or failed processing image: {base_filename} ---")
                self._log_to_markdown(self._format_log_entry(agent_name="DocentAgent", action="Image Processing Incomplete", summary="Image processing did not complete successfully or was skipped.", error="One or more stages failed or image_processed_successfully flag was false.", image_context=base_filename))
 
        # Step 5: Final Workflow Summary
        print("\n--- Workflow Summary ---")
        print(f"Selected stages: {' -> '.join(selected_stages)}")
        print(f"Total images found: {num_images}")
        print(f"Successfully processed: {processed_count}")
        print(f"Skipped/Failed: {skipped_count}")
        print(f"Final outputs (if Placard ran) are in: '{self.output_folder}'")
        print(f"Intermediate files (if not cleaned) are in: '{self.workspace_folder}'")
        print("--------------------------")
        logger.info(f"DocentAgent workflow finished. Processed: {processed_count}, Skipped/Failed: {skipped_count}.")
        self._log_to_markdown(f"\n## Workflow Run Ended (User Selectable)\n- **Timestamp:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n- **Summary:**\n  - Selected Stages: `{' -> '.join(selected_stages)}`\n  - Total Images Found: {num_images}\n  - Successfully Processed: {processed_count}\n  - Skipped/Failed: {skipped_count}\n  - Output Folder: `{self.output_folder}`\n  - Workspace Folder: `{self.workspace_folder}`\n---")


    def _get_image_paths(self, folder_path) -> List[str]:
        """Finds supported image files in the folder and returns a list of their paths."""
        # Ensure logger is accessible, assuming it's defined as self.logger or globally as logger
        log = getattr(self, 'logger', logging.getLogger(__name__)) # Use self.logger if available, else module logger

        log.info(f"Scanning folder for images: {folder_path}")
        image_paths = []
        supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')

        try:
            if not os.path.isdir(folder_path):
                log.error(f"Input folder does not exist or is not a directory: {folder_path}")
                print(f"Error: Input folder '{folder_path}' not found.")
                return [] # Return empty list on critical error

            for filename in os.listdir(folder_path):
                log.debug(f"Processing entry: {filename}") # Log every entry

                if filename.startswith('.'):
                    log.debug(f"Skipping hidden entry: {filename}")
                    continue

                file_path = os.path.join(folder_path, filename)

                # Check if it's a file *before* checking extension
                if not os.path.isfile(file_path):
                    log.debug(f"Skipping non-file entry: {filename}")
                    continue

                # Check if extension is supported
                if not filename.lower().endswith(supported_extensions):
                    log.debug(f"Skipping unsupported file extension: {filename}")
                    continue

                # If all checks pass, add the file path
                image_paths.append(file_path)
                log.debug(f"Added supported image file: {filename}")

            # Report final count after loop
            if image_paths:
                log.info(f"Found {len(image_paths)} supported images in {folder_path}.")
            else:
                log.warning(f"No supported images found in {folder_path}.")
                # Keep the print statement for user visibility if desired
                print(f"Warning: No supported images (.png, .jpg, .jpeg, .bmp, .gif, .tiff, .webp) found in '{folder_path}'.")

        except PermissionError:
            log.error(f"Permission denied when trying to scan folder: {folder_path}")
            print(f"Error: Permission denied for folder '{folder_path}'.")
            return [] # Return empty list on permission error
        except Exception as e:
            log.exception(f"An unexpected error occurred scanning input folder {folder_path}: {e}") # Use log.exception to include traceback
            print(f"Error scanning input folder: {e}")
            return [] # Return empty list on other exceptions

        # Return the list (potentially empty) after successful scan
        return image_paths

    def _think_with_grok(self, prompt: str):
        """
        Internal reasoning using Grok LLM (grok-3-mini-fast-high-beta).
        """
        import os
        grok_api_key = os.environ.get("GROK_API_KEY")
        if not grok_api_key:
            logger.warning("[DocentAgent] Grok API key not found. Skipping internal reasoning.")
            return None
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=grok_api_key,
                base_url="https://api.x.ai/v1",
            )
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="grok-3-mini-fast-high-beta",
                temperature=0.2,
                max_tokens=256
            )
            reasoning = response.choices[0].message.content
            logger.info(f"[DocentAgent] Grok thinking output: {reasoning}")
            return reasoning
        except Exception as e:
            logger.error(f"[DocentAgent] Grok thinking failed: {e}")
            return None

# --- Integrated Thinking Step for Agent Refinement ---
    def _refine_agent_input_with_grok(
        self,
        target_agent_name: str,
        current_image_path: str,
        previous_stage_results: Optional[dict],
        user_context_input: str
    ) -> Optional[dict]:
        """
        Calls _think_with_grok to refine parameters/instructions for the next agent stage.
        Returns a validated dictionary of parameters for the target agent, or empty dict if none.
        """
        # Define expected keys for each agent
        expected_keys = {
            "Vision": ["focus_areas", "crop_preference"],
            "Upscale": ["upscale_model", "preserve_texture_level"],
            "Placard": ["placard_style", "additional_notes"]
        }
        keys = expected_keys.get(target_agent_name, [])

        # Summarize previous results for prompt
        prev_summary = ""
        if previous_stage_results:
            try:
                prev_summary = json.dumps(previous_stage_results, ensure_ascii=False)
            except Exception:
                prev_summary = str(previous_stage_results)
        else:
            prev_summary = "None"

        prompt = (
            f"Generate refined parameters/instructions for the upcoming {target_agent_name} stage.\n"
            f"Previous stage results: {prev_summary}\n"
            f"User preference: '{user_context_input}'.\n"
            f"The target image is {current_image_path}.\n"
            f"Respond ONLY with a JSON dictionary containing keys relevant for the {target_agent_name} agent, "
            f"such as {keys}. If no specific refinement is needed, return an empty JSON object {{}}."
        )

        response = self._think_with_grok(prompt)
        if not response:
            logger.warning(f"[DocentAgent] No response from Grok for {target_agent_name} thinking step.")
            return {}

        # Try to extract JSON from response
        import re
        json_str = None
        # Try to find a JSON object in the response
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            json_str = match.group(0)
        else:
            json_str = response.strip()

        try:
            parsed = json.loads(json_str)
            if not isinstance(parsed, dict):
                logger.warning(f"[DocentAgent] Grok output for {target_agent_name} is not a dict: {parsed}")
                return {}
        except Exception as e:
            logger.warning(f"[DocentAgent] Failed to parse Grok output for {target_agent_name}: {e}. Raw output: {response}")
            return {}

        # Validate keys
        if keys:
            valid = all((k in keys for k in parsed.keys()))
            if not valid:
                logger.warning(f"[DocentAgent] Grok output for {target_agent_name} contains unexpected keys: {parsed.keys()}")
                # Optionally filter to only expected keys
                parsed = {k: v for k, v in parsed.items() if k in keys}
        return parsed

    @staticmethod
    def _format_confidence_score(score: Any) -> str:
        """Formats a confidence score for display."""
        if score is None: return "N/A"
        try:
            return f"{float(score)*100:.1f}%"
        except (ValueError, TypeError):
            return "Invalid"
 
    def handle_request(self, task_description: str, image_path: str) -> Dict[str, Any]:
        """
        Handles a specific request to process an image through the full workflow.
        This is a non-interactive method intended for programmatic calls (e.g., from tests or other agents).

        Args:
            task_description (str): A description of the task (e.g., "Analyze and crop the image.").
                                    Currently used for logging.
            image_path (str): The path to the input image.

        Returns:
            Dict[str, Any]: A dictionary containing the path to the final processed image
                            and any collected metadata.
                            Example: {"final_image_path": "path/to/output.jpg", "metadata": {...}}
                            Returns an error structure if processing fails.
        """
        base_filename = os.path.basename(image_path)
        logger.info(f"--- Handling request for image: {base_filename} ---")
        logger.info(f"Task description: {task_description}")

        self._log_to_markdown(f"\n## Workflow Run Started (Programmatic Request: {task_description})\n- **Timestamp:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n- **Input Image:** `{image_path}`")

        if not os.path.exists(image_path):
            logger.error(f"Input image not found: {image_path}")
            error_msg = f"Input image not found: {image_path}"
            self._log_to_markdown(self._format_log_entry(agent_name="DocentAgent", action="Workflow Initialization", error=error_msg, image_context=base_filename))
            return {"error": error_msg, "final_image_path": None, "metadata": {}}
 
        # Simulate full workflow selection
        run_research = True
        run_vision = True
        run_upscale = True
        run_placard = True

        current_image_path = image_path
        metadata: Dict[str, Any] = {}
        analysis_results: Optional[Dict[str, Any]] = None
        genre = 'Default'
        cropped_intermediate_path = None
        upscaled_intermediate_path = None
        
        # Track called agents for User Instruction 4
        called_agents_log = []


        # --- Research Stage ---
        if run_research:
            called_agents_log.append("ResearchAgent")
            logger.info(f"[{base_filename}] Running Research stage...")
            self._log_to_markdown(self._format_log_entry(agent_name="DocentAgent", action="Initializing Research", details=[f"Delegating metadata extraction to `ResearchAgent`.", f"Passing image `{os.path.basename(current_image_path)}` for analysis."], image_context=base_filename))
            research_stage_error_log = None
            research_summary_md_log = "Research stage did not complete as expected."
            try:
                research_agent = ResearchAgent(self.config, self.vision_agent_classes)
                research_output = research_agent.research_and_process(current_image_path)
                if research_output and isinstance(research_output, dict) and "consolidated_results" in research_output:
                    consolidated_metadata = research_output.get("consolidated_results")
                    if consolidated_metadata and isinstance(consolidated_metadata, dict) and 'error' not in consolidated_metadata:
                        metadata = consolidated_metadata
                        genre = metadata.get('genre') or 'Default'
                        logger.info(f"[{base_filename}] Research complete. Genre: {genre}. Metadata collected.")
                        research_summary_md_log = f"Completed. Genre: {genre}, Confidence: {DocentAgent._format_confidence_score(metadata.get('confidence_score'))}, Grounding: {metadata.get('grounding_used', 'N/A')}, Metadata keys: {list(metadata.keys())}"
                    else:
                        err_msg = consolidated_metadata.get('error') if isinstance(consolidated_metadata, dict) else 'Unknown consolidation error'
                        logger.warning(f"[{base_filename}] Research consolidation error: {err_msg}")
                        research_stage_error_log = f"Consolidation error: {err_msg}"
                        research_summary_md_log = f"Research consolidation error: {err_msg}"
                else:
                    logger.warning(f"[{base_filename}] Research output invalid or missing consolidated_results.")
                    research_stage_error_log = "Research output invalid or missing consolidated_results."
                    research_summary_md_log = "Research output invalid."
            except Exception as research_err:
                logger.error(f"[{base_filename}] Research stage failed: {research_err}", exc_info=True)
                metadata = {"error": f"Research stage failed: {research_err}"} # Keep this for return value
                research_stage_error_log = str(research_err)
                research_summary_md_log = f"Research FAILED: {research_err}"
            finally:
                self._log_to_markdown(self._format_log_entry(agent_name="ResearchAgent", action="Findings", summary=research_summary_md_log, error=research_stage_error_log, image_context=base_filename))
 
        # --- Vision Stage ---
        if run_vision:
            vision_agent_to_log = "VisionAgent (Unknown)"
            logger.info(f"[{base_filename}] Running Vision stage (Genre: {genre})...")
            vision_details_md_log = [f"Input image: `{os.path.basename(current_image_path)}`", f"Determined Genre for Vision: `{genre}`"]
            if metadata and 'error' not in metadata : vision_details_md_log.append(f"Metadata (keys): {list(metadata.keys())}")

            vision_summary_md_log = "Vision stage skipped."
            vision_error_md_log = None

            vision_agent_class = self.vision_agent_classes.get(genre)
            if not vision_agent_class:
                logger.warning(f"[{base_filename}] No specific VisionAgent for genre '{genre}'. Using Default.")
                vision_details_md_log.append(f"Decision: No specific VisionAgent for genre `{genre}`. Using Default.")
                vision_agent_class = self.vision_agent_classes.get('Default')
 
            if vision_agent_class:
                vision_agent_to_log = vision_agent_class.__name__
                vision_details_md_log.append(f"Decision: Selected `VisionAgent`: `{vision_agent_to_log}`.")
                self._log_to_markdown(self._format_log_entry(agent_name="DocentAgent", action="Initializing Vision", details=vision_details_md_log, image_context=base_filename))
                
                try:
                    vision_agent = vision_agent_class(self.config)
                    called_agents_log.append(vision_agent_to_log) # Log agent name for return value
                    logger.debug(f"[{base_filename}] Instantiated Vision Agent: {vision_agent_to_log}")
                    
                    analysis_log_summary = "Not performed or not applicable."
                    if hasattr(vision_agent, 'analyze_image'):
                        try:
                            analysis_results = vision_agent.analyze_image(current_image_path, metadata)
                            analysis_log_summary = f"Analysis complete. Result keys: {list(analysis_results.keys()) if analysis_results else 'None'}"
                            self._log_to_markdown(self._format_log_entry(agent_name=vision_agent_to_log, action="Image Analysis", summary=analysis_log_summary, image_context=base_filename))
                        except Exception as analyze_err:
                            analysis_log_summary = f"Analysis FAILED: {analyze_err}"
                            vision_error_md_log = f"Analysis Error: {analyze_err}"
                            self._log_to_markdown(self._format_log_entry(agent_name=vision_agent_to_log, action="Image Analysis", summary=analysis_log_summary, error=str(analyze_err), image_context=base_filename))
                    
                    cropped_filename = f"{os.path.splitext(base_filename)[0]}_cropped{os.path.splitext(base_filename)[1]}"
                    cropped_output_path = os.path.join(self.workspace_folder, cropped_filename)
                    crop_details_md = [f"Target crop output path: `{cropped_output_path}`"]
                    if analysis_results: crop_details_md.append(f"Analysis results passed to crop: {analysis_results}")

                    cropped_path_result = None
                    if hasattr(vision_agent, 'copy_and_crop_image'):
                        cropped_path_result = vision_agent.copy_and_crop_image(current_image_path, cropped_output_path, analysis_results)
                    
                    if cropped_path_result and os.path.exists(cropped_path_result) and cropped_path_result != current_image_path:
                        current_image_path = cropped_path_result
                        cropped_intermediate_path = cropped_path_result
                        logger.info(f"[{base_filename}] Vision crop successful. New path: {current_image_path}")
                        vision_summary_md_log = f"Crop successful. Output: `{os.path.basename(current_image_path)}`"
                    elif cropped_path_result == current_image_path:
                         logger.info(f"[{base_filename}] Vision agent did not perform crop. Using previous path.")
                         vision_summary_md_log = f"Crop not performed. Image unchanged: `{os.path.basename(current_image_path)}`"
                    else: # Crop failed or skipped by agent logic
                        logger.warning(f"[{base_filename}] Vision crop failed or output path invalid. Using previous path.")
                        vision_summary_md_log = f"Crop FAILED or skipped by agent. Output path: {cropped_path_result}. Using previous image: `{os.path.basename(current_image_path)}`"
                        if cropped_path_result is not None : vision_error_md_log = (vision_error_md_log + "; " if vision_error_md_log else "") + "Crop output invalid or not found."
                    
                    self._log_to_markdown(self._format_log_entry(agent_name=vision_agent_to_log, action="Image Cropping", details=crop_details_md, summary=vision_summary_md_log, error=vision_error_md_log, image_context=base_filename))

                except Exception as vision_err:
                    logger.error(f"[{base_filename}] Vision stage failed for {vision_agent_to_log}: {vision_err}", exc_info=True)
                    vision_summary_md_log = f"Vision stage FAILED: {vision_err}. Using previous image: `{os.path.basename(current_image_path)}`"
                    vision_error_md_log = str(vision_err)
                    self._log_to_markdown(self._format_log_entry(agent_name=vision_agent_to_log, action="Image Cropping", summary=vision_summary_md_log, error=vision_error_md_log, image_context=base_filename))
            else: # Default VisionAgent not found
                logger.error(f"[{base_filename}] Default VisionAgent not found. Skipping Vision stage.")
                called_agents_log.append(f"VisionAgent (Error: Default not found for genre {genre})") # Log for return value
                vision_summary_md_log = "SKIPPED (Default VisionAgent not found)"
                vision_error_md_log = "Default VisionAgent not found."
                self._log_to_markdown(self._format_log_entry(agent_name="DocentAgent", action="Initializing Vision", details=vision_details_md_log, summary=vision_summary_md_log, error=vision_error_md_log, image_context=base_filename))
 
        # --- Upscale Stage ---
        if run_upscale:
            called_agents_log.append("UpscaleAgent")
            logger.info(f"[{base_filename}] Running Upscale stage...")
            upscale_details_md_log = [f"Input image for upscale: `{os.path.basename(current_image_path)}`"]
            self._log_to_markdown(self._format_log_entry(agent_name="DocentAgent", action="Initializing Upscale", details=[f"Delegating to `UpscaleAgent`."] + upscale_details_md_log, image_context=base_filename))
            upscale_summary_md_log = "Upscale stage skipped."
            upscale_error_md_log = None

            if not os.path.exists(current_image_path):
                logger.error(f"[{base_filename}] Input image for Upscale not found: {current_image_path}. Skipping.")
                upscale_summary_md_log = f"SKIPPED (Input not found: {current_image_path})"
                upscale_error_md_log = "Input image for upscale not found."
                self._log_to_markdown(self._format_log_entry(agent_name="UpscaleAgent", action="Image Upscaling", details=upscale_details_md_log, summary=upscale_summary_md_log, error=upscale_error_md_log, image_context=base_filename))
            else:
                try:
                    upscale_agent = self.agents['upscale']
                    name, ext = os.path.splitext(os.path.basename(current_image_path))
                    base_name_for_upscale = name.replace('_cropped', '')
                    upscaled_filename = f"{base_name_for_upscale}_upscaled{ext}"
                    upscale_output_path = os.path.join(self.workspace_folder, upscaled_filename)
                    upscale_details_md_log.append(f"Target upscale output path: `{upscale_output_path}`")
                    
                    upscaled_path_result = upscale_agent.upscale_image(current_image_path, upscale_output_path)
                    if upscaled_path_result and os.path.exists(upscaled_path_result):
                        current_image_path = upscaled_path_result
                        upscaled_intermediate_path = upscaled_path_result
                        logger.info(f"[{base_filename}] Upscaling successful. New path: {current_image_path}")
                        upscale_summary_md_log = f"Upscaling successful. Output: `{os.path.basename(current_image_path)}`"
                    else:
                        logger.warning(f"[{base_filename}] Upscaling failed or output path invalid. Using previous path.")
                        upscale_summary_md_log = f"Upscaling FAILED (Output invalid: {upscaled_path_result}). Using previous image: `{os.path.basename(current_image_path)}`"
                        upscale_error_md_log = "Upscale output invalid or not found."
                    self._log_to_markdown(self._format_log_entry(agent_name="UpscaleAgent", action="Image Upscaling", details=upscale_details_md_log, summary=upscale_summary_md_log, error=upscale_error_md_log, image_context=base_filename))
                except Exception as upscale_err:
                    logger.error(f"[{base_filename}] Upscale stage failed: {upscale_err}", exc_info=True)
                    upscale_summary_md_log = f"Upscale stage FAILED: {upscale_err}. Using previous image: `{os.path.basename(current_image_path)}`"
                    upscale_error_md_log = str(upscale_err)
                    self._log_to_markdown(self._format_log_entry(agent_name="UpscaleAgent", action="Image Upscaling", details=upscale_details_md_log, summary=upscale_summary_md_log, error=upscale_error_md_log, image_context=base_filename))
 
        # --- Placard Stage ---
        if run_placard:
            called_agents_log.append("PlacardAgent")
            logger.info(f"[{base_filename}] Running Placard stage...")
            placard_details_md_log = [f"Input image for placard: `{os.path.basename(current_image_path)}`"]
            if metadata and 'error' not in metadata: placard_details_md_log.append(f"Metadata provided (keys): {list(metadata.keys())}")
            elif 'error' in metadata : placard_details_md_log.append(f"Metadata contains error: {metadata.get('error')}")
            else: placard_details_md_log.append("No metadata provided.")
            self._log_to_markdown(self._format_log_entry(agent_name="DocentAgent", action="Initializing Placard", details=[f"Delegating to `PlacardAgent`."] + placard_details_md_log, image_context=base_filename))
            placard_summary_md_log = "Placard stage skipped."
            placard_error_md_log = None

            if not os.path.exists(current_image_path):
                logger.error(f"[{base_filename}] Input image for Placard not found: {current_image_path}. Skipping.")
                placard_summary_md_log = f"SKIPPED (Input not found: {current_image_path})"
                placard_error_md_log = "Input image for placard not found."
                self._log_to_markdown(self._format_log_entry(agent_name="PlacardAgent", action="Placard Creation", details=placard_details_md_log, summary=placard_summary_md_log, error=placard_error_md_log, image_context=base_filename))
            else:
                try:
                    placard_agent = self.agents['placard']
                    name, ext = os.path.splitext(os.path.basename(current_image_path))
                    original_base_name = name.replace('_upscaled', '').replace('_cropped', '')
                    final_filename = f"{original_base_name}_final{ext}"
                    final_output_path = os.path.join(self.output_folder, final_filename)
                    placard_details_md_log.append(f"Target final output path: `{final_output_path}`")
                    
                    placarded_path_result = placard_agent.add_plaque(current_image_path, final_output_path, metadata)
                    if placarded_path_result and os.path.exists(placarded_path_result):
                        current_image_path = placarded_path_result # This is the final image
                        logger.info(f"[{base_filename}] Placard addition successful. Final output: {current_image_path}")
                        placard_summary_md_log = f"Placard addition successful. Final output: `{os.path.basename(current_image_path)}`"
                        
                        cleanup_enabled = self.config.get('cleanup_workspace', True)
                        if cleanup_enabled:
                            # ... (cleanup logging can be added here if desired, but keeping it concise for now) ...
                            pass
                    else:
                        logger.error(f"[{base_filename}] Placard addition failed or output path invalid.")
                        placard_summary_md_log = f"Placard addition FAILED (Output invalid: {placarded_path_result})."
                        placard_error_md_log = "Placard output invalid or not found."
                    self._log_to_markdown(self._format_log_entry(agent_name="PlacardAgent", action="Placard Creation", details=placard_details_md_log, summary=placard_summary_md_log, error=placard_error_md_log, image_context=base_filename))
                except Exception as placard_err:
                    logger.error(f"[{base_filename}] Placard stage failed: {placard_err}", exc_info=True)
                    placard_summary_md_log = f"Placard stage FAILED: {placard_err}."
                    placard_error_md_log = str(placard_err)
                    self._log_to_markdown(self._format_log_entry(agent_name="PlacardAgent", action="Placard Creation", details=placard_details_md_log, summary=placard_summary_md_log, error=placard_error_md_log, image_context=base_filename))
        
        logger.info(f"--- Finished handling request for image: {base_filename} ---")
        self._log_to_markdown(self._format_log_entry(agent_name="DocentAgent", action="Request Handling Complete", summary=f"Finished processing request for `{base_filename}`. Final image path: `{current_image_path}`. Metadata keys: {list(metadata.keys())}", image_context=base_filename))
        self._log_to_markdown(f"\n## Workflow Run Ended (Programmatic Request: {task_description})\n- **Timestamp:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n- **Final Image Path:** `{current_image_path}`\n- **Final Metadata:** {json.dumps(metadata, indent=2)}\n---")

        # Add called agents to metadata for User Instruction 4
        metadata["called_agents_workflow"] = called_agents_log
 
        return {"final_image_path": current_image_path, "metadata": metadata}
 
 # --- End of Class ---
# Worker methods (_initial_image_processor, _upscale_worker, _placard_worker) are removed.