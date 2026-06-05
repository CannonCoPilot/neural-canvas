import os
import logging
import datetime
import yaml
import shutil
import sys

# Ensure the art_agent_team package can be found
# This assumes the script is run from the workspace root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from art_agent_team.agents.vision_agent_still_life import VisionAgentStillLife, CorruptedImageError, UnsupportedImageFormatError
from art_agent_team.agents.upscale_agent import UpscaleAgent
from art_agent_team.agents.placard_agent import PlacardAgent

# --- Configuration ---
LOG_DIR = "logs"
LIVE_TEST_LOG_FILE_NAME = f"live_test_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
CONFIG_FILE_PATH = "art_agent_team/config/config.yaml"

# Input and Output Base Paths (relative to workspace root)
INPUT_IMAGE_DIR = "art_agent_team/tests/test_data/input"
OUTPUT_BASE_DIR = "art_agent_team/tests/test_data/output/live_test_run" # Timestamped subfolder will be created

# Asset Paths for Placard Agent (relative to workspace root)
PLACARD_FONT_DIR = "art_agent_team/assets/fonts" # Assuming fonts like arial.ttf are here
PLACARD_BACKGROUND_DIR = "art_agent_team/assets/backgrounds" # Assuming card_stock.jpg is here

# Test Images
TEST_IMAGES = [
    "Frank Brangwyn, Swans, c.1921.jpg",
    "Litzlberg am Attersee (1915).jpeg"
]

def setup_logging(log_file_path):
    """Configures logging for the test run."""
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Logging setup complete. Log file: {log_file_path}")

def load_main_config():
    """Loads API keys and other global configurations."""
    try:
        with open(CONFIG_FILE_PATH, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Successfully loaded main configuration from {CONFIG_FILE_PATH}")
        return config
    except FileNotFoundError:
        logging.error(f"ERROR: Main configuration file not found at {CONFIG_FILE_PATH}. Aborting.")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"ERROR: Could not parse main configuration file {CONFIG_FILE_PATH}. Error: {e}. Aborting.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"ERROR: An unexpected error occurred while loading config from {CONFIG_FILE_PATH}. Error: {e}. Aborting.")
        sys.exit(1)

def process_image(image_filename, main_config, current_output_dir):
    """Processes a single image through the agent workflow."""
    logging.info(f"--- Starting processing for image: {image_filename} ---")
    base_image_name = os.path.splitext(image_filename)[0]
    input_image_path = os.path.join(INPUT_IMAGE_DIR, image_filename)

    if not os.path.exists(input_image_path):
        logging.error(f"Input image not found: {input_image_path}. Skipping this image.")
        return

    # Define output paths for this specific image
    vision_output_dir = os.path.join(current_output_dir, "vision_output")
    upscaled_output_dir = os.path.join(current_output_dir, "upscaled_output")
    placard_output_dir = os.path.join(current_output_dir, "placard_output")

    os.makedirs(vision_output_dir, exist_ok=True)
    os.makedirs(upscaled_output_dir, exist_ok=True)
    os.makedirs(placard_output_dir, exist_ok=True)

    vision_agent_output_path_prefix = os.path.join(vision_output_dir, base_image_name)
    upscaled_image_path = os.path.join(upscaled_output_dir, f"{base_image_name}_upscaled.png") # Upscaler might change format
    placard_image_path = os.path.join(placard_output_dir, f"{base_image_name}_placarded.jpg")


    # 1. Vision Agent
    logging.info(f"--- Step 1: Vision Agent for {image_filename} ---")
    vision_agent_config = {
        'gcp_project_id': main_config.get('gcp_project_id'),
        'gcp_location': main_config.get('gcp_location', 'us-central1'),
        'vertex_model_id': main_config.get('vertex_model_id_still_life'), # Assuming a still life model for now
        'output_folder': vision_output_dir # Vision agent saves its own files here
    }
    # Ensure output_folder is correctly passed and used by VisionAgent for its internal saves
    # The `_save_labeled_version` and `_save_masked_version` in VisionAgentStillLife use self.output_folder

    vision_agent = VisionAgentStillLife(config=vision_agent_config)
    vision_analysis_result = None
    try:
        # Mock research data for now, as per current VisionAgentStillLife signature
        mock_research_data = {
            "primary_subject": "still life element",
            "secondary_subjects": ["composition", "lighting"]
        }
        vision_analysis_result = vision_agent.analyze_image(input_image_path, research_data=mock_research_data)
        if vision_analysis_result:
            logging.info(f"Vision Agent analysis successful for {image_filename}.")
            logging.info(f"  Visualization saved to: {vision_analysis_result.get('visualization_path')}")
            logging.info(f"  Masked version saved to: {vision_analysis_result.get('masked_version_path')}")
        else:
            logging.error(f"Vision Agent analysis failed or returned no result for {image_filename}.")
    except CorruptedImageError as e:
        logging.error(f"Vision Agent: Corrupted image {input_image_path}. Error: {e}")
    except UnsupportedImageFormatError as e:
        logging.error(f"Vision Agent: Unsupported image format for {input_image_path}. Error: {e}")
    except Exception as e:
        logging.error(f"Vision Agent: Unexpected error during analysis of {input_image_path}. Error: {e}", exc_info=True)


    # 2. Upscale Agent
    logging.info(f"--- Step 2: Upscale Agent for {image_filename} ---")
    upscale_agent_config = {
        'image_upscaling_id': main_config.get('image_upscaling_id'),
        'stablediffusion_api_key': main_config.get('stablediffusion_api_key'),
        'upscaler_preference': main_config.get('upscaler_preference', 'fallback')
    }
    upscale_agent = UpscaleAgent(config=upscale_agent_config)
    actual_upscaled_path = None
    try:
        # Upscale the original image
        actual_upscaled_path = upscale_agent.upscale_image(input_image_path, upscaled_image_path)
        if actual_upscaled_path:
            logging.info(f"Upscale Agent successful for {image_filename}. Output: {actual_upscaled_path}")
        else:
            logging.warning(f"Upscale Agent failed for {image_filename}. Proceeding with original image for placard if possible.")
    except Exception as e:
        logging.error(f"Upscale Agent: Unexpected error during upscaling of {input_image_path}. Error: {e}", exc_info=True)
        logging.warning(f"Proceeding with original image for placard due to upscaling error.")

    image_for_placard = actual_upscaled_path if actual_upscaled_path and os.path.exists(actual_upscaled_path) else input_image_path

    # 3. Placard Agent
    logging.info(f"--- Step 3: Placard Agent for {image_filename} (using {os.path.basename(image_for_placard)}) ---")
    placard_agent_config = {
        'font_path': os.path.join(PLACARD_FONT_DIR, main_config.get('placard_font_regular', 'arial.ttf')),
        'font_path_bold': os.path.join(PLACARD_FONT_DIR, main_config.get('placard_font_bold', 'arialbd.ttf')),
        'background_image_path': os.path.join(PLACARD_BACKGROUND_DIR, main_config.get('placard_background_default', 'card_stock.jpg')),
        'plaque_opacity': main_config.get('placard_plaque_opacity', 0.9),
        'margin_percent': main_config.get('placard_margin_percent', 5),
        'text_color': tuple(main_config.get('placard_text_color', [0,0,0,255])) # Ensure it's a tuple
    }
    placard_agent = PlacardAgent(config=placard_agent_config)

    # Prepare metadata for placard - use Vision Agent output if available
    placard_metadata = {
        'title': base_image_name.replace('_', ' '), # Basic title from filename
        'author': 'Unknown Artist',
        'date': 'N/A',
        'nationality': 'N/A',
        'style': 'N/A',
        'genre': 'General' # Default genre
    }
    if vision_analysis_result and vision_analysis_result.get('analysis'):
        va_analysis = vision_analysis_result['analysis']
        # This is a placeholder - actual mapping depends on VisionAgent's output structure
        placard_metadata['title'] = va_analysis.get('title', placard_metadata['title'])
        placard_metadata['author'] = va_analysis.get('artist', placard_metadata['author']) # Assuming 'artist' key
        placard_metadata['date'] = va_analysis.get('creation_date', placard_metadata['date']) # Assuming 'creation_date'
        placard_metadata['style'] = va_analysis.get('style', placard_metadata['style'])
        placard_metadata['genre'] = va_analysis.get('genre', placard_metadata['genre'])
        # Add more fields as available and relevant from vision_analysis_result['analysis']
        if vision_analysis_result.get('important_objects'):
             placard_metadata['description'] = f"Key elements: {', '.join(vision_analysis_result['important_objects'][:3])}"


    try:
        final_placard_path = placard_agent.add_plaque(image_for_placard, placard_image_path, placard_metadata)
        if final_placard_path:
            logging.info(f"Placard Agent successful for {image_filename}. Output: {final_placard_path}")
        else:
            logging.error(f"Placard Agent failed for {image_filename}.")
    except Exception as e:
        logging.error(f"Placard Agent: Unexpected error during placarding of {image_for_placard}. Error: {e}", exc_info=True)

    logging.info(f"--- Finished processing for image: {image_filename} ---")


def main():
    """Main function to run the live test workflow."""
    timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    current_run_output_dir = os.path.join(OUTPUT_BASE_DIR, f"run_{timestamp_str}")
    os.makedirs(current_run_output_dir, exist_ok=True)

    log_file_full_path = os.path.join(LOG_DIR, f"live_test_run_{timestamp_str}.log")
    setup_logging(log_file_full_path)

    logging.info("Starting Live Test Run for Multi-Agent Image Processing Workflow")
    logging.info(f"Output for this run will be in: {current_run_output_dir}")

    main_cfg = load_main_config()
    if not main_cfg:
        return # load_main_config logs error and exits

    # Check for essential config keys (GCP for Vision, at least one upscaler API)
    if not main_cfg.get('gcp_project_id') or not main_cfg.get('vertex_model_id_still_life'):
        logging.error("GCP project ID or Vertex Model ID for Still Life not found in config. VisionAgent may fail.")
        # Decide if to proceed or exit, for now, we proceed with a warning.
    if not main_cfg.get('image_upscaling_id') and not main_cfg.get('stablediffusion_api_key'):
        logging.warning("Neither image-upscaling.net ID nor Stability AI API key found in config. Upscaling will likely fail.")


    for image_file in TEST_IMAGES:
        process_image(image_file, main_cfg, current_run_output_dir)

    logging.info("Live Test Run Completed.")
    logging.info(f"All outputs for this run are in: {current_run_output_dir}")
    logging.info(f"Detailed log file: {log_file_full_path}")

if __name__ == "__main__":
    main()