import os
import sys
import logging
import yaml
import argparse # For command-line arguments
# Adjust the path to include the parent directory for module resolution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.upscale_agent import UpscaleAgent
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found at {config_path}.")
        return {}
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}

def test_upscale_agent(upscaler_preference: str):
    """
    Test the UpscaleAgent functionality.
    :param upscaler_preference: 'image_upscaling_net', 'stability_ai', or 'fallback'.
    """
    logger.info(f"Starting test for UpscaleAgent with preference: {upscaler_preference}")
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')
    config = load_config(config_path)
    
    if not config:
        logger.error("Test failed: Config could not be loaded.")
        return False

    # Add the upscaler_preference to the config for the agent
    config['upscaler_preference'] = upscaler_preference

    # Check for necessary API keys based on preference
    if upscaler_preference == 'image_upscaling_net' or upscaler_preference == 'fallback':
        if 'image_upscaling_id' not in config:
            logger.error(f"Test failed: 'image_upscaling_id' not found in config, required for '{upscaler_preference}' preference.")
            # If fallback, we might still proceed if stability_ai is configured.
            if upscaler_preference == 'image_upscaling_net':
                 return False
    
    if upscaler_preference == 'stability_ai' or upscaler_preference == 'fallback':
        if 'stablediffusion_api_key' not in config:
            logger.error(f"Test failed: 'stablediffusion_api_key' not found in config, required for '{upscaler_preference}' preference.")
            if upscaler_preference == 'stability_ai':
                return False
    
    if upscaler_preference == 'fallback' and 'image_upscaling_id' not in config and 'stablediffusion_api_key' not in config:
        logger.error("Test failed for 'fallback': Neither 'image_upscaling_id' nor 'stablediffusion_api_key' found in config.")
        return False

    input_folder = config.get('input_folder')
    output_folder = config.get('output_folder')
    
    if not input_folder or not output_folder:
        logger.error("Test failed: 'input_folder' or 'output_folder' not found in config.")
        return False
    
    # Use a different output filename for each preference to avoid conflicts if run sequentially
    # And use .png as Stability AI default is webp, but we can request png.
    # The agent itself will handle output_format for Stability AI based on output_path extension.
    output_filename = f"test_upscale_output_{upscaler_preference.replace('_', '')}.png"
    input_path = os.path.join(input_folder, 'test_upscale_input.jpg') # Assuming JPG input
    output_path = os.path.join(output_folder, output_filename)
    
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    
    # Prepare input image
    demo_image_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'test_image.jpg') # Path relative to project root
    # Corrected path for demo_image_path, assuming tests is a sibling of art_agent_team, and test_image.jpg is in project root.
    # If test_image.jpg is inside art_agent_team, adjust accordingly.
    # For this example, let's assume test_image.jpg is in the root of Art_AI project.
    # A more robust way would be to have test assets within the test_data folder.
    # Using a fixed path for the input image for simplicity in this test script.
    # The original script used `os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_image.jpg')`
    # which means test_image.jpg is expected in the `art_agent_team` directory.
    # Let's stick to that assumption for minimal changes to existing logic if `test_image.jpg` is there.
    # If `test_image.jpg` is in the project root:
    # demo_image_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'test_image.jpg')
    # If `test_image.jpg` is in `art_agent_team/input/`:
    # demo_image_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'input', 'test_image.jpg')
    # For now, let's assume it's in the `input_folder` as `test_upscale_input.jpg`
    
    if not os.path.exists(input_path):
        logger.info(f"Test input file {input_path} not found. Trying to use demo image or create one.")
        # Path to the original demo image, assuming it's in the project root as `download_demo_image.py` suggests
        project_root_demo_image = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'test_image.jpg')
        if os.path.exists(project_root_demo_image):
            try:
                with Image.open(project_root_demo_image) as img:
                    img.save(input_path) # Save it to the configured input_folder
                logger.info(f"Used demo image from {project_root_demo_image} and saved to {input_path}")
            except Exception as e:
                logger.error(f"Failed to use demo image: {e}")
                # Create a dummy if demo image fails
                try:
                    img = Image.new('RGB', (100, 100), color='blue')
                    img.save(input_path)
                    logger.info(f"Created dummy test image at {input_path}")
                except Exception as e_create:
                    logger.error(f"Test failed: Failed to create test image at {input_path}: {e_create}")
                    return False
        else: # If demo image not found at root, create a dummy
            try:
                img = Image.new('RGB', (100, 100), color='blue')
                img.save(input_path)
                logger.info(f"Created dummy test image at {input_path} as demo image was not found at {project_root_demo_image}")
            except Exception as e:
                logger.error(f"Test failed: Failed to create test image at {input_path}: {e}")
                return False
    else:
        logger.info(f"Using existing test input image at {input_path}")

    agent = UpscaleAgent(config=config)
    
    # Check if any API is ready based on preference
    agent_ready = False
    if upscaler_preference == 'image_upscaling_net' and agent.image_upscaling_net_ready:
        agent_ready = True
    elif upscaler_preference == 'stability_ai' and agent.stability_ai_ready:
        agent_ready = True
    elif upscaler_preference == 'fallback' and (agent.image_upscaling_net_ready or agent.stability_ai_ready):
        agent_ready = True
    
    if not agent_ready:
        logger.error(f"Test failed: UpscaleAgent not ready for preference '{upscaler_preference}'. Check API keys and config.")
        # Log specific readiness states
        if upscaler_preference == 'image_upscaling_net' or upscaler_preference == 'fallback':
            logger.info(f"image-upscaling.net ready: {agent.image_upscaling_net_ready}")
        if upscaler_preference == 'stability_ai' or upscaler_preference == 'fallback':
            logger.info(f"Stability AI ready: {agent.stability_ai_ready}")
        return False
    
    logger.info(f"Attempting to upscale image from {input_path} to {output_path} using {upscaler_preference} preference.")
    result = agent.upscale_image(input_path, output_path)
    
    if result:
        logger.info(f"Upscaling call successful. Output expected at {result}")
        if os.path.exists(result):
            try:
                with Image.open(result) as img:
                    width, height = img.size
                    logger.info(f"Upscaled image dimensions: {width}x{height}")
                    # Basic check: image should be larger than 100x100 (our dummy/small input)
                    if width > 100 and height > 100:
                        logger.info(f"Test passed for '{upscaler_preference}': Upscaling completed successfully.")
                        return True
                    else:
                        logger.error(f"Test failed for '{upscaler_preference}': Upscaled image dimensions ({width}x{height}) are not significantly larger than input.")
                        return False
            except Exception as e:
                logger.error(f"Test failed for '{upscaler_preference}': Error reading upscaled image at {result}: {e}")
                return False
        else:
            logger.error(f"Test failed for '{upscaler_preference}': Output file does not exist at {result}.")
            return False
    else:
        logger.error(f"Test failed for '{upscaler_preference}': Upscaling process returned None.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test UpscaleAgent with different API preferences.")
    parser.add_argument(
        "--preference",
        type=str,
        default="stability_ai", # Default to Stability AI due to current issues with the other API
        choices=["image_upscaling_net", "stability_ai", "fallback"],
        help="Upscaler preference to test."
    )
    args = parser.parse_args()

    success = test_upscale_agent(args.preference)
    logger.info(f"UpscaleAgent test with preference '{args.preference}' {'passed' if success else 'failed'}")