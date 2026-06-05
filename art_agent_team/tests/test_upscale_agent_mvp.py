import os
import sys
import logging
from pathlib import Path
import yaml

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.upscale_agent import UpscaleAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_config():
    """Load the configuration from config.yaml"""
    config_path = Path(__file__).parents[1] / 'config' / 'config.yaml'
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        return None

def test_upscale_agent(config):
    """Test UpscaleAgent with different configurations and input images"""
    
    # Setup test paths
    test_input_dir = Path(__file__).parent / 'test_data' / 'input'
    test_output_dir = Path(__file__).parent / 'test_data' / 'output' / 'upscale_verification'
    test_output_dir.mkdir(parents=True, exist_ok=True)

    test_images = [
        'Frank Brangwyn, Swans, c.1921.jpg',
        'Litzlberg am Attersee (1915).jpeg'
    ]

    test_configs = {
        'both_apis': config,
        'image_upscaling_only': {
            'image_upscaling_id': config.get('image_upscaling_id'),
            'upscaler_preference': 'image_upscaling_net'
        },
        'stability_only': {
            'stablediffusion_api_key': config.get('stablediffusion_api_key'),
            'upscaler_preference': 'stability_ai'
        },
        'fallback': {
            'image_upscaling_id': config.get('image_upscaling_id'),
            'stablediffusion_api_key': config.get('stablediffusion_api_key'),
            'upscaler_preference': 'fallback'
        }
    }

    results = []

    for config_name, test_config in test_configs.items():
        logging.info(f"\nTesting configuration: {config_name}")
        agent = UpscaleAgent(config=test_config)
        
        for image_name in test_images:
            input_path = test_input_dir / image_name
            output_name = f"{Path(image_name).stem}_{config_name}_upscaled{Path(image_name).suffix}"
            output_path = test_output_dir / output_name

            logging.info(f"Testing upscale of {image_name} with {config_name} configuration")
            
            try:
                result = agent.upscale_image(
                    str(input_path),
                    str(output_path)
                )
                
                success = result is not None and os.path.exists(output_path)
                results.append({
                    'config': config_name,
                    'image': image_name,
                    'success': success,
                    'output_path': str(output_path) if success else None
                })
                
                logging.info(f"Upscale {'succeeded' if success else 'failed'} for {image_name}")
                
            except Exception as e:
                logging.error(f"Error during upscale test: {e}")
                results.append({
                    'config': config_name,
                    'image': image_name,
                    'success': False,
                    'error': str(e)
                })

    return results

def main():
    """Main test execution function"""
    config = load_config()
    if not config:
        logging.error("Failed to load configuration. Aborting tests.")
        sys.exit(1)

    logging.info("Starting UpscaleAgent verification tests")
    results = test_upscale_agent(config)
    
    # Log results summary
    logging.info("\n=== Test Results Summary ===")
    success_count = sum(1 for r in results if r['success'])
    total_tests = len(results)
    
    logging.info(f"Total Tests: {total_tests}")
    logging.info(f"Successful: {success_count}")
    logging.info(f"Failed: {total_tests - success_count}")
    
    logging.info("\nDetailed Results:")
    for result in results:
        status = "✓ Success" if result['success'] else "✗ Failed"
        logging.info(f"{status} - Config: {result['config']}, Image: {result['image']}")
        if not result['success'] and 'error' in result:
            logging.info(f"  Error: {result['error']}")
        elif result['success']:
            logging.info(f"  Output: {result['output_path']}")

    # Exit with status code based on test results
    sys.exit(0 if success_count == total_tests else 1)

if __name__ == "__main__":
    main()