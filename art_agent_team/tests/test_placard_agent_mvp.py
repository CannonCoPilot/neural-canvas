"""
MVP Verification Test for PlacardAgent
Tests basic placard generation and overlay functionality
"""

import os
import sys
from PIL import Image

# Add the parent directory to the Python path to import the agent
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.placard_agent import PlacardAgent

def verify_image_dimensions(input_path, output_path):
    """Verify the output image dimensions are valid relative to input"""
    with Image.open(input_path) as input_img:
        input_size = input_img.size
    with Image.open(output_path) as output_img:
        output_size = output_img.size
        
    # Output should have same dimensions as input since placard is overlaid
    assert output_size == input_size, f"Output dimensions {output_size} don't match input {input_size}"
    return True

def test_placard_basic_functionality():
    """Test basic placard generation and overlay with complete metadata"""
    
    # Test data paths
    input_image = os.path.abspath("art_agent_team/tests/test_data/input/Frank Brangwyn, Swans, c.1921.jpg")
    output_image = os.path.abspath("art_agent_team/tests/test_data/output/test_placard_swans.jpg")
    
    # Test metadata
    metadata = {
        'title': 'Swans',
        'author': 'Frank Brangwyn',
        'date': 'c.1921',
        'nationality': 'British',
        'style': 'Impressionist',
        'genre': 'animal'  # Genre helps with background selection
    }
    
    # Initialize agent with correct paths relative to workspace root
    agent = PlacardAgent(config={
        'background_image_path': 'input/card_stock.jpg',
        'font_path': 'fonts/arial.ttf',
        'font_path_bold': 'fonts/arialbd.ttf'
    })
    
    # Process image
    print("\nTesting placard generation for Swans...")
    result_path = agent.add_plaque(input_image, output_image, metadata)
    
    # Verify results
    if result_path and os.path.exists(result_path):
        print("✓ Placard generated successfully")
        print(f"✓ Output saved to: {result_path}")
        
        # Verify image dimensions
        try:
            verify_image_dimensions(input_image, result_path)
            print("✓ Output dimensions verification passed")
        except Exception as e:
            print(f"✗ Dimension verification failed: {e}")
            return False
        
        # Basic image integrity check
        try:
            with Image.open(result_path) as img:
                img.verify()
            print("✓ Output image integrity verification passed")
        except Exception as e:
            print(f"✗ Image integrity check failed: {e}")
            return False
            
        return True
    else:
        print("✗ Placard generation failed")
        return False

def test_placard_minimal_metadata():
    """Test placard generation with minimal metadata"""
    
    # Test data paths
    input_image = os.path.abspath("art_agent_team/tests/test_data/input/Litzlberg am Attersee (1915).jpeg")
    output_image = os.path.abspath("art_agent_team/tests/test_data/output/test_placard_litzlberg.jpg")
    
    # Minimal metadata
    metadata = {
        'title': 'Litzlberg am Attersee',
        'date': '1915'
        # Intentionally omitting other fields to test fallback handling
    }
    
    # Initialize agent with correct paths relative to workspace root
    agent = PlacardAgent(config={
        'background_image_path': 'input/card_stock.jpg',
        'font_path': 'fonts/arial.ttf',
        'font_path_bold': 'fonts/arialbd.ttf'
    })
    
    # Process image
    print("\nTesting placard generation for Litzlberg...")
    result_path = agent.add_plaque(input_image, output_image, metadata)
    
    # Verify results
    if result_path and os.path.exists(result_path):
        print("✓ Placard generated successfully")
        print(f"✓ Output saved to: {result_path}")
        
        # Verify image dimensions
        try:
            verify_image_dimensions(input_image, result_path)
            print("✓ Output dimensions verification passed")
        except Exception as e:
            print(f"✗ Dimension verification failed: {e}")
            return False
        
        # Basic image integrity check
        try:
            with Image.open(result_path) as img:
                img.verify()
            print("✓ Output image integrity verification passed")
        except Exception as e:
            print(f"✗ Image integrity check failed: {e}")
            return False
            
        return True
    else:
        print("✗ Placard generation failed")
        return False

if __name__ == "__main__":
    print("Starting PlacardAgent MVP Verification Tests...")
    
    # Run tests
    basic_test_passed = test_placard_basic_functionality()
    minimal_test_passed = test_placard_minimal_metadata()
    
    # Report results
    print("\nTest Results:")
    print(f"Basic Functionality Test: {'✓ Passed' if basic_test_passed else '✗ Failed'}")
    print(f"Minimal Metadata Test: {'✓ Passed' if minimal_test_passed else '✗ Failed'}")
    
    # Overall status for CI/CD
    if basic_test_passed and minimal_test_passed:
        print("\n✓ All tests passed - PlacardAgent MVP verification complete")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed - see above for details")
        sys.exit(1)