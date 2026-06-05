import os
import sys
import pytest

# Ensure the root directory is in the system path to recognize the package structure
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

def pytest_configure(config):
    """
    Configure pytest to ensure the package context is set correctly.
    This helps in resolving relative imports by setting __package__ appropriately.
    """
    os.environ.setdefault('PYTHONPATH', f"{root_dir}:{os.environ.get('PYTHONPATH', '')}")

@pytest.fixture(scope="session", autouse=True)
def configure_test_environment():
    """
    Fixture to run before any tests, ensuring the environment is set up correctly.
    """
    # Set up mock configuration for test environment to avoid using real API keys
    os.environ['GOOGLE_API_KEY'] = 'test_google_api_key'
    os.environ['GROK_API_KEY'] = 'test_grok_api_key'
    os.environ['IMAGE_UPSCALING_ID'] = 'test_image_upscaling_id'
    os.environ['STABLEDIFFUSION_API_KEY'] = 'test_stablediffusion_api_key'
    os.environ['VERTEX_MODEL_ID'] = 'test_vertex_model_id'
    os.environ['GCP_PROJECT_ID'] = 'test_gcp_project_id' # Added for VisionAgent

@pytest.fixture(scope="function", autouse=True)
def mock_config(request, monkeypatch): # Added request
    """
    Fixture to override configuration loading in tests with mock values.
    This ensures tests use mock API keys and settings without modifying the actual config file.
    It also sets 'self.config' on unittest test instances when @pytest.mark.usefixtures is used.
    """
    def _mock_load_config_dict():
        return {
            'image_upscaling_id': os.environ.get('IMAGE_UPSCALING_ID', 'test_image_upscaling_id'),
            'stablediffusion_api_key': os.environ.get('STABLEDIFFUSION_API_KEY', 'test_stablediffusion_api_key'),
            'google_api_key': os.environ.get('GOOGLE_API_KEY', 'test_google_api_key'),
            'grok_api_key': os.environ.get('GROK_API_KEY', 'test_grok_api_key'),
            'vertex_model_id': os.environ.get('VERTEX_MODEL_ID', 'test_vertex_model_id'),
            'gcp_project_id': os.environ.get('GCP_PROJECT_ID', 'test_gcp_project_id'),
            'input_folder': 'input',
            'output_folder': 'output'
        }
    
    config_data = _mock_load_config_dict()

    # Apply mock to all relevant test modules that might load config
    monkeypatch.setattr("art_agent_team.docent_agent.DocentAgent._load_config", _mock_load_config_dict)
    monkeypatch.setattr("art_agent_team.main.load_and_set_env_from_config", lambda config_path: None) # Prevents loading from actual file

    # For unittest.TestCase compatibility when used with @pytest.mark.usefixtures
    if hasattr(request, 'instance'):
        request.instance.config = config_data
    
    return config_data # Return for standard pytest injection if needed elsewhere