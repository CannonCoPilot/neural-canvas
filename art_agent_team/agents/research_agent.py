import logging
import os
import json
import time
import random
import base64
import re # Added for date validation
import threading # Added for thread safety
import concurrent.futures
from typing import List, Dict, Optional, Any, Union
from datetime import datetime

# --- LLM Client Imports ---
import openai # Used for Grok and OpenRouter clients
from openai import OpenAI # Explicit import for type hints
import vertexai
from vertexai.preview import generative_models as vertex_genai # Use alias
from vertexai.preview.generative_models import GenerativeModel, Part
import importlib

# --- Google GenerativeAI Import ---
try:
    google_genai = importlib.import_module("google.generativeai")
except ImportError:
    google_genai = None

# --- Logging Setup ---
# Assumes basic logging is configured by the calling script (e.g., DocentAgent)

# --- Constants ---
# Predefined Styles List (as per requirements)
# Predefined Styles List (as per requirements) - Updated with art movements from movements.txt
PREDEFINED_STYLES = [
    "landscape", "portrait", "surrealist", "genre-scene", "animal", "religious",
    "historical", "still life", "abstract"
    # Art movements have been removed from this list to ensure clear separation
    # between style and movement. Movements are now handled exclusively via movements.txt.
]

def _is_supported_google_genai_content_type(obj):
    # Acceptable: bytes, str, dict, PIL.Image.Image, IPython.display.Image
    try:
        from PIL import Image as PILImage
    except ImportError:
        PILImage = None
    try:
        from IPython.display import Image as IPyImage
    except ImportError:
        IPyImage = None
    allowed_types = [bytes, str, dict]
    if PILImage:
        allowed_types.append(PILImage)
    if IPyImage:
        allowed_types.append(IPyImage)
    # Blob is not imported, but if present, allow by name
    if type(obj).__name__ == "Blob":
        return True
    return isinstance(obj, tuple(allowed_types))

# API Base URLs
XAI_BASE_URL = "https://api.x.ai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# --- New Model Constants (Refactored Workflow) ---
# USER REQUIREMENT: DO NOT CHANGE THESE MODEL DEFINITIONS.
## Vision Models (Includes models potentially capable of text too)
## Vision Models to be called through the OpenRouter API using OpenRouter API Key
OPENROUTER_LLAMA_MAVERICK = "meta-llama/llama-4-maverick:free"
OPENROUTER_LLAMA_SCOUT = "meta-llama/llama-4-scout:free"
OPENROUTER_QWEN_VL = "qwen/qwen2.5-vl-72b-instruct:free"
OPENROUTER_INTERNVL = "opengvlab/internvl3-14b:free"
OPENROUTER_MISTRAL3 = "mistralai/mistral-small-3.1-24b-instruct:free"
OPENROUTER_GEMMA3 = "google/gemma-3-27b-it:free"
## Vision Models to be called through the Google Gemini API using Gemini API Key, or Google Vertex AI API using Vertex AI project credentials
GEMINI_IMAGE_SEARCH_MODEL = "gemini-2.5-pro-exp-03-25"
FLASH_IMAGE_SEARCH_MODEL = "gemini-2.0-flash-exp"
## Vision Models to be called through the Grok OpenAI-compatible API using Grok API Key
GROK_IMAGE_SEARCH_MODEL = "grok-2-vision-1212"
# Consolidation Model to be called through the Grok OpenAI-compatible API using Grok API Key
CONSOLIDATION_MODEL = "grok-3-mini-fast-high-beta"


# --- Model Registry ---
class ModelRegistry:
    """Manages LLM models, API clients, and configurations."""
    def __init__(self, config: Dict, agent_id: int):
        self.models = {}
        self.clients = {} # Stores initialized clients (OpenAI instances, VertexAI GenerativeModel instances, or True for Vertex SDK init)
        self.config = config
        self.agent_id = agent_id
        self._vertex_client_lock = threading.Lock()  # Initialize thread lock for Vertex AI client instantiation
        self._initialize_clients()
        self._register_models()

    def _initialize_clients(self):
        """Initializes base API clients based on available keys. Logs explicit warnings for missing configs."""
        tag = f"[ModelRegistry {self.agent_id}]"
        # Grok (xAI) Client
        grok_api_key = os.getenv('GROK_API_KEY')
        if not grok_api_key:
            logging.warning(f"{tag} GROK_API_KEY not set in environment. Grok models unavailable.")
        else:
            try:
                self.clients['grok'] = OpenAI(api_key=grok_api_key, base_url=XAI_BASE_URL)
                logging.info(f"{tag} Grok (xAI) client initialized.")
            except Exception as e:
                logging.error(f"{tag} Failed to initialize Grok client: {e}")

        # OpenRouter Client
        openrouter_api_key = self.config.get('openrouter_api_key') or os.getenv('OPENROUTER_API_KEY')
        if not openrouter_api_key:
            logging.warning(f"{tag} OPENROUTER_API_KEY not set in config or environment. OpenRouter models unavailable.")
        else:
            try:
                self.clients['openrouter'] = OpenAI(
                    base_url=OPENROUTER_BASE_URL,
                    api_key=openrouter_api_key,
                )
                logging.info(f"{tag} OpenRouter client initialized.")
            except Exception as e:
                logging.error(f"{tag} Failed to initialize OpenRouter client: {e}")
        
        # Google GenerativeAI (Gemini) Client Initialization
        # Check for Google API Key in config or environment (support both cases)
        google_api_key = (
            self.config.get('google_api_key')
            or os.environ.get('GOOGLE_API_KEY')
            or os.environ.get('google_api_key')
        )
        if not google_api_key:
            logging.warning(f"{tag} google_api_key not set in config or environment. Gemini models unavailable.")
        elif google_genai is None:
            logging.warning(f"{tag} google.generativeai library not installed. Gemini models unavailable.")
        else:
            try:
                google_genai.configure(api_key=google_api_key)
                self.clients['google_genai'] = google_genai
                logging.info(f"{tag} Google GenerativeAI client initialized for Gemini models.")
            except Exception as e:
                logging.error(f"{tag} Failed to initialize Google GenerativeAI client: {e}")

        # Vertex AI (Google Gemini) SDK Initialization
        try:
            # Check for Vertex AI project_id and location in config or environment (support both cases)
            project_id = (
                self.config.get('vertex_project_id')
                or os.environ.get('VERTEX_PROJECT_ID')
                or os.environ.get('vertex_project_id')
            )
            location = (
                self.config.get('vertex_location')
                or os.environ.get('VERTEX_LOCATION')
                or os.environ.get('vertex_location')
            )
            if not project_id:
                logging.warning(f"{tag} vertex_project_id not set in config or environment. Vertex AI models unavailable.")
            if not location:
                logging.warning(f"{tag} vertex_location not set in config or environment. Vertex AI models unavailable.")
            
            google_credentials_path_config = self.config.get('google_credentials_path')
            current_google_creds_env = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

            if project_id and location:
                if google_credentials_path_config:
                    # Ensure the path from config is absolute or resolve it relative to a known base if necessary
                    # For now, assume it's a relative path from the workspace root or an absolute path.
                    # If it's relative, it might need adjustment based on where the script is run from.
                    # For robustness, consider making it an absolute path in config or resolving it.
                    resolved_credentials_path = os.path.abspath(google_credentials_path_config)
                    if os.path.exists(resolved_credentials_path):
                        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = resolved_credentials_path
                        logging.info(f"{tag} Using GOOGLE_APPLICATION_CREDENTIALS from config: {resolved_credentials_path}")
                    else:
                        logging.warning(f"{tag} google_credentials_path '{google_credentials_path_config}' (resolved to '{resolved_credentials_path}') from config not found. Vertex AI may fail if ADC is not configured.")
                        if current_google_creds_env:
                             logging.info(f"{tag} Falling back to existing GOOGLE_APPLICATION_CREDENTIALS environment variable: {current_google_creds_env}")
                        else:
                             logging.warning(f"{tag} No GOOGLE_APPLICATION_CREDENTIALS in environment either. Vertex AI will attempt ADC.")
                elif current_google_creds_env:
                    logging.info(f"{tag} Using existing GOOGLE_APPLICATION_CREDENTIALS environment variable: {current_google_creds_env}")
                else:
                    logging.warning(f"{tag} GOOGLE_APPLICATION_CREDENTIALS not set in config or environment. Vertex AI will attempt to use Application Default Credentials (ADC).")

                vertexai.init(project=project_id, location=location)
                self.clients['vertexai_sdk_initialized'] = True
                logging.info(f"{tag} Vertex AI SDK initialized for project {project_id} in {location}.")
            else:
                self.clients['vertexai_sdk_initialized'] = False
        except ImportError as e:
            logging.error(f"{tag} Vertex AI libraries not installed? Error: {e}")
            self.clients['vertexai_sdk_initialized'] = False
        except Exception as e:
            logging.error(f"{tag} Failed to initialize Vertex AI SDK: {e}")
            self.clients['vertexai_sdk_initialized'] = False


    def _register_models(self):
        """Registers models specified in the requirements, linking them to client types."""
        tag = f"[ModelRegistry {self.agent_id}]"
        # Define models with their client type and capabilities ('vision' or 'text')
        model_definitions = {
            # Vision Models
            OPENROUTER_LLAMA_MAVERICK: {'client_type': 'openrouter', 'capability': 'vision', 'provider': 'openrouter'},
            OPENROUTER_LLAMA_SCOUT: {'client_type': 'openrouter', 'capability': 'vision', 'provider': 'openrouter'},
            OPENROUTER_QWEN_VL: {'client_type': 'openrouter', 'capability': 'vision', 'provider': 'openrouter'},
            OPENROUTER_INTERNVL: {'client_type': 'openrouter', 'capability': 'vision', 'provider': 'openrouter'},
            GEMINI_IMAGE_SEARCH_MODEL: {'client_type': 'vertexai', 'capability': 'vision', 'provider': 'google'},
            FLASH_IMAGE_SEARCH_MODEL: {'client_type': 'vertexai', 'capability': 'vision', 'provider': 'google'},
            GROK_IMAGE_SEARCH_MODEL: {'client_type': 'grok', 'capability': 'vision', 'provider': 'grok'},
            # Text Models (can include vision models if they also handle text well)
            OPENROUTER_MISTRAL3: {'client_type': 'openrouter', 'capability': 'text', 'provider': 'openrouter'},
            OPENROUTER_GEMMA3: {'client_type': 'openrouter', 'capability': 'text', 'provider': 'openrouter'},
            # Consolidation Model (must be text-capable)
            CONSOLIDATION_MODEL: {'client_type': 'grok', 'capability': 'text', 'provider': 'grok'},  # Changed to use Grok client per model requirements
        }

        for name, details in model_definitions.items():
            client_type = details['client_type']
            # Check if the corresponding client is available
            client_available = False
            if client_type == 'vertexai':
                client_available = self.clients.get('vertexai_sdk_initialized', False)
            else:
                client_available = client_type in self.clients # Check if Grok or OpenRouter client exists

            if client_available:
                self.models[name] = details
                logging.debug(f"{tag} Registered model: {name} (Client Type: {client_type}, Capability: {details['capability']})")
            else:
                logging.warning(f"{tag} Skipping registration of model '{name}' because required client type '{client_type}' is unavailable or not configured. This is expected if API keys or configs are missing.")

    def get_client_for_model(self, model_name: str) -> Optional[Any]:
        """Gets the initialized client instance for a given model name. Instantiates Vertex AI models on demand.
        For Gemini models, returns the standard google.generativeai client, not Vertex AI."""
        tag = f"[ModelRegistry {self.agent_id}]"
        model_info = self.models.get(model_name)
        if not model_info:
            logging.warning(f"{tag} Model '{model_name}' not registered or unavailable.")
            return None

        client_type = model_info['client_type']

        # Special handling for Gemini models: use google.generativeai client, not Vertex AI
        if model_info['provider'] == 'google' and model_name in [GEMINI_IMAGE_SEARCH_MODEL, FLASH_IMAGE_SEARCH_MODEL]:
            genai_client = self.clients.get('google_genai')
            if not genai_client:
                logging.error(f"{tag} Google GenerativeAI client not initialized for Gemini model {model_name}.")
            return genai_client

        # Handle Vertex AI model instantiation (for other Google/Vertex models)
        if client_type == 'vertexai':
            if not self.clients.get('vertexai_sdk_initialized'):
                logging.error(f"{tag} Vertex AI SDK not initialized. Cannot get client for {model_name}.")
                return None
            with self._vertex_client_lock:
                if model_name in self.clients:
                    return self.clients[model_name]
                else:
                    try:
                        safety_settings = {
                            vertex_genai.HarmCategory.HARM_CATEGORY_HARASSMENT: vertex_genai.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                            vertex_genai.HarmCategory.HARM_CATEGORY_HATE_SPEECH: vertex_genai.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                            vertex_genai.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: vertex_genai.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                            vertex_genai.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: vertex_genai.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                        }
                        logging.info(f"{tag} Instantiating Vertex AI model: {model_name}")
                        model_instance = GenerativeModel(model_name=model_name, safety_settings=safety_settings)
                        self.clients[model_name] = model_instance
                        logging.info(f"{tag} Successfully instantiated and cached Vertex AI model: {model_name}")
                        return model_instance
                    except Exception as e:
                        logging.error(f"{tag} Failed to instantiate Vertex AI model {model_name}. Error type: {type(e).__name__}. Message: {e}. Check model name validity, permissions, and SDK compatibility.", exc_info=True)
                        return None
        else:
            client_instance = self.clients.get(client_type)
            if not client_instance:
                logging.warning(f"{tag} Client instance for type '{client_type}' (needed for model {model_name}) not found.")
            return client_instance

    def get_models_by_capability(self, capability: str) -> List[str]:
        """Returns a list of registered model names matching the specified capability ('vision' or 'text')."""
        return [name for name, details in self.models.items() if details['capability'] == capability]

    def get_model_details(self, model_name: str) -> Optional[Dict]:
        """Returns the details dictionary for a specific model."""
        return self.models.get(model_name)

# --- Prompt Template System ---
class PromptTemplate:
    """Manages master and model-specific prompt generation."""
    def __init__(self):
        self.predefined_styles = PREDEFINED_STYLES # Use constant defined above
        
        self.movements_data_dict: Dict[str, str] = {}
        self.movement_names_list: List[str] = []
        self.movement_list_for_prompt_str: str = ""
        self._load_movements_data() # Load and parse movements.txt

        self.master_template = self._load_master_template()
        self.consolidation_template = self._load_consolidation_template()

    def _load_movements_data(self) -> None:
        """
        Reads the movement list from input/movements.txt, parses it into a dictionary
        (name -> details with date), a list of names, and a formatted string for prompts.
        """
        movements_dict = {}
        movement_lines_for_prompt = []
        try:
            # Assuming input/movements.txt is relative to the execution directory
            # Consider making this path configurable or more robust if issues arise
            with open("input/movements.txt", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    match = re.match(r"^(.*?)\\s*\\((.*)\\)$", line)
                    if match:
                        name, details = match.groups()
                        movements_dict[name.strip()] = details.strip()
                        movement_lines_for_prompt.append(f"* {name.strip()} ({details.strip()})")
                    else:
                        # Handle lines that might not fit the expected "Name (Details)" format,
                        # though movements.txt is expected to follow this.
                        # For now, we'll just use the line as a name if no parentheses.
                        if '(' not in line and ')' not in line: # Simple check
                             movements_dict[line] = "" # No details
                             movement_lines_for_prompt.append(f"* {line}")
                        else:
                             logging.warning(f"[PromptTemplate] Could not parse movement line: {line}")
            
            self.movements_data_dict = movements_dict
            self.movement_names_list = list(movements_dict.keys())
            self.movement_list_for_prompt_str = "\\n".join(movement_lines_for_prompt)
            logging.info(f"[PromptTemplate] Loaded {len(self.movements_data_dict)} movements from movements.txt.")

        except Exception as e:
            logging.error(f"[PromptTemplate] Failed to load or parse movement list: {e}")
            # Fallback to empty structures
            self.movements_data_dict = {}
            self.movement_names_list = []
            self.movement_list_for_prompt_str = ""

    def _load_master_template(self) -> str:
        """Loads the finalized master vision prompt from Prompt Design Specifications."""
        style_list_str = ", ".join(self.predefined_styles)
        # Use self.movement_list_for_prompt_str which is already formatted
        return (
            "Analyze the image and filename tokens provided. Extract the following fields:\\n"
            "1. author\\n"
            "2. title\\n"
            "3. date\\n"
            "4. nationality\\n"
            f"5. style (from the predefined list: {style_list_str})\\n"
            "6. movement (from the predefined list below, including their typical date ranges):\\n" # Clarified instruction
            f"{self.movement_list_for_prompt_str}\\n"
            ")\\n"
            "7. primary_subjects\\n"
            "8. secondary_subjects\\n"
            "9. brief_description\\n"
            "10. confidence_score\\n"
            "11. grounding_used\\n"
            "12. limitations\\n"
            "\\n"
            "Validate the identified 'date' against the time range associated with the identified 'movement' from the provided list. Ensure the date falls within the movement's time range.\\n"
            "\\n"
            "Output the results in JSON format only.\\n"
            "\\n"
            "Additional instructions for the model: {model_specific_instructions}\\n"
            "\\n"
            "Filename tokens: {filename_tokens}\\n"
        )

    def _load_consolidation_template(self) -> str:
        """Loads the finalized consolidation prompt from Prompt Design Specifications."""
        # Use self.movement_list_for_prompt_str which is already formatted
        return (
            "Consolidate the results from multiple vision models provided as a JSON string. "
            "Reference the 'movement' field and use the provided movement list (including their typical date ranges) below to validate "
            "the consolidated 'date' against the consolidated 'movement'\\'s time range. "
            "If there are conflicts or low confidence, choose the most plausible movement.\\n"
            f"{self.movement_list_for_prompt_str}\\n"
            "\\n"
            "Output the consolidated results in JSON format only.\\n"
        )
    def get_vision_prompt(self, model_name: str, filename_tokens: str) -> str:
        """
        Generates a prompt customized for a specific vision model and input.
        """
        if not isinstance(model_name, str) or not model_name:
            logging.error("[PromptTemplate] Invalid model_name provided for vision prompt.")
            model_name = "unknown_model"
        if not isinstance(filename_tokens, str):
            logging.warning(f"[PromptTemplate] Invalid filename_tokens type ({type(filename_tokens)}). Converting to empty string.")
            filename_tokens = ""
        model_specific_instructions = f"Note: This analysis is being performed by vision model {model_name}."
        if "gemini" in model_name.lower() or "grok" in model_name.lower():
            model_specific_instructions += "\\nPlease utilize web search grounding or internal knowledge if available to improve accuracy."
        if "qwen" in model_name.lower():
            model_specific_instructions += "\\nProvide detailed analysis based on the image."
        
        # The master_template is already formatted with the necessary lists in its definition
        prompt = self.master_template.format(
            # style_list is embedded via style_list_str in _load_master_template
            # movement_list is embedded via self.movement_list_for_prompt_str in _load_master_template
            filename_tokens=filename_tokens,
            model_specific_instructions=model_specific_instructions
        )
        return prompt

    def get_consolidation_prompt(self, results_json_string: str) -> str:
        """Creates the prompt for the consolidation model using the finalized template."""
        return (
            f"{self.consolidation_template}\\n"
            f"Input Data (JSON string containing results from various models):\\n{results_json_string}\\n"
        )

# --- Main Research Agent Class (Refactored) ---
class ResearchAgent:
    """
    Researches artwork using multiple vision LLMs concurrently and consolidates findings.
    Refactored (2025-05-05) for unified vision workflow, new models, and parallel processing.
    """

    def _think_with_grok(self, thought: str) -> str:
        """
        USER REQUIREMENT: Internal 'thinking' step using grok-3-mini-fast-high-beta.
        This method sends the agent's reasoning or validation prompt to the Grok model and returns the response.
        """
        import openai
        api_key = self.config.get("GROK_API_KEY") or os.environ.get("GROK_API_KEY")
        if not api_key:
            logging.warning("[ResearchAgent] No GROK_API_KEY found for Grok thinking step.")
            return ""
        try:
            client = openai.OpenAI(api_key=api_key, base_url=XAI_BASE_URL)
            response = client.chat.completions.create(
                model="grok-3-mini-fast-high-beta",
                messages=[
                    {"role": "system", "content": "You are an expert research agent reasoning about your next step."},
                    {"role": "user", "content": thought}
                ],
                max_tokens=256,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"[ResearchAgent] Grok thinking step failed: {e}")
            return ""

    def __init__(self, config: Optional[Dict] = None, vision_agent_classes: Optional[Dict] = None):
        """
        Initialize the ResearchAgent with configuration, ModelRegistry, and PromptTemplate.
        """
        self.config = config if config is not None else {}
        self.agent_id = random.randint(10000, 99999)
        self.vision_agent_classes = vision_agent_classes # Retained for DocentAgent compatibility if needed elsewhere

        logging.info(f"[ResearchAgent {self.agent_id}] Initializing...")

        # --- Initialize Core Components ---
        self.model_registry = ModelRegistry(self.config, self.agent_id)
        self.prompt_template = PromptTemplate()

        # --- Configuration ---
        self.max_concurrent_calls = self.config.get('max_concurrent_calls', 5) # Limit parallel calls
        self.api_call_timeout = self.config.get('api_call_timeout', 120) # Timeout for individual LLM calls (seconds)
        self.max_retries = self.config.get('max_retries', 2) # Retries = max_retries + 1 initial attempt = 3 total attempts

        # --- Readiness Check ---
        available_vision_models = self.model_registry.get_models_by_capability('vision')
        consolidation_model_available = bool(self.model_registry.get_client_for_model(CONSOLIDATION_MODEL))

        if available_vision_models and consolidation_model_available:
            logging.info(f"[ResearchAgent {self.agent_id}] Initialization complete. Available Vision Models: {len(available_vision_models)}. Consolidation Model ({CONSOLIDATION_MODEL}) available.")
        else:
            logging.warning(f"[ResearchAgent {self.agent_id}] Initialization potentially incomplete. Available Vision Models: {len(available_vision_models)}. Consolidation Model Available: {consolidation_model_available}. Check API keys and config.")
            if not available_vision_models:
                 logging.warning(f"[ResearchAgent {self.agent_id}] No vision models seem to be available. Research will likely fail.")
            if not consolidation_model_available:
                 logging.warning(f"[ResearchAgent {self.agent_id}] Consolidation model ({CONSOLIDATION_MODEL}) unavailable. Consolidation step will fail.")

    # --- Public Method ---
# --- Compatibility Layer (Added 2025-05-05) ---
    def research_and_process(self, image_path: str) -> Dict[str, Any]:
        """
        Compatibility method for older callers (like DocentAgent) that expect
        'research_and_process'. This method simply calls the new unified
        research workflow.
    
        Args:
            image_path: Path to the image file.
    
        Returns:
            A dictionary containing 'research_attempts' (formerly 'individual_results')
            and 'consolidated_findings' (formerly 'consolidated_results')
            from the unified research process.
        """
        self._think_with_grok(f"About to process research for image: {os.path.basename(image_path)}. What should I consider before starting?")
        logging.info(f"[ResearchAgent {self.agent_id}] Called compatibility method 'research_and_process' for {os.path.basename(image_path)}. Redirecting to 'unified_research'.")
        # Call the main unified research method and return its output directly
        return self.unified_research(image_path)
    def categorize_artwork(self, image_path: str) -> str:
        """
        Categorize artwork by style, refining the movement using Grok and the movement list.
        """
        self._think_with_grok(f"About to categorize artwork for image: {os.path.basename(image_path)}. What reasoning steps should I take?")
        logging.info(f"[ResearchAgent {getattr(self, 'agent_id', 'N/A')}] Categorizing {os.path.basename(image_path)} using unified research...")

        # Run unified research as before
        research_output = self.unified_research(image_path)
        consolidated_results = research_output.get("consolidated_results", {}) # Ensure this key matches actual output
        if not consolidated_results and "consolidated_findings" in research_output: # Check for new key
            consolidated_results = research_output.get("consolidated_findings", {})

        style = consolidated_results.get("style", "unknown")
        movement = consolidated_results.get("movement", None)
        date = consolidated_results.get("date", None)

        # Use the movement list string directly from PromptTemplate
        # This string is already formatted as "* Movement Name (Details)\\n..."
        movement_list_for_grok_prompt = self.prompt_template.movement_list_for_prompt_str
        
        # Also get the plain list of movement names for parsing Grok's response
        valid_movement_names = self.prompt_template.movement_names_list

        # Compose prompt for Grok to check movement/date consistency
        grok_prompt = (
            f"Given the following artwork metadata:\\n"
            f"- Date: {date}\\n"
            f"- Movement: {movement}\\n"
            f"Here is a list of recognized art movements and their time ranges:\\n"
            f"{movement_list_for_grok_prompt}\\n" # Use the formatted list from PromptTemplate
            "\\n\\n"
            "Analyze whether the provided date fits the movement's time range. "
            "If not, suggest the most appropriate movement from the list above based on the date. "
            "Return only the final movement string (from the list above) that best fits the date, or the original movement if it is consistent."
        )

        grok_response = self._think_with_grok(grok_prompt)
        final_movement = movement # Default to original movement

        if grok_response:
            grok_response_clean = grok_response.strip().lower()
            # Try to find a direct match from the valid movement names in Grok's response
            # This makes parsing more robust by checking against known movement names.
            best_match = None
            for m_name in valid_movement_names:
                if m_name.lower() in grok_response_clean:
                    # Simple substring check. Could be improved with fuzzy matching if needed.
                    if best_match is None or len(m_name) > len(best_match): # Prefer longer match if multiple
                        best_match = m_name
            
            if best_match:
                final_movement = best_match
                logging.info(f"[ResearchAgent] Grok refined movement to: {final_movement} based on response: '{grok_response_clean}'")
            else:
                logging.warning(f"[ResearchAgent] Could not clearly extract a movement from Grok's response: '{grok_response_clean}'. Using original: {movement}")
        else:
            logging.warning(f"[ResearchAgent] Grok response was empty for movement refinement. Using original: {movement}")


        # Update consolidated_results with the final movement
        # This assumes consolidated_results is a mutable dictionary that might be used later.
        # If research_output is what's returned and used, update it there.
        # For now, let's assume consolidated_results is the primary dict to update.
        if isinstance(consolidated_results, dict):
            consolidated_results["movement"] = final_movement
        
        # The method is expected to return the 'style' string
        return style

    # --- Core Unified Workflow ---
    def unified_research(self, image_path: str) -> Dict[str, Any]:
        """
        Performs the unified research workflow: preprocess, parallel vision search, consolidate.
        Returns a dictionary containing individual model results and the final consolidated data.
        """
        tag = f"[ResearchAgent {self.agent_id}][Unified Research]"
        self._think_with_grok(f"Planning unified research workflow for image: {os.path.basename(image_path)}. What steps should I follow and how will I validate the results?")
        logging.info(f"{tag} Starting for image: {os.path.basename(image_path)}")

        # 1. Preprocess Input
        preprocess_result = self._preprocess_input(image_path)
        if "error" in preprocess_result:
            logging.error(f"{tag} Preprocessing failed: {preprocess_result['error']}")
            # Return structure consistent with final output, indicating failure early
            return {
                "research_attempts": {},
                "consolidated_findings": {"error": f"Preprocessing failed: {preprocess_result['error']}", "confidence_score": 0.0, "grounding_used": False, "artist": None, "title": None, "date": None, "description": f"Preprocessing failed: {preprocess_result['error']}", "style": None, "subjects": [], "structured_sentence": ""}
            }
        image_bytes = preprocess_result["image_bytes"]
        mime_type = preprocess_result["mime_type"]
        filename_tokens = preprocess_result["filename_tokens"]

        # 2. Execute Parallel Searches (Vision Models Only)
        vision_models = self.model_registry.get_models_by_capability('vision')
        if not vision_models:
             logging.error(f"{tag} No vision models available in registry. Cannot perform research.")
             return {
                 "research_attempts": {},
                 "consolidated_findings": {"error": "No vision models available", "confidence_score": 0.0, "grounding_used": False, "artist": None, "title": None, "date": None, "description": "No vision models available", "style": None, "subjects": [], "structured_sentence": ""}
             }

        logging.info(f"{tag} Executing parallel searches for {len(vision_models)} vision models...")
        individual_results = self._execute_parallel_searches(vision_models, image_path, image_bytes, mime_type, filename_tokens) # Pass image_path

        # 3. Consolidate Results
        self._think_with_grok("About to consolidate results from multiple vision models. How should I validate and synthesize the findings for completeness and quality?")
        logging.info(f"{tag} Consolidating results from {len(individual_results)} individual calls...")
        consolidated_results = self._consolidate_results(individual_results)

        # 4. Format and Return Output
        output_data = {
            "research_attempts": individual_results, # Changed key
            "consolidated_findings": consolidated_results # Changed key
        }
        
        # Save findings to markdown log file
        if consolidated_results and not ("error" in consolidated_results and consolidated_results.get("confidence_score", 1.0) == 0.0):
            self._save_findings_to_markdown_file(consolidated_results, image_path)

        logging.info(f"{tag} Workflow completed for {os.path.basename(image_path)}.")
        logging.debug(f"{tag} Final Output Data: {json.dumps(output_data, indent=2)}") # Log full output for debugging
        return output_data

    # --- Helper Methods ---
    def _format_findings_to_markdown(self, findings: Dict, image_path: str) -> str:
        """Formats the research findings into a markdown string."""
        image_basename = os.path.basename(image_path)
        md_lines = [f"# Research Agent Findings for {image_basename}\\n"]
        md_lines.append("## Consolidated Report\\n")

        # Define an order for keys to make the report consistent
        ordered_keys = [
            "author", "title", "date", "nationality", "style", "movement", "genre",
            "primary_subjects", "secondary_subjects", "brief_description",
            "confidence_score", "grounding_used", "limitations", "error"
        ]

        for key in ordered_keys:
            if key not in findings:
                continue # Skip keys not present in the findings

            value = findings[key]
            
            if key == "error" and not value: # Don't show empty error field
                continue

            if value is None:
                formatted_value = "_N/A_"
            elif isinstance(value, list):
                if not value:
                    formatted_value = "_N/A_"
                else:
                    list_items = "\\n".join([f"- {item}" for item in value])
                    formatted_value = f"\\n{list_items}"
            elif isinstance(value, bool):
                formatted_value = str(value)
            elif isinstance(value, (float, int)):
                 if key == "confidence_score":
                    formatted_value = f"{value:.2f}" # Format confidence score
                 else:
                    formatted_value = str(value)
            else: # string or other
                formatted_value = str(value).strip()
                if not formatted_value:
                    formatted_value = "_N/A_"
            
            display_key = key.replace("_", " ").title()
            md_lines.append(f"**{display_key}:** {formatted_value}\\n")
        
        return "\\n".join(md_lines)

    def _save_findings_to_markdown_file(self, findings: Dict, image_path: str):
        """Saves the formatted findings to a markdown file in the logs directory."""
        tag = f"[ResearchAgent {self.agent_id}][Markdown Log]"
        
        # Double check if we should skip logging (already checked before calling, but good for standalone use)
        if not findings or ("error" in findings and findings.get("confidence_score", 1.0) == 0.0 and not any(v for k, v in findings.items() if k != "error")):
            logging.info(f"{tag} Skipping markdown log generation due to error or effectively empty findings for {os.path.basename(image_path)}.")
            return

        try:
            markdown_content = self._format_findings_to_markdown(findings, image_path)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_basename_no_ext = os.path.splitext(os.path.basename(image_path))[0]
            
            # Sanitize image_basename_no_ext for use in filename
            safe_image_basename = re.sub(r'[^a-zA-Z0-9_-]', '_', image_basename_no_ext)

            log_filename = f"research_agent_findings_{timestamp}_{safe_image_basename}.md"
            
            # Ensure logs directory exists
            logs_dir = "logs"
            os.makedirs(logs_dir, exist_ok=True)
            
            log_filepath = os.path.join(logs_dir, log_filename)

            with open(log_filepath, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            logging.info(f"{tag} Successfully saved findings to {log_filepath}")
        except Exception as e:
            logging.error(f"{tag} Failed to save findings for {os.path.basename(image_path)} to markdown: {e}", exc_info=True)

    def _preprocess_input(self, image_path: str) -> Dict[str, Any]:
        """Validates input, extracts filename tokens, loads image bytes, and gets MIME type."""
        tag = f"[ResearchAgent {self.agent_id}][Preprocess]"
        try:
            # Basic file checks
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            if os.path.getsize(image_path) == 0:
                raise ValueError(f"Empty file detected: {image_path}")

            # MIME type and format check
            mime_type = self._get_mime_type(image_path)
            if mime_type == 'application/octet-stream': # Means extension wasn't recognized
                 ext = os.path.splitext(image_path)[1].lower()
                 raise ValueError(f"Unsupported file extension: {ext}")

            # Load image bytes
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()
            if not image_bytes:
                 raise ValueError("Failed to read image bytes (file might be empty after check).") # Should be caught earlier

            # Extract filename tokens
            filename_tokens = os.path.splitext(os.path.basename(image_path))[0].replace("_", " ").replace("-", " ")
            logging.info(f"{tag} Successfully processed: {os.path.basename(image_path)} ({len(image_bytes)} bytes, type: {mime_type}). Tokens: '{filename_tokens}'")

            return {
                "image_bytes": image_bytes,
                "mime_type": mime_type,
                "filename_tokens": filename_tokens
            }
        except Exception as e:
            logging.error(f"{tag} Error during preprocessing {image_path}: {e}", exc_info=True)
            return {"error": str(e)}

    def _get_mime_type(self, image_path: str) -> str:
        """Determines the MIME type based on file extension."""
        ext = os.path.splitext(image_path)[1].lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            # Add more if needed
        }
        return mime_map.get(ext, "application/octet-stream") # Default if extension unknown

    def _prepare_model_call_args(self, model_name: str, image_path: str, image_bytes: bytes, mime_type: str, filename_tokens: str) -> Optional[Dict]:
        """Prepares the prompt and content arguments for a specific model call."""
        tag = f"[ResearchAgent {self.agent_id}][Prepare Args {model_name}]"
        model_details = self.model_registry.get_model_details(model_name)
        client = self.model_registry.get_client_for_model(model_name)

        if not model_details or not client:
            logging.error(f"{tag} Cannot prepare args. Model details or client unavailable.")
            return None

        # Generate prompt using the template system
        prompt_text = self.prompt_template.get_vision_prompt(model_name, filename_tokens)
        content = None

        try:
            # --- Format content based on client type ---
            if model_details['provider'] in ['grok', 'openrouter']: # OpenAI API format
                base64_image = base64.b64encode(image_bytes).decode("utf-8")
                content = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
                            },
                        ],
                    }
                ]
            elif model_details['provider'] == 'google':
                # Distinguish between Vertex AI GenerativeModel instance and standard google.generativeai module
                if isinstance(client, vertex_genai.GenerativeModel): # Check if it's a Vertex AI model instance
                    # Vertex AI format: expects Part object
                    image_part = Part.from_data(data=image_bytes, mime_type=mime_type)
                    content = [prompt_text, image_part]
                elif client is google_genai: # Check if it's the standard google.generativeai module
                    # Standard Gemini API: expects prompt and image_bytes directly
                    content = [prompt_text, image_bytes]
                    # Perform type check here for standard google.generativeai content
                    for c_item in content:
                        if not _is_supported_google_genai_content_type(c_item): # Part objects are not supported
                            logging.error(f"{tag} Invalid content type '{type(c_item).__name__}' for standard Google GenerativeAI model '{model_name}'. Supported types: bytes, str, dict, PIL.Image, IPython.Image, Blob.")
                            return None
                else:
                    # Fallback or error for unexpected client type under 'google' provider
                    logging.error(f"{tag} Unexpected client type '{type(client).__name__}' for Google provider model '{model_name}'. Cannot prepare content.")
                    return None
            else:
                logging.error(f"{tag} Unsupported provider '{model_details['provider']}' for content formatting.")
                return None

            return {
                "client": client,
                "model_name": model_name,
                "image_path": image_path, # Add image_path
                "content": content,
                "is_json_output": True, # All vision calls expect JSON
                "max_tokens": 2000, # Default, can be adjusted
                "temperature": 0.1 # Default, can be adjusted
            }
        except Exception as e:
            logging.error(f"{tag} Failed to prepare content (e.g., base64 encoding, Part creation): {e}", exc_info=True)
            return None


    def _execute_parallel_searches(self, model_names: List[str], image_path: str, image_bytes: bytes, mime_type: str, filename_tokens: str) -> Dict[str, Dict]:
        """Executes LLM calls for the given models in parallel using ThreadPoolExecutor."""
        tag = f"[ResearchAgent {self.agent_id}][Parallel Search]"
        results = {}
        # Use ThreadPoolExecutor for I/O-bound tasks (API calls)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrent_calls) as executor:
            # Prepare future tasks
            future_to_model = {}
            for model_name in model_names:
                call_args = self._prepare_model_call_args(model_name, image_path, image_bytes, mime_type, filename_tokens) # Pass image_path
                if call_args:
                    # Submit the task to the executor, calling _make_llm_call_with_retry
                    # Ensure image_path is included in the arguments passed to the submit call
                    future = executor.submit(self._make_llm_call_with_retry, **call_args)
                    future_to_model[future] = model_name
                else:
                    # Store error if args couldn't be prepared
                    results[model_name] = {"error": f"Failed to prepare arguments for model {model_name}"}
                    logging.warning(f"{tag} Skipping call for {model_name} due to argument preparation failure.")

            # Process completed futures
            for future in concurrent.futures.as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    # Get the result from the future (the dictionary returned by _make_llm_call_with_retry)
                    result_data = future.result(timeout=self.api_call_timeout + 30) # Add buffer to timeout
                    results[model_name] = result_data
                    if isinstance(result_data, dict) and "error" in result_data:
                         logging.warning(f"{tag} Call failed for {model_name} after retries: {result_data['error']}")
                    else:
                         logging.info(f"{tag} Call successful for {model_name}.")
                         logging.debug(f"{tag} Result for {model_name}: {result_data}")

                except concurrent.futures.TimeoutError:
                    logging.error(f"{tag} Call timed out for model {model_name} after {self.api_call_timeout}s (plus buffer).")
                    results[model_name] = {"error": f"API call timed out after {self.api_call_timeout}s"}
                except Exception as exc:
                    logging.error(f"{tag} Call generated an exception for model {model_name}: {exc}", exc_info=True)
                    results[model_name] = {"error": f"Exception during API call: {str(exc)}"}

        logging.info(f"{tag} Finished parallel execution. Got results for {len(results)}/{len(model_names)} models.")
        return results

    def _consolidate_results(self, individual_results: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Consolidates results from multiple vision models using the CONSOLIDATION_MODEL.
        """
        tag = f"[ResearchAgent {self.agent_id}][Consolidate Results]"
        logging.info(f"{tag} Starting consolidation of {len(individual_results)} individual results.")

        if not individual_results:
            logging.warning(f"{tag} No individual results to consolidate.")
            return self._create_error_json("No individual results to consolidate.")

        valid_results_for_consolidation = {
            model: res for model, res in individual_results.items()
            if isinstance(res, dict) and "error" not in res
        }

        if not valid_results_for_consolidation:
            all_errors = [res.get("error", "Unknown error") for res in individual_results.values() if isinstance(res, dict)]
            error_summary = f"All vision models returned errors or no valid data. Details: {'; '.join(all_errors)[:500]}"
            logging.warning(f"{tag} {error_summary}")
            return self._create_error_json(error_summary)

        results_to_consolidate_json = json.dumps(valid_results_for_consolidation, indent=2)
        consolidation_prompt = self.prompt_template.get_consolidation_prompt(results_to_consolidate_json)

        consolidation_client = self.model_registry.get_client_for_model(CONSOLIDATION_MODEL)
        model_details = self.model_registry.get_model_details(CONSOLIDATION_MODEL)

        if not consolidation_client or not model_details:
            logging.error(f"{tag} Consolidation model '{CONSOLIDATION_MODEL}' or its client is not available.")
            return self._create_error_json(f"Consolidation model {CONSOLIDATION_MODEL} unavailable.")

        logging.info(f"{tag} Calling consolidation model: {CONSOLIDATION_MODEL}")

        content_for_consolidation: List[Dict[str, Any]]
        if model_details['provider'] in ['grok', 'openrouter']: # OpenAI API format
            content_for_consolidation = [{"role": "user", "content": consolidation_prompt}]
        else:
            logging.error(f"{tag} Consolidation model {CONSOLIDATION_MODEL} has unsupported provider {model_details['provider']}.")
            return self._create_error_json(f"Unsupported provider for consolidation model {CONSOLIDATION_MODEL}.")

        consolidated_data = self._make_llm_call_with_retry(
            client=consolidation_client,
            model_name=CONSOLIDATION_MODEL,
            image_path="consolidation_task",
            content=content_for_consolidation,
            is_json_output=True,
            max_tokens=3000,
            temperature=0.1
        )

        if isinstance(consolidated_data, dict) and "error" in consolidated_data:
            logging.error(f"{tag} Consolidation failed: {consolidated_data['error']}")
            return consolidated_data

        if not isinstance(consolidated_data, dict):
            logging.error(f"{tag} Consolidation returned non-dict data: {type(consolidated_data)}. Response: {str(consolidated_data)[:200]}")
            return self._create_error_json(f"Consolidation model returned unexpected data type: {type(consolidated_data)}")

        self._ensure_standard_fields(consolidated_data, CONSOLIDATION_MODEL)
        logging.info(f"{tag} Consolidation successful.")
        logging.debug(f"{tag} Consolidated Data: {json.dumps(consolidated_data, indent=2)}")
        return consolidated_data

    def _make_llm_call_with_retry(self, client: Any, model_name: str, image_path: str, content: Union[str, List[Union[str, Part]], List[Dict]], is_json_output: bool = False, max_tokens: int = 2000, temperature: float = 0.2) -> Dict:
        """
        Wrapper for _make_llm_call that implements retry logic.
        Returns a dictionary, guaranteed to have 'error' key on failure.
        """
        if image_path is None or not isinstance(image_path, str) or not image_path.strip():
            raise ValueError("image_path must be a non-empty string.")
        tag = f"[ResearchAgent {self.agent_id}][Retry Call {model_name}]"
        for attempt in range(self.max_retries + 1):
            try:
                result = self._make_llm_call(
                    client=client,
                    model_name=model_name,
                    image_path=image_path, # Pass image_path
                    content=content,
                    is_json_output=is_json_output,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                # Check if the result indicates an error occurred within _make_llm_call
                if isinstance(result, dict) and "error" in result:
                    # Don't retry logical errors (like bad request, content blocked), only connection/transient issues
                    # This check could be more sophisticated based on error types/messages
                    error_msg = result.get("error", "").lower()
                    if "badrequest" in error_msg or "blocked" in error_msg or "invalid" in error_msg or "unsupported" in error_msg:
                         logging.warning(f"{tag} Attempt {attempt + 1} failed with non-retryable error: {result['error']}. Not retrying.")
                         return result # Return the error dict immediately
                    else:
                         # Log the retryable error
                         logging.warning(f"{tag} Attempt {attempt + 1} failed: {result['error']}. Retrying...")
                         # Fall through to retry logic below
                else:
                    # Success!
                    return result # Return successful result (dict or string)

            except Exception as e:
                # Catch exceptions raised directly by _make_llm_call or the client library during the call
                logging.error(f"{tag} Attempt {attempt + 1} raised exception: {e}", exc_info=True)
                # Treat unexpected exceptions as potentially retryable
                result = {"error": f"Exception during API call attempt {attempt + 1}: {str(e)}"}
                # Fall through to retry logic

            # --- Retry Logic ---
            if attempt < self.max_retries:
                wait_time = (2 ** attempt) + random.uniform(0, 1) # Exponential backoff with jitter
                logging.info(f"{tag} Waiting {wait_time:.2f} seconds before retry {attempt + 2}...")
                time.sleep(wait_time)
            else:
                logging.error(f"{tag} Call failed after {self.max_retries + 1} attempts.")
                # Return the error from the last attempt
                return result if isinstance(result, dict) else {"error": "Unknown error after max retries"}

        # Should not be reachable, but as a fallback:
        return {"error": "Max retries reached, final attempt failed."}


    def _make_llm_call(self, client: Any, model_name: str, image_path: str, content: Union[str, List[Union[str, Part]], List[Dict]], is_json_output: bool = False, max_tokens: int = 2000, temperature: float = 0.2) -> Union[str, Dict]:
        """
        Generic helper to make LLM calls with robust error handling and JSON parsing.
        Handles OpenAI/xAI/OpenRouter (expects content=List[Dict]) and Vertex AI GenerativeModel (expects content=Union[str, List[Union[str, Part]]]).
        Returns parsed JSON dictionary (with standard fields ensured) if is_json_output=True, otherwise raw text string.
        Returns a dictionary with an 'error' key on failure.
        """
        tag = f"[ResearchAgent {self.agent_id}][LLM Call {model_name}]"
        filename = os.path.basename(image_path) # Extract filename for logging context
        if not client:
            # This check might be redundant if called via _make_llm_call_with_retry which gets client from registry
            logging.warning(f"{tag} Client not available.")
            return {"error": f"Client for model {model_name} not available."}

        # Check Vertex AI SDK readiness specifically (client might be True placeholder)
        if isinstance(client, bool) and client is True and model_name in self.model_registry.models and self.model_registry.models[model_name]['client_type'] == 'vertexai':
             # Need to get the actual instantiated model
             client = self.model_registry.get_client_for_model(model_name)
             if not client:
                  logging.error(f"{tag} Failed to get instantiated Vertex AI client for {model_name}.")
                  return {"error": f"Vertex AI client for {model_name} could not be instantiated."}
        elif isinstance(client, bool): # Should not happen if logic is correct
             logging.error(f"{tag} Invalid client placeholder encountered.")
             return {"error": "Invalid client state."}


        logging.debug(f"{tag} Attempting call. JSON output: {is_json_output}")
        raw_response_text = None # Store raw text for debugging
        result = None # Initialize result

        try:
            # --- Handle OpenAI/xAI/OpenRouter Client (duck typing) ---
            if hasattr(client, 'chat') and hasattr(client.chat, 'completions') and hasattr(client.chat.completions, 'create'):
                if not isinstance(content, list) or not all(isinstance(item, dict) for item in content):
                    #logging.error(f"{tag} Invalid content format for OpenAI-compatible client. Expected list of dicts (messages). Got: {type(content)}")
                    return {"error": "Invalid content format for OpenAI-compatible call."}

                response_format_config = {"type": "json_object"} if is_json_output else None
                try:
                    completion = client.chat.completions.create(
                        messages=content,
                        model=model_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format=response_format_config
                    )
                    if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
                         result = completion.choices[0].message.content
                    else:
                         finish_reason = completion.choices[0].finish_reason if completion.choices else "unknown"
                         logging.warning(f"{tag} OpenAI-compatible response structure invalid or empty content. Finish Reason: {finish_reason}.")
                         result = None # Ensure result is None if content is missing

                except openai.BadRequestError as e:
                     # Log specific details if available
                     error_body = e.body.get('message', str(e)) if e.body else str(e)
                     logging.error(f"{tag} OpenAI BadRequestError: {error_body}", exc_info=False) # Less verbose logging for bad requests
                     return {"error": f"OpenAI BadRequestError: {error_body}"}
                # Catch other potential OpenAI errors (RateLimitError, APIError, etc.)
                except openai.OpenAIError as e:
                     logging.error(f"{tag} OpenAI API error: {e}", exc_info=True)
                     return {"error": f"OpenAI API error: {type(e).__name__} - {str(e)}"}
                except Exception as e: # Catch unexpected errors during the call
                     logging.error(f"{tag} Unexpected error during OpenAI-compatible call: {e}", exc_info=True)
                     return {"error": f"Unexpected error during OpenAI-compatible call: {str(e)}"}


            # --- Handle Google GenerativeAI Client (Standard API) ---
            # Check if the client is the imported module and the model is a standard Gemini one
            elif client is google_genai and model_name in [GEMINI_IMAGE_SEARCH_MODEL, FLASH_IMAGE_SEARCH_MODEL]:
                 if not isinstance(content, list): # Expects [prompt, image_bytes]
                      #logging.error(f"{tag} Invalid content format for Google GenerativeAI client. Expected list [prompt, image_bytes]. Got: {type(content)}")
                      return {"error": "Invalid content format for Google GenerativeAI call."}
                 # Final type check for Google GenerativeAI
                 for c in content:
                     if type(c).__name__ == "Part" or not _is_supported_google_genai_content_type(c):
                         return {"error": f"Invalid content type for Google GenerativeAI: {type(c)}. Must be bytes, dict, PIL.Image.Image, IPython.display.Image, or Blob."}
                 try:
                      # Instantiate the model here using the module
                      genai_model = client.GenerativeModel(model_name) # Use the passed model_name
                      # Prepare generation config (adjust as needed for google.generativeai)
                      generation_config = {
                          "temperature": temperature,
                          "max_output_tokens": max_tokens,
                          # JSON mode might need specific handling if required by google.generativeai API
                      }
                      # Make the call
                      response = genai_model.generate_content(
                          contents=content, # Should be [prompt, image_bytes]
                          generation_config=generation_config,
                          # stream=False # Ensure non-streaming
                      )

                      # --- Process Google GenerativeAI Response ---
                      # Similar processing logic to Vertex AI, check response structure
                      if not response.candidates:
                           # Check for blocking reasons (structure might differ slightly from Vertex)
                           block_reason = response.prompt_feedback.block_reason.name if hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason else 'N/A'
                           safety_ratings_str = ", ".join([f"{rating.category.name}: {rating.probability.name}" for rating in response.prompt_feedback.safety_ratings]) if hasattr(response.prompt_feedback, 'safety_ratings') and response.prompt_feedback.safety_ratings else 'N/A'
                           logging.warning(f"{tag} Google GenerativeAI response was empty or blocked. Block Reason: {block_reason}, Safety Ratings: [{safety_ratings_str}]")
                           return {"error": f"Blocked response from {model_name}. Reason: {block_reason}"}

                      candidate = response.candidates[0]
                      # Use getattr for safer access to finish_reason and potential name attribute
                      finish_reason_enum = getattr(candidate, 'finish_reason', None)
                      finish_reason = getattr(finish_reason_enum, 'name', 'UNKNOWN')


                      if finish_reason not in ["STOP", "MAX_TOKENS"]:
                           logging.warning(f"{tag} Google GenerativeAI response finished with reason: {finish_reason}. Content might be incomplete or missing.")
                           # Check content existence more carefully
                           text_content_exists = (
                               hasattr(candidate, 'content') and
                               hasattr(candidate.content, 'parts') and
                               len(candidate.content.parts) > 0 and
                               hasattr(candidate.content.parts[0], 'text')
                           )
                           if not text_content_exists:
                                return {"error": f"Response finished due to {finish_reason}, no usable content."}

                      # Extract text using response.text helper if available, otherwise manually
                      if hasattr(response, 'text'):
                           result = response.text
                      elif (hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and
                            len(candidate.content.parts) > 0 and hasattr(candidate.content.parts[0], 'text')):
                           result = candidate.content.parts[0].text
                      else:
                           logging.warning(f"{tag} Google GenerativeAI response has no text part despite finish reason {finish_reason}.")
                           return {"error": f"No text content found in response from {model_name}. Finish Reason: {finish_reason}"}


                 except Exception as e:
                      # Only log error type, sanitized message, agent ID, and model name. Do not log full object or traceback.
                      # Suppress verbose multi-line details (e.g., "Got a: ..." and "Value: ...") from Google GenerativeAI errors.
                      sanitized_msg = str(e).splitlines()[0] if str(e) else type(e).__name__
                      if len(sanitized_msg) > 300:
                          sanitized_msg = sanitized_msg[:300] + " ...[truncated]"
                      logging.error(f"{tag} Google GenerativeAI API call failed. Error Type: {type(e).__name__}. Message: {sanitized_msg} [Agent: {self.agent_id}, Model: {model_name}]")
                      # Check for specific API key errors if possible
                      if "API key not valid" in sanitized_msg:
                           return {"error": f"Google GenerativeAI API key not valid for {model_name}."}
                      return {"error": f"Google GenerativeAI API call failed for {model_name}: {type(e).__name__} - {sanitized_msg}"}


            # --- Handle Google Vertex AI Client ---
            # Ensure this only catches actual VertexAI GenerativeModel instances, not the module
            elif isinstance(client, vertex_genai.GenerativeModel):
                 if not isinstance(content, (str, list)):
                      #logging.error(f"{tag} Invalid content format for Vertex AI client. Expected str or list. Got: {type(content)}")
                      return {"error": "Invalid content format for Vertex AI call."}

                 generation_config = vertex_genai.GenerationConfig(
                     temperature=temperature,
                     max_output_tokens=max_tokens,
                     # Specify JSON only if requested, otherwise default to text
                     response_mime_type="application/json" if is_json_output else "text/plain"
                 )

                 try:
                      response = client.generate_content(
                          contents=content,
                          generation_config=generation_config,
                          # stream=False # Ensure non-streaming for single response
                      )

                      # --- Process Vertex AI Response ---
                      if not response.candidates:
                          # Check for blocking reasons
                          block_reason = response.prompt_feedback.block_reason.name if hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason else 'N/A'
                          safety_ratings_str = ", ".join([f"{rating.category.name}: {rating.probability.name}" for rating in response.prompt_feedback.safety_ratings]) if hasattr(response.prompt_feedback, 'safety_ratings') and response.prompt_feedback.safety_ratings else 'N/A'
                          logging.warning(f"{tag} Vertex AI response was empty or blocked. Block Reason: {block_reason}, Safety Ratings: [{safety_ratings_str}]")
                          # Return specific error about blocking
                          return {"error": f"Blocked response from {model_name}. Reason: {block_reason}"}

                      # Get the first candidate (usually only one for non-streaming)
                      candidate = response.candidates[0]

                      # Check finish reason
                      finish_reason = candidate.finish_reason.name if hasattr(candidate, 'finish_reason') and candidate.finish_reason else 'UNKNOWN'
                      if finish_reason not in ["STOP", "MAX_TOKENS"]: # Treat other reasons (SAFETY, RECITATION, OTHER) as potential issues
                           logging.warning(f"{tag} Vertex AI response finished with reason: {finish_reason}. Content might be incomplete or missing.")
                           # If content is missing due to safety, etc., it's an error state
                           if not (candidate.content and candidate.content.parts and hasattr(candidate.content.parts[0], 'text')):
                                return {"error": f"Response finished due to {finish_reason}, no usable content."}

                      # Extract text content
                      if candidate.content and candidate.content.parts and hasattr(candidate.content.parts[0], 'text'):
                          result = candidate.text # .text conveniently joins parts
                      else:
                          # This case should be less likely if finish_reason is STOP/MAX_TOKENS, but handle defensively
                          logging.warning(f"{tag} Vertex AI response has no text part despite finish reason {finish_reason}.")
                          return {"error": f"No text content found in response from {model_name}. Finish Reason: {finish_reason}"}

                 except Exception as e: # Catch errors during the Vertex AI call itself
                      logging.error(f"{tag} Vertex AI API call failed: {e}", exc_info=True)
                      return {"error": f"Vertex AI API call failed: {str(e)}"}

            else:
                 logging.error(f"{tag} Unsupported client type provided: {type(client)}")
                 return {"error": f"Unsupported client type: {type(client)}"}

            # --- Post-processing and JSON Parsing ---
            raw_response_text = result # Store raw text before parsing

            if not result: # Handle empty but successful responses
                logging.warning(f"{tag} Received empty but successful response.")
                if is_json_output:
                    # Return standard error structure for empty JSON
                    logging.warning(f"{tag} Received empty but successful response for image '{filename}'.")
                    return self._create_error_json(f"Empty response from {model_name} for image '{filename}'")
                else:
                    logging.warning(f"{tag} Received empty but successful non-JSON response for image '{filename}'.")
                    return "" # Return empty string for non-JSON empty response

            if is_json_output:
                parsed_json = None
                try:
                    # Clean potential markdown code blocks
                    result_cleaned = result.strip()
                    if result_cleaned.startswith("```json"):
                        result_cleaned = result_cleaned[7:]
                    elif result_cleaned.startswith("```"): # Handle case with just ```
                         result_cleaned = result_cleaned[3:]
                    if result_cleaned.endswith("```"):
                        result_cleaned = result_cleaned[:-3]
                    result_cleaned = result_cleaned.strip()

                    if not result_cleaned:
                         logging.warning(f"{tag} Response was empty after cleaning markdown/whitespace for image '{filename}'.")
                         return self._create_error_json(f"Empty response after cleaning from {model_name} for image '{filename}'")

                    parsed_json = json.loads(result_cleaned)

                    if not isinstance(parsed_json, dict):
                        logging.error(f"{tag} Parsed JSON is not a dictionary for image '{filename}'. Type: {type(parsed_json)}.")
                        return self._create_error_json(f"Parsed JSON is not a dictionary from {model_name} for image '{filename}'")

                    # --- JSON Field Censoring ---
                    # Iterate through the parsed JSON and replace values matching the binary pattern
                    binary_pattern = r'(?:\\.{1,10}){10,}' # Pattern for short segments separated by \
                    keys_to_censor = []
                    for key, value in parsed_json.items():
                        if isinstance(value, str) and re.search(binary_pattern, value):
                            keys_to_censor.append(key)

                    if keys_to_censor:
                        logging.warning(f"{tag} Found and censoring binary-like data in fields: {keys_to_censor} for image '{filename}'")
                        for key in keys_to_censor:
                            parsed_json[key] = "<binary_data_field_removed>"
                    # --- End JSON Field Censoring ---

                    # Ensure standard fields exist (confidence, grounding, subjects)
                    self._ensure_standard_fields(parsed_json, model_name)
                    return parsed_json # Return the validated/completed dictionary

                except json.JSONDecodeError as json_e:
                    error_msg = f"Failed to parse JSON response from {model_name} for image '{filename}'. Error: {json_e}"
                    logging.error(f"{tag} {error_msg}. Context: Error occurred processing image '{filename}'.", exc_info=False) # Log context instead of raw response
                    return self._create_error_json(error_msg)
            else:
                # Return raw text if JSON output was not requested
                return result

        except Exception as e:
            # Catch-all for unexpected errors during the process
            # Censor the raw response before including it in the log context
            censored_response_snippet = self._censor_response_text(raw_response_text)
            error_context = f" Raw response snippet (censored): '{censored_response_snippet}'" if censored_response_snippet else ""
            logging.error(f"{tag} Unexpected error during LLM call processing for image '{filename}': {e}.{error_context}", exc_info=True)
            if is_json_output:
                return self._create_error_json(f"Unexpected error in _make_llm_call for {model_name} processing image '{filename}': {str(e)}")
            else:
                # For non-JSON, still return an error dict for consistency in retry logic
                return {"error": f"Unexpected error in _make_llm_call for {model_name}: {str(e)}"}

    def _censor_response_text(self, text: Optional[str]) -> Optional[str]:
        """
        Placeholder for censoring sensitive information in raw response text for logging.
        Currently, it just returns a snippet or a generic message if binary-like data is detected.
        """
        if not text:
            return None
        # Basic heuristic for binary-like data (multiple backslashes with short segments)
        binary_pattern = r'(?:\\.{1,10}){10,}'
        if re.search(binary_pattern, text):
            return "[Potential binary data detected and censored in log]"
        # Return a snippet for long text
        return text[:200] + "..." if len(text) > 200 else text

    def _create_error_json(self, error_message: str) -> Dict:
        """Helper to create a standardized error dictionary for JSON responses."""
        error_dict = {
            "error": error_message,
            "confidence_score": 0.0, # Default confidence for errors
            "grounding_used": False, # Default grounding for errors
            "author": None,
            "title": None,
            "date": None,
            "nationality": None,
            "style": None,
            "genre": None,
            "primary_subjects": [],
            "secondary_subjects": [],
            "brief_description": None,
            "limitations": "Error occurred during processing." # Default limitation for errors
        }
        # Removed raw_response inclusion
        return error_dict

    def _ensure_standard_fields(self, data: Dict, model_name: str):
        """Ensures standard fields exist and performs basic validation/type correction."""
        tag = f"[ResearchAgent {self.agent_id}][Ensure Fields {model_name}]"
        # Regex for basic date patterns (YYYY, c. YYYY, YYYYs, YYYY-YYYY)
        date_pattern = re.compile(r"^(c\.\s?)?\d{4}(s|'s)?(-\d{4})?$")
        defaults = {
            "confidence_score": 0.5, # Default to neutral confidence if missing
            "grounding_used": False, # Default to false if missing
            "primary_subjects": [],
            "secondary_subjects": [],
            "author": None,
            "title": None,
            "date": None,
            "nationality": None,
            "style": None,
            "genre": None,
            "brief_description": None,
            "limitations": None # Default to null if missing
        }
        for key, default_value in defaults.items():
            if key not in data:
                data[key] = default_value
                logging.debug(f"{tag} Added default {key}={default_value}")

            # --- Enhanced Validation ---
            value = data.get(key) # Use .get() for safety, though loop ensures key exists if not added

            if key == "confidence_score":
                original_value = value
                try:
                    score = float(value)
                    if not (0.0 <= score <= 1.0):
                        logging.warning(f"{tag} Confidence score '{original_value}' out of range [0.0, 1.0]. Clamping to {max(0.0, min(1.0, score))}.")
                        data[key] = max(0.0, min(1.0, score)) # Clamp to range
                    else:
                        data[key] = score # Store valid float
                except (ValueError, TypeError):
                    logging.warning(f"{tag} Invalid type/value for confidence_score ('{original_value}', type {type(value)}). Setting to default 0.0.")
                    data[key] = 0.0

            elif key == "grounding_used":
                # Accept bool, int, str representations of True/False
                if isinstance(value, bool):
                    data[key] = value
                elif isinstance(value, int):
                    data[key] = bool(value)
                    logging.warning(f"{tag} Coerced int '{value}' to bool for grounding_used.")
                elif isinstance(value, str):
                    val_lower = value.strip().lower()
                    if val_lower in ("true", "yes", "1"):
                        data[key] = True
                        logging.warning(f"{tag} Coerced string '{value}' to True for grounding_used.")
                    elif val_lower in ("false", "no", "0", ""):
                        data[key] = False
                        logging.warning(f"{tag} Coerced string '{value}' to False for grounding_used.")
                    else:
                        data[key] = False
                        logging.warning(f"{tag} Unrecognized string '{value}' for grounding_used. Set to False.")
                else:
                    logging.warning(f"{tag} Invalid type for grounding_used ('{value}', type {type(value)}). Setting to False.")
                    data[key] = False

            elif key in ["primary_subjects", "secondary_subjects"]:
                if not isinstance(value, list):
                    logging.warning(f"{tag} Correcting invalid type for {key} ('{value}', type {type(value)}). Setting to [].")
                    data[key] = []
                else: # Ensure all elements in list are strings
                    corrected_list = [str(item) for item in value if isinstance(item, (str, int, float))] # Convert basic types to string
                    if len(corrected_list) != len(value):
                        logging.warning(f"{tag} Removed non-string elements from {key}. Original: {value}, Corrected: {corrected_list}")
                    data[key] = corrected_list

            elif key == "date" and value is not None:
                # Accept YYYY, c. YYYY, YYYYs, YYYY-YYYY, and allow for some flexibility
                if not isinstance(value, str):
                    logging.warning(f"{tag} Non-string date '{value}' (type {type(value)}). Setting to None.")
                    data[key] = None
                else:
                    val = value.strip()
                    # Acceptable: 4-digit year, c. 4-digit, 4-digit-4-digit, 4-digit's, 4-digit s
                    flexible_pattern = re.compile(r"^(c\.\s*)?\d{4}([sS]'?s?)?(-\d{4})?$")
                    if flexible_pattern.match(val):
                        data[key] = val
                    else:
                        logging.warning(f"{tag} Invalid or non-matching date format '{value}'. Setting to None.")
                        data[key] = None

            elif key == "nationality" and value is not None:
                if not isinstance(value, str) or not value.strip():
                    logging.warning(f"{tag} Invalid or empty nationality format '{value}'. Setting to None.")
                    data[key] = None
                else:
                    data[key] = value.strip() # Store cleaned string

            elif key == "genre" and value is not None: # Assuming 'movement' meant style/genre
                if not isinstance(value, str) or not value.strip():
                    logging.warning(f"{tag} Invalid or empty genre format '{value}'. Setting to None.")
                    data[key] = None
                else:
                    data[key] = value.strip() # Store cleaned string

            elif key == "style" and value is not None:
                 if not isinstance(value, str) or not value.strip():
                     logging.warning(f"{tag} Invalid or empty style format '{value}'. Setting to None.")
                     data[key] = None
                 elif value.strip() not in self.prompt_template.predefined_styles:
                     logging.warning(f"{tag} Style '{value}' provided by model is not in the predefined list {self.prompt_template.predefined_styles}. Keeping it for consolidation but flagging.")
                     # Keep the style for now, but it's flagged. Could enforce null here if strict adherence is required.
                     # data[key] = None # Uncomment to enforce null if style not in list