import sys
import os
import logging
import yaml
from art_agent_team.docent_agent import DocentAgent

# Configure basic logging (can be further configured in config.yaml later)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout) # Explicitly use stdout

import re # Add import for regex

# --- Custom Filter Definition ---
class TruncateImageDataFilter(logging.Filter):
    """
    Filters log records to truncate long 'data' values within dictionary-like strings,
    often containing image data.
    """
    MAX_DATA_LEN = 100  # Max length of data string to show before truncating
    # Regex to find 'data': '...' or "data": "..." patterns
    # It captures the opening quote (1), the key 'data', the closing quote (1),
    # the opening quote for the value (2), the value itself (3), and the closing quote (2).
    # Making the value capture non-greedy (.*?) is important.
    DATA_PATTERN_REGEX = re.compile(r"""
        (['"])data\1:  # Match 'data' or "data" followed by :
        \s*           # Optional whitespace
        (['"])        # Capture the opening quote of the value (' or ")
        (.*?)         # Capture the value itself (non-greedy)
        \2            # Match the closing quote of the value
    """, re.VERBOSE | re.DOTALL) # DOTALL allows . to match newline if data spans lines

    def _replacer(self, match):
        """Replacement function for re.sub"""
        opening_quote_key = match.group(1)
        opening_quote_value = match.group(2)
        data_value = match.group(3)
        # closing_quote_value = match.group(2) # Same as opening

        if len(data_value) > self.MAX_DATA_LEN:
            truncated_data = data_value[:self.MAX_DATA_LEN]
            # Basic truncation, assuming data doesn't contain problematic internal quotes
            return f"{opening_quote_key}data{opening_quote_key}: {opening_quote_value}{truncated_data}...[TRUNCATED]{opening_quote_value}"
        else:
            # Return the original matched string if data is not too long
            return match.group(0)

    def filter(self, record):
        # We operate on the formatted message as the structure might come from format strings + args
        try:
            message_content = record.getMessage() # Get the fully formatted message

            # Use re.sub with the replacer function
            # This finds all occurrences and replaces them if needed
            modified_message, num_replacements = self.DATA_PATTERN_REGEX.subn(self._replacer, message_content)

            if num_replacements > 0:
                # If modified, replace record.msg with the modified string
                # and clear args, as the original format string is no longer valid.
                record.msg = modified_message
                record.args = () # Clear args as the message is now pre-formatted

        except Exception as e:
            # Log filter error without crashing the application
            logging.getLogger(__name__).error(f"Error in TruncateImageDataFilter: {e}", exc_info=False)
            # Allow original record to pass through on error
            pass

        return True # Always allow the record to pass (possibly modified)
# --- End Custom Filter Definition ---


# Get the root logger
root_logger = logging.getLogger()

# Create a file handler
log_file_path = 'terminal_output.log'
file_handler = logging.FileHandler(log_file_path, mode='a') # Use 'a' to append

# Create the custom filter
image_data_filter = TruncateImageDataFilter()

# Add the filter to the file handler
file_handler.addFilter(image_data_filter)

# Add the filter to the console handler(s) as well
# basicConfig adds a StreamHandler to the root logger's handlers list
for handler in root_logger.handlers:
    if isinstance(handler, logging.StreamHandler) and handler.stream in (sys.stdout, sys.stderr):
         # Check if the filter is not already added (though unlikely here)
         if image_data_filter not in handler.filters:
              handler.addFilter(image_data_filter)
              logging.info(f"Added TruncateImageDataFilter to console handler: {handler}")

# Get the formatter from the existing handler(s) if available, or create one
if root_logger.handlers:
    formatter = root_logger.handlers[0].formatter
else: # Fallback if basicConfig didn't add a handler for some reason
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler.setFormatter(formatter)

# Add the file handler to the root logger
root_logger.addHandler(file_handler)

logging.info(f"Logging configured to output to console and '{log_file_path}'")


def load_and_set_env_from_config(config_path):
    """Loads config from YAML and sets specified keys as environment variables."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            if not isinstance(config, dict):
                raise ValueError("Config file did not load as a dictionary.")
    except Exception as e:
        logging.error(f"Failed to load or parse config file '{config_path}': {e}")
        raise RuntimeError(f"Failed to load or parse config file '{config_path}': {e}")

    # Define keys to load from config and their corresponding environment variable names
    api_keys_to_set = {
        'GROK_API_KEY': config.get('grok_api_key'),
        'OPENROUTER_API_KEY': config.get('openrouter_api_key'),
        'GOOGLE_API_KEY': config.get('google_api_key'), # Keep setting this if needed elsewhere
        # Use standard env var for Google credentials path
        'GOOGLE_APPLICATION_CREDENTIALS': config.get('google_credentials_path')
    }

    folder_paths_to_set = {
        'INPUT_FOLDER': config.get('input_folder'),
        'OUTPUT_FOLDER': config.get('output_folder'),
        'WORKSPACE_FOLDER': config.get('workspace_folder')
    }

    # Set API keys as environment variables
    logging.info("Setting environment variables from config...")
    keys_set_count = 0
    keys_missing_count = 0
    keys_placeholder_count = 0

    for env_var, value in api_keys_to_set.items():
        if value and isinstance(value, str) and value != "YOUR_OPENROUTER_API_KEY_HERE" and not value.startswith("YOUR_"): # Check for actual value and avoid common placeholders
            os.environ[env_var] = str(value) # Ensure value is string
            logging.info(f"Set environment variable {env_var} from config.")
            keys_set_count += 1
        elif not value:
            logging.warning(f"Config key for {env_var} not found or empty in '{config_path}'.")
            keys_missing_count += 1
        else: # Is placeholder or invalid type
            logging.warning(f"Config key for {env_var} in '{config_path}' is a placeholder or invalid type. Environment variable not set.")
            keys_placeholder_count += 1

    # Set folder paths as environment variables
    for env_var, value in folder_paths_to_set.items():
         if value and isinstance(value, str):
             os.environ[env_var] = value
             logging.info(f"Set environment variable {env_var}='{value}' from config.")
         else:
             logging.warning(f"Config key for {env_var} not found or invalid in '{config_path}'.")

    logging.info(f"Environment variable setting complete. Set: {keys_set_count}, Missing: {keys_missing_count}, Placeholders: {keys_placeholder_count}")

    # Optional: Raise error if critical keys are missing (e.g., if Grok is essential)
    # if not os.getenv('GROK_API_KEY'):
    #     logging.error("Critical environment variable GROK_API_KEY is not set. Exiting.")
    #     raise RuntimeError("Critical environment variable GROK_API_KEY is not set.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the Art Agent Team workflow.")
    parser.add_argument("--input_folder", type=str, default=None, help="Path to the input folder containing images.")
    parser.add_argument("--config", type=str, default="art_agent_team/config/config.yaml", help="Path to the config YAML file.")
    args = parser.parse_args()
    
    config_path = args.config
    load_and_set_env_from_config(config_path)
    
    # Use CLI input_folder if provided, else from config/env
    input_folder = args.input_folder if args.input_folder else os.environ.get("INPUT_FOLDER", "input")
    
    docent = DocentAgent(config_path=config_path)
    docent.input_folder = input_folder
    docent.start_workflow()