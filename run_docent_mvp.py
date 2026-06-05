import os
import logging
import sys

# Ensure the project root is in sys.path if script is not run from there directly
# This helps in locating the art_agent_team package.
# Assuming run_docent_mvp.py is in the project root.
# If art_agent_team is installed or PYTHONPATH is set, this might not be strictly necessary.
# However, for local development, it's a good safeguard.
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from art_agent_team.docent_agent import DocentAgent
except ImportError as e:
    print(f"Failed to import DocentAgent. Ensure 'art_agent_team' is in PYTHONPATH or script is in project root: {e}")
    sys.exit(1)

# Configure logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting MVP Demo Script for DocentAgent workflow.")

    # Define the input directory for the images
    input_dir = "art_agent_team/tests/test_data/input"
    
    # Ensure the input directory exists
    if not os.path.isdir(input_dir):
        logger.error(f"Input directory '{input_dir}' not found. Please create it and add images.")
        print(f"Error: Input directory '{input_dir}' not found. Cannot proceed.")
        return

    try:
        logger.info("Initializing DocentAgent...")
        # DocentAgent will attempt to load its default config from 'art_agent_team/config/config.yaml'.
        # This config might specify input_folder, output_folder, workspace_folder.
        # We will explicitly override the input_folder after initialization for this demo.
        docent = DocentAgent() 
        logger.info("DocentAgent initialized successfully.")

        # Override the input_folder to use the specified test data directory
        docent.input_folder = input_dir
        logger.info(f"DocentAgent input_folder has been explicitly set to: {docent.input_folder}")
        
        # Log the effective output and workspace folders that will be used by the agent
        logger.info(f"DocentAgent output_folder (for final outputs if Placard stage runs): {docent.output_folder}")
        logger.info(f"DocentAgent workspace_folder (for intermediate files): {docent.workspace_folder}")
        logger.info(f"DocentAgent markdown_log_file (for detailed workflow logging): {docent.markdown_log_file}")

        # Start the interactive workflow.
        # The DocentAgent's start_workflow() method will prompt the user to select stages.
        logger.info("Starting DocentAgent.start_workflow() (this will be interactive)...")
        print("\n" + "="*50)
        print("--- Starting Docent Agent Workflow ---")
        print(f"The DocentAgent will process images from the folder: '{docent.input_folder}'")
        print("You will be prompted to select the workflow stages to run.")
        print("For this MVP demo, please enter '1' when prompted to select the 'Research' stage.")
        print(f"Detailed progress, including ResearchAgent findings, will be logged to: '{docent.markdown_log_file}'")
        print("="*50 + "\n")
        
        docent.start_workflow() # This call is interactive and will block until user input is provided in the terminal.

        logger.info("DocentAgent.start_workflow() has completed.")
        print("\n" + "="*50)
        print("--- Docent Agent Workflow Complete ---")
        print(f"Please review the console output above for real-time messages from the workflow.")
        print(f"A detailed log of the workflow, including ResearchAgent's findings for each image, can be found in: '{docent.markdown_log_file}'")
        print(f"The designated output folders 'output/Docent_Output' and 'output/Research_Output' created earlier are not directly populated by this script with individual files from these specific stages. DocentAgent's `start_workflow` manages outputs differently: research findings are logged and passed internally, and final image products (e.g., from Placard stage if run) go to DocentAgent's configured `output_folder` (typically '{docent.output_folder}').")
        print("="*50)

    except Exception as e:
        logger.error(f"An critical error occurred during the DocentAgent workflow execution: {e}", exc_info=True)
        print(f"An error occurred: {e}. Please check the logs for more details.")

if __name__ == "__main__":
    main()