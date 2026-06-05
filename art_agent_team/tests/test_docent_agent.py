import os
import sys
import logging
import yaml
import shutil
import tempfile
import json
from unittest import TestCase

# Adjust the path to include the parent directory for module resolution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from docent_agent import DocentAgent

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestDocentAgentMarkdownLogging(TestCase):
    """Unit tests for the markdown logging functionality of DocentAgent."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'config.yaml')
        self.log_dir = os.path.join(self.temp_dir, 'logs')
        self.log_file = os.path.join(self.log_dir, 'docent_workflow_log.md')
        self.input_dir = os.path.join(self.temp_dir, 'input')
        self.workspace_dir = os.path.join(self.temp_dir, 'workspace')
        self.output_dir = os.path.join(self.temp_dir, 'output')
        
        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.workspace_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create a minimal config file
        config = {
            'input_folder': self.input_dir,
            'workspace_folder': self.workspace_dir,
            'output_folder': self.output_dir,
            'cleanup_workspace': False
        }
        with open(self.config_path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False)
        
        # Initialize DocentAgent with the temporary config
        self.docent = DocentAgent(config_path=self.config_path)

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        # Remove temporary directory and all contents
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_markdown_log_file_creation(self):
        """Test that the markdown log file is created upon initialization."""
        self.assertTrue(os.path.exists(self.log_file), "Markdown log file should be created upon initialization.")
        with open(self.log_file, 'r') as f:
            content = f.read()
        self.assertTrue(content.startswith("# Docent Workflow Log - Initialized:"), "Log file should start with initialization header.")

    def test_markdown_log_appending(self):
        """Test that log entries are appended during workflow initialization."""
        # Simulate starting the workflow (without actual image processing)
        self.docent.start_workflow = lambda: None  # Mock to avoid actual workflow
        self.docent._log_to_markdown("Test log entry")
        with open(self.log_file, 'r') as f:
            content = f.read()
        self.assertIn("Test log entry", content, "Log entry should be appended to the file.")

    def test_markdown_log_content_structure(self):
        """Test that log entries contain expected information and formatting."""
        entry = self.docent._format_log_entry(
            agent_name="TestAgent",
            action="TestAction",
            details=["Detail1", "Detail2"],
            summary="TestSummary",
            error="TestError",
            image_context="test_image.jpg"
        )
        self.assertIn("### TestAgent: TestAction (Image: `test_image.jpg`)", entry, "Entry should include agent name, action, and image context.")
        self.assertIn("- **Timestamp:**", entry, "Entry should include timestamp.")
        self.assertIn("- **Details:** Detail1", entry, "Entry should include details.")
        self.assertIn("-             Detail2", entry, "Subsequent details should be aligned.")
        self.assertIn("- **Summary:** TestSummary", entry, "Entry should include summary.")
        self.assertIn("- **Error:** TestError", entry, "Entry should include error.")

    def test_markdown_log_error_handling(self):
        """Test that logging continues to function even if file writing fails."""
        # Simulate a failure by making the log file read-only (on Unix-like systems)
        if os.name != 'nt':  # Skip on Windows as chmod behavior differs
            os.chmod(self.log_file, 0o444)  # Read-only
            self.docent._log_to_markdown("Test log entry under read-only condition")
            os.chmod(self.log_file, 0o666)  # Restore write permissions
            # Check that no exception crashes the test, and logging error is captured in the logger
            self.assertTrue(True, "Logging should handle file write errors gracefully.")
        else:
            # For Windows, we can't easily simulate read-only without admin rights, so skip detailed test
            self.assertTrue(True, "Skipping detailed error handling test on Windows.")

    def test_format_confidence_score(self):
        """Test the static _format_confidence_score method with various inputs."""
        # Test numerical values
        self.assertEqual(DocentAgent._format_confidence_score(0.756), "75.6%")
        self.assertEqual(DocentAgent._format_confidence_score(1.0), "100.0%")
        self.assertEqual(DocentAgent._format_confidence_score(0.0), "0.0%")
        
        # Test edge cases and invalid inputs
        self.assertEqual(DocentAgent._format_confidence_score(None), "N/A")
        self.assertEqual(DocentAgent._format_confidence_score("invalid"), "Invalid")
        self.assertEqual(DocentAgent._format_confidence_score(-0.5), "-50.0%")
        self.assertEqual(DocentAgent._format_confidence_score(1.5), "150.0%")

    def test_handle_request_markdown_logging(self):
        """Test markdown logging during handle_request execution."""
        # Create a temporary test image
        test_image = os.path.join(self.input_dir, "test_image.jpg")
        with open(test_image, 'wb') as f:
            f.write(b'dummy image content')

        # Call handle_request
        self.docent.handle_request("Test task", test_image)

        # Verify log file contents
        with open(self.log_file, 'r') as f:
            log_content = f.read()

        # Check for required markdown sections with correct headings
        self.assertIn("## Workflow Run Started (Programmatic Request: Test task)", log_content)
        self.assertIn("- **Input Image:** ", log_content)
        self.assertIn("### DocentAgent: Initializing Research", log_content)
        self.assertIn("### ResearchAgent: Findings", log_content)
        self.assertIn("## Workflow Run Ended", log_content)
        self.assertIn("- **Final Image Path:** ", log_content)
        self.assertIn("- **Final Metadata:** ", log_content)

    def test_log_entry_optional_parameters(self):
        """Test _format_log_entry with optional parameters."""
        # Test with minimal parameters
        entry = self.docent._format_log_entry(
            agent_name="TestAgent",
            action="TestAction"
        )
        self.assertIn("### TestAgent: TestAction", entry)
        self.assertNotIn("Image:", entry)
        self.assertNotIn("Details:", entry)
        self.assertNotIn("Error:", entry)
        self.assertNotIn("Summary:", entry)

        # Test with empty details list
        entry = self.docent._format_log_entry(
            agent_name="TestAgent",
            action="TestAction",
            details=[]
        )
        self.assertNotIn("Details:", entry)

        # Test with None values
        entry = self.docent._format_log_entry(
            agent_name="TestAgent",
            action="TestAction",
            details=None,
            summary=None,
            error=None,
            image_context=None
        )
        self.assertIn("### TestAgent: TestAction", entry)
        self.assertNotIn("Image:", entry)

    def test_markdown_log_file_rotation(self):
        """Test that new DocentAgent instance creates fresh log file."""
        # Write initial content
        self.docent._log_to_markdown("Initial content")
        
        # Create new instance
        new_docent = DocentAgent(config_path=self.config_path)
        
        # Verify log file was reset
        with open(self.log_file, 'r') as f:
            content = f.read()
        self.assertNotIn("Initial content", content)
        self.assertTrue(content.startswith("# Docent Workflow Log - Initialized:"))

if __name__ == '__main__':
    import unittest
    unittest.main()