"""Live test monitoring for multi-agent image processing workflow.

This script monitors the live test execution, integrating validation checks
and error detection while the workflow runs.
"""

import os
import sys
import json
import logging
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .test_logging_handler import (
    TestLogHandler,
    TestResultType,
    get_test_logger
)
from .test_workflow_validation import WorkflowValidator
from .test_error_detection import ErrorDetector, ErrorCategory, ErrorContext

class LiveTestMonitor:
    """Monitors and validates the live test execution."""
    
    def __init__(self, base_output_dir: str):
        """Initialize the live test monitor.
        
        Args:
            base_output_dir: Base directory for test outputs
        """
        self.base_output_dir = base_output_dir
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Set up logging
        log_dir = os.path.join("logs", f"live_test_{timestamp}")
        os.makedirs(log_dir, exist_ok=True)
        self.logger = get_test_logger(log_dir)
        
        # Initialize components
        self.validator = WorkflowValidator(base_output_dir, self.logger)
        self.error_detector = ErrorDetector(self.logger)
        
        self.logger.log_test_result(
            TestResultType.STEP_START,
            "monitor_initialization",
            "passed",
            {"timestamp": timestamp}
        )
    
    def monitor_workflow_step(
        self,
        step_name: str,
        image_name: str,
        output_dir: str
    ) -> bool:
        """Monitor and validate a single workflow step.
        
        Args:
            step_name: Name of the workflow step
            image_name: Name of the input image
            output_dir: Directory containing step outputs
            
        Returns:
            True if step passed validation, False otherwise
        """
        self.logger.log_test_result(
            TestResultType.STEP_START,
            step_name,
            "started",
            {
                "image": image_name,
                "output_dir": output_dir
            }
        )
        
        # Run validations based on step
        success = True
        if step_name == "vision_agent":
            success, errors = self.validator.validate_vision_agent_output(
                image_name,
                output_dir
            )
            # Check for hallucinations
            hallucination_errors = self.error_detector.check_for_hallucinations(
                output_dir,
                image_name
            )
            if hallucination_errors:
                success = False
                
        elif step_name == "upscale_agent":
            success, errors = self.validator.validate_upscale_agent_output(
                image_name,
                output_dir
            )
            
        elif step_name == "placard_agent":
            success, errors = self.validator.validate_placard_agent_output(
                image_name,
                output_dir
            )
        
        # Check for missing outputs
        missing_errors = self.error_detector.check_for_missing_outputs(
            os.path.dirname(output_dir),  # Parent dir containing all step outputs
            image_name
        )
        if missing_errors:
            success = False
        
        # Validate schema if output exists
        try:
            analysis_file = None
            if step_name == "vision_agent":
                analysis_file = os.path.join(
                    output_dir,
                    f"{Path(image_name).stem}_analysis.json"
                )
            
            if analysis_file and os.path.exists(analysis_file):
                with open(analysis_file, 'r') as f:
                    data = json.load(f)
                schema_errors = self.error_detector.validate_schema(
                    step_name,
                    data
                )
                if schema_errors:
                    success = False
        except Exception as e:
            self.logger.log_test_result(
                TestResultType.ERROR_DETECTED,
                step_name,
                "failed",
                {
                    "error_type": "schema_validation_error",
                    "error_msg": str(e)
                }
            )
            success = False
        
        # Log step completion
        self.logger.log_test_result(
            TestResultType.STEP_COMPLETE,
            step_name,
            "passed" if success else "failed",
            {
                "image": image_name,
                "output_dir": output_dir
            }
        )
        
        return success
    
    def monitor_complete_workflow(
        self,
        image_name: str,
        run_output_dir: str
    ) -> bool:
        """Monitor and validate the complete workflow for an image.
        
        Args:
            image_name: Name of the input image
            run_output_dir: Directory containing all outputs for this run
            
        Returns:
            True if workflow passed validation, False otherwise
        """
        self.logger.log_test_result(
            TestResultType.STEP_START,
            "complete_workflow",
            "started",
            {
                "image": image_name,
                "output_dir": run_output_dir
            }
        )
        
        # Monitor each step
        vision_success = self.monitor_workflow_step(
            "vision_agent",
            image_name,
            os.path.join(run_output_dir, "vision_output")
        )
        
        upscale_success = self.monitor_workflow_step(
            "upscale_agent",
            image_name,
            os.path.join(run_output_dir, "upscaled_output")
        )
        
        placard_success = self.monitor_workflow_step(
            "placard_agent",
            image_name,
            os.path.join(run_output_dir, "placard_output")
        )
        
        # Validate complete workflow
        workflow_success, errors_by_stage = self.validator.validate_complete_workflow(
            image_name,
            run_output_dir
        )
        
        success = all([
            vision_success,
            upscale_success,
            placard_success,
            workflow_success
        ])
        
        # Generate and save test summary
        summary = self.logger.generate_summary()
        
        self.logger.log_test_result(
            TestResultType.STEP_COMPLETE,
            "complete_workflow",
            "passed" if success else "failed",
            {
                "image": image_name,
                "output_dir": run_output_dir,
                "summary": summary
            }
        )
        
        return success

def main():
    """Main function to run the live test monitor."""
    parser = argparse.ArgumentParser(
        description="Monitor live testing of multi-agent image processing workflow"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Base output directory for the workflow"
    )
    args = parser.parse_args()
    
    monitor = LiveTestMonitor(args.output_dir)
    
    # Get list of test images from main workflow configuration
    test_images = [
        "Frank Brangwyn, Swans, c.1921.jpg",
        "Litzlberg am Attersee (1915).jpeg"
    ]
    
    all_success = True
    for image in test_images:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
        
        success = monitor.monitor_complete_workflow(image, run_dir)
        all_success = all_success and success
    
    sys.exit(0 if all_success else 1)

if __name__ == "__main__":
    main()