"""Workflow validation for live testing of multi-agent image processing.

This module provides validation functions to verify outputs and intermediate results
at each stage of the workflow, integrating with test_logging_handler.py for
comprehensive test reporting.
"""

import os
import json
import imghdr
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .test_logging_handler import (
    TestLogHandler,
    TestResultType,
    get_test_logger
)

class ValidationError(Exception):
    """Custom exception for validation failures."""
    pass

class WorkflowValidator:
    """Validates the multi-agent workflow execution and outputs."""
    
    def __init__(self, base_output_dir: str, logger: Optional[TestLogHandler] = None):
        """Initialize the workflow validator.
        
        Args:
            base_output_dir: Base directory for workflow outputs
            logger: Optional test logger instance
        """
        self.base_output_dir = base_output_dir
        self.logger = logger or get_test_logger("logs")
    
    def validate_vision_agent_output(
        self,
        image_name: str,
        output_dir: str
    ) -> Tuple[bool, List[str]]:
        """Validate Vision Agent outputs for an image.
        
        Args:
            image_name: Name of the input image
            output_dir: Directory containing Vision Agent outputs
            
        Returns:
            Tuple of (success: bool, errors: List[str])
        """
        base_name = Path(image_name).stem
        errors = []
        
        # Expected output files
        visualization_path = os.path.join(output_dir, f"{base_name}_labeled.jpg")
        masked_path = os.path.join(output_dir, f"{base_name}_masked.jpg")
        analysis_path = os.path.join(output_dir, f"{base_name}_analysis.json")
        
        # Check file presence
        for path, desc in [
            (visualization_path, "labeled visualization"),
            (masked_path, "masked version"),
            (analysis_path, "analysis JSON")
        ]:
            if not os.path.exists(path):
                errors.append(f"Missing {desc} at {path}")
                continue
            
            # Validate image files
            if path.endswith('.jpg'):
                if not self._validate_image_file(path):
                    errors.append(f"Invalid image file: {path}")
            
            # Validate JSON structure
            if path.endswith('.json'):
                try:
                    with open(path, 'r') as f:
                        analysis = json.load(f)
                    
                    # Check required fields
                    required_fields = ['title', 'artist', 'creation_date', 'style', 'genre']
                    missing = [f for f in required_fields if f not in analysis]
                    if missing:
                        errors.append(f"Analysis JSON missing required fields: {', '.join(missing)}")
                except json.JSONDecodeError:
                    errors.append(f"Invalid JSON file: {path}")
                except Exception as e:
                    errors.append(f"Error reading analysis file: {str(e)}")
        
        success = len(errors) == 0
        self.logger.log_test_result(
            TestResultType.VALIDATION_CHECK,
            "vision_agent",
            "passed" if success else "failed",
            {
                "image": image_name,
                "output_dir": output_dir
            },
            {"errors": errors} if errors else None
        )
        
        return success, errors
    
    def validate_upscale_agent_output(
        self,
        image_name: str,
        output_dir: str
    ) -> Tuple[bool, List[str]]:
        """Validate Upscale Agent outputs for an image.
        
        Args:
            image_name: Name of the input image
            output_dir: Directory containing Upscale Agent outputs
            
        Returns:
            Tuple of (success: bool, errors: List[str])
        """
        base_name = Path(image_name).stem
        errors = []
        
        # Expected upscaled image
        upscaled_path = os.path.join(output_dir, f"{base_name}_upscaled.png")
        
        if not os.path.exists(upscaled_path):
            errors.append(f"Missing upscaled image at {upscaled_path}")
        elif not self._validate_image_file(upscaled_path):
            errors.append(f"Invalid upscaled image file: {upscaled_path}")
        
        success = len(errors) == 0
        self.logger.log_test_result(
            TestResultType.VALIDATION_CHECK,
            "upscale_agent",
            "passed" if success else "failed",
            {
                "image": image_name,
                "output_dir": output_dir
            },
            {"errors": errors} if errors else None
        )
        
        return success, errors
    
    def validate_placard_agent_output(
        self,
        image_name: str,
        output_dir: str
    ) -> Tuple[bool, List[str]]:
        """Validate Placard Agent outputs for an image.
        
        Args:
            image_name: Name of the input image
            output_dir: Directory containing Placard Agent outputs
            
        Returns:
            Tuple of (success: bool, errors: List[str])
        """
        base_name = Path(image_name).stem
        errors = []
        
        # Expected placard image
        placard_path = os.path.join(output_dir, f"{base_name}_placarded.jpg")
        
        if not os.path.exists(placard_path):
            errors.append(f"Missing placard image at {placard_path}")
        elif not self._validate_image_file(placard_path):
            errors.append(f"Invalid placard image file: {placard_path}")
        
        success = len(errors) == 0
        self.logger.log_test_result(
            TestResultType.VALIDATION_CHECK,
            "placard_agent",
            "passed" if success else "failed",
            {
                "image": image_name,
                "output_dir": output_dir
            },
            {"errors": errors} if errors else None
        )
        
        return success, errors
    
    def validate_complete_workflow(
        self,
        image_name: str,
        run_output_dir: str
    ) -> Tuple[bool, Dict[str, List[str]]]:
        """Validate all outputs for a complete workflow run on an image.
        
        Args:
            image_name: Name of the input image
            run_output_dir: Directory containing all outputs for this run
            
        Returns:
            Tuple of (success: bool, errors_by_stage: Dict[str, List[str]])
        """
        errors_by_stage = {}
        
        # Validate Vision Agent outputs
        vision_success, vision_errors = self.validate_vision_agent_output(
            image_name,
            os.path.join(run_output_dir, "vision_output")
        )
        if vision_errors:
            errors_by_stage["vision_agent"] = vision_errors
        
        # Validate Upscale Agent outputs
        upscale_success, upscale_errors = self.validate_upscale_agent_output(
            image_name,
            os.path.join(run_output_dir, "upscaled_output")
        )
        if upscale_errors:
            errors_by_stage["upscale_agent"] = upscale_errors
        
        # Validate Placard Agent outputs
        placard_success, placard_errors = self.validate_placard_agent_output(
            image_name,
            os.path.join(run_output_dir, "placard_output")
        )
        if placard_errors:
            errors_by_stage["placard_agent"] = placard_errors
        
        success = vision_success and upscale_success and placard_success
        
        # Log overall workflow validation result
        self.logger.log_test_result(
            TestResultType.VALIDATION_CHECK,
            "complete_workflow",
            "passed" if success else "failed",
            {
                "image": image_name,
                "output_dir": run_output_dir
            },
            {"errors_by_stage": errors_by_stage} if errors_by_stage else None
        )
        
        return success, errors_by_stage
    
    def _validate_image_file(self, path: str) -> bool:
        """Validate that a file is a valid image.
        
        Args:
            path: Path to the image file
            
        Returns:
            True if file is a valid image, False otherwise
        """
        try:
            if not os.path.exists(path):
                return False
            
            # Check if it's a valid image file
            img_type = imghdr.what(path)
            if img_type not in ['jpeg', 'png']:
                return False
            
            # Could add more image validation here if needed
            return True
        except Exception:
            return False