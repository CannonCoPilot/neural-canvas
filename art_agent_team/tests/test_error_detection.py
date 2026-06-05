"""Error detection for live testing of multi-agent image processing.

This module provides error detection and analysis capabilities, focusing on:
1. Anomaly detection (hallucinations, unexpected outputs)
2. Missing output detection
3. Inter-agent communication validation
4. Schema validation for agent payloads
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from .test_logging_handler import (
    TestLogHandler,
    TestResultType,
    get_test_logger
)

class ErrorCategory(Enum):
    """Categories of errors that can be detected."""
    HALLUCINATION = "hallucination"
    MISSING_OUTPUT = "missing_output"
    SCHEMA_VIOLATION = "schema_violation"
    COMMUNICATION_ERROR = "communication_error"
    DATA_CORRUPTION = "data_corruption"

@dataclass
class ErrorContext:
    """Context information for detected errors."""
    category: ErrorCategory
    step_name: str
    details: Dict[str, Any]
    severity: str
    timestamp: str

class ErrorDetector:
    """Detects and analyzes errors in the multi-agent workflow."""
    
    def __init__(self, logger: Optional[TestLogHandler] = None):
        """Initialize the error detector.
        
        Args:
            logger: Optional test logger instance
        """
        self.logger = logger or get_test_logger("logs")
        self._load_expected_schemas()
    
    def _load_expected_schemas(self) -> None:
        """Load expected schemas for agent outputs and communication."""
        # Define expected schemas for each agent's outputs
        self.expected_schemas = {
            "vision_agent": {
                "analysis": {
                    "required": [
                        "title",
                        "artist",
                        "creation_date",
                        "style",
                        "genre"
                    ],
                    "optional": [
                        "important_objects",
                        "composition_notes",
                        "confidence_score"
                    ]
                }
            },
            "upscale_agent": {
                "metadata": {
                    "required": [
                        "original_size",
                        "upscaled_size",
                        "upscaler_used"
                    ],
                    "optional": [
                        "quality_score",
                        "processing_time"
                    ]
                }
            },
            "placard_agent": {
                "metadata": {
                    "required": [
                        "title",
                        "description",
                        "font_used",
                        "background_used"
                    ],
                    "optional": [
                        "text_color",
                        "margin_size"
                    ]
                }
            }
        }
    
    def check_for_hallucinations(
        self,
        vision_output_dir: str,
        image_name: str
    ) -> List[ErrorContext]:
        """Check for potential hallucinations in Vision Agent output.
        
        Args:
            vision_output_dir: Directory containing Vision Agent outputs
            image_name: Name of the input image
            
        Returns:
            List of detected hallucination errors
        """
        errors: List[ErrorContext] = []
        base_name = Path(image_name).stem
        analysis_path = os.path.join(vision_output_dir, f"{base_name}_analysis.json")
        
        try:
            with open(analysis_path, 'r') as f:
                analysis = json.load(f)
            
            # Check for potential hallucination indicators
            
            # 1. Highly improbable combinations
            if analysis.get('creation_date') and not self._is_plausible_date(analysis['creation_date']):
                errors.append(ErrorContext(
                    category=ErrorCategory.HALLUCINATION,
                    step_name="vision_agent",
                    details={
                        "type": "implausible_date",
                        "value": analysis['creation_date']
                    },
                    severity="warning",
                    timestamp=self.logger.current_run_id
                ))
            
            # 2. Inconsistent metadata
            if analysis.get('style') and analysis.get('creation_date'):
                if not self._is_style_consistent_with_period(
                    analysis['style'],
                    analysis['creation_date']
                ):
                    errors.append(ErrorContext(
                        category=ErrorCategory.HALLUCINATION,
                        step_name="vision_agent",
                        details={
                            "type": "style_period_mismatch",
                            "style": analysis['style'],
                            "date": analysis['creation_date']
                        },
                        severity="warning",
                        timestamp=self.logger.current_run_id
                    ))
            
            # Log any detected hallucinations
            for error in errors:
                self.logger.log_test_result(
                    TestResultType.ANOMALY_DETECTED,
                    error.step_name,
                    "failed",
                    {
                        "category": error.category.value,
                        "details": error.details
                    }
                )
        
        except Exception as e:
            self.logger.log_test_result(
                TestResultType.ERROR_DETECTED,
                "vision_agent",
                "failed",
                {
                    "error_type": "analysis_check_failed",
                    "error_msg": str(e)
                }
            )
        
        return errors
    
    def check_for_missing_outputs(
        self,
        run_output_dir: str,
        image_name: str
    ) -> List[ErrorContext]:
        """Check for any missing outputs in the workflow.
        
        Args:
            run_output_dir: Directory containing all outputs for this run
            image_name: Name of the input image
            
        Returns:
            List of detected missing output errors
        """
        errors: List[ErrorContext] = []
        base_name = Path(image_name).stem
        
        # Expected outputs for each stage
        expected_outputs = {
            "vision_agent": [
                f"{base_name}_labeled.jpg",
                f"{base_name}_masked.jpg",
                f"{base_name}_analysis.json"
            ],
            "upscale_agent": [
                f"{base_name}_upscaled.png"
            ],
            "placard_agent": [
                f"{base_name}_placarded.jpg"
            ]
        }
        
        for agent, expected_files in expected_outputs.items():
            agent_dir = os.path.join(run_output_dir, f"{agent.split('_')[0]}_output")
            
            for expected_file in expected_files:
                file_path = os.path.join(agent_dir, expected_file)
                if not os.path.exists(file_path):
                    error = ErrorContext(
                        category=ErrorCategory.MISSING_OUTPUT,
                        step_name=agent,
                        details={
                            "missing_file": expected_file,
                            "expected_path": file_path
                        },
                        severity="error",
                        timestamp=self.logger.current_run_id
                    )
                    errors.append(error)
                    
                    self.logger.log_test_result(
                        TestResultType.ERROR_DETECTED,
                        agent,
                        "failed",
                        {
                            "category": "missing_output",
                            "details": error.details
                        }
                    )
        
        return errors
    
    def validate_schema(
        self,
        agent_name: str,
        data: Dict[str, Any]
    ) -> List[ErrorContext]:
        """Validate data against expected schema for an agent.
        
        Args:
            agent_name: Name of the agent
            data: Data to validate
            
        Returns:
            List of detected schema violations
        """
        errors: List[ErrorContext] = []
        schema = self.expected_schemas.get(agent_name, {})
        
        for output_type, fields in schema.items():
            if output_type not in data:
                errors.append(ErrorContext(
                    category=ErrorCategory.SCHEMA_VIOLATION,
                    step_name=agent_name,
                    details={
                        "missing_section": output_type
                    },
                    severity="error",
                    timestamp=self.logger.current_run_id
                ))
                continue
            
            # Check required fields
            missing_required = [
                field for field in fields["required"]
                if field not in data[output_type]
            ]
            if missing_required:
                errors.append(ErrorContext(
                    category=ErrorCategory.SCHEMA_VIOLATION,
                    step_name=agent_name,
                    details={
                        "missing_required_fields": missing_required
                    },
                    severity="error",
                    timestamp=self.logger.current_run_id
                ))
            
            # Log any schema violations
            for error in errors:
                self.logger.log_test_result(
                    TestResultType.ERROR_DETECTED,
                    agent_name,
                    "failed",
                    {
                        "category": "schema_violation",
                        "details": error.details
                    }
                )
        
        return errors
    
    def _is_plausible_date(self, date_str: str) -> bool:
        """Check if a date string is plausible for artwork.
        
        Args:
            date_str: Date string to check
            
        Returns:
            True if date is plausible, False otherwise
        """
        try:
            # Simple check for now - could be more sophisticated
            if "c." in date_str.lower():
                date_str = date_str.lower().replace("c.", "").strip()
            
            year = int(''.join(filter(str.isdigit, date_str)))
            return 1000 <= year <= 2025  # Reasonable range for artwork
        except ValueError:
            return False
    
    def _is_style_consistent_with_period(
        self,
        style: str,
        date_str: str
    ) -> bool:
        """Check if an art style is consistent with a time period.
        
        Args:
            style: Art style
            date_str: Date string
            
        Returns:
            True if style is consistent with period, False otherwise
        """
        # This is a simplified check - could be more comprehensive
        style_periods = {
            "renaissance": (1300, 1600),
            "baroque": (1600, 1750),
            "romantic": (1780, 1850),
            "impressionist": (1850, 1900),
            "modern": (1900, 1970),
            "contemporary": (1970, 2025)
        }
        
        try:
            year = int(''.join(filter(str.isdigit, date_str)))
            style_lower = style.lower()
            
            for known_style, (start, end) in style_periods.items():
                if style_lower in known_style and not (start <= year <= end):
                    return False
            
            return True
        except ValueError:
            return True  # If we can't parse the date, assume it's consistent