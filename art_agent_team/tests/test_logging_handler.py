"""Test logging handler for live testing workflow.

This module extends the base logging functionality to provide structured logging
for test results, error aggregation, and summary generation.
"""

import os
import json
import logging
import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class TestResultType(Enum):
    """Enumeration of possible test result types."""
    STEP_START = "step_start"
    STEP_COMPLETE = "step_complete"
    STEP_ERROR = "step_error"
    VALIDATION_CHECK = "validation_check"
    ERROR_DETECTED = "error_detected"
    ANOMALY_DETECTED = "anomaly_detected"

@dataclass
class TestResult:
    """Container for test result data."""
    timestamp: str
    result_type: TestResultType
    step_name: str
    status: str
    details: Dict[str, Any]
    error_info: Optional[Dict[str, Any]] = None

class TestLogHandler(logging.Handler):
    """Custom logging handler for test execution and validation."""
    
    def __init__(self, base_log_dir: str):
        """Initialize the test log handler.
        
        Args:
            base_log_dir: Base directory for storing log files
        """
        super().__init__()
        self.base_log_dir = base_log_dir
        self.test_results: List[TestResult] = []
        self.current_run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_log_dir = os.path.join(base_log_dir, f"run_{self.current_run_id}")
        os.makedirs(self.run_log_dir, exist_ok=True)
        
        # Set up structured logging format
        self.formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        )
    
    def emit(self, record: logging.LogRecord) -> None:
        """Process a log record.
        
        Args:
            record: The log record to process
        """
        try:
            # Format the message
            msg = self.format(record)
            
            # Save to the appropriate log files based on log level
            self._write_to_log_file(msg, record.levelno)
            
            # If this is a test result, process it
            if hasattr(record, 'test_result'):
                self._process_test_result(record.test_result)
        except Exception as e:
            # Log any errors in handler to a separate file
            self._write_error(f"Error in TestLogHandler: {str(e)}")
    
    def _write_to_log_file(self, msg: str, level: int) -> None:
        """Write message to appropriate log file based on level.
        
        Args:
            msg: The formatted log message
            level: The logging level
        """
        # Determine log file based on level
        if level >= logging.ERROR:
            log_file = os.path.join(self.run_log_dir, "errors.log")
        else:
            log_file = os.path.join(self.run_log_dir, "test_execution.log")
        
        # Append message to file
        with open(log_file, 'a') as f:
            f.write(msg + '\n')
    
    def _write_error(self, error_msg: str) -> None:
        """Write handler errors to a separate log file.
        
        Args:
            error_msg: The error message to log
        """
        error_log = os.path.join(self.run_log_dir, "handler_errors.log")
        with open(error_log, 'a') as f:
            f.write(f"{datetime.datetime.now().isoformat()} - {error_msg}\n")
    
    def _process_test_result(self, result: TestResult) -> None:
        """Process and store a test result.
        
        Args:
            result: The test result to process
        """
        self.test_results.append(result)
        
        # Write result to JSON file
        results_file = os.path.join(self.run_log_dir, "test_results.json")
        try:
            with open(results_file, 'w') as f:
                json.dump(
                    [asdict(r) for r in self.test_results],
                    f,
                    indent=2,
                    default=str
                )
        except Exception as e:
            self._write_error(f"Error saving test results: {str(e)}")
    
    def log_test_result(
        self,
        result_type: TestResultType,
        step_name: str,
        status: str,
        details: Dict[str, Any],
        error_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a test result.
        
        Args:
            result_type: Type of test result
            step_name: Name of the step being tested
            status: Status of the test (e.g., "passed", "failed")
            details: Additional details about the test
            error_info: Optional error information if test failed
        """
        result = TestResult(
            timestamp=datetime.datetime.now().isoformat(),
            result_type=result_type,
            step_name=step_name,
            status=status,
            details=details,
            error_info=error_info
        )
        
        # Create a log record with the test result
        logger = logging.getLogger(__name__)
        record = logger.makeRecord(
            __name__,
            logging.INFO if status == "passed" else logging.ERROR,
            __file__,
            0,
            f"Test result: {step_name} - {status}",
            (),
            None
        )
        record.test_result = result
        
        self.handle(record)
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of test results.
        
        Returns:
            Dict containing test execution summary
        """
        summary = {
            "run_id": self.current_run_id,
            "total_tests": len(self.test_results),
            "passed": len([r for r in self.test_results if r.status == "passed"]),
            "failed": len([r for r in self.test_results if r.status == "failed"]),
            "errors": [
                {
                    "step": r.step_name,
                    "error": r.error_info
                }
                for r in self.test_results
                if r.error_info is not None
            ],
            "steps": {}
        }
        
        # Aggregate results by step
        for result in self.test_results:
            if result.step_name not in summary["steps"]:
                summary["steps"][result.step_name] = {
                    "total": 0,
                    "passed": 0,
                    "failed": 0
                }
            
            step_stats = summary["steps"][result.step_name]
            step_stats["total"] += 1
            if result.status == "passed":
                step_stats["passed"] += 1
            else:
                step_stats["failed"] += 1
        
        # Save summary to file
        summary_file = os.path.join(self.run_log_dir, "test_summary.json")
        try:
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            self._write_error(f"Error saving test summary: {str(e)}")
        
        return summary

def get_test_logger(log_dir: str) -> logging.Logger:
    """Get a logger configured with the test logging handler.
    
    Args:
        log_dir: Directory for storing log files
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("test_logger")
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Add our custom handler
    handler = TestLogHandler(log_dir)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    
    # Ensure we capture all messages
    logger.setLevel(logging.DEBUG)
    
    return logger