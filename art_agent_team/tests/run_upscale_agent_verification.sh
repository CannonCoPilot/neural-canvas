#!/bin/bash

# Set up error handling
set -e

# Setup directories
echo "Setting up test directories..."
mkdir -p "./test_data/output/upscale_verification"

# Run the verification tests
echo "Running UpscaleAgent verification tests..."
python3 test_upscale_agent_mvp.py 2>&1 | tee test_upscale_results.log

# Get the exit code of the Python script
TEST_EXIT_CODE=${PIPESTATUS[0]}

# Parse results and update verification log
echo "Updating verification log..."
{
    echo -e "\n## Test Execution Results - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "### Test Run Summary"
    if [ $TEST_EXIT_CODE -eq 0 ]; then
        echo "✓ All tests passed successfully"
    else
        echo "✗ Some tests failed"
    fi
    echo -e "\n### Detailed Results"
    echo '```'
    cat test_upscale_results.log
    echo '```'
} >> ../../logs/debug_upscale_agent_verification_log.md

# Cleanup temporary log file
rm test_upscale_results.log

# Exit with the Python script's exit code
exit $TEST_EXIT_CODE