#!/bin/bash

# Set up Python environment and dependencies
echo "Setting up test environment..."
export PYTHONPATH="/Users/nathaniel.cannon/Documents/VScodeWork/Art_AI:$PYTHONPATH"

# Create required directories if they don't exist
mkdir -p "/Users/nathaniel.cannon/Documents/VScodeWork/Art_AI/art_agent_team/tests/test_data/input"
mkdir -p "/Users/nathaniel.cannon/Documents/VScodeWork/Art_AI/art_agent_team/tests/test_data/output"

# Copy test image if not present
if [ ! -f "/Users/nathaniel.cannon/Documents/VScodeWork/Art_AI/art_agent_team/tests/test_data/input/Frank Brangwyn, Swans, c.1921.jpg" ]; then
    cp "/Users/nathaniel.cannon/Documents/VScodeWork/Art_AI/art_agent_team/tests/test_data/input/Frank Brangwyn, Swans, c.1921.jpg" \
       "/Users/nathaniel.cannon/Documents/VScodeWork/Art_AI/art_agent_team/tests/test_data/input/"
fi

# Run the tests with detailed output
echo "Running artistic integrity tests..."
# Ensure we run from the root directory to set the correct package context
cd /Users/nathaniel.cannon/Documents/VScodeWork/Art_AI
python3 -m pytest art_agent_team/tests/test_artistic_integrity.py -v --capture=no

# Check test results and summarize
TEST_EXIT_CODE=$?
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "All tests passed successfully!"
else
    echo "Some tests failed. Please check the output above for details."
    echo "See art_agent_team/debug/error_investigation_log.md for known issues and solutions."
fi

exit $TEST_EXIT_CODE