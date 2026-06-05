#!/bin/bash

# Run Docent workflow tests with focus on markdown logging
echo "Running Docent Agent Markdown Workflow Tests..."
echo "=============================================="

# Set Python path to include project root
export PYTHONPATH=$PYTHONPATH:$(dirname $(dirname $PWD))

# Run tests with detailed output
python -m unittest test_docent_agent.py -v

# Check test execution status
if [ $? -eq 0 ]; then
    echo "=============================================="
    echo "✅ All Docent markdown workflow tests passed!"
else
    echo "=============================================="
    echo "❌ Some tests failed. Check output above for details."
    exit 1
fi