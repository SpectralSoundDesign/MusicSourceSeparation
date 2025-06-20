#!/bin/bash

# Simple test runner

# Install test dependencies
pip install pytest

# Run all tests
echo "Running tests..."
pytest test/ -v

# Run specific test modules
# pytest test/test_model.py -v
# pytest test/test_processing.py -v
# pytest test/test_musdb.py -v