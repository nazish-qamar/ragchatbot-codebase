#!/bin/bash

# Run code quality checks for the project

if [ "$1" = "--fix" ]; then
    echo "Formatting code with black..."
    uv run black backend/ main.py
else
    echo "Checking code formatting with black..."
    uv run black --check backend/ main.py
fi
