#!/bin/bash

# This script runs the blind assistance system.
# It checks for and installs the required Python libraries before running the main script.

# Activate virtual environment if it exists
if [ -d "venv" ]; then
  source venv/bin/activate
fi

# Check if requirements are installed
if ! pip freeze | grep -q -f requirements.txt; then
  echo "Installing required libraries..."
  pip install -r requirements.txt
fi

# Run the main script in Raspberry Pi mode
python3 main.py --pi --no-display