#!/bin/bash

# This script runs the blind assistance system in Raspberry Pi mode.
# It is recommended to run this script from the root of the project directory.

# Activate virtual environment if it exists
if [ -d "venv" ]; then
  source venv/bin/activate
fi

# Run the main script
python3 main.py --pi --no-display
