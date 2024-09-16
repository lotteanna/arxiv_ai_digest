#!/bin/bash

# Get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate the virtual environment
source "$DIR/venv/bin/activate"

# Run the Python script
python "$DIR/arxiv_ai_summarizer.py"

# Deactivate the virtual environment when done
deactivate