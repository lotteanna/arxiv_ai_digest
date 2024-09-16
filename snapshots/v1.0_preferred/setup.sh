#!/bin/bash

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install arxiv transformers torch python-dotenv sentence-transformers scikit-learn

# Create requirements.txt
pip freeze > requirements.txt

# Make the main script executable
chmod +x arxiv_ai_summarizer.py

echo "Setup complete. Activate the virtual environment with 'source venv/bin/activate'"