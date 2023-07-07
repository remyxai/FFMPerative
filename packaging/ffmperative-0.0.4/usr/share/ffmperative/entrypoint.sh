#!/bin/sh

python3 -c "import os; from huggingface_hub import HfFolder; HfFolder.save_token(os.environ['HUGGINGFACE_TOKEN'])"

# Template for running ffmperative command given user prompt as variable $1
ffmp do --p "$1"

