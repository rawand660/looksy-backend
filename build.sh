#!/usr/bin/env bash
# exit on error
set -o errexit

# Install system dependencies like cmake
apt-get update && apt-get install -y cmake

# Install Python dependencies using the requirements.txt
pip install -r requirements.txt