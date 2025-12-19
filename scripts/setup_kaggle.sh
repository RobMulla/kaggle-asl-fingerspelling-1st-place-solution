#!/bin/bash

# Setup script for Kaggle environment
# Installs uv, creates venv, and links data

set -e

echo "Starting Kaggle setup..."

# 1. Install uv
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
else
    echo "uv is already installed"
fi

# 2. Setup Python environment
echo "Setting up virtual environment..."
uv venv .venv --python 3.10 --allow-existing
source .venv/bin/activate

echo "Installing dependencies..."
# Use uv pip install for faster resolution
uv pip install -r requirements.txt

# 3. Setup Data Directory Structure
echo "Setting up data directories..."
mkdir -p datamount/weights

# Symlink Datasets
# Adjust these paths if your Kaggle dataset mount points differ
KAGGLE_INPUT="/kaggle/input"
SUPP_DS="$KAGGLE_INPUT/asl-fingerspelling-preprocessed-supp-dataset"
TRAIN_DS="$KAGGLE_INPUT/asl-fingerspelling-preprocessing-train-dataset"

# Link train_landmarks_npy
if [ -d "$TRAIN_DS/train_landmarks_npy" ]; then
    echo "Linking train_landmarks_npy..."
    ln -sf "$TRAIN_DS/train_landmarks_npy" datamount/train_landmarks_npy
elif [ -d "$TRAIN_DS" ]; then
    # Fallback if train_landmarks_npy is directly inside the mount
    echo "Linking train_landmarks_npy (fallback)..."
    ln -sf "$TRAIN_DS" datamount/train_landmarks_npy
else
    echo "WARNING: train_landmarks_npy not found in expected locations."
fi

# Link character_to_prediction_index.json
if [ -f "$SUPP_DS/character_to_prediction_index.json" ]; then
    echo "Linking character_to_prediction_index.json..."
    ln -sf "$SUPP_DS/character_to_prediction_index.json" datamount/character_to_prediction_index.json
else
    echo "WARNING: character_to_prediction_index.json not found."
fi

# Link symmetry.csv (if exists)
if [ -f "$SUPP_DS/symmetry.csv" ]; then
    echo "Linking symmetry.csv..."
    ln -sf "$SUPP_DS/symmetry.csv" datamount/symmetry.csv
else
    # Check if it's in the repo already or needs to be downloaded
    echo "WARNING: symmetry.csv not found in dataset. Checking repo..."
    if [ ! -f "datamount/symmetry.csv" ]; then
         echo "NOTE: You may need to upload symmetry.csv to datamount/ manually."
    fi
fi

# Link train_folded_oof_supp.csv (if exists)
if [ -f "$SUPP_DS/train_folded_oof_supp.csv" ]; then
    echo "Linking train_folded_oof_supp.csv..."
    ln -sf "$SUPP_DS/train_folded_oof_supp.csv" datamount/train_folded_oof_supp.csv
elif [ -f "$SUPP_DS/supplemental_metadata.csv" ]; then
    echo "Found supplemental_metadata.csv. You might need to generate train_folded_oof_supp.csv using scripts."
    # Potentially link it as base if code supports it, but likely strict naming is needed
fi

# Link validation weights for OOF generation (if using that script)
# (Optional logic here)

echo "Setup complete. Activate environment with 'source .venv/bin/activate'"
