#!/bin/bash
# VANTAGE - Complete pipeline execution script

set -e

echo "============================================"
echo "VANTAGE - Candidate Ranking System"
echo "============================================"
echo ""

# Check if candidates file exists
if [ ! -f "data/raw/candidates.jsonl" ]; then
    echo "ERROR: candidates.jsonl not found in data/raw/"
    echo "Please place the candidates file first."
    exit 1
fi

# Check if JD file exists
if [ ! -f "data/raw/job_description.md" ]; then
    echo "ERROR: job_description.md not found in data/raw/"
    exit 1
fi

# Create necessary directories
mkdir -p data/processed models output

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Step 1: Pre-compute JD requirements
echo "[1/4] Parsing job description..."
python scripts/parse_jd.py
echo ""

# Step 2: Pre-compute embeddings (skip if already done)
if [ ! -f "data/processed/candidate_embeddings.npy" ]; then
    echo "[2/4] Pre-computing candidate embeddings (5-10 minutes)..."
    python scripts/precompute_embeddings.py
else
    echo "[2/4] Embeddings already exist, skipping..."
fi
echo ""

# Step 3: Pre-compute features (skip if already done)
if [ ! -f "data/processed/candidate_features.parquet" ]; then
    echo "[3/4] Pre-computing candidate features (3-5 minutes)..."
    python scripts/precompute_features.py
else
    echo "[3/4] Features already exist, skipping..."
fi
echo ""

# Step 4: Train ranker (skip if already done)
if [ ! -f "models/xgboost_ranker.json" ]; then
    echo "[4/4] Training XGBoost ranker..."
    python scripts/train_ranker.py
else
    echo "[4/4] Ranker model already exists, skipping..."
fi
echo ""

# Step 5: Run the ranking pipeline
echo "============================================"
echo "Running VANTAGE ranking pipeline..."
echo "============================================"
python src/pipeline.py \
    --candidates data/raw/candidates.jsonl \
    --jd data/raw/job_description.md \
    --output output/submission.csv

echo ""
echo "============================================"
echo "Pipeline complete!"
echo "Output: output/submission.csv"
echo "============================================"
echo ""
echo "Validate with:"
echo "  python scripts/validate_submission.py output/submission.csv"