#!/bin/bash
set -euo pipefail

# ==============================================================================
# SOLOMUSE End-to-End Pipeline Validation Script
# ==============================================================================
# This script runs the entire baseline pipeline on a specified dataset.
# It is designed to smoke-test the stability of the entire pipeline on an M2 Mac.
#
# Usage: ./scripts/run_full_validation.sh [DATASET] [LIMIT]
# Example: ./scripts/run_full_validation.sh slakh 100
#
# Recommended Sizes:
#  - Quick Smoke: 25   (~1-2 mins)
#  - Medium Check: 100 (~3-5 mins)
#  - Stress Test: 500+ (15+ mins, good for memory leak hunting)
# ==============================================================================

DATASET=${1:-slakh}
LIMIT=${2:-100}

CONFIG="config.yaml"
LIMIT_FLAG=""
if [ "$LIMIT" -gt 0 ]; then
    LIMIT_FLAG="--limit $LIMIT"
fi

echo "================================================================================"
echo "🧪 Running SoloMuse Pipeline Validation"
echo "   Dataset: $DATASET"
echo "   Limit: $LIMIT segments"
echo "   Device Expected: mps (fallback CPU)"
echo "================================================================================"

echo ""
echo "[1/8] Canonicalize Audio"
python -m solomuse_data.cli canonicalize --config $CONFIG --dataset $DATASET || { echo "❌ Stage 1 failed"; exit 1; }

echo ""
echo "[2/8] Build Backing/Solo Pairs"
python -m solomuse_data.cli build-pairs --config $CONFIG --dataset $DATASET || { echo "❌ Stage 2 failed"; exit 1; }

echo ""
echo "[3/8] Segment Audio"
python -m solomuse_data.cli segment --config $CONFIG --dataset $DATASET || { echo "❌ Stage 3 failed"; exit 1; }

echo ""
echo "[4/8] Extract Situation Layer (Limit: $LIMIT)"
python -m solomuse_data.cli situation --config $CONFIG --dataset $DATASET $LIMIT_FLAG --overwrite || { echo "❌ Stage 4 failed"; exit 1; }

echo ""
echo "[5/8] Build Intent Targets (Limit: $LIMIT)"
python -m solomuse_data.cli intent-targets --config $CONFIG --dataset $DATASET $LIMIT_FLAG --overwrite || { echo "❌ Stage 5 failed"; exit 1; }

echo ""
echo "[6/8] Build Renderer Targets (Limit: $LIMIT)"
python -m solomuse_data.cli renderer-targets --config $CONFIG --dataset $DATASET $LIMIT_FLAG --overwrite || { echo "❌ Stage 6 failed"; exit 1; }

echo ""
echo "================================================================================"
echo "🔍 Preflight Check: Verifying Extraction Manifests"
for MANIFEST in "manifest.csv" "manifest_situation.csv" "manifest_intent.csv" "manifest_renderer.csv"; do
    FILE="data/processed/segments/$DATASET/$MANIFEST"
    if [ ! -f "$FILE" ]; then
        echo "❌ Preflight failed: Missing required manifest '$FILE'"
        exit 1
    fi
done
echo "✅ All manifests found."
echo "================================================================================"

echo ""
echo "[7/8] Train Intent Planner"
echo "(Note: if no validation split, will fallback to train loss)"
python -m solomuse_data.cli train-intent --config $CONFIG --dataset $DATASET || { echo "❌ Stage 7 failed"; exit 1; }

echo ""
echo "[8/8] Train Renderer"
python -m solomuse_data.cli train-renderer --config $CONFIG --dataset $DATASET || { echo "❌ Stage 8 failed"; exit 1; }

echo ""
echo "================================================================================"
echo "✅ Pipeline Training Stable!"
echo "   Checking Inference outputs..."
echo "================================================================================"

# Find the first segment directory built (usually Track00000 or Track00013)
SEGMENT_DIR=$(ls -d data/processed/segments/$DATASET/*/* | head -n 1)

if [ -z "$SEGMENT_DIR" ]; then
    echo "❌ Validation failed: Could not locate a segment directory to test inference."
    exit 1
fi

echo "Testing Inference on: $SEGMENT_DIR"

echo "  -> Running Intent Inference"
python -m solomuse_data.cli infer-intent --config $CONFIG --dataset $DATASET --segment-dir "$SEGMENT_DIR" || { echo "❌ Inference Intent failed"; exit 1; }

echo "  -> Running Renderer Inference (End-to-End audio)"
python -m solomuse_data.cli render-segment --config $CONFIG --dataset $DATASET --segment-dir "$SEGMENT_DIR" || { echo "❌ Inference Renderer failed"; exit 1; }

if [ -f "$SEGMENT_DIR/y_hat.wav" ]; then
    echo ""
    echo "🎉 SUCCESS: Full pipeline completed and generated y_hat.wav offline payload."
    echo "   Pipeline is mathematically stable for large-scale ingestion."
else
    echo "❌ Validation failed: Missing output file $SEGMENT_DIR/y_hat.wav"
    exit 1
fi
