#!/usr/bin/env bash
# Run all visualization scripts inside the CPU Docker container.
# Usage: bash scripts/run_all_viz.sh
set -e

IMAGE="airbnb-price-cpu:latest"
OUTPUTS_HOST="/mnt/nvme_data/linux_sys/aribnb_outputs"
REPO_HOST="$(cd "$(dirname "$0")/.." && pwd)"

docker run --rm \
  -v "$REPO_HOST":/workspace \
  -v "$OUTPUTS_HOST":/workspace/outputs \
  -w /workspace \
  "$IMAGE" \
  bash -c "
    pip install matplotlib --quiet
    python scripts/viz_model_comparison.py
    python scripts/viz_training_curves.py
    python scripts/viz_error_analysis.py
  "

echo ""
echo "All figures written to outputs/figures/"
