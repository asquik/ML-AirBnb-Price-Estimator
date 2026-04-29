#!/usr/bin/env bash
# run_image_lora_cleaned.sh — ImageLoRA 224px, cleaned_bc then cleaned_raw.
#
# Usage (from HOST):
#
#   docker rm -f lora_train 2>/dev/null; \
#   docker run -d --name lora_train \
#     --gpus all --shm-size=8g \
#     -v /home/admin/ML-AirBnb-Price-Estimator:/workspace \
#     -v /mnt/nvme_data/linux_sys/ml_images:/workspace/images \
#     -v /mnt/nvme_data/linux_sys/ml:/hf_cache \
#     -w /workspace \
#     -e HF_HOME=/hf_cache \
#     -e TRANSFORMERS_CACHE=/hf_cache/hub \
#     airbnb-gpu:latest \
#     bash scripts/run_image_lora_cleaned.sh
#
# Follow logs: docker logs -f lora_train

set -euo pipefail

PASS=0
FAIL=0
FAILED_NAMES=()

banner() {
  echo
  echo "================================================================"
  echo "  $1"
  echo "================================================================"
}

run_python() {
  local label="$1"; shift
  banner "$label"
  if python scripts/models/$@; then
    echo "  [PASS] $label"
    PASS=$((PASS + 1))
  else
    local rc=$?
    echo "  [FAIL] $label — exit $rc"
    FAIL=$((FAIL + 1))
    FAILED_NAMES+=("$label")
  fi
}

# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

banner "SMOKE TESTS"

run_python "smoke / image_lora / 224px / rank16 / deep_256 / cleaned_bc" \
  image_lora.py --variant cleaned_bc --image-size 224 --lora-rank 16 \
  --fusion-head deep_256 --smoke-test --workers 0

run_python "smoke / image_lora / 224px / rank16 / deep_256 / cleaned_raw" \
  image_lora.py --variant cleaned_raw --image-size 224 --lora-rank 16 \
  --fusion-head deep_256 --smoke-test --workers 0

banner "SMOKE SUMMARY"
echo "  Passed: $PASS / $((PASS + FAIL))"
if [[ $FAIL -gt 0 ]]; then
  echo "  Failed:"
  for n in "${FAILED_NAMES[@]}"; do echo "    - $n"; done
  echo "  Aborting — fix errors before full training."
  exit 1
fi
echo "  All smoke tests passed."

PASS=0; FAIL=0; FAILED_NAMES=()

# ---------------------------------------------------------------------------
# Full training — sequential, one GPU
# ---------------------------------------------------------------------------

banner "FULL TRAINING"

run_python "image_lora / 224px / rank16 / deep_256 / cleaned_bc" \
  image_lora.py --variant cleaned_bc --image-size 224 --lora-rank 16 \
  --fusion-head deep_256 --lr-head 5e-5 \
  --batch-size 16 --accum-steps 2 --workers 4 \
  --run-name cleaned_bc_rank16_deep256_224px

run_python "image_lora / 224px / rank16 / deep_256 / cleaned_raw" \
  image_lora.py --variant cleaned_raw --image-size 224 --lora-rank 16 \
  --fusion-head deep_256 --lr-head 1e-6 \
  --batch-size 16 --accum-steps 2 --workers 4 \
  --run-name cleaned_raw_rank16_deep256_224px

# ---------------------------------------------------------------------------

banner "ALL DONE"
echo "  Passed: $PASS / $((PASS + FAIL))"
if [[ ${#FAILED_NAMES[@]} -gt 0 ]]; then
  echo "  Failed runs:"
  for n in "${FAILED_NAMES[@]}"; do echo "    - $n"; done
  exit 1
fi
echo "  All runs completed successfully."
