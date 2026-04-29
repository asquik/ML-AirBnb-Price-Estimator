#!/usr/bin/env bash
# run_proj_and_text_lora.sh
# Queue (sequential, one GPU):
#   1. FusionLoRA cleaned_bc 224px rank16 deep_256 proj256x128
#   2. TextLoRA   cleaned_bc rank16 deep_256
#   3. TextLoRA   cleaned_raw rank16 deep_256
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
#     bash scripts/run_proj_and_text_lora.sh

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
    echo "  [FAIL] $label — exit $?"
    FAIL=$((FAIL + 1))
    FAILED_NAMES+=("$label")
  fi
}

# ---------------------------------------------------------------------------
# 1. FusionLoRA cleaned_bc — with modality projection heads (text→256, image→128)
# ---------------------------------------------------------------------------

run_python "fusion_lora / 224px / rank16 / deep_256 / cleaned_bc / proj256x128 / lr_adapters=5e-6" \
  fusion_lora.py \
    --variant cleaned_bc \
    --image-size 224 \
    --lora-rank 16 \
    --fusion-head deep_256 \
    --text-proj-dim 256 \
    --image-proj-dim 128 \
    --lr-adapters 5e-6 \
    --batch-size 16 \
    --accum-steps 2 \
    --workers 4 \
    --run-name cleaned_bc_rank16_deep256_224px_proj256x128_lra5e6

# ---------------------------------------------------------------------------
# 2. TextLoRA cleaned_bc
# ---------------------------------------------------------------------------

run_python "text_lora / rank16 / deep_256 / cleaned_bc" \
  text_lora.py \
    --variant cleaned_bc \
    --lora-rank 16 \
    --fusion-head deep_256 \
    --batch-size 32 \
    --accum-steps 1 \
    --workers 4 \
    --run-name cleaned_bc_rank16_deep256

# ---------------------------------------------------------------------------
# 3. TextLoRA cleaned_raw
# ---------------------------------------------------------------------------

run_python "text_lora / rank16 / deep_256 / cleaned_raw" \
  text_lora.py \
    --variant cleaned_raw \
    --lora-rank 16 \
    --fusion-head deep_256 \
    --batch-size 32 \
    --accum-steps 1 \
    --workers 4 \
    --run-name cleaned_raw_rank16_deep256

# ---------------------------------------------------------------------------

banner "ALL DONE"
echo "  Passed: $PASS / $((PASS + FAIL))"
if [[ ${#FAILED_NAMES[@]} -gt 0 ]]; then
  echo "  Failed runs:"
  for n in "${FAILED_NAMES[@]}"; do echo "    - $n"; done
  exit 1
fi
echo "  All runs completed successfully."
