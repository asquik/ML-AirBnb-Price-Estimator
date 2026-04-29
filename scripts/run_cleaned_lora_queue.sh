#!/usr/bin/env bash
# run_cleaned_lora_queue.sh — ImageLoRA cleaned_bc, then FusionLoRA cleaned_bc + cleaned_bc with projections.
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
#     bash scripts/run_cleaned_lora_queue.sh
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
    echo "  [FAIL] $label — exit $?"
    FAIL=$((FAIL + 1))
    FAILED_NAMES+=("$label")
  fi
}

# ---------------------------------------------------------------------------
# 1. ImageLoRA 224px cleaned_bc
#    (cleaned_raw skipped — image_lora uses MSE loss which explodes on raw $ targets)
# ---------------------------------------------------------------------------

run_python "image_lora / 224px / rank16 / deep_256 / cleaned_bc" \
  image_lora.py \
    --variant cleaned_bc \
    --image-size 224 \
    --lora-rank 16 \
    --fusion-head deep_256 \
    --lr-head 5e-5 \
    --batch-size 16 \
    --accum-steps 2 \
    --workers 4 \
    --run-name cleaned_bc_rank16_deep256_224px

# ---------------------------------------------------------------------------
# 2. FusionLoRA 224px cleaned_bc — standard head (baseline for projection ablation)
# ---------------------------------------------------------------------------

run_python "fusion_lora / 224px / rank16 / deep_256 / cleaned_bc / no-proj" \
  fusion_lora.py \
    --variant cleaned_bc \
    --image-size 224 \
    --lora-rank 16 \
    --fusion-head deep_256 \
    --lr-head 5e-5 \
    --batch-size 16 \
    --accum-steps 2 \
    --workers 4 \
    --run-name cleaned_bc_rank16_deep256_224px

# ---------------------------------------------------------------------------
# 3. FusionLoRA 224px cleaned_bc — with modality projection heads (text→256, image→128)
# ---------------------------------------------------------------------------

run_python "fusion_lora / 224px / rank16 / deep_256 / cleaned_bc / proj256-128" \
  fusion_lora.py \
    --variant cleaned_bc \
    --image-size 224 \
    --lora-rank 16 \
    --fusion-head deep_256 \
    --lr-head 5e-5 \
    --text-proj-dim 256 \
    --image-proj-dim 128 \
    --batch-size 16 \
    --accum-steps 2 \
    --workers 4 \
    --run-name cleaned_bc_rank16_deep256_224px_proj256x128

# ---------------------------------------------------------------------------

banner "ALL DONE"
echo "  Passed: $PASS / $((PASS + FAIL))"
if [[ ${#FAILED_NAMES[@]} -gt 0 ]]; then
  echo "  Failed runs:"
  for n in "${FAILED_NAMES[@]}"; do echo "    - $n"; done
  exit 1
fi
echo "  All runs completed successfully."
