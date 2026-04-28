#!/usr/bin/env bash
# run_all_lora.sh — runs inside the container as a single process.
#
# Usage (from the HOST, launch one long-lived detached container):
#
#   docker rm -f lora_train 2>/dev/null; \
#   docker run -d --name lora_train \
#     --gpus all --shm-size=8g \
#     -v /home/admin/ML-AirBnb-Price-Estimator:/workspace \
#     -w /workspace \
#     -v /mnt/nvme_data/linux_sys/ml:/hf_cache \
#     -e HF_HOME=/hf_cache \
#     -e TRANSFORMERS_CACHE=/hf_cache/hub \
#     airbnb-gpu \
#     bash scripts/run_all_lora.sh
#
# Follow logs: docker logs -f lora_train
# Add --smoke-only to run smoke tests then exit.

set -euo pipefail

SMOKE_ONLY=0
for arg in "$@"; do
  [[ "$arg" == "--smoke-only" ]] && SMOKE_ONLY=1
done

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
# Phase 1: Smoke tests
# ---------------------------------------------------------------------------

banner "PHASE 1 — SMOKE TESTS"

run_python "smoke / fusion_lora / 224px / rank16 / deep_256" \
  fusion_lora.py --variant normal_bc --image-size 224 --lora-rank 16 --fusion-head deep_256 --smoke-test --workers 0

run_python "smoke / fusion_lora / 224px / rank16 / deep_256 / run32" \
  fusion_lora.py --variant normal_bc --image-size 224 --lora-rank 16 --fusion-head deep_256 --smoke-test --workers 0

run_python "smoke / text_lora / rank16 / deep_256" \
  text_lora.py --variant normal_bc --lora-rank 16 --fusion-head deep_256 --smoke-test --workers 0

run_python "smoke / image_lora / 336px / rank16 / deep_256 / run33" \
  image_lora.py --variant normal_bc --image-size 336 --lora-rank 16 --fusion-head deep_256 --smoke-test --workers 0 --batch-size 8

run_python "smoke / image_lora / 224px / rank16 / deep_256 / run33b" \
  image_lora.py --variant normal_bc --image-size 224 --lora-rank 16 --fusion-head deep_256 --smoke-test --workers 0

run_python "smoke / fusion_lora / 224px / rank16 / shallow_64 / run35" \
  fusion_lora.py --variant normal_bc --image-size 224 --lora-rank 16 --fusion-head shallow_64 --smoke-test --workers 0

run_python "smoke / fusion_lora / 224px / rank8 / deep_256 / run36" \
  fusion_lora.py --variant normal_bc --image-size 224 --lora-rank 8 --fusion-head deep_256 --smoke-test --workers 0

# ---------------------------------------------------------------------------
# Smoke summary
# ---------------------------------------------------------------------------

banner "SMOKE TEST SUMMARY"
echo "  Passed: $PASS / $((PASS + FAIL))"
if [[ $FAIL -gt 0 ]]; then
  echo "  Failed:"
  for n in "${FAILED_NAMES[@]}"; do echo "    - $n"; done
  echo
  echo "  Aborting — fix errors above before running full training."
  exit 1
fi
echo "  All smoke tests passed."

if [[ $SMOKE_ONLY -eq 1 ]]; then
  echo "  --smoke-only flag set. Exiting."
  exit 0
fi

# ---------------------------------------------------------------------------
# Phase 2: Full training (sequential — one GPU, same container)
# ---------------------------------------------------------------------------

banner "PHASE 2 — FULL TRAINING (sequential)"
echo "  Estimated total: ~12-16 hours depending on early stopping."
PASS=0; FAIL=0; FAILED_NAMES=()

run_python "run31 / fusion_lora / 224px / rank16 / deep_256 / priority1" \
  fusion_lora.py --variant normal_bc --image-size 224 --lora-rank 16 --fusion-head deep_256 \
  --batch-size 16 --accum-steps 2 --workers 4 --run-name priority1

run_python "run32 / fusion_lora / 224px / rank16 / deep_256 / priority2" \
  fusion_lora.py --variant normal_bc --image-size 224 --lora-rank 16 --fusion-head deep_256 \
  --batch-size 16 --accum-steps 2 --workers 4 --run-name priority2

run_python "run34 / text_lora / rank16 / deep_256 / ablation" \
  text_lora.py --variant normal_bc --lora-rank 16 --fusion-head deep_256 \
  --batch-size 32 --accum-steps 1 --workers 4 --run-name ablation

run_python "run33 / image_lora / 336px / rank16 / deep_256 / ablation" \
  image_lora.py --variant normal_bc --image-size 336 --lora-rank 16 --fusion-head deep_256 \
  --batch-size 8 --accum-steps 4 --workers 4 --run-name ablation

run_python "run33b / image_lora / 224px / rank16 / deep_256 / ablation" \
  image_lora.py --variant normal_bc --image-size 224 --lora-rank 16 --fusion-head deep_256 \
  --batch-size 16 --accum-steps 2 --workers 4 --run-name ablation

run_python "run35 / fusion_lora / 224px / rank16 / shallow_64 / head_ablation" \
  fusion_lora.py --variant normal_bc --image-size 224 --lora-rank 16 --fusion-head shallow_64 \
  --batch-size 16 --accum-steps 2 --workers 4 --run-name head_ablation

run_python "run36 / fusion_lora / 224px / rank8 / deep_256 / rank_ablation" \
  fusion_lora.py --variant normal_bc --image-size 224 --lora-rank 8 --fusion-head deep_256 \
  --batch-size 16 --accum-steps 2 --workers 4 --run-name rank_ablation

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

banner "ALL DONE"
echo "  Passed: $PASS / $((PASS + FAIL))"
if [[ ${#FAILED_NAMES[@]} -gt 0 ]]; then
  echo "  Failed runs:"
  for n in "${FAILED_NAMES[@]}"; do echo "    - $n"; done
  exit 1
fi
echo "  All runs completed successfully."
