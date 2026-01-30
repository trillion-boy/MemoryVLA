#!/bin/bash
# CALVIN Evaluation Script for MemoryVLA
#
# CALVIN is a benchmark for language-conditioned long-horizon robot manipulation.
# This script evaluates MemoryVLA on CALVIN tasks without training (zero-shot).
#
# Usage:
#   bash script/eval/calvin/eval_calvin.sh
#
# Prerequisites:
#   1. Install CALVIN: git clone https://github.com/mees/calvin && cd calvin && pip install -e .
#   2. Start MemoryVLA server: python deploy.py --checkpoint <path>
#   3. Run this evaluation script

set -e

# Configuration
CHECKPOINT_PATH="${1:-/PATH/TO/YOUR/CHECKPOINT}"
PORT="${2:-6800}"
NUM_SEQUENCES="${3:-100}"
ENV_NAME="${4:-calvin_env_D}"
LOG_DIR="${5:-./logs/eval_calvin}"
SEED="${6:-42}"
GPU_ID="${7:-0}"

echo "=========================================="
echo "CALVIN Evaluation for MemoryVLA"
echo "=========================================="
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Port: ${PORT}"
echo "Num Sequences: ${NUM_SEQUENCES}"
echo "Environment: ${ENV_NAME}"
echo "Log Directory: ${LOG_DIR}"
echo "Seed: ${SEED}"
echo "GPU: ${GPU_ID}"
echo "=========================================="

# Create log directory
mkdir -p "${LOG_DIR}"

# Run evaluation
CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluation/calvin/eval_calvin.py \
    --checkpoint "${CHECKPOINT_PATH}" \
    --port ${PORT} \
    --num-sequences ${NUM_SEQUENCES} \
    --env-name "${ENV_NAME}" \
    --log-dir "${LOG_DIR}" \
    --seed ${SEED} \
    2>&1 | tee "${LOG_DIR}/eval_log.txt"

echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: ${LOG_DIR}"
echo "=========================================="
