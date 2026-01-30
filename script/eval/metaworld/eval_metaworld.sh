#!/bin/bash
# Meta-World Evaluation Script for MemoryVLA
#
# Meta-World is a benchmark with 50 distinct robotic manipulation tasks.
# This script evaluates MemoryVLA on Meta-World tasks without training (zero-shot).
#
# Usage:
#   bash script/eval/metaworld/eval_metaworld.sh
#
# Prerequisites:
#   1. Install Meta-World: pip install metaworld
#   2. Start MemoryVLA server: python deploy.py --checkpoint <path>
#   3. Run this evaluation script

set -e

# Configuration
CHECKPOINT_PATH="${1:-/PATH/TO/YOUR/CHECKPOINT}"
PORT="${2:-6800}"
BENCHMARK="${3:-MT10}"  # MT10, MT50, or specific task name
NUM_EPISODES="${4:-50}"
LOG_DIR="${5:-./logs/eval_metaworld}"
SEED="${6:-42}"
GPU_ID="${7:-0}"

echo "=========================================="
echo "Meta-World Evaluation for MemoryVLA"
echo "=========================================="
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Port: ${PORT}"
echo "Benchmark: ${BENCHMARK}"
echo "Episodes per task: ${NUM_EPISODES}"
echo "Log Directory: ${LOG_DIR}"
echo "Seed: ${SEED}"
echo "GPU: ${GPU_ID}"
echo "=========================================="

# Create log directory
mkdir -p "${LOG_DIR}"

# Run evaluation
CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluation/metaworld/eval_metaworld.py \
    --checkpoint "${CHECKPOINT_PATH}" \
    --port ${PORT} \
    --benchmark "${BENCHMARK}" \
    --num-episodes ${NUM_EPISODES} \
    --log-dir "${LOG_DIR}" \
    --seed ${SEED} \
    2>&1 | tee "${LOG_DIR}/eval_log.txt"

echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: ${LOG_DIR}"
echo "=========================================="
