#!/bin/bash
# =============================================================================
# ManiSkill2 Zero-Shot Evaluation for MemoryVLA (Bridge Checkpoint)
# =============================================================================
#
# This script evaluates MemoryVLA trained on Bridge dataset (WidowX robot)
# on ManiSkill2 tasks (Franka Panda robot) to test cross-domain generalization.
#
# Usage:
#   bash script/eval/maniskill2/eval_maniskill2_bridge.sh <checkpoint_path>
#
# Example:
#   bash script/eval/maniskill2/eval_maniskill2_bridge.sh ./checkpoints/memvla-bridge
#
# =============================================================================

set -e

# Configuration
CHECKPOINT_PATH="${1:-./checkpoints/memvla-bridge}"
NUM_EPISODES="${2:-20}"
LOG_DIR="${3:-./logs/eval_maniskill2_bridge}"
GPU_ID="${4:-0}"

echo "============================================================"
echo "ManiSkill2 Zero-Shot Evaluation (Bridge Checkpoint)"
echo "============================================================"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Episodes per task: ${NUM_EPISODES}"
echo "Log directory: ${LOG_DIR}"
echo "GPU: ${GPU_ID}"
echo "============================================================"
echo ""
echo "Testing Cross-Domain Generalization:"
echo "  Train: Bridge (WidowX robot, real2sim)"
echo "  Test:  ManiSkill2 (Franka Panda, pure simulation)"
echo "============================================================"

# Create log directory
mkdir -p "${LOG_DIR}"

# Run evaluation
CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluation/maniskill2/eval_maniskill2_bridge.py \
    --checkpoint "${CHECKPOINT_PATH}" \
    --num-episodes ${NUM_EPISODES} \
    --max-steps 200 \
    --save-videos \
    --log-dir "${LOG_DIR}" \
    --seed 42 \
    2>&1 | tee "${LOG_DIR}/eval_log.txt"

echo ""
echo "============================================================"
echo "✅ Evaluation Complete!"
echo "Results: ${LOG_DIR}/results.json"
echo "Videos:  ${LOG_DIR}/videos/"
echo "============================================================"
