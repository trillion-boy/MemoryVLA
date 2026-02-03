#!/bin/bash
# ============================================================================
# Cross-Environment Generalization Evaluation Script
# ============================================================================
# Tests LIBERO-trained MemoryVLA models on ManiSkill2 Franka Panda environments
#
# Key Point: Same robot (Franka Panda), different environments and objects
# - Training: LIBERO (Franka Panda) - kitchen/tabletop manipulation
# - Testing: ManiSkill2 (Franka Panda) - various pick/place tasks
#
# This tests the model's ability to generalize to UNSEEN environments
# while keeping the robot embodiment fixed.
# ============================================================================

# ============================================
# Configuration - MODIFY THESE
# ============================================

# Your LIBERO checkpoint path
# Examples:
#   - LIBERO-Spatial: ./checkpoints/libero_spatial/step-XXXXX.pt
#   - LIBERO-Object:  ./checkpoints/libero_object/step-XXXXX.pt
#   - LIBERO-Goal:    ./checkpoints/libero_goal/step-XXXXX.pt
#   - LIBERO-100:     ./checkpoints/libero_100/step-XXXXX.pt
CKPT_PATH="/PATH/TO/YOUR/LIBERO_CHECKPOINT.pt"

# Unnormalization key - must match your LIBERO training dataset
# Options: libero_spatial_no_noops, libero_object_no_noops, libero_goal_no_noops, libero_90_no_noops
UNNORM_KEY="libero_spatial_no_noops"

# GPU to use
GPU_ID=0

# Number of evaluation episodes per task
NUM_EPISODES=50

# Output directory
EVAL_DIR="./eval_results/maniskill2_generalization/$(basename ${CKPT_PATH%.pt})"
mkdir -p ${EVAL_DIR}

# ============================================
# ManiSkill2 Franka Panda Tasks
# ============================================
# All these tasks use Franka Panda robot (same as LIBERO)
# Reference: https://maniskill2.github.io/

echo "============================================"
echo "Cross-Environment Generalization Evaluation"
echo "============================================"
echo "Checkpoint: ${CKPT_PATH}"
echo "Unnorm Key: ${UNNORM_KEY}"
echo "Output Dir: ${EVAL_DIR}"
echo "============================================"

# Task 1: PickCube-v0
# Simple cube picking - tests basic grasping generalization
echo -e "\n[1/5] Evaluating PickCube-v0..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluation/maniskill2/maniskill2_evaluator.py \
    --ckpt-path ${CKPT_PATH} \
    --env-name "PickCube-v0" \
    --task-instruction "pick up the red cube" \
    --unnorm-key ${UNNORM_KEY} \
    --num-episodes ${NUM_EPISODES} \
    --max-steps 100 \
    --save-dir ${EVAL_DIR} \
    2>&1 | tee ${EVAL_DIR}/PickCube.log

# Task 2: StackCube-v0
# Stack cube on cube - tests sequential manipulation
echo -e "\n[2/5] Evaluating StackCube-v0..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluation/maniskill2/maniskill2_evaluator.py \
    --ckpt-path ${CKPT_PATH} \
    --env-name "StackCube-v0" \
    --task-instruction "stack the red cube on the green cube" \
    --unnorm-key ${UNNORM_KEY} \
    --num-episodes ${NUM_EPISODES} \
    --max-steps 150 \
    --save-dir ${EVAL_DIR} \
    2>&1 | tee ${EVAL_DIR}/StackCube.log

# Task 3: PickSingleYCB-v0
# Pick YCB objects - tests object shape generalization (unseen objects)
echo -e "\n[3/5] Evaluating PickSingleYCB-v0..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluation/maniskill2/maniskill2_evaluator.py \
    --ckpt-path ${CKPT_PATH} \
    --env-name "PickSingleYCB-v0" \
    --task-instruction "pick up the object" \
    --unnorm-key ${UNNORM_KEY} \
    --num-episodes ${NUM_EPISODES} \
    --max-steps 100 \
    --save-dir ${EVAL_DIR} \
    2>&1 | tee ${EVAL_DIR}/PickSingleYCB.log

# Task 4: PickSingleEGAD-v0
# Pick EGAD objects - tests more diverse shape generalization
echo -e "\n[4/5] Evaluating PickSingleEGAD-v0..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluation/maniskill2/maniskill2_evaluator.py \
    --ckpt-path ${CKPT_PATH} \
    --env-name "PickSingleEGAD-v0" \
    --task-instruction "pick up the object" \
    --unnorm-key ${UNNORM_KEY} \
    --num-episodes ${NUM_EPISODES} \
    --max-steps 100 \
    --save-dir ${EVAL_DIR} \
    2>&1 | tee ${EVAL_DIR}/PickSingleEGAD.log

# Task 5: PickClutterYCB-v0
# Pick from clutter - tests visual reasoning in complex scenes
echo -e "\n[5/5] Evaluating PickClutterYCB-v0..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluation/maniskill2/maniskill2_evaluator.py \
    --ckpt-path ${CKPT_PATH} \
    --env-name "PickClutterYCB-v0" \
    --task-instruction "pick up the target object" \
    --unnorm-key ${UNNORM_KEY} \
    --num-episodes ${NUM_EPISODES} \
    --max-steps 150 \
    --save-dir ${EVAL_DIR} \
    2>&1 | tee ${EVAL_DIR}/PickClutterYCB.log

# ============================================
# Summary
# ============================================
echo -e "\n============================================"
echo "Evaluation Complete!"
echo "============================================"
echo "Results saved to: ${EVAL_DIR}"
echo ""
echo "To extract summary:"
echo "  python script/eval/maniskill2_franka/extract_maniskill2_franka_results.py --eval-dir ${EVAL_DIR}"
echo "============================================"
