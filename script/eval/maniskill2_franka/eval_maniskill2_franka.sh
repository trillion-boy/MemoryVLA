#!/bin/bash
# Evaluation script for testing LIBERO-trained models on ManiSkill2 Franka Panda environments
# This enables cross-environment generalization testing with the same robot (Franka Panda)

# LIBERO checkpoint paths (trained on LIBERO with Franka Panda)
ckpt_paths=(
/PATH/TO/YOUR/LIBERO_CHECKPOINT_1
/PATH/TO/YOUR/LIBERO_CHECKPOINT_2
# Example: ./checkpoints/libero_spatial/step-032500-epoch-03-loss=0.0455.pt
)

gpu_id=0

# ManiSkill2 Franka Panda environments for generalization testing
# These environments use Franka Panda robot, same as LIBERO

for ckpt_path in "${ckpt_paths[@]}"; do
    eval_dir=$(dirname $(dirname ${ckpt_path}))/eval_maniskill2_franka/$(basename ${ckpt_path})
    mkdir -p ${eval_dir}

    # Common settings for Franka Panda in ManiSkill2
    robot=panda
    scene_name=defaults

    # Robot initial position (typical for ManiSkill2 tabletop tasks)
    robot_init_x=0.0
    robot_init_y=0.0

    # ============================================
    # Task 1: PickCube-v0
    # Simple cube picking task
    # ============================================
    echo "Evaluating PickCube-v0..."
    CUDA_VISIBLE_DEVICES=${gpu_id} python evaluation/simpler_env/simpler_env_inference.py \
      --ckpt-path ${ckpt_path} \
      --robot ${robot} \
      --policy-setup franka_panda \
      --control-freq 3 \
      --sim-freq 513 \
      --max-episode-steps 100 \
      --env-name PickCube-v0 \
      --scene-name ${scene_name} \
      --robot-init-x ${robot_init_x} ${robot_init_x} 1 \
      --robot-init-y ${robot_init_y} ${robot_init_y} 1 \
      --obj-variation-mode episode \
      --obj-episode-range 0 50 \
      --robot-init-rot-quat-center 0 0 0 1 \
      --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      | tee ${eval_dir}/PickCube.txt

    # ============================================
    # Task 2: StackCube-v0
    # Stack one cube on another
    # ============================================
    echo "Evaluating StackCube-v0..."
    CUDA_VISIBLE_DEVICES=${gpu_id} python evaluation/simpler_env/simpler_env_inference.py \
      --ckpt-path ${ckpt_path} \
      --robot ${robot} \
      --policy-setup franka_panda \
      --control-freq 3 \
      --sim-freq 513 \
      --max-episode-steps 150 \
      --env-name StackCube-v0 \
      --scene-name ${scene_name} \
      --robot-init-x ${robot_init_x} ${robot_init_x} 1 \
      --robot-init-y ${robot_init_y} ${robot_init_y} 1 \
      --obj-variation-mode episode \
      --obj-episode-range 0 50 \
      --robot-init-rot-quat-center 0 0 0 1 \
      --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      | tee ${eval_dir}/StackCube.txt

    # ============================================
    # Task 3: PickSingleYCB-v0
    # Pick various YCB objects
    # ============================================
    echo "Evaluating PickSingleYCB-v0..."
    CUDA_VISIBLE_DEVICES=${gpu_id} python evaluation/simpler_env/simpler_env_inference.py \
      --ckpt-path ${ckpt_path} \
      --robot ${robot} \
      --policy-setup franka_panda \
      --control-freq 3 \
      --sim-freq 513 \
      --max-episode-steps 100 \
      --env-name PickSingleYCB-v0 \
      --scene-name ${scene_name} \
      --robot-init-x ${robot_init_x} ${robot_init_x} 1 \
      --robot-init-y ${robot_init_y} ${robot_init_y} 1 \
      --obj-variation-mode episode \
      --obj-episode-range 0 50 \
      --robot-init-rot-quat-center 0 0 0 1 \
      --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      | tee ${eval_dir}/PickSingleYCB.txt

    # ============================================
    # Task 4: PickSingleEGAD-v0
    # Pick EGAD objects (more diverse shapes)
    # ============================================
    echo "Evaluating PickSingleEGAD-v0..."
    CUDA_VISIBLE_DEVICES=${gpu_id} python evaluation/simpler_env/simpler_env_inference.py \
      --ckpt-path ${ckpt_path} \
      --robot ${robot} \
      --policy-setup franka_panda \
      --control-freq 3 \
      --sim-freq 513 \
      --max-episode-steps 100 \
      --env-name PickSingleEGAD-v0 \
      --scene-name ${scene_name} \
      --robot-init-x ${robot_init_x} ${robot_init_x} 1 \
      --robot-init-y ${robot_init_y} ${robot_init_y} 1 \
      --obj-variation-mode episode \
      --obj-episode-range 0 50 \
      --robot-init-rot-quat-center 0 0 0 1 \
      --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      | tee ${eval_dir}/PickSingleEGAD.txt

    # ============================================
    # Task 5: PickClutterYCB-v0
    # Pick objects from cluttered scene
    # ============================================
    echo "Evaluating PickClutterYCB-v0..."
    CUDA_VISIBLE_DEVICES=${gpu_id} python evaluation/simpler_env/simpler_env_inference.py \
      --ckpt-path ${ckpt_path} \
      --robot ${robot} \
      --policy-setup franka_panda \
      --control-freq 3 \
      --sim-freq 513 \
      --max-episode-steps 150 \
      --env-name PickClutterYCB-v0 \
      --scene-name ${scene_name} \
      --robot-init-x ${robot_init_x} ${robot_init_x} 1 \
      --robot-init-y ${robot_init_y} ${robot_init_y} 1 \
      --obj-variation-mode episode \
      --obj-episode-range 0 50 \
      --robot-init-rot-quat-center 0 0 0 1 \
      --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      | tee ${eval_dir}/PickClutterYCB.txt

    echo "Done evaluating: ${ckpt_path}"
done

wait
echo "All evaluations complete!"
