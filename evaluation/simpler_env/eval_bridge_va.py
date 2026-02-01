#!/usr/bin/env python3
"""
SimplerEnv-Bridge Visual Aggregation (VA) Evaluation

This script evaluates MemoryVLA on SimplerEnv-Bridge tasks with Visual Aggregation:
- VM (Visual Matching): Uses rgb_overlay to match real-world appearance
- VA (Visual Aggregation): Uses different scenes WITHOUT overlay for generalization testing

Key insight: VA tests visual generalization while keeping robot (WidowX) and action space same.
This isolates visual domain gap from cross-embodiment challenges.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from argparse import Namespace

# Environment setup
os.environ["DISPLAY"] = ""
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"


def setup_simpler_env():
    """Setup SimplerEnv environment variables"""
    # Try to find SimplerEnv installation
    simpler_path = None
    possible_paths = [
        "/content/SimplerEnv",
        os.path.expanduser("~/SimplerEnv"),
        "/home/user/SimplerEnv",
    ]
    for p in possible_paths:
        if os.path.exists(p):
            simpler_path = p
            break

    if simpler_path:
        if simpler_path not in sys.path:
            sys.path.insert(0, simpler_path)
        print(f"   SimplerEnv found at: {simpler_path}")
    return simpler_path


def load_model(ckpt_path: str, policy_setup: str = "widowx_bridge"):
    """Load MemoryVLA model for SimplerEnv evaluation"""
    from evaluation.simpler_env.vla_policy import VLAInference

    model = VLAInference(
        saved_model_path=ckpt_path,
        policy_setup=policy_setup,
        cfg_scale=1.5,
        use_ddim=True,
        num_ddim_steps=10,
        action_ensemble=True,
        adaptive_ensemble_alpha=0.1,
    )
    return model


def get_bridge_tasks():
    """
    Get Bridge evaluation tasks

    Returns list of (env_name, instruction) tuples
    """
    # Core Bridge tasks from SimplerEnv
    tasks = [
        # Pick and place tasks
        ("PutCarrotOnPlateInScene-v0", "put carrot on plate"),
        ("PutSpoonOnTableClothInScene-v0", "put spoon on table cloth"),
        ("StackGreenCubeOnYellowCubeBakedTexInScene-v0", "stack green cube on yellow cube"),
        ("PutEggplantInBasketScene-v0", "put eggplant in basket"),
    ]
    return tasks


def get_scene_configs():
    """
    Get scene configurations for VM and VA testing

    VM (Visual Matching): Uses rgb_overlay to match real appearance
    VA (Visual Aggregation): Different scenes without overlay
    """
    configs = {
        # Visual Matching - with overlay for baseline
        "vm": {
            "scene_name": "bridge_table_1_v1",
            "rgb_overlay_path": "real",  # Will be resolved to actual overlay path
            "description": "Visual Matching (baseline with overlay)"
        },
        # Visual Aggregation - without overlay, different scene
        "va_scene1": {
            "scene_name": "bridge_table_1_v1",
            "rgb_overlay_path": None,  # No overlay - pure sim appearance
            "description": "VA: Default scene, no overlay"
        },
        "va_scene2": {
            "scene_name": "bridge_table_1_v2",
            "rgb_overlay_path": None,
            "description": "VA: Alternative scene, no overlay"
        },
    }
    return configs


def run_single_episode(
    model,
    env_name: str,
    scene_name: str,
    instruction: str,
    rgb_overlay_path: str = None,
    max_steps: int = 80,
    save_video: bool = True,
    video_dir: str = "./eval_videos",
):
    """
    Run a single evaluation episode

    Args:
        model: VLAInference model
        env_name: SimplerEnv environment name
        scene_name: Scene configuration name
        instruction: Task instruction
        rgb_overlay_path: Path to RGB overlay image (None for VA)
        max_steps: Maximum episode steps
        save_video: Whether to save video
        video_dir: Directory for saving videos

    Returns:
        success: bool, whether task was successful
        frames: list of frames for video
    """
    from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

    robot_name = "widowx"
    control_mode = get_robot_control_mode(robot_name, "MemoryVLA")

    # Build environment
    env = build_maniskill2_env(
        env_name,
        obs_mode="rgbd",
        robot=robot_name,
        sim_freq=513,
        control_mode=control_mode,
        control_freq=3,
        max_episode_steps=max_steps,
        scene_name=scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=rgb_overlay_path,
    )

    # Reset environment
    obs, _ = env.reset()

    # Get task instruction
    task_description = instruction or env.get_language_instruction()

    # Initialize
    model.reset(task_description)
    frames = []
    success = False

    image = get_image_from_maniskill2_obs_dict(env, obs)
    episode_first_frame = 'True'

    for step in range(max_steps):
        # Save frame
        if save_video:
            frames.append(image.copy())

        # Get action from model
        raw_action, action = model.step(
            image,
            task_description,
            episode_first_frame=episode_first_frame,
        )
        episode_first_frame = 'False'

        # Check for termination signal
        if bool(action["terminate_episode"][0] > 0):
            break

        # Step environment
        env_action = np.concatenate([
            action["world_vector"],
            action["rot_axangle"],
            action["gripper"]
        ])
        obs, reward, done, truncated, info = env.step(env_action)

        if done:
            success = True
            break
        if truncated:
            break

        image = get_image_from_maniskill2_obs_dict(env, obs)

    env.close()
    return success, frames


def evaluate_task(
    model,
    env_name: str,
    instruction: str,
    scene_config: dict,
    num_episodes: int = 5,
    max_steps: int = 80,
    save_video: bool = True,
    video_dir: str = "./eval_videos",
):
    """
    Evaluate a single task with given scene configuration
    """
    scene_name = scene_config["scene_name"]
    rgb_overlay_path = scene_config.get("rgb_overlay_path")
    description = scene_config.get("description", "")

    print(f"\n{'='*60}")
    print(f"Task: {env_name}")
    print(f"Config: {description}")
    print(f"Scene: {scene_name}, Overlay: {rgb_overlay_path is not None}")
    print(f"{'='*60}")

    successes = []

    for ep in range(num_episodes):
        try:
            success, frames = run_single_episode(
                model=model,
                env_name=env_name,
                scene_name=scene_name,
                instruction=instruction,
                rgb_overlay_path=rgb_overlay_path,
                max_steps=max_steps,
                save_video=save_video,
                video_dir=video_dir,
            )
            successes.append(float(success))
            status = "Success" if success else "Failure"
            print(f"  Episode {ep+1}/{num_episodes}: {status}")

        except Exception as e:
            print(f"  Episode {ep+1}/{num_episodes}: Error - {e}")
            successes.append(0.0)

    success_rate = np.mean(successes) if successes else 0.0
    print(f"  Success Rate: {success_rate*100:.1f}%")

    return {
        "success_rate": success_rate,
        "successes": successes,
        "env_name": env_name,
        "scene_config": description,
    }


def run_va_evaluation(
    ckpt_path: str,
    num_episodes: int = 5,
    max_steps: int = 80,
    save_video: bool = True,
    test_configs: list = None,
):
    """
    Run Visual Aggregation evaluation on SimplerEnv-Bridge

    Args:
        ckpt_path: Path to MemoryVLA checkpoint
        num_episodes: Episodes per task/config
        max_steps: Max steps per episode
        save_video: Save evaluation videos
        test_configs: List of config names to test (default: all)
    """
    print("="*60)
    print("SimplerEnv-Bridge Visual Aggregation Evaluation")
    print("="*60)
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Episodes per config: {num_episodes}")
    print(f"  Max steps: {max_steps}")
    print("="*60)

    # Setup
    setup_simpler_env()

    # Load model
    print("\n1. Loading model...")
    model = load_model(ckpt_path, policy_setup="widowx_bridge")
    print("   Model loaded successfully")

    # Get tasks and configs
    tasks = get_bridge_tasks()
    scene_configs = get_scene_configs()

    if test_configs is None:
        test_configs = ["va_scene1"]  # Default to VA without overlay

    # Run evaluation
    all_results = {}

    for config_name in test_configs:
        if config_name not in scene_configs:
            print(f"Warning: Unknown config {config_name}, skipping")
            continue

        config = scene_configs[config_name]
        config_results = []

        print(f"\n{'#'*60}")
        print(f"# Testing: {config['description']}")
        print(f"{'#'*60}")

        for env_name, instruction in tasks:
            try:
                result = evaluate_task(
                    model=model,
                    env_name=env_name,
                    instruction=instruction,
                    scene_config=config,
                    num_episodes=num_episodes,
                    max_steps=max_steps,
                    save_video=save_video,
                )
                config_results.append(result)
            except Exception as e:
                print(f"Error evaluating {env_name}: {e}")
                config_results.append({
                    "success_rate": 0.0,
                    "env_name": env_name,
                    "error": str(e),
                })

        all_results[config_name] = config_results

    # Summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)

    for config_name, results in all_results.items():
        config = scene_configs.get(config_name, {})
        print(f"\n{config.get('description', config_name)}:")

        total_successes = []
        for r in results:
            sr = r.get("success_rate", 0) * 100
            print(f"  {r['env_name']}: {sr:.1f}%")
            if "successes" in r:
                total_successes.extend(r["successes"])

        if total_successes:
            overall = np.mean(total_successes) * 100
            print(f"  Overall: {overall:.1f}%")

    print("="*60)
    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SimplerEnv-Bridge VA Evaluation")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to MemoryVLA checkpoint")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Episodes per task/config")
    parser.add_argument("--steps", type=int, default=80,
                        help="Max steps per episode")
    parser.add_argument("--configs", nargs="+", default=["va_scene1"],
                        choices=["vm", "va_scene1", "va_scene2"],
                        help="Scene configs to test")
    parser.add_argument("--no-video", action="store_true",
                        help="Disable video saving")
    args = parser.parse_args()

    run_va_evaluation(
        ckpt_path=args.ckpt,
        num_episodes=args.episodes,
        max_steps=args.steps,
        save_video=not args.no_video,
        test_configs=args.configs,
    )
