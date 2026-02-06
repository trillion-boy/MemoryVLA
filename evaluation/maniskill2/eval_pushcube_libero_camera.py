"""
ManiSkill2 PushCube evaluation with LIBERO camera configuration.

Tests MemoryVLA model on PushCube-v1 environment using LIBERO-style camera settings.
Tracks terminated vs truncated episode endings for detailed analysis.

Usage (in Colab):
    # After loading the model as 'vla'
    from evaluation.maniskill2.eval_pushcube_libero_camera import run_evaluation
    results = run_evaluation(vla, num_trials=8)
"""

import gymnasium as gym
import mani_skill.envs
import numpy as np
from PIL import Image
import torch
import imageio
import os
from sapien import Pose
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple


# Default configuration
DEFAULT_CONFIG = {
    "save_dir": "/content/eval_results_pushcube_libero",
    "num_trials": 8,
    "max_steps": 100,
    "unnorm_key": "libero_object_no_noops",
    "task_instruction": "push the cube to the target",
    "cfg_scale": 1.5,
    "use_ddim": True,
    "num_ddim_steps": 10,
}

# LIBERO camera configuration
LIBERO_CAMERA_CONFIG = {
    "position": [0.5, 0, 1.2],
    "quaternion": [0, -0.433189, 0, 0.901303],  # wxyz format for Sapien
    "fovy": 0.7854,  # 45 degrees in radians
}


def get_rgb_from_obs(obs: dict) -> np.ndarray:
    """
    Extract RGB image from ManiSkill observation.

    Handles both ManiSkill (mani_skill) and older observation formats.

    Args:
        obs: Observation dictionary from environment

    Returns:
        RGB image as numpy array (H, W, 3), uint8
    """
    # ManiSkill (new) format with sensor_data
    if "sensor_data" in obs:
        for cam_name in obs["sensor_data"]:
            if "rgb" in obs["sensor_data"][cam_name]:
                rgb = obs["sensor_data"][cam_name]["rgb"]
                if isinstance(rgb, torch.Tensor):
                    rgb = rgb.cpu().numpy()
                if rgb.ndim == 4:
                    rgb = rgb[0]  # Remove batch dimension
                if rgb.max() <= 1.0:
                    rgb = (rgb * 255).astype(np.uint8)
                return rgb

    # ManiSkill2 format with image
    if "image" in obs:
        for cam_name in ["base_camera", "hand_camera"]:
            if cam_name in obs["image"] and "rgb" in obs["image"][cam_name]:
                rgb = obs["image"][cam_name]["rgb"]
                if isinstance(rgb, torch.Tensor):
                    rgb = rgb.cpu().numpy()
                if rgb.dtype != np.uint8:
                    rgb = (rgb * 255).astype(np.uint8)
                return rgb

    raise ValueError("No RGB observation found in obs dict")


def setup_libero_camera(env, config: Optional[dict] = None):
    """
    Configure camera to LIBERO-style viewpoint.

    Args:
        env: ManiSkill environment
        config: Camera configuration dict (optional, uses LIBERO_CAMERA_CONFIG if None)
    """
    if config is None:
        config = LIBERO_CAMERA_CONFIG

    base_env = env.unwrapped
    camera = base_env._sensors['base_camera']
    sapien_cam = camera.camera._render_cameras[0]

    # Set LIBERO camera pose
    new_pose = Pose(
        p=config["position"],
        q=config["quaternion"]
    )
    sapien_cam.set_local_pose(new_pose)
    sapien_cam.set_fovy(config["fovy"])


def run_episode(
    env,
    vla,
    task_instruction: str,
    max_steps: int,
    unnorm_key: str,
    cfg_scale: float = 1.5,
    use_ddim: bool = True,
    num_ddim_steps: int = 10,
) -> Dict[str, Any]:
    """
    Run a single evaluation episode.

    Args:
        env: ManiSkill environment
        vla: MemoryVLA model
        task_instruction: Natural language task
        max_steps: Maximum steps per episode
        unnorm_key: Action unnormalization key
        cfg_scale: Classifier-free guidance scale
        use_ddim: Whether to use DDIM sampling
        num_ddim_steps: Number of DDIM steps

    Returns:
        Dictionary with episode results including:
        - success: bool
        - terminated: bool
        - truncated: bool
        - total_reward: float
        - steps: int
        - frames: list of RGB frames
        - end_reason: str describing how episode ended
    """
    obs, info = env.reset()
    setup_libero_camera(env)

    # Re-render after camera setup and get updated observation
    env.unwrapped.render_sensors()
    obs = env.unwrapped.get_obs()

    frames = []
    total_reward = 0.0
    terminated = False
    truncated = False

    for step in range(max_steps):
        # Get RGB observation
        rgb = get_rgb_from_obs(obs)
        pil_image = Image.fromarray(rgb)
        frames.append(rgb.copy())

        # Predict action
        episode_first = 'True' if step == 0 else 'False'

        actions, _ = vla.predict_action(
            image=pil_image,
            instruction=task_instruction,
            unnorm_key=unnorm_key,
            cfg_scale=cfg_scale,
            use_ddim=use_ddim,
            num_ddim_steps=num_ddim_steps,
            episode_first_frame=episode_first,
        )

        # Execute action
        action = actions[0]
        obs, reward, terminated, truncated, info = env.step(
            torch.tensor(action).unsqueeze(0)
        )

        # Handle tensor types
        if isinstance(reward, torch.Tensor):
            reward = reward.item()
        if isinstance(terminated, torch.Tensor):
            terminated = terminated.item()
        if isinstance(truncated, torch.Tensor):
            truncated = truncated.item()

        total_reward += reward

        # Check for episode end
        if terminated or truncated:
            # Capture final frame
            rgb = get_rgb_from_obs(obs)
            frames.append(rgb.copy())
            break

    # Determine success
    success = info.get('success', False)
    if isinstance(success, torch.Tensor):
        success = success.item()

    # Determine end reason
    if success:
        end_reason = "SUCCESS"
    elif terminated:
        end_reason = "TERMINATED (task failed)"
    elif truncated:
        end_reason = "TRUNCATED (max steps reached)"
    else:
        end_reason = "MAX_STEPS (loop ended)"

    return {
        "success": success,
        "terminated": terminated,
        "truncated": truncated,
        "total_reward": total_reward,
        "steps": step + 1,
        "frames": frames,
        "end_reason": end_reason,
    }


def run_evaluation(
    vla,
    num_trials: int = 8,
    max_steps: int = 100,
    task_instruction: str = "push the cube to the target",
    unnorm_key: str = "libero_object_no_noops",
    save_dir: str = "/content/eval_results_pushcube_libero",
    cfg_scale: float = 1.5,
    use_ddim: bool = True,
    num_ddim_steps: int = 10,
    save_videos: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run full evaluation with multiple trials.

    Args:
        vla: MemoryVLA model
        num_trials: Number of evaluation episodes (default: 8)
        max_steps: Maximum steps per episode (default: 100)
        task_instruction: Natural language task
        unnorm_key: Action unnormalization key
        save_dir: Directory to save results and videos
        cfg_scale: Classifier-free guidance scale
        use_ddim: Whether to use DDIM sampling
        num_ddim_steps: Number of DDIM steps
        save_videos: Whether to save episode videos
        verbose: Print detailed progress

    Returns:
        Dictionary with aggregated results
    """
    os.makedirs(save_dir, exist_ok=True)

    # Create environment
    if verbose:
        print("Creating PushCube environment with LIBERO camera...")

    env = gym.make(
        "PushCube-v1",
        obs_mode="rgbd",
        control_mode="pd_ee_delta_pose",
        robot_uids="panda",
        num_envs=1,
        render_mode="rgb_array",
        max_episode_steps=max_steps,
    )

    # Run trials
    results = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"LIBERO Camera Evaluation: PushCube-v1")
        print(f"{'='*60}")
        print(f"Task: {task_instruction}")
        print(f"Trials: {num_trials}")
        print(f"Max steps: {max_steps}")
        print(f"Unnorm key: {unnorm_key}")
        print(f"Camera: LIBERO style (position={LIBERO_CAMERA_CONFIG['position']})")
        print(f"{'='*60}\n")

    for trial in range(num_trials):
        result = run_episode(
            env=env,
            vla=vla,
            task_instruction=task_instruction,
            max_steps=max_steps,
            unnorm_key=unnorm_key,
            cfg_scale=cfg_scale,
            use_ddim=use_ddim,
            num_ddim_steps=num_ddim_steps,
        )

        result["trial"] = trial
        results.append(result)

        # Save video
        if save_videos:
            status_str = "success" if result["success"] else "failure"
            end_tag = "term" if result["terminated"] else ("trunc" if result["truncated"] else "max")
            video_path = os.path.join(
                save_dir,
                f"trial_{trial:02d}_{status_str}_{end_tag}.mp4"
            )
            imageio.mimsave(video_path, result["frames"], fps=10)

        # Print progress
        if verbose:
            success_count = sum(r["success"] for r in results)
            status = "SUCCESS" if result["success"] else "failure"

            print(
                f"Trial {trial+1:2d}/{num_trials}: {status:10s} | "
                f"Reward: {result['total_reward']:7.3f} | "
                f"Steps: {result['steps']:3d} | "
                f"End: {result['end_reason']:30s} | "
                f"Success: {success_count}/{trial+1}"
            )

    env.close()

    # Compute statistics
    total_success = sum(r["success"] for r in results)
    total_terminated = sum(r["terminated"] and not r["success"] for r in results)
    total_truncated = sum(r["truncated"] for r in results)
    avg_reward = np.mean([r["total_reward"] for r in results])
    avg_steps = np.mean([r["steps"] for r in results])

    summary = {
        "env_name": "PushCube-v1",
        "camera_style": "LIBERO",
        "num_trials": num_trials,
        "max_steps": max_steps,
        "success_count": total_success,
        "success_rate": 100 * total_success / num_trials,
        "terminated_count": total_terminated,
        "truncated_count": total_truncated,
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
        "results": results,
        "save_dir": save_dir,
    }

    # Print summary
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluation Summary - PushCube (LIBERO Camera)")
        print(f"{'='*60}")
        print(f"Success Rate: {total_success}/{num_trials} ({summary['success_rate']:.1f}%)")
        print(f"Avg Reward:   {avg_reward:.3f}")
        print(f"Avg Steps:    {avg_steps:.1f}")
        print(f"{'='*60}")
        print(f"\nEpisode Endings:")
        print(f"  - SUCCESS:    {total_success:3d} episodes")
        print(f"  - TERMINATED: {total_terminated:3d} episodes (task failed before max steps)")
        print(f"  - TRUNCATED:  {total_truncated:3d} episodes (reached max steps)")
        print(f"{'='*60}")
        if save_videos:
            print(f"Videos saved to: {save_dir}")

    # Save results to file
    result_file = os.path.join(save_dir, "evaluation_results.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(result_file, "w") as f:
        f.write(f"PushCube Evaluation Results (LIBERO Camera) - {timestamp}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Environment: PushCube-v1\n")
        f.write(f"  Camera: LIBERO style\n")
        f.write(f"    - Position: {LIBERO_CAMERA_CONFIG['position']}\n")
        f.write(f"    - Quaternion: {LIBERO_CAMERA_CONFIG['quaternion']}\n")
        f.write(f"    - FOV: {LIBERO_CAMERA_CONFIG['fovy']}\n")
        f.write(f"  Task: {task_instruction}\n")
        f.write(f"  Unnorm key: {unnorm_key}\n")
        f.write(f"  Max steps: {max_steps}\n")
        f.write(f"  CFG scale: {cfg_scale}\n")
        f.write(f"  DDIM steps: {num_ddim_steps}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Success Rate: {total_success}/{num_trials} ({summary['success_rate']:.1f}%)\n")
        f.write(f"  Avg Reward: {avg_reward:.3f}\n")
        f.write(f"  Avg Steps: {avg_steps:.1f}\n\n")
        f.write(f"Episode Endings:\n")
        f.write(f"  SUCCESS:    {total_success}\n")
        f.write(f"  TERMINATED: {total_terminated}\n")
        f.write(f"  TRUNCATED:  {total_truncated}\n\n")
        f.write(f"Per-trial Results:\n")
        f.write(f"{'-'*60}\n")
        for r in results:
            f.write(
                f"Trial {r['trial']+1:2d}: "
                f"{'SUCCESS' if r['success'] else 'FAILURE':8s} | "
                f"Reward: {r['total_reward']:7.3f} | "
                f"Steps: {r['steps']:3d} | "
                f"{r['end_reason']}\n"
            )

    return summary


# For direct execution in Colab
if __name__ == "__main__":
    print("This script is designed to be run in Colab with a loaded VLA model.")
    print("\nUsage:")
    print("  from evaluation.maniskill2.eval_pushcube_libero_camera import run_evaluation")
    print("  results = run_evaluation(vla, num_trials=8)")
