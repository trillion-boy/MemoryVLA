"""
ManiSkill2 PushCube evaluation with Training-Free Depth-Aware Pipeline.

Combines depth feature injection (Step A) and action correction (Step B)
to enhance MemoryVLA's spatial perception for cross-environment generalization.

Usage (in Colab):
    # 1. Load VLA model
    from vla import load_vla
    vla = load_vla(...)

    # 2. (Optional) Load Depth Anything V2
    from evaluation.maniskill2.depth_aware_pipeline import load_depth_anything_v2
    depth_model = load_depth_anything_v2("Small")

    # 3. Run evaluation
    from evaluation.maniskill2.eval_pushcube_depth_aware import run_evaluation
    results = run_evaluation(vla, depth_model=depth_model, num_trials=8)
"""

import gymnasium as gym
import mani_skill.envs
import numpy as np
from PIL import Image
import torch
import imageio
import os
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

from evaluation.maniskill2.depth_aware_pipeline import DepthAwarePipeline


# Default configuration
DEFAULT_CONFIG = {
    "save_dir": "/content/eval_results_pushcube_depth",
    "num_trials": 8,
    "max_steps": 100,
    "unnorm_key": "libero_object_no_noops",
    "task_instruction": (
        "Push the blue cube forward along the table toward the red and white "
        "circular target. Approach the cube from behind and slide it forward "
        "until it reaches the target zone."
    ),
    "cfg_scale": 1.5,
    "use_ddim": True,
    "num_ddim_steps": 10,
    # Depth pipeline parameters
    "alpha": 0.3,
    "use_depth_injection": True,
    "use_action_correction": True,
}


def get_rgb_from_obs(obs: dict) -> Tuple[np.ndarray, str]:
    """Extract RGB image from ManiSkill observation."""
    if "sensor_data" in obs:
        for cam_name in ["base_camera", "hand_camera"]:
            if cam_name in obs["sensor_data"] and "rgb" in obs["sensor_data"][cam_name]:
                rgb = obs["sensor_data"][cam_name]["rgb"]
                if isinstance(rgb, torch.Tensor):
                    rgb = rgb.cpu().numpy()
                if rgb.ndim == 4:
                    rgb = rgb[0]
                if rgb.max() <= 1.0:
                    rgb = (rgb * 255).astype(np.uint8)
                return rgb, cam_name

        cam_name = list(obs["sensor_data"].keys())[0]
        if "rgb" in obs["sensor_data"][cam_name]:
            rgb = obs["sensor_data"][cam_name]["rgb"]
            if isinstance(rgb, torch.Tensor):
                rgb = rgb.cpu().numpy()
            if rgb.ndim == 4:
                rgb = rgb[0]
            if rgb.max() <= 1.0:
                rgb = (rgb * 255).astype(np.uint8)
            return rgb, cam_name

    if "image" in obs:
        for cam_name in ["base_camera", "hand_camera"]:
            if cam_name in obs["image"] and "rgb" in obs["image"][cam_name]:
                rgb = obs["image"][cam_name]["rgb"]
                if isinstance(rgb, torch.Tensor):
                    rgb = rgb.cpu().numpy()
                if rgb.dtype != np.uint8:
                    rgb = (rgb * 255).astype(np.uint8)
                return rgb, cam_name

    raise ValueError("No RGB observation found in obs dict")


def run_episode(
    env,
    pipeline: DepthAwarePipeline,
    task_instruction: str,
    max_steps: int,
    unnorm_key: str,
    cfg_scale: float = 1.5,
    use_ddim: bool = True,
    num_ddim_steps: int = 10,
    save_depth_vis: bool = False,
) -> Dict[str, Any]:
    """
    Run a single evaluation episode with the depth-aware pipeline.

    Args:
        env: ManiSkill environment
        pipeline: DepthAwarePipeline instance
        task_instruction: Natural language task
        max_steps: Maximum steps per episode
        unnorm_key: Action unnormalization key
        cfg_scale: Classifier-free guidance scale
        use_ddim: Whether to use DDIM sampling
        num_ddim_steps: Number of DDIM steps
        save_depth_vis: Whether to save depth visualization frames

    Returns:
        Dictionary with episode results
    """
    obs, info = env.reset()

    frames = []
    depth_frames = []
    total_reward = 0.0
    terminated = False
    truncated = False
    camera_name = None

    for step in range(max_steps):
        # Get RGB observation
        rgb, camera_name = get_rgb_from_obs(obs)
        pil_image = Image.fromarray(rgb)
        frames.append(rgb.copy())

        # Save depth visualization if requested
        if save_depth_vis:
            gt_depth = pipeline.get_depth_from_obs(obs)
            if gt_depth is not None:
                depth_vis = pipeline.depth_to_colormap(gt_depth)
                depth_frames.append(np.array(depth_vis))

        # Predict action WITH depth pipeline
        episode_first = 'True' if step == 0 else 'False'

        actions, _ = pipeline.predict_action(
            image=pil_image,
            instruction=task_instruction,
            obs=obs,  # Provides GT depth for Step B + fallback for Step A
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

        if terminated or truncated:
            rgb, _ = get_rgb_from_obs(obs)
            frames.append(rgb.copy())
            break

    # Determine success
    success = info.get('success', False)
    if isinstance(success, torch.Tensor):
        success = success.item()

    if success:
        end_reason = "SUCCESS"
    elif terminated:
        end_reason = "TERMINATED (task failed)"
    elif truncated:
        end_reason = "TRUNCATED (max steps reached)"
    else:
        end_reason = "MAX_STEPS (loop ended)"

    result = {
        "success": success,
        "terminated": terminated,
        "truncated": truncated,
        "total_reward": total_reward,
        "steps": step + 1,
        "frames": frames,
        "end_reason": end_reason,
        "camera_name": camera_name,
    }

    if save_depth_vis and depth_frames:
        result["depth_frames"] = depth_frames

    return result


def run_evaluation(
    vla,
    depth_model=None,
    num_trials: int = 8,
    max_steps: int = 100,
    task_instruction: str = DEFAULT_CONFIG["task_instruction"],
    unnorm_key: str = "libero_object_no_noops",
    save_dir: str = "/content/eval_results_pushcube_depth",
    cfg_scale: float = 1.5,
    use_ddim: bool = True,
    num_ddim_steps: int = 10,
    save_videos: bool = True,
    save_depth_vis: bool = True,
    verbose: bool = True,
    # Sensor resolution (ManiSkill default is 128x128)
    # Use 128x128 for fair comparison with baseline MemoryVLA
    # Use 256x256 for higher quality depth features (separate experiment)
    sensor_width: int = 128,
    sensor_height: int = 128,
    # Depth pipeline parameters
    alpha: float = 0.3,
    use_depth_injection: bool = True,
    use_action_correction: bool = True,
) -> Dict[str, Any]:
    """
    Run full PushCube evaluation with depth-aware pipeline.

    Args:
        vla: MemoryVLA model
        depth_model: Depth Anything V2 model (optional).
                     If None, uses ManiSkill GT depth for Step A.
        num_trials: Number of evaluation episodes
        max_steps: Maximum steps per episode
        task_instruction: Natural language task
        unnorm_key: Action unnormalization key
        save_dir: Directory to save results
        cfg_scale: Classifier-free guidance scale
        use_ddim: Whether to use DDIM sampling
        num_ddim_steps: Number of DDIM steps
        save_videos: Whether to save episode videos
        save_depth_vis: Whether to save depth visualization videos
        verbose: Print detailed progress
        alpha: Depth injection mixing weight
        use_depth_injection: Enable Step A
        use_action_correction: Enable Step B

    Returns:
        Dictionary with aggregated results
    """
    os.makedirs(save_dir, exist_ok=True)

    # Create depth-aware pipeline
    pipeline = DepthAwarePipeline(
        vla=vla,
        depth_model=depth_model,
        alpha=alpha,
        use_depth_injection=use_depth_injection,
        use_action_correction=use_action_correction,
    )

    depth_source = "Depth Anything V2" if depth_model is not None else "ManiSkill GT"

    if verbose:
        print("Creating PushCube environment...")

    env = gym.make(
        "PushCube-v1",
        obs_mode="rgbd",
        control_mode="pd_ee_delta_pose",
        robot_uids="panda",
        num_envs=1,
        render_mode="rgb_array",
        max_episode_steps=max_steps,
        sensor_configs=dict(width=sensor_width, height=sensor_height),
    )

    # Get camera info
    obs, _ = env.reset()
    rgb, camera_name = get_rgb_from_obs(obs)

    # Run trials
    results = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"Depth-Aware Evaluation: PushCube-v1")
        print(f"{'='*60}")
        print(f"Task: {task_instruction}")
        print(f"Trials: {num_trials}")
        print(f"Max steps: {max_steps}")
        print(f"Unnorm key: {unnorm_key}")
        print(f"Camera: {camera_name}")
        print(f"Sensor resolution: {sensor_width}x{sensor_height}")
        print(f"{'='*60}")
        print(f"Depth Pipeline Config:")
        print(f"  Step A (Depth Injection): {'ON' if use_depth_injection else 'OFF'}")
        print(f"    - Alpha: {alpha}")
        print(f"    - Depth source: {depth_source}")
        print(f"    - Colormap: INFERNO")
        print(f"  Step B (Action Correction): {'ON' if use_action_correction else 'OFF'}")
        print(f"    - Depth source: ManiSkill GT")
        print(f"{'='*60}\n")

    for trial in range(num_trials):
        result = run_episode(
            env=env,
            pipeline=pipeline,
            task_instruction=task_instruction,
            max_steps=max_steps,
            unnorm_key=unnorm_key,
            cfg_scale=cfg_scale,
            use_ddim=use_ddim,
            num_ddim_steps=num_ddim_steps,
            save_depth_vis=save_depth_vis,
        )

        result["trial"] = trial
        results.append(result)

        # Save RGB video
        if save_videos:
            status_str = "success" if result["success"] else "failure"
            end_tag = (
                "term" if result["terminated"]
                else ("trunc" if result["truncated"] else "max")
            )
            video_path = os.path.join(
                save_dir,
                f"trial_{trial:02d}_{status_str}_{end_tag}.mp4"
            )
            imageio.mimsave(video_path, result["frames"], fps=10)

            # Save depth visualization video
            if save_depth_vis and "depth_frames" in result and result["depth_frames"]:
                depth_video_path = os.path.join(
                    save_dir,
                    f"trial_{trial:02d}_{status_str}_depth.mp4"
                )
                imageio.mimsave(depth_video_path, result["depth_frames"], fps=10)

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
        "pipeline": "Depth-Aware",
        "depth_source_stepA": depth_source,
        "alpha": alpha,
        "use_depth_injection": use_depth_injection,
        "use_action_correction": use_action_correction,
        "sensor_resolution": f"{sensor_width}x{sensor_height}",
        "camera_name": camera_name,
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

    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluation Summary - PushCube (Depth-Aware Pipeline)")
        print(f"{'='*60}")
        print(f"Success Rate: {total_success}/{num_trials} ({summary['success_rate']:.1f}%)")
        print(f"Avg Reward:   {avg_reward:.3f}")
        print(f"Avg Steps:    {avg_steps:.1f}")
        print(f"{'='*60}")
        print(f"\nEpisode Endings:")
        print(f"  - SUCCESS:    {total_success:3d} episodes")
        print(f"  - TERMINATED: {total_terminated:3d} episodes")
        print(f"  - TRUNCATED:  {total_truncated:3d} episodes")
        print(f"{'='*60}")
        print(f"\nDepth Pipeline: alpha={alpha}, "
              f"injection={'ON' if use_depth_injection else 'OFF'}, "
              f"correction={'ON' if use_action_correction else 'OFF'}")
        if save_videos:
            print(f"Videos saved to: {save_dir}")

    # Save results to file
    result_file = os.path.join(save_dir, "evaluation_results.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(result_file, "w") as f:
        f.write(f"PushCube Depth-Aware Evaluation Results - {timestamp}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Depth Pipeline Configuration:\n")
        f.write(f"  Step A (Depth Injection): {'ON' if use_depth_injection else 'OFF'}\n")
        f.write(f"    - Alpha: {alpha}\n")
        f.write(f"    - Depth source: {depth_source}\n")
        f.write(f"    - Colormap: INFERNO\n")
        f.write(f"  Step B (Action Correction): {'ON' if use_action_correction else 'OFF'}\n")
        f.write(f"    - Depth source: ManiSkill GT\n\n")
        f.write(f"Environment Configuration:\n")
        f.write(f"  Environment: PushCube-v1\n")
        f.write(f"  Sensor resolution: {sensor_width}x{sensor_height}\n")
        f.write(f"  Camera: {camera_name}\n")
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
    print("  # Basic (uses GT depth for Step A):")
    print("  from evaluation.maniskill2.eval_pushcube_depth_aware import run_evaluation")
    print("  results = run_evaluation(vla, num_trials=8)")
    print()
    print("  # With Depth Anything V2:")
    print("  from evaluation.maniskill2.depth_aware_pipeline import load_depth_anything_v2")
    print("  depth_model = load_depth_anything_v2('Small')")
    print("  results = run_evaluation(vla, depth_model=depth_model, num_trials=8)")
    print()
    print("  # Ablation: Step A only (no action correction):")
    print("  results = run_evaluation(vla, alpha=0.3, use_action_correction=False)")
    print()
    print("  # Ablation: Step B only (no depth injection):")
    print("  results = run_evaluation(vla, use_depth_injection=False, use_action_correction=True)")
    print()
    print("  # Fair comparison experiments (same resolution for baseline & depth):")
    print("  # 1. Baseline @128:  results = run_evaluation(vla, use_depth_injection=False, use_action_correction=False)")
    print("  # 2. Depth   @128:  results = run_evaluation(vla, alpha=0.3)")
    print("  # 3. Baseline @256:  results = run_evaluation(vla, use_depth_injection=False, use_action_correction=False, sensor_width=256, sensor_height=256)")
    print("  # 4. Depth   @256:  results = run_evaluation(vla, alpha=0.3, sensor_width=256, sensor_height=256)")
