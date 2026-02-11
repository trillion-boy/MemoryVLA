"""
Pure MemoryVLA baseline evaluation on ManiSkill2 PickCube.

No ControlMLLM, no depth model, no visual prompt optimization.
Just MemoryVLA + proper action conversion (LIBERO → ManiSkill2 format).

Usage (Colab):
    from evaluation.maniskill2.eval_pickcube_baseline import run_baseline
    results = run_baseline(vla, save_dir="/content/eval_pure_baseline")
"""

import gymnasium as gym
try:
    import mani_skill.envs
except ModuleNotFoundError:
    import mani_skill2.envs
import numpy as np
from PIL import Image
import torch
import imageio
import os
from datetime import datetime
from typing import Dict, Any
from transforms3d.euler import euler2axangle


TASK_INSTRUCTION = (
    "Pick up the small cube from the table. Move the gripper directly above "
    "the cube, lower it down to grasp the cube, close the gripper firmly, "
    "and lift the cube upward off the table surface."
)


def get_rgb_from_obs(obs):
    """Extract RGB from ManiSkill2 observation."""
    if "sensor_data" in obs:
        for cam in ["base_camera", "hand_camera"]:
            if cam in obs["sensor_data"] and "rgb" in obs["sensor_data"][cam]:
                rgb = obs["sensor_data"][cam]["rgb"]
                if isinstance(rgb, torch.Tensor):
                    rgb = rgb.cpu().numpy()
                if rgb.ndim == 4:
                    rgb = rgb[0]
                if rgb.max() <= 1.0:
                    rgb = (rgb * 255).astype(np.uint8)
                return rgb
    raise ValueError("No RGB found in obs")


def convert_action_to_maniskill2(raw_action: np.ndarray, action_scale: float = 1.0) -> np.ndarray:
    """
    Convert MemoryVLA raw action (LIBERO format) to ManiSkill2 format.

    LIBERO:     [x, y, z, roll, pitch, yaw, gripper(0~1)]
    ManiSkill2: [dx, dy, dz, ax, ay, az, gripper(-1~+1)]
    """
    delta_pos = raw_action[:3] * action_scale

    roll, pitch, yaw = raw_action[3], raw_action[4], raw_action[5]
    axis, angle = euler2axangle(roll, pitch, yaw)
    delta_rot_axangle = axis * angle * action_scale

    gripper = raw_action[6]
    gripper_normalized = 2.0 * (gripper > 0.5) - 1.0

    return np.concatenate([delta_pos, delta_rot_axangle, [gripper_normalized]]).astype(np.float32)


def run_baseline(
    vla,
    num_trials: int = 4,
    max_steps: int = 100,
    task_instruction: str = TASK_INSTRUCTION,
    unnorm_key: str = "libero_object_no_noops",
    save_dir: str = "/content/eval_pure_baseline",
    cfg_scale: float = 1.5,
    save_videos: bool = True,
    sensor_resolution: int = 128,
) -> Dict[str, Any]:
    """
    Pure MemoryVLA baseline. No depth, no ControlMLLM, no pipeline.
    Only action format conversion (euler→axis-angle, gripper normalization).
    """
    os.makedirs(save_dir, exist_ok=True)

    env = gym.make(
        "PickCube-v1",
        obs_mode="rgbd",
        control_mode="pd_ee_delta_pose",
        robot_uids="panda",
        num_envs=1,
        render_mode="rgb_array",
        max_episode_steps=max_steps,
        sensor_configs=dict(
            base_camera=dict(width=sensor_resolution, height=sensor_resolution),
        ),
    )

    print(f"\n{'='*60}")
    print(f"PURE BASELINE: MemoryVLA only (no depth, no ControlMLLM)")
    print(f"{'='*60}")
    print(f"Task: {task_instruction}")
    print(f"Trials: {num_trials}, Max steps: {max_steps}")
    print(f"unnorm_key: {unnorm_key}")
    print(f"Action conversion: euler→axis-angle + gripper [-1,+1]")
    print(f"{'='*60}\n")

    results = []

    for trial in range(num_trials):
        obs, _ = env.reset()
        frames = []
        total_reward = 0.0

        for step in range(max_steps):
            rgb = get_rgb_from_obs(obs)
            pil_image = Image.fromarray(rgb)
            frames.append(rgb.copy())

            episode_first = "True" if step == 0 else "False"

            actions, _ = vla.predict_action(
                image=pil_image,
                instruction=task_instruction,
                unnorm_key=unnorm_key,
                cfg_scale=cfg_scale,
                use_ddim=True,
                num_ddim_steps=10,
                episode_first_frame=episode_first,
            )

            raw_action = actions[0]
            action = convert_action_to_maniskill2(raw_action)
            obs, reward, terminated, truncated, info = env.step(
                torch.tensor(action).unsqueeze(0)
            )

            if isinstance(reward, torch.Tensor):
                reward = reward.item()
            if isinstance(terminated, torch.Tensor):
                terminated = terminated.item()
            if isinstance(truncated, torch.Tensor):
                truncated = truncated.item()

            total_reward += reward

            if terminated or truncated:
                rgb = get_rgb_from_obs(obs)
                frames.append(rgb.copy())
                break

        success = info.get("success", False)
        if isinstance(success, torch.Tensor):
            success = success.item()

        result = {
            "trial": trial,
            "success": success,
            "total_reward": total_reward,
            "steps": step + 1,
        }
        results.append(result)

        if save_videos and frames:
            status = "success" if success else "failure"
            video_path = os.path.join(save_dir, f"trial_{trial:02d}_{status}.mp4")
            imageio.mimsave(video_path, frames, fps=10)

        sc = sum(r["success"] for r in results)
        status = "SUCCESS" if success else "failure"
        print(
            f"Trial {trial+1:2d}/{num_trials}: {status:10s} | "
            f"Reward: {total_reward:7.3f} | Steps: {step+1:3d} | "
            f"Success: {sc}/{trial+1}"
        )

    env.close()

    total_success = sum(r["success"] for r in results)
    avg_reward = np.mean([r["total_reward"] for r in results])

    print(f"\n{'='*60}")
    print(f"PURE BASELINE Summary")
    print(f"{'='*60}")
    print(f"Success Rate: {total_success}/{num_trials} ({100*total_success/num_trials:.1f}%)")
    print(f"Avg Reward:   {avg_reward:.3f}")
    print(f"{'='*60}")

    return {
        "pipeline": "PURE_BASELINE",
        "success_rate": 100 * total_success / num_trials,
        "avg_reward": avg_reward,
        "results": results,
    }
