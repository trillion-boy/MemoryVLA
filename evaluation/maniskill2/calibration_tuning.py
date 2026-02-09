"""
Action Space Calibration Tuning for LIBERO → ManiSkill transfer.

Lightweight script to observe raw VLA action outputs and tune
axis mapping / scale factors. Run with few trials and steps.

Usage (Colab):
    from evaluation.maniskill2.calibration_tuning import run_calibration
    run_calibration(vla, num_trials=2, max_steps=30)

    # After finding good scale factors:
    run_calibration(vla, scale=[1.0, 1.0, 1.0], num_trials=2, max_steps=30)
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
from typing import Optional, List


DEFAULT_INSTRUCTION = (
    "First, identify the blue cube and the red and white circular target. "
    "Position the end-effector directly behind the center of the blue cube "
    "to align with the target. Then, smoothly slide the cube forward in a "
    "straight line until it is fully inside the target zone. Ensure the cube "
    "remains flat on the table throughout the motion."
)


def get_rgb_from_obs(obs):
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
    raise ValueError("No RGB found")


def run_calibration(
    vla,
    num_trials: int = 2,
    max_steps: int = 30,
    scale: Optional[List[float]] = None,
    axis_swap: Optional[List[int]] = None,
    task_instruction: str = DEFAULT_INSTRUCTION,
    unnorm_key: str = "libero_object_no_noops",
    cfg_scale: float = 1.5,
    save_dir: str = "/content/calibration_results",
):
    """
    Run calibration trials, printing raw & calibrated actions per step.

    Args:
        vla: MemoryVLA model
        num_trials: Number of trials (2~3 is enough)
        max_steps: Steps per trial (30~50)
        scale: [scale_x, scale_y, scale_z] multipliers. Default [1,1,1]
        axis_swap: [i,j,k] axis remapping. e.g. [1,0,2] swaps X↔Y. Default [0,1,2]
        task_instruction: Task description
        unnorm_key: Action unnorm key
        cfg_scale: CFG scale
        save_dir: Directory for videos
    """
    if scale is None:
        scale = [1.0, 1.0, 1.0]
    if axis_swap is None:
        axis_swap = [0, 1, 2]

    os.makedirs(save_dir, exist_ok=True)

    env = gym.make(
        "PushCube-v1",
        obs_mode="rgbd",
        control_mode="pd_ee_delta_pose",
        robot_uids="panda",
        num_envs=1,
        render_mode="rgb_array",
        max_episode_steps=max_steps,
    )

    print(f"{'='*70}")
    print(f"Action Space Calibration")
    print(f"{'='*70}")
    print(f"Scale:     X={scale[0]:.2f}, Y={scale[1]:.2f}, Z={scale[2]:.2f}")
    print(f"Axis swap: {axis_swap}  (0=X, 1=Y, 2=Z)")
    print(f"Trials: {num_trials}, Max steps: {max_steps}")
    print(f"{'='*70}\n")

    for trial in range(num_trials):
        obs, _ = env.reset()
        frames = []
        total_reward = 0.0

        print(f"\n--- Trial {trial+1}/{num_trials} ---")
        print(f"{'Step':>4} | {'raw_dx':>8} {'raw_dy':>8} {'raw_dz':>8} | "
              f"{'cal_dx':>8} {'cal_dy':>8} {'cal_dz':>8} | "
              f"{'grip':>6} | {'reward':>7}")
        print("-" * 90)

        for step in range(max_steps):
            rgb = get_rgb_from_obs(obs)
            pil_image = Image.fromarray(rgb)
            frames.append(rgb.copy())

            episode_first = 'True' if step == 0 else 'False'

            actions, _ = vla.predict_action(
                image=pil_image,
                instruction=task_instruction,
                unnorm_key=unnorm_key,
                cfg_scale=cfg_scale,
                use_ddim=True,
                num_ddim_steps=10,
                episode_first_frame=episode_first,
            )

            action = actions[0].copy()
            raw_xyz = action[:3].copy()

            # Apply axis swap
            swapped = np.array([raw_xyz[axis_swap[0]],
                                raw_xyz[axis_swap[1]],
                                raw_xyz[axis_swap[2]]])

            # Apply scale
            calibrated = swapped * np.array(scale)
            action[:3] = calibrated

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

            print(f"{step+1:4d} | "
                  f"{raw_xyz[0]:8.4f} {raw_xyz[1]:8.4f} {raw_xyz[2]:8.4f} | "
                  f"{calibrated[0]:8.4f} {calibrated[1]:8.4f} {calibrated[2]:8.4f} | "
                  f"{action[6]:6.2f} | "
                  f"{reward:7.3f}")

            if terminated or truncated:
                rgb = get_rgb_from_obs(obs)
                frames.append(rgb.copy())
                break

        success = info.get('success', False)
        if isinstance(success, torch.Tensor):
            success = success.item()

        print(f"\nTrial {trial+1} result: {'SUCCESS' if success else 'FAILURE'} | "
              f"Total reward: {total_reward:.3f} | Steps: {step+1}")

        # Save video
        video_path = os.path.join(save_dir, f"cal_trial_{trial:02d}.mp4")
        imageio.mimsave(video_path, frames, fps=10)
        print(f"Video: {video_path}")

    env.close()
    print(f"\n{'='*70}")
    print("Tips:")
    print("  - If robot moves RIGHT when it should go FORWARD → try axis_swap=[1,0,2]")
    print("  - If robot moves too little → increase scale (e.g. scale=[2.0, 2.0, 1.0])")
    print("  - If robot moves opposite direction → use negative scale (e.g. scale=[-1.0, 1.0, 1.0])")
    print(f"{'='*70}")
