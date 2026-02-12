"""
PickCube evaluation with ControlMLLM-style visual prompt optimization.

Tests whether attention-guided visual prompt optimization can improve
cross-environment generalization (LIBERO → ManiSkill).

Usage (Colab):
    from evaluation.maniskill2.depth_aware_pipeline import load_depth_anything_v2
    depth_model = load_depth_anything_v2("Small")

    from evaluation.maniskill2.eval_pickcube_controlmllm import run_evaluation
    results = run_evaluation(vla, depth_model=depth_model, num_trials=4)

    # Attention diagnosis (run FIRST to find the right layers):
    from evaluation.maniskill2.controlmllm_vla_pipeline import ControlMLLMVLAPipeline
    pipeline = ControlMLLMVLAPipeline(vla, depth_model=depth_model)
    # Get one observation from env, then:
    pipeline.visualize_attention_by_layer(pil_image, instruction)
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
from typing import Optional, Dict, Any
from transforms3d.euler import euler2axangle

from evaluation.maniskill2.controlmllm_vla_pipeline import ControlMLLMVLAPipeline


TASK_INSTRUCTION = (
    "Pick up the small cube from the table. Move the gripper directly above "
    "the cube, lower it down to grasp the cube, close the gripper firmly, "
    "and lift the cube upward off the table surface."
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
    raise ValueError("No RGB found in obs")


def convert_action_to_maniskill2(raw_action: np.ndarray, action_scale: float = 1.0) -> np.ndarray:
    """
    Convert MemoryVLA raw action (LIBERO format) to ManiSkill2 format.

    MemoryVLA outputs: [x, y, z, roll, pitch, yaw, gripper]
      - rotation in euler angles
      - gripper in [0, 1] (LIBERO convention)

    ManiSkill2 pd_ee_delta_pose expects: [dx, dy, dz, ax, ay, az, gripper]
      - rotation in axis-angle
      - gripper in [-1, +1] (-1=close, +1=open)
    """
    delta_pos = raw_action[:3] * action_scale

    # Euler (roll, pitch, yaw) → axis-angle
    roll, pitch, yaw = raw_action[3], raw_action[4], raw_action[5]
    axis, angle = euler2axangle(roll, pitch, yaw)
    delta_rot_axangle = axis * angle * action_scale

    # Gripper: LIBERO [0,1] → ManiSkill2 [-1,+1], binarized
    gripper = raw_action[6]
    gripper_normalized = 2.0 * (gripper > 0.5) - 1.0

    return np.concatenate([delta_pos, delta_rot_axangle, [gripper_normalized]]).astype(np.float32)


def run_evaluation(
    vla,
    depth_model=None,
    num_trials: int = 4,
    max_steps: int = 100,
    task_instruction: str = TASK_INSTRUCTION,
    unnorm_key: str = "libero_object_no_noops",
    save_dir: str = "/content/eval_results_pickcube_controlmllm",
    cfg_scale: float = 1.5,
    save_videos: bool = True,
    verbose: bool = True,
    debug_masks: bool = True,
    # ControlMLLM parameters
    T: int = 10,
    lr: float = 5.0,
    alpha_loss: float = 400.0,
    layer_start: int = 14,
    layer_end: int = 26,
    optimize_freq: int = 5,
    optimizer: str = "sgd",
    init_scale: float = 0.05,
    sensor_resolution: int = 224,
    baseline_mode: bool = False,
) -> Dict[str, Any]:
    """
    Run PickCube evaluation with ControlMLLM visual prompt optimization.

    Args:
        vla: MemoryVLA model
        depth_model: Depth Anything V2 model
        num_trials: Number of evaluation episodes
        max_steps: Max steps per episode
        task_instruction: Natural language task
        unnorm_key: Action unnormalization key
        save_dir: Directory for results
        cfg_scale: CFG scale
        save_videos: Save episode videos
        verbose: Print progress
        debug_masks: Save mask debug images for first trial
        T: Optimization iterations per step
        lr: Learning rate for visual prompt
        alpha_loss: Loss scaling factor
        layer_start: First LLM layer for attention
        layer_end: Last LLM layer for attention
        optimize_freq: Optimize every N steps
        optimizer: "adam" or "sgd"
        sensor_resolution: Camera resolution (128 or 256)

    Returns:
        Dictionary with results
    """
    os.makedirs(save_dir, exist_ok=True)

    # Create pipeline
    pipeline = ControlMLLMVLAPipeline(
        vla=vla,
        depth_model=depth_model,
        T=T,
        lr=lr,
        alpha_loss=alpha_loss,
        layer_start=layer_start,
        layer_end=layer_end,
        optimize_freq=optimize_freq,
        optimizer=optimizer,
        init_scale=init_scale,
    )

    depth_source = "Depth Anything V2" if depth_model else "None"

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

    mode_name = "BASELINE (no ControlMLLM)" if baseline_mode else "ControlMLLM-VLA"

    if verbose:
        print(f"\n{'='*60}")
        print(f"{mode_name} Evaluation: PickCube-v1")
        print(f"{'='*60}")
        print(f"Task: {task_instruction}")
        print(f"Trials: {num_trials}, Max steps: {max_steps}")
        print(f"Depth source: {depth_source}")
        print(f"Resolution: {sensor_resolution}x{sensor_resolution}")
        if baseline_mode:
            print(f"MODE: BASELINE — no visual prompt optimization")
        else:
            print(f"{'='*60}")
            print(f"ControlMLLM Config:")
            print(f"  T (iterations):    {T}")
            print(f"  lr:                {lr}")
            print(f"  alpha_loss:        {alpha_loss}")
            print(f"  Layers:            {layer_start}-{layer_end}")
            print(f"  Optimizer:         {optimizer}")
            print(f"  Optimize freq:     every {optimize_freq} step(s)")
        print(f"{'='*60}\n")

    results = []

    for trial in range(num_trials):
        obs, _ = env.reset()
        pipeline.reset()

        frames = []
        total_reward = 0.0
        terminated = False
        truncated = False

        # Debug mask dir for first trial only
        trial_debug_dir = None
        if debug_masks and trial == 0:
            trial_debug_dir = os.path.join(save_dir, "debug_masks_trial0")

        for step in range(max_steps):
            rgb = get_rgb_from_obs(obs)
            pil_image = Image.fromarray(rgb)
            frames.append(rgb.copy())

            episode_first = "True" if step == 0 else "False"

            # Save full debug visualization for first frame of each trial
            if step == 0:
                frame_debug_path = os.path.join(
                    save_dir, f"debug_first_frame_trial{trial}.png"
                )
                pipeline.debug_first_frame(pil_image, save_path=frame_debug_path)

            if baseline_mode:
                # Baseline: run MemoryVLA directly, no Pv optimization
                actions, _ = vla.predict_action(
                    image=pil_image,
                    instruction=task_instruction,
                    unnorm_key=unnorm_key,
                    cfg_scale=cfg_scale,
                    use_ddim=True,
                    num_ddim_steps=10,
                    episode_first_frame=episode_first,
                )
            else:
                # ControlMLLM: optimize visual prompt then predict
                step_debug_dir = None
                if trial_debug_dir and step < 5:
                    step_debug_dir = trial_debug_dir

                actions, _ = pipeline.predict_action(
                    image=pil_image,
                    instruction=task_instruction,
                    obs=obs,
                    unnorm_key=unnorm_key,
                    cfg_scale=cfg_scale,
                    use_ddim=True,
                    num_ddim_steps=10,
                    episode_first_frame=episode_first,
                    debug_save_dir=step_debug_dir,
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
            "terminated": terminated,
            "truncated": truncated,
        }
        results.append(result)

        # Save video
        if save_videos and frames:
            status = "success" if success else "failure"
            video_path = os.path.join(save_dir, f"trial_{trial:02d}_{status}.mp4")
            imageio.mimsave(video_path, frames, fps=10)

        if verbose:
            sc = sum(r["success"] for r in results)
            status = "SUCCESS" if success else "failure"
            print(
                f"Trial {trial+1:2d}/{num_trials}: {status:10s} | "
                f"Reward: {total_reward:7.3f} | Steps: {step+1:3d} | "
                f"Success: {sc}/{trial+1}"
            )

    env.close()

    # Summary
    total_success = sum(r["success"] for r in results)
    avg_reward = np.mean([r["total_reward"] for r in results])
    avg_steps = np.mean([r["steps"] for r in results])

    summary = {
        "env_name": "PickCube-v1",
        "pipeline": "BASELINE" if baseline_mode else "ControlMLLM-VLA",
        "depth_source": depth_source,
        "optimizer": optimizer,
        "T": T,
        "lr": lr,
        "alpha_loss": alpha_loss,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "optimize_freq": optimize_freq,
        "sensor_resolution": sensor_resolution,
        "num_trials": num_trials,
        "success_count": total_success,
        "success_rate": 100 * total_success / num_trials,
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
        "results": results,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Summary - PickCube ({mode_name})")
        print(f"{'='*60}")
        print(f"Success Rate: {total_success}/{num_trials} ({summary['success_rate']:.1f}%)")
        print(f"Avg Reward:   {avg_reward:.3f}")
        print(f"Avg Steps:    {avg_steps:.1f}")
        print(f"Config: T={T}, lr={lr}, optim={optimizer}, layers={layer_start}-{layer_end}")
        print(f"{'='*60}")

    # Save results
    result_file = os.path.join(save_dir, "results.txt")
    with open(result_file, "w") as f:
        f.write(f"PickCube {mode_name} Results - {datetime.now()}\n")
        f.write(f"{'='*60}\n")
        f.write(f"Config: T={T}, lr={lr}, optim={optimizer}, alpha={alpha_loss}, "
                f"layers={layer_start}-{layer_end}, freq={optimize_freq}\n")
        f.write(f"Depth: {depth_source}\n")
        f.write(f"Resolution: {sensor_resolution}x{sensor_resolution}\n\n")
        f.write(f"Success: {total_success}/{num_trials} ({summary['success_rate']:.1f}%)\n")
        f.write(f"Avg Reward: {avg_reward:.3f}\n\n")
        for r in results:
            f.write(f"Trial {r['trial']+1}: {'SUCCESS' if r['success'] else 'FAIL':7s} | "
                    f"Reward: {r['total_reward']:7.3f} | Steps: {r['steps']:3d}\n")

    return summary
