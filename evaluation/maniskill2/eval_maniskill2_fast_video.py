#!/usr/bin/env python3
"""
ManiSkill Zero-Shot Evaluation with Video Recording
- Faster inference with reduced DDIM steps
- Action scaling for cross-domain transfer
- Video recording for failure analysis
"""

import os
import sys
import numpy as np
import torch
from PIL import Image
import imageio
from pathlib import Path
from datetime import datetime

# Environment setup
os.environ["DISPLAY"] = ""
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

# Mock modules
class FakeModule:
    def __getattr__(self, name):
        return FakeModule()
    def __call__(self, *args, **kwargs):
        return FakeModule()

for mod in ['dlimp', 'dlimp.dataset', 'tensorflow_graphics',
            'tensorflow_graphics.geometry', 'tensorflow_graphics.geometry.transformation']:
    sys.modules[mod] = FakeModule()


def load_model(ckpt_path: str):
    """Load MemoryVLA model"""
    from vla.load import load_vla
    model = load_vla(ckpt_path)
    model.eval()
    return model


def create_env(task_name: str, render_mode: str = "rgb_array"):
    """Create ManiSkill environment"""
    import mani_skill.envs
    import gymnasium as gym

    env = gym.make(
        task_name,
        obs_mode="rgbd",
        control_mode="pd_ee_delta_pos",
        render_mode=render_mode,
        sensor_configs=dict(width=256, height=256),
    )
    return env


def get_image_from_obs(obs) -> Image.Image:
    """Extract RGB image from observation"""
    if "sensor_data" in obs:
        # ManiSkill v3 format
        cam_data = obs["sensor_data"].get("base_camera",
                   obs["sensor_data"].get("hand_camera",
                   list(obs["sensor_data"].values())[0]))
        rgb = cam_data["rgb"]
    elif "image" in obs:
        rgb = obs["image"].get("base_camera", list(obs["image"].values())[0])
    else:
        raise ValueError(f"Unknown obs format: {obs.keys()}")

    if isinstance(rgb, torch.Tensor):
        rgb = rgb.cpu().numpy()

    if rgb.ndim == 4:
        rgb = rgb[0]

    if rgb.dtype != np.uint8:
        if rgb.max() <= 1.0:
            rgb = (rgb * 255).astype(np.uint8)
        else:
            rgb = rgb.astype(np.uint8)

    return Image.fromarray(rgb)


def predict_action_fast(model, image: Image.Image, instruction: str,
                        is_first_frame: bool, num_ddim_steps: int = 5):
    """
    Fast action prediction with reduced DDIM steps

    Args:
        model: MemoryVLA model
        image: RGB image
        instruction: Task instruction
        is_first_frame: Whether this is the first frame
        num_ddim_steps: Number of DDIM steps (default 5, original 10)

    Returns:
        action: 7-dim action array
    """
    action_output = model.predict_action(
        image=image,
        instruction=instruction,
        episode_first_frame='True' if is_first_frame else 'False',
        use_ddim=True,
        num_ddim_steps=num_ddim_steps,  # Reduced for speed
    )

    # Handle tuple return
    if isinstance(action_output, tuple):
        action = action_output[0]
    else:
        action = action_output

    # Convert to numpy
    if isinstance(action, torch.Tensor):
        action = action.cpu().numpy()

    if action.ndim > 1:
        action = action.squeeze()

    return action


def convert_action_to_maniskill(action_7d: np.ndarray,
                                 scale_factor: float = 10.0) -> np.ndarray:
    """
    Convert 7-dim Bridge action to 4-dim ManiSkill action

    Bridge action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
    ManiSkill pd_ee_delta_pos: [dx, dy, dz, gripper]

    Args:
        action_7d: 7-dimensional action from MemoryVLA
        scale_factor: Scale up the action (Bridge actions are small)

    Returns:
        action_4d: 4-dimensional action for ManiSkill
    """
    ms_action = np.zeros(4)

    # Position deltas with scaling
    ms_action[0] = action_7d[0] * scale_factor  # dx
    ms_action[1] = action_7d[1] * scale_factor  # dy
    ms_action[2] = action_7d[2] * scale_factor  # dz

    # Gripper: Bridge uses continuous, ManiSkill uses -1 to 1
    # Threshold at 0 to convert to binary open/close
    gripper = action_7d[6] if len(action_7d) > 6 else action_7d[-1]
    ms_action[3] = 1.0 if gripper > 0 else -1.0

    # Clip to valid range
    ms_action[:3] = np.clip(ms_action[:3], -1.0, 1.0)

    return ms_action


def evaluate_task(model, task_name: str, instruction: str,
                  num_episodes: int = 3, max_steps: int = 50,
                  num_ddim_steps: int = 5, action_scale: float = 10.0,
                  save_video: bool = True, video_dir: str = "./eval_videos"):
    """
    Evaluate a single task with video recording

    Args:
        model: MemoryVLA model
        task_name: ManiSkill task name
        instruction: Task instruction for the model
        num_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode
        num_ddim_steps: DDIM steps (lower = faster)
        action_scale: Scale factor for actions
        save_video: Whether to save video
        video_dir: Directory to save videos

    Returns:
        results: Dictionary with success rate and video paths
    """
    print(f"\n{'='*60}")
    print(f"평가: {task_name}")
    print(f"  Instruction: {instruction}")
    print(f"  DDIM steps: {num_ddim_steps}, Action scale: {action_scale}")
    print(f"{'='*60}")

    # Create video directory
    if save_video:
        video_path = Path(video_dir) / task_name
        video_path.mkdir(parents=True, exist_ok=True)

    try:
        env = create_env(task_name)
        print("   ✅ 환경 생성 성공")
    except Exception as e:
        print(f"   ❌ 환경 생성 실패: {e}")
        return {"success_rate": 0.0, "error": str(e)}

    successes = []
    video_files = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        frames = []  # For video
        is_first_frame = True
        success = False

        for step in range(max_steps):
            # Get image
            try:
                pil_image = get_image_from_obs(obs)
            except Exception as e:
                print(f"   Image extraction error: {e}")
                break

            # Save frame for video
            if save_video:
                # Get render frame (higher quality)
                render_frame = env.render()
                if render_frame is not None:
                    if isinstance(render_frame, torch.Tensor):
                        render_frame = render_frame.cpu().numpy()
                    if render_frame.ndim == 4:
                        render_frame = render_frame[0]
                    if render_frame.dtype != np.uint8:
                        if render_frame.max() <= 1.0:
                            render_frame = (render_frame * 255).astype(np.uint8)
                        else:
                            render_frame = render_frame.astype(np.uint8)
                    frames.append(render_frame)

            # Predict action
            try:
                action_7d = predict_action_fast(
                    model, pil_image, instruction,
                    is_first_frame, num_ddim_steps
                )
                is_first_frame = False

                # Print first action for debugging
                if step == 0:
                    print(f"   Ep {ep+1} first action: {action_7d[:4]}")

            except Exception as e:
                print(f"   Prediction error: {e}")
                action_7d = np.zeros(7)

            # Convert action
            ms_action = convert_action_to_maniskill(action_7d, action_scale)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(ms_action)

            if terminated:
                success = info.get("success", False)
                break
            if truncated:
                break

        # Record result
        successes.append(float(success))
        status = "✅" if success else "❌"
        print(f"   Episode {ep+1}/{num_episodes}: {status} (steps: {step+1})")

        # Save video
        if save_video and len(frames) > 0:
            timestamp = datetime.now().strftime("%H%M%S")
            result_str = "success" if success else "failure"
            video_file = video_path / f"ep{ep+1}_{result_str}_{timestamp}.mp4"

            try:
                imageio.mimsave(str(video_file), frames, fps=10)
                video_files.append(str(video_file))
                print(f"   📹 Video saved: {video_file.name}")
            except Exception as e:
                print(f"   ⚠️ Video save error: {e}")

    env.close()

    success_rate = np.mean(successes)
    print(f"   Success Rate: {success_rate*100:.1f}%")

    return {
        "success_rate": success_rate,
        "successes": successes,
        "videos": video_files
    }


def run_evaluation(ckpt_path: str,
                   num_episodes: int = 3,
                   max_steps: int = 50,
                   num_ddim_steps: int = 5,
                   action_scale: float = 10.0,
                   save_video: bool = True):
    """
    Run full evaluation on ManiSkill tasks
    """
    print("="*60)
    print("ManiSkill Zero-Shot Evaluation")
    print(f"  DDIM steps: {num_ddim_steps} (faster inference)")
    print(f"  Action scale: {action_scale}x")
    print(f"  Episodes per task: {num_episodes}")
    print(f"  Max steps: {max_steps}")
    print(f"  Save videos: {save_video}")
    print("="*60)

    # Load model
    print("\n1. 모델 로딩...")
    model = load_model(ckpt_path)
    print(f"   ✅ 모델 로딩 완료 ({torch.cuda.memory_allocated()/1e9:.2f} GB)")

    # Tasks to evaluate
    tasks = [
        ("PickCube-v1", "pick up the red cube"),
        ("StackCube-v1", "stack the red cube on the green cube"),
        ("PegInsertionSide-v1", "insert the peg into the hole"),
    ]

    results = {}

    for task_name, instruction in tasks:
        result = evaluate_task(
            model=model,
            task_name=task_name,
            instruction=instruction,
            num_episodes=num_episodes,
            max_steps=max_steps,
            num_ddim_steps=num_ddim_steps,
            action_scale=action_scale,
            save_video=save_video,
        )
        results[task_name] = result

    # Summary
    print("\n" + "="*60)
    print("평가 결과 요약")
    print("="*60)

    total_success = []
    for task_name, result in results.items():
        sr = result.get("success_rate", 0) * 100
        print(f"  {task_name}: {sr:.1f}%")
        if "successes" in result:
            total_success.extend(result["successes"])
        if "videos" in result and result["videos"]:
            print(f"    Videos: {len(result['videos'])} saved")

    if total_success:
        overall = np.mean(total_success) * 100
        print(f"\n전체 Success Rate: {overall:.1f}%")

    print("="*60)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="checkpoints/memvla-bridge/checkpoints/memvla-bridge.pt")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--ddim-steps", type=int, default=5)
    parser.add_argument("--action-scale", type=float, default=10.0)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    run_evaluation(
        ckpt_path=args.ckpt,
        num_episodes=args.episodes,
        max_steps=args.steps,
        num_ddim_steps=args.ddim_steps,
        action_scale=args.action_scale,
        save_video=not args.no_video,
    )
