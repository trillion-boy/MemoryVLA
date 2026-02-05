"""
ManiSkill2 native evaluator for cross-environment generalization testing.

Tests LIBERO-trained models on ManiSkill2 Franka Panda environments.
No SimplerEnv dependency - uses ManiSkill2 API directly.

Usage:
    python evaluation/maniskill2/maniskill2_evaluator.py \
        --ckpt-path /path/to/libero_checkpoint.pt \
        --env-name PickCube-v0 \
        --task-instruction "pick up the cube" \
        --num-episodes 50
"""

import os
import argparse
import numpy as np
from typing import Optional
from pathlib import Path
from datetime import datetime

# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def create_maniskill2_env(
    env_name: str,
    obs_mode: str = "rgbd",
    control_mode: str = "pd_ee_delta_pose",
    render_mode: Optional[str] = None,
    max_episode_steps: int = 200,
):
    """
    Create a ManiSkill2 environment.

    Args:
        env_name: Environment name (e.g., "PickCube-v0", "StackCube-v0")
        obs_mode: Observation mode ("rgbd", "pointcloud", "state")
        control_mode: Control mode ("pd_ee_delta_pose" for EEF control)
        render_mode: Render mode ("human", "rgb_array", None)
        max_episode_steps: Maximum steps per episode

    Returns:
        ManiSkill2 gym environment
    """
    try:
        import gymnasium as gym
        from gymnasium.wrappers import TimeLimit
        import mani_skill2.envs  # Register ManiSkill2 environments
        env = gym.make(
            env_name,
            obs_mode=obs_mode,
            control_mode=control_mode,
            render_mode=render_mode,
        )
        # Wrap with TimeLimit to enforce max_episode_steps
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    except ImportError:
        # Fallback to older gym
        import gym
        from gym.wrappers import TimeLimit
        import mani_skill2.envs
        env = gym.make(
            env_name,
            obs_mode=obs_mode,
            control_mode=control_mode,
        )
        # Wrap with TimeLimit to enforce max_episode_steps
        env = TimeLimit(env, max_episode_steps=max_episode_steps)

    return env


def get_rgb_from_obs(obs: dict) -> np.ndarray:
    """
    Extract RGB image from ManiSkill2 observation.

    ManiSkill2 RGBD observation structure:
    - obs['image']['base_camera']['rgb']: (H, W, 3) uint8
    - obs['image']['hand_camera']['rgb']: (H, W, 3) uint8 (wrist camera)

    Args:
        obs: ManiSkill2 observation dictionary

    Returns:
        RGB image as numpy array (H, W, 3), uint8
    """
    if 'image' in obs:
        # Try base camera first (third-person view, similar to LIBERO)
        if 'base_camera' in obs['image']:
            rgb = obs['image']['base_camera']['rgb']
        elif 'hand_camera' in obs['image']:
            rgb = obs['image']['hand_camera']['rgb']
        else:
            # Get first available camera
            camera_name = list(obs['image'].keys())[0]
            rgb = obs['image'][camera_name]['rgb']

        # Ensure uint8
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).astype(np.uint8)

        return rgb
    else:
        raise ValueError("No image observation found. Make sure obs_mode='rgbd'")


def evaluate_single_episode(
    env,
    policy,
    task_instruction: str,
    max_steps: int = 200,
    verbose: bool = False,
) -> tuple:
    """
    Evaluate a single episode.

    Args:
        env: ManiSkill2 environment
        policy: VLA policy with predict_action method
        task_instruction: Natural language task instruction
        max_steps: Maximum steps per episode
        verbose: Print step-by-step info

    Returns:
        (success, total_reward, num_steps)
    """
    obs, info = env.reset()
    policy.reset(task_instruction)

    total_reward = 0.0
    success = False

    for step in range(max_steps):
        # Get RGB observation
        rgb = get_rgb_from_obs(obs)

        # Predict action
        action = policy.predict_action(
            rgb,
            episode_first_frame=(step == 0)
        )

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if verbose and step % 20 == 0:
            print(f"  Step {step}: reward={reward:.3f}, total={total_reward:.3f}")

        # Check success
        if 'success' in info and info['success']:
            success = True
            if verbose:
                print(f"  SUCCESS at step {step}!")
            break

        if terminated or truncated:
            break

    return success, total_reward, step + 1


def evaluate_maniskill2(
    ckpt_path: str,
    env_name: str,
    task_instruction: str,
    num_episodes: int = 50,
    max_steps: int = 200,
    unnorm_key: str = "libero_spatial_no_noops",
    control_mode: str = "pd_ee_delta_pose",
    obs_mode: str = "rgbd",
    action_scale: float = 1.0,
    cfg_scale: float = 1.5,
    save_dir: Optional[str] = None,
    seed: int = 42,
    verbose: bool = False,
    **kwargs,
):
    """
    Evaluate a MemoryVLA model on ManiSkill2 environment.

    Args:
        ckpt_path: Path to the trained checkpoint
        env_name: ManiSkill2 environment name
        task_instruction: Natural language task instruction
        num_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode
        unnorm_key: Action normalization key (from training dataset)
        control_mode: ManiSkill2 control mode
        obs_mode: ManiSkill2 observation mode
        action_scale: Action scaling factor
        cfg_scale: Classifier-free guidance scale
        save_dir: Directory to save results
        seed: Random seed
        verbose: Print detailed info

    Returns:
        dict with evaluation results
    """
    from evaluation.maniskill2.maniskill2_policy import ManiSkill2VLAPolicy

    # Set random seed
    np.random.seed(seed)

    # Create environment
    print(f"\n{'='*60}")
    print(f"ManiSkill2 Cross-Environment Generalization Evaluation")
    print(f"{'='*60}")
    print(f"Environment: {env_name}")
    print(f"Task: {task_instruction}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Unnorm key: {unnorm_key}")
    print(f"Control mode: {control_mode}")
    print(f"Num episodes: {num_episodes}")
    print(f"{'='*60}\n")

    env = create_maniskill2_env(
        env_name=env_name,
        obs_mode=obs_mode,
        control_mode=control_mode,
        max_episode_steps=max_steps,
    )

    # Create policy
    policy = ManiSkill2VLAPolicy(
        saved_model_path=ckpt_path,
        unnorm_key=unnorm_key,
        action_scale=action_scale,
        cfg_scale=cfg_scale,
        **kwargs,
    )

    # Evaluate episodes
    successes = []
    rewards = []
    steps_list = []

    for ep in range(num_episodes):
        success, reward, num_steps = evaluate_single_episode(
            env, policy, task_instruction, max_steps, verbose
        )

        successes.append(success)
        rewards.append(reward)
        steps_list.append(num_steps)

        status = "SUCCESS" if success else "FAILURE"
        print(f"Episode {ep+1:3d}/{num_episodes}: {status} | "
              f"Reward: {reward:7.2f} | Steps: {num_steps:3d}")

    env.close()

    # Compute statistics
    success_rate = np.mean(successes) * 100
    avg_reward = np.mean(rewards)
    avg_steps = np.mean(steps_list)

    results = {
        'env_name': env_name,
        'task_instruction': task_instruction,
        'ckpt_path': ckpt_path,
        'unnorm_key': unnorm_key,
        'num_episodes': num_episodes,
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'avg_steps': avg_steps,
        'successes': successes,
        'rewards': rewards,
        'steps': steps_list,
    }

    print(f"\n{'='*60}")
    print(f"Results Summary")
    print(f"{'='*60}")
    print(f"Success Rate: {success_rate:.1f}% ({sum(successes)}/{num_episodes})")
    print(f"Avg Reward:   {avg_reward:.2f}")
    print(f"Avg Steps:    {avg_steps:.1f}")
    print(f"{'='*60}\n")

    # Save results
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = save_dir / f"{env_name}_{timestamp}.txt"

        with open(result_file, 'w') as f:
            f.write(f"Environment: {env_name}\n")
            f.write(f"Task: {task_instruction}\n")
            f.write(f"Checkpoint: {ckpt_path}\n")
            f.write(f"Unnorm key: {unnorm_key}\n")
            f.write(f"Success Rate: {success_rate:.1f}%\n")
            f.write(f"Avg Reward: {avg_reward:.2f}\n")
            f.write(f"Avg Steps: {avg_steps:.1f}\n")
            f.write(f"\nPer-episode results:\n")
            for i, (s, r, st) in enumerate(zip(successes, rewards, steps_list)):
                f.write(f"Episode {i+1}: {'success' if s else 'failure'} | "
                       f"reward={r:.2f} | steps={st}\n")

        print(f"Results saved to: {result_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate MemoryVLA on ManiSkill2 (Cross-Environment Generalization)'
    )

    # Required arguments
    parser.add_argument('--ckpt-path', type=str, required=True,
                        help='Path to the trained MemoryVLA checkpoint')
    parser.add_argument('--env-name', type=str, required=True,
                        help='ManiSkill2 environment name (e.g., PickCube-v0)')
    parser.add_argument('--task-instruction', type=str, required=True,
                        help='Natural language task instruction')

    # Evaluation settings
    parser.add_argument('--num-episodes', type=int, default=50,
                        help='Number of episodes to evaluate')
    parser.add_argument('--max-steps', type=int, default=200,
                        help='Maximum steps per episode')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Model settings
    parser.add_argument('--unnorm-key', type=str, default='libero_spatial_no_noops',
                        help='Action normalization key from training dataset')
    parser.add_argument('--action-scale', type=float, default=1.0,
                        help='Action scaling factor')
    parser.add_argument('--cfg-scale', type=float, default=1.5,
                        help='Classifier-free guidance scale')
    parser.add_argument('--use-ddim', action='store_true', default=True,
                        help='Use DDIM sampling')
    parser.add_argument('--num-ddim-steps', type=int, default=10,
                        help='Number of DDIM steps')

    # ManiSkill2 settings
    parser.add_argument('--control-mode', type=str, default='pd_ee_delta_pose',
                        help='ManiSkill2 control mode')
    parser.add_argument('--obs-mode', type=str, default='rgbd',
                        help='ManiSkill2 observation mode')

    # Model architecture
    parser.add_argument('--action-model-type', type=str, default='DiT-L',
                        help='Action model type')
    parser.add_argument('--future-action-window-size', type=int, default=15,
                        help='Future action window size')
    parser.add_argument('--action-dim', type=int, default=7,
                        help='Action dimension')

    # Output settings
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save results')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed step-by-step info')

    args = parser.parse_args()

    # Run evaluation
    results = evaluate_maniskill2(
        ckpt_path=args.ckpt_path,
        env_name=args.env_name,
        task_instruction=args.task_instruction,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        unnorm_key=args.unnorm_key,
        control_mode=args.control_mode,
        obs_mode=args.obs_mode,
        action_scale=args.action_scale,
        cfg_scale=args.cfg_scale,
        save_dir=args.save_dir,
        seed=args.seed,
        verbose=args.verbose,
        action_model_type=args.action_model_type,
        future_action_window_size=args.future_action_window_size,
        action_dim=args.action_dim,
        use_ddim=args.use_ddim,
        num_ddim_steps=args.num_ddim_steps,
    )

    return results


if __name__ == "__main__":
    main()
