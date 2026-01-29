"""
RoboCasa Zero-Shot Evaluation for MemoryVLA
Evaluate LIBERO-pretrained checkpoint on RoboCasa tasks (OOD generalization)
"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import json
from datetime import datetime

# RoboCasa imports
sys.path.append('/content/robocasa')
from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS
from robocasa.utils.env_utils import create_env

# MemoryVLA imports
sys.path.append('/content/MemoryVLA')
from memvla.models import load_vla


class MemVLAPolicyForRoboCasa:
    """MemoryVLA policy wrapper for RoboCasa evaluation"""

    def __init__(self, checkpoint_path, unnorm_key, use_bf16=True, action_chunking_window=8):
        print(f"Loading MemoryVLA from {checkpoint_path}")
        self.vla = load_vla(model_id_or_path=checkpoint_path, load_for_training=False)
        self.vla = self.vla.to("cuda").eval()

        if use_bf16:
            self.vla = self.vla.to(torch.bfloat16)
            print("Using bfloat16 precision")

        self.unnorm_key = unnorm_key
        self.action_chunking_window = action_chunking_window
        self.action_queue = []

        print(f"✅ MemoryVLA loaded (unnorm_key: {unnorm_key}, chunking: {action_chunking_window})")

    def start_episode(self):
        """Reset action queue for new episode"""
        self.action_queue = []

    def predict(self, obs, instruction, episode_first_frame=False):
        """
        Predict action from observation

        Args:
            obs: RoboCasa observation dict with 'robot0_agentview_left_image'
            instruction: Language instruction string
            episode_first_frame: Whether this is the first frame

        Returns:
            action: 7D action array [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        # Use action queue if available
        if len(self.action_queue) > 0:
            return self.action_queue.pop(0)

        # Get image from RoboCasa observation
        # RoboCasa uses 'robot0_agentview_left_image' as main camera
        image = obs['robot0_agentview_left_image']  # Shape: (H, W, 3), RGB

        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))

        # Predict actions using MemoryVLA
        with torch.no_grad():
            unnormed_actions, _ = self.vla.predict_action(
                image=image,
                instruction=instruction,
                unnorm_key=self.unnorm_key,
                cfg_scale=1.5,
                use_ddim=True,
                num_ddim_steps=10,
                episode_first_frame='True' if episode_first_frame else 'False',
            )

        # Fill action queue with chunked actions
        for i in range(min(self.action_chunking_window, len(unnormed_actions))):
            action = unnormed_actions[i]
            if torch.is_tensor(action):
                action = action.cpu().numpy()
            self.action_queue.append(action)

        # Return first action
        return self.action_queue.pop(0) if self.action_queue else unnormed_actions[0].cpu().numpy()


def evaluate_task(env_name, policy, num_episodes=10, max_steps=400, render=False):
    """
    Evaluate policy on a single RoboCasa task

    Args:
        env_name: RoboCasa environment name (e.g., 'PnPCounterToCab')
        policy: MemVLAPolicyForRoboCasa instance
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        render: Whether to render (only works on local, not Colab)

    Returns:
        results: Dict with success_rate, episode_rewards, etc.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {env_name}")
    print(f"{'='*60}")

    # Create environment
    env = create_env(
        env_name=env_name,
        render_onscreen=render,
        seed=0,
        horizon=max_steps,
    )

    successes = []
    rewards = []
    episode_lengths = []

    for ep in tqdm(range(num_episodes), desc=f"{env_name}"):
        # Reset episode
        policy.start_episode()
        obs = env.reset()

        # Get language instruction
        ep_meta = env.get_ep_meta()
        instruction = ep_meta.get("lang", "complete the task")

        total_reward = 0
        success = False

        for step in range(max_steps):
            # Predict action
            action = policy.predict(
                obs=obs,
                instruction=instruction,
                episode_first_frame=(step == 0)
            )

            # Execute action
            obs, reward, done, info = env.step(action)
            total_reward += reward

            # Check success
            if 'success' in info:
                success = info['success']

            if done or success:
                break

        successes.append(1 if success else 0)
        rewards.append(total_reward)
        episode_lengths.append(step + 1)

        if (ep + 1) % 5 == 0:
            current_sr = np.mean(successes) * 100
            print(f"  Episode {ep+1}/{num_episodes} | Success Rate: {current_sr:.1f}%")

    env.close()

    results = {
        'env_name': env_name,
        'success_rate': np.mean(successes) * 100,
        'num_successes': sum(successes),
        'num_episodes': num_episodes,
        'mean_reward': np.mean(rewards),
        'mean_episode_length': np.mean(episode_lengths),
    }

    print(f"\n✅ {env_name} Results:")
    print(f"   Success Rate: {results['success_rate']:.1f}% ({results['num_successes']}/{num_episodes})")
    print(f"   Mean Reward: {results['mean_reward']:.2f}")
    print(f"   Mean Length: {results['mean_episode_length']:.1f} steps")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to MemoryVLA checkpoint')
    parser.add_argument('--unnorm_key', type=str, default='libero_spatial_no_noops',
                        help='Action unnormalization key')
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='Number of episodes per task')
    parser.add_argument('--max_steps', type=int, default=400,
                        help='Maximum steps per episode')
    parser.add_argument('--use_bf16', action='store_true', default=True,
                        help='Use bfloat16 precision')
    parser.add_argument('--action_chunking_window', type=int, default=8,
                        help='Action chunking window size')
    parser.add_argument('--tasks', type=str, nargs='+', default=None,
                        help='Specific tasks to evaluate (default: sample 10 atomic tasks)')
    parser.add_argument('--log_dir', type=str, default='/content/logs/robocasa_zero_shot',
                        help='Directory to save results')

    args = parser.parse_args()

    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)

    # Load policy
    policy = MemVLAPolicyForRoboCasa(
        checkpoint_path=args.checkpoint_path,
        unnorm_key=args.unnorm_key,
        use_bf16=args.use_bf16,
        action_chunking_window=args.action_chunking_window,
    )

    # Select tasks to evaluate
    if args.tasks is None:
        # Sample 10 atomic tasks (easier for zero-shot)
        atomic_tasks = [
            'PnPCounterToCab',
            'PnPCabToCounter',
            'PnPCounterToSink',
            'PnPSinkToCounter',
            'PnPCounterToMicrowave',
            'PnPMicrowaveToCounter',
            'OpenSingleDoor',
            'CloseSingleDoor',
            'TurnOnMicrowave',
            'TurnOffMicrowave',
        ]
        tasks = atomic_tasks[:10]
    else:
        tasks = args.tasks

    print(f"\n{'='*60}")
    print(f"RoboCasa Zero-Shot Evaluation")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Tasks: {len(tasks)}")
    print(f"Episodes per task: {args.num_episodes}")
    print(f"Total episodes: {len(tasks) * args.num_episodes}")
    print(f"{'='*60}\n")

    # Evaluate all tasks
    all_results = []
    for task_name in tasks:
        results = evaluate_task(
            env_name=task_name,
            policy=policy,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            render=False,
        )
        all_results.append(results)

    # Compute overall statistics
    overall_sr = np.mean([r['success_rate'] for r in all_results])

    print(f"\n{'='*60}")
    print(f"OVERALL RESULTS")
    print(f"{'='*60}")
    print(f"Mean Success Rate: {overall_sr:.1f}%")
    print(f"Tasks evaluated: {len(tasks)}")
    for r in all_results:
        print(f"  {r['env_name']}: {r['success_rate']:.1f}%")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.log_dir, f'results_{timestamp}.json')

    summary = {
        'checkpoint_path': args.checkpoint_path,
        'unnorm_key': args.unnorm_key,
        'num_episodes_per_task': args.num_episodes,
        'overall_success_rate': overall_sr,
        'task_results': all_results,
        'timestamp': timestamp,
    }

    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Results saved to: {results_file}")


if __name__ == "__main__":
    main()
