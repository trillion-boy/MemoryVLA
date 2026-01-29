"""
CALVIN evaluation script for MemoryVLA (Colab-optimized)
- Zero-shot transfer from LIBERO pretrained checkpoint
- No training, only inference
- Tests generalization to unseen environment
"""

import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add MemoryVLA to path
sys.path.insert(0, '/content/MemoryVLA')

from vla import load_vla

# CALVIN imports
import gym
import calvin_env

# TensorFlow CPU only
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')


class DirectMemVLAPolicy:
    """Direct VLA policy for CALVIN evaluation"""

    def __init__(self, checkpoint_path, unnorm_key, use_bf16=True, action_chunking_window=8):
        print(f"Loading MemoryVLA from {checkpoint_path}...")

        self.vla = load_vla(
            model_id_or_path=checkpoint_path,
            load_for_training=False,
        )
        self.vla = self.vla.to("cuda").eval()

        if use_bf16:
            self.vla = self.vla.to(torch.bfloat16)
            print("✅ Using bfloat16 inference mode")
        else:
            print("✅ Using float32 inference mode")

        self.unnorm_key = unnorm_key
        self.action_chunking_window = action_chunking_window
        print(f"✅ MemoryVLA loaded successfully")

    def reset(self):
        """Reset episode state"""
        pass

    def predict(self, image, instruction, episode_first_frame='False'):
        """Predict action from image and instruction"""
        # Convert numpy to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Model inference
        with torch.no_grad():
            unnormed_actions, _ = self.vla.predict_action(
                image=image,
                instruction=instruction,
                unnorm_key=self.unnorm_key,
                cfg_scale=1.5,
                use_ddim=True,
                num_ddim_steps=10,
                episode_first_frame=episode_first_frame,
            )

        # Action chunking
        actions = []
        for i in range(min(self.action_chunking_window, len(unnormed_actions))):
            action = unnormed_actions[i]
            # Handle both torch tensors and numpy arrays
            if torch.is_tensor(action):
                action = action.cpu().numpy()
            actions.append(action)

        return actions


def eval_calvin_colab(
    checkpoint_path,
    dataset_path,
    num_episodes=50,
    unnorm_key="libero_spatial_no_noops",
    use_bf16=True,
    action_chunking_window=8,
    log_dir=None,
):
    """
    Zero-shot evaluation of MemoryVLA on CALVIN benchmark

    Args:
        checkpoint_path: Path to pretrained MemoryVLA checkpoint (e.g., LIBERO)
        dataset_path: Path to CALVIN dataset
        num_episodes: Number of episodes to evaluate
        unnorm_key: Unnormalization key from training dataset
        use_bf16: Use bfloat16 for memory efficiency
        action_chunking_window: Number of actions per observation
        log_dir: Directory to save logs
    """

    # Setup log directory
    if log_dir is None:
        log_dir = f"/content/logs/calvin_zero_shot"
    os.makedirs(log_dir, exist_ok=True)
    print(f"📂 Logs will be saved to: {log_dir}")

    # Initialize MemoryVLA policy
    policy = DirectMemVLAPolicy(
        checkpoint_path=checkpoint_path,
        unnorm_key=unnorm_key,
        use_bf16=use_bf16,
        action_chunking_window=action_chunking_window,
    )

    # Load CALVIN environment
    print(f"\n{'='*70}")
    print(f"🎯 Loading CALVIN environment from {dataset_path}")
    print(f"{'='*70}\n")

    env = gym.make(dataset_path)

    # Get task instructions from CALVIN
    # CALVIN uses language annotations
    task_instructions = [
        "rotate the blue block to the right",
        "push the red block to the left",
        "lift the pink block",
        "open the drawer",
        "close the drawer",
        # Add more CALVIN tasks as needed
    ]

    total_episodes = 0
    total_successes = 0
    episode_results = []

    print(f"📊 Evaluating {num_episodes} episodes")
    print(f"{'='*70}\n")

    for episode_idx in tqdm(range(num_episodes), desc="Episodes"):
        # Reset environment
        obs = env.reset()
        policy.reset()

        # Select random instruction (CALVIN is language-conditioned)
        instruction = np.random.choice(task_instructions)

        episode_first_frame = 'True'
        episode_reward = 0
        done = False
        step = 0
        max_steps = 360  # CALVIN episode length

        while not done and step < max_steps:
            # Get RGB image from observation
            # CALVIN provides multiple camera views
            rgb_static = obs['rgb_obs']['rgb_static']  # Static camera

            # Predict actions
            actions = policy.predict(
                image=rgb_static,
                instruction=instruction,
                episode_first_frame=episode_first_frame,
            )
            episode_first_frame = 'False'

            # Execute action chunking
            for action in actions:
                # CALVIN uses 7D action space (same as LIBERO)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                step += 1

                if done:
                    break

        # Record results
        success = done  # CALVIN sets done=True on success
        total_episodes += 1
        if success:
            total_successes += 1

        episode_results.append({
            'episode': episode_idx,
            'instruction': instruction,
            'success': success,
            'reward': episode_reward,
            'steps': step,
        })

        # Print progress
        status = "✅ Success" if success else "❌ Fail"
        print(f"  Episode {episode_idx+1}/{num_episodes}: {status} | Reward: {episode_reward:.2f} | Steps: {step}")

    # Calculate success rate
    success_rate = total_successes / total_episodes * 100

    # Print final results
    print(f"\n{'='*70}")
    print(f"🎯 FINAL RESULTS - Zero-shot CALVIN Evaluation")
    print(f"{'='*70}")
    print(f"Total Episodes: {total_episodes}")
    print(f"Total Successes: {total_successes}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"{'='*70}\n")

    # Save results
    results_file = os.path.join(log_dir, "results.txt")
    with open(results_file, 'w') as f:
        f.write(f"MemoryVLA Zero-shot CALVIN Evaluation\n")
        f.write(f"{'='*70}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Unnorm Key: {unnorm_key}\n")
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Episodes: {num_episodes}\n")
        f.write(f"{'='*70}\n\n")

        for result in episode_results:
            f.write(f"Episode {result['episode']}: {result['instruction']}\n")
            f.write(f"  Success: {result['success']}, Reward: {result['reward']:.2f}, Steps: {result['steps']}\n\n")

        f.write(f"{'='*70}\n")
        f.write(f"Overall Success Rate: {success_rate:.1f}% ({total_successes}/{total_episodes})\n")

    print(f"📄 Results saved to: {results_file}")

    return episode_results, success_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot CALVIN evaluation for MemoryVLA")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to MemoryVLA checkpoint")
    parser.add_argument("--dataset_path", type=str, default="calvin_env/playdata/task_D_D/validation",
                        help="CALVIN dataset path")
    parser.add_argument("--num_episodes", type=int, default=50, help="Number of episodes")
    parser.add_argument("--unnorm_key", type=str, default="libero_spatial_no_noops",
                        help="Unnormalization key from training dataset")
    parser.add_argument("--use_bf16", action="store_true", default=True, help="Use bfloat16")
    parser.add_argument("--action_chunking_window", type=int, default=8, help="Action chunking window")
    parser.add_argument("--log_dir", type=str, default=None, help="Log directory")

    args = parser.parse_args()

    results, success_rate = eval_calvin_colab(
        checkpoint_path=args.checkpoint_path,
        dataset_path=args.dataset_path,
        num_episodes=args.num_episodes,
        unnorm_key=args.unnorm_key,
        use_bf16=args.use_bf16,
        action_chunking_window=args.action_chunking_window,
        log_dir=args.log_dir,
    )

    print(f"\n✅ Evaluation complete! Success rate: {success_rate:.1f}%")
