"""
Colab-optimized LIBERO evaluation script for MemoryVLA
- No Flask server required (direct model inference)
- Reduced trials for faster testing
- bfloat16 support for memory efficiency
"""

import os
os.environ['MUJOCO_GL'] = 'osmesa'

import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import math

from libero.libero import benchmark
from evaluation.libero.libero_utils import (
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from evaluation.libero.robot_utils import set_seed_everywhere, DATE_TIME
from vla import load_vla

# TensorFlow CPU only
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')


class DirectMemVLAPolicy:
    """Direct VLA policy without Flask server"""

    def __init__(self, checkpoint_path, unnorm_key, use_bf16=True, action_chunking_window=8):
        print(f"Loading model from {checkpoint_path}...")

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
        print(f"✅ Model loaded successfully")

    def reset(self):
        """Reset episode state"""
        pass

    def predict(self, image, task_description, episode_first_frame='False'):
        """Predict action from image and task description"""
        # Convert numpy to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Model inference
        with torch.no_grad():
            unnormed_actions, _ = self.vla.predict_action(
                image=image,
                instruction=task_description,
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


def resize_image(image, size=(224, 224)):
    """Resize and center crop image to match training preprocessing"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    w, h = image.size

    # Center crop to square
    left_margin = (w - h) // 2
    left_margin = min(max(left_margin, 0), w - h)
    image = image.crop((left_margin, 0, left_margin + h, h))

    # Resize to target size
    image = image.resize(size, resample=Image.LANCZOS)

    # Apply scale augmentation (matching training)
    scale = 0.9
    new_w = int(size[0] * math.sqrt(scale))
    new_h = int(size[1] * math.sqrt(scale))
    margin_w = (size[0] - new_w) // 2
    margin_h = (size[1] - new_h) // 2

    image = image.crop((margin_w, margin_h, margin_w + new_w, margin_h + new_h))
    image = image.resize(size, resample=Image.LANCZOS)

    return image


def eval_libero_colab(
    checkpoint_path,
    task_suite_name="libero_spatial",
    unnorm_key="libero_spatial_no_noops",
    num_trials_per_task=10,
    seed=7,
    use_bf16=True,
    action_chunking_window=8,
    log_dir=None,
    specific_task_ids=None,
):
    """
    Evaluate MemoryVLA on LIBERO benchmark (Colab-optimized)

    Args:
        checkpoint_path: Path to pretrained checkpoint
        task_suite_name: LIBERO task suite (libero_spatial, libero_object, libero_goal, libero_10, libero_90)
        unnorm_key: Unnormalization key for action denormalization
        num_trials_per_task: Number of trials per task (default: 10 for Colab)
        seed: Random seed
        use_bf16: Use bfloat16 for memory efficiency
        action_chunking_window: Number of actions to predict per step
        log_dir: Directory to save logs and videos
        specific_task_ids: List of specific task IDs to evaluate (None = all tasks)
    """

    # Set random seed
    set_seed_everywhere(seed)

    # Setup log directory
    if log_dir is None:
        log_dir = f"/content/logs/{task_suite_name}-{DATE_TIME}"
    os.makedirs(log_dir, exist_ok=True)
    print(f"📂 Logs will be saved to: {log_dir}")

    # Initialize policy
    policy = DirectMemVLAPolicy(
        checkpoint_path=checkpoint_path,
        unnorm_key=unnorm_key,
        use_bf16=use_bf16,
        action_chunking_window=action_chunking_window,
    )

    # Load LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks = task_suite.n_tasks
    print(f"\n{'='*70}")
    print(f"📋 Task Suite: {task_suite_name}")
    print(f"📋 Total Tasks: {num_tasks}")
    print(f"📋 Trials per Task: {num_trials_per_task}")
    print(f"{'='*70}\n")

    # Task-specific max steps
    max_steps_dict = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    max_steps = max_steps_dict.get(task_suite_name, 300)
    num_wait_steps = 10  # Wait for objects to stabilize

    # Evaluation loop
    total_episodes = 0
    total_successes = 0
    task_results = {}

    task_ids_to_eval = specific_task_ids if specific_task_ids is not None else range(num_tasks)

    for task_id in task_ids_to_eval:
        # Get task
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, resolution=256)

        print(f"\n{'='*70}")
        print(f"🎯 Task {task_id+1}/{num_tasks}: {task_description}")
        print(f"{'='*70}")

        task_successes = 0
        task_episodes = 0

        for episode_idx in tqdm(range(num_trials_per_task), desc=f"Task {task_id+1}"):
            # Reset environment
            env.reset()
            policy.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            # Episode variables
            episode_first_frame = 'True'
            replay_images = []
            t = 0
            done = False

            # Episode rollout
            while t < max_steps + num_wait_steps:
                try:
                    # Wait for objects to stabilize
                    if t < num_wait_steps:
                        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                        t += 1
                        continue

                    # Get and preprocess image
                    img = get_libero_image(obs, resize_size=256)
                    replay_images.append(img)
                    resized_img = resize_image(img, size=(224, 224))

                    # Predict actions
                    actions = policy.predict(
                        image=resized_img,
                        task_description=task_description,
                        episode_first_frame=episode_first_frame,
                    )
                    episode_first_frame = 'False'

                    # Execute action chunking
                    done_flag = False
                    for action in actions:
                        # Convert gripper action (1.0 → -1.0, 0.0 → 1.0)
                        if action[6] == 1.0:
                            action[6] = -1.0
                        elif action[6] == 0.0:
                            action[6] = 1.0

                        # Step environment
                        obs, reward, done, info = env.step(action)
                        t += 1

                        if done:
                            task_successes += 1
                            total_successes += 1
                            done_flag = True
                            break

                    if done_flag:
                        break

                except Exception as e:
                    print(f"❌ Exception during episode: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save rollout video
            video_dir = os.path.join(log_dir, "videos")
            os.makedirs(video_dir, exist_ok=True)

            save_rollout_video(
                replay_images,
                total_episodes,
                success=done,
                task_description=task_description,
                log_file=None,
                rollout_dir=video_dir,
            )

            # Print episode result
            status = "✅ Success" if done else "❌ Fail"
            print(f"  Episode {episode_idx+1}/{num_trials_per_task}: {status}")

        # Task summary
        task_success_rate = task_successes / task_episodes * 100
        task_results[task_id] = {
            'description': task_description,
            'successes': task_successes,
            'episodes': task_episodes,
            'success_rate': task_success_rate,
        }

        print(f"\n📊 Task {task_id+1} Results:")
        print(f"   Success Rate: {task_success_rate:.1f}% ({task_successes}/{task_episodes})")

    # Final summary
    total_success_rate = total_successes / total_episodes * 100

    print(f"\n{'='*70}")
    print(f"🎯 FINAL EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"Total Episodes: {total_episodes}")
    print(f"Total Successes: {total_successes}")
    print(f"Overall Success Rate: {total_success_rate:.1f}%")
    print(f"{'='*70}\n")

    # Save detailed results
    results_file = os.path.join(log_dir, "results.txt")
    with open(results_file, 'w') as f:
        f.write(f"MemoryVLA LIBERO Evaluation Results\n")
        f.write(f"{'='*70}\n")
        f.write(f"Task Suite: {task_suite_name}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Trials per Task: {num_trials_per_task}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"{'='*70}\n\n")

        for task_id, result in task_results.items():
            f.write(f"Task {task_id+1}: {result['description']}\n")
            f.write(f"  Success Rate: {result['success_rate']:.1f}% ")
            f.write(f"({result['successes']}/{result['episodes']})\n\n")

        f.write(f"{'='*70}\n")
        f.write(f"Overall Success Rate: {total_success_rate:.1f}% ")
        f.write(f"({total_successes}/{total_episodes})\n")

    print(f"📄 Results saved to: {results_file}")
    print(f"🎥 Videos saved to: {video_dir}")

    return task_results, total_success_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MemoryVLA on LIBERO (Colab-optimized)")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--task_suite_name", type=str, default="libero_spatial",
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"])
    parser.add_argument("--unnorm_key", type=str, default="libero_spatial_no_noops",
                        help="Unnormalization key")
    parser.add_argument("--num_trials_per_task", type=int, default=10,
                        help="Number of trials per task")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--use_bf16", action="store_true", default=True,
                        help="Use bfloat16 inference")
    parser.add_argument("--action_chunking_window", type=int, default=8,
                        help="Action chunking window size")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Log directory (default: /content/logs/<suite>-<timestamp>)")
    parser.add_argument("--specific_task_ids", type=int, nargs='+', default=None,
                        help="Specific task IDs to evaluate (e.g., --specific_task_ids 0 1 2)")

    args = parser.parse_args()

    # Run evaluation
    task_results, success_rate = eval_libero_colab(
        checkpoint_path=args.checkpoint_path,
        task_suite_name=args.task_suite_name,
        unnorm_key=args.unnorm_key,
        num_trials_per_task=args.num_trials_per_task,
        seed=args.seed,
        use_bf16=args.use_bf16,
        action_chunking_window=args.action_chunking_window,
        log_dir=args.log_dir,
        specific_task_ids=args.specific_task_ids,
    )

    print(f"\n✅ Evaluation complete! Success rate: {success_rate:.1f}%")
