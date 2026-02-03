"""
Extract evaluation results from ManiSkill2 Franka Panda generalization experiments.
This script summarizes success rates across different tasks when testing LIBERO-trained
models on ManiSkill2 environments (cross-environment generalization with same robot).
"""

import os
import re
import argparse
from pathlib import Path


def extract_success_rate(log_file: str) -> tuple:
    """
    Extract success rate from a single evaluation log file.

    Returns:
        tuple: (num_success, num_total, success_rate)
    """
    if not os.path.exists(log_file):
        return None, None, None

    with open(log_file, 'r') as f:
        content = f.read()

    # Count success and failure
    successes = content.count('success')
    failures = content.count('failure')
    total = successes + failures

    if total == 0:
        return None, None, None

    success_rate = successes / total * 100
    return successes, total, success_rate


def main():
    parser = argparse.ArgumentParser(description='Extract ManiSkill2 Franka evaluation results')
    parser.add_argument('--eval-dir', type=str, required=True,
                        help='Directory containing evaluation results')
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)

    # Tasks to look for
    tasks = [
        ('PickCube', 'PickCube.txt'),
        ('StackCube', 'StackCube.txt'),
        ('PickSingleYCB', 'PickSingleYCB.txt'),
        ('PickSingleEGAD', 'PickSingleEGAD.txt'),
        ('PickClutterYCB', 'PickClutterYCB.txt'),
    ]

    print("=" * 70)
    print("ManiSkill2 Franka Panda Cross-Environment Generalization Results")
    print("=" * 70)
    print()

    total_success = 0
    total_episodes = 0
    results = []

    for task_name, log_file in tasks:
        log_path = eval_dir / log_file
        num_success, num_total, success_rate = extract_success_rate(str(log_path))

        if success_rate is not None:
            results.append((task_name, num_success, num_total, success_rate))
            total_success += num_success
            total_episodes += num_total
            print(f"{task_name:20s}: {success_rate:6.2f}% ({num_success}/{num_total})")
        else:
            print(f"{task_name:20s}: Not found or empty")

    print()
    print("-" * 70)

    if total_episodes > 0:
        avg_success_rate = total_success / total_episodes * 100
        print(f"{'Overall':20s}: {avg_success_rate:6.2f}% ({total_success}/{total_episodes})")
    else:
        print("No valid results found.")

    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
