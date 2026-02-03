"""
Extract evaluation results from ManiSkill2 Franka Panda generalization experiments.

This script summarizes success rates across different tasks when testing LIBERO-trained
models on ManiSkill2 environments (cross-environment generalization with same robot).

Usage:
    python script/eval/maniskill2_franka/extract_maniskill2_franka_results.py \
        --eval-dir ./eval_results/maniskill2_generalization/checkpoint_name/
"""

import os
import re
import argparse
from pathlib import Path
from glob import glob


def extract_success_rate_from_log(log_file: str) -> tuple:
    """
    Extract success rate from evaluation log file.

    Supports two formats:
    1. New format: "Success Rate: XX.X% (N/M)"
    2. Legacy format: counting 'success' and 'failure' keywords

    Returns:
        tuple: (num_success, num_total, success_rate) or (None, None, None) if not found
    """
    if not os.path.exists(log_file):
        return None, None, None

    with open(log_file, 'r') as f:
        content = f.read()

    # Try new format first: "Success Rate: XX.X% (N/M)"
    match = re.search(r'Success Rate:\s*([\d.]+)%\s*\((\d+)/(\d+)\)', content)
    if match:
        success_rate = float(match.group(1))
        num_success = int(match.group(2))
        num_total = int(match.group(3))
        return num_success, num_total, success_rate

    # Fallback to counting keywords
    # Count episode results: "Episode X: SUCCESS" or "Episode X: FAILURE"
    successes = len(re.findall(r'Episode\s+\d+.*?:\s*SUCCESS', content, re.IGNORECASE))
    failures = len(re.findall(r'Episode\s+\d+.*?:\s*FAILURE', content, re.IGNORECASE))

    # If not found, try lowercase
    if successes == 0 and failures == 0:
        successes = content.lower().count('success')
        failures = content.lower().count('failure')

    total = successes + failures
    if total == 0:
        return None, None, None

    success_rate = successes / total * 100
    return successes, total, success_rate


def extract_success_rate_from_txt(txt_file: str) -> tuple:
    """
    Extract success rate from .txt result file (saved by evaluator).

    Returns:
        tuple: (num_success, num_total, success_rate) or (None, None, None)
    """
    if not os.path.exists(txt_file):
        return None, None, None

    with open(txt_file, 'r') as f:
        content = f.read()

    # Look for "Success Rate: XX.X%"
    match = re.search(r'Success Rate:\s*([\d.]+)%', content)
    if match:
        success_rate = float(match.group(1))

        # Try to find episode count
        episode_match = re.search(r'\((\d+)/(\d+)\)', content)
        if episode_match:
            num_success = int(episode_match.group(1))
            num_total = int(episode_match.group(2))
        else:
            # Estimate from success rate
            num_total = 50  # default
            num_success = int(success_rate * num_total / 100)

        return num_success, num_total, success_rate

    return None, None, None


def main():
    parser = argparse.ArgumentParser(
        description='Extract ManiSkill2 Franka Panda cross-environment generalization results'
    )
    parser.add_argument('--eval-dir', type=str, required=True,
                        help='Directory containing evaluation results')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for summary (optional)')
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)

    # Tasks to look for (in order)
    tasks = [
        ('PickCube', ['PickCube.log', 'PickCube.txt', 'PickCube-v0*.txt']),
        ('StackCube', ['StackCube.log', 'StackCube.txt', 'StackCube-v0*.txt']),
        ('PickSingleYCB', ['PickSingleYCB.log', 'PickSingleYCB.txt', 'PickSingleYCB-v0*.txt']),
        ('PickSingleEGAD', ['PickSingleEGAD.log', 'PickSingleEGAD.txt', 'PickSingleEGAD-v0*.txt']),
        ('PickClutterYCB', ['PickClutterYCB.log', 'PickClutterYCB.txt', 'PickClutterYCB-v0*.txt']),
    ]

    print()
    print("=" * 70)
    print("  ManiSkill2 Cross-Environment Generalization Results")
    print("  (LIBERO checkpoint tested on ManiSkill2 Franka Panda environments)")
    print("=" * 70)
    print(f"  Eval directory: {eval_dir}")
    print("=" * 70)
    print()
    print(f"{'Task':<20} {'Success Rate':>15} {'Episodes':>15}")
    print("-" * 50)

    total_success = 0
    total_episodes = 0
    results = []

    for task_name, file_patterns in tasks:
        num_success, num_total, success_rate = None, None, None

        # Try each file pattern
        for pattern in file_patterns:
            if '*' in pattern:
                # Glob pattern
                matches = list(eval_dir.glob(pattern))
                if matches:
                    file_path = matches[0]  # Take first match
                else:
                    continue
            else:
                file_path = eval_dir / pattern

            # Try to extract from this file
            if file_path.suffix == '.log':
                num_success, num_total, success_rate = extract_success_rate_from_log(str(file_path))
            else:
                num_success, num_total, success_rate = extract_success_rate_from_txt(str(file_path))

            if success_rate is not None:
                break

        if success_rate is not None:
            results.append({
                'task': task_name,
                'success': num_success,
                'total': num_total,
                'rate': success_rate
            })
            total_success += num_success
            total_episodes += num_total
            print(f"{task_name:<20} {success_rate:>14.1f}% {num_success:>7}/{num_total:<7}")
        else:
            print(f"{task_name:<20} {'N/A':>15} {'N/A':>15}")

    print("-" * 50)

    if total_episodes > 0:
        avg_success_rate = total_success / total_episodes * 100
        print(f"{'AVERAGE':<20} {avg_success_rate:>14.1f}% {total_success:>7}/{total_episodes:<7}")
    else:
        print("No valid results found.")

    print("=" * 70)
    print()

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            f.write("ManiSkill2 Cross-Environment Generalization Results\n")
            f.write(f"Eval directory: {eval_dir}\n\n")
            for r in results:
                f.write(f"{r['task']}: {r['rate']:.1f}% ({r['success']}/{r['total']})\n")
            if total_episodes > 0:
                f.write(f"\nAverage: {avg_success_rate:.1f}% ({total_success}/{total_episodes})\n")
        print(f"Results saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
