"""
LIBERO environment per_attn diagnostic script.

Loads a LIBERO spatial task, captures initial observation images,
and runs the 3-level per_attn diagnostics to verify whether
per_attn is alive in-domain (LIBERO ckpt + LIBERO env).

Usage (from repo root):
    export MUJOCO_GL='osmesa'
    python scripts/diagnose_libero_per_attn.py \
        --model_path <path_to_ckpt> \
        --task_suite_name libero_spatial \
        --task_id 0

Compares with cross-domain (Maniskill) results to determine if
per_attn is inherently dead vs killed by domain shift.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import torch
from PIL import Image

# LIBERO imports
from libero.libero import benchmark
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evaluation", "libero"))
from libero_utils import get_libero_env, get_libero_image, quat2axisangle

# MemoryVLA imports
from vla.load import load_vla
from scripts.colab_inference_with_diagnostics import (
    diagnose_per_attn_weights,
    diagnose_per_attn_activation,
    diagnose_output_sensitivity,
)

# Suppress TF GPU usage (avoids conflict with PyTorch)
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")


def capture_libero_observation(
    task_suite_name: str = "libero_spatial",
    task_id: int = 0,
    episode_idx: int = 0,
    resolution: int = 256,
    num_steps_wait: int = 10,
):
    """
    Initialize a LIBERO env, reset to initial state, wait for stabilization,
    and return (pil_image, task_description, raw_obs).
    """
    print(f"\n{'='*60}")
    print(f"  Capturing LIBERO observation")
    print(f"  Suite: {task_suite_name}, Task: {task_id}, Episode: {episode_idx}")
    print(f"{'='*60}")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)

    env, task_description = get_libero_env(task, resolution=resolution)
    print(f"  Task: {task_description}")

    # Reset and set initial state
    env.reset()
    obs = env.set_init_state(initial_states[episode_idx])

    # Wait for objects to stabilize
    for _ in range(num_steps_wait):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])

    # Get preprocessed image (same pipeline as eval_libero.py)
    img_np = get_libero_image(obs, resolution)
    pil_image = Image.fromarray(img_np)
    print(f"  Image shape: {img_np.shape}, dtype: {img_np.dtype}")
    print(f"  Instruction: \"{task_description}\"")

    env.close()
    return pil_image, task_description, obs


def run_libero_diagnostics(
    model_path: str,
    task_suite_name: str = "libero_spatial",
    task_id: int = 0,
    episode_idx: int = 0,
    unnorm_key: str = "libero_spatial_no_noops",
    cfg_scale: float = 1.5,
):
    """Run full 3-level diagnostics with a LIBERO observation."""

    # Step 1: Capture LIBERO observation
    pil_image, task_description, raw_obs = capture_libero_observation(
        task_suite_name=task_suite_name,
        task_id=task_id,
        episode_idx=episode_idx,
    )

    # Step 2: Load model
    print(f"\n{'='*60}")
    print(f"  Loading model from: {model_path}")
    print(f"{'='*60}")
    vla = load_vla(model_path)
    vla.eval()
    device = next(vla.parameters()).device
    print(f"  Device: {device}")

    # Step 3: Run diagnostics
    results = {}

    # --- Level 1: Weight magnitude ---
    print(f"\n{'='*60}")
    print(f"  Level 1: Weight Magnitude")
    print(f"{'='*60}")
    lv1 = diagnose_per_attn_weights(vla)
    results["level1"] = lv1
    for b in lv1["blocks"][:3]:
        print(f"  Block {b['block_idx']:2d}: in_proj mean={b['in_proj_mean_abs']:.6f} max={b['in_proj_max_abs']:.6f}  risk={b['risk']}")
    if len(lv1["blocks"]) > 3:
        print(f"  ... ({len(lv1['blocks'])} blocks total)")
    print(f"  Verdict: {lv1['verdict']}")
    print(f"  >> {lv1['summary']}")

    # --- Level 2: Activation ratio ---
    print(f"\n{'='*60}")
    print(f"  Level 2: Activation Ratio ||x_c|| / ||x||")
    print(f"{'='*60}")
    lv2 = diagnose_per_attn_activation(vla)
    results["level2"] = lv2
    for r in lv2["ratios"][:3]:
        print(f"  Block {r['block_idx']:2d}: ||x_c||={r['x_c_norm']:.6f}  ||x||={r['x_norm']:.2f}  ratio={r['ratio']:.2e}")
    if len(lv2["ratios"]) > 3:
        print(f"  ... ({len(lv2['ratios'])} blocks total)")
    print(f"  Mean ratio: {lv2['mean_ratio']:.2e}")
    print(f"  Verdict: {lv2['verdict']}")
    print(f"  >> {lv2['summary']}")

    # --- Level 3: Paired-seed output sensitivity (LIBERO image) ---
    print(f"\n{'='*60}")
    print(f"  Level 3: Paired-Seed Output Sensitivity (LIBERO image)")
    print(f"{'='*60}")
    print(f"  Image: LIBERO {task_suite_name} task {task_id}")
    print(f"  Instruction: \"{task_description}\"")
    print()

    lv3 = diagnose_output_sensitivity(
        vla, pil_image, task_description,
        unnorm_key=unnorm_key, cfg_scale=cfg_scale,
    )
    results["level3"] = lv3

    print()
    print("  Per-seed paired deltas:")
    for r in lv3["per_seed"]:
        print(f"    seed={r['seed']:5d}:  A→B={r['delta_ab']:.6f}  A→C={r['delta_ac']:.6f}  B→C={r['delta_bc']:.6f}")

    print()
    print(f"  Null baseline (A vs A, different seeds):")
    for i, nd in enumerate(lv3["null_deltas"]):
        print(f"    pair {i}: {nd:.6f}")
    print(f"    mean noise floor = {lv3['null_baseline']:.6f}")

    print()
    print(f"  Mean paired A→B (per_attn signal) : {lv3['mean_delta_AB']:.6f}")
    print(f"  Mean paired A→C (bypass signal)   : {lv3['mean_delta_AC']:.6f}")
    print(f"  Mean paired B→C (bypass adds)     : {lv3['mean_delta_BC']:.6f}")
    print(f"  Noise floor (A vs A)              : {lv3['null_baseline']:.6f}")
    print(f"  SNR A→B (signal/noise)            : {lv3['snr_AB']:.1f}x")
    print(f"  SNR A→C (signal/noise)            : {lv3['snr_AC']:.1f}x")
    print()
    print(f"  per_attn effective? {lv3['per_attn_effective']}  (need SNR > 2x)")
    print(f"  bypass effective?   {lv3['bypass_effective']}  (need SNR > 2x)")
    print(f"  Verdict: {lv3['verdict']}")
    print(f"  >> {lv3['summary']}")

    # --- Final Summary ---
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY (LIBERO in-domain)")
    print(f"{'='*60}")
    print(f"  Level 1 (weights):     {lv1['verdict']}")
    print(f"  Level 2 (activations): {lv2['verdict']}")
    print(f"  Level 3 (sensitivity): {lv3['verdict']}")
    print()

    if lv3["verdict"] == "PER_ATTN_ALIVE":
        print("  per_attn is ALIVE in-domain.")
        print("  => Maniskill failure is domain shift, not architecture.")
    elif lv3["verdict"] in ("PER_ATTN_DEAD_BYPASS_WORKS", "BOTH_DEAD"):
        print("  per_attn is DEAD even in-domain (LIBERO ckpt + LIBERO env).")
        print("  => This is an architecture/training issue, NOT domain shift.")
        print("  => LIBERO 90%+ success was achieved via cog_tokens alone.")
    print()

    results["task_description"] = task_description
    results["task_suite"] = task_suite_name
    results["task_id"] = task_id
    results["recommendation"] = lv3["verdict"]
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LIBERO per_attn diagnostics")
    parser.add_argument("--model_path", type=str, required=True, help="Path to MemoryVLA checkpoint")
    parser.add_argument("--task_suite_name", type=str, default="libero_spatial", help="LIBERO task suite")
    parser.add_argument("--task_id", type=int, default=0, help="Task ID within suite")
    parser.add_argument("--episode_idx", type=int, default=0, help="Episode/initial state index")
    parser.add_argument("--unnorm_key", type=str, default="libero_spatial_no_noops", help="Unnormalization key")
    parser.add_argument("--cfg_scale", type=float, default=1.5, help="Classifier-free guidance scale")
    args = parser.parse_args()

    results = run_libero_diagnostics(
        model_path=args.model_path,
        task_suite_name=args.task_suite_name,
        task_id=args.task_id,
        episode_idx=args.episode_idx,
        unnorm_key=args.unnorm_key,
        cfg_scale=args.cfg_scale,
    )
