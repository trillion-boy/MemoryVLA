"""
LIBERO in-domain per_attn diagnostic.

Loads a LIBERO spatial task, captures initial observation,
and runs 3-level per_attn diagnostics to determine whether
per_attn is inherently dead (training issue) vs killed by domain shift.

Usage (Colab):
    import os
    os.environ["MUJOCO_GL"] = "osmesa"

    from scripts.diagnose_libero_per_attn import (
        capture_libero_observation,
        run_libero_diagnostics,
    )

    # If model is already loaded:
    result = run_libero_diagnostics(vla_model=vla)

    # If model needs loading:
    result = run_libero_diagnostics(model_path="/path/to/ckpt")
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from PIL import Image


def _patch_torch_load():
    """PyTorch 2.6+ changed torch.load default to weights_only=True.
    LIBERO init_states are pickled numpy arrays, which fail with this.
    Monkey-patch torch.load to default weights_only=False."""
    import functools
    _original = torch.load
    if getattr(_original, "_patched_for_libero", False):
        return
    @functools.wraps(_original)
    def _patched_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _original(*args, **kwargs)
    _patched_load._patched_for_libero = True
    torch.load = _patched_load


def _ensure_libero_imports():
    """Lazy-import LIBERO utilities (handles sys.path for Colab)."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Add third_libs/LIBERO to path (editable install may not be visible to Colab kernel)
    libero_dir = os.path.join(repo_root, "third_libs", "LIBERO")
    if os.path.isdir(libero_dir) and libero_dir not in sys.path:
        sys.path.insert(0, libero_dir)

    # Add evaluation/libero to path for libero_utils
    eval_libero_dir = os.path.join(repo_root, "evaluation", "libero")
    if eval_libero_dir not in sys.path:
        sys.path.insert(0, eval_libero_dir)

    # Fix PyTorch 2.6+ vs LIBERO pickle incompatibility
    _patch_torch_load()

    # Suppress TF GPU (avoids conflict with PyTorch)
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
    and return (pil_image, task_description).

    The image goes through the exact same preprocessing as eval_libero.py:
    180-degree rotation → JPEG encode/decode → lanczos3 resize.
    """
    _ensure_libero_imports()
    from libero.libero import benchmark
    from libero_utils import get_libero_env, get_libero_image

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

    # Wait for objects to stabilize (same as eval_libero.py)
    for _ in range(num_steps_wait):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])

    # Preprocess image (identical pipeline to eval_libero.py)
    img_np = get_libero_image(obs, resolution)
    pil_image = Image.fromarray(img_np)
    print(f"  Image shape: {img_np.shape}, dtype: {img_np.dtype}")
    print(f"  Instruction: \"{task_description}\"")

    env.close()
    return pil_image, task_description


def run_libero_diagnostics(
    vla_model=None,
    model_path: str = None,
    task_suite_name: str = "libero_spatial",
    task_id: int = 0,
    episode_idx: int = 0,
    unnorm_key: str = "libero_spatial_no_noops",
    cfg_scale: float = 1.5,
):
    """
    Run full 3-level diagnostics with a LIBERO observation.

    Args:
        vla_model: Already-loaded MemoryVLA model (preferred in Colab).
        model_path: Path to checkpoint. Used only if vla_model is None.
        task_suite_name: LIBERO task suite name.
        task_id: Task index within suite.
        episode_idx: Initial state / episode index.
        unnorm_key: Unnormalization key for action decoding.
        cfg_scale: Classifier-free guidance scale.

    Returns:
        dict with level1/level2/level3 results and recommendation.
    """
    from scripts.colab_inference_with_diagnostics import (
        diagnose_per_attn_weights,
        diagnose_per_attn_activation,
        diagnose_output_sensitivity,
    )

    # --- Step 1: Capture LIBERO observation ---
    pil_image, task_description = capture_libero_observation(
        task_suite_name=task_suite_name,
        task_id=task_id,
        episode_idx=episode_idx,
    )

    # --- Step 2: Get model ---
    if vla_model is None:
        assert model_path is not None, "Provide either vla_model or model_path"
        from vla.load import load_vla
        print(f"\n  Loading model from: {model_path}")
        vla_model = load_vla(model_path)
    vla_model.eval()
    print(f"  Device: {next(vla_model.parameters()).device}")

    # --- Step 3: Run 3-level diagnostics ---
    results = {}

    # Level 1
    print(f"\n{'='*60}")
    print(f"  Level 1: Weight Magnitude")
    print(f"{'='*60}")
    lv1 = diagnose_per_attn_weights(vla_model)
    results["level1"] = lv1
    for b in lv1["blocks"][:3]:
        print(
            f"  Block {b['block_idx']:2d}: "
            f"in_proj mean={b['in_proj_mean_abs']:.6f} "
            f"max={b['in_proj_max_abs']:.6f}  "
            f"risk={b['risk']}"
        )
    if len(lv1["blocks"]) > 3:
        print(f"  ... ({len(lv1['blocks'])} blocks total)")
    print(f"  Verdict: {lv1['verdict']}")
    print(f"  >> {lv1['summary']}")

    # Level 2
    print(f"\n{'='*60}")
    print(f"  Level 2: Activation Ratio ||x_c|| / ||x||")
    print(f"{'='*60}")
    lv2 = diagnose_per_attn_activation(vla_model)
    results["level2"] = lv2
    for r in lv2["ratios"][:3]:
        print(
            f"  Block {r['block_idx']:2d}: "
            f"||x_c||={r['x_c_norm']:.6f}  "
            f"||x||={r['x_norm']:.2f}  "
            f"ratio={r['ratio']:.2e}"
        )
    if len(lv2["ratios"]) > 3:
        print(f"  ... ({len(lv2['ratios'])} blocks total)")
    print(f"  Mean ratio: {lv2['mean_ratio']:.2e}")
    print(f"  Verdict: {lv2['verdict']}")
    print(f"  >> {lv2['summary']}")

    # Level 3 (with LIBERO image)
    print(f"\n{'='*60}")
    print(f"  Level 3: Paired-Seed Output Sensitivity (LIBERO image)")
    print(f"{'='*60}")
    print(f"  Image: LIBERO {task_suite_name} task {task_id}")
    print(f"  Instruction: \"{task_description}\"")
    print()

    lv3 = diagnose_output_sensitivity(
        vla_model, pil_image, task_description,
        unnorm_key=unnorm_key, cfg_scale=cfg_scale,
    )
    results["level3"] = lv3

    print()
    print("  Per-seed paired deltas:")
    for r in lv3["per_seed"]:
        print(
            f"    seed={r['seed']:5d}:  "
            f"A→B={r['delta_ab']:.6f}  "
            f"A→C={r['delta_ac']:.6f}  "
            f"B→C={r['delta_bc']:.6f}"
        )

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
