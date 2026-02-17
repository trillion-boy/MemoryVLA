"""
Diffusion-based action confidence for MemoryVLA.

Measures how confident the DiT action model is about the predicted action,
NOT the LLaMA language confidence. This directly measures action quality.

Two methods implemented:
  1. Trajectory Consistency (recommended)
     — Captures pred_xstart at every denoising step from the SAME inference pass.
     — Zero extra cost: action + confidence come from one predict_action call.
     — Principle: if the model is confident, it predicts the same action even
       from highly noisy inputs (early denoising steps).

  2. Sample Variance
     — Runs predict_action N times with different random seeds.
     — Cost: N × inference time.
     — Principle: if the model is confident, different noise samples converge
       to the same action.

Usage:
    from scripts.colab_diffusion_confidence import (
        DiffusionConfidenceConfig,
        predict_with_trajectory_confidence,
        predict_with_sample_variance,
    )

    cfg = DiffusionConfidenceConfig()
    result = predict_with_trajectory_confidence(vla_model, image, instruction, cfg)
    print(result["action_confidence"])   # 0-1 scalar, higher = more confident
    print(result["per_dim_confidence"])  # [7] per action dimension
    print(result["actions"])             # same actions as predict_action
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class DiffusionConfidenceConfig:
    """Configuration for diffusion-based confidence measurement."""

    unnorm_key: str = "libero_object_no_noops"

    # -- diffusion sampling --
    cfg_scale: float = 1.5
    use_ddim: bool = False
    num_ddim_steps: int = 10

    # -- per-token prior (Latent L) --
    per_token_prior: Optional[torch.Tensor] = None
    per_token_prior_strength: float = 0.1
    per_token_prior_mode: str = "add"
    per_token_cond_scale: float = 0.0

    # -- trajectory consistency params --
    tail_fraction: float = 0.5
    """Fraction of denoising steps to use for variance computation.
    0.5 = use second half of steps. Larger fraction includes noisier
    early predictions, giving stronger discrimination."""

    tau: float = 0.1
    """Temperature for confidence sigmoid: conf = 1 / (1 + var / tau).
    Smaller tau makes confidence more sensitive to variance."""

    # -- sample variance params --
    n_samples: int = 5
    """Number of independent samples for sample variance method."""


# ---------------------------------------------------------------------------
# Method 1: Trajectory Consistency (RECOMMENDED — zero extra cost)
# ---------------------------------------------------------------------------
@contextmanager
def _capture_pred_xstarts(vla_model, use_ddim: bool = False,
                          num_ddim_steps: int = 10):
    """
    Context manager that monkey-patches the diffusion sampling loop to
    capture pred_xstart at every denoising step.

    After exiting, captured["pred_xstarts"] contains a list of tensors,
    one per denoising step, each of shape [B, T_action, action_dim].

    The original methods are restored on exit.
    """
    captured: Dict[str, list] = {"pred_xstarts": []}

    if use_ddim:
        # Ensure DDIM diffusion exists
        if vla_model.action_model.ddim_diffusion is None:
            vla_model.action_model.create_ddim(ddim_step=num_ddim_steps)
        diffusion = vla_model.action_model.ddim_diffusion
        original_fn = diffusion.ddim_sample_loop

        def _patched_ddim_loop(model, shape, noise=None, clip_denoised=True,
                               denoised_fn=None, cond_fn=None,
                               model_kwargs=None, device=None,
                               progress=False, eta=0.0):
            captured["pred_xstarts"] = []
            final = None
            for sample in diffusion.ddim_sample_loop_progressive(
                model, shape, noise=noise, clip_denoised=clip_denoised,
                denoised_fn=denoised_fn, cond_fn=cond_fn,
                model_kwargs=model_kwargs, device=device,
                progress=progress, eta=eta,
            ):
                captured["pred_xstarts"].append(
                    sample["pred_xstart"].detach().clone()
                )
                final = sample
            return final["sample"]

        diffusion.ddim_sample_loop = _patched_ddim_loop
    else:
        diffusion = vla_model.action_model.diffusion
        original_fn = diffusion.p_sample_loop

        def _patched_ddpm_loop(model, shape, noise=None, clip_denoised=True,
                               denoised_fn=None, cond_fn=None,
                               model_kwargs=None, device=None,
                               progress=False):
            captured["pred_xstarts"] = []
            final = None
            for sample in diffusion.p_sample_loop_progressive(
                model, shape, noise=noise, clip_denoised=clip_denoised,
                denoised_fn=denoised_fn, cond_fn=cond_fn,
                model_kwargs=model_kwargs, device=device,
                progress=progress,
            ):
                captured["pred_xstarts"].append(
                    sample["pred_xstart"].detach().clone()
                )
                final = sample
            return final["sample"]

        diffusion.p_sample_loop = _patched_ddpm_loop

    try:
        yield captured
    finally:
        # Restore original method
        if use_ddim:
            diffusion.ddim_sample_loop = original_fn
        else:
            diffusion.p_sample_loop = original_fn


def _compute_trajectory_confidence(
    pred_xstarts: List[torch.Tensor],
    using_cfg: bool,
    tail_fraction: float = 0.5,
    tau: float = 0.1,
) -> dict:
    """
    Compute confidence from the trajectory of pred_xstart predictions.

    Args:
        pred_xstarts: List of [B, T, D] tensors, one per denoising step.
                      B may be 2*actual_B if CFG was used.
        using_cfg: If True, take only first half of batch (conditioned part).
        tail_fraction: Use this fraction of steps from the end.
        tau: Temperature for sigmoid confidence.

    Returns:
        dict with action_confidence (scalar), per_dim_confidence ([D]),
        per_step_variance ([num_steps]), etc.
    """
    # Stack: [num_steps, B, T, D]
    stacked = torch.stack(pred_xstarts, dim=0)

    # If CFG was used, batch is doubled — take conditioned half only
    if using_cfg:
        B = stacked.shape[1] // 2
        stacked = stacked[:, :B, :, :]

    num_steps = stacked.shape[0]

    # Use the tail portion of denoising steps
    # (early steps are pure noise, later steps show convergence)
    tail_start = max(1, int(num_steps * (1.0 - tail_fraction)))
    tail = stacked[tail_start:]  # [K, B, T, D]

    # --- Per-step variance (shows convergence curve) ---
    # Variance at each step relative to the final prediction
    final_pred = stacked[-1:]  # [1, B, T, D]
    per_step_mse = ((stacked - final_pred) ** 2).mean(dim=(1, 2, 3))  # [num_steps]

    # --- Tail variance (main confidence metric) ---
    # How much does pred_xstart vary across the tail steps?
    tail_var = tail.var(dim=0)  # [B, T, D]

    # Scalar confidence: 1 / (1 + mean_variance / tau)
    mean_var = tail_var.mean().item()
    action_confidence = 1.0 / (1.0 + mean_var / tau)

    # Per action-dimension confidence: [D]
    per_dim_var = tail_var.mean(dim=(0, 1))  # [D]
    per_dim_confidence = 1.0 / (1.0 + per_dim_var / tau)
    per_dim_confidence = per_dim_confidence.cpu().numpy()

    # Per timestep confidence: [T_action]
    per_timestep_var = tail_var.mean(dim=(0, 2))  # [T]
    per_timestep_confidence = 1.0 / (1.0 + per_timestep_var / tau)
    per_timestep_confidence = per_timestep_confidence.cpu().numpy()

    return {
        "action_confidence": action_confidence,
        "per_dim_confidence": per_dim_confidence,
        "per_timestep_confidence": per_timestep_confidence,
        "mean_variance": mean_var,
        "per_step_mse": per_step_mse.cpu().numpy(),
        "num_denoising_steps": num_steps,
        "tail_steps_used": len(tail),
    }


@torch.inference_mode()
def predict_with_trajectory_confidence(
    vla_model,
    image: Image.Image,
    instruction: str,
    cfg: DiffusionConfidenceConfig,
    episode_first_frame: str = "True",
) -> dict:
    """
    Run predict_action and simultaneously capture trajectory confidence.

    This is the RECOMMENDED method: zero extra cost, action + confidence
    from the same inference pass.

    Returns:
        dict with:
          - actions: np.ndarray [T, 7]
          - normalized_actions: np.ndarray [T, 7]
          - action_confidence: float in [0, 1], higher = more confident
          - per_dim_confidence: np.ndarray [7], per action dimension
          - per_timestep_confidence: np.ndarray [T], per future timestep
          - mean_variance: float, raw variance (lower = more confident)
          - convergence_curve: np.ndarray [num_steps], MSE vs final pred
    """
    using_cfg = cfg.cfg_scale > 1.0

    with _capture_pred_xstarts(
        vla_model, use_ddim=cfg.use_ddim, num_ddim_steps=cfg.num_ddim_steps
    ) as captured:
        actions, normalized_actions = vla_model.predict_action(
            image=image,
            instruction=instruction,
            unnorm_key=cfg.unnorm_key,
            cfg_scale=cfg.cfg_scale,
            use_ddim=cfg.use_ddim,
            num_ddim_steps=cfg.num_ddim_steps,
            episode_first_frame=episode_first_frame,
            per_token_prior=cfg.per_token_prior,
            per_token_prior_strength=cfg.per_token_prior_strength,
            per_token_prior_mode=cfg.per_token_prior_mode,
            per_token_cond_scale=cfg.per_token_cond_scale,
        )

    # Compute trajectory consistency confidence
    traj = _compute_trajectory_confidence(
        captured["pred_xstarts"],
        using_cfg=using_cfg,
        tail_fraction=cfg.tail_fraction,
        tau=cfg.tau,
    )

    dim_labels = ["dx", "dy", "dz", "rx", "ry", "rz", "grip"]
    print("=" * 60)
    print("  Diffusion Action Confidence (Trajectory Consistency)")
    print("=" * 60)
    print(f"  Action Confidence : {traj['action_confidence']:.4f}  (0=uncertain, 1=confident)")
    print(f"  Raw Variance      : {traj['mean_variance']:.6f}")
    print(f"  Denoising Steps   : {traj['num_denoising_steps']} (tail {traj['tail_steps_used']} used)")
    print(f"  Per-dim confidence:")
    for i, (label, c) in enumerate(zip(dim_labels, traj["per_dim_confidence"])):
        bar = "#" * int(c * 20)
        print(f"    {label:5s}: {c:.4f}  [{bar:<20s}]")
    print()

    return {
        "actions": actions,
        "normalized_actions": normalized_actions,
        "action_confidence": traj["action_confidence"],
        "per_dim_confidence": traj["per_dim_confidence"],
        "per_timestep_confidence": traj["per_timestep_confidence"],
        "mean_variance": traj["mean_variance"],
        "convergence_curve": traj["per_step_mse"],
    }


# ---------------------------------------------------------------------------
# Method 2: Sample Variance (secondary — N× cost)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def predict_with_sample_variance(
    vla_model,
    image: Image.Image,
    instruction: str,
    cfg: DiffusionConfidenceConfig,
    episode_first_frame: str = "True",
) -> dict:
    """
    Run predict_action N times and measure variance across samples.

    More expensive (N × inference time) but intuitive and robust.

    Returns:
        dict with:
          - actions: np.ndarray [T, 7] (mean of N samples)
          - all_actions: list of np.ndarray, each [T, 7]
          - action_confidence: float in [0, 1]
          - per_dim_confidence: np.ndarray [7]
          - per_dim_std: np.ndarray [7], raw standard deviation
    """
    all_actions = []
    all_norm_actions = []

    for i in range(cfg.n_samples):
        actions, norm_actions = vla_model.predict_action(
            image=image,
            instruction=instruction,
            unnorm_key=cfg.unnorm_key,
            cfg_scale=cfg.cfg_scale,
            use_ddim=cfg.use_ddim,
            num_ddim_steps=cfg.num_ddim_steps,
            episode_first_frame=episode_first_frame,
            per_token_prior=cfg.per_token_prior,
            per_token_prior_strength=cfg.per_token_prior_strength,
            per_token_prior_mode=cfg.per_token_prior_mode,
            per_token_cond_scale=cfg.per_token_cond_scale,
        )
        all_actions.append(actions)
        all_norm_actions.append(norm_actions)

    # Stack: [N, T, 7]
    stacked = np.stack(all_actions, axis=0)
    stacked_norm = np.stack(all_norm_actions, axis=0)

    # Mean action (consensus)
    mean_actions = stacked.mean(axis=0)  # [T, 7]
    mean_norm = stacked_norm.mean(axis=0)

    # Per-dim standard deviation across samples
    per_dim_std = stacked.std(axis=0).mean(axis=0)  # [7]

    # Confidence: 1 / (1 + std / tau)
    per_dim_confidence = 1.0 / (1.0 + per_dim_std / cfg.tau)
    action_confidence = float(per_dim_confidence.mean())

    dim_labels = ["dx", "dy", "dz", "rx", "ry", "rz", "grip"]
    print("=" * 60)
    print(f"  Diffusion Action Confidence (Sample Variance, N={cfg.n_samples})")
    print("=" * 60)
    print(f"  Action Confidence : {action_confidence:.4f}  (0=uncertain, 1=confident)")
    print(f"  Per-dim confidence and std:")
    for i, (label, c, s) in enumerate(
        zip(dim_labels, per_dim_confidence, per_dim_std)
    ):
        bar = "#" * int(c * 20)
        print(f"    {label:5s}: conf={c:.4f}  std={s:.6f}  [{bar:<20s}]")
    print()

    return {
        "actions": mean_actions,
        "normalized_actions": mean_norm,
        "all_actions": all_actions,
        "action_confidence": action_confidence,
        "per_dim_confidence": per_dim_confidence,
        "per_dim_std": per_dim_std,
    }


# ---------------------------------------------------------------------------
# Combined: both methods for comparison
# ---------------------------------------------------------------------------
@torch.inference_mode()
def predict_with_both_confidences(
    vla_model,
    image: Image.Image,
    instruction: str,
    cfg: DiffusionConfidenceConfig,
    episode_first_frame: str = "True",
) -> dict:
    """
    Run both trajectory consistency and sample variance, plus LLaMA confidence
    for comparison.

    Returns dict with all three confidence metrics.
    """
    # 1. Trajectory consistency (single pass, captures pred_xstarts)
    traj_result = predict_with_trajectory_confidence(
        vla_model, image, instruction, cfg, episode_first_frame
    )

    # 2. Sample variance (N additional passes)
    sv_result = predict_with_sample_variance(
        vla_model, image, instruction, cfg, episode_first_frame
    )

    # 3. LLaMA confidence (one more pass)
    _, _, llm_conf = vla_model.predict_action(
        image=image,
        instruction=instruction,
        unnorm_key=cfg.unnorm_key,
        cfg_scale=cfg.cfg_scale,
        use_ddim=cfg.use_ddim,
        num_ddim_steps=cfg.num_ddim_steps,
        episode_first_frame=episode_first_frame,
        per_token_prior=cfg.per_token_prior,
        per_token_prior_strength=cfg.per_token_prior_strength,
        per_token_prior_mode=cfg.per_token_prior_mode,
        per_token_cond_scale=cfg.per_token_cond_scale,
        return_confidence=True,
        confidence_type="max_prob",
    )

    print("=" * 60)
    print("  Confidence Comparison (3 Methods)")
    print("=" * 60)
    print(f"  Trajectory Consistency : {traj_result['action_confidence']:.4f}  "
          f"(action-level, 0 extra cost)")
    print(f"  Sample Variance        : {sv_result['action_confidence']:.4f}  "
          f"(action-level, {cfg.n_samples}x cost)")
    print(f"  LLaMA Token Confidence : {float(llm_conf.mean()):.4f}  "
          f"(language-level, not action quality)")
    print()

    return {
        "actions": traj_result["actions"],
        "normalized_actions": traj_result["normalized_actions"],
        "trajectory_confidence": traj_result["action_confidence"],
        "trajectory_per_dim": traj_result["per_dim_confidence"],
        "trajectory_convergence": traj_result["convergence_curve"],
        "sample_variance_confidence": sv_result["action_confidence"],
        "sample_variance_per_dim": sv_result["per_dim_confidence"],
        "sample_variance_std": sv_result["per_dim_std"],
        "llm_confidence": float(llm_conf.mean()),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Diffusion-based Action Confidence for MemoryVLA")
    print()
    print("Recommended: predict_with_trajectory_confidence()")
    print("  — Zero extra cost, action + confidence from same pass")
    print()
    print("Alternative: predict_with_sample_variance()")
    print("  — N × cost, but intuitive and robust")
    print()
    print("Compare all: predict_with_both_confidences()")
    print("  — Trajectory + Sample Variance + LLaMA, side by side")
    print()
    print("Example:")
    print("  from scripts.colab_diffusion_confidence import (")
    print("      DiffusionConfidenceConfig, predict_with_trajectory_confidence")
    print("  )")
    print("  cfg = DiffusionConfidenceConfig()")
    print("  result = predict_with_trajectory_confidence(vla, image, instr, cfg)")
    print('  print(result["action_confidence"])')
