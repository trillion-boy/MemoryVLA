"""
SPSA-based latent (L) optimization for MemoryVLA.

Implements the mentor flow:
    for i in range(N):
        1. I, Img + L  -> MVLA -> A, conf
        2. SPSA perturbation: L-h, L+h -> conf comparison
        3. Update L  (Goal: maximize conf)
    -> L* (optimal L)
    I, Img + L* -> MVLA -> A -> Simulation

Key idea: optimize a per-token prior (L) using only the VLM's token
confidence signal — no simulation reward needed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from vla.load import load_vla
from vla.spatial_prior import build_colab_prior


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class SPSAConfig:
    """Hyper-parameters for SPSA latent optimization."""

    # -- model --
    ckpt_path: str = ""
    unnorm_key: str = "libero_object_no_noops"

    # -- latent --
    latent_dim: int = 256
    """Dimension of the per-token prior vector (must match per_token_size)."""

    # -- SPSA --
    num_iters: int = 30
    """Number of SPSA iterations."""
    a: float = 0.1
    """Step-size numerator (ak = a / (k + 1 + A)^alpha)."""
    c: float = 0.05
    """Perturbation scale numerator (ck = c / (k + 1)^gamma)."""
    A: float = 5.0
    """Step-size stability constant (typically ~10% of num_iters)."""
    alpha: float = 0.602
    """Step-size decay exponent (standard SPSA value)."""
    gamma: float = 0.101
    """Perturbation decay exponent (standard SPSA value)."""

    # -- injection --
    prior_strength: float = 0.1
    prior_mode: str = "add"
    per_token_cond_scale: float = 0.0

    # -- confidence --
    confidence_type: str = "max_prob"
    """One of {'max_prob', 'max_logit', 'token_prob'}."""

    # -- misc --
    seed: Optional[int] = 42
    verbose: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _gain_ak(k: int, cfg: SPSAConfig) -> float:
    """SPSA step-size schedule: a_k = a / (k + 1 + A)^alpha."""
    return cfg.a / ((k + 1 + cfg.A) ** cfg.alpha)


def _gain_ck(k: int, cfg: SPSAConfig) -> float:
    """SPSA perturbation scale: c_k = c / (k + 1)^gamma."""
    return cfg.c / ((k + 1) ** cfg.gamma)


@torch.inference_mode()
def _eval_conf(
    vla_model,
    image: Image.Image,
    instruction: str,
    latent: torch.Tensor,
    cfg: SPSAConfig,
) -> float:
    """Run one forward pass and return scalar confidence."""
    _, _, confidence = vla_model.predict_action(
        image=image,
        instruction=instruction,
        unnorm_key=cfg.unnorm_key,
        episode_first_frame="True",
        per_token_prior=latent,
        per_token_prior_strength=cfg.prior_strength,
        per_token_prior_mode=cfg.prior_mode,
        per_token_cond_scale=cfg.per_token_cond_scale,
        return_confidence=True,
        confidence_type=cfg.confidence_type,
    )
    return float(confidence.mean())


# ---------------------------------------------------------------------------
# Core SPSA routine
# ---------------------------------------------------------------------------
def optimize_latent_spsa(
    vla_model,
    image: Image.Image,
    instruction: str,
    cfg: SPSAConfig,
    init_latent: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, List[float]]:
    """
    Maximize VLM confidence w.r.t. a per-token prior latent using SPSA.

    Args:
        vla_model: Loaded MemoryVLA model (eval mode, on GPU).
        image: Single PIL image for the current observation.
        instruction: Language instruction.
        cfg: SPSA hyper-parameter configuration.
        init_latent: Optional starting latent [1, latent_dim].
                     If None, starts from zeros.

    Returns:
        latent_star: Optimized latent tensor [1, latent_dim].
        conf_history: List of confidence values per iteration.
    """
    device = next(vla_model.parameters()).device

    if init_latent is not None:
        latent = init_latent.clone().to(device=device, dtype=torch.float32)
    else:
        latent = torch.zeros((1, cfg.latent_dim), device=device, dtype=torch.float32)

    if cfg.seed is not None:
        rng = torch.Generator(device=device).manual_seed(cfg.seed)
    else:
        rng = None

    conf_history: List[float] = []

    for k in range(cfg.num_iters):
        ak = _gain_ak(k, cfg)
        ck = _gain_ck(k, cfg)

        # Bernoulli +/-1 perturbation vector
        delta = torch.where(
            torch.rand(latent.shape, device=device, generator=rng) < 0.5,
            torch.ones_like(latent),
            -torch.ones_like(latent),
        )

        lp = latent + ck * delta
        lm = latent - ck * delta

        conf_p = _eval_conf(vla_model, image, instruction, lp, cfg)
        conf_m = _eval_conf(vla_model, image, instruction, lm, cfg)

        # SPSA gradient estimate (we MAXIMIZE conf, so we ascend)
        ghat = ((conf_p - conf_m) / (2.0 * ck)) * delta

        latent = latent + ak * ghat

        conf_cur = 0.5 * (conf_p + conf_m)
        conf_history.append(conf_cur)

        if cfg.verbose:
            print(
                f"[SPSA iter {k+1:3d}/{cfg.num_iters}]  "
                f"ak={ak:.5f}  ck={ck:.5f}  "
                f"conf+={conf_p:.6f}  conf-={conf_m:.6f}  "
                f"avg={conf_cur:.6f}"
            )

    return latent.detach(), conf_history


# ---------------------------------------------------------------------------
# End-to-end entry point
# ---------------------------------------------------------------------------
def run_spsa_optimization(
    image: Image.Image,
    instruction: str,
    cfg: SPSAConfig,
    sam_mask: Optional[torch.Tensor] = None,
    depth_map: Optional[torch.Tensor] = None,
) -> dict:
    """
    Full pipeline:
      1. Load model
      2. (Optional) Build initial latent from SAM mask
      3. SPSA optimize latent L*
      4. Final action prediction with L*

    Args:
        image: PIL observation image.
        instruction: Language instruction.
        cfg: SPSA configuration.
        sam_mask: Optional [H,W] SAM mask for warm-starting the latent.
        depth_map: Optional [H,W] depth map for warm-starting.

    Returns:
        Dictionary with optimized latent, actions, confidence history, etc.
    """
    vla_model = load_vla(cfg.ckpt_path, load_for_training=False)

    # Optional: warm-start from a spatial prior
    init_latent = None
    if sam_mask is not None:
        init_latent = build_colab_prior(
            sam_mask=sam_mask,
            depth_map=depth_map,
            num_patches=cfg.latent_dim,
        )  # [1, latent_dim]

    # SPSA optimization
    latent_star, conf_history = optimize_latent_spsa(
        vla_model=vla_model,
        image=image,
        instruction=instruction,
        cfg=cfg,
        init_latent=init_latent,
    )

    # Final action with optimized latent
    actions, normalized_actions, final_conf = vla_model.predict_action(
        image=image,
        instruction=instruction,
        unnorm_key=cfg.unnorm_key,
        episode_first_frame="True",
        per_token_prior=latent_star,
        per_token_prior_strength=cfg.prior_strength,
        per_token_prior_mode=cfg.prior_mode,
        per_token_cond_scale=cfg.per_token_cond_scale,
        return_confidence=True,
        confidence_type=cfg.confidence_type,
    )

    # Store optimized prior in model for subsequent timesteps
    vla_model.set_per_token_prior(
        latent_star, strength=cfg.prior_strength, mode=cfg.prior_mode
    )

    return {
        "model": vla_model,
        "latent_star": latent_star,
        "conf_history": conf_history,
        "final_confidence": float(final_conf.mean()),
        "actions": actions,
        "normalized_actions": normalized_actions,
    }


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("SPSA Latent Optimization for MemoryVLA")
    print("Usage: import and call run_spsa_optimization() or optimize_latent_spsa()")
    print()
    print("Example:")
    print("  from scripts.colab_spsa_latent_opt import SPSAConfig, run_spsa_optimization")
    print('  cfg = SPSAConfig(ckpt_path="path/to/ckpt.pt")')
    print("  result = run_spsa_optimization(image, instruction, cfg)")
    print('  print(result["final_confidence"], result["actions"])')
