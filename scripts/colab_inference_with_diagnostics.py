"""
Full inference pipeline with per_attn diagnostics and LLaMA-based confidence.

Two main capabilities:
  1. diagnose_per_attn()  — check if per_attn weights in DiT are dead
  2. run_inference()       — predict_action with LLaMA token confidence

Usage:
    from scripts.colab_inference_with_diagnostics import (
        InferenceConfig, diagnose_per_attn, run_inference,
        run_full_diagnostics_and_inference,
    )

    cfg = InferenceConfig(ckpt_path="path/to/ckpt.pt")
    result = run_full_diagnostics_and_inference(image, instruction, cfg)
    print(result["per_attn_report"])
    print(result["llm_confidence"])
    print(result["actions"])
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from PIL import Image

from vla.load import load_vla


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class InferenceConfig:
    """Configuration for inference with diagnostics."""

    ckpt_path: str = ""
    unnorm_key: str = "libero_object_no_noops"

    # -- diffusion sampling --
    cfg_scale: float = 1.5
    use_ddim: bool = False
    num_ddim_steps: int = 10

    # -- LLaMA confidence --
    confidence_type: str = "max_prob"
    """One of {'max_prob', 'max_logit', 'token_prob'}.
       max_prob  : softmax(logits).max()    — range [0,1], most intuitive
       max_logit : logits.max()             — unbounded, for relative comparison
       token_prob: prob of generated token  — range [0,1]
    """

    # -- per-token prior (Latent L) --
    per_token_prior: Optional[torch.Tensor] = None
    per_token_prior_strength: float = 0.1
    per_token_prior_mode: str = "add"
    per_token_cond_scale: float = 0.0


# ---------------------------------------------------------------------------
# per_attn Diagnostics
# ---------------------------------------------------------------------------
def diagnose_per_attn(vla_model) -> dict:
    """
    Check whether per_attn weights in each DiT block are alive or dead.

    Dead per_attn means per_token (visual spatial info + Latent L)
    is NOT reaching the DiT action generation process.

    Returns:
        dict with keys:
          - blocks: list of per-block weight statistics
          - is_dead: bool, True if all per_attn are near-zero
          - summary: human-readable diagnosis string
    """
    dit = vla_model.action_model.net
    report = {"blocks": [], "is_dead": True, "summary": ""}

    has_per_attn = False
    for i, block in enumerate(dit.blocks):
        if not block.use_per_attn:
            continue

        has_per_attn = True
        w_in = block.per_attn.in_proj_weight.data
        w_out = block.per_attn.out_proj.weight.data

        in_mean = w_in.abs().mean().item()
        in_max = w_in.abs().max().item()
        out_mean = w_out.abs().mean().item()
        out_max = w_out.abs().max().item()

        block_info = {
            "block_idx": i,
            "in_proj_mean_abs": in_mean,
            "in_proj_max_abs": in_max,
            "out_proj_mean_abs": out_mean,
            "out_proj_max_abs": out_max,
        }
        report["blocks"].append(block_info)

        # Threshold: if mean abs weight > 0.01, consider alive
        if in_mean > 0.01 or out_mean > 0.01:
            report["is_dead"] = False

    if not has_per_attn:
        report["is_dead"] = True
        report["summary"] = "No per_attn found in DiT blocks (use_per_attn=False)."
    elif report["is_dead"]:
        report["summary"] = (
            "per_attn is DEAD (weights near zero). "
            "per_token (visual spatial info + Latent L) is NOT reaching DiT. "
            "Consider: (1) enable per_token_cond_scale > 0 as bypass, or "
            "(2) use SpatialGatingControl to override per_attn."
        )
    else:
        report["summary"] = (
            "per_attn is ALIVE. per_token information is reaching DiT."
        )

    return report


# ---------------------------------------------------------------------------
# Inference with LLaMA-based confidence
# ---------------------------------------------------------------------------
@torch.inference_mode()
def run_inference(
    vla_model,
    image: Image.Image,
    instruction: str,
    cfg: InferenceConfig,
    episode_first_frame: str = "True",
) -> dict:
    """
    Run predict_action with LLaMA-based confidence.

    Returns:
        dict with keys:
          - actions: np.ndarray [T, 7] unnormalized action
          - normalized_actions: np.ndarray [T, 7] normalized action [-1, 1]
          - llm_confidence: float (scalar confidence value)
          - confidence_type: str (which type was used)
    """
    actions, normalized_actions, confidence = vla_model.predict_action(
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
        confidence_type=cfg.confidence_type,
    )

    return {
        "actions": actions,
        "normalized_actions": normalized_actions,
        "llm_confidence": float(confidence.mean()),
        "confidence_type": cfg.confidence_type,
    }


# ---------------------------------------------------------------------------
# End-to-end: diagnostics + inference
# ---------------------------------------------------------------------------
def run_full_diagnostics_and_inference(
    image: Image.Image,
    instruction: str,
    cfg: InferenceConfig,
) -> dict:
    """
    Full pipeline:
      1. Load model
      2. Diagnose per_attn (alive or dead?)
      3. Run inference with LLaMA confidence

    Returns:
        dict with per_attn_report, actions, llm_confidence, model, etc.
    """
    # Load model
    vla_model = load_vla(cfg.ckpt_path, load_for_training=False)
    vla_model = vla_model.to("cuda").eval()

    # --- Step 1: per_attn diagnostics ---
    per_attn_report = diagnose_per_attn(vla_model)

    print("=" * 60)
    print("  per_attn Diagnostics")
    print("=" * 60)
    status = "DEAD" if per_attn_report["is_dead"] else "ALIVE"
    print(f"  Status: {status}")
    for b in per_attn_report["blocks"]:
        print(
            f"  Block {b['block_idx']:2d}: "
            f"in_proj mean={b['in_proj_mean_abs']:.6f} max={b['in_proj_max_abs']:.6f}  "
            f"out_proj mean={b['out_proj_mean_abs']:.6f} max={b['out_proj_max_abs']:.6f}"
        )
    print(f"  >> {per_attn_report['summary']}")
    print()

    # --- Step 2: inference with LLaMA confidence ---
    result = run_inference(vla_model, image, instruction, cfg)

    print("=" * 60)
    print("  Inference Result (LLaMA Confidence)")
    print("=" * 60)
    print(f"  Action shape     : {result['actions'].shape}")
    print(f"  First step action: {np.array2string(result['actions'][0], precision=4)}")
    print(f"  Confidence type  : {result['confidence_type']}")
    print(f"  LLM Confidence   : {result['llm_confidence']:.6f}")
    print()

    # Interpretation guide
    if result["confidence_type"] == "max_prob":
        c = result["llm_confidence"]
        if c > 0.9:
            interp = "Very high — LLM is very confident about understanding"
        elif c > 0.7:
            interp = "Moderate — LLM has reasonable understanding"
        elif c > 0.5:
            interp = "Low — LLM is uncertain"
        else:
            interp = "Very low — LLM is highly uncertain"
        print(f"  Interpretation   : {interp}")
        print(
            f"  Note: This measures LLaMA's language token confidence, "
            f"NOT direct action quality."
        )
    print()

    result["per_attn_report"] = per_attn_report
    result["model"] = vla_model
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Inference with Diagnostics for MemoryVLA")
    print("Usage: import and call run_full_diagnostics_and_inference()")
    print()
    print("Example:")
    print("  from scripts.colab_inference_with_diagnostics import (")
    print("      InferenceConfig, run_full_diagnostics_and_inference")
    print("  )")
    print('  cfg = InferenceConfig(ckpt_path="path/to/ckpt.pt")')
    print("  result = run_full_diagnostics_and_inference(image, instruction, cfg)")
