"""
Full inference pipeline with 3-level per_attn diagnostics and LLaMA-based confidence.

Diagnostics (3 levels):
  Level 1: diagnose_per_attn_weights()   — weight magnitude check (fast, no image)
  Level 2: diagnose_per_attn_activation() — actual ||x_c|| / ||x|| ratio during forward
  Level 3: diagnose_output_sensitivity()  — per_token on/off → action/confidence change

Inference:
  run_inference()                          — predict_action with LLaMA token confidence

Usage:
    from scripts.colab_inference_with_diagnostics import (
        InferenceConfig,
        diagnose_per_attn_weights,
        diagnose_per_attn_activation,
        diagnose_output_sensitivity,
        run_full_3level_diagnostics,
    )

    result = run_full_3level_diagnostics(vla, image, instruction, unnorm_key)
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional

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
    unnorm_key: str = "libero_spatial_no_noops"

    # -- diffusion sampling --
    cfg_scale: float = 1.5
    use_ddim: bool = False
    num_ddim_steps: int = 10

    # -- LLaMA confidence --
    confidence_type: str = "max_prob"
    """One of {'max_prob', 'max_logit', 'token_prob'}."""

    # -- per-token prior (Latent L) --
    per_token_prior: Optional[torch.Tensor] = None
    per_token_prior_strength: float = 0.1
    per_token_prior_mode: str = "add"
    per_token_cond_scale: float = 0.0


# ===================================================================
# Level 1: Weight-level diagnosis (no image needed)
# ===================================================================
def diagnose_per_attn_weights(vla_model) -> dict:
    """
    Level 1: Check per_attn weight magnitudes.

    in_proj is the bottleneck — projects inputs to Q/K/V.
    If in_proj ≈ 0, cross-attention output ≈ 0 regardless of out_proj.

    Verdict criteria:
      in_proj_mean < 1e-4 AND in_proj_max < 1e-3 → HIGH RISK dead
      in_proj_mean < 0.005                        → LIKELY dead
      in_proj_mean >= 0.005                       → possibly alive
    """
    dit = vla_model.action_model.net
    report = {"blocks": [], "verdict": "UNKNOWN", "level": 1}

    has_per_attn = False
    high_risk_count = 0
    likely_dead_count = 0
    total_blocks = 0

    for i, block in enumerate(dit.blocks):
        if not block.use_per_attn:
            continue

        has_per_attn = True
        total_blocks += 1
        w_in = block.per_attn.in_proj_weight.data
        w_out = block.per_attn.out_proj.weight.data

        in_mean = w_in.abs().mean().item()
        in_max = w_in.abs().max().item()
        out_mean = w_out.abs().mean().item()
        out_max = w_out.abs().max().item()

        # Per-block risk assessment
        if in_mean < 1e-4 and in_max < 1e-3:
            block_risk = "HIGH_RISK"
            high_risk_count += 1
        elif in_mean < 0.005:
            block_risk = "LIKELY_DEAD"
            likely_dead_count += 1
        else:
            block_risk = "OK"

        report["blocks"].append({
            "block_idx": i,
            "in_proj_mean_abs": in_mean,
            "in_proj_max_abs": in_max,
            "out_proj_mean_abs": out_mean,
            "out_proj_max_abs": out_max,
            "risk": block_risk,
        })

    if not has_per_attn:
        report["verdict"] = "NO_PER_ATTN"
        report["summary"] = "No per_attn in DiT blocks."
    elif high_risk_count == total_blocks:
        report["verdict"] = "HIGH_RISK_DEAD"
        report["summary"] = (
            f"All {total_blocks} blocks: in_proj < 1e-4. "
            "HIGH RISK dead. Need Level 2/3 to confirm."
        )
    elif likely_dead_count + high_risk_count == total_blocks:
        report["verdict"] = "LIKELY_DEAD"
        report["summary"] = (
            f"All {total_blocks} blocks: in_proj < 0.005. "
            "LIKELY dead. Need Level 2/3 to confirm."
        )
    else:
        report["verdict"] = "POSSIBLY_ALIVE"
        alive = total_blocks - likely_dead_count - high_risk_count
        report["summary"] = (
            f"{alive}/{total_blocks} blocks have in_proj >= 0.005. "
            "Possibly alive. Level 2/3 will confirm."
        )

    return report


# ===================================================================
# Level 2: Activation-level diagnosis (needs model + dummy input)
# ===================================================================
@torch.inference_mode()
def diagnose_per_attn_activation(vla_model) -> dict:
    """
    Level 2: Measure actual per_attn contribution during a forward pass.

    For each DiT block, computes:
      ratio = ||x_c|| / ||x||
    where x_c is the per_attn output and x is the main pathway signal.

    If ratio ≈ 0 → per_attn is adding nothing → functionally dead.
    """
    dit = vla_model.action_model.net
    device = next(dit.parameters()).device
    dtype = next(dit.parameters()).dtype

    B = 1
    T = dit.future_action_window_size + 1

    # Create dummy inputs with correct dimensions
    noise_action = torch.randn(B, T, dit.in_channels, device=device, dtype=dtype)
    t = torch.tensor([50], device=device)

    # z_embedder expects (B, 1, token_size) where token_size = LLaMA dim (4096)
    cog_dim = dit.z_embedder.linear.in_features  # token_size, NOT hidden_size
    cog = torch.randn(B, 1, cog_dim, device=device, dtype=dtype)

    # per_token_embedder expects (B, seq_len, per_token_size)
    if dit.use_per_attn:
        per_token_dim = dit.per_token_embedder.linear.in_features
        per_token = torch.randn(B, 256, per_token_dim, device=device, dtype=dtype)
    else:
        per_token = None

    # Hook into each DiTBlock to capture ||x_c|| and ||x||
    activation_ratios = []
    hooks = []

    def _make_hook(block_idx):
        def hook_fn(module, args, output):
            """Capture x_c norm by running per_attn manually."""
            x_input = args[0]  # x entering this block
            per_tok = args[1]  # per_token

            if not module.use_per_attn:
                return

            with torch.no_grad():
                # After self-attn
                x_after_sa = x_input + module.attn(module.norm1(x_input))
                # per_attn output
                x_c, _ = module.per_attn(
                    module.norm3(x_after_sa), per_tok, per_tok
                )

                x_norm = x_after_sa.norm().item()
                xc_norm = x_c.norm().item()
                ratio = xc_norm / (x_norm + 1e-10)

                activation_ratios.append({
                    "block_idx": block_idx,
                    "x_norm": x_norm,
                    "x_c_norm": xc_norm,
                    "ratio": ratio,
                })
        return hook_fn

    for i, block in enumerate(dit.blocks):
        if block.use_per_attn:
            h = block.register_forward_hook(_make_hook(i))
            hooks.append(h)

    # Run DiT forward
    x = dit.x_embedder(noise_action)
    t_emb = dit.t_embedder(t)
    z = dit.z_embedder(cog, False)
    per_tok_emb = dit.per_token_embedder(per_token) if dit.use_per_attn else None

    c = t_emb.unsqueeze(1) + z
    x = torch.cat((c, x), dim=1)
    x = x + dit.positional_embedding

    for block in dit.blocks:
        x = block(x, per_tok_emb)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Summarize
    if not activation_ratios:
        return {"ratios": [], "mean_ratio": 0.0, "verdict": "NO_PER_ATTN", "level": 2}

    mean_ratio = np.mean([r["ratio"] for r in activation_ratios])

    if mean_ratio < 1e-4:
        verdict = "DEAD"
        summary = f"Mean ||x_c||/||x|| = {mean_ratio:.2e}. per_attn output is negligible."
    elif mean_ratio < 0.01:
        verdict = "WEAK"
        summary = f"Mean ||x_c||/||x|| = {mean_ratio:.2e}. per_attn has minimal effect."
    else:
        verdict = "ACTIVE"
        summary = f"Mean ||x_c||/||x|| = {mean_ratio:.4f}. per_attn is contributing."

    return {
        "ratios": activation_ratios,
        "mean_ratio": mean_ratio,
        "verdict": verdict,
        "summary": summary,
        "level": 2,
    }


# ===================================================================
# Level 3: Output sensitivity (needs image — most definitive)
# ===================================================================
@torch.inference_mode()
def diagnose_output_sensitivity(
    vla_model,
    image: Image.Image,
    instruction: str,
    unnorm_key: str = "libero_spatial_no_noops",
    cfg_scale: float = 1.5,
    n_repeats: int = 3,
) -> dict:
    """
    Level 3: End-to-end test. Change per_token, observe action change.

    Runs predict_action in three conditions:
      A. Baseline (no prior, cond_scale=0.0)
      B. Random prior injected (strength=0.3, cond_scale=0.0)
      C. Random prior + bypass ON (strength=0.3, cond_scale=0.2)

    If A≈B → per_attn path is dead (prior has no effect)
    If B≈C → bypass adds nothing (per_attn already works)
    If A≈B but A≠C → per_attn dead, but bypass works → use bypass
    """
    device = next(vla_model.parameters()).device
    per_dim = vla_model.per_token_size

    # Generate a fixed random prior
    random_prior = torch.randn(1, per_dim, device=device) * 0.5

    def _run(prior, strength, cond_scale, label):
        all_actions = []
        all_confs = []
        for _ in range(n_repeats):
            acts, _, conf = vla_model.predict_action(
                image=image,
                instruction=instruction,
                unnorm_key=unnorm_key,
                cfg_scale=cfg_scale,
                episode_first_frame="True",
                per_token_prior=prior,
                per_token_prior_strength=strength,
                per_token_prior_mode="add",
                per_token_cond_scale=cond_scale,
                return_confidence=True,
                confidence_type="max_prob",
            )
            all_actions.append(acts)
            all_confs.append(float(conf.mean()))
        mean_act = np.stack(all_actions).mean(axis=0)
        mean_conf = np.mean(all_confs)
        return {"label": label, "mean_action": mean_act, "mean_conf": mean_conf}

    # Condition A: baseline
    cond_a = _run(None, 0.0, 0.0, "A: Baseline (no prior)")
    # Condition B: prior injected, bypass OFF
    cond_b = _run(random_prior, 0.3, 0.0, "B: Prior ON, bypass OFF")
    # Condition C: prior injected, bypass ON
    cond_c = _run(random_prior, 0.3, 0.2, "C: Prior ON, bypass ON")

    # Compute deltas
    delta_ab = np.abs(cond_a["mean_action"] - cond_b["mean_action"]).mean()
    delta_ac = np.abs(cond_a["mean_action"] - cond_c["mean_action"]).mean()
    delta_bc = np.abs(cond_b["mean_action"] - cond_c["mean_action"]).mean()

    conf_delta_ab = abs(cond_a["mean_conf"] - cond_b["mean_conf"])
    conf_delta_ac = abs(cond_a["mean_conf"] - cond_c["mean_conf"])

    # Interpretation
    per_attn_effective = delta_ab > 0.005
    bypass_effective = delta_ac > 0.005
    bypass_adds_over_per_attn = delta_bc > 0.005

    if per_attn_effective:
        verdict = "PER_ATTN_ALIVE"
        summary = (
            f"per_attn IS working. Prior changes action by {delta_ab:.6f}. "
            f"No bypass needed."
        )
    elif bypass_effective:
        verdict = "PER_ATTN_DEAD_BYPASS_WORKS"
        summary = (
            f"per_attn is dead (A→B delta={delta_ab:.6f}). "
            f"BUT bypass works (A→C delta={delta_ac:.6f}). "
            f"Use per_token_cond_scale > 0."
        )
    else:
        verdict = "BOTH_DEAD"
        summary = (
            f"per_attn dead (delta={delta_ab:.6f}), "
            f"bypass also ineffective (delta={delta_ac:.6f}). "
            f"per_token has no path to DiT. Try SpatialGatingControl."
        )

    return {
        "conditions": [cond_a, cond_b, cond_c],
        "delta_action_AB": delta_ab,
        "delta_action_AC": delta_ac,
        "delta_action_BC": delta_bc,
        "delta_conf_AB": conf_delta_ab,
        "delta_conf_AC": conf_delta_ac,
        "per_attn_effective": per_attn_effective,
        "bypass_effective": bypass_effective,
        "verdict": verdict,
        "summary": summary,
        "level": 3,
    }


# ===================================================================
# Combined: all 3 levels
# ===================================================================
@torch.inference_mode()
def run_full_3level_diagnostics(
    vla_model,
    image: Image.Image,
    instruction: str,
    unnorm_key: str = "libero_spatial_no_noops",
    cfg_scale: float = 1.5,
) -> dict:
    """
    Run all 3 diagnostic levels and print comprehensive report.
    """
    results = {}

    # ── Level 1: Weight check ──
    print("=" * 60)
    print("  Level 1: Weight Magnitude")
    print("=" * 60)
    lv1 = diagnose_per_attn_weights(vla_model)
    results["level1"] = lv1

    for b in lv1["blocks"][:3]:  # Show first 3 blocks
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
    print()

    # ── Level 2: Activation ratio ──
    print("=" * 60)
    print("  Level 2: Activation Ratio ||x_c|| / ||x||")
    print("=" * 60)
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
    print()

    # ── Level 3: Output sensitivity ──
    print("=" * 60)
    print("  Level 3: Output Sensitivity (per_token on/off)")
    print("=" * 60)
    print("  Running 3 conditions × 3 repeats ...")
    lv3 = diagnose_output_sensitivity(
        vla_model, image, instruction,
        unnorm_key=unnorm_key, cfg_scale=cfg_scale,
    )
    results["level3"] = lv3

    for cond in lv3["conditions"]:
        print(f"  {cond['label']}")
        print(f"    action[0] = {np.array2string(cond['mean_action'][0], precision=4)}")
        print(f"    LLM conf  = {cond['mean_conf']:.6f}")

    print()
    print(f"  Action delta A→B (per_attn path) : {lv3['delta_action_AB']:.6f}")
    print(f"  Action delta A→C (bypass path)   : {lv3['delta_action_AC']:.6f}")
    print(f"  Action delta B→C (bypass adds)   : {lv3['delta_action_BC']:.6f}")
    print(f"  Conf   delta A→B                 : {lv3['delta_conf_AB']:.6f}")
    print(f"  Conf   delta A→C                 : {lv3['delta_conf_AC']:.6f}")
    print()
    print(f"  per_attn effective? {lv3['per_attn_effective']}")
    print(f"  bypass effective?   {lv3['bypass_effective']}")
    print(f"  Verdict: {lv3['verdict']}")
    print(f"  >> {lv3['summary']}")
    print()

    # ── Final Summary ──
    print("=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"  Level 1 (weights):     {lv1['verdict']}")
    print(f"  Level 2 (activations): {lv2['verdict']}")
    print(f"  Level 3 (sensitivity): {lv3['verdict']}")
    print()

    if lv3["verdict"] == "PER_ATTN_ALIVE":
        print("  Conclusion: per_attn works. Latent L can reach DiT normally.")
    elif lv3["verdict"] == "PER_ATTN_DEAD_BYPASS_WORKS":
        print("  Conclusion: per_attn is dead BUT bypass works.")
        print("  Action: set per_token_cond_scale > 0 (e.g., 0.1~0.2)")
    else:
        print("  Conclusion: both paths inactive.")
        print("  Action: use SpatialGatingControl to override per_attn.")
    print()

    results["recommendation"] = lv3["verdict"]
    return results


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
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("3-Level per_attn Diagnostics for MemoryVLA")
    print()
    print("Quick usage (on Colab after model loaded):")
    print("  from scripts.colab_inference_with_diagnostics import (")
    print("      run_full_3level_diagnostics")
    print("  )")
    print('  result = run_full_3level_diagnostics(vla, image, instruction, "libero_spatial_no_noops")')
