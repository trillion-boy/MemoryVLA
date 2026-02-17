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
    seeds: tuple = (42, 1337, 2024),
) -> dict:
    """
    Level 3: Paired-seed output sensitivity test.

    For each seed, runs three conditions with IDENTICAL diffusion noise:
      A. Baseline (no prior, cond_scale=0.0)
      B. Random prior injected (strength=0.3, cond_scale=0.0)
      C. Random prior + bypass ON (strength=0.3, cond_scale=0.2)

    Also measures null baseline: A(seed_i) vs A(seed_j) to quantify
    intrinsic noise variance, then compares signal vs noise floor.

    Uses DDIM 10 steps (eta=0) so noise is fully deterministic given seed.
    """
    device = next(vla_model.parameters()).device
    per_dim = vla_model.per_token_size

    # Generate a fixed random prior (seeded for reproducibility)
    gen = torch.Generator(device=device).manual_seed(9999)
    random_prior = torch.randn(1, per_dim, device=device, generator=gen) * 0.5

    n_seeds = len(seeds)
    total_calls = n_seeds * 3
    call_count = [0]

    def _run_one(seed, prior, strength, cond_scale, label):
        """Run a single predict_action with a fixed seed."""
        call_count[0] += 1
        print(f"    [{call_count[0]}/{total_calls}] seed={seed} {label} ...", flush=True)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        acts, _, conf = vla_model.predict_action(
            image=image,
            instruction=instruction,
            unnorm_key=unnorm_key,
            cfg_scale=cfg_scale,
            use_ddim=True,
            num_ddim_steps=10,
            episode_first_frame="True",
            per_token_prior=prior,
            per_token_prior_strength=strength,
            per_token_prior_mode="add",
            per_token_cond_scale=cond_scale,
            return_confidence=True,
            confidence_type="max_prob",
        )
        return acts

    # Run paired experiments: same seed → A, B, C
    paired_deltas_ab = []
    paired_deltas_ac = []
    paired_deltas_bc = []
    all_a_actions = []
    per_seed_results = []

    for seed in seeds:
        act_a = _run_one(seed, None, 0.0, 0.0, "A: Baseline")
        act_b = _run_one(seed, random_prior, 0.3, 0.0, "B: Prior ON, bypass OFF")
        act_c = _run_one(seed, random_prior, 0.3, 0.2, "C: Prior ON, bypass ON")

        d_ab = np.abs(act_a - act_b).mean()
        d_ac = np.abs(act_a - act_c).mean()
        d_bc = np.abs(act_b - act_c).mean()

        paired_deltas_ab.append(d_ab)
        paired_deltas_ac.append(d_ac)
        paired_deltas_bc.append(d_bc)
        all_a_actions.append(act_a)
        per_seed_results.append({
            "seed": seed, "act_a": act_a, "act_b": act_b, "act_c": act_c,
            "delta_ab": d_ab, "delta_ac": d_ac, "delta_bc": d_bc,
        })

    # Null baseline: A(seed_i) vs A(seed_j) → intrinsic noise variance
    null_deltas = []
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            null_deltas.append(
                np.abs(all_a_actions[i] - all_a_actions[j]).mean()
            )

    mean_delta_ab = np.mean(paired_deltas_ab)
    mean_delta_ac = np.mean(paired_deltas_ac)
    mean_delta_bc = np.mean(paired_deltas_bc)
    mean_null = np.mean(null_deltas) if null_deltas else 0.0

    # Signal-to-noise: paired delta vs null baseline
    snr_ab = mean_delta_ab / (mean_null + 1e-10)
    snr_ac = mean_delta_ac / (mean_null + 1e-10)

    # Interpretation: signal must exceed noise floor by 2x to be meaningful
    per_attn_effective = mean_delta_ab > mean_null * 2 and mean_delta_ab > 0.001
    bypass_effective = mean_delta_ac > mean_null * 2 and mean_delta_ac > 0.001
    bypass_adds_over_per_attn = mean_delta_bc > mean_null * 2 and mean_delta_bc > 0.001

    if per_attn_effective:
        verdict = "PER_ATTN_ALIVE"
        summary = (
            f"per_attn IS working. Paired A→B={mean_delta_ab:.6f} vs "
            f"noise floor={mean_null:.6f} (SNR={snr_ab:.1f}x)."
        )
    elif bypass_effective:
        verdict = "PER_ATTN_DEAD_BYPASS_WORKS"
        summary = (
            f"per_attn dead (A→B={mean_delta_ab:.6f}, noise={mean_null:.6f}, "
            f"SNR={snr_ab:.1f}x). BUT bypass works (A→C={mean_delta_ac:.6f}, "
            f"SNR={snr_ac:.1f}x). Use per_token_cond_scale > 0."
        )
    else:
        verdict = "BOTH_DEAD"
        summary = (
            f"Both paths inactive. A→B={mean_delta_ab:.6f}, A→C={mean_delta_ac:.6f}, "
            f"noise floor={mean_null:.6f}. Signal ≤ noise. "
            f"per_token has no path to DiT."
        )

    return {
        "per_seed": per_seed_results,
        "mean_delta_AB": mean_delta_ab,
        "mean_delta_AC": mean_delta_ac,
        "mean_delta_BC": mean_delta_bc,
        "null_baseline": mean_null,
        "null_deltas": null_deltas,
        "snr_AB": snr_ab,
        "snr_AC": snr_ac,
        "per_attn_effective": per_attn_effective,
        "bypass_effective": bypass_effective,
        "bypass_adds_over_per_attn": bypass_adds_over_per_attn,
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

    # ── Level 3: Paired-seed output sensitivity ──
    print("=" * 60)
    print("  Level 3: Paired-Seed Output Sensitivity")
    print("=" * 60)
    print("  Same seed → same noise → delta = pure per_attn effect")
    print("  DDIM 10 steps, eta=0 (deterministic given seed)")
    print()
    lv3 = diagnose_output_sensitivity(
        vla_model, image, instruction,
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
        print(f"  (Signal {lv3['snr_AB']:.1f}x above noise floor)")
    elif lv3["verdict"] == "PER_ATTN_DEAD_BYPASS_WORKS":
        print("  Conclusion: per_attn is dead BUT bypass works.")
        print("  Action: set per_token_cond_scale > 0 (e.g., 0.1~0.2)")
        print(f"  (Bypass signal {lv3['snr_AC']:.1f}x above noise floor)")
    else:
        print("  Conclusion: both paths inactive. Signal ≤ noise floor.")
        print("  Action: per_token has no effect via any path.")
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
