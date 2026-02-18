"""
cog_attn (z pathway) 진단 스크립트.

per_attn이 dead인 것을 확인한 후, L을 z(cog_tokens) 경로에 넣으려면
이 경로가 실제로 반응하는지 먼저 확인해야 한다.

DiT 아키텍처에서 cog_tokens는:
  z_embedder(z) → c = t + z → cat(c, action_tokens) → self-attention
self-attention(cog_attn)은 Xavier init이라 살아있을 것으로 예상하지만,
실제 checkpoint에서 확인이 필요하다.

3-Level 진단:
  Level 1: self-attention weight magnitude (Xavier init 확인)
  Level 2: z 경로 activation ratio (condition token의 기여도)
  Level 3: paired-seed z perturbation test (가장 결정적)

Usage (Colab):
    from scripts.diagnose_cog_attn import run_cog_attn_diagnostics
    result = run_cog_attn_diagnostics(vla, image, instruction, "libero_spatial_no_noops")
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Optional


# ===================================================================
# Level 1: Self-attention weight magnitude
# ===================================================================
def diagnose_self_attn_weights(vla_model) -> dict:
    """
    Level 1: Check self-attention (cog_attn) weight magnitudes.

    Self-attention uses timm's Attention module with a fused qkv projection.
    If weights are near-zero (like per_attn), the z pathway would be dead too.
    Xavier-initialized weights should have mean_abs ≈ 0.01~0.1.
    """
    dit = vla_model.action_model.net
    report = {"blocks": [], "verdict": "UNKNOWN", "level": 1}

    for i, block in enumerate(dit.blocks):
        # timm Attention stores Q/K/V as a fused linear: block.attn.qkv
        qkv_w = block.attn.qkv.weight.data  # [3*hidden_size, hidden_size]
        proj_w = block.attn.proj.weight.data  # [hidden_size, hidden_size]

        qkv_mean = qkv_w.abs().mean().item()
        qkv_max = qkv_w.abs().max().item()
        proj_mean = proj_w.abs().mean().item()
        proj_max = proj_w.abs().max().item()

        report["blocks"].append({
            "block_idx": i,
            "qkv_mean_abs": qkv_mean,
            "qkv_max_abs": qkv_max,
            "proj_mean_abs": proj_mean,
            "proj_max_abs": proj_max,
        })

    # Also check z_embedder
    z_emb_w = dit.z_embedder.linear.weight.data
    report["z_embedder"] = {
        "mean_abs": z_emb_w.abs().mean().item(),
        "max_abs": z_emb_w.abs().max().item(),
    }

    # Compare with per_attn for reference
    per_attn_means = []
    for block in dit.blocks:
        if block.use_per_attn:
            per_attn_means.append(
                block.per_attn.in_proj_weight.data.abs().mean().item()
            )
    report["per_attn_mean_ref"] = np.mean(per_attn_means) if per_attn_means else None

    # Verdict
    sa_means = [b["qkv_mean_abs"] for b in report["blocks"]]
    avg_sa = np.mean(sa_means)

    if avg_sa < 1e-4:
        report["verdict"] = "DEAD"
        report["summary"] = (
            f"Self-attention qkv mean_abs={avg_sa:.2e}. "
            "Near-zero like per_attn — z pathway likely dead."
        )
    elif avg_sa < 0.005:
        report["verdict"] = "WEAK"
        report["summary"] = (
            f"Self-attention qkv mean_abs={avg_sa:.2e}. "
            "Suspiciously low — needs Level 2/3 confirmation."
        )
    else:
        report["verdict"] = "ALIVE"
        report["summary"] = (
            f"Self-attention qkv mean_abs={avg_sa:.2e}. "
            "Healthy magnitude (Xavier range). z pathway should be active."
        )

    return report


# ===================================================================
# Level 2: z pathway activation ratio
# ===================================================================
@torch.inference_mode()
def diagnose_z_activation(vla_model) -> dict:
    """
    Level 2: Measure how much the condition token (z) contributes
    to the action tokens through self-attention.

    For each DiT block, we compare:
    - Full forward (condition + action tokens together)
    - Condition token zeroed out
    The difference shows how much z contributes.
    """
    dit = vla_model.action_model.net
    device = next(dit.parameters()).device
    dtype = next(dit.parameters()).dtype

    B = 1
    T = dit.future_action_window_size + 1

    # Create realistic-scale inputs
    noise_action = torch.randn(B, T, dit.in_channels, device=device, dtype=dtype)
    t = torch.tensor([50], device=device)

    cog_dim = dit.z_embedder.linear.in_features
    cog = torch.randn(B, 1, cog_dim, device=device, dtype=dtype)
    cog_zero = torch.zeros_like(cog)

    per_token_dim = dit.per_token_embedder.linear.in_features if dit.use_per_attn else None
    per_token = torch.randn(B, 256, per_token_dim, device=device, dtype=dtype) if per_token_dim else None

    # Run with real cog_tokens
    out_real = dit(noise_action.clone(), t, cog, per_token=per_token)

    # Run with zeroed cog_tokens
    out_zero = dit(noise_action.clone(), t, cog_zero, per_token=per_token)

    # Compute difference
    delta = (out_real - out_zero).abs()
    delta_mean = delta.mean().item()
    delta_max = delta.max().item()
    out_norm = out_real.abs().mean().item()
    ratio = delta_mean / (out_norm + 1e-10)

    if ratio < 1e-4:
        verdict = "DEAD"
        summary = f"z contribution ratio={ratio:.2e}. z pathway has no effect on output."
    elif ratio < 0.01:
        verdict = "WEAK"
        summary = f"z contribution ratio={ratio:.2e}. z has minimal effect."
    else:
        verdict = "ACTIVE"
        summary = f"z contribution ratio={ratio:.4f}. z pathway is actively contributing."

    return {
        "delta_mean": delta_mean,
        "delta_max": delta_max,
        "output_mean": out_norm,
        "ratio": ratio,
        "verdict": verdict,
        "summary": summary,
        "level": 2,
    }


# ===================================================================
# Level 3: Paired-seed z perturbation test (most definitive)
# ===================================================================
@torch.inference_mode()
def diagnose_z_sensitivity(
    vla_model,
    image: Image.Image,
    instruction: str,
    unnorm_key: str = "libero_spatial_no_noops",
    cfg_scale: float = 1.5,
    seeds: tuple = (42, 1337, 2024),
    perturbation_scale: float = 0.1,
) -> dict:
    """
    Level 3: Paired-seed output sensitivity for z (cog_tokens) pathway.

    For each seed, runs two conditions with IDENTICAL diffusion noise:
      A. Baseline (unmodified cog_tokens)
      B. cog_tokens + small perturbation (z pathway test)

    By hooking into the model's internal state, we inject a perturbation
    to cog_tokens AFTER memory bank processing but BEFORE DiT.

    Uses DDIM 10 steps (eta=0) so noise is fully deterministic given seed.
    """
    device = next(vla_model.parameters()).device
    cog_dim = vla_model.action_model.net.z_embedder.linear.in_features  # 4096

    # Fixed perturbation (seeded for reproducibility)
    gen = torch.Generator(device=device).manual_seed(7777)
    z_perturb = torch.randn(1, 1, cog_dim, device=device, generator=gen) * perturbation_scale

    n_seeds = len(seeds)
    total_calls = n_seeds * 2
    call_count = [0]

    # We need to hook into predict_action to intercept cog_tokens.
    # Strategy: monkey-patch cog_mem_bank.process_batch to add perturbation.
    original_process_batch = vla_model.cog_mem_bank.process_batch
    inject_perturbation = [False]  # mutable flag

    def patched_process_batch(*args, **kwargs):
        result = original_process_batch(*args, **kwargs)
        if inject_perturbation[0]:
            result = result + z_perturb.to(result.device, dtype=result.dtype)
        return result

    def _run_one(seed, with_perturbation, label):
        call_count[0] += 1
        print(f"    [{call_count[0]}/{total_calls}] seed={seed} {label} ...", flush=True)

        inject_perturbation[0] = with_perturbation

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
            return_confidence=True,
            confidence_type="max_prob",
        )
        return acts

    # Monkey-patch
    vla_model.cog_mem_bank.process_batch = patched_process_batch

    try:
        paired_deltas = []
        all_a_actions = []
        per_seed_results = []

        for seed in seeds:
            act_a = _run_one(seed, False, "A: Baseline z")
            act_b = _run_one(seed, True, f"B: z + perturbation(scale={perturbation_scale})")

            d_ab = np.abs(act_a - act_b).mean()
            paired_deltas.append(d_ab)
            all_a_actions.append(act_a)
            per_seed_results.append({
                "seed": seed,
                "act_a_first": act_a[0].tolist(),
                "act_b_first": act_b[0].tolist(),
                "delta": d_ab,
            })

        # Null baseline: A(seed_i) vs A(seed_j) → intrinsic noise variance
        null_deltas = []
        for i in range(n_seeds):
            for j in range(i + 1, n_seeds):
                null_deltas.append(np.abs(all_a_actions[i] - all_a_actions[j]).mean())

        mean_delta = np.mean(paired_deltas)
        mean_null = np.mean(null_deltas) if null_deltas else 0.0
        snr = mean_delta / (mean_null + 1e-10)

        z_responsive = mean_delta > mean_null * 2 and mean_delta > 0.001

        if z_responsive:
            verdict = "Z_PATHWAY_ALIVE"
            summary = (
                f"z pathway IS responsive. "
                f"Paired delta={mean_delta:.6f} vs noise floor={mean_null:.6f} "
                f"(SNR={snr:.1f}x). "
                f"L injected into cog_tokens WILL affect action output."
            )
        else:
            verdict = "Z_PATHWAY_DEAD"
            summary = (
                f"z pathway NOT responsive. "
                f"Paired delta={mean_delta:.6f} vs noise floor={mean_null:.6f} "
                f"(SNR={snr:.1f}x). "
                f"z perturbation has no effect — self-attention may not propagate z."
            )

        return {
            "per_seed": per_seed_results,
            "mean_delta": mean_delta,
            "null_baseline": mean_null,
            "null_deltas": null_deltas,
            "snr": snr,
            "z_responsive": z_responsive,
            "perturbation_scale": perturbation_scale,
            "verdict": verdict,
            "summary": summary,
            "level": 3,
        }

    finally:
        # Always restore
        vla_model.cog_mem_bank.process_batch = original_process_batch
        inject_perturbation[0] = False


# ===================================================================
# Combined: all 3 levels
# ===================================================================
@torch.inference_mode()
def run_cog_attn_diagnostics(
    vla_model,
    image: Image.Image,
    instruction: str,
    unnorm_key: str = "libero_spatial_no_noops",
    cfg_scale: float = 1.5,
    perturbation_scale: float = 0.1,
) -> dict:
    """
    Run all 3 diagnostic levels for z (cog_tokens) pathway and print report.
    """
    results = {}

    # ── Level 1: Self-attention weight magnitudes ──
    print("=" * 60)
    print("  Level 1: Self-Attention Weight Magnitude (cog_attn)")
    print("=" * 60)
    lv1 = diagnose_self_attn_weights(vla_model)
    results["level1"] = lv1

    for b in lv1["blocks"][:3]:
        print(
            f"  Block {b['block_idx']:2d}: "
            f"qkv mean={b['qkv_mean_abs']:.6f} "
            f"max={b['qkv_max_abs']:.6f}  "
            f"proj mean={b['proj_mean_abs']:.6f}"
        )
    if len(lv1["blocks"]) > 3:
        print(f"  ... ({len(lv1['blocks'])} blocks total)")

    print(f"\n  z_embedder: mean={lv1['z_embedder']['mean_abs']:.6f} "
          f"max={lv1['z_embedder']['max_abs']:.6f}")

    if lv1["per_attn_mean_ref"] is not None:
        sa_avg = np.mean([b["qkv_mean_abs"] for b in lv1["blocks"]])
        print(f"\n  Comparison:")
        print(f"    self-attn qkv mean : {sa_avg:.6f}")
        print(f"    per_attn in_proj   : {lv1['per_attn_mean_ref']:.6f}")
        print(f"    ratio (SA/per)     : {sa_avg / (lv1['per_attn_mean_ref'] + 1e-10):.0f}x")

    print(f"\n  Verdict: {lv1['verdict']}")
    print(f"  >> {lv1['summary']}")
    print()

    # ── Level 2: z contribution ratio ──
    print("=" * 60)
    print("  Level 2: z Pathway Activation (cog vs zero)")
    print("=" * 60)
    lv2 = diagnose_z_activation(vla_model)
    results["level2"] = lv2

    print(f"  DiT output with real z   : mean_abs={lv2['output_mean']:.6f}")
    print(f"  Delta (real z vs zero z) : mean={lv2['delta_mean']:.6f} max={lv2['delta_max']:.6f}")
    print(f"  Contribution ratio       : {lv2['ratio']:.4f}")
    print(f"  Verdict: {lv2['verdict']}")
    print(f"  >> {lv2['summary']}")
    print()

    # ── Level 3: Paired-seed z sensitivity ──
    print("=" * 60)
    print("  Level 3: Paired-Seed z Perturbation Test")
    print("=" * 60)
    print(f"  Same seed → same noise → delta = pure z pathway effect")
    print(f"  Perturbation scale: {perturbation_scale}")
    print(f"  DDIM 10 steps, eta=0 (deterministic)")
    print()
    lv3 = diagnose_z_sensitivity(
        vla_model, image, instruction,
        unnorm_key=unnorm_key, cfg_scale=cfg_scale,
        perturbation_scale=perturbation_scale,
    )
    results["level3"] = lv3

    print()
    print("  Per-seed paired deltas (z vs z+perturbation):")
    for r in lv3["per_seed"]:
        print(f"    seed={r['seed']:5d}: delta={r['delta']:.6f}")
        print(f"      A: {[f'{v:.4f}' for v in r['act_a_first']]}")
        print(f"      B: {[f'{v:.4f}' for v in r['act_b_first']]}")

    print()
    print(f"  Null baseline (A vs A, different seeds):")
    for i, nd in enumerate(lv3["null_deltas"]):
        print(f"    pair {i}: {nd:.6f}")
    print(f"    mean noise floor = {lv3['null_baseline']:.6f}")

    print()
    print(f"  Mean paired delta (z signal) : {lv3['mean_delta']:.6f}")
    print(f"  Noise floor (A vs A)         : {lv3['null_baseline']:.6f}")
    print(f"  SNR (signal/noise)           : {lv3['snr']:.1f}x")
    print()
    print(f"  z pathway responsive? {lv3['z_responsive']}  (need SNR > 2x)")
    print(f"  Verdict: {lv3['verdict']}")
    print(f"  >> {lv3['summary']}")
    print()

    # ── Final Summary ──
    print("=" * 60)
    print("  FINAL SUMMARY: z (cog_tokens) Pathway")
    print("=" * 60)
    print(f"  Level 1 (weights):     {lv1['verdict']}")
    print(f"  Level 2 (activations): {lv2['verdict']}")
    print(f"  Level 3 (sensitivity): {lv3['verdict']}")
    print()

    if lv3["verdict"] == "Z_PATHWAY_ALIVE":
        print("  ✓ z pathway is ALIVE and responsive.")
        print("  → L injected into cog_tokens (z = z + αL) will affect actions.")
        print(f"    Signal is {lv3['snr']:.1f}x above noise floor.")
        print()
        print("  Next steps:")
        print("    1. Create L_z = zeros(1, 1, 4096) as learnable latent")
        print("    2. Inject after cog_mem_bank: z = cog_tokens + α * L_z")
        print("    3. Optimize L_z via SPSA or gradient-based methods")
    else:
        print("  ✗ z pathway is NOT responsive.")
        print("  → Even cog_tokens perturbation doesn't change output.")
        print("  → This would be very unexpected (self-attention should be alive)")
        print("  → Check: is CFG scale too high? Is the model loaded correctly?")
    print()

    results["recommendation"] = lv3["verdict"]
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("cog_attn (z pathway) Diagnostics for MemoryVLA")
    print()
    print("Usage (Colab):")
    print("  from scripts.diagnose_cog_attn import run_cog_attn_diagnostics")
    print('  result = run_cog_attn_diagnostics(vla, image, instruction, "libero_spatial_no_noops")')
