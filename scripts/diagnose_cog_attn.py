"""
cog_attn (z pathway) 진단 스크립트 v2.

v1 대비 개선사항 (비판 반영):
  1. 결정론적 바닥(same-seed repeat)과 cross-seed 분산을 분리
  2. raw (pre-clip/binarize) + final action 이중 출력 공간 측정
  3. perturbation scale sweep (dose-response 곡선)
  4. CFG scale sweep
  5. 내용 민감도 테스트 (z=0 vs z_real vs z_shuffled vs z+δ)

v1 Level 3의 문제점:
  - predict_action 후처리(clip, gripper 이진화)가 delta를 축소
  - cross-seed A/A를 noise floor로 사용하면 과대추정
  - 단일 perturbation scale(0.1)로는 dead vs weak 구분 불가
  - CFG가 z 신호를 증폭하는지 미확인
  - z on/off vs z content 구분 불가

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
# Level 3 Enhanced: z pathway sensitivity (v2)
# ===================================================================

def _ensure_ddim(vla_model, steps=10):
    """Pre-create DDIM so hooks on ddim_sample_loop work reliably."""
    if vla_model.action_model.ddim_diffusion is None:
        vla_model.action_model.create_ddim(ddim_step=steps)


@torch.inference_mode()
def _run_one_v2(vla_model, image, instruction, unnorm_key, cfg_scale, seed,
                cog_patch_fn=None):
    """
    Run predict_action and capture BOTH:
      - raw:   pre-clip/binarize (normalized space, direct diffusion output)
      - final: post-clip/binarize/unnorm (action space)

    cog_patch_fn: optional f(cog_tokens) -> modified cog_tokens.
                  Applied after memory bank, before DiT.
    """
    _ensure_ddim(vla_model)

    # ── Hook 1: cog_tokens patching ──
    original_process = vla_model.cog_mem_bank.process_batch
    if cog_patch_fn is not None:
        def patched_process(*args, **kwargs):
            result = original_process(*args, **kwargs)
            return cog_patch_fn(result)
        vla_model.cog_mem_bank.process_batch = patched_process

    # ── Hook 2: capture raw diffusion output (before clip/binarize) ──
    raw_holder = {}
    ddim = vla_model.action_model.ddim_diffusion
    original_loop = ddim.ddim_sample_loop

    def capture_loop(*args, **kwargs):
        result = original_loop(*args, **kwargs)
        raw_holder['samples'] = result.detach().cpu().clone()
        return result
    ddim.ddim_sample_loop = capture_loop

    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        final_actions, _, _ = vla_model.predict_action(
            image=image, instruction=instruction,
            unnorm_key=unnorm_key, cfg_scale=cfg_scale,
            use_ddim=True, num_ddim_steps=10,
            episode_first_frame="True",
            return_confidence=True, confidence_type="max_prob",
        )

        raw = raw_holder.get('samples')
        if raw is not None:
            if cfg_scale > 1.0:
                raw, _ = raw.chunk(2, dim=0)
            raw_actions = raw[0].numpy()
        else:
            raw_actions = final_actions.copy()

        return {'raw': raw_actions, 'final': final_actions}
    finally:
        vla_model.cog_mem_bank.process_batch = original_process
        ddim.ddim_sample_loop = original_loop


# -------------------------------------------------------------------
#  Sub-test 3a: Deterministic reproducibility floor
# -------------------------------------------------------------------
@torch.inference_mode()
def _test_3a_deterministic_floor(vla_model, image, instruction, unnorm_key,
                                  cfg_scale, seed=42):
    """Same seed, same z, run twice. If DDIM eta=0 is truly deterministic,
    delta should be ~0.  This is the TRUE noise floor (not cross-seed)."""
    r1 = _run_one_v2(vla_model, image, instruction, unnorm_key, cfg_scale, seed)
    r2 = _run_one_v2(vla_model, image, instruction, unnorm_key, cfg_scale, seed)
    return {
        'delta_raw':  float(np.abs(r1['raw']  - r2['raw']).mean()),
        'delta_final': float(np.abs(r1['final'] - r2['final']).mean()),
        'max_raw':     float(np.abs(r1['raw']  - r2['raw']).max()),
    }


# -------------------------------------------------------------------
#  Sub-test 3b: Perturbation scale sweep (dose-response)
# -------------------------------------------------------------------
@torch.inference_mode()
def _test_3b_perturbation_sweep(vla_model, image, instruction, unnorm_key,
                                 cfg_scale, seed=42,
                                 scales=(0.03, 0.1, 0.3, 1.0)):
    """Fixed direction, varying magnitude. Monotonic increase → alive."""
    device = next(vla_model.parameters()).device
    cog_dim = vla_model.action_model.net.z_embedder.linear.in_features

    gen = torch.Generator(device=device).manual_seed(7777)
    direction = torch.randn(1, 1, cog_dim, device=device, generator=gen)
    direction = direction / direction.norm()  # unit vector

    baseline = _run_one_v2(vla_model, image, instruction, unnorm_key, cfg_scale, seed)

    sweep = []
    for scale in scales:
        p = direction * scale
        patch_fn = lambda z, _p=p: z + _p.to(z.device, dtype=z.dtype)
        r = _run_one_v2(vla_model, image, instruction, unnorm_key,
                        cfg_scale, seed, cog_patch_fn=patch_fn)
        sweep.append({
            'scale': scale,
            'delta_raw':  float(np.abs(baseline['raw']  - r['raw']).mean()),
            'delta_final': float(np.abs(baseline['final'] - r['final']).mean()),
            'max_delta_raw': float(np.abs(baseline['raw'] - r['raw']).max()),
        })

    raw_deltas = [s['delta_raw'] for s in sweep]
    is_monotonic = all(raw_deltas[i] <= raw_deltas[i+1] * 1.05
                       for i in range(len(raw_deltas) - 1))

    return {
        'baseline_first': baseline['raw'][0].tolist() if baseline['raw'].ndim > 1
                          else baseline['raw'].tolist(),
        'sweep': sweep,
        'is_monotonic': is_monotonic,
    }


# -------------------------------------------------------------------
#  Sub-test 3c: CFG scale sweep
# -------------------------------------------------------------------
@torch.inference_mode()
def _test_3c_cfg_sweep(vla_model, image, instruction, unnorm_key,
                        seed=42, perturbation_scale=0.1,
                        cfg_scales=(1.0, 1.5, 3.0)):
    """Higher CFG should amplify conditional signal ⇒ amplify z effect."""
    device = next(vla_model.parameters()).device
    cog_dim = vla_model.action_model.net.z_embedder.linear.in_features

    gen = torch.Generator(device=device).manual_seed(7777)
    direction = torch.randn(1, 1, cog_dim, device=device, generator=gen)
    direction = direction / direction.norm()
    p = direction * perturbation_scale
    patch_fn = lambda z, _p=p: z + _p.to(z.device, dtype=z.dtype)

    results = []
    for cfg in cfg_scales:
        b = _run_one_v2(vla_model, image, instruction, unnorm_key, cfg, seed)
        r = _run_one_v2(vla_model, image, instruction, unnorm_key, cfg, seed,
                        cog_patch_fn=patch_fn)
        results.append({
            'cfg': cfg,
            'delta_raw':  float(np.abs(b['raw']  - r['raw']).mean()),
            'delta_final': float(np.abs(b['final'] - r['final']).mean()),
        })

    cfg_amplifies = (len(results) >= 2 and
                     results[-1]['delta_raw'] > results[0]['delta_raw'] * 1.3)

    return {'sweep': results, 'cfg_amplifies': cfg_amplifies}


# -------------------------------------------------------------------
#  Sub-test 3d: Content sensitivity (z=0, z_shuffled, z+delta)
# -------------------------------------------------------------------
@torch.inference_mode()
def _test_3d_content_sensitivity(vla_model, image, instruction, unnorm_key,
                                  cfg_scale, seed=42):
    """
    Compare z_real  vs  z_zero / z_shuffled / z+delta.
    - z_zero large, z_shuffled small  → on/off only (BIAS_ONLY)
    - z_zero large, z_shuffled large  → content matters (ALIVE)
    - all small                       → dead
    """
    device = next(vla_model.parameters()).device
    cog_dim = vla_model.action_model.net.z_embedder.linear.in_features

    # z_real (baseline - no patching)
    r_real = _run_one_v2(vla_model, image, instruction, unnorm_key, cfg_scale, seed)

    # z_zero
    r_zero = _run_one_v2(vla_model, image, instruction, unnorm_key, cfg_scale, seed,
                          cog_patch_fn=lambda z: torch.zeros_like(z))

    # z_shuffled (permute feature dims — same norm, different content)
    perm = torch.randperm(cog_dim, generator=torch.Generator().manual_seed(9999))
    def patch_shuffle(z, _p=perm):
        return z[:, :, _p.to(z.device)].contiguous()
    r_shuffle = _run_one_v2(vla_model, image, instruction, unnorm_key, cfg_scale, seed,
                             cog_patch_fn=patch_shuffle)

    # z + delta (small perturbation, scale=0.1)
    gen = torch.Generator(device=device).manual_seed(7777)
    delta = torch.randn(1, 1, cog_dim, device=device, generator=gen) * 0.1
    r_delta = _run_one_v2(vla_model, image, instruction, unnorm_key, cfg_scale, seed,
                           cog_patch_fn=lambda z, _d=delta: z + _d.to(z.device, dtype=z.dtype))

    def _fmt(r):
        return r['raw'][0].tolist() if r['raw'].ndim > 1 else r['raw'].tolist()

    results = {}
    for name, r in [('z_zero', r_zero), ('z_shuffled', r_shuffle), ('z_perturbed', r_delta)]:
        results[name] = {
            'delta_raw':  float(np.abs(r_real['raw']  - r['raw']).mean()),
            'delta_final': float(np.abs(r_real['final'] - r['final']).mean()),
            'first_raw': _fmt(r),
        }
    results['z_real'] = {'first_raw': _fmt(r_real)}

    d_zero    = results['z_zero']['delta_raw']
    d_shuffle = results['z_shuffled']['delta_raw']
    d_perturb = results['z_perturbed']['delta_raw']

    # Content sensitivity: shuffling destroys content → large delta means content used
    content_sensitive = (d_shuffle > d_perturb * 3) and (d_zero > d_perturb * 3)

    results['analysis'] = {
        'content_sensitive': content_sensitive,
        'd_zero': d_zero, 'd_shuffle': d_shuffle, 'd_perturb': d_perturb,
    }
    return results


# -------------------------------------------------------------------
#  Verdict computation
# -------------------------------------------------------------------
def _compute_verdict(floor, sweep, cfg, content):
    """
    Combine sub-test results into a nuanced verdict.

    Returns (verdict_str, summary_str).
    Possible verdicts: DEAD, BIAS_ONLY, WEAK, ALIVE
    """
    floor_raw = floor['delta_raw']
    sig_threshold = max(floor_raw * 10, 1e-5)

    d_zero    = content['analysis']['d_zero']
    d_shuffle = content['analysis']['d_shuffle']
    max_sweep = max(s['delta_raw'] for s in sweep['sweep'])

    # (1) No response at any scale
    if max_sweep < sig_threshold and d_zero < sig_threshold:
        return "DEAD", "No measurable z response at any perturbation scale or condition."

    # (2) z=0 differs from z_real → z has *some* contribution
    #     but z_shuffled ≈ z_real → content doesn't matter, only on/off
    if d_zero >= sig_threshold and d_shuffle < sig_threshold:
        return "BIAS_ONLY", (
            f"z acts as static bias (on/off). "
            f"d_zero={d_zero:.6f} but d_shuffle={d_shuffle:.6f} (content ignored)."
        )

    # (3) Monotonic dose-response + content sensitive → alive
    if sweep['is_monotonic'] and content['analysis']['content_sensitive']:
        # Distinguish WEAK from ALIVE by absolute magnitude
        mid = sweep['sweep'][1]['delta_raw']  # scale ≈ 0.1
        if mid < 0.005:
            return "WEAK", (
                f"z pathway responsive & content-sensitive, but low gain. "
                f"delta@0.1={mid:.6f}."
            )
        return "ALIVE", (
            f"z pathway is responsive and content-sensitive. "
            f"delta@0.1={mid:.6f}, monotonic dose-response confirmed."
        )

    # (4) Monotonic but content-insensitive → weak
    if sweep['is_monotonic']:
        return "WEAK", (
            f"z pathway responds to perturbation scale (monotonic) "
            f"but content sensitivity is unclear."
        )

    # (5) Fallback
    return "WEAK", "z pathway shows partial response. Further investigation needed."


# ===================================================================
# Level 3 combined runner
# ===================================================================
@torch.inference_mode()
def diagnose_z_sensitivity_v2(
    vla_model,
    image: Image.Image,
    instruction: str,
    unnorm_key: str = "libero_spatial_no_noops",
    cfg_scale: float = 1.5,
    seed: int = 42,
    perturbation_scales: tuple = (0.03, 0.1, 0.3, 1.0),
    cfg_scales: tuple = (1.0, 1.5, 3.0),
    include_cross_seed_ref: bool = True,
) -> dict:
    """
    Level 3 Enhanced: Comprehensive z pathway sensitivity analysis.

    Sub-tests:
      3a. Deterministic floor (same-seed repeat)
      3b. Perturbation scale sweep (dose-response curve)
      3c. CFG scale sweep
      3d. Content sensitivity (z=0, z_shuffled, z+delta)
    """
    results = {"level": 3}

    # ── 3a. Deterministic Floor ──
    print("\n  3a. Deterministic Floor (same-seed repeat)")
    print("  " + "-" * 44)

    floor = _test_3a_deterministic_floor(
        vla_model, image, instruction, unnorm_key, cfg_scale, seed)
    results['3a'] = floor

    is_det = floor['delta_raw'] < 1e-4
    print(f"    raw   : delta_mean={floor['delta_raw']:.2e}  delta_max={floor['max_raw']:.2e}")
    print(f"    final : delta_mean={floor['delta_final']:.2e}")
    print(f"    -> {'Deterministic (floor ~ 0)' if is_det else 'Non-deterministic (floor > 0)'}")

    # ── 3b. Perturbation Scale Sweep ──
    print(f"\n  3b. Perturbation Scale Sweep")
    print("  " + "-" * 44)

    sweep = _test_3b_perturbation_sweep(
        vla_model, image, instruction, unnorm_key, cfg_scale, seed,
        scales=perturbation_scales)
    results['3b'] = sweep

    print(f"\n    {'scale':>6} | {'d_raw':>10} | {'d_final':>10} | {'raw/floor':>10}")
    print(f"    {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for s in sweep['sweep']:
        rf = f"{s['delta_raw']/(floor['delta_raw']+1e-15):.0f}x" if floor['delta_raw'] > 1e-10 else "inf"
        print(f"    {s['scale']:6.2f} | {s['delta_raw']:10.6f} | {s['delta_final']:10.6f} | {rf:>10}")

    compr_ratios = [s['delta_final'] / (s['delta_raw'] + 1e-15) for s in sweep['sweep']
                    if s['delta_raw'] > 1e-8]
    compr = np.mean(compr_ratios) if compr_ratios else 1.0
    print(f"\n    Monotonically increasing: {'YES' if sweep['is_monotonic'] else 'NO'}")
    print(f"    Post-processing keeps {compr:.0%} of raw signal (rest lost to clip/binarize)")

    # ── 3c. CFG Scale Sweep ──
    print(f"\n  3c. CFG Scale Sweep (perturbation=0.1)")
    print("  " + "-" * 44)

    cfg_res = _test_3c_cfg_sweep(
        vla_model, image, instruction, unnorm_key, seed,
        perturbation_scale=0.1, cfg_scales=cfg_scales)
    results['3c'] = cfg_res

    print(f"\n    {'cfg':>6} | {'d_raw':>10} | {'d_final':>10}")
    print(f"    {'-'*6}-+-{'-'*10}-+-{'-'*10}")
    for c in cfg_res['sweep']:
        print(f"    {c['cfg']:6.1f} | {c['delta_raw']:10.6f} | {c['delta_final']:10.6f}")

    if len(cfg_res['sweep']) >= 2:
        amp = cfg_res['sweep'][-1]['delta_raw'] / (cfg_res['sweep'][0]['delta_raw'] + 1e-15)
        print(f"\n    CFG amplification (cfg={cfg_scales[-1]} vs {cfg_scales[0]}): {amp:.1f}x")
        print(f"    Higher CFG amplifies z signal: {'YES' if cfg_res['cfg_amplifies'] else 'NO'}")

    # ── 3d. Content Sensitivity ──
    print(f"\n  3d. Content Sensitivity")
    print("  " + "-" * 44)

    content = _test_3d_content_sensitivity(
        vla_model, image, instruction, unnorm_key, cfg_scale, seed)
    results['3d'] = content

    print(f"\n    {'condition':>14} | {'d_raw':>10} | {'d_final':>10}")
    print(f"    {'-'*14}-+-{'-'*10}-+-{'-'*10}")
    for name in ['z_zero', 'z_shuffled', 'z_perturbed']:
        c = content[name]
        print(f"    {name:>14} | {c['delta_raw']:10.6f} | {c['delta_final']:10.6f}")

    a = content['analysis']
    print(f"\n    z=0 (on/off signal)    : {a['d_zero']:.6f}")
    print(f"    z_shuffled (content)   : {a['d_shuffle']:.6f}")
    print(f"    z+delta (perturbation) : {a['d_perturb']:.6f}")
    print(f"    Content-sensitive: {'YES' if a['content_sensitive'] else 'NO'}")

    # ── Optional: Cross-seed reference (informational only) ──
    if include_cross_seed_ref:
        print(f"\n  Ref: Cross-seed variance (informational, NOT noise floor)")
        print("  " + "-" * 44)
        ref_seeds = (42, 1337, 2024)
        ref_actions = []
        for s in ref_seeds:
            r = _run_one_v2(vla_model, image, instruction, unnorm_key, cfg_scale, s)
            ref_actions.append(r)
        cross_deltas = []
        for i in range(len(ref_seeds)):
            for j in range(i + 1, len(ref_seeds)):
                d = float(np.abs(ref_actions[i]['raw'] - ref_actions[j]['raw']).mean())
                cross_deltas.append(d)
                print(f"    seed {ref_seeds[i]} vs {ref_seeds[j]}: {d:.6f}")
        results['cross_seed_ref'] = {
            'mean': float(np.mean(cross_deltas)),
            'deltas': cross_deltas,
        }
        print(f"    mean = {np.mean(cross_deltas):.6f}")
        print(f"    (This is NOT used for verdict — shown for reference only)")

    # ── Verdict ──
    verdict, summary = _compute_verdict(floor, sweep, cfg_res, content)
    results['verdict'] = verdict
    results['summary'] = summary

    return results


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
    seed: int = 42,
    perturbation_scales: tuple = (0.03, 0.1, 0.3, 1.0),
    cfg_scales: tuple = (1.0, 1.5, 3.0),
) -> dict:
    """Run all 3 diagnostic levels for z (cog_tokens) pathway and print report."""
    results = {}

    # ── Level 1 ──
    print("=" * 60)
    print("  Level 1: Self-Attention Weight Magnitude (cog_attn)")
    print("=" * 60)
    lv1 = diagnose_self_attn_weights(vla_model)
    results["level1"] = lv1

    for b in lv1["blocks"][:3]:
        print(f"  Block {b['block_idx']:2d}: "
              f"qkv mean={b['qkv_mean_abs']:.6f} max={b['qkv_max_abs']:.6f}  "
              f"proj mean={b['proj_mean_abs']:.6f}")
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
    print(f"  >> {lv1['summary']}\n")

    # ── Level 2 ──
    print("=" * 60)
    print("  Level 2: z Pathway Activation (cog vs zero)")
    print("=" * 60)
    lv2 = diagnose_z_activation(vla_model)
    results["level2"] = lv2

    print(f"  DiT output with real z   : mean_abs={lv2['output_mean']:.6f}")
    print(f"  Delta (real z vs zero z) : mean={lv2['delta_mean']:.6f} max={lv2['delta_max']:.6f}")
    print(f"  Contribution ratio       : {lv2['ratio']:.4f}")
    print(f"  Verdict: {lv2['verdict']}")
    print(f"  >> {lv2['summary']}\n")

    # ── Level 3 Enhanced ──
    print("=" * 60)
    print("  Level 3 Enhanced: z Pathway Sensitivity Analysis")
    print("=" * 60)
    print(f"  DDIM 10 steps, eta=0 | seed={seed}")
    print(f"  Perturbation scales: {perturbation_scales}")
    print(f"  CFG scales: {cfg_scales}")

    lv3 = diagnose_z_sensitivity_v2(
        vla_model, image, instruction,
        unnorm_key=unnorm_key, cfg_scale=cfg_scale, seed=seed,
        perturbation_scales=perturbation_scales, cfg_scales=cfg_scales,
    )
    results["level3"] = lv3

    # ── Final Summary ──
    print()
    print("=" * 60)
    print("  FINAL SUMMARY: z (cog_tokens) Pathway")
    print("=" * 60)
    print(f"  Level 1 (weights):        {lv1['verdict']}")
    print(f"  Level 2 (activations):    {lv2['verdict']}")
    print(f"  Level 3 (sensitivity):    {lv3['verdict']}")
    print(f"    3a det. floor (raw):    {lv3['3a']['delta_raw']:.2e}")
    print(f"    3b monotonic:           {lv3['3b']['is_monotonic']}")
    print(f"    3c CFG amplifies:       {lv3['3c']['cfg_amplifies']}")
    print(f"    3d content-sensitive:   {lv3['3d']['analysis']['content_sensitive']}")
    print()

    v = lv3['verdict']
    if v == "ALIVE":
        print("  ALIVE: z pathway is responsive and content-sensitive.")
        print("  -> SPSA on cog_tokens is viable.")
        print(f"  >> {lv3['summary']}")
    elif v == "WEAK":
        print("  WEAK: z pathway responds but with low gain.")
        print("  -> SPSA may work with larger alpha or more iterations.")
        print(f"  >> {lv3['summary']}")
    elif v == "BIAS_ONLY":
        print("  BIAS_ONLY: z acts as static bias (on/off), content not utilized.")
        print("  -> SPSA on z alone is insufficient; consider architectural changes")
        print("     (e.g., AdaLN, broadcast z to all tokens, gating).")
        print(f"  >> {lv3['summary']}")
    elif v == "DEAD":
        print("  DEAD: No measurable z response.")
        print("  -> z pathway is non-functional. Investigate model loading/training.")
        print(f"  >> {lv3['summary']}")
    else:
        print(f"  {v}: {lv3['summary']}")
    print()

    results["recommendation"] = v
    return results


# ---------------------------------------------------------------------------
# Backward-compatible alias for v1 callers
# ---------------------------------------------------------------------------
diagnose_z_sensitivity = diagnose_z_sensitivity_v2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("cog_attn (z pathway) Diagnostics v2 for MemoryVLA")
    print()
    print("Usage (Colab):")
    print("  from scripts.diagnose_cog_attn import run_cog_attn_diagnostics")
    print('  result = run_cog_attn_diagnostics(vla, image, instruction, "libero_spatial_no_noops")')
