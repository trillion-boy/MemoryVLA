"""
SPSA optimization on the z (cog_tokens) pathway for MemoryVLA.

Why z pathway instead of per_token pathway:
  - per_attn: DEAD (zero-init, in_proj ≈ 0.0003)
  - z pathway: WEAK but ALIVE (content-sensitive, linear dose-response)
  - The old colab_spsa_latent_opt.py injected L into per_tokens → ineffective
  - This script injects L_z into cog_tokens → z = cog_tokens + L_z

Key improvements over the old SPSA:
  1. Targets z pathway (cog_tokens) instead of dead per_token pathway
  2. Composite objective: J = w_lang * conf_llm + w_act * conf_action
     - conf_llm: LLM token probability (language-level)
     - conf_action: trajectory consistency (action-level, zero extra cost)
  3. Phase A calibration → Phase B full optimization
  4. Stability monitoring (L_z norm, action jerk)

Diagnostic basis (from diagnose_cog_attn.py v2):
  - 3a: deterministic floor = 0 (DDIM eta=0)
  - 3b: linear dose-response (delta ∝ perturbation scale)
  - 3c: CFG amplifies z signal 1.8x at cfg=3.0
  - 3d: content-sensitive (z_shuffled delta ≈ z_zero delta >> z_perturb)
  - Verdict: WEAK — alive and content-sensitive, but low gain

Usage (Colab):
    from scripts.colab_spsa_z_pathway import (
        ZSPSAConfig, calibrate_z_spsa, optimize_z_spsa, run_z_spsa_full,
    )

    # Quick: Phase A calibration + Phase B in one call
    cfg = ZSPSAConfig(unnorm_key="libero_spatial_no_noops")
    result = run_z_spsa_full(vla, image, instruction, cfg)
    print(result["best_params"])
    print(result["final_actions"])

    # Manual: calibrate first, then optimize
    cal = calibrate_z_spsa(vla, image, instruction, cfg)
    print(cal["best"])
    # Update cfg with best params, then run Phase B
    result = optimize_z_spsa(vla, image, instruction, cfg)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# Reuse trajectory capture from existing confidence module
from scripts.colab_diffusion_confidence import (
    _capture_pred_xstarts,
    _compute_trajectory_confidence,
)


# ===================================================================
# Configuration
# ===================================================================
@dataclass
class ZSPSAConfig:
    """Hyper-parameters for z-pathway SPSA optimization."""

    # -- model / task --
    unnorm_key: str = "libero_spatial_no_noops"

    # -- z pathway --
    cog_dim: int = 4096
    """Dimension of cog_tokens (LLM hidden size)."""

    # -- SPSA schedule --
    num_iters: int = 50
    """Number of SPSA iterations for Phase B."""
    a: float = 0.3
    """Step-size numerator. Larger than per_token SPSA (0.1) to compensate
    for z pathway's low gain."""
    c: float = 0.15
    """Perturbation scale numerator. Larger than per_token (0.05) because
    z pathway has low gain — need bigger perturbation for measurable signal."""
    A: float = 10.0
    """Step-size stability constant (~10-20% of num_iters)."""
    alpha: float = 0.602
    """Step-size decay exponent (standard SPSA)."""
    gamma: float = 0.101
    """Perturbation decay exponent (standard SPSA)."""

    # -- inference --
    cfg_scale: float = 1.5
    """CFG scale. Higher amplifies z signal (1.8x at 3.0 per diagnostic)."""
    use_ddim: bool = True
    num_ddim_steps: int = 10

    # -- composite objective --
    w_lang: float = 0.5
    """Weight for LLM confidence in composite J."""
    w_act: float = 0.5
    """Weight for action trajectory confidence in composite J."""
    normalize_objectives: bool = True
    """If True, normalize conf_llm and conf_action to comparable scales
    using running statistics from the first few iterations. This prevents
    one metric from dominating the composite objective when their raw
    scales differ (e.g., conf_llm ≈ 0.95 vs conf_action ≈ 0.3)."""
    confidence_type: str = "max_prob"
    """LLM confidence type: 'max_prob', 'max_logit', 'token_prob'."""
    traj_tail_fraction: float = 0.5
    """Fraction of denoising steps to use for trajectory confidence."""
    traj_tau: float = 0.1
    """Temperature for trajectory confidence sigmoid."""

    # -- stability --
    max_L_norm: float = 50.0
    """Clip L_z norm to prevent explosion."""

    # -- calibration (Phase A) --
    cal_iters: int = 15
    """Short iterations per configuration in Phase A."""
    cal_cfg_scales: Tuple[float, ...] = (1.5, 2.0, 3.0)
    cal_c_multipliers: Tuple[float, ...] = (1.0, 2.0)
    cal_a_multipliers: Tuple[float, ...] = (1.0, 2.0)

    # -- misc --
    seed: Optional[int] = 42
    verbose: bool = True


# ===================================================================
# SPSA schedule helpers
# ===================================================================
# ===================================================================
# Objective scale normalizer
# ===================================================================
class _RunningNormalizer:
    """Tracks running mean/std of two signals and normalizes them to [0, 1].

    Solves the scale mismatch problem:
      conf_llm  might live in [0.85, 0.99]  (narrow, high)
      conf_action might live in [0.1, 0.5]  (wide, low)
    Without normalization, J = 0.5*llm + 0.5*act is dominated by llm.

    After warmup (default 4 samples), both signals are z-scored then
    mapped to [0, 1] via sigmoid, so w_lang and w_act control true
    relative importance.
    """

    def __init__(self, warmup: int = 4):
        self.warmup = warmup
        self._llm_vals: List[float] = []
        self._act_vals: List[float] = []

    @property
    def ready(self) -> bool:
        return len(self._llm_vals) >= self.warmup

    def update(self, conf_llm: float, conf_action: float):
        self._llm_vals.append(conf_llm)
        self._act_vals.append(conf_action)

    def normalize(self, conf_llm: float, conf_action: float) -> Tuple[float, float]:
        """Return normalized scores in ~[0, 1] with comparable spread."""
        if not self.ready:
            return conf_llm, conf_action

        llm_mean = np.mean(self._llm_vals)
        llm_std = max(np.std(self._llm_vals), 1e-6)
        act_mean = np.mean(self._act_vals)
        act_std = max(np.std(self._act_vals), 1e-6)

        # z-score → sigmoid → [0, 1]
        def _sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        nlm = _sigmoid((conf_llm - llm_mean) / llm_std)
        nac = _sigmoid((conf_action - act_mean) / act_std)
        return float(nlm), float(nac)


def _gain_ak(k: int, cfg: ZSPSAConfig) -> float:
    """Step-size: a_k = a / (k + 1 + A)^alpha."""
    return cfg.a / ((k + 1 + cfg.A) ** cfg.alpha)


def _gain_ck(k: int, cfg: ZSPSAConfig) -> float:
    """Perturbation scale: c_k = c / (k + 1)^gamma."""
    return cfg.c / ((k + 1) ** cfg.gamma)


# ===================================================================
# Composite objective evaluation
# ===================================================================
def _ensure_ddim(vla_model, steps=10):
    """Pre-create DDIM so hooks on ddim_sample_loop work reliably."""
    if vla_model.action_model.ddim_diffusion is None:
        vla_model.action_model.create_ddim(ddim_step=steps)


@torch.inference_mode()
def _eval_z_objective(
    vla_model,
    image: Image.Image,
    instruction: str,
    L_z: torch.Tensor,
    cfg: ZSPSAConfig,
) -> Tuple[float, dict]:
    """
    Evaluate composite objective with L_z injected into z pathway.

    J = w_lang * conf_llm + w_act * conf_action

    Returns:
        J: scalar (higher = better)
        info: dict with conf_llm, conf_action, actions, etc.
    """
    _ensure_ddim(vla_model, cfg.num_ddim_steps)

    # ── Hook: inject L_z into cog_tokens ──
    original_process = vla_model.cog_mem_bank.process_batch

    def patched_process(*args, **kwargs):
        result = original_process(*args, **kwargs)
        return result + L_z.to(result.device, dtype=result.dtype)

    vla_model.cog_mem_bank.process_batch = patched_process

    try:
        # ── Capture trajectory + LLM confidence in single pass ──
        with _capture_pred_xstarts(
            vla_model, use_ddim=cfg.use_ddim,
            num_ddim_steps=cfg.num_ddim_steps,
        ) as captured:
            actions, norm_actions, llm_conf = vla_model.predict_action(
                image=image,
                instruction=instruction,
                unnorm_key=cfg.unnorm_key,
                cfg_scale=cfg.cfg_scale,
                use_ddim=cfg.use_ddim,
                num_ddim_steps=cfg.num_ddim_steps,
                episode_first_frame="True",
                return_confidence=True,
                confidence_type=cfg.confidence_type,
            )

        conf_llm = float(llm_conf.mean())

        # Trajectory consistency confidence
        if captured["pred_xstarts"]:
            traj = _compute_trajectory_confidence(
                captured["pred_xstarts"],
                using_cfg=(cfg.cfg_scale > 1.0),
                tail_fraction=cfg.traj_tail_fraction,
                tau=cfg.traj_tau,
            )
            conf_action = traj["action_confidence"]
            traj_var = traj["mean_variance"]
        else:
            conf_action = conf_llm
            traj_var = 0.0

        J = cfg.w_lang * conf_llm + cfg.w_act * conf_action

        return J, {
            "conf_llm": conf_llm,
            "conf_action": conf_action,
            "traj_variance": traj_var,
            "J": J,
            "actions": actions,
            "norm_actions": norm_actions,
        }

    finally:
        vla_model.cog_mem_bank.process_batch = original_process


# ===================================================================
# Core SPSA loop for z pathway
# ===================================================================
@torch.inference_mode()
def optimize_z_spsa(
    vla_model,
    image: Image.Image,
    instruction: str,
    cfg: ZSPSAConfig,
    init_L: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, List[dict]]:
    """
    Maximize composite objective w.r.t. L_z via SPSA on z pathway.

    L_z is injected as: cog_tokens_out = cog_mem_bank(cog_tokens) + L_z

    Args:
        vla_model: Loaded MemoryVLA (eval mode, on GPU).
        image: PIL observation image.
        instruction: Language instruction.
        cfg: SPSA configuration.
        init_L: Optional warm-start latent [1, 1, cog_dim].

    Returns:
        L_star: Optimized latent [1, 1, cog_dim].
        history: List of per-iteration records.
    """
    device = next(vla_model.parameters()).device
    dtype = next(vla_model.action_model.net.parameters()).dtype

    if init_L is not None:
        L_z = init_L.clone().to(device=device, dtype=torch.float32)
    else:
        L_z = torch.zeros(1, 1, cfg.cog_dim, device=device, dtype=torch.float32)

    if cfg.seed is not None:
        rng = torch.Generator(device=device).manual_seed(cfg.seed)
    else:
        rng = None

    history: List[dict] = []
    prev_actions = None
    normalizer = _RunningNormalizer(warmup=4) if cfg.normalize_objectives else None

    for k in range(cfg.num_iters):
        ak = _gain_ak(k, cfg)
        ck = _gain_ck(k, cfg)

        # Bernoulli ±1 perturbation
        delta = torch.where(
            torch.rand(L_z.shape, device=device, generator=rng) < 0.5,
            torch.ones_like(L_z),
            -torch.ones_like(L_z),
        )

        L_plus = L_z + ck * delta
        L_minus = L_z - ck * delta

        J_p, info_p = _eval_z_objective(vla_model, image, instruction, L_plus, cfg)
        J_m, info_m = _eval_z_objective(vla_model, image, instruction, L_minus, cfg)

        # Scale normalization: prevent one metric from dominating
        if normalizer is not None:
            normalizer.update(info_p["conf_llm"], info_p["conf_action"])
            normalizer.update(info_m["conf_llm"], info_m["conf_action"])
            if normalizer.ready:
                nlm_p, nac_p = normalizer.normalize(info_p["conf_llm"], info_p["conf_action"])
                nlm_m, nac_m = normalizer.normalize(info_m["conf_llm"], info_m["conf_action"])
                J_p = cfg.w_lang * nlm_p + cfg.w_act * nac_p
                J_m = cfg.w_lang * nlm_m + cfg.w_act * nac_m

        # SPSA gradient estimate (ascent → maximize J)
        ghat = ((J_p - J_m) / (2.0 * ck)) * delta

        L_z = L_z + ak * ghat

        # Stability: clip L_z norm
        L_norm = float(L_z.norm().item())
        if L_norm > cfg.max_L_norm:
            L_z = L_z * (cfg.max_L_norm / L_norm)
            L_norm = cfg.max_L_norm

        # Action jerk monitoring
        cur_actions = 0.5 * (info_p["actions"] + info_m["actions"])
        jerk = float(np.abs(cur_actions - prev_actions).mean()) if prev_actions is not None else 0.0
        prev_actions = cur_actions

        J_cur = 0.5 * (J_p + J_m)
        raw_llm = 0.5 * (info_p["conf_llm"] + info_m["conf_llm"])
        raw_act = 0.5 * (info_p["conf_action"] + info_m["conf_action"])
        record = {
            "iter": k + 1,
            "J": J_cur,
            "J_p": J_p,
            "J_m": J_m,
            "conf_llm": raw_llm,
            "conf_action": raw_act,
            "traj_var": 0.5 * (info_p["traj_variance"] + info_m["traj_variance"]),
            "L_norm": L_norm,
            "ak": ak,
            "ck": ck,
            "jerk": jerk,
        }
        history.append(record)

        if cfg.verbose:
            norm_tag = ""
            if normalizer is not None and normalizer.ready:
                norm_tag = " [normalized]"
            print(
                f"  [iter {k+1:3d}/{cfg.num_iters}]  "
                f"J={J_cur:.4f}{norm_tag}  "
                f"llm={raw_llm:.4f}  "
                f"act={raw_act:.4f}  "
                f"|L|={L_norm:.2f}  "
                f"jerk={jerk:.4f}  "
                f"ak={ak:.4f} ck={ck:.4f}"
            )

    return L_z.detach(), history


# ===================================================================
# Phase A: Calibration sweep
# ===================================================================
@torch.inference_mode()
def calibrate_z_spsa(
    vla_model,
    image: Image.Image,
    instruction: str,
    cfg: ZSPSAConfig,
) -> dict:
    """
    Phase A: Short SPSA sweeps over (cfg_scale, c, a) to find best config.

    Runs cal_iters per combination, measures J improvement and stability.

    Returns:
        dict with:
          - best: best configuration dict
          - all_results: list of all tested configurations
          - baseline_J: objective without any L_z
    """
    print("=" * 60)
    print("  Phase A: Calibration Sweep")
    print("=" * 60)

    # Baseline: no L_z
    device = next(vla_model.parameters()).device
    L_zero = torch.zeros(1, 1, cfg.cog_dim, device=device, dtype=torch.float32)
    J_base, info_base = _eval_z_objective(vla_model, image, instruction, L_zero, cfg)
    print(f"  Baseline (L_z=0): J={J_base:.4f} "
          f"(llm={info_base['conf_llm']:.4f}, act={info_base['conf_action']:.4f})")
    print()

    all_results = []

    configs = [
        (cs, cm, am)
        for cs in cfg.cal_cfg_scales
        for cm in cfg.cal_c_multipliers
        for am in cfg.cal_a_multipliers
    ]
    total = len(configs)

    for idx, (cfg_s, c_mult, a_mult) in enumerate(configs):
        cal_cfg = copy.copy(cfg)
        cal_cfg.cfg_scale = cfg_s
        cal_cfg.c = cfg.c * c_mult
        cal_cfg.a = cfg.a * a_mult
        cal_cfg.num_iters = cfg.cal_iters
        cal_cfg.verbose = False

        print(f"  [{idx+1}/{total}] cfg={cfg_s:.1f}  c={cal_cfg.c:.3f} (x{c_mult})  "
              f"a={cal_cfg.a:.3f} (x{a_mult}) ... ", end="", flush=True)

        L_z, history = optimize_z_spsa(vla_model, image, instruction, cal_cfg)

        J_start = history[0]["J"]
        J_end = history[-1]["J"]
        improvement = J_end - J_start
        L_norm = history[-1]["L_norm"]
        max_jerk = max(h["jerk"] for h in history[1:]) if len(history) > 1 else 0.0
        stable = L_norm < cfg.max_L_norm * 0.9

        # J trend: monotonically improving?
        J_vals = [h["J"] for h in history]
        # Check if second half is generally better than first half
        mid = len(J_vals) // 2
        trend_up = np.mean(J_vals[mid:]) > np.mean(J_vals[:mid])

        result = {
            "cfg_scale": cfg_s,
            "c": cal_cfg.c,
            "a": cal_cfg.a,
            "c_mult": c_mult,
            "a_mult": a_mult,
            "J_start": J_start,
            "J_end": J_end,
            "improvement": improvement,
            "L_norm": L_norm,
            "max_jerk": max_jerk,
            "stable": stable,
            "trend_up": trend_up,
            "history": history,
        }
        all_results.append(result)

        status = "OK" if stable else "UNSTABLE"
        trend = "UP" if trend_up else "flat/down"
        print(f"J: {J_start:.4f} -> {J_end:.4f} ({improvement:+.4f})  "
              f"|L|={L_norm:.1f}  [{status}, {trend}]")

    # Select best: stable + highest improvement
    stable_results = [r for r in all_results if r["stable"]]
    if not stable_results:
        print("\n  WARNING: No stable configurations found. Using least unstable.")
        stable_results = sorted(all_results, key=lambda r: r["L_norm"])[:3]

    best = max(stable_results, key=lambda r: r["improvement"])

    print()
    print(f"  Best: cfg={best['cfg_scale']:.1f}  "
          f"c={best['c']:.3f} (x{best['c_mult']})  "
          f"a={best['a']:.3f} (x{best['a_mult']})")
    print(f"    J: {best['J_start']:.4f} -> {best['J_end']:.4f} "
          f"(improvement={best['improvement']:+.4f})")
    print()

    return {
        "best": best,
        "all_results": all_results,
        "baseline_J": J_base,
        "baseline_info": info_base,
    }


# ===================================================================
# Phase B: Full optimization with best params
# ===================================================================
@torch.inference_mode()
def run_phase_b(
    vla_model,
    image: Image.Image,
    instruction: str,
    cfg: ZSPSAConfig,
    cal_result: dict,
) -> dict:
    """
    Phase B: Full SPSA optimization using best params from Phase A.

    Args:
        cal_result: Output from calibrate_z_spsa().

    Returns:
        dict with L_star, history, final actions, etc.
    """
    best = cal_result["best"]

    # Apply best params
    run_cfg = copy.copy(cfg)
    run_cfg.cfg_scale = best["cfg_scale"]
    run_cfg.c = best["c"]
    run_cfg.a = best["a"]

    print("=" * 60)
    print("  Phase B: Full Optimization")
    print("=" * 60)
    print(f"  cfg_scale={run_cfg.cfg_scale:.1f}  c={run_cfg.c:.3f}  a={run_cfg.a:.3f}")
    print(f"  num_iters={run_cfg.num_iters}  w_lang={run_cfg.w_lang}  w_act={run_cfg.w_act}")
    print()

    L_star, history = optimize_z_spsa(vla_model, image, instruction, run_cfg)

    # Final evaluation
    J_final, info_final = _eval_z_objective(
        vla_model, image, instruction, L_star, run_cfg
    )

    # Summary
    J_start = history[0]["J"]
    J_end = history[-1]["J"]

    print()
    print("  " + "-" * 44)
    print(f"  J: {J_start:.4f} -> {J_end:.4f} -> final={J_final:.4f}")
    print(f"  conf_llm:    {history[0]['conf_llm']:.4f} -> {info_final['conf_llm']:.4f}")
    print(f"  conf_action: {history[0]['conf_action']:.4f} -> {info_final['conf_action']:.4f}")
    print(f"  |L_z|: {history[-1]['L_norm']:.2f}")
    print()

    return {
        "L_star": L_star,
        "history": history,
        "final_J": J_final,
        "final_info": info_final,
        "final_actions": info_final["actions"],
        "final_norm_actions": info_final["norm_actions"],
        "run_cfg": run_cfg,
    }


# ===================================================================
# Combined: Phase A + Phase B
# ===================================================================
@torch.inference_mode()
def run_z_spsa_full(
    vla_model,
    image: Image.Image,
    instruction: str,
    cfg: ZSPSAConfig,
) -> dict:
    """
    Full pipeline: Phase A (calibration) -> Phase B (optimization).

    Usage:
        cfg = ZSPSAConfig(unnorm_key="libero_spatial_no_noops")
        result = run_z_spsa_full(vla, image, instruction, cfg)
        print(result["final_actions"])
        print(result["best_params"])
    """
    # Phase A
    cal = calibrate_z_spsa(vla_model, image, instruction, cfg)

    # Phase B
    phase_b = run_phase_b(vla_model, image, instruction, cfg, cal)

    # Final summary
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Baseline J : {cal['baseline_J']:.4f}")
    print(f"  Final J    : {phase_b['final_J']:.4f}")
    print(f"  Improvement: {phase_b['final_J'] - cal['baseline_J']:+.4f}")
    print(f"  Best params: cfg={cal['best']['cfg_scale']:.1f}  "
          f"c={cal['best']['c']:.3f}  a={cal['best']['a']:.3f}")
    print(f"  |L_z|      : {float(phase_b['L_star'].norm()):.2f}")
    print()

    return {
        "calibration": cal,
        "phase_b": phase_b,
        "L_star": phase_b["L_star"],
        "final_actions": phase_b["final_actions"],
        "final_J": phase_b["final_J"],
        "baseline_J": cal["baseline_J"],
        "best_params": {
            "cfg_scale": cal["best"]["cfg_scale"],
            "c": cal["best"]["c"],
            "a": cal["best"]["a"],
        },
    }


# ===================================================================
# Utility: inject L_z for subsequent predict_action calls
# ===================================================================
def set_z_latent(vla_model, L_z: torch.Tensor):
    """
    Permanently patch cog_mem_bank to add L_z for all future calls.
    Call with L_z=None to remove the patch.

    Usage after optimization:
        set_z_latent(vla, result["L_star"])
        # Now every predict_action call includes L_z
        actions, _ = vla.predict_action(image, instruction, ...)
    """
    # Remove existing patch if any
    if hasattr(vla_model, "_z_latent_original_process"):
        vla_model.cog_mem_bank.process_batch = vla_model._z_latent_original_process
        del vla_model._z_latent_original_process

    if L_z is None:
        return

    vla_model._z_latent_original_process = vla_model.cog_mem_bank.process_batch
    original = vla_model._z_latent_original_process
    _L = L_z.detach().clone()

    def patched(*args, **kwargs):
        result = original(*args, **kwargs)
        return result + _L.to(result.device, dtype=result.dtype)

    vla_model.cog_mem_bank.process_batch = patched


# ===================================================================
# CLI
# ===================================================================
if __name__ == "__main__":
    print("Z-Pathway SPSA Optimization for MemoryVLA")
    print()
    print("Usage (Colab):")
    print("  from scripts.colab_spsa_z_pathway import ZSPSAConfig, run_z_spsa_full")
    print('  cfg = ZSPSAConfig(unnorm_key="libero_spatial_no_noops")')
    print("  result = run_z_spsa_full(vla, image, instruction, cfg)")
    print()
    print("After optimization, inject L_z for all future calls:")
    print("  from scripts.colab_spsa_z_pathway import set_z_latent")
    print('  set_z_latent(vla, result["L_star"])')
    print("  actions, _ = vla.predict_action(image, instruction, ...)")
