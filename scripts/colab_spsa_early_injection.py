"""
Phase 1: Early injection SPSA — L_z into projected_patch_embeddings (BEFORE LLM).

Structural difference from colab_spsa_z_pathway.py (late injection):
  Late  injection: cog_mem_bank output + L_z → AFTER LLM → conf_llm INVARIANT
  Early injection: projected_patch_embeddings + L_z → BEFORE LLM → conf_llm CAN RESPOND

This is the key A/B experiment for ControlMLLM-style memory steering:
  - If conf_llm responds to early L_z, the optimization landscape becomes richer
    (two gradient signals instead of one).
  - If not, the two injection sites are functionally equivalent and only differ
    in propagation depth.

Design constraints (Phase 1 — position effect only):
  - L_z shape: [1, 1, 4096] — token-shared, broadcast to all patches.
    Same shape as late injection so the comparison is fair (same DoF).
  - Same SPSA infrastructure: paired noise, calibration, normalization.
  - Same seed/iter/image/instruction for reproducible A/B.

Hook mechanism:
  Late  injection hooks `cog_mem_bank.process_batch` (monkey-patch).
  Early injection hooks `vlm.projector` via `register_forward_hook` (cleaner).
  The projector hook fires once per generate() call (prefill pass only — cached
  autoregressive steps skip vision entirely, see PrismaticVLM.forward line 330).

Usage (Colab):
    from scripts.colab_spsa_early_injection import (
        EarlySPSAConfig, run_early_spsa_full,
        set_early_z_latent, clear_early_z_latent,
    )
    # Reuse visualization from z_pathway:
    from scripts.colab_spsa_z_pathway import (
        plot_input_frame, plot_optimization_curves, plot_action_trajectory,
        render_action_rollout, closed_loop_rollout_multichunk,
        save_z_result, load_z_result,
    )

    cfg = EarlySPSAConfig(unnorm_key="libero_spatial_no_noops")
    result = run_early_spsa_full(vla, image, instruction, cfg)

    # Compare with late injection baseline:
    #   late_result  = load_z_result("L_star_late.pt")
    #   early_result = result
    #   print(f"Late  conf_llm delta: {late_result['baseline_info']...}")
    #   print(f"Early conf_llm delta: ...")
"""
from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass
import io
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# ── Shared utilities from late-injection SPSA ──
from scripts.colab_spsa_z_pathway import (
    ZSPSAConfig,
    _RunningNormalizer,
    _gain_ak,
    _gain_ck,
    _ensure_ddim,
    _compute_smoothness,
)
from scripts.colab_diffusion_confidence import (
    _capture_pred_xstarts,
    _compute_trajectory_confidence,
)


# ===================================================================
# Configuration
# ===================================================================
@dataclass
class EarlySPSAConfig(ZSPSAConfig):
    """Config for early injection SPSA.

    Inherits all fields from ZSPSAConfig. Overrides defaults where the
    injection site changes expected behavior.
    """

    # -- override: w_lang can now be meaningful --
    w_lang: float = 0.0
    """Weight for LLM confidence in composite J.
    Default 0.0 for Phase 1 (position-effect-only test).
    Unlike late injection, conf_llm MAY respond to L_z here.
    Phase 2: if auto-gate probe confirms responsiveness, increase to ~0.3."""

    w_act: float = 1.0
    """Weight for action confidence. Kept at 1.0 for Phase 1 parity."""

    # -- override: tighter norm for pre-LLM injection --
    max_L_norm: float = 30.0
    """Tighter clip than late injection (50.0) because L_z propagates through
    the full LLM — large perturbations could destabilize attention.
    Start conservative; increase if optimization plateaus early."""

    # -- override: smaller perturbation default --
    c: float = 0.10
    """Perturbation scale. Smaller than late injection (0.15) because the
    signal path is longer (projector → LLM → cog_tokens → DiT).
    Calibration Phase A will find the best multiplier anyway."""

    # -- early-injection specific --
    injection_target: str = "projector"
    """Where to inject L_z in the VLM forward pass.
    'projector': after self.projector(patch_features), before LLM.
    This is the only option for Phase 1."""

    def __post_init__(self):
        super().__post_init__()
        if self.injection_target not in ("projector",):
            raise ValueError(
                f"injection_target must be 'projector', got '{self.injection_target}'"
            )


# ===================================================================
# DDP/wrapper-safe model unwrapping
# ===================================================================
def _unwrap_model(vla_model):
    """Unwrap DDP/FSDP/DataParallel wrapper to get the raw MemoryVLA.

    Handles:
      - torch.nn.parallel.DistributedDataParallel  (.module)
      - torch.nn.parallel.DataParallel              (.module)
      - torch.distributed.fsdp.FullyShardedDataParallel (.module)
      - No wrapper (returns as-is)
    """
    model = vla_model
    # Peel off wrapper layers (handles nested wrappers too)
    while hasattr(model, "module"):
        model = model.module
    return model


# ===================================================================
# Projector hook helpers
# ===================================================================
def _validate_L_z_shape(L_z: torch.Tensor, cog_dim: int):
    """Validate L_z shape is [1, 1, cog_dim] for token-shared injection.

    Raises ValueError with clear diagnostic if shape is wrong.
    """
    expected = (1, 1, cog_dim)
    if L_z.shape != expected:
        raise ValueError(
            f"L_z shape mismatch: got {tuple(L_z.shape)}, expected {expected}. "
            f"Phase 1 uses token-shared injection — L_z must be [1, 1, {cog_dim}]."
        )


def _make_projector_hook(L_z: torch.Tensor):
    """Create a forward_hook function that adds L_z to projector output.

    The projector outputs [bsz, num_patches, llm_embed_dim].
    L_z is [1, 1, 4096] and broadcasts across bsz and num_patches dimensions.

    Returns:
        hook_fn: Callable for register_forward_hook.
    """
    def hook_fn(module, input, output):
        # output: [bsz, num_patches, 4096]
        # L_z:    [1, 1, 4096] → broadcasts to [bsz, num_patches, 4096]
        return output + L_z.to(device=output.device, dtype=output.dtype)
    return hook_fn


def _get_projector(vla_model):
    """Get the projector module from a MemoryVLA model (wrapper-safe).

    Unwraps DDP/FSDP before accessing vlm.projector.
    """
    return _unwrap_model(vla_model).vlm.projector


# ===================================================================
# Composite objective evaluation (early injection)
# ===================================================================
@torch.inference_mode()
def _eval_early_objective(
    vla_model,
    image: Image.Image,
    instruction: str,
    L_z: torch.Tensor,
    cfg: EarlySPSAConfig,
    noise_seed: Optional[int] = None,
    gate_disabled: bool = False,
) -> Tuple[float, dict]:
    """
    Evaluate composite objective with L_z injected BEFORE LLM (early injection).

    J = (gate * w_lang * conf_llm) + (w_act * conf_action)

    Key difference from _eval_z_objective (late):
      - L_z is added to projected_patch_embeddings via projector forward hook.
      - This means L_z flows through the LLM, potentially influencing conf_llm.
      - The hook is registered before predict_action and removed after.

    Args:
        L_z: [1, 1, cog_dim] latent perturbation (token-shared, broadcasts).
        noise_seed: If set, fixes all RNG for paired SPSA evaluation.
        gate_disabled: If True, gate is forced to 1.0.

    Returns:
        J: scalar (higher = better)
        info: dict with conf_llm, conf_action, actions, raw_actions, etc.
    """
    # Unwrap once — all internal access uses raw_model to avoid DDP mismatch
    raw_model = _unwrap_model(vla_model)

    _ensure_ddim(raw_model, cfg.num_ddim_steps)
    _validate_L_z_shape(L_z, cfg.cog_dim)

    # ── Save and fix all RNG sources for paired SPSA evaluation ──
    _rng_states_saved = None
    if noise_seed is not None:
        _rng_states_saved = {
            "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda": [torch.cuda.get_rng_state(d)
                           for d in range(torch.cuda.device_count())]
                          if torch.cuda.is_available() else [],
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
        torch.manual_seed(noise_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(noise_seed)
        np.random.seed(noise_seed % (2**31))
        random.seed(noise_seed)

    # ── Guard: temporarily remove any persistent early hook ──
    # Check on raw_model (where set_early_z_latent stores state) to avoid
    # DDP wrapper mismatch where eval checks wrapper but state lives on inner model.
    had_persistent_hook = hasattr(raw_model, "_early_z_hook_handle")
    persistent_handle = None
    if had_persistent_hook:
        if cfg.verbose:
            print("  [warn] persistent early L_z hook detected — "
                  "temporarily removed for isolated eval")
        persistent_handle = raw_model._early_z_hook_handle
        persistent_handle.remove()

    # ── Register projector hook for this evaluation ──
    projector = _get_projector(vla_model)
    hook_fn = _make_projector_hook(L_z)
    handle = projector.register_forward_hook(hook_fn)

    try:
        # ── Capture trajectory + LLM confidence in single pass ──
        # Suppress "** reset memory **" spam from predict_action (prints once
        # per call × hundreds of calls during calibration = unreadable output).
        with _capture_pred_xstarts(
            vla_model, use_ddim=cfg.use_ddim,
            num_ddim_steps=cfg.num_ddim_steps,
        ) as captured, contextlib.redirect_stdout(io.StringIO()):
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

        # -- Sub-metrics for conf_action --
        raw_actions = None
        if captured["pred_xstarts"]:
            traj = _compute_trajectory_confidence(
                captured["pred_xstarts"],
                using_cfg=(cfg.cfg_scale > 1.0),
                tail_fraction=cfg.traj_tail_fraction,
                tau=cfg.traj_tau,
            )
            conf_traj = traj["action_confidence"]
            traj_var = traj["mean_variance"]

            raw_pred = captured["pred_xstarts"][-1]
            if cfg.cfg_scale > 1.0 and raw_pred.shape[0] > 1:
                raw_pred = raw_pred[: raw_pred.shape[0] // 2]
            raw_actions = raw_pred.cpu().numpy()
        else:
            conf_traj = conf_llm
            traj_var = 0.0

        # Smoothness
        if raw_actions is not None:
            smooth_source = np.clip(raw_actions, -1.0, 1.0)
        else:
            smooth_source = norm_actions
        conf_smooth = _compute_smoothness(smooth_source, cfg.smooth_tau)

        # Combine sub-metrics
        conf_action = cfg.w_traj * conf_traj + cfg.w_smooth * conf_smooth

        # -- Gate --
        if gate_disabled or cfg.gate_mode == "disabled":
            gate = 1.0
        else:
            gate = min(1.0, conf_llm / max(cfg.llm_gate_floor, 1e-6))

        # Gate dampens only the lang term; action term is never gated.
        J = (gate * cfg.w_lang * conf_llm) + (cfg.w_act * conf_action)

        return J, {
            "conf_llm": conf_llm,
            "conf_action": conf_action,
            "conf_traj": conf_traj,
            "conf_smooth": conf_smooth,
            "traj_variance": traj_var,
            "gate": gate,
            "J": J,
            "actions": actions,
            "norm_actions": norm_actions,
            "raw_actions": raw_actions,
        }

    finally:
        # Remove eval hook
        handle.remove()
        # Restore persistent hook if it was active (on raw_model, not wrapper)
        if had_persistent_hook and persistent_handle is not None:
            projector = _get_projector(raw_model)
            L_persistent = raw_model._early_z_latent
            new_handle = projector.register_forward_hook(
                _make_projector_hook(L_persistent)
            )
            raw_model._early_z_hook_handle = new_handle
        # Restore RNG states
        if _rng_states_saved is not None:
            torch.random.set_rng_state(_rng_states_saved["torch_cpu"])
            for d, state in enumerate(_rng_states_saved["torch_cuda"]):
                torch.cuda.set_rng_state(state, d)
            np.random.set_state(_rng_states_saved["numpy"])
            random.setstate(_rng_states_saved["python"])


# ===================================================================
# Gate mode resolution (early injection)
# ===================================================================
@torch.inference_mode()
def _resolve_gate_mode_early(
    vla_model,
    image: Image.Image,
    instruction: str,
    cfg: EarlySPSAConfig,
) -> bool:
    """Probe whether conf_llm responds to early L_z.

    This is the critical Phase 1 experiment:
      - Late injection: conf_llm is ALWAYS invariant (proven).
      - Early injection: conf_llm SHOULD respond because L_z flows through LLM.

    Returns:
        True if gate should be disabled (conf_llm invariant — same as late).
        False if conf_llm responds (early injection advantage confirmed!).
    """
    if cfg.gate_mode == "disabled":
        return True
    if cfg.gate_mode == "soft":
        return False

    # "auto": probe with L=0 vs L=perturbation
    device = next(vla_model.parameters()).device
    L_zero = torch.zeros(1, 1, cfg.cog_dim, device=device, dtype=torch.float32)
    L_pert = torch.randn(1, 1, cfg.cog_dim, device=device) * cfg.c

    probe_seed = (cfg.seed + 99999) if cfg.seed is not None else None
    _, info_0 = _eval_early_objective(
        vla_model, image, instruction, L_zero, cfg,
        noise_seed=probe_seed, gate_disabled=True,
    )
    _, info_p = _eval_early_objective(
        vla_model, image, instruction, L_pert, cfg,
        noise_seed=probe_seed, gate_disabled=True,
    )

    llm_diff = abs(info_0["conf_llm"] - info_p["conf_llm"])
    act_diff = abs(info_0["conf_action"] - info_p["conf_action"])

    # Use RELATIVE threshold: delta must be >1% of base value to count as
    # responsive.  Absolute 1e-5 is too tight — early injection routes L_z
    # through the full LLM, so floating-point noise alone can produce
    # delta ~ 1e-4 without any meaningful gradient signal.
    llm_base = max(abs(info_0["conf_llm"]), 1e-8)
    llm_rel = llm_diff / llm_base
    REL_THRESHOLD = 0.01  # 1% relative change required
    disabled = llm_rel < REL_THRESHOLD

    if cfg.verbose:
        print()
        print("  " + "=" * 50)
        print("  EARLY INJECTION — conf_llm responsiveness probe")
        print("  " + "=" * 50)
        print(f"  conf_llm:    {info_0['conf_llm']:.6f} (L=0) vs "
              f"{info_p['conf_llm']:.6f} (L=pert)")
        print(f"    abs delta={llm_diff:.2e}  "
              f"rel delta={llm_rel:.4f} ({llm_rel*100:.2f}%)  "
              f"threshold={REL_THRESHOLD:.0%}")
        print(f"  conf_action: {info_0['conf_action']:.6f} (L=0) vs "
              f"{info_p['conf_action']:.6f} (L=pert) — delta={act_diff:.2e}")
        if disabled:
            print(f"  RESULT: conf_llm delta {llm_rel*100:.2f}% < {REL_THRESHOLD*100:.0f}% threshold")
            print("          Treating as INVARIANT (numerical noise, not real signal)")
            print("          gate DISABLED, w_lang stays 0.0")
        else:
            print(f"  RESULT: conf_llm delta {llm_rel*100:.2f}% >= {REL_THRESHOLD*100:.0f}% threshold")
            print("          conf_llm RESPONDS to early L_z!")
            print(f"          gate: soft mode ACTIVE (floor={cfg.llm_gate_floor})")
            print("          Phase 2: consider increasing w_lang > 0")
        print("  " + "=" * 50)
        print()

    return disabled


# ===================================================================
# Core SPSA loop (early injection)
# ===================================================================
@torch.inference_mode()
def optimize_early_spsa(
    vla_model,
    image: Image.Image,
    instruction: str,
    cfg: EarlySPSAConfig,
    init_L: Optional[torch.Tensor] = None,
    gate_disabled: bool = False,
) -> Tuple[torch.Tensor, List[dict], Optional[_RunningNormalizer]]:
    """
    Maximize composite objective w.r.t. L_z via SPSA with early injection.

    L_z is injected as: projected_patch_embeddings += L_z (broadcast)
    This flows through LLM attention → cog_tokens → DiT.

    Returns:
        L_star: Optimized latent [1, 1, cog_dim].
        history: List of per-iteration records.
        normalizer: The frozen normalizer (or None).
    """
    device = next(vla_model.parameters()).device

    if init_L is not None:
        L_z = init_L.clone().to(device=device, dtype=torch.float32)
    else:
        L_z = torch.zeros(1, 1, cfg.cog_dim, device=device, dtype=torch.float32)

    if cfg.seed is not None:
        rng = torch.Generator(device=device).manual_seed(cfg.seed)
    else:
        rng = None
        if cfg.verbose:
            print("  [warn] cfg.seed is None — paired SPSA evals will have "
                  "different diffusion noise.")

    history: List[dict] = []
    prev_actions = None
    normalizer = _RunningNormalizer(warmup=4) if cfg.normalize_objectives else None
    _normalizer_freeze_logged = False

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

        noise_seed_k = (cfg.seed + k) if cfg.seed is not None else None

        J_p, info_p = _eval_early_objective(
            vla_model, image, instruction, L_plus, cfg,
            noise_seed=noise_seed_k, gate_disabled=gate_disabled,
        )
        J_m, info_m = _eval_early_objective(
            vla_model, image, instruction, L_minus, cfg,
            noise_seed=noise_seed_k, gate_disabled=gate_disabled,
        )

        # Scale normalization
        if normalizer is not None:
            normalizer.update(info_p["conf_llm"], info_p["conf_action"])
            normalizer.update(info_m["conf_llm"], info_m["conf_action"])
            if normalizer.ready and not _normalizer_freeze_logged:
                if cfg.verbose:
                    print(f"  [info] normalizer frozen at iter {k+1}: "
                          f"llm μ={normalizer._frozen_llm_mean:.4f} "
                          f"σ={normalizer._frozen_llm_std:.4f}, "
                          f"act μ={normalizer._frozen_act_mean:.4f} "
                          f"σ={normalizer._frozen_act_std:.4f}")
                _normalizer_freeze_logged = True
            if normalizer.ready:
                nlm_p, nac_p = normalizer.normalize(
                    info_p["conf_llm"], info_p["conf_action"])
                nlm_m, nac_m = normalizer.normalize(
                    info_m["conf_llm"], info_m["conf_action"])
                J_p = info_p["gate"] * (cfg.w_lang * nlm_p + cfg.w_act * nac_p)
                J_m = info_m["gate"] * (cfg.w_lang * nlm_m + cfg.w_act * nac_m)

        # SPSA gradient estimate (ascent → maximize J)
        ghat = ((J_p - J_m) / (2.0 * ck)) * delta

        L_z = L_z + ak * ghat

        # Stability: clip L_z norm
        L_norm = float(L_z.norm().item())
        if L_norm > cfg.max_L_norm:
            L_z = L_z * (cfg.max_L_norm / L_norm)
            L_norm = cfg.max_L_norm

        # Action jerk monitoring
        cur_actions = 0.5 * (info_p["norm_actions"] + info_m["norm_actions"])
        jerk = (float(np.abs(cur_actions - prev_actions).mean())
                if prev_actions is not None else 0.0)
        prev_actions = cur_actions

        J_cur = 0.5 * (J_p + J_m)
        raw_llm = 0.5 * (info_p["conf_llm"] + info_m["conf_llm"])
        raw_act = 0.5 * (info_p["conf_action"] + info_m["conf_action"])
        raw_traj = 0.5 * (info_p["conf_traj"] + info_m["conf_traj"])
        raw_smooth = 0.5 * (info_p["conf_smooth"] + info_m["conf_smooth"])
        avg_gate = 0.5 * (info_p["gate"] + info_m["gate"])
        record = {
            "iter": k + 1,
            "J": J_cur,
            "J_p": J_p,
            "J_m": J_m,
            "conf_llm": raw_llm,
            "conf_action": raw_act,
            "conf_traj": raw_traj,
            "conf_smooth": raw_smooth,
            "gate": avg_gate,
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
                f"act={raw_act:.4f} (traj={raw_traj:.3f} smooth={raw_smooth:.3f})  "
                f"gate={avg_gate:.2f}  "
                f"|L|={L_norm:.2f}  "
                f"jerk(norm)={jerk:.4f}"
            )

    return L_z.detach(), history, normalizer


# ===================================================================
# Phase A: Calibration sweep (early injection)
# ===================================================================
@torch.inference_mode()
def calibrate_early_spsa(
    vla_model,
    image: Image.Image,
    instruction: str,
    cfg: EarlySPSAConfig,
    gate_disabled: bool = False,
) -> dict:
    """Phase A: Short SPSA sweeps over (cfg_scale, c, a) to find best config."""
    print("=" * 60)
    print("  Phase A: Calibration Sweep (EARLY injection)")
    print("=" * 60)

    device = next(vla_model.parameters()).device
    L_zero = torch.zeros(1, 1, cfg.cog_dim, device=device, dtype=torch.float32)
    J_base, info_base = _eval_early_objective(
        vla_model, image, instruction, L_zero, cfg,
        gate_disabled=gate_disabled,
    )
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
        cal_cfg.normalize_objectives = False

        print(f"  [{idx+1}/{total}] cfg={cfg_s:.1f}  c={cal_cfg.c:.3f} (x{c_mult})  "
              f"a={cal_cfg.a:.3f} (x{a_mult}) ... ", end="", flush=True)

        J_init, _ = _eval_early_objective(
            vla_model, image, instruction, L_zero, cal_cfg,
            gate_disabled=gate_disabled,
        )

        L_z, history, _ = optimize_early_spsa(
            vla_model, image, instruction, cal_cfg,
            gate_disabled=gate_disabled,
        )

        J_end = history[-1]["J"]
        improvement = J_end - J_init
        L_norm = history[-1]["L_norm"]
        max_jerk = max(h["jerk"] for h in history[1:]) if len(history) > 1 else 0.0
        stable = L_norm < cfg.max_L_norm * 0.9

        J_vals = [h["J"] for h in history]
        mid = len(J_vals) // 2
        trend_up = np.mean(J_vals[mid:]) > np.mean(J_vals[:mid])

        end_act = history[-1].get("conf_action", 0.0) if history else 0.0
        end_llm = history[-1].get("conf_llm", 0.0) if history else 0.0

        result = {
            "cfg_scale": cfg_s,
            "c": cal_cfg.c,
            "a": cal_cfg.a,
            "c_mult": c_mult,
            "a_mult": a_mult,
            "J_init": J_init,
            "J_end": J_end,
            "improvement": improvement,
            "L_norm": L_norm,
            "max_jerk": max_jerk,
            "end_conf_action": end_act,
            "end_conf_llm": end_llm,
            "stable": stable,
            "trend_up": trend_up,
            "history": history,
        }
        all_results.append(result)

        status = "OK" if stable else "UNSTABLE"
        trend = "UP" if trend_up else "flat/down"
        print(f"J: {J_init:.4f} -> {J_end:.4f} ({improvement:+.4f})  "
              f"|L|={L_norm:.1f}  jerk={max_jerk:.4f}  "
              f"llm={end_llm:.4f}  [{status}, {trend}]")

    # Select best (same Pareto logic as late injection)
    stable_results = [r for r in all_results if r["stable"]]
    if not stable_results:
        print("\n  WARNING: No stable configurations found. Using least unstable.")
        stable_results = sorted(all_results, key=lambda r: r["L_norm"])[:3]

    def _pareto_score(r: dict) -> Tuple[float, float]:
        improvements = [s["improvement"] for s in stable_results]
        jerks = [s["max_jerk"] for s in stable_results]
        acts = [s["end_conf_action"] for s in stable_results]

        imp_range = max(improvements) - min(improvements) if len(improvements) > 1 else 1.0
        jerk_range = max(jerks) - min(jerks) if len(jerks) > 1 else 1.0
        act_range = max(acts) - min(acts) if len(acts) > 1 else 1.0

        norm_imp = (r["improvement"] - min(improvements)) / max(imp_range, 1e-8)
        norm_jerk = 1.0 - (r["max_jerk"] - min(jerks)) / max(jerk_range, 1e-8)
        norm_act = (r["end_conf_action"] - min(acts)) / max(act_range, 1e-8)

        score = 0.5 * norm_imp + 0.2 * norm_jerk + 0.3 * norm_act
        return (score, r["improvement"])

    best = max(stable_results, key=_pareto_score)

    print()
    print(f"  Best: cfg={best['cfg_scale']:.1f}  "
          f"c={best['c']:.3f} (x{best['c_mult']})  "
          f"a={best['a']:.3f} (x{best['a_mult']})")
    print(f"    J: {best['J_init']:.4f} -> {best['J_end']:.4f} "
          f"(improvement={best['improvement']:+.4f})")
    print()

    return {
        "best": best,
        "all_results": all_results,
        "baseline_J": J_base,
        "baseline_info": info_base,
    }


# ===================================================================
# Phase B: Full optimization (early injection)
# ===================================================================
@torch.inference_mode()
def run_phase_b_early(
    vla_model,
    image: Image.Image,
    instruction: str,
    cfg: EarlySPSAConfig,
    cal_result: dict,
    gate_disabled: bool = False,
) -> dict:
    """Phase B: Full SPSA optimization using best params from Phase A."""
    best = cal_result["best"]

    run_cfg = copy.copy(cfg)
    run_cfg.cfg_scale = best["cfg_scale"]
    run_cfg.c = best["c"]
    run_cfg.a = best["a"]

    print("=" * 60)
    print("  Phase B: Full Optimization (EARLY injection)")
    print("=" * 60)
    print(f"  cfg_scale={run_cfg.cfg_scale:.1f}  c={run_cfg.c:.3f}  a={run_cfg.a:.3f}")
    print(f"  num_iters={run_cfg.num_iters}  w_lang={run_cfg.w_lang}  w_act={run_cfg.w_act}")
    print(f"  max_L_norm={run_cfg.max_L_norm}  injection={run_cfg.injection_target}")
    print()

    device = next(vla_model.parameters()).device
    L_init = torch.zeros(1, 1, run_cfg.cog_dim, device=device, dtype=torch.float32)
    J_init, info_init = _eval_early_objective(
        vla_model, image, instruction, L_init, run_cfg,
        gate_disabled=gate_disabled,
    )
    print(f"  J_init (L=0): {J_init:.4f}  "
          f"(llm={info_init['conf_llm']:.4f}, act={info_init['conf_action']:.4f})")

    L_star, history, opt_normalizer = optimize_early_spsa(
        vla_model, image, instruction, run_cfg,
        gate_disabled=gate_disabled,
    )

    # Final evaluation (raw scale)
    J_final, info_final = _eval_early_objective(
        vla_model, image, instruction, L_star, run_cfg,
        gate_disabled=gate_disabled,
    )

    J_end_inloop = history[-1]["J"]

    print()
    print("  " + "-" * 44)
    print(f"  J (raw): {J_init:.4f} -> {J_final:.4f}  "
          f"(improvement={J_final - J_init:+.4f})")
    print(f"  J (in-loop last): {J_end_inloop:.4f}"
          f"{'  [normalized]' if opt_normalizer is not None and opt_normalizer.ready else ''}")
    print(f"  conf_llm:    {info_init['conf_llm']:.4f} -> {info_final['conf_llm']:.4f}")
    print(f"  conf_action: {info_init['conf_action']:.4f} -> {info_final['conf_action']:.4f}")
    print(f"    traj:   {info_final['conf_traj']:.4f}  smooth: {info_final['conf_smooth']:.4f}")
    print(f"  gate:        {info_final['gate']:.3f}")
    print(f"  |L_z|: {history[-1]['L_norm']:.2f}")

    # Sanity check: J_end vs J_final
    if opt_normalizer is not None and opt_normalizer.ready:
        nlm_f, nac_f = opt_normalizer.normalize(
            info_final["conf_llm"], info_final["conf_action"]
        )
        J_final_norm = info_final["gate"] * (
            run_cfg.w_lang * nlm_f + run_cfg.w_act * nac_f
        )
        j_divergence = abs(J_end_inloop - J_final_norm)
        if j_divergence > 0.1 * max(abs(J_end_inloop), abs(J_final_norm), 1e-6):
            print(f"  WARNING: J_end_norm ({J_end_inloop:.4f}) vs J_final_norm "
                  f"({J_final_norm:.4f}) diverge by {j_divergence:.4f}")
    else:
        j_divergence = abs(J_end_inloop - J_final)
        if j_divergence > 0.1 * max(abs(J_end_inloop), abs(J_final), 1e-6):
            print(f"  WARNING: J_end ({J_end_inloop:.4f}) vs J_final ({J_final:.4f}) "
                  f"diverge by {j_divergence:.4f}.")
    print()

    return {
        "L_star": L_star,
        "history": history,
        "final_J": J_final,
        "final_info": info_final,
        "final_actions": info_final["actions"],
        "final_norm_actions": info_final["norm_actions"],
        "baseline_norm_actions": info_init["norm_actions"],
        "run_cfg": run_cfg,
    }


# ===================================================================
# Combined: Phase A + Phase B (early injection)
# ===================================================================
@torch.inference_mode()
def run_early_spsa_full(
    vla_model,
    image: Image.Image,
    instruction: str,
    cfg: EarlySPSAConfig,
) -> dict:
    """
    Full pipeline: gate probe → Phase A → Phase B for early injection.

    Usage:
        cfg = EarlySPSAConfig(unnorm_key="libero_spatial_no_noops")
        result = run_early_spsa_full(vla, image, instruction, cfg)
        print(result["final_actions"])
        print(result["best_params"])
        print(result["gate_probe"])  # "responsive" or "invariant"
    """
    # Resolve gate mode — THE critical Phase 1 experiment
    gate_off = _resolve_gate_mode_early(vla_model, image, instruction, cfg)

    # Phase A
    cal = calibrate_early_spsa(vla_model, image, instruction, cfg,
                               gate_disabled=gate_off)

    # Phase B
    phase_b = run_phase_b_early(vla_model, image, instruction, cfg, cal,
                                gate_disabled=gate_off)

    # Final summary
    gate_status = "invariant" if gate_off else "responsive"
    print("=" * 60)
    print("  SUMMARY (EARLY INJECTION)")
    print("=" * 60)
    print(f"  Injection    : {cfg.injection_target} (before LLM)")
    print(f"  conf_llm gate: {gate_status}")
    print(f"  Baseline J   : {cal['baseline_J']:.4f}")
    print(f"  Final J      : {phase_b['final_J']:.4f}")
    print(f"  Improvement  : {phase_b['final_J'] - cal['baseline_J']:+.4f}")
    print(f"  Best params  : cfg={cal['best']['cfg_scale']:.1f}  "
          f"c={cal['best']['c']:.3f}  a={cal['best']['a']:.3f}")
    print(f"  |L_z|        : {float(phase_b['L_star'].norm()):.2f}")

    # Phase 2 guidance
    if not gate_off:
        print()
        print("  >>> conf_llm is RESPONSIVE to early L_z!")
        print("  >>> Phase 2: re-run with w_lang=0.3 to exploit LLM signal.")
        print("  >>>   cfg = EarlySPSAConfig(w_lang=0.3, w_act=0.7, ...)")
    else:
        print()
        print("  >>> conf_llm is invariant — same as late injection.")
        print("  >>> Early injection provides no LLM-level advantage.")
    print()

    return {
        "calibration": cal,
        "phase_b": phase_b,
        "L_star": phase_b["L_star"],
        "final_actions": phase_b["final_actions"],
        "final_norm_actions": phase_b["final_norm_actions"],
        "baseline_norm_actions": phase_b["baseline_norm_actions"],
        "final_J": phase_b["final_J"],
        "baseline_J": cal["baseline_J"],
        "best_params": {
            "cfg_scale": cal["best"]["cfg_scale"],
            "c": cal["best"]["c"],
            "a": cal["best"]["a"],
        },
        "gate_probe": gate_status,
        "gate_disabled": gate_off,
    }


# ===================================================================
# Utility: persistent early injection for deployment
# ===================================================================
def set_early_z_latent(vla_model, L_z: torch.Tensor, cog_dim: int = 4096):
    """
    Permanently hook projector to add L_z for all future predict_action calls.

    Uses register_forward_hook (cleaner than monkey-patching).
    Call clear_early_z_latent() to remove.

    Wrapper-safe: unwraps DDP/FSDP before accessing projector and storing
    state, so _eval_early_objective's guard always finds the hook.

    Args:
        vla_model: MemoryVLA (possibly DDP-wrapped).
        L_z: [1, 1, cog_dim] latent. None to just clear.
        cog_dim: Expected last dimension (default 4096). Used for validation.

    Usage after optimization:
        set_early_z_latent(vla, result["L_star"])
        actions, _ = vla.predict_action(image, instruction, ...)
    """
    # Remove existing hook if any
    clear_early_z_latent(vla_model)

    if L_z is None:
        return

    _validate_L_z_shape(L_z, cog_dim)

    raw_model = _unwrap_model(vla_model)
    _L = L_z.detach().clone()
    projector = _get_projector(raw_model)
    handle = projector.register_forward_hook(_make_projector_hook(_L))

    # Store on raw (unwrapped) model — _eval_early_objective checks raw_model
    raw_model._early_z_hook_handle = handle
    raw_model._early_z_latent = _L


def clear_early_z_latent(vla_model):
    """Remove persistent early injection hook (wrapper-safe)."""
    raw_model = _unwrap_model(vla_model)
    if hasattr(raw_model, "_early_z_hook_handle"):
        raw_model._early_z_hook_handle.remove()
        del raw_model._early_z_hook_handle
    if hasattr(raw_model, "_early_z_latent"):
        del raw_model._early_z_latent


# ===================================================================
# Checkpoint: save / load early L* result
# ===================================================================
def save_early_result(result: dict, path: str) -> None:
    """Save run_early_spsa_full() result to a .pt checkpoint.

    Usage:
        result = run_early_spsa_full(vla, image, instruction, cfg)
        save_early_result(result, "L_star_early_libero.pt")
    """
    import os
    payload = {
        "injection_type": "early",
        "L_star": result["L_star"].detach().cpu(),
        "final_J": result["final_J"],
        "baseline_J": result["baseline_J"],
        "best_params": result["best_params"],
        "gate_probe": result["gate_probe"],
        "gate_disabled": result["gate_disabled"],
        "final_actions": result["final_actions"],
        "final_norm_actions": result["final_norm_actions"],
        "baseline_norm_actions": result["baseline_norm_actions"],
        "history": result["phase_b"]["history"],
        "final_info": result["phase_b"]["final_info"],
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)
    norm = float(payload["L_star"].norm())
    print(f"  Saved early L* checkpoint -> {path}")
    print(f"    |L*|={norm:.2f}  J={result['final_J']:.4f}  "
          f"(baseline={result['baseline_J']:.4f})  "
          f"gate={result['gate_probe']}")


def load_early_result(path: str, device: str = "cuda") -> dict:
    """Load a saved early L* checkpoint.

    Usage:
        ckpt = load_early_result("L_star_early_libero.pt")
        set_early_z_latent(vla, ckpt["L_star"])
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ckpt["L_star"] = ckpt["L_star"].to(device)
    norm = float(ckpt["L_star"].norm())
    print(f"  Loaded early L* checkpoint <- {path}")
    print(f"    |L*|={norm:.2f}  J={ckpt['final_J']:.4f}  "
          f"(baseline={ckpt['baseline_J']:.4f})  "
          f"gate={ckpt.get('gate_probe', 'unknown')}")
    return ckpt


# ===================================================================
# A/B comparison helper
# ===================================================================
def compare_early_vs_late(early_result: dict, late_result: dict) -> dict:
    """Print side-by-side comparison of early vs late injection results.

    Args:
        early_result: Output from run_early_spsa_full().
        late_result: Output from run_z_spsa_full() (or load_z_result()).

    Returns:
        dict with comparison metrics.
    """
    e = early_result
    l = late_result

    e_J = e["final_J"]
    l_J = l["final_J"]
    e_base = e["baseline_J"]
    l_base = l["baseline_J"]
    e_imp = e_J - e_base
    l_imp = l_J - l_base

    e_info = e["phase_b"]["final_info"]
    l_info = l.get("phase_b", {}).get("final_info", {})

    # Late injection might come from load_z_result (no phase_b)
    if not l_info:
        l_llm_final = l.get("conf_llm_final", "N/A")
        l_act_final = l.get("conf_action_final", "N/A")
    else:
        l_llm_final = l_info.get("conf_llm", "N/A")
        l_act_final = l_info.get("conf_action", "N/A")

    print("=" * 65)
    print("  EARLY vs LATE INJECTION — A/B Comparison")
    print("=" * 65)
    print(f"  {'Metric':<25} {'Late (after LLM)':>18} {'Early (before LLM)':>18}")
    print(f"  {'-'*25} {'-'*18} {'-'*18}")
    print(f"  {'Baseline J':<25} {l_base:>18.4f} {e_base:>18.4f}")
    print(f"  {'Final J':<25} {l_J:>18.4f} {e_J:>18.4f}")
    print(f"  {'Improvement':<25} {l_imp:>+18.4f} {e_imp:>+18.4f}")
    print(f"  {'|L_z|':<25} {float(l['L_star'].norm()):>18.2f} "
          f"{float(e['L_star'].norm()):>18.2f}")

    if isinstance(l_llm_final, float):
        print(f"  {'conf_llm (final)':<25} {l_llm_final:>18.4f} "
              f"{e_info['conf_llm']:>18.4f}")
    if isinstance(l_act_final, float):
        print(f"  {'conf_action (final)':<25} {l_act_final:>18.4f} "
              f"{e_info['conf_action']:>18.4f}")

    gate_probe = e.get("gate_probe", "unknown")
    print(f"  {'conf_llm responsive?':<25} {'NO (invariant)':>18} "
          f"{gate_probe.upper():>18}")
    print("=" * 65)

    winner = "early" if e_imp > l_imp else "late" if l_imp > e_imp else "tie"
    margin = abs(e_imp - l_imp)
    print(f"  Winner: {winner} injection (margin={margin:+.4f})")
    print()

    return {
        "early_improvement": e_imp,
        "late_improvement": l_imp,
        "winner": winner,
        "margin": margin,
        "gate_probe": gate_probe,
    }


# ===================================================================
# CLI
# ===================================================================
if __name__ == "__main__":
    print("Phase 1: Early Injection SPSA for MemoryVLA")
    print()
    print("Usage (Colab):")
    print("  from scripts.colab_spsa_early_injection import (")
    print("      EarlySPSAConfig, run_early_spsa_full,")
    print("      set_early_z_latent, clear_early_z_latent,")
    print("      compare_early_vs_late,")
    print("  )")
    print("  from scripts.colab_spsa_z_pathway import (")
    print("      plot_input_frame, plot_optimization_curves, plot_action_trajectory,")
    print("      render_action_rollout, closed_loop_rollout_multichunk,")
    print("      save_z_result, load_z_result,")
    print("  )")
    print()
    print('  cfg = EarlySPSAConfig(unnorm_key="libero_spatial_no_noops")')
    print("  result = run_early_spsa_full(vla, image, instruction, cfg)")
    print()
    print("  # Visualize:")
    print('  plot_optimization_curves(result["phase_b"]["history"],')
    print('      title="SPSA Early Injection")')
    print('  plot_action_trajectory(result["final_norm_actions"],')
    print('      baseline_actions=result["baseline_norm_actions"])')
    print()
    print("  # Deploy L*:")
    print('  set_early_z_latent(vla, result["L_star"])')
    print("  actions, _ = vla.predict_action(image, instruction, ...)")
    print()
    print("  # A/B comparison with late injection baseline:")
    print('  late_ckpt = load_z_result("L_star_late.pt")')
    print("  compare_early_vs_late(result, late_ckpt)")
    print()
    print("  # If conf_llm responds (Phase 2):")
    print('  cfg2 = EarlySPSAConfig(w_lang=0.3, w_act=0.7, unnorm_key="...")')
    print("  result2 = run_early_spsa_full(vla, image, instruction, cfg2)")
