"""
SPSA optimization on the z (cog_tokens) pathway for MemoryVLA.

Why z pathway instead of per_token pathway:
  - per_attn: DEAD (zero-init, in_proj ≈ 0.0003)
  - z pathway: WEAK but ALIVE (content-sensitive, linear dose-response)
  - The old colab_spsa_latent_opt.py injected L into per_tokens → ineffective
  - This script injects L_z into cog_tokens → z = cog_tokens + L_z

Key improvements over the old SPSA:
  1. Targets z pathway (cog_tokens) instead of dead per_token pathway
  2. Composite objective: J = (gate * w_lang * conf_llm) + (w_act * conf_action)
     - conf_llm: LLM token probability (INVARIANT to L_z — see gate_mode)
     - conf_action: trajectory consistency (action-level, zero extra cost)
     NOTE: L_z injects after LLM generate, so conf_llm cannot respond to L_z.
     Default: w_lang=0, w_act=1, gate_mode='auto' (auto-detects and disables gate).
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
        plot_input_frame, plot_optimization_curves, plot_action_trajectory,
        render_action_rollout, closed_loop_rollout_multichunk,
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

    # Closed-loop with base_camera (front view, LIBERO-aligned):
    #   Create env with 224x224 sensor resolution to match model input:
    #     env = gym.make("PickCube-v1", obs_mode="rgbd", render_mode="rgb_array",
    #                    sensor_configs=dict(base_camera=dict(width=224, height=224)))
    #   frames, logs, env = closed_loop_rollout_multichunk(
    #       env, vla, instruction, cfg,
    #       use_base_camera=True,   # front-view obs camera (default)
    #   )
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import random
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
    w_lang: float = 0.0
    """Weight for LLM confidence in composite J.
    Default 0.0 because L_z injects AFTER LLM token generation:
      LLM generate → conf_llm computed → cog_tokens extracted → L_z added
    So conf_llm is structurally invariant to L_z (zero gradient signal).
    Only increase if architecture changes to inject L_z before LLM."""
    w_act: float = 1.0
    """Weight for action confidence in composite J."""
    w_traj: float = 0.7
    """Sub-weight for trajectory certainty within conf_action."""
    w_smooth: float = 0.3
    """Sub-weight for intra-trajectory smoothness within conf_action.
    Smoothness = how smooth the predicted future actions are (no jitter)."""
    llm_gate_floor: float = 0.3
    """Soft gate floor (only used when gate_mode='soft').
    If conf_llm < this, J is penalized proportionally."""
    gate_mode: str = "auto"
    """How to apply the conf_llm gate:
      'auto': Check conf_llm variance during warmup. If σ ≈ 0 (invariant
              to L_z), disable gate and warn. Otherwise use soft gate.
      'soft': Always apply gate = min(1, conf_llm / llm_gate_floor).
      'disabled': Gate is always 1.0 (no penalty from conf_llm)."""
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
    smooth_tau: float = 0.05
    """Temperature for smoothness sigmoid: conf = 1/(1 + mean_jerk/tau).
    Lower = more sensitive to action jitter."""

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

    def __post_init__(self):
        wsum = self.w_traj + self.w_smooth
        if abs(wsum - 1.0) > 0.01:
            raise ValueError(
                f"w_traj ({self.w_traj}) + w_smooth ({self.w_smooth}) = {wsum}, "
                f"expected ~1.0. conf_action scale will be distorted."
            )
        if self.gate_mode not in ("auto", "soft", "disabled"):
            raise ValueError(
                f"gate_mode must be 'auto', 'soft', or 'disabled', "
                f"got '{self.gate_mode}'"
            )


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

    Collects statistics during warmup, then FREEZES them so the SPSA
    objective landscape stays stationary. This prevents the normalizer
    itself from shifting the gradient signal that SPSA is trying to follow.
    """

    def __init__(self, warmup: int = 4):
        self.warmup = warmup
        self._llm_vals: List[float] = []
        self._act_vals: List[float] = []
        self._frozen = False
        self._frozen_llm_mean: float = 0.0
        self._frozen_llm_std: float = 1.0
        self._frozen_act_mean: float = 0.0
        self._frozen_act_std: float = 1.0
        self._llm_is_constant: bool = False

    @property
    def ready(self) -> bool:
        return self._frozen or len(self._llm_vals) >= self.warmup

    def update(self, conf_llm: float, conf_action: float):
        if self._frozen:
            return
        self._llm_vals.append(conf_llm)
        self._act_vals.append(conf_action)
        # Freeze once warmup is reached
        if len(self._llm_vals) >= self.warmup and not self._frozen:
            self._frozen_llm_mean = float(np.mean(self._llm_vals))
            raw_llm_std = float(np.std(self._llm_vals))
            self._llm_is_constant = raw_llm_std < 1e-5
            self._frozen_llm_std = max(raw_llm_std, 1e-6)
            self._frozen_act_mean = float(np.mean(self._act_vals))
            self._frozen_act_std = max(float(np.std(self._act_vals)), 1e-6)
            self._frozen = True

    def normalize(self, conf_llm: float, conf_action: float) -> Tuple[float, float]:
        """Return normalized scores in ~[0, 1] with comparable spread."""
        if not self.ready:
            return conf_llm, conf_action

        # z-score → sigmoid → [0, 1]  (using frozen statistics)
        def _sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        # When llm σ ≈ 0 (constant), z-score would be extreme → use neutral 0.5.
        # This is safe because w_lang=0 means nlm has no effect, and if user
        # sets w_lang>0 with constant llm, the neutral value avoids instability.
        if self._llm_is_constant:
            nlm = 0.5
        else:
            nlm = _sigmoid((conf_llm - self._frozen_llm_mean) / self._frozen_llm_std)
        nac = _sigmoid((conf_action - self._frozen_act_mean) / self._frozen_act_std)
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


def _compute_smoothness(norm_actions: np.ndarray, tau: float) -> float:
    """Compute smoothness confidence from inter-step jitter in action sequence.

    norm_actions: [T, D] or [1, T, D] — the predicted action chunk.
    Returns scalar in (0, 1]: conf = 1 / (1 + mean_jerk / tau).
    If T <= 1, returns 1.0 (single-step is trivially smooth).
    """
    a = np.asarray(norm_actions)
    if a.ndim == 3:
        a = a[0]  # [T, D]
    if a.shape[0] <= 1:
        return 1.0
    # First-order finite differences → "jerk" = mean abs delta
    diffs = np.diff(a, axis=0)  # [T-1, D]
    mean_jerk = float(np.abs(diffs).mean())
    return 1.0 / (1.0 + mean_jerk / tau)


@torch.inference_mode()
def _eval_z_objective(
    vla_model,
    image: Image.Image,
    instruction: str,
    L_z: torch.Tensor,
    cfg: ZSPSAConfig,
    noise_seed: Optional[int] = None,
    gate_disabled: bool = False,
) -> Tuple[float, dict]:
    """
    Evaluate composite objective with L_z injected into z pathway.

    J = (gate * w_lang * conf_llm) + (w_act * conf_action)

    where:
      conf_action = w_traj * trajectory_certainty + w_smooth * smoothness
      gate = min(1.0, conf_llm / llm_gate_floor)  — soft penalty when
             the model doesn't understand the instruction at all.

    Args:
        noise_seed: If set, fixes the diffusion noise via torch.manual_seed()
            before calling predict_action. This ensures paired L+/L- evals
            in SPSA see the same stochastic conditions.
        gate_disabled: If True, gate is forced to 1.0 regardless of gate_mode.
            Set by _resolve_gate_mode() when conf_llm is invariant to L_z.

    Returns:
        J: scalar (higher = better)
        info: dict with conf_llm, conf_action, actions, raw_actions, etc.
    """
    _ensure_ddim(vla_model, cfg.num_ddim_steps)

    # ── Save and fix all RNG sources for paired SPSA evaluation ──
    # Restore on exit so external notebook RNG is not corrupted.
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

    # ── Guard: temporarily remove set_z_latent global patch if active ──
    # Otherwise L_z would be added twice: once by set_z_latent, once here.
    had_global_patch = hasattr(vla_model, "_z_latent_original_process")
    if had_global_patch:
        if cfg.verbose:
            print("  [warn] set_z_latent global patch detected — "
                  "temporarily disabled for isolated L_z eval")
        global_patched = vla_model.cog_mem_bank.process_batch
        vla_model.cog_mem_bank.process_batch = vla_model._z_latent_original_process

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

        # -- Sub-metrics for conf_action --
        raw_actions = None
        if captured["pred_xstarts"]:
            # Trajectory certainty (from pred_xstart variance across steps)
            traj = _compute_trajectory_confidence(
                captured["pred_xstarts"],
                using_cfg=(cfg.cfg_scale > 1.0),
                tail_fraction=cfg.traj_tail_fraction,
                tau=cfg.traj_tau,
            )
            conf_traj = traj["action_confidence"]
            traj_var = traj["mean_variance"]

            # Raw actions: final pred_xstart before clip/binarize
            raw_pred = captured["pred_xstarts"][-1]
            if cfg.cfg_scale > 1.0 and raw_pred.shape[0] > 1:
                raw_pred = raw_pred[: raw_pred.shape[0] // 2]
            raw_actions = raw_pred.cpu().numpy()
        else:
            conf_traj = conf_llm
            traj_var = 0.0

        # Smoothness (from action sequence inter-step jitter)
        # Prefer raw_actions (pre-clip/binarize) to avoid binarization distortion,
        # but clamp to [-1, 1] so smooth_tau is on the same scale as norm_actions.
        if raw_actions is not None:
            smooth_source = np.clip(raw_actions, -1.0, 1.0)
        else:
            smooth_source = norm_actions
        conf_smooth = _compute_smoothness(smooth_source, cfg.smooth_tau)

        # Combine sub-metrics
        conf_action = cfg.w_traj * conf_traj + cfg.w_smooth * conf_smooth

        # -- Gate: penalize when conf_llm is too low --
        # gate_disabled is resolved once by _resolve_gate_mode() and passed
        # explicitly — no dynamic attributes on cfg.
        if gate_disabled or cfg.gate_mode == "disabled":
            gate = 1.0
        else:  # "soft" or "auto" (pre-resolved to not-disabled)
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
        vla_model.cog_mem_bank.process_batch = original_process
        # Restore set_z_latent global patch if it was active before this call
        if had_global_patch:
            vla_model.cog_mem_bank.process_batch = global_patched
        # Restore RNG states so external code is not affected
        if _rng_states_saved is not None:
            torch.random.set_rng_state(_rng_states_saved["torch_cpu"])
            for d, state in enumerate(_rng_states_saved["torch_cuda"]):
                torch.cuda.set_rng_state(state, d)
            np.random.set_state(_rng_states_saved["numpy"])
            random.setstate(_rng_states_saved["python"])


# ===================================================================
# Gate mode resolution
# ===================================================================
@torch.inference_mode()
def _resolve_gate_mode(
    vla_model,
    image: Image.Image,
    instruction: str,
    cfg: ZSPSAConfig,
) -> bool:
    """Probe whether conf_llm responds to L_z changes.

    Runs 2 evals (L=0 vs L=perturbation) with the same noise seed.
    If conf_llm is identical → invariant to L_z → gate should be disabled.

    This is called ONCE before optimization so that J_init, J_end, and
    J_final all use the same gate state (fixes scale mismatch).

    Returns:
        True if gate should be disabled (conf_llm invariant to L_z).
    """
    if cfg.gate_mode == "disabled":
        return True
    if cfg.gate_mode == "soft":
        return False

    # "auto": probe with L=0 vs L=random perturbation
    device = next(vla_model.parameters()).device
    L_zero = torch.zeros(1, 1, cfg.cog_dim, device=device, dtype=torch.float32)
    L_pert = torch.randn(1, 1, cfg.cog_dim, device=device) * cfg.c

    probe_seed = (cfg.seed + 99999) if cfg.seed is not None else None
    _, info_0 = _eval_z_objective(
        vla_model, image, instruction, L_zero, cfg,
        noise_seed=probe_seed, gate_disabled=True,
    )
    _, info_p = _eval_z_objective(
        vla_model, image, instruction, L_pert, cfg,
        noise_seed=probe_seed, gate_disabled=True,
    )

    llm_diff = abs(info_0["conf_llm"] - info_p["conf_llm"])
    disabled = llm_diff < 1e-5

    if cfg.verbose:
        if disabled:
            print(f"  [auto-gate] conf_llm: {info_0['conf_llm']:.6f} (L=0) vs "
                  f"{info_p['conf_llm']:.6f} (L=pert) — diff={llm_diff:.2e} ≈ 0")
            print(f"  [auto-gate] conf_llm is invariant to L_z → gate DISABLED")
        else:
            print(f"  [auto-gate] conf_llm: {info_0['conf_llm']:.6f} (L=0) vs "
                  f"{info_p['conf_llm']:.6f} (L=pert) — diff={llm_diff:.4f}")
            print(f"  [auto-gate] conf_llm responds to L_z → soft gate ACTIVE")

    return disabled


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
    gate_disabled: bool = False,
) -> Tuple[torch.Tensor, List[dict], Optional[_RunningNormalizer]]:
    """
    Maximize composite objective w.r.t. L_z via SPSA on z pathway.

    L_z is injected as: cog_tokens_out = cog_mem_bank(cog_tokens) + L_z

    Args:
        vla_model: Loaded MemoryVLA (eval mode, on GPU).
        image: PIL observation image.
        instruction: Language instruction.
        cfg: SPSA configuration.
        init_L: Optional warm-start latent [1, 1, cog_dim].
        gate_disabled: Pre-resolved gate state from _resolve_gate_mode().

    Returns:
        L_star: Optimized latent [1, 1, cog_dim].
        history: List of per-iteration records.
        normalizer: The frozen normalizer (or None if normalization disabled).
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
        if cfg.verbose:
            print("  [warn] cfg.seed is None — paired L+/L- evals will use "
                  "different diffusion noise. Set seed for reproducible gradients.")

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

        # Same noise_seed for L+ and L-: ensures paired eval sees identical
        # diffusion noise, so gradient estimate reflects only L_z difference.
        # Critical for WEAK pathway where signal is small relative to noise.
        noise_seed_k = (cfg.seed + k) if cfg.seed is not None else None

        J_p, info_p = _eval_z_objective(vla_model, image, instruction, L_plus, cfg,
                                         noise_seed=noise_seed_k,
                                         gate_disabled=gate_disabled)
        J_m, info_m = _eval_z_objective(vla_model, image, instruction, L_minus, cfg,
                                         noise_seed=noise_seed_k,
                                         gate_disabled=gate_disabled)

        # Scale normalization: prevent one metric from dominating
        if normalizer is not None:
            normalizer.update(info_p["conf_llm"], info_p["conf_action"])
            normalizer.update(info_m["conf_llm"], info_m["conf_action"])
            if normalizer.ready and not _normalizer_freeze_logged:
                if cfg.verbose:
                    print(f"  [info] normalizer frozen at iter {k+1}: "
                          f"llm μ={normalizer._frozen_llm_mean:.4f} σ={normalizer._frozen_llm_std:.4f}, "
                          f"act μ={normalizer._frozen_act_mean:.4f} σ={normalizer._frozen_act_std:.4f}")
                _normalizer_freeze_logged = True
            if normalizer.ready:
                nlm_p, nac_p = normalizer.normalize(info_p["conf_llm"], info_p["conf_action"])
                nlm_m, nac_m = normalizer.normalize(info_m["conf_llm"], info_m["conf_action"])
                # Gate only dampens the lang term; action term is never gated.
                J_p = (info_p["gate"] * cfg.w_lang * nlm_p) + (cfg.w_act * nac_p)
                J_m = (info_m["gate"] * cfg.w_lang * nlm_m) + (cfg.w_act * nac_m)

        # SPSA gradient estimate (ascent → maximize J)
        ghat = ((J_p - J_m) / (2.0 * ck)) * delta

        L_z = L_z + ak * ghat

        # Stability: clip L_z norm
        L_norm = float(L_z.norm().item())
        if L_norm > cfg.max_L_norm:
            L_z = L_z * (cfg.max_L_norm / L_norm)
            L_norm = cfg.max_L_norm

        # Action jerk monitoring (use norm_actions for task-independent scale)
        cur_actions = 0.5 * (info_p["norm_actions"] + info_m["norm_actions"])
        jerk = float(np.abs(cur_actions - prev_actions).mean()) if prev_actions is not None else 0.0
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
# Phase A: Calibration sweep
# ===================================================================
@torch.inference_mode()
def calibrate_z_spsa(
    vla_model,
    image: Image.Image,
    instruction: str,
    cfg: ZSPSAConfig,
    gate_disabled: bool = False,
) -> dict:
    """
    Phase A: Short SPSA sweeps over (cfg_scale, c, a) to find best config.

    Runs cal_iters per combination, measures J improvement and stability.

    Args:
        gate_disabled: Pre-resolved gate state from _resolve_gate_mode().

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
    J_base, info_base = _eval_z_objective(
        vla_model, image, instruction, L_zero, cfg, gate_disabled=gate_disabled)
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
        # Disable normalization during calibration so J_init and J_end
        # are on the same raw scale — prevents apples-to-oranges comparison
        cal_cfg.normalize_objectives = False

        print(f"  [{idx+1}/{total}] cfg={cfg_s:.1f}  c={cal_cfg.c:.3f} (x{c_mult})  "
              f"a={cal_cfg.a:.3f} (x{a_mult}) ... ", end="", flush=True)

        # Explicit J_init at L=0 with this config's cfg_scale
        J_init, _ = _eval_z_objective(
            vla_model, image, instruction, L_zero, cal_cfg,
            gate_disabled=gate_disabled,
        )

        L_z, history, _ = optimize_z_spsa(
            vla_model, image, instruction, cal_cfg,
            gate_disabled=gate_disabled,
        )

        J_end = history[-1]["J"]
        improvement = J_end - J_init
        L_norm = history[-1]["L_norm"]
        max_jerk = max(h["jerk"] for h in history[1:]) if len(history) > 1 else 0.0
        stable = L_norm < cfg.max_L_norm * 0.9

        # J trend: monotonically improving?
        J_vals = [h["J"] for h in history]
        # Check if second half is generally better than first half
        mid = len(J_vals) // 2
        trend_up = np.mean(J_vals[mid:]) > np.mean(J_vals[:mid])

        # Conf_action at end of calibration run
        end_act = history[-1].get("conf_action", 0.0) if history else 0.0

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
            "stable": stable,
            "trend_up": trend_up,
            "history": history,
        }
        all_results.append(result)

        status = "OK" if stable else "UNSTABLE"
        trend = "UP" if trend_up else "flat/down"
        print(f"J: {J_init:.4f} -> {J_end:.4f} ({improvement:+.4f})  "
              f"|L|={L_norm:.1f}  jerk={max_jerk:.4f}  [{status}, {trend}]")

    # Select best: Pareto-style multi-criteria scoring
    # Criteria: improvement (primary), low jerk (stability), conf_action end value
    stable_results = [r for r in all_results if r["stable"]]
    if not stable_results:
        print("\n  WARNING: No stable configurations found. Using least unstable.")
        stable_results = sorted(all_results, key=lambda r: r["L_norm"])[:3]

    def _pareto_score(r: dict) -> Tuple[float, float]:
        """Multi-criteria score: balance improvement, stability, and action quality.

        Returns (score, improvement) — the second element is the tie-break:
        when Pareto scores are near-identical (range collapse), prefer higher
        raw improvement.
        """
        # Normalize each axis to [0, 1] relative to the candidate set
        improvements = [s["improvement"] for s in stable_results]
        jerks = [s["max_jerk"] for s in stable_results]
        acts = [s["end_conf_action"] for s in stable_results]

        imp_range = max(improvements) - min(improvements) if len(improvements) > 1 else 1.0
        jerk_range = max(jerks) - min(jerks) if len(jerks) > 1 else 1.0
        act_range = max(acts) - min(acts) if len(acts) > 1 else 1.0

        norm_imp = (r["improvement"] - min(improvements)) / max(imp_range, 1e-8)
        # Lower jerk is better → invert
        norm_jerk = 1.0 - (r["max_jerk"] - min(jerks)) / max(jerk_range, 1e-8)
        norm_act = (r["end_conf_action"] - min(acts)) / max(act_range, 1e-8)

        # Weighted combination: improvement matters most
        score = 0.5 * norm_imp + 0.2 * norm_jerk + 0.3 * norm_act
        # Tie-break: raw improvement (when score is near-identical due to
        # range collapse, this ensures deterministic and sensible ordering)
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
# Phase B: Full optimization with best params
# ===================================================================
@torch.inference_mode()
def run_phase_b(
    vla_model,
    image: Image.Image,
    instruction: str,
    cfg: ZSPSAConfig,
    cal_result: dict,
    gate_disabled: bool = False,
) -> dict:
    """
    Phase B: Full SPSA optimization using best params from Phase A.

    Args:
        cal_result: Output from calibrate_z_spsa().
        gate_disabled: Pre-resolved gate state from _resolve_gate_mode().

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

    # Explicit J_init at starting state (before any SPSA updates)
    # gate_disabled is pre-resolved — same state for J_init, optimize, and J_final
    device = next(vla_model.parameters()).device
    L_init = torch.zeros(1, 1, run_cfg.cog_dim, device=device, dtype=torch.float32)
    J_init, info_init = _eval_z_objective(
        vla_model, image, instruction, L_init, run_cfg,
        gate_disabled=gate_disabled,
    )
    print(f"  J_init (L=0): {J_init:.4f}  "
          f"(llm={info_init['conf_llm']:.4f}, act={info_init['conf_action']:.4f})")

    L_star, history, opt_normalizer = optimize_z_spsa(
        vla_model, image, instruction, run_cfg,
        gate_disabled=gate_disabled,
    )

    # Final evaluation (raw scale — definitive measurement)
    J_final, info_final = _eval_z_objective(
        vla_model, image, instruction, L_star, run_cfg,
        gate_disabled=gate_disabled,
    )

    # Summary: use raw J_init and J_final for definitive improvement
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

    # Sanity check: compare J_end vs J_final on SAME scale.
    # If normalizer was used, transform J_final to normalized space for comparison.
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
                  f"({J_final_norm:.4f}) diverge by {j_divergence:.4f} — "
                  f"possible stochastic instability.")
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
    # Resolve gate mode once — ensures consistent gate state across
    # all J evaluations (J_init, in-loop, J_final) in both phases.
    gate_off = _resolve_gate_mode(vla_model, image, instruction, cfg)

    # Phase A
    cal = calibrate_z_spsa(vla_model, image, instruction, cfg,
                           gate_disabled=gate_off)

    # Phase B
    phase_b = run_phase_b(vla_model, image, instruction, cfg, cal,
                          gate_disabled=gate_off)

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
        "final_norm_actions": phase_b["final_norm_actions"],
        "baseline_norm_actions": phase_b["baseline_norm_actions"],
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
# Checkpoint: save / load L* result
# ===================================================================
def save_z_result(result: dict, path: str) -> None:
    """Save run_z_spsa_full() result to a .pt file.

    Saves L_star tensor, metrics, best_params, and optimization history.
    Usage:
        result = run_z_spsa_full(vla, image, instruction, cfg)
        save_z_result(result, "L_star_libero_base.pt")
    """
    import os
    payload = {
        "L_star": result["L_star"].detach().cpu(),
        "final_J": result["final_J"],
        "baseline_J": result["baseline_J"],
        "best_params": result["best_params"],
        "final_actions": result["final_actions"],
        "final_norm_actions": result["final_norm_actions"],
        "baseline_norm_actions": result["baseline_norm_actions"],
        "history": result["phase_b"]["history"],
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)
    norm = float(payload["L_star"].norm())
    print(f"  Saved L* checkpoint -> {path}")
    print(f"    |L*|={norm:.2f}  J={result['final_J']:.4f}  "
          f"(baseline={result['baseline_J']:.4f})")


def load_z_result(path: str, device: str = "cuda") -> dict:
    """Load a saved L* checkpoint.

    Returns a dict with L_star (on device), metrics, and history.
    Usage:
        ckpt = load_z_result("L_star_libero_base.pt")
        set_z_latent(vla, ckpt["L_star"])
    """
    ckpt = torch.load(path, map_location="cpu")
    ckpt["L_star"] = ckpt["L_star"].to(device)
    norm = float(ckpt["L_star"].norm())
    print(f"  Loaded L* checkpoint <- {path}")
    print(f"    |L*|={norm:.2f}  J={ckpt['final_J']:.4f}  "
          f"(baseline={ckpt['baseline_J']:.4f})")
    return ckpt


# ===================================================================
# Visualization (Colab)
# ===================================================================
def plot_input_frame(image: Image.Image, instruction: str = ""):
    """Display the input observation image in Colab/Jupyter.

    Args:
        image: PIL observation image (robot + scene).
        instruction: Optional instruction text shown as title.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(image)
    ax.set_axis_off()
    if instruction:
        ax.set_title(instruction, fontsize=10, wrap=True)
    plt.tight_layout()
    plt.show()


def plot_optimization_curves(history: List[dict], title: str = "SPSA Optimization"):
    """Plot optimization metrics over iterations.

    Shows 4 subplots:
      1. Objective J over iterations
      2. conf_action breakdown (traj + smooth)
      3. Action jerk (stability)
      4. |L_z| norm (latent magnitude)

    Args:
        history: List of per-iteration records from optimize_z_spsa().
        title: Plot title.
    """
    import matplotlib.pyplot as plt

    if not history:
        print("  [viz] No history to plot.")
        return

    iters = [h["iter"] for h in history]
    J_vals = [h["J"] for h in history]
    act_vals = [h["conf_action"] for h in history]
    traj_vals = [h["conf_traj"] for h in history]
    smooth_vals = [h["conf_smooth"] for h in history]
    gate_vals = [h["gate"] for h in history]
    jerk_vals = [h["jerk"] for h in history]
    L_norms = [h["L_norm"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=13)

    # 1. Objective J
    ax = axes[0, 0]
    ax.plot(iters, J_vals, "b-", linewidth=1.5, label="J (composite)")
    ax.set_ylabel("J")
    ax.set_xlabel("Iteration")
    ax.set_title("Objective")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 2. conf_action breakdown
    ax = axes[0, 1]
    ax.plot(iters, act_vals, "g-", linewidth=1.5, label="conf_action")
    ax.plot(iters, traj_vals, "g--", linewidth=1, alpha=0.7, label="traj")
    ax.plot(iters, smooth_vals, "g:", linewidth=1, alpha=0.7, label="smooth")
    ax.plot(iters, gate_vals, "r-", linewidth=1, alpha=0.5, label="gate")
    ax.set_ylabel("Confidence")
    ax.set_xlabel("Iteration")
    ax.set_title("Action Confidence & Gate")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 3. Action jerk
    ax = axes[1, 0]
    ax.plot(iters, jerk_vals, "orange", linewidth=1.5)
    ax.set_ylabel("Jerk (norm-space)")
    ax.set_xlabel("Iteration")
    ax.set_title("Action Stability")
    ax.grid(True, alpha=0.3)

    # 4. |L_z| norm
    ax = axes[1, 1]
    ax.plot(iters, L_norms, "purple", linewidth=1.5)
    ax.set_ylabel("|L_z|")
    ax.set_xlabel("Iteration")
    ax.set_title("Latent Magnitude")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def render_action_rollout(
    env,
    actions: np.ndarray,
    title: str = "Action Rollout",
    reset: bool = True,
    max_steps: Optional[int] = None,
):
    """Execute predicted actions in a gym env and display as inline video.

    Works with ManiSkill3 (gymnasium) environments.
    Handles shape mismatches automatically:
      - Adds batch dimension (1, D) for ManiSkill3 vectorized envs
      - Pads/truncates action dim to match env.action_space

    Args:
        env: Gymnasium env with render_mode="rgb_array".
        actions: [T, D] or [1, T, D] array of actions to execute.
            Should be un-normalized (env-scale) actions from result["final_actions"].
        title: Title for the video.
        reset: If True, reset env before rollout.
        max_steps: Limit number of steps (default: all actions).

    Returns:
        List of RGB frames (numpy arrays) for further use.

    Usage (Colab):
        env2 = gym.make("PickCube-v1", obs_mode="rgbd", render_mode="rgb_array")
        frames = render_action_rollout(env2, result["final_actions"])
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from IPython.display import HTML, display

    a = np.asarray(actions, dtype=np.float32)
    if a.ndim == 3:
        a = a[0]  # [1, T, D] → [T, D]
    if a.ndim == 1:
        a = a.reshape(1, -1)  # [D] → [1, D]
    T, D_model = a.shape
    if max_steps is not None:
        T = min(T, max_steps)

    # Detect expected action shape from env
    act_shape = env.action_space.shape  # e.g. (8,) or (1, 8)
    D_env = act_shape[-1]
    needs_batch = len(act_shape) == 2  # ManiSkill3: (num_envs, action_dim)

    if D_model != D_env:
        print(f"  [rollout] action dim mismatch: model={D_model}, env={D_env} "
              f"— {'padding' if D_model < D_env else 'truncating'}")

    if reset:
        try:
            env.reset()
        except AttributeError as exc:
            if "_reset_mask" in str(exc):
                raise RuntimeError(
                    "ManiSkill env reset failed because scene is None. "
                    "Recreate the environment object and retry rollout. "
                    "Example: env = gym.make(...); render_action_rollout(env, actions)."
                ) from exc
            raise

    def _grab_frame():
        frame = env.render()
        if hasattr(frame, 'cpu'):
            frame = frame.cpu().numpy()
        frame = np.squeeze(frame)
        if frame.ndim == 3 and frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        return frame

    frames = [_grab_frame()]

    for t in range(T):
        action_t = a[t]  # (D_model,)

        # Pad or truncate to match env action dim
        if D_model < D_env:
            action_t = np.concatenate([action_t, np.zeros(D_env - D_model, dtype=np.float32)])
        elif D_model > D_env:
            action_t = action_t[:D_env]

        # Add batch dim for ManiSkill3 vectorized envs
        if needs_batch:
            action_t = action_t.reshape(1, -1)

        env.step(action_t)
        frames.append(_grab_frame())

    # Create animation
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_axis_off()
    im = ax.imshow(frames[0])

    def _update(i):
        im.set_data(frames[i])
        ax.set_title(f"{title} — step {i}/{len(frames)-1}", fontsize=10)
        return [im]

    anim = animation.FuncAnimation(
        fig, _update, frames=len(frames), interval=200, blit=True,
    )
    plt.close(fig)
    display(HTML(anim.to_html5_video()))

    return frames


def closed_loop_rollout_multichunk(
    env,
    vla_model,
    instruction: str,
    cfg: ZSPSAConfig,
    n_chunks: int = 4,
    steps_per_chunk: int = 16,
    title: str = "Closed-loop Multi-chunk Rollout",
    reset: bool = True,
    env_factory=None,
    use_base_camera: bool = True,
    camera_name: str = "base_camera",
):
    """Closed-loop rollout that replans actions every chunk.

    This utility is intended for Colab experiments where a single 16-step chunk
    may under-shoot final grasp depth.  It repeatedly:
      1) captures current observation,
      2) calls predict_action(...),
      3) executes a short action chunk,
      4) replans from the latest frame.

    Args:
        env: Gym/Gymnasium/ManiSkill environment.
        vla_model: MemoryVLA model instance.
        instruction: Text instruction.
        cfg: ZSPSAConfig used for unnorm_key / cfg_scale / DDIM args.
        n_chunks: Number of replan chunks.
        steps_per_chunk: Number of executed steps per chunk.
        title: Video title.
        reset: Whether to reset env before rollout.
        env_factory: Optional callable to rebuild env if reset fails due to
            ManiSkill scene lifecycle issues.
        use_base_camera: If True, use obs-based sensor camera (front view,
            closer to LIBERO agentview distribution) for model input instead
            of env.render() (diagonal render_camera). Default True.
        camera_name: Sensor camera name to extract from obs. Default
            "base_camera" (ManiSkill3 PickCube front-facing camera at
            [0.3, 0, 0.6]).

    Returns:
        (frames, logs, env):
            frames: list of rendered uint8 RGB frames (render_camera for video),
            logs: per-chunk metadata dicts,
            env: possibly replaced env (if env_factory was used).
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from IPython.display import HTML, display

    def _safe_reset(_env):
        """Reset env and return (env, obs). obs may be None if reset=False."""
        if not reset:
            return _env, None
        try:
            obs = _env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]  # gymnasium returns (obs, info)
            return _env, obs
        except AttributeError as exc:
            if "_reset_mask" not in str(exc):
                raise
            if env_factory is None:
                raise RuntimeError(
                    "ManiSkill env reset failed with scene=None (_reset_mask). "
                    "Pass env_factory to recreate env automatically, or recreate "
                    "env manually and retry."
                ) from exc
            print("  [closed-loop] reset failed; rebuilding env via env_factory()")
            _env = env_factory()
            obs = _env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            return _env, obs

    def _grab_render_frame(_env):
        """Grab frame from render_camera (for video visualization only)."""
        frame = _env.render()
        if hasattr(frame, "cpu"):
            frame = frame.cpu().numpy()
        frame = np.squeeze(frame)
        if frame.ndim == 3 and frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        return frame

    def _obs_to_model_frame(obs):
        """Extract base_camera RGB from obs dict as uint8 numpy array.

        ManiSkill3 obs structure: obs["sensor_data"][camera_name]["rgb"]
        Returns (H, W, 3) uint8 array suitable for PIL conversion.
        """
        rgb = None
        # ManiSkill3: obs["sensor_data"]["base_camera"]["rgb"]
        if isinstance(obs, dict) and "sensor_data" in obs:
            cam_data = obs["sensor_data"].get(camera_name, {})
            if isinstance(cam_data, dict) and "rgb" in cam_data:
                rgb = cam_data["rgb"]
        # Fallback: obs["image"]["base_camera"]["rgb"]
        if rgb is None and isinstance(obs, dict) and "image" in obs:
            cam_data = obs["image"].get(camera_name, {})
            if isinstance(cam_data, dict) and "rgb" in cam_data:
                rgb = cam_data["rgb"]
        if rgb is None:
            print(f"  [WARN] base_camera not found in obs, falling back to env.render()")
            return None

        if hasattr(rgb, "cpu"):
            rgb = rgb.cpu().numpy()
        rgb = np.squeeze(rgb)
        # Keep only RGB channels (drop alpha if present)
        if rgb.ndim == 3 and rgb.shape[-1] == 4:
            rgb = rgb[:, :, :3]
        if rgb.ndim == 3 and rgb.max() <= 1.0:
            rgb = (rgb * 255).astype(np.uint8)
        return rgb.astype(np.uint8)

    def _to_pil(_frame):
        return Image.fromarray(_frame)

    def _adapt_action(_action_t, _env):
        action_t = np.asarray(_action_t, dtype=np.float32).reshape(-1)
        act_shape = _env.action_space.shape
        d_env = act_shape[-1]
        needs_batch = len(act_shape) == 2
        d_model = action_t.shape[0]
        if d_model < d_env:
            action_t = np.concatenate(
                [action_t, np.zeros(d_env - d_model, dtype=np.float32)]
            )
        elif d_model > d_env:
            action_t = action_t[:d_env]
        if needs_batch:
            action_t = action_t.reshape(1, -1)
        return action_t

    def _predict_chunk(_pil_img, _first_frame: bool):
        kwargs = dict(
            image=_pil_img,
            instruction=instruction,
            unnorm_key=cfg.unnorm_key,
            cfg_scale=cfg.cfg_scale,
            use_ddim=cfg.use_ddim,
            num_ddim_steps=cfg.num_ddim_steps,
            episode_first_frame="True" if _first_frame else "False",
        )
        try:
            out = vla_model.predict_action(
                **kwargs,
                return_confidence=True,
                confidence_type=cfg.confidence_type,
            )
        except TypeError:
            out = vla_model.predict_action(**kwargs)

        if isinstance(out, tuple):
            actions = out[0]
            llm_conf = out[2] if len(out) >= 3 else None
        else:
            actions, llm_conf = out, None

        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 3:
            actions = actions[0]
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        return actions, llm_conf

    # ── Execute ──
    env, obs = _safe_reset(env)

    # Model-input frame: base_camera (front view) if available, else render
    model_frame = None
    if use_base_camera and obs is not None:
        model_frame = _obs_to_model_frame(obs)
    if model_frame is None:
        model_frame = _grab_render_frame(env)
        if use_base_camera:
            print("  [closed-loop] base_camera unavailable at reset; "
                  "using env.render() for chunk 0")

    # Video frames always come from render_camera (for human viewing)
    render_frame = _grab_render_frame(env)
    frames = [render_frame]
    logs: List[dict] = []
    done = False

    if use_base_camera:
        print(f"  [closed-loop] model input: obs[sensor_data][{camera_name}][rgb] "
              f"(front view, shape={model_frame.shape})")
    else:
        print("  [closed-loop] model input: env.render() (render_camera, diagonal)")

    for ck in range(n_chunks):
        if done:
            break

        pil_img = _to_pil(model_frame)
        actions, llm_conf = _predict_chunk(pil_img, _first_frame=(ck == 0))

        exec_steps = min(steps_per_chunk, actions.shape[0])
        logs.append({
            "chunk": ck,
            "pred_steps": int(actions.shape[0]),
            "exec_steps": int(exec_steps),
            "llm_conf": (
                None if llm_conf is None
                else float(np.asarray(llm_conf).mean())
            ),
            "model_input": camera_name if use_base_camera else "render_camera",
        })

        for t in range(exec_steps):
            step_out = env.step(_adapt_action(actions[t], env))
            # Extract obs from step output
            if len(step_out) == 5:
                obs_t, _, terminated, truncated, _ = step_out
                done = bool(terminated or truncated)
            else:
                obs_t, _, done_flag, _ = step_out
                done = bool(done_flag)

            # Update model-input frame from base_camera obs
            if use_base_camera and obs_t is not None:
                new_model_frame = _obs_to_model_frame(obs_t)
                if new_model_frame is not None:
                    model_frame = new_model_frame

            # Video frame from render_camera
            render_frame = _grab_render_frame(env)
            frames.append(render_frame)
            if done:
                break

    # ── Video ──
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_axis_off()
    im = ax.imshow(frames[0])

    def _update(i):
        im.set_data(frames[i])
        ax.set_title(f"{title} — frame {i}/{len(frames)-1}", fontsize=10)
        return [im]

    anim = animation.FuncAnimation(
        fig, _update, frames=len(frames), interval=150, blit=True,
    )
    plt.close(fig)
    display(HTML(anim.to_html5_video()))

    print("=== Closed-loop chunk logs ===")
    for row in logs:
        print(row)

    return frames, logs, env


def plot_action_trajectory(
    norm_actions: np.ndarray,
    baseline_actions: Optional[np.ndarray] = None,
    title: str = "Predicted Actions",
):
    """Visualize predicted action sequence as line plots per dimension.

    Args:
        norm_actions: [T, D] or [1, T, D] normalized actions from the model.
        baseline_actions: Optional [T, D] baseline (L_z=0) for comparison.
        title: Plot title.
    """
    import matplotlib.pyplot as plt

    a = np.asarray(norm_actions)
    if a.ndim == 3:
        a = a[0]
    T, D = a.shape
    dim_labels = ["x", "y", "z", "rx", "ry", "rz", "grip"][:D]

    fig, axes = plt.subplots(1, min(D, 7), figsize=(min(D, 7) * 2.2, 3))
    if D == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=11)

    for d in range(min(D, 7)):
        ax = axes[d]
        ax.plot(range(T), a[:, d], "b-", linewidth=1.5, label="optimized")
        if baseline_actions is not None:
            b = np.asarray(baseline_actions)
            if b.ndim == 3:
                b = b[0]
            ax.plot(range(T), b[:, d], "r--", linewidth=1, alpha=0.6, label="baseline")
        ax.set_title(dim_labels[d], fontsize=9)
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, alpha=0.3)
        if d == 0:
            ax.legend(fontsize=7)

    plt.tight_layout()
    plt.show()


# ===================================================================
# CLI
# ===================================================================
if __name__ == "__main__":
    print("Z-Pathway SPSA Optimization for MemoryVLA")
    print()
    print("Usage (Colab):")
    print("  from scripts.colab_spsa_z_pathway import (")
    print("      ZSPSAConfig, run_z_spsa_full,")
    print("      plot_input_frame, plot_optimization_curves, plot_action_trajectory,")
    print("      render_action_rollout, closed_loop_rollout_multichunk,")
    print("  )")
    print('  cfg = ZSPSAConfig(unnorm_key="libero_spatial_no_noops")')
    print()
    print("  # Show input scene")
    print("  plot_input_frame(image, instruction)")
    print()
    print("  # Run optimization")
    print("  result = run_z_spsa_full(vla, image, instruction, cfg)")
    print()
    print("  # Visualize results")
    print('  plot_optimization_curves(result["phase_b"]["history"])')
    print("  plot_action_trajectory(")
    print('      result["final_norm_actions"],')
    print('      baseline_actions=result["baseline_norm_actions"],')
    print("  )")
    print()
    print("After optimization, inject L_z for all future calls:")
    print("  from scripts.colab_spsa_z_pathway import set_z_latent")
    print('  set_z_latent(vla, result["L_star"])')
    print("  actions, _ = vla.predict_action(image, instruction, ...)")
    print()
    print("Closed-loop rollout (multi-chunk re-planning):")
    print('  set_z_latent(vla, result["L_star"])')
    print("  frames, logs, env = closed_loop_rollout_multichunk(")
    print("      env, vla, instruction, cfg,")
    print("      n_chunks=4, steps_per_chunk=16,")
    print("  )")
