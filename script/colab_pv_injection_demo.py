from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from vla.load import load_vla
from vla.spatial_prior import build_colab_prior


@dataclass
class PriorConfig:
    ckpt_path: str
    unnorm_key: str = "libero_object_no_noops"
    strengths: Tuple[float, ...] = (0.05, 0.1, 0.2, 0.3)
    depth_weight: float = 0.15
    mode: str = "add"
    per_token_cond_scales: Tuple[float, ...] = (0.0, 0.05, 0.1)
    repeats_per_strength: int = 3


def set_update_fused_ablation(vla_model, enabled: bool = False) -> None:
    vla_model.update_fused = enabled
    vla_model.cog_mem_bank.update_fused = enabled
    vla_model.per_mem_bank.update_fused = enabled


def action_stability_score(actions_list: Sequence[np.ndarray]) -> float:
    stacked = np.stack(actions_list, axis=0)
    return float(stacked.std(axis=0).mean())


def evaluate_strength(
    vla_model,
    image: Image.Image,
    instruction: str,
    prior_patch: torch.Tensor,
    unnorm_key: str,
    strength: float,
    per_token_cond_scale: float,
    repeats: int = 3,
    episode_first_frame: str = "True",
    mode: str = "add",
):
    all_actions: List[np.ndarray] = []
    all_norm_actions: List[np.ndarray] = []

    for _ in range(repeats):
        actions, norm_actions = vla_model.predict_action(
            image=image,
            instruction=instruction,
            unnorm_key=unnorm_key,
            episode_first_frame=episode_first_frame,
            per_token_prior=prior_patch,
            per_token_prior_strength=strength,
            per_token_prior_mode=mode,
            per_token_cond_scale=per_token_cond_scale,
        )
        all_actions.append(actions)
        all_norm_actions.append(norm_actions)

    score = action_stability_score(all_actions)
    best_idx = int(np.argmin([np.abs(a).mean() for a in all_actions]))
    return score, all_actions[best_idx], all_norm_actions[best_idx]


def tune_prior_strength(vla_model, image, instruction, prior_patch, cfg: PriorConfig):
    best = (float("inf"), cfg.strengths[0], cfg.per_token_cond_scales[0], None, None)

    for strength in cfg.strengths:
        for cond_scale in cfg.per_token_cond_scales:
            score, actions, norm_actions = evaluate_strength(
                vla_model=vla_model,
                image=image,
                instruction=instruction,
                prior_patch=prior_patch,
                unnorm_key=cfg.unnorm_key,
                strength=strength,
                per_token_cond_scale=cond_scale,
                repeats=cfg.repeats_per_strength,
                episode_first_frame="True",
                mode=cfg.mode,
            )
            print(f"[strength={strength:.3f}, cond_scale={cond_scale:.3f}] stability={score:.6f}")
            if score < best[0]:
                best = (score, strength, cond_scale, actions, norm_actions)

    _, best_strength, best_cond_scale, best_actions, best_norm_actions = best
    return best_strength, best_cond_scale, best_actions, best_norm_actions


def run_demo(image, instruction, sam_mask, depth_map: Optional[torch.Tensor], cfg: PriorConfig):
    vla_model = load_vla(cfg.ckpt_path, load_for_training=False)
    set_update_fused_ablation(vla_model, enabled=False)

    prior_patch = build_colab_prior(
        sam_mask=sam_mask,
        depth_map=depth_map,
        num_patches=256,
        depth_weight=cfg.depth_weight,
    )

    best_strength, best_cond_scale, best_actions, best_norm_actions = tune_prior_strength(
        vla_model=vla_model,
        image=image,
        instruction=instruction,
        prior_patch=prior_patch,
        cfg=cfg,
    )

    vla_model.set_per_token_prior(prior_patch, strength=best_strength, mode=cfg.mode)

    return {
        "model": vla_model,
        "prior_patch": prior_patch,
        "best_strength": best_strength,
        "best_cond_scale": best_cond_scale,
        "actions": best_actions,
        "normalized_actions": best_norm_actions,
    }
