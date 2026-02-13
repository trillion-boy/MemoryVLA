"""Utilities for building per-token spatial priors for MemoryVLA inference."""
from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn.functional as F


def normalize_map(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize each map to [0, 1] over spatial dimensions."""
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.dim() != 3:
        raise ValueError(f"Expected [H,W] or [B,H,W], got {tuple(x.shape)}")

    flat = x.flatten(1)
    x_min = flat.amin(dim=1, keepdim=True).unsqueeze(-1)
    x_max = flat.amax(dim=1, keepdim=True).unsqueeze(-1)
    return (x - x_min) / (x_max - x_min + eps)


def mask_to_patch_prior(
    mask: torch.Tensor,
    num_patches: int = 256,
    threshold: Optional[float] = None,
) -> torch.Tensor:
    """Convert [H,W] or [B,H,W] mask into [B,N] patch priors."""
    side = int(math.sqrt(num_patches))
    if side * side != num_patches:
        raise ValueError(f"num_patches must be square, got {num_patches}")

    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if mask.dim() != 3:
        raise ValueError(f"Expected [H,W] or [B,H,W], got {tuple(mask.shape)}")

    mask = normalize_map(mask).unsqueeze(1)
    patch = F.interpolate(mask, size=(side, side), mode="bilinear", align_corners=False)
    patch = patch.squeeze(1)

    if threshold is not None:
        patch = (patch >= threshold).to(patch.dtype)

    return patch.flatten(1)


def blend_priors(
    semantic_prior: torch.Tensor,
    depth_prior: Optional[torch.Tensor] = None,
    depth_weight: float = 0.2,
) -> torch.Tensor:
    """Blend semantic and depth priors safely."""
    semantic_prior = normalize_map(semantic_prior)

    if depth_prior is None:
        return semantic_prior

    depth_prior = normalize_map(depth_prior)
    return (1.0 - depth_weight) * semantic_prior + depth_weight * depth_prior


def build_colab_prior(
    sam_mask: torch.Tensor,
    depth_map: Optional[torch.Tensor] = None,
    num_patches: int = 256,
    depth_weight: float = 0.2,
) -> torch.Tensor:
    """Convenience helper for Colab experiments."""
    prior = blend_priors(sam_mask, depth_map, depth_weight=depth_weight)
    return mask_to_patch_prior(prior, num_patches=num_patches)
