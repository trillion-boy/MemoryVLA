"""
Training-free spatial control for MemoryVLA DiT Action Model (Method A).

Instead of optimizing a latent perturbation through dead per_attn (ControlDiT),
this directly replaces per_attn with manual spatial attention using SAM masks.

Problem: per_attn was zero-initialized and never trained (in_proj ≈ 0.0003~0.0006),
so ControlDiT's gradient-based optimization on pv is ineffective.

Solution: Bypass per_attn entirely. Use SAM mask as attention weights directly,
computing a spatial-weighted average of per_token features as the cross-attention
output. This requires no optimization and no training.

Usage:
    from action_model.spatial_control import SpatialGatingControl

    controller = SpatialGatingControl(action_model_net=vla.action_model.net)

    # target_grid: [256] binary from Grounding DINO + SAM
    # Use as context manager around diffusion sampling:
    with controller.spatial_attn_override(target_grid):
        # DiT.forward() now uses manual spatial attention instead of dead per_attn
        actions = diffusion_sample(...)
"""

import torch
import torch.nn as nn
import numpy as np
from contextlib import contextmanager
from typing import Optional


class SpatialGatingControl:
    """
    Training-free visual grounding by replacing dead per_attn with
    manual spatial attention derived from SAM masks.

    Key insight: per_attn in the checkpoint has near-zero weights (zero-init
    that never trained), so ControlDiT optimization on pv is ineffective.
    Instead, we directly compute spatial attention using the SAM mask.

    The manual attention computes:
        context = sum(attn_weight_i * per_token_i)  for target patches
    and adds this to each action token via the residual connection.
    """

    def __init__(self, action_model_net: nn.Module, scale: float = 0.5):
        """
        Args:
            action_model_net: The DiT network (vla.action_model.net)
            scale: Magnitude of the spatial attention output.
                   Start small (0.1-1.0) since the model was trained with
                   per_attn output ≈ 0. Too large may destabilize.
        """
        self.dit = action_model_net
        self.device = next(action_model_net.parameters()).device
        self.dtype = next(action_model_net.parameters()).dtype
        self.scale = scale

    def compute_spatial_weights(self, target_grid: torch.Tensor) -> torch.Tensor:
        """
        Convert binary target grid to normalized attention weights.

        Args:
            target_grid: [N_patches] or [B, N_patches] binary mask

        Returns:
            attn_weights: [B, N_patches] normalized (sum to 1 over target patches)
        """
        target_grid = target_grid.to(self.device, dtype=self.dtype)
        if target_grid.ndim == 1:
            target_grid = target_grid.unsqueeze(0)

        attn_weights = target_grid / (target_grid.sum(dim=-1, keepdim=True) + 1e-8)
        return attn_weights

    @contextmanager
    def spatial_attn_override(self, target_grid: torch.Tensor):
        """
        Context manager that temporarily replaces per_attn in all DiT blocks
        with manual spatial attention weighted by target_grid.

        Inside this context:
        - per_attn(query, key, value) → spatial_weighted_sum(value) * scale
        - All action tokens receive the same spatial context
        - Attention weights match SAM mask (target patches only)

        Args:
            target_grid: [N_patches] or [B, N_patches] binary mask
        """
        attn_weights = self.compute_spatial_weights(target_grid)
        scale = self.scale

        original_forwards = []
        blocks_with_per_attn = []

        for block in self.dit.blocks:
            if block.use_per_attn:
                original_forwards.append(block.per_attn.forward)
                blocks_with_per_attn.append(block)

                num_heads = block.per_attn.num_heads

                def _make_manual_attn(aw, sc, nh):
                    def manual_attn(
                        query, key, value,
                        need_weights=False,
                        average_attn_weights=True,
                        **kwargs,
                    ):
                        B, T, D = query.shape
                        # aw: [B_mask, N_patches], value: [B, N_patches, D]
                        # Expand aw to match batch size (handles CFG doubling)
                        aw_expanded = aw.expand(B, -1)

                        # Spatial-weighted sum of per_token values
                        context = torch.einsum('bn,bnd->bd', aw_expanded, value)
                        output = context.unsqueeze(1).expand(B, T, D) * sc

                        if need_weights:
                            w = aw_expanded.unsqueeze(1).unsqueeze(1).expand(
                                B, 1, T, -1
                            )
                            if not average_attn_weights:
                                w = w.expand(B, nh, T, -1)
                            return output, w
                        return output, None

                    return manual_attn

                block.per_attn.forward = _make_manual_attn(
                    attn_weights, scale, num_heads
                )

        try:
            yield
        finally:
            for block, orig_fwd in zip(blocks_with_per_attn, original_forwards):
                block.per_attn.forward = orig_fwd

    def diagnose(self, per_tokens: torch.Tensor, cog_tokens: torch.Tensor,
                 target_grid: torch.Tensor) -> dict:
        """
        Diagnostic: compare per_attn output with vs without spatial override.

        Returns dict with norms and attention patterns for analysis.
        """
        B = per_tokens.shape[0]
        t = torch.tensor([50] * B, device=self.device)
        noise_action = torch.randn(
            B, self.dit.future_action_window_size + 1,
            self.dit.in_channels,
            device=self.device, dtype=self.dtype,
        )

        # Run with original (dead) per_attn
        with torch.no_grad():
            _, attn_orig = self.dit.forward(
                x=noise_action, t=t, z=cog_tokens,
                per_token=per_tokens, return_attn_weights=True,
            )

        # Run with spatial override
        with torch.no_grad():
            with self.spatial_attn_override(target_grid):
                _, attn_spatial = self.dit.forward(
                    x=noise_action, t=t, z=cog_tokens,
                    per_token=per_tokens, return_attn_weights=True,
                )

        target_grid_dev = target_grid.to(self.device, dtype=self.dtype)
        if target_grid_dev.ndim == 1:
            target_grid_dev = target_grid_dev.unsqueeze(0)

        def _compute_focus(attn_stack):
            if attn_stack is None:
                return 0.0
            attn_avg = attn_stack.mean(dim=(0, 2, 3))  # [B, N_patches]
            inside = (attn_avg * target_grid_dev).sum(dim=-1)
            total = attn_avg.sum(dim=-1) + 1e-8
            return (inside / total).mean().item()

        info = {
            "original_focus": _compute_focus(attn_orig),
            "spatial_focus": _compute_focus(attn_spatial),
            "scale": self.scale,
            "n_target_patches": (target_grid > 0).sum().item(),
            "n_total_patches": target_grid.numel(),
        }

        expected_random = info["n_target_patches"] / info["n_total_patches"]
        info["expected_random_focus"] = expected_random

        print(f"=== Spatial Control Diagnostics ===")
        print(f"  Target patches: {info['n_target_patches']}/{info['n_total_patches']}")
        print(f"  Expected random focus: {expected_random:.4f}")
        print(f"  Original per_attn focus: {info['original_focus']:.4f}")
        print(f"  Spatial override focus:  {info['spatial_focus']:.4f}")
        print(f"  Scale: {self.scale}")
        print(f"  Improvement: {info['spatial_focus'] - info['original_focus']:.4f}")

        return info
