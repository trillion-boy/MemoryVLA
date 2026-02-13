"""
ControlMLLM for MemoryVLA DiT Action Model (Option A).

Training-free visual grounding: optimizes a latent perturbation (pv) on per_tokens
so the DiT cross-attention focuses on the target region (e.g., red cube).

Reference: ControlMLLM (Liu et al., 2024) - adapted for DiT per-token attention.

Usage:
    from action_model.control_dit import ControlDiT
    controller = ControlDiT(action_model_net=vla.action_model.net)

    # target_grid: [256] binary from Grounding DINO + SAM
    pv = controller.optimize(
        per_tokens=per_tokens,  # [B, 256, 256]
        cog_tokens=cog_tokens,  # [B, 1, 4096]
        target_grid=target_grid,  # [256]
        T=5, alpha=100.0,
    )

    # Use optimized per_tokens for diffusion sampling
    per_tokens_controlled = per_tokens + pv
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict


class ControlDiT:
    """
    ControlMLLM adapted for MemoryVLA's DiT action model.

    Optimizes a latent variable pv added to per_tokens such that
    DiT cross-attention (per_attn) concentrates on the target region.
    """

    def __init__(self, action_model_net: nn.Module):
        """
        Args:
            action_model_net: The DiT network (vla.action_model.net)
        """
        self.dit = action_model_net
        self.device = next(action_model_net.parameters()).device
        self.dtype = next(action_model_net.parameters()).dtype

    @torch.enable_grad()
    def optimize(
        self,
        per_tokens: torch.Tensor,
        cog_tokens: torch.Tensor,
        target_grid: torch.Tensor,
        T: int = 5,
        alpha: float = 100.0,
        diffusion_timestep: int = 50,
        noise_action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Optimize latent perturbation pv via ControlMLLM energy function.

        Args:
            per_tokens: [B, N_patches, per_dim] from BottleneckSE + MemoryBank
            cog_tokens: [B, 1, cog_dim] from LLM + MemoryBank
            target_grid: [N_patches] binary mask (1 = target region)
            T: Number of optimization steps (paper recommends 5)
            alpha: Learning rate for gradient descent
            diffusion_timestep: Which diffusion timestep to use for probing
                                (mid-range ~50 works well)
            noise_action: Optional [B, action_len, action_dim] noise;
                          if None, random noise is used

        Returns:
            pv: [B, N_patches, per_dim] optimized perturbation (detached)
        """
        B = per_tokens.shape[0]
        per_tokens = per_tokens.detach()
        cog_tokens = cog_tokens.detach()

        # Target grid to device
        target_grid = target_grid.to(self.device, dtype=self.dtype)
        if target_grid.ndim == 1:
            target_grid = target_grid.unsqueeze(0)  # [1, N_patches]

        # Create noise action for probing DiT attention
        if noise_action is None:
            noise_action = torch.randn(
                B,
                self.dit.future_action_window_size + 1,
                self.dit.in_channels,
                device=self.device, dtype=self.dtype,
            )
        else:
            noise_action = noise_action.detach()

        # Diffusion timestep
        t = torch.tensor([diffusion_timestep] * B, device=self.device)

        # Initialize pv (latent perturbation)
        pv = torch.zeros_like(per_tokens, requires_grad=True)

        # Optimization loop
        losses = []
        for step in range(T):
            if pv.grad is not None:
                pv.grad.zero_()

            # Inject: add pv to per_tokens
            modified_per = per_tokens + pv

            # Forward DiT with attention extraction
            _, attn_stack = self.dit.forward(
                x=noise_action,
                t=t,
                z=cog_tokens,
                per_token=modified_per,
                return_attn_weights=True,
            )
            # attn_stack: [num_blocks, B, heads, T+1, N_patches]

            # Average attention across blocks, heads, and action tokens
            # → [B, N_patches]: how much each patch is attended to
            attn_avg = attn_stack.mean(dim=(0, 2, 3))  # [B, N_patches]

            # Energy function (ControlMLLM Eq. 5)
            # Score = attention on target region / total attention
            inside = (attn_avg * target_grid).sum(dim=-1)    # [B]
            total = attn_avg.sum(dim=-1) + 1e-8              # [B]
            score = inside / total                            # [B]
            loss = ((1.0 - score) ** 2).mean()               # scalar

            # Backward
            loss.backward()

            # Gradient descent on pv
            with torch.no_grad():
                pv = pv - alpha * pv.grad
                pv = pv.detach().requires_grad_(True)

            losses.append(loss.item())

        return pv.detach(), losses

    @torch.enable_grad()
    def optimize_with_schedule(
        self,
        per_tokens: torch.Tensor,
        cog_tokens: torch.Tensor,
        target_grid: torch.Tensor,
        T: int = 5,
        alpha_start: float = 150.0,
        alpha_end: float = 50.0,
        probe_timesteps: Optional[list] = None,
        noise_action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Optimize with learning rate schedule and multi-timestep probing.

        Probes attention at multiple diffusion timesteps for robustness.
        """
        B = per_tokens.shape[0]
        per_tokens = per_tokens.detach()
        cog_tokens = cog_tokens.detach()

        target_grid = target_grid.to(self.device, dtype=self.dtype)
        if target_grid.ndim == 1:
            target_grid = target_grid.unsqueeze(0)

        if noise_action is None:
            noise_action = torch.randn(
                B,
                self.dit.future_action_window_size + 1,
                self.dit.in_channels,
                device=self.device, dtype=self.dtype,
            )
        else:
            noise_action = noise_action.detach()

        if probe_timesteps is None:
            probe_timesteps = [30, 50, 70]

        pv = torch.zeros_like(per_tokens, requires_grad=True)

        losses = []
        for step in range(T):
            if pv.grad is not None:
                pv.grad.zero_()

            # Linear LR schedule
            progress = step / max(T - 1, 1)
            alpha = alpha_start + (alpha_end - alpha_start) * progress

            modified_per = per_tokens + pv

            # Probe at multiple timesteps, accumulate loss
            total_loss = torch.tensor(0.0, device=self.device)
            for ts in probe_timesteps:
                t = torch.tensor([ts] * B, device=self.device)
                _, attn_stack = self.dit.forward(
                    x=noise_action, t=t, z=cog_tokens,
                    per_token=modified_per,
                    return_attn_weights=True,
                )
                attn_avg = attn_stack.mean(dim=(0, 2, 3))
                inside = (attn_avg * target_grid).sum(dim=-1)
                total = attn_avg.sum(dim=-1) + 1e-8
                score = inside / total
                total_loss = total_loss + ((1.0 - score) ** 2).mean()

            total_loss = total_loss / len(probe_timesteps)
            total_loss.backward()

            with torch.no_grad():
                pv = pv - alpha * pv.grad
                pv = pv.detach().requires_grad_(True)

            losses.append(total_loss.item())

        return pv.detach(), losses


def create_target_grid_from_mask(
    mask: np.ndarray,
    grid_h: int = 16,
    grid_w: int = 16,
    threshold: float = 0.1,
) -> torch.Tensor:
    """
    Convert pixel-level binary mask to patch grid for ControlDiT.

    Args:
        mask: Binary mask [H, W] (from SAM)
        grid_h, grid_w: Patch grid size (16x16 = 256 patches for MemoryVLA)
        threshold: Fraction of pixels in a patch to activate it

    Returns:
        target_grid: [grid_h * grid_w] binary tensor
    """
    mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    pooled = F.adaptive_avg_pool2d(mask_t, (grid_h, grid_w))
    grid = (pooled > threshold).float()
    return grid.flatten()  # [256]
