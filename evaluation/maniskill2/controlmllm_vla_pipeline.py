"""
ControlMLLM-style Visual Prompt Optimization for MemoryVLA.

Test-time adaptation: optimizes a visual prompt using the frozen LLM's
self-attention as signal, guided by depth-derived spatial masks.

Key difference from naive depth injection:
    Naive:       Pv = fixed depth_tokens (no optimization)
    ControlMLLM: Pv = zeros → iterative gradient optimization → optimized prompt

Pipeline:
    1. DA V2 → depth map → spatial mask (where objects are)
    2. Initialize visual_prompt (Pv) = zeros [1, N_patches, 4096]
    3. Optimization loop (T iterations):
        - Add Pv to image token embeddings
        - Forward through frozen LLM → extract attention maps
        - Loss = alpha * (1 - attention_inside_mask)^2
        - Update Pv via Adam or SGD
    4. Inject optimized Pv into MemoryVLA's predict_action via hook

Reference: ControlMLLM (NeurIPS 2024) - https://github.com/mrwu-mac/ControlMLLM
"""

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional, Tuple, Any
from transformers import LlamaTokenizerFast
import os


class ControlMLLMVLAPipeline:
    """
    Test-time visual prompt optimization for MemoryVLA.

    Optimizes a perturbation (visual_prompt) added to image token embeddings
    so that the LLM's attention concentrates on depth-indicated object regions.
    """

    def __init__(
        self,
        vla,
        depth_model=None,
        T: int = 10,
        lr: float = 0.1,
        alpha_loss: float = 400.0,
        layer_start: int = 14,
        layer_end: int = 26,
        optimize_freq: int = 1,
        optimizer: str = "sgd",
    ):
        """
        Args:
            vla: MemoryVLA model (frozen)
            depth_model: Depth Anything V2 model
            T: Number of optimization iterations per step
            lr: Learning rate for visual prompt optimization
            alpha_loss: Loss scaling factor (ControlMLLM uses 400)
            layer_start: First LLM layer for attention extraction
            layer_end: Last LLM layer for attention extraction
            optimize_freq: Optimize every N steps (1=every step, 5=every 5th step)
            optimizer: "adam" or "sgd" (sgd is better for short T)
        """
        self.vla = vla
        self.depth_model = depth_model
        self.T = T
        self.lr = lr
        self.alpha_loss = alpha_loss
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.optimize_freq = optimize_freq
        self.optimizer = optimizer.lower()

        # Model references
        self.vlm = vla.vlm
        self.llm = self.vlm.llm_backbone.llm
        self.tokenizer = self.vlm.llm_backbone.tokenizer
        self.image_transform = self.vlm.vision_backbone.image_transform

        # State
        self._visual_prompt = None
        self._hook_handle = None
        self._step_count = 0

    # ================================================================
    # Depth → Spatial Mask
    # ================================================================

    def estimate_depth(self, image: Image.Image) -> Optional[np.ndarray]:
        """Get depth map from Depth Anything V2."""
        if self.depth_model is None:
            return None
        rgb_np = np.array(image)
        if hasattr(self.depth_model, 'infer_image'):
            return self.depth_model.infer_image(rgb_np)
        elif hasattr(self.depth_model, '__call__'):
            result = self.depth_model(image)
            depth = result["depth"]
            if isinstance(depth, Image.Image):
                return np.array(depth).astype(np.float32)
            return np.array(depth)
        return None

    def create_spatial_mask(
        self, depth_map: np.ndarray, grid_h: int = 16, grid_w: int = 16,
        debug_save_path: Optional[str] = None,
        rgb_image: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        Create attention mask using HSV saturation peak detection.

        The cube is the MOST SATURATED object in the scene:
        - Cube: vivid red/green/blue (high saturation)
        - Table: brown wood (moderate saturation)
        - Robot arm: gray/silver (low saturation)
        - Background sky: gray (low saturation)

        Simply take the top-K most saturated patches = cube location.

        Args:
            depth_map: Depth values (unused, kept for API compat)
            grid_h, grid_w: Attention grid size (16x16 for 256 patches)
            debug_save_path: If set, save visualization
            rgb_image: RGB image (REQUIRED)

        Returns:
            mask: Tensor [1, grid_h * grid_w] (normalized)
        """
        if rgb_image is None:
            mask_tensor = torch.ones(grid_h * grid_w) / (grid_h * grid_w)
            return mask_tensor.unsqueeze(0)

        img_h, img_w = rgb_image.shape[:2]
        block_h = img_h // grid_h
        block_w = img_w // grid_w

        # --- Step 1: Compute per-patch saturation ---
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        grid_sat = np.zeros((grid_h, grid_w), dtype=np.float32)

        for i in range(grid_h):
            for j in range(grid_w):
                block = hsv[i * block_h:(i + 1) * block_h,
                            j * block_w:(j + 1) * block_w, 1]
                grid_sat[i, j] = np.mean(block)

        # --- Step 2: Find THE most saturated patch (= cube center) ---
        # argmax → single peak patch + 3x3 neighborhood = max 9 patches
        flat_sat = grid_sat.flatten()
        peak_idx = int(np.argmax(flat_sat))
        peak_row, peak_col = peak_idx // grid_w, peak_idx % grid_w
        peak_threshold = flat_sat[peak_idx]

        # Single peak patch
        peak_patches = np.zeros((grid_h, grid_w), dtype=np.float32)
        peak_patches[peak_row, peak_col] = 1.0

        # 3x3 neighborhood around peak = max 9 patches
        mask_grid = np.zeros((grid_h, grid_w), dtype=np.float32)
        for di in range(-1, 2):
            for dj in range(-1, 2):
                r, c = peak_row + di, peak_col + dj
                if 0 <= r < grid_h and 0 <= c < grid_w:
                    mask_grid[r, c] = 1.0

        # --- Debug visualization ---
        if debug_save_path:
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            axes[0].imshow(rgb_image)
            axes[0].set_title("RGB input")
            axes[1].imshow(grid_sat, cmap="hot", interpolation="nearest")
            axes[1].set_title(f"Saturation (per patch)\nmax={grid_sat.max():.1f}")
            axes[2].imshow(peak_patches, cmap="gray",
                          interpolation="nearest")
            axes[2].set_title(f"Peak (argmax)\nsat={peak_threshold:.1f} @({peak_row},{peak_col})")
            axes[3].imshow(mask_grid, cmap="hot", interpolation="nearest")
            axes[3].set_title(f"Final mask (3x3)\n{int(mask_grid.sum())} patches")
            for ax in axes:
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(debug_save_path, dpi=100, bbox_inches="tight")
            plt.close()

        # Convert to normalized tensor
        mask_tensor = torch.tensor(mask_grid, dtype=torch.float32).flatten()
        if mask_tensor.sum() > 1e-6:
            mask_tensor = mask_tensor / mask_tensor.sum()
        else:
            mask_tensor = torch.ones_like(mask_tensor) / mask_tensor.numel()

        return mask_tensor.unsqueeze(0)  # [1, 256]

    # ================================================================
    # Attention Loss (ControlMLLM)
    # ================================================================

    def compute_attention_loss(
        self,
        attentions: tuple,
        mask: torch.Tensor,
        image_start: int,
        image_end: int,
    ) -> torch.Tensor:
        """
        Attention alignment loss with temperature scaling for stronger gradients.

        Post-softmax attention is very peaked (1-2 patches dominate), causing
        vanishing gradients. We re-scale attention with temperature to soften
        the distribution, making it easier for gradient to redistribute attention.

        Loss = alpha * (1 - activation_value)^2
        where activation_value = sum(softmax(attn/T) * mask)
        """
        device = mask.device

        end = min(self.layer_end, len(attentions))
        start = min(self.layer_start, end)

        selected = []
        for i in range(start, end):
            selected.append(attentions[i].to(device))

        if not selected:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Average across selected layers: [batch, heads, seq, seq]
        mean_attn = torch.stack(selected).mean(dim=0)

        # Average across heads: [batch, seq, seq]
        mean_attn = mean_attn.mean(dim=1)

        # Attention from last token to image tokens: [batch, N_patches]
        last_to_image = mean_attn[:, -1, image_start:image_end]

        # Handle CLS token
        n_image = image_end - image_start
        if mask.shape[1] < n_image:
            pad = torch.zeros(1, n_image - mask.shape[1], device=device)
            mask_padded = torch.cat([pad, mask], dim=1)
        else:
            mask_padded = mask[:, :n_image]

        # Temperature-scaled re-normalization for stronger gradients
        # Post-softmax attention is too peaked → gradient vanishes
        # Re-apply softmax with high temperature to soften the distribution
        # This preserves the gradient graph while making optimization easier
        temperature = 5.0
        log_attn = torch.log(last_to_image.clamp(min=1e-10))
        attn_rescaled = torch.softmax(log_attn / temperature, dim=-1)

        # Activation: fraction of rescaled attention inside mask
        activation = (attn_rescaled * mask_padded).sum(dim=-1)

        # Loss: push activation toward 1.0
        loss = self.alpha_loss * ((1.0 - activation) ** 2).mean()

        return loss

    # ================================================================
    # Embedding Construction
    # ================================================================

    def _get_embeddings(
        self, image: Image.Image, instruction: str
    ) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        Construct multimodal embeddings (replicates predict_action setup).

        Returns:
            multimodal_embeddings: [1, seq_len, 4096]
            n_image_tokens: number of image tokens (257)
            input_ids: tokenized input
        """
        model_dtype = next(self.vla.parameters()).dtype
        device = self.vlm.device

        # Tokenize
        prompt_builder = self.vlm.get_prompt_builder()
        prompt_builder.add_turn(
            role="human",
            message=f"What action should the robot take to {instruction.lower()}?",
        )
        prompt_text = prompt_builder.get_prompt()

        input_ids = self.tokenizer(
            prompt_text, truncation=True, return_tensors="pt"
        ).input_ids.to(device)

        if isinstance(self.tokenizer, LlamaTokenizerFast):
            input_ids = torch.cat(
                (input_ids, torch.tensor([[29871, 2]], device=device)), dim=1
            )

        # Process image through vision backbone + projector
        pixel_values = self.image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(device, dtype=model_dtype)
        elif isinstance(pixel_values, dict):
            pixel_values = {
                k: v[None, ...].to(device, dtype=model_dtype)
                for k, v in pixel_values.items()
            }

        with torch.no_grad():
            patch_features = self.vlm.vision_backbone(pixel_values)
            self.vlm.vision_feats = patch_features
            projected = self.vlm.projector(patch_features)

        # Text embeddings
        with torch.no_grad():
            input_embeddings = self.vlm.llm_backbone.embed_input_ids(input_ids)

        # Construct multimodal embeddings: [BOS] + [Image tokens] + [Text tokens]
        multimodal_embeddings = torch.cat(
            [
                input_embeddings[:, :1, :],
                projected,
                input_embeddings[:, 1:, :],
            ],
            dim=1,
        )

        n_image_tokens = projected.shape[1]  # 257

        return multimodal_embeddings, n_image_tokens, input_ids

    # ================================================================
    # Visual Prompt Optimization
    # ================================================================

    def optimize_visual_prompt(
        self,
        image: Image.Image,
        instruction: str,
        depth_map: np.ndarray,
        debug_save_dir: Optional[str] = None,
    ) -> torch.Tensor:
        """
        ControlMLLM-style test-time optimization.

        Optimizes visual_prompt so that the LLM's attention focuses
        on depth-indicated object regions.

        Args:
            image: RGB observation
            instruction: task instruction
            depth_map: depth from DA V2
            debug_save_dir: if set, save mask debug images here

        Returns:
            visual_prompt: optimized [1, N_patches, 4096]
        """
        model_dtype = next(self.vla.parameters()).dtype
        device = self.vlm.device

        # 1. Create spatial mask from depth (with CCA filtering)
        mask_debug_path = None
        if debug_save_dir:
            os.makedirs(debug_save_dir, exist_ok=True)
            mask_debug_path = os.path.join(
                debug_save_dir, f"mask_step_{self._step_count:03d}.png"
            )
        rgb_np = np.array(image)
        mask = self.create_spatial_mask(
            depth_map, debug_save_path=mask_debug_path,
            rgb_image=rgb_np,
        )  # [1, 256]
        mask = mask.to(device)

        # Log mask stats
        n_active = (mask > 0).sum().item()
        print(f"    Mask: {n_active}/256 active patches")

        # 2. Get base embeddings
        base_embeddings, n_image_tokens, _ = self._get_embeddings(image, instruction)
        image_start = 1
        image_end = 1 + n_image_tokens

        # 3. Initialize Pv = zeros (ControlMLLM style)
        visual_prompt = torch.zeros(
            1, n_image_tokens, base_embeddings.shape[-1],
            device=device, dtype=torch.float32,
        ).requires_grad_(True)

        # Optimizer state (Adam only, SGD needs no state)
        if self.optimizer == "adam":
            m = torch.zeros_like(visual_prompt)
            s = torch.zeros_like(visual_prompt)
            beta1, beta2, eps = 0.9, 0.999, 1e-3

        # 4. Optimization loop
        for t in range(1, self.T + 1):
            # Add Pv to image token positions
            modified = base_embeddings.clone().float()
            modified[:, image_start:image_end, :] += visual_prompt

            # Forward through LLM with attention output
            with torch.enable_grad():
                outputs = self.llm(
                    inputs_embeds=modified.to(model_dtype),
                    output_attentions=True,
                    return_dict=True,
                )

            # Compute attention alignment loss
            loss = self.compute_attention_loss(
                outputs.attentions, mask, image_start, image_end
            )

            # Backprop
            grad = torch.autograd.grad(loss, visual_prompt)[0]
            grad_norm = grad.norm().item()

            if self.optimizer == "adam":
                # Adam update
                m = beta1 * m + (1 - beta1) * grad
                s = beta2 * s + (1 - beta2) * grad.pow(2)
                m_hat = m / (1 - beta1 ** t)
                s_hat = s / (1 - beta2 ** t)
                visual_prompt = (
                    visual_prompt - self.lr * m_hat / (torch.sqrt(s_hat) + eps)
                ).detach().requires_grad_(True)
            else:
                # SGD update (simpler, better for short T)
                visual_prompt = (
                    visual_prompt - self.lr * grad
                ).detach().requires_grad_(True)

            if t == 1 or t == self.T or t % 5 == 0:
                pv_norm = visual_prompt.norm().item()
                print(f"    [Optim step {t}/{self.T}] loss={loss.item():.4f} "
                      f"grad_norm={grad_norm:.6f} pv_norm={pv_norm:.4f}")

        return visual_prompt.detach().to(model_dtype)

    # ================================================================
    # Main Prediction
    # ================================================================

    def predict_action(
        self,
        image: Image.Image,
        instruction: str,
        obs: Optional[dict] = None,
        unnorm_key: str = "libero_object_no_noops",
        cfg_scale: float = 1.5,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        episode_first_frame: str = "False",
        **kwargs,
    ) -> Tuple[np.ndarray, Any]:
        """
        Predict action with ControlMLLM-optimized visual prompt.

        Steps:
            1. Estimate depth (DA V2)
            2. Optimize visual prompt (attention alignment)
            3. Inject optimized prompt via hook
            4. Run MemoryVLA predict_action
        """
        self._step_count += 1

        # 1. Estimate depth
        depth_map = self.estimate_depth(image)

        # 2. Optimize visual prompt (at specified frequency)
        if depth_map is not None and (self._step_count % self.optimize_freq == 1
                                       or self.optimize_freq == 1
                                       or self._visual_prompt is None):
            print(f"  [Step {self._step_count}] Optimizing visual prompt...")
            # Save debug masks for first 3 steps
            debug_dir = kwargs.pop("debug_save_dir", None)
            self._visual_prompt = self.optimize_visual_prompt(
                image, instruction, depth_map,
                debug_save_dir=debug_dir,
            )

        # 3. Register hook to inject during predict_action
        if self._visual_prompt is not None:
            vp = self._visual_prompt

            def projector_hook(module, input, output):
                return output + vp

            self._hook_handle = self.vlm.projector.register_forward_hook(
                projector_hook
            )

        # 4. Run original predict_action
        try:
            actions, extra = self.vla.predict_action(
                image=image,
                instruction=instruction,
                unnorm_key=unnorm_key,
                cfg_scale=cfg_scale,
                use_ddim=use_ddim,
                num_ddim_steps=num_ddim_steps,
                episode_first_frame=episode_first_frame,
                **kwargs,
            )
        finally:
            if self._hook_handle is not None:
                self._hook_handle.remove()
                self._hook_handle = None

        return actions, extra

    def reset(self):
        """Reset state for new episode."""
        self._visual_prompt = None
        self._step_count = 0

    def debug_first_frame(
        self,
        image: Image.Image,
        save_path: str = "/content/debug_first_frame.png",
    ):
        """
        Save RGB + depth + mask visualization for a single frame.
        Call this to see what the model sees and what the mask looks like.
        """
        rgb = np.array(image)
        # Depth is optional now (mask uses RGB color saliency)
        depth = self.estimate_depth(image)
        if depth is None:
            depth = np.zeros(rgb.shape[:2], dtype=np.float32)

        # Create mask with debug
        self.create_spatial_mask(
            depth, debug_save_path=save_path, rgb_image=rgb
        )
        print(f"Debug visualization saved to: {save_path}")

    # ================================================================
    # Attention Diagnostics
    # ================================================================

    def visualize_attention_by_layer(
        self,
        image: Image.Image,
        instruction: str,
        save_dir: str = "/content/attention_diagnosis",
    ) -> dict:
        """
        Diagnostic: visualize which layers attend to image tokens.

        Runs ONE forward pass and saves per-layer attention heatmaps
        (last token → image tokens) overlaid on 16x16 grid.
        Use this BEFORE running optimization to find the right layer range.

        Args:
            image: RGB observation
            instruction: task instruction
            save_dir: directory to save heatmaps

        Returns:
            dict mapping layer_idx → activation stats
        """
        os.makedirs(save_dir, exist_ok=True)
        model_dtype = next(self.vla.parameters()).dtype

        base_embeddings, n_image_tokens, _ = self._get_embeddings(image, instruction)
        image_start = 1
        image_end = 1 + n_image_tokens

        # Determine grid size: 256 spatial patches → 16x16
        # n_image_tokens can be 256 (no CLS) or 257 (CLS + 256 spatial)
        if n_image_tokens == 257:
            spatial_offset = 1  # skip CLS token
            n_spatial = 256
        else:
            spatial_offset = 0
            n_spatial = n_image_tokens
        grid_side = int(np.sqrt(n_spatial))
        print(f"  Image tokens: {n_image_tokens} (spatial: {n_spatial}, grid: {grid_side}x{grid_side})")

        with torch.no_grad():
            outputs = self.llm(
                inputs_embeds=base_embeddings.to(model_dtype),
                output_attentions=True,
                return_dict=True,
            )

        layer_stats = {}
        n_layers = len(outputs.attentions)

        # Create summary figure: all layers in one grid
        cols = 8
        rows = (n_layers + cols - 1) // cols
        fig_all, axes_all = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes_flat = axes_all.flatten()

        for layer_idx in range(n_layers):
            attn = outputs.attentions[layer_idx]  # [1, heads, seq, seq]

            # Average across heads
            attn_mean = attn.mean(dim=1)  # [1, seq, seq]

            # Last token → image tokens
            last_to_img = attn_mean[0, -1, image_start:image_end]  # [n_image_tokens]

            # Get spatial tokens only (skip CLS if present)
            spatial_attn = last_to_img[spatial_offset:].detach().cpu().numpy()  # [n_spatial]

            # Normalize for visualization
            if spatial_attn.max() > 0:
                spatial_vis = spatial_attn / spatial_attn.max()
            else:
                spatial_vis = spatial_attn

            heatmap = spatial_vis.reshape(grid_side, grid_side)

            # Stats
            layer_stats[layer_idx] = {
                "mean": float(spatial_attn.mean()),
                "max": float(spatial_attn.max()),
                "std": float(spatial_attn.std()),
                "sum": float(spatial_attn.sum()),
            }

            # Plot in grid
            ax = axes_flat[layer_idx]
            ax.imshow(heatmap, cmap="hot", interpolation="nearest", vmin=0, vmax=1)
            ax.set_title(
                f"L{layer_idx}\nmax={spatial_attn.max():.4f}",
                fontsize=8,
            )
            ax.axis("off")

        # Hide unused axes
        for idx in range(n_layers, len(axes_flat)):
            axes_flat[idx].axis("off")

        fig_all.suptitle(
            "Attention: last token → image patches (per layer)", fontsize=14
        )
        plt.tight_layout()
        fig_all.savefig(
            os.path.join(save_dir, "all_layers_attention.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig_all)

        # Print summary: which layers have strongest image attention
        print(f"\n{'='*60}")
        print("Attention Diagnosis: last token → image patches")
        print(f"{'='*60}")
        print(f"{'Layer':>6} {'Mean':>10} {'Max':>10} {'Std':>10}")
        print(f"{'-'*40}")
        for idx in sorted(layer_stats.keys()):
            s = layer_stats[idx]
            marker = " <<<" if s["max"] > 0.01 else ""
            print(f"  L{idx:2d}   {s['mean']:.6f}  {s['max']:.6f}  {s['std']:.6f}{marker}")

        print(f"\nHeatmaps saved to: {save_dir}/all_layers_attention.png")
        return layer_stats
