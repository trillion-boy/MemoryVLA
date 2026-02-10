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
        - Update Pv via Adam
    4. Inject optimized Pv into MemoryVLA's predict_action via hook

Reference: ControlMLLM (NeurIPS 2024) - https://github.com/mrwu-mac/ControlMLLM
"""

import cv2
import numpy as np
import torch
from PIL import Image
from typing import Optional, Tuple, Any
from transformers import LlamaTokenizerFast


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
        T: int = 5,
        lr: float = 0.03,
        alpha_loss: float = 400.0,
        layer_start: int = 14,
        layer_end: int = 26,
        optimize_freq: int = 1,
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
        """
        self.vla = vla
        self.depth_model = depth_model
        self.T = T
        self.lr = lr
        self.alpha_loss = alpha_loss
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.optimize_freq = optimize_freq

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
        self, depth_map: np.ndarray, grid_h: int = 16, grid_w: int = 16
    ) -> torch.Tensor:
        """
        Create attention mask from depth map.

        Identifies object regions via depth edges and deviation from
        the median (table surface). Returns a soft mask for the
        attention alignment loss.

        Args:
            depth_map: Depth values (H, W) from DA V2
            grid_h, grid_w: Attention grid size (16x16 for 256 patches)

        Returns:
            mask: Tensor [1, grid_h * grid_w] (normalized soft mask)
        """
        d = depth_map.copy().astype(np.float32)
        d_min, d_max = d.min(), d.max()
        if d_max - d_min > 1e-6:
            d_norm = (d - d_min) / (d_max - d_min)
        else:
            d_norm = np.zeros_like(d)

        d_uint8 = (d_norm * 255).astype(np.uint8)

        # Edge detection → object boundaries
        edges = cv2.Canny(d_uint8, 30, 100)
        kernel = np.ones((5, 5), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)

        # Regions significantly different from median (objects on table)
        median_depth = np.median(d_norm)
        object_mask = (np.abs(d_norm - median_depth) > 0.08).astype(np.float32)

        # Combine edge and object detection
        combined = np.maximum(edges_dilated / 255.0, object_mask)

        # Resize to attention grid
        mask_resized = cv2.resize(
            combined, (grid_w, grid_h), interpolation=cv2.INTER_LINEAR
        )

        # Convert to soft mask (normalized)
        mask_tensor = torch.tensor(mask_resized, dtype=torch.float32).flatten()
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
        ControlMLLM attention alignment loss.

        Measures what fraction of the last token's attention falls
        inside the mask region, and penalizes if it's low.

        Loss = alpha * (1 - activation_value)^2
        where activation_value = sum(attention * mask) / sum(attention)

        Args:
            attentions: Tuple of attention tensors per layer
                        Each: [batch, heads, seq_len, seq_len]
            mask: Spatial mask [1, N_spatial_patches]
            image_start: Start index of image tokens
            image_end: End index of image tokens

        Returns:
            loss: scalar tensor
        """
        device = mask.device

        # Select middle-to-late layers (ControlMLLM uses layers 14-26)
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

        # Handle CLS token: mask is [1, 256] but image tokens might be 257
        n_image = image_end - image_start
        if mask.shape[1] < n_image:
            # Pad mask with 0 for CLS token (position 0 of image tokens)
            pad = torch.zeros(1, n_image - mask.shape[1], device=device)
            mask_padded = torch.cat([pad, mask], dim=1)  # CLS first, then spatial
        else:
            mask_padded = mask[:, :n_image]

        # Normalize attention
        attn_sum = last_to_image.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        last_to_image_norm = last_to_image / attn_sum

        # Activation: fraction of attention inside mask
        activation = (last_to_image_norm * mask_padded).sum(dim=-1)

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
    ) -> torch.Tensor:
        """
        ControlMLLM-style test-time optimization.

        Optimizes visual_prompt so that the LLM's attention focuses
        on depth-indicated object regions.

        Args:
            image: RGB observation
            instruction: task instruction
            depth_map: depth from DA V2

        Returns:
            visual_prompt: optimized [1, N_patches, 4096]
        """
        model_dtype = next(self.vla.parameters()).dtype
        device = self.vlm.device

        # 1. Create spatial mask from depth
        mask = self.create_spatial_mask(depth_map)  # [1, 256]
        mask = mask.to(device)

        # 2. Get base embeddings
        base_embeddings, n_image_tokens, _ = self._get_embeddings(image, instruction)
        image_start = 1
        image_end = 1 + n_image_tokens

        # 3. Initialize Pv = zeros (ControlMLLM style)
        visual_prompt = torch.zeros(
            1, n_image_tokens, base_embeddings.shape[-1],
            device=device, dtype=torch.float32,
        ).requires_grad_(True)

        # Adam optimizer state
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

            # Adam update
            m = beta1 * m + (1 - beta1) * grad
            s = beta2 * s + (1 - beta2) * grad.pow(2)
            m_hat = m / (1 - beta1 ** t)
            s_hat = s / (1 - beta2 ** t)

            visual_prompt = (
                visual_prompt - self.lr * m_hat / (torch.sqrt(s_hat) + eps)
            ).detach().requires_grad_(True)

            if t == 1 or t == self.T:
                print(f"    [Optim step {t}/{self.T}] loss={loss.item():.4f}")

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
            self._visual_prompt = self.optimize_visual_prompt(
                image, instruction, depth_map
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
