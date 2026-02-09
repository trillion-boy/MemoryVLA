"""
Training-Free Depth-Aware Pipeline for MemoryVLA.

Enhances MemoryVLA's spatial perception by injecting depth features
at the input level and correcting actions at the output level,
WITHOUT modifying any model weights.

Architecture:
    Step A - Depth Feature Injection (ControlMLLM-inspired):
        RGB  → [DINOv2+SigLIP] → [MLP Projector] → rgb_projected   [B,257,4096]
        Depth → INFERNO colormap → [DINOv2+SigLIP] → [MLP Projector] → depth_projected [B,257,4096]
        final_visual_tokens = rgb_projected + alpha * depth_projected

    Step B - Hybrid Action Correction:
        Uses ManiSkill ground-truth depth for Z-axis safety (prevent table collision).

Usage (Colab):
    from evaluation.maniskill2.depth_aware_pipeline import DepthAwarePipeline

    pipeline = DepthAwarePipeline(vla, depth_model=depth_model, alpha=0.3)
    actions, extra = pipeline.predict_action(
        image=pil_image, instruction="...", obs=obs,
        unnorm_key="libero_object_no_noops",
    )
"""

import cv2
import numpy as np
import torch
from PIL import Image
from typing import Optional, Tuple, Any


class DepthAwarePipeline:
    """
    Training-free depth-aware wrapper for MemoryVLA.

    Injects depth features into the visual token stream via a forward hook
    on the frozen MLP projector (Step A), and applies Z-axis safety
    correction using ground-truth depth (Step B).

    All MemoryVLA weights remain frozen. No training required.
    """

    def __init__(
        self,
        vla,
        depth_model=None,
        alpha: float = 0.3,
        safe_z_margin: float = 0.85,
        use_depth_injection: bool = True,
        use_action_correction: bool = True,
        colormap: int = cv2.COLORMAP_INFERNO,
    ):
        """
        Args:
            vla: Loaded MemoryVLA model (weights frozen)
            depth_model: Depth Anything V2 model (optional). If None, uses GT depth
                         from ManiSkill observation for Step A as well.
            alpha: Mixing weight for depth token injection.
                   visual_tokens = rgb_tokens + alpha * depth_tokens
                   Recommended range: 0.1 ~ 0.5
            safe_z_margin: Proximity threshold for Z-axis safety (0~1 normalized).
                          Higher = more conservative. 0.85 means clamp when gripper
                          depth is within top 15% of the depth range.
            use_depth_injection: Enable Step A (depth feature injection)
            use_action_correction: Enable Step B (action correction)
            colormap: OpenCV colormap for depth-to-RGB conversion.
                      cv2.COLORMAP_INFERNO recommended (DINOv2 is color-aware).
        """
        self.vla = vla
        self.depth_model = depth_model
        self.alpha = alpha
        self.safe_z_margin = safe_z_margin
        self.use_depth_injection = use_depth_injection
        self.use_action_correction = use_action_correction
        self.colormap = colormap

        # Internal state for hook
        self._depth_projected = None
        self._hook_handle = None

    # ================================================================
    # Step A: Depth Feature Injection
    # ================================================================

    def depth_to_colormap(self, depth_map: np.ndarray) -> Image.Image:
        """
        Convert 1-channel depth map to 3-channel INFERNO-colorized image.

        DINOv2 is trained on natural RGB images and leverages color information.
        INFERNO colormap encodes depth as a gradient:
            dark purple (near) → orange (mid) → bright yellow (far)
        This gives the backbone meaningful color variation for spatial reasoning.

        Args:
            depth_map: Depth values as numpy array (H, W) or (H, W, 1).

        Returns:
            PIL Image (H, W, 3) with colormap applied.
        """
        if isinstance(depth_map, torch.Tensor):
            depth_map = depth_map.cpu().numpy()

        if depth_map.ndim == 3:
            depth_map = depth_map.squeeze(-1)

        # Clip to foreground range (exclude background/sky)
        # Background pixels can dominate even the 99th percentile,
        # so we use median-based clipping: median is robust to background outliers
        # since the table surface (foreground) occupies >50% of pixels.
        valid = depth_map[(depth_map > 0) & np.isfinite(depth_map)]
        if len(valid) == 0:
            depth_norm = np.zeros_like(depth_map, dtype=np.uint8)
        else:
            median = np.median(valid)
            d_min = np.percentile(valid, 1)
            d_max = median * 1.5  # Anything beyond 1.5x median is background
            if d_max - d_min > 1e-6:
                clipped = np.clip(depth_map, d_min, d_max)
                depth_norm = ((clipped - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            else:
                depth_norm = np.zeros_like(depth_map, dtype=np.uint8)

        # CLAHE: boost local contrast so subtle depth differences
        # (e.g. cube vs table surface) become visible
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        depth_norm = clahe.apply(depth_norm)

        # Apply INFERNO colormap (returns BGR) → convert to RGB
        depth_bgr = cv2.applyColorMap(depth_norm, self.colormap)
        depth_rgb = cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2RGB)

        return Image.fromarray(depth_rgb)

    def compute_depth_tokens(self, depth_image: Image.Image) -> torch.Tensor:
        """
        Compute depth visual tokens by passing INFERNO-colored depth image
        through the SAME frozen vision backbone + MLP projector.

        This is truly training-free: we reuse the existing frozen components.
        The backbone (DINOv2+SigLIP) extracts spatial features from the colormap,
        and the projector maps them to the LLM embedding space.

        Pipeline:
            depth_image → image_transform → DINOv2+SigLIP → [B,257,2176]
                        → MLP Projector → [B,257,4096]

        Args:
            depth_image: PIL Image with INFERNO colormap applied.

        Returns:
            depth_projected: Tensor [B, 257, 4096] (same shape as rgb_projected)
        """
        vlm = self.vla.vlm
        model_dtype = next(self.vla.parameters()).dtype

        # Use the SAME image transform as the RGB pipeline
        image_transform = vlm.vision_backbone.image_transform

        # Transform depth image (resize, normalize, etc.)
        depth_pixel_values = image_transform(depth_image)
        if isinstance(depth_pixel_values, torch.Tensor):
            depth_pixel_values = depth_pixel_values[None, ...].to(vlm.device, dtype=model_dtype)
        elif isinstance(depth_pixel_values, dict):
            depth_pixel_values = {
                k: v[None, ...].to(vlm.device, dtype=model_dtype)
                for k, v in depth_pixel_values.items()
            }

        # Run through frozen backbone + projector (no grad needed)
        with torch.inference_mode():
            depth_features = vlm.vision_backbone(depth_pixel_values)
            depth_projected = vlm.projector(depth_features)

        return depth_projected

    def _projector_hook(self, module, input, output):
        """
        Forward hook injected on the MLP Projector.

        Fires when the projector processes RGB features during the main forward pass.
        Adds pre-computed depth tokens:
            output = rgb_projected + alpha * depth_projected
        """
        if self._depth_projected is not None:
            return output + self.alpha * self._depth_projected
        return output

    # ================================================================
    # Step B: Action Correction
    # ================================================================

    def correct_action(
        self,
        actions: np.ndarray,
        gt_depth: np.ndarray,
    ) -> np.ndarray:
        """
        Z-axis safety correction using ManiSkill ground-truth depth.

        Prevents the gripper from pressing into the table surface by analyzing
        the depth distribution in the gripper operating region.

        Logic:
            1. Compute depth statistics in the center region (gripper area)
            2. Estimate proximity: how close is the closest surface to the camera
               relative to the table surface depth
            3. If proximity exceeds threshold, clamp downward (negative dz) motion

        Args:
            actions: VLA output actions [N, 7] (dx, dy, dz, drx, dry, drz, gripper)
            gt_depth: Ground-truth depth from ManiSkill (H, W)

        Returns:
            Corrected actions [N, 7]
        """
        corrected = actions.copy()

        if gt_depth is None:
            return corrected

        if isinstance(gt_depth, torch.Tensor):
            gt_depth = gt_depth.cpu().numpy()
        if gt_depth.ndim == 3:
            gt_depth = gt_depth.squeeze(-1)

        h, w = gt_depth.shape

        # Center region: proxy for where the gripper operates
        center_depth = gt_depth[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3]
        valid = center_depth[center_depth > 0]

        if len(valid) == 0:
            return corrected

        # Depth statistics
        closest_surface = np.percentile(valid, 5)    # Near surface (object/gripper)
        table_surface = np.percentile(valid, 95)      # Far surface (table)

        if table_surface < 1e-6:
            return corrected

        # Normalized proximity: 0 = far from table, 1 = at table level
        # When closest_surface ≈ table_surface, proximity ≈ 0 (everything is flat = at table)
        proximity = 1.0 - (closest_surface / table_surface)

        # If the depth range is very small, the gripper is likely near the table
        # In that case, restrict downward motion
        if proximity < (1.0 - self.safe_z_margin):
            for i in range(corrected.shape[0]):
                if corrected[i, 2] < -0.002:
                    corrected[i, 2] = -0.002  # Allow only minimal descent

        return corrected

    # ================================================================
    # Depth Extraction Helpers
    # ================================================================

    def get_depth_from_obs(self, obs: dict) -> Optional[np.ndarray]:
        """
        Extract ground-truth depth from ManiSkill RGBD observation.

        Args:
            obs: ManiSkill observation dictionary

        Returns:
            Depth map (H, W) as numpy array, or None if not available.
        """
        # ManiSkill (new) format: obs["sensor_data"][cam]["depth"]
        if "sensor_data" in obs:
            for cam in ["base_camera", "hand_camera"]:
                if cam in obs["sensor_data"] and "depth" in obs["sensor_data"][cam]:
                    depth = obs["sensor_data"][cam]["depth"]
                    if isinstance(depth, torch.Tensor):
                        depth = depth.cpu().numpy()
                    if depth.ndim == 4:
                        depth = depth[0]  # Remove batch dim
                    if depth.ndim == 3:
                        depth = depth.squeeze(-1)  # Remove channel dim
                    return depth

            # Fallback: first available camera
            for cam in obs["sensor_data"]:
                if "depth" in obs["sensor_data"][cam]:
                    depth = obs["sensor_data"][cam]["depth"]
                    if isinstance(depth, torch.Tensor):
                        depth = depth.cpu().numpy()
                    if depth.ndim == 4:
                        depth = depth[0]
                    if depth.ndim == 3:
                        depth = depth.squeeze(-1)
                    return depth

        # ManiSkill2 (old) format: obs["image"][cam]["depth"]
        if "image" in obs:
            for cam in ["base_camera", "hand_camera"]:
                if cam in obs["image"] and "depth" in obs["image"][cam]:
                    depth = obs["image"][cam]["depth"]
                    if isinstance(depth, torch.Tensor):
                        depth = depth.cpu().numpy()
                    if depth.ndim == 3:
                        depth = depth.squeeze(-1)
                    return depth

        return None

    def estimate_depth(self, image: Image.Image) -> Optional[np.ndarray]:
        """
        Run Depth Anything V2 on an RGB image.

        Args:
            image: PIL Image (RGB)

        Returns:
            Estimated depth map (H, W) as numpy array, or None.
        """
        if self.depth_model is None:
            return None

        rgb_np = np.array(image)

        # Support different Depth Anything V2 interfaces
        if hasattr(self.depth_model, 'infer_image'):
            # Official Depth Anything V2 API
            return self.depth_model.infer_image(rgb_np)
        elif hasattr(self.depth_model, '__call__'):
            # HuggingFace pipeline interface
            result = self.depth_model(image)
            depth = result["depth"]
            if isinstance(depth, Image.Image):
                return np.array(depth).astype(np.float32)
            return np.array(depth)

        return None

    # ================================================================
    # Main Prediction Method
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
        episode_first_frame: str = 'False',
        **kwargs,
    ) -> Tuple[np.ndarray, Any]:
        """
        Predict action with depth-aware enhancements.

        Step A: Inject depth features into visual token stream
        Step B: Correct actions using GT depth (Z-axis safety)

        Args:
            image: RGB observation as PIL Image
            instruction: Task instruction string
            obs: ManiSkill observation dict (provides GT depth for Step B,
                 and fallback depth for Step A if no depth_model)
            unnorm_key: Action unnormalization key
            cfg_scale: Classifier-free guidance scale
            use_ddim: Use DDIM sampling
            num_ddim_steps: Number of DDIM steps
            episode_first_frame: 'True' or 'False'

        Returns:
            Tuple of (actions [N, action_dim], extra_info)
        """
        # ==================================
        # Step A: Depth Feature Injection
        # ==================================
        if self.use_depth_injection:
            # Obtain depth map for injection
            depth_for_injection = None

            if self.depth_model is not None:
                # Primary: Use Depth Anything V2 (monocular estimation)
                depth_for_injection = self.estimate_depth(image)
            elif obs is not None:
                # Fallback: Use ManiSkill GT depth
                depth_for_injection = self.get_depth_from_obs(obs)

            if depth_for_injection is not None:
                # Convert depth → INFERNO colormap → 3-channel image
                depth_colored = self.depth_to_colormap(depth_for_injection)

                # Compute depth tokens through same frozen backbone + projector
                self._depth_projected = self.compute_depth_tokens(depth_colored)

                # Register hook: projector output += alpha * depth_projected
                self._hook_handle = self.vla.vlm.projector.register_forward_hook(
                    self._projector_hook
                )

        # ==================================
        # Run MemoryVLA (with depth hook active)
        # ==================================
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
            # Always cleanup hook (even if prediction fails)
            if self._hook_handle is not None:
                self._hook_handle.remove()
                self._hook_handle = None
            self._depth_projected = None

        # ==================================
        # Step B: Action Correction
        # ==================================
        if self.use_action_correction and obs is not None:
            gt_depth = self.get_depth_from_obs(obs)
            if gt_depth is not None:
                actions = self.correct_action(actions, gt_depth)

        return actions, extra


def load_depth_anything_v2(model_size: str = "Small", device: str = "cuda"):
    """
    Load Depth Anything V2 from HuggingFace.

    Args:
        model_size: "Small" (~25MB VRAM), "Base" (~100MB), or "Large" (~400MB)
        device: Device to load model on

    Returns:
        Depth estimation model with .infer_image() method

    Usage in Colab:
        !pip install transformers -q
        depth_model = load_depth_anything_v2("Small")
    """
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    model_id = f"depth-anything/Depth-Anything-V2-{model_size}-hf"
    print(f"Loading Depth Anything V2 ({model_size}) from {model_id}...")

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    model = model.to(device).eval()

    class _DepthAnythingV2Wrapper:
        """Wrapper to provide .infer_image() interface."""

        def __init__(self, model, processor, device):
            self._model = model
            self._processor = processor
            self._device = device

        @torch.inference_mode()
        def infer_image(self, rgb: np.ndarray) -> np.ndarray:
            """
            Estimate depth from RGB image.

            Args:
                rgb: numpy array (H, W, 3), uint8

            Returns:
                depth: numpy array (H, W), float32 (relative depth)
            """
            pil_image = Image.fromarray(rgb) if isinstance(rgb, np.ndarray) else rgb
            inputs = self._processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            outputs = self._model(**inputs)
            depth = outputs.predicted_depth

            # Interpolate to original image size
            h, w = rgb.shape[:2] if isinstance(rgb, np.ndarray) else (pil_image.height, pil_image.width)
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()

            return depth

    wrapper = _DepthAnythingV2Wrapper(model, processor, device)
    print(f"Depth Anything V2 ({model_size}) loaded successfully.")
    return wrapper
