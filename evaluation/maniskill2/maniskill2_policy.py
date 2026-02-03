"""
ManiSkill2 native policy wrapper for MemoryVLA.
No SimplerEnv dependency - directly uses ManiSkill2 API.

This enables cross-environment generalization testing:
- Train on LIBERO (Franka Panda)
- Test on ManiSkill2 (Franka Panda) with unseen environments and objects
"""

import os
from collections import deque
from typing import Optional

import cv2 as cv
import numpy as np
import torch
from PIL import Image
from transforms3d.euler import euler2axangle

from vla import load_vla


class ManiSkill2VLAPolicy:
    """
    VLA Policy wrapper for ManiSkill2 environments.

    Converts MemoryVLA outputs to ManiSkill2 action format.
    Supports pd_ee_delta_pose control mode (default for Franka Panda).
    """

    def __init__(
        self,
        saved_model_path: str,
        unnorm_key: str = "libero_spatial_no_noops",
        image_size: list[int] = [224, 224],
        action_dim: int = 7,
        action_model_type: str = "DiT-L",
        future_action_window_size: int = 15,
        action_scale: float = 1.0,
        cfg_scale: float = 1.5,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        use_bf16: bool = False,
        action_ensemble: bool = True,
        action_ensemble_horizon: int = 7,
        adaptive_ensemble_alpha: float = 0.1,
        **kwargs,
    ):
        """
        Initialize the ManiSkill2 VLA Policy.

        Args:
            saved_model_path: Path to the trained MemoryVLA checkpoint
            unnorm_key: Key for action unnormalization statistics (e.g., "libero_spatial_no_noops")
            image_size: Input image size for the model
            action_dim: Action dimension (7 for EEF pose + gripper)
            action_scale: Scaling factor for actions
            cfg_scale: Classifier-free guidance scale
            use_ddim: Whether to use DDIM sampling
            num_ddim_steps: Number of DDIM steps
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.unnorm_key = unnorm_key
        self.image_size = image_size
        self.action_scale = action_scale
        self.cfg_scale = cfg_scale
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.action_ensemble = action_ensemble
        self.action_ensemble_horizon = action_ensemble_horizon

        # Load the VLA model
        print(f"Loading MemoryVLA from: {saved_model_path}")
        print(f"Using unnorm_key: {unnorm_key}")

        self.vla = load_vla(
            saved_model_path,
            load_for_training=False,
            action_model_type=action_model_type,
            future_action_window_size=future_action_window_size,
            action_dim=action_dim,
            **kwargs,
        )

        if use_bf16:
            self.vla.vlm = self.vla.vlm.to(torch.bfloat16)
        self.vla = self.vla.to("cuda").eval()

        # Action ensemble for smoother control
        if self.action_ensemble:
            from evaluation.simpler_env.adaptive_ensemble import AdaptiveEnsembler
            self.action_ensembler = AdaptiveEnsembler(
                action_ensemble_horizon,
                adaptive_ensemble_alpha
            )
        else:
            self.action_ensembler = None

        # State tracking
        self.task_description = None
        self.previous_gripper_action = None

    def reset(self, task_description: str):
        """
        Reset the policy for a new episode.

        Args:
            task_description: Natural language task instruction
        """
        self.task_description = task_description
        self.previous_gripper_action = None
        if self.action_ensembler is not None:
            self.action_ensembler.reset()

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to model input size."""
        return cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)

    def predict_action(
        self,
        image: np.ndarray,
        episode_first_frame: bool = False,
    ) -> np.ndarray:
        """
        Predict action from RGB image observation.

        Args:
            image: RGB image observation (H, W, 3), uint8
            episode_first_frame: Whether this is the first frame of the episode

        Returns:
            action: np.ndarray of shape (7,) for ManiSkill2 pd_ee_delta_pose control
                    [delta_x, delta_y, delta_z, delta_ax, delta_ay, delta_az, gripper]
        """
        assert self.task_description is not None, "Call reset() with task description first"
        assert image.dtype == np.uint8, "Image should be uint8"

        # Convert to PIL Image for model input
        pil_image = Image.fromarray(image)

        # Get raw action prediction from VLA
        raw_actions, _ = self.vla.predict_action(
            image=pil_image,
            instruction=self.task_description,
            unnorm_key=self.unnorm_key,
            cfg_scale=self.cfg_scale,
            use_ddim=self.use_ddim,
            num_ddim_steps=self.num_ddim_steps,
            episode_first_frame='True' if episode_first_frame else 'False',
        )

        # Apply action ensemble if enabled
        if self.action_ensemble and self.action_ensembler is not None:
            raw_actions = self.action_ensembler.ensemble_action(raw_actions)[None]

        # Extract action components
        # raw_actions shape: (1, 7) -> [x, y, z, roll, pitch, yaw, gripper]
        delta_pos = raw_actions[0, :3] * self.action_scale
        delta_rot_euler = raw_actions[0, 3:6]  # roll, pitch, yaw
        gripper_action = raw_actions[0, 6]

        # Convert euler to axis-angle for ManiSkill2
        roll, pitch, yaw = delta_rot_euler
        axis, angle = euler2axangle(roll, pitch, yaw)
        delta_rot_axangle = axis * angle * self.action_scale

        # Process gripper action: normalize to [-1, +1] and binarize
        # LIBERO uses [0, 1] -> need to convert to [-1, +1]
        # For ManiSkill2 pd_ee_delta_pose: -1 = close, +1 = open (verify this)
        gripper_normalized = 2.0 * (gripper_action > 0.5) - 1.0

        # Combine into ManiSkill2 action format
        # pd_ee_delta_pose expects: [delta_x, delta_y, delta_z, delta_ax, delta_ay, delta_az, gripper]
        action = np.concatenate([
            delta_pos,
            delta_rot_axangle,
            [gripper_normalized]
        ])

        return action.astype(np.float32)

    def predict_action_dict(
        self,
        image: np.ndarray,
        episode_first_frame: bool = False,
    ) -> dict:
        """
        Predict action and return as dictionary (for debugging/logging).

        Returns:
            dict with keys: 'delta_pos', 'delta_rot', 'gripper', 'full_action'
        """
        action = self.predict_action(image, episode_first_frame)

        return {
            'delta_pos': action[:3],
            'delta_rot': action[3:6],
            'gripper': action[6],
            'full_action': action,
        }
