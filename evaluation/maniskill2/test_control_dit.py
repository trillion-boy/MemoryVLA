"""
End-to-end test: Grounding DINO + SAM → ControlDiT → Action Prediction.

Tests the full ControlMLLM pipeline on ManiSkill2 PickCube-v1:
1. Render image from ManiSkill2
2. Resize to 224x224, run Grounding DINO + SAM → binary mask
3. Convert mask to 16x16 patch grid (target_grid)
4. Run MemoryVLA with ControlDiT optimization
5. Compare actions with vs without control

Usage in Colab (after cells 1-4 have loaded the model):

    # Cell: Install Grounding DINO deps (if not already)
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from transformers import SamModel, SamProcessor

    # Cell: Run test
    from evaluation.maniskill2.test_control_dit import run_control_test
    results = run_control_test(vla=vla, save_dir="/content/control_dit_test")
"""

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Optional, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ================================================================
# Grounding DINO + SAM → Target Grid
# ================================================================

def generate_target_grid(
    image_224: Image.Image,
    text_prompt: str = "red cube",
    box_threshold: float = 0.15,
    gdino_processor=None,
    gdino_model=None,
    sam_processor=None,
    sam_model=None,
    grid_size: int = 16,
) -> dict:
    """
    Full pipeline: image → Grounding DINO → SAM → 16x16 patch grid.

    Args:
        image_224: PIL Image (should be 224x224 for coordinate alignment)
        text_prompt: Object description for Grounding DINO
        box_threshold: Detection confidence threshold
        gdino_processor/model: Pre-loaded Grounding DINO (loaded if None)
        sam_processor/model: Pre-loaded SAM (loaded if None)
        grid_size: Patch grid dimension (16 for MemoryVLA)

    Returns:
        Dict with target_grid [256], sam_mask [H,W], detection info
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models if not provided
    if gdino_processor is None or gdino_model is None:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        print("  Loading Grounding DINO...")
        gdino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-tiny"
        ).eval().to(device)

    if sam_processor is None or sam_model is None:
        from transformers import SamModel, SamProcessor
        print("  Loading SAM...")
        sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        sam_model = SamModel.from_pretrained("facebook/sam-vit-base").eval().to(device)

    # Step 1: Grounding DINO detection
    text = text_prompt if text_prompt.endswith(".") else text_prompt + "."
    inputs = gdino_processor(images=image_224, text=text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = gdino_model(**inputs)

    results = gdino_processor.post_process_grounded_object_detection(
        outputs, inputs["input_ids"],
        box_threshold=box_threshold,
        text_threshold=box_threshold,
        target_sizes=[image_224.size[::-1]],
    )[0]

    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()

    if len(boxes) == 0:
        print(f"  WARNING: No detection for '{text_prompt}' (threshold={box_threshold})")
        # Return empty grid
        return {
            "target_grid": torch.zeros(grid_size * grid_size),
            "sam_mask": np.zeros((image_224.size[1], image_224.size[0]), dtype=np.uint8),
            "detection": None,
            "gdino_processor": gdino_processor, "gdino_model": gdino_model,
            "sam_processor": sam_processor, "sam_model": sam_model,
        }

    # Take best detection
    best_idx = scores.argmax()
    best_box = boxes[best_idx]
    best_score = scores[best_idx]
    print(f"  Detection: '{text_prompt}' score={best_score:.3f} "
          f"box=({best_box[0]:.0f},{best_box[1]:.0f},{best_box[2]:.0f},{best_box[3]:.0f})")

    # Step 2: SAM segmentation from box
    input_boxes = [[[float(best_box[0]), float(best_box[1]),
                     float(best_box[2]), float(best_box[3])]]]
    sam_inputs = sam_processor(image_224, input_boxes=input_boxes, return_tensors="pt")
    sam_inputs = {k: v.to(device) for k, v in sam_inputs.items()}

    with torch.no_grad():
        sam_outputs = sam_model(**sam_inputs)

    masks = sam_processor.image_processor.post_process_masks(
        sam_outputs.pred_masks.cpu(),
        sam_inputs["original_sizes"].cpu(),
        sam_inputs["reshaped_input_sizes"].cpu(),
    )
    sam_mask = masks[0][0]
    if sam_mask.ndim == 3:
        if hasattr(sam_outputs, "iou_scores"):
            best_mask_idx = sam_outputs.iou_scores[0].argmax().item()
        else:
            best_mask_idx = 0
        sam_mask = sam_mask[best_mask_idx]
    sam_mask = sam_mask.numpy().astype(np.uint8)
    print(f"  SAM mask: {sam_mask.sum()} pixels")

    # Step 3: Convert to 16x16 patch grid
    mask_t = torch.from_numpy(sam_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    pooled = F.adaptive_avg_pool2d(mask_t, (grid_size, grid_size))
    target_grid = (pooled > 0.1).float().flatten()
    n_active = (target_grid > 0).sum().item()
    print(f"  Target grid: {n_active}/{grid_size*grid_size} active patches")

    return {
        "target_grid": target_grid,
        "sam_mask": sam_mask,
        "detection": {"box": best_box, "score": best_score},
        "gdino_processor": gdino_processor, "gdino_model": gdino_model,
        "sam_processor": sam_processor, "sam_model": sam_model,
    }


# ================================================================
# Visualization
# ================================================================

def visualize_control_result(
    image_224: Image.Image,
    sam_mask: np.ndarray,
    target_grid: torch.Tensor,
    losses: list,
    actions_baseline: np.ndarray,
    actions_control: np.ndarray,
    title: str = "",
    save_path: Optional[str] = None,
):
    """Visualize: original, SAM mask, target grid, loss curve, action comparison."""
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    # 1: Original image
    axes[0].imshow(image_224)
    axes[0].set_title("Input (224x224)")
    axes[0].axis("off")

    # 2: SAM mask overlay
    overlay = np.array(image_224).copy().astype(np.float32)
    mask_colored = np.zeros_like(overlay)
    mask_colored[:, :, 1] = 255
    mask_3d = np.stack([sam_mask] * 3, axis=-1).astype(np.float32)
    overlay = overlay * (1 - mask_3d * 0.4) + mask_colored * (mask_3d * 0.4)
    axes[1].imshow(overlay.astype(np.uint8))
    axes[1].set_title(f"SAM Mask ({sam_mask.sum()} px)")
    axes[1].axis("off")

    # 3: 16x16 patch grid
    grid_2d = target_grid.view(16, 16).cpu().numpy()
    axes[2].imshow(grid_2d, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
    n_active = (grid_2d > 0).sum()
    axes[2].set_title(f"Target Grid ({n_active}/256)")
    axes[2].axis("off")

    # 4: Loss curve
    axes[3].plot(range(1, len(losses) + 1), losses, "b-o", linewidth=2, markersize=8)
    axes[3].set_xlabel("Optimization Step")
    axes[3].set_ylabel("Loss")
    axes[3].set_title(f"ControlDiT Loss ({losses[-1]:.4f})")
    axes[3].grid(True, alpha=0.3)

    # 5: Action comparison (first timestep)
    action_labels = ["dx", "dy", "dz", "rx", "ry", "rz", "grip"]
    x_pos = np.arange(len(action_labels))
    w = 0.35
    axes[4].bar(x_pos - w/2, actions_baseline[0], w, label="Baseline", alpha=0.7, color="gray")
    axes[4].bar(x_pos + w/2, actions_control[0], w, label="Controlled", alpha=0.7, color="red")
    axes[4].set_xticks(x_pos)
    axes[4].set_xticklabels(action_labels, fontsize=8)
    axes[4].set_title("Action Comparison (t=0)")
    axes[4].legend(fontsize=8)
    axes[4].grid(True, alpha=0.3, axis="y")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


# ================================================================
# Main Test
# ================================================================

def run_control_test(
    vla,
    save_dir: str = "/content/control_dit_test",
    env_name: str = "PickCube-v1",
    num_seeds: int = 3,
    text_prompt: str = "red cube",
    task_instruction: str = "pick up the small red cube from the table",
    unnorm_key: str = "libero_object_no_noops",
    control_steps: int = 5,
    control_alpha: float = 100.0,
    cfg_scale: float = 1.5,
    use_ddim: bool = True,
    num_ddim_steps: int = 10,
):
    """
    Full ControlDiT test: render → detect → optimize → compare actions.

    Args:
        vla: Loaded MemoryVLA model
        save_dir: Where to save results
        env_name: ManiSkill2 environment name
        num_seeds: Number of random seeds to test
        text_prompt: Grounding DINO prompt (e.g., "red cube")
        task_instruction: MemoryVLA task instruction
        unnorm_key: Dataset key for action un-normalization
        control_steps: T for ControlDiT optimization
        control_alpha: Learning rate for pv
    """
    os.makedirs(save_dir, exist_ok=True)

    # ====== Step 1: Render ManiSkill2 images ======
    print("=" * 60)
    print("STEP 1: Rendering ManiSkill2 images (224x224)")
    print("=" * 60)
    images_224 = _render_images_224(env_name, num_seeds, save_dir)

    if not images_224:
        print("ERROR: No images rendered!")
        return None

    # ====== Step 2: Load detection models (once) ======
    print("\n" + "=" * 60)
    print("STEP 2: Loading Grounding DINO + SAM")
    print("=" * 60)
    gdino_proc, gdino_model, sam_proc, sam_model = None, None, None, None

    results = []
    for idx, (pil_224, seed) in enumerate(images_224):
        print(f"\n{'='*60}")
        print(f"IMAGE {idx+1}/{len(images_224)}: seed={seed}")
        print(f"{'='*60}")

        # ====== Step 3: Generate target grid ======
        print("\n--- Generating target grid ---")
        grid_result = generate_target_grid(
            pil_224, text_prompt=text_prompt, box_threshold=0.15,
            gdino_processor=gdino_proc, gdino_model=gdino_model,
            sam_processor=sam_proc, sam_model=sam_model,
        )
        target_grid = grid_result["target_grid"]
        sam_mask = grid_result["sam_mask"]
        # Cache models
        gdino_proc = grid_result["gdino_processor"]
        gdino_model = grid_result["gdino_model"]
        sam_proc = grid_result["sam_processor"]
        sam_model = grid_result["sam_model"]

        if grid_result["detection"] is None:
            print("  Skipping (no detection)")
            continue

        # ====== Step 4: Baseline action (no control) ======
        print("\n--- Baseline action prediction ---")
        t0 = time.time()
        actions_base, norm_actions_base = vla.predict_action(
            pil_224, task_instruction,
            unnorm_key=unnorm_key, cfg_scale=cfg_scale,
            use_ddim=use_ddim, num_ddim_steps=num_ddim_steps,
            episode_first_frame='True',
        )
        t_base = time.time() - t0
        print(f"  Baseline action[0]: {actions_base[0].round(4)}")
        print(f"  Time: {t_base:.2f}s")

        # ====== Step 5: Controlled action (with ControlDiT) ======
        print("\n--- ControlDiT action prediction ---")
        t0 = time.time()
        actions_ctrl, norm_actions_ctrl, losses = vla.predict_action_with_control(
            pil_224, task_instruction, target_grid,
            unnorm_key=unnorm_key, cfg_scale=cfg_scale,
            use_ddim=use_ddim, num_ddim_steps=num_ddim_steps,
            episode_first_frame='True',
            control_steps=control_steps, control_alpha=control_alpha,
        )
        t_ctrl = time.time() - t0
        print(f"  Controlled action[0]: {actions_ctrl[0].round(4)}")
        print(f"  Losses: {[f'{l:.4f}' for l in losses]}")
        print(f"  Time: {t_ctrl:.2f}s (overhead: {t_ctrl-t_base:.2f}s)")

        # ====== Step 6: Action difference ======
        diff = np.abs(actions_ctrl - actions_base)
        print(f"\n  Action diff (L1): {diff.mean():.4f} (max: {diff.max():.4f})")

        # Visualize
        visualize_control_result(
            pil_224, sam_mask, target_grid, losses,
            actions_base, actions_ctrl,
            title=f"ControlDiT: {env_name} seed={seed}, prompt='{text_prompt}'",
            save_path=os.path.join(save_dir, f"control_seed{seed}.png"),
        )

        results.append({
            "seed": seed,
            "baseline_action": actions_base,
            "controlled_action": actions_ctrl,
            "losses": losses,
            "detection_score": grid_result["detection"]["score"],
            "time_baseline": t_base,
            "time_controlled": t_ctrl,
        })

    # ====== Summary ======
    if results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for r in results:
            diff = np.abs(r["controlled_action"] - r["baseline_action"]).mean()
            print(f"  seed={r['seed']}: detect={r['detection_score']:.3f}, "
                  f"loss={r['losses'][0]:.4f}→{r['losses'][-1]:.4f}, "
                  f"action_diff={diff:.4f}, "
                  f"overhead={r['time_controlled']-r['time_baseline']:.2f}s")

    return results


def _render_images_224(env_name, num_seeds, save_dir):
    """Render ManiSkill2 images and resize to 224x224."""
    import gymnasium as gym
    try:
        import mani_skill.envs
    except ModuleNotFoundError:
        import mani_skill2.envs

    images = []
    print(f"  Creating {env_name}...")
    try:
        env = gym.make(env_name, obs_mode="rgbd", control_mode="pd_ee_delta_pose")
    except Exception as e:
        print(f"  Failed: {e}")
        return images

    for seed in range(num_seeds):
        obs, _ = env.reset(seed=seed)
        rgb = _extract_rgb(obs)
        if rgb is None:
            continue

        # Resize to 224x224
        pil = Image.fromarray(rgb)
        pil_224 = pil.resize((224, 224), Image.BILINEAR)

        save_path = os.path.join(save_dir, "raw_images", f"seed{seed}_224.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pil_224.save(save_path)

        images.append((pil_224, seed))
        print(f"  Seed {seed}: {rgb.shape} → 224x224")

    env.close()
    return images


def _extract_rgb(obs):
    """Extract RGB from ManiSkill observation."""
    for key in ["sensor_data", "image"]:
        if key in obs:
            for cam in ["base_camera", "hand_camera"]:
                if cam in obs[key] and "rgb" in obs[key][cam]:
                    rgb = obs[key][cam]["rgb"]
                    if isinstance(rgb, torch.Tensor):
                        rgb = rgb.cpu().numpy()
                    if rgb.ndim == 4:
                        rgb = rgb[0]
                    if rgb.max() <= 1.0:
                        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
                    return rgb[:, :, :3]
    return None
