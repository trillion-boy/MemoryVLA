"""
Grounding DINO + SAM detection test on ManiSkill2 rendered images.

Tests whether Grounding DINO can detect objects (cubes) in simulation-rendered
images, as a replacement for depth-based mask generation in the ControlMLLM pipeline.

Usage (Colab):
    # ============================================
    # Cell 1: Install Grounding DINO + SAM deps
    # ============================================
    !pip install -q segment-anything-py

    # ============================================
    # Cell 2: Run the test
    # ============================================
    from evaluation.maniskill2.test_grounding_dino_detection import run_full_test
    results = run_full_test(save_dir="/content/grounding_dino_test")
"""

import os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ================================================================
# ManiSkill2 Image Rendering
# ================================================================

def render_maniskill2_images(
    num_seeds: int = 5,
    env_names: List[str] = None,
    save_dir: Optional[str] = None,
) -> Dict[str, List[Tuple[np.ndarray, dict]]]:
    """
    Render RGB images from ManiSkill2 environments.

    Returns:
        Dict[env_name -> List[(rgb_image, env_info)]]
    """
    import gymnasium as gym
    try:
        import mani_skill.envs
    except ModuleNotFoundError:
        import mani_skill2.envs

    if env_names is None:
        env_names = ["PickCube-v1"]

    results = {}
    for env_name in env_names:
        images = []
        print(f"\n  Rendering {env_name}...")
        try:
            env = gym.make(
                env_name,
                obs_mode="rgbd",
                control_mode="pd_ee_delta_pose",
            )
        except Exception as e:
            print(f"    Failed to create {env_name}: {e}")
            continue

        for seed in range(num_seeds):
            try:
                obs, info = env.reset(seed=seed)

                # Extract RGB from observation
                rgb = _extract_rgb(obs)
                if rgb is None:
                    print(f"    Seed {seed}: Could not extract RGB")
                    continue

                # Also extract depth if available
                depth = _extract_depth(obs)

                env_info = {
                    "env_name": env_name,
                    "seed": seed,
                    "depth": depth,
                }
                images.append((rgb, env_info))
                print(f"    Seed {seed}: RGB {rgb.shape}, "
                      f"range [{rgb.min()}-{rgb.max()}]"
                      + (f", depth range [{depth.min():.3f}-{depth.max():.3f}]"
                         if depth is not None else ""))

                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    pil = Image.fromarray(rgb)
                    pil.save(os.path.join(save_dir, f"{env_name}_seed{seed}.png"))

            except Exception as e:
                print(f"    Seed {seed}: Error - {e}")

        env.close()
        results[env_name] = images
        print(f"  {env_name}: {len(images)} images rendered")

    return results


def _extract_rgb(obs) -> Optional[np.ndarray]:
    """Extract RGB from ManiSkill observation (handles v2 and v3 formats)."""
    # ManiSkill3 format: obs["sensor_data"]["base_camera"]["rgb"]
    if "sensor_data" in obs:
        for cam in ["base_camera", "hand_camera"]:
            if cam in obs["sensor_data"] and "rgb" in obs["sensor_data"][cam]:
                rgb = obs["sensor_data"][cam]["rgb"]
                if isinstance(rgb, torch.Tensor):
                    rgb = rgb.cpu().numpy()
                if rgb.ndim == 4:
                    rgb = rgb[0]
                if rgb.dtype == np.float32 or rgb.max() <= 1.0:
                    rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
                return rgb[:, :, :3]  # Drop alpha if present

    # ManiSkill2 format: obs["image"]["base_camera"]["rgb"]
    if "image" in obs:
        for cam in ["base_camera", "hand_camera"]:
            if cam in obs["image"] and "rgb" in obs["image"][cam]:
                rgb = obs["image"][cam]["rgb"]
                if isinstance(rgb, torch.Tensor):
                    rgb = rgb.cpu().numpy()
                if rgb.ndim == 4:
                    rgb = rgb[0]
                if rgb.dtype != np.uint8:
                    rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
                return rgb[:, :, :3]

    return None


def _extract_depth(obs) -> Optional[np.ndarray]:
    """Extract depth from ManiSkill observation."""
    for key in ["sensor_data", "image"]:
        if key in obs:
            for cam in ["base_camera", "hand_camera"]:
                if cam in obs[key] and "depth" in obs[key][cam]:
                    depth = obs[key][cam]["depth"]
                    if isinstance(depth, torch.Tensor):
                        depth = depth.cpu().numpy()
                    if depth.ndim == 4:
                        depth = depth[0]
                    if depth.ndim == 3:
                        depth = depth[:, :, 0]
                    return depth.astype(np.float32)
    return None


# ================================================================
# Grounding DINO Detection
# ================================================================

def load_grounding_dino(model_id: str = "IDEA-Research/grounding-dino-tiny"):
    """Load Grounding DINO from HuggingFace transformers."""
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    print(f"  Loading Grounding DINO: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        print("  -> Moved to CUDA")

    return processor, model


def detect_objects(
    processor,
    model,
    image: Image.Image,
    text_prompts: List[str],
    box_threshold: float = 0.2,
    text_threshold: float = 0.2,
) -> List[Dict]:
    """
    Run Grounding DINO detection.

    Args:
        processor: GroundingDINO processor
        model: GroundingDINO model
        image: PIL Image
        text_prompts: List of text queries, e.g. ["red cube", "cube"]
        box_threshold: Confidence threshold for boxes
        text_threshold: Confidence threshold for text matching

    Returns:
        List of detections, each with {box, score, label, prompt}
    """
    device = next(model.parameters()).device
    all_detections = []

    for prompt in text_prompts:
        # Grounding DINO expects text ending with "."
        text = prompt if prompt.endswith(".") else prompt + "."

        inputs = processor(images=image, text=text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],  # (H, W)
        )[0]

        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"]

        for i in range(len(boxes)):
            all_detections.append({
                "box": boxes[i],       # [x1, y1, x2, y2]
                "score": float(scores[i]),
                "label": labels[i],
                "prompt": prompt,
            })

    # Sort by score descending
    all_detections.sort(key=lambda d: d["score"], reverse=True)
    return all_detections


# ================================================================
# SAM Segmentation
# ================================================================

def load_sam(model_id: str = "facebook/sam-vit-base"):
    """Load SAM from HuggingFace transformers."""
    from transformers import SamModel, SamProcessor

    print(f"  Loading SAM: {model_id}")
    processor = SamProcessor.from_pretrained(model_id)
    model = SamModel.from_pretrained(model_id)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        print("  -> Moved to CUDA")

    return processor, model


def segment_from_box(
    sam_processor,
    sam_model,
    image: Image.Image,
    box: np.ndarray,
) -> np.ndarray:
    """
    Generate segmentation mask from bounding box using SAM.

    Args:
        sam_processor: SAM processor
        sam_model: SAM model
        image: PIL Image
        box: [x1, y1, x2, y2] bounding box

    Returns:
        Binary mask (H, W) as np.ndarray
    """
    device = next(sam_model.parameters()).device

    # SAM expects boxes as [[x1, y1, x2, y2]]
    input_boxes = [[[float(box[0]), float(box[1]), float(box[2]), float(box[3])]]]

    inputs = sam_processor(
        image,
        input_boxes=input_boxes,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = sam_model(**inputs)

    masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )

    # Take the best mask (highest IoU prediction)
    mask = masks[0][0]  # [num_masks, H, W]
    if mask.ndim == 3:
        # SAM outputs 3 mask candidates; pick the one with highest score
        if hasattr(outputs, "iou_scores"):
            best_idx = outputs.iou_scores[0].argmax().item()
        else:
            best_idx = 0
        mask = mask[best_idx]

    return mask.numpy().astype(np.uint8)


# ================================================================
# Mask → ControlMLLM 16x16 Grid Conversion
# ================================================================

def mask_to_patch_grid(
    mask: np.ndarray,
    grid_h: int = 16,
    grid_w: int = 16,
    threshold: float = 0.1,
) -> torch.Tensor:
    """
    Convert pixel-level binary mask to 16x16 patch grid for ControlMLLM.

    Args:
        mask: Binary mask (H, W)
        grid_h, grid_w: Patch grid dimensions (MemoryVLA uses 16x16 = 256 patches)
        threshold: Minimum fraction of mask pixels in a patch to activate it

    Returns:
        Tensor [1, grid_h*grid_w] with values 0 or 1
    """
    import torch.nn.functional as F

    h, w = mask.shape[:2]
    mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    # Average pool to grid size
    pooled = F.adaptive_avg_pool2d(mask_tensor, (grid_h, grid_w))

    # Threshold: if >threshold fraction of pixels in patch are masked
    grid = (pooled > threshold).float()

    return grid.view(1, grid_h * grid_w)


# ================================================================
# Visualization
# ================================================================

def visualize_detection(
    image: Image.Image,
    detections: List[Dict],
    sam_mask: Optional[np.ndarray] = None,
    patch_grid: Optional[torch.Tensor] = None,
    title: str = "",
    save_path: Optional[str] = None,
):
    """Visualize detection results: boxes, SAM mask, and patch grid."""
    n_panels = 2 + (1 if sam_mask is not None else 0) + (1 if patch_grid is not None else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    # Panel 1: Original image
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Panel 2: Detections (boxes)
    img_with_boxes = np.array(image).copy()
    import cv2
    for det in detections:
        box = det["box"].astype(int)
        score = det["score"]
        label = det["label"]
        color = (0, 255, 0) if score > 0.3 else (255, 165, 0)
        cv2.rectangle(img_with_boxes, (box[0], box[1]), (box[2], box[3]), color, 2)
        text = f'{label}: {score:.2f}'
        cv2.putText(img_with_boxes, text, (box[0], max(box[1] - 5, 15)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    axes[1].imshow(img_with_boxes)
    axes[1].set_title(f"Grounding DINO ({len(detections)} det)")
    axes[1].axis("off")

    panel_idx = 2

    # Panel 3: SAM mask overlay
    if sam_mask is not None:
        overlay = np.array(image).copy().astype(np.float32)
        mask_colored = np.zeros_like(overlay)
        mask_colored[:, :, 1] = 255  # Green
        mask_3d = np.stack([sam_mask] * 3, axis=-1).astype(np.float32)
        overlay = overlay * (1 - mask_3d * 0.4) + mask_colored * (mask_3d * 0.4)
        axes[panel_idx].imshow(overlay.astype(np.uint8))
        n_pixels = sam_mask.sum()
        axes[panel_idx].set_title(f"SAM Mask ({n_pixels} px)")
        axes[panel_idx].axis("off")
        panel_idx += 1

    # Panel 4: 16x16 patch grid
    if patch_grid is not None:
        grid_2d = patch_grid.view(16, 16).numpy()
        axes[panel_idx].imshow(grid_2d, cmap="Reds", vmin=0, vmax=1,
                                interpolation="nearest")
        n_active = (grid_2d > 0).sum()
        axes[panel_idx].set_title(f"Patch Grid ({n_active}/256 active)")
        axes[panel_idx].axis("off")
        panel_idx += 1

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


# ================================================================
# Main Test Functions
# ================================================================

def test_grounding_dino_on_images(
    images: List[Tuple[np.ndarray, dict]],
    text_prompts: List[str] = None,
    box_threshold: float = 0.2,
    text_threshold: float = 0.2,
    save_dir: Optional[str] = None,
    gdino_model_id: str = "IDEA-Research/grounding-dino-tiny",
    sam_model_id: str = "facebook/sam-vit-base",
) -> List[Dict]:
    """
    Test Grounding DINO + SAM on a list of images.

    Args:
        images: List of (rgb_np, info_dict) from render_maniskill2_images
        text_prompts: Prompts to test, default covers various descriptions
        box_threshold: Detection confidence threshold
        save_dir: Directory to save visualizations

    Returns:
        List of result dicts per image
    """
    if text_prompts is None:
        text_prompts = [
            "red cube",
            "cube",
            "small red object",
            "red block",
            "object on table",
        ]

    # Load models
    print("\n=== Loading Models ===")
    gdino_proc, gdino_model = load_grounding_dino(gdino_model_id)
    sam_proc, sam_model = load_sam(sam_model_id)

    results = []
    print(f"\n=== Testing {len(images)} images, {len(text_prompts)} prompts each ===")

    for idx, (rgb, info) in enumerate(images):
        env_name = info.get("env_name", "unknown")
        seed = info.get("seed", idx)
        print(f"\n--- Image {idx}: {env_name} seed={seed} ---")

        pil_image = Image.fromarray(rgb)

        # Run Grounding DINO with all prompts
        detections = detect_objects(
            gdino_proc, gdino_model, pil_image,
            text_prompts, box_threshold, text_threshold,
        )

        print(f"  Total detections: {len(detections)}")
        for det in detections:
            box = det["box"]
            print(f"    [{det['prompt']}] score={det['score']:.3f} "
                  f"box=({box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}) "
                  f"label='{det['label']}'")

        # SAM: segment the best detection (if any)
        sam_mask = None
        patch_grid = None
        if detections:
            best = detections[0]
            print(f"  Best detection: [{best['prompt']}] score={best['score']:.3f}")
            sam_mask = segment_from_box(sam_proc, sam_model, pil_image, best["box"])
            patch_grid = mask_to_patch_grid(sam_mask)
            n_active = (patch_grid > 0).sum().item()
            print(f"  SAM mask: {sam_mask.sum()} pixels, "
                  f"Patch grid: {n_active}/256 active patches")

        # Per-prompt summary
        prompt_results = {}
        for prompt in text_prompts:
            prompt_dets = [d for d in detections if d["prompt"] == prompt]
            best_score = prompt_dets[0]["score"] if prompt_dets else 0.0
            prompt_results[prompt] = {
                "n_detections": len(prompt_dets),
                "best_score": best_score,
                "detected": best_score > box_threshold,
            }

        result = {
            "env_name": env_name,
            "seed": seed,
            "detections": detections,
            "sam_mask": sam_mask,
            "patch_grid": patch_grid,
            "prompt_results": prompt_results,
        }
        results.append(result)

        # Visualize
        if save_dir:
            visualize_detection(
                pil_image, detections, sam_mask, patch_grid,
                title=f"{env_name} seed={seed}",
                save_path=os.path.join(save_dir, f"{env_name}_seed{seed}_detection.png"),
            )

    return results


def run_full_test(
    save_dir: str = "/content/grounding_dino_test",
    num_seeds: int = 5,
    env_names: List[str] = None,
    text_prompts: List[str] = None,
    box_threshold: float = 0.15,
    gdino_model_id: str = "IDEA-Research/grounding-dino-tiny",
    sam_model_id: str = "facebook/sam-vit-base",
):
    """
    Full test pipeline: render ManiSkill2 images → Grounding DINO → SAM → report.

    Usage in Colab:
        from evaluation.maniskill2.test_grounding_dino_detection import run_full_test
        results = run_full_test(save_dir="/content/grounding_dino_test")
    """
    if env_names is None:
        env_names = ["PickCube-v1"]

    os.makedirs(save_dir, exist_ok=True)

    # Step 1: Render images
    print("=" * 60)
    print("STEP 1: Rendering ManiSkill2 images")
    print("=" * 60)
    rendered = render_maniskill2_images(
        num_seeds=num_seeds,
        env_names=env_names,
        save_dir=os.path.join(save_dir, "raw_images"),
    )

    # Flatten all images
    all_images = []
    for env_name, imgs in rendered.items():
        all_images.extend(imgs)

    if not all_images:
        print("ERROR: No images rendered! Check ManiSkill2 installation.")
        return None

    # Step 2: Test detection
    print("\n" + "=" * 60)
    print("STEP 2: Grounding DINO + SAM Detection")
    print("=" * 60)
    results = test_grounding_dino_on_images(
        all_images,
        text_prompts=text_prompts,
        box_threshold=box_threshold,
        save_dir=os.path.join(save_dir, "detections"),
        gdino_model_id=gdino_model_id,
        sam_model_id=sam_model_id,
    )

    # Step 3: Summary report
    print("\n" + "=" * 60)
    print("STEP 3: Detection Summary")
    print("=" * 60)
    _print_summary(results)

    return results


def test_on_single_image(
    image: np.ndarray,
    text_prompts: List[str] = None,
    box_threshold: float = 0.15,
    save_path: Optional[str] = None,
    gdino_model_id: str = "IDEA-Research/grounding-dino-tiny",
    sam_model_id: str = "facebook/sam-vit-base",
) -> Dict:
    """
    Test on a single image (no ManiSkill2 needed).

    Usage in Colab:
        import numpy as np
        from PIL import Image
        img = np.array(Image.open("/path/to/maniskill_screenshot.png"))
        from evaluation.maniskill2.test_grounding_dino_detection import test_on_single_image
        result = test_on_single_image(img, save_path="/content/test_result.png")
    """
    if text_prompts is None:
        text_prompts = ["red cube", "cube", "small red object",
                        "red block", "object on table"]

    images = [(image, {"env_name": "custom", "seed": 0})]
    results = test_grounding_dino_on_images(
        images,
        text_prompts=text_prompts,
        box_threshold=box_threshold,
        save_dir=os.path.dirname(save_path) if save_path else None,
        gdino_model_id=gdino_model_id,
        sam_model_id=sam_model_id,
    )
    return results[0] if results else {}


def _print_summary(results: List[Dict]):
    """Print a detection summary table."""
    if not results:
        print("  No results to summarize.")
        return

    # Collect all prompts
    all_prompts = set()
    for r in results:
        all_prompts.update(r["prompt_results"].keys())
    all_prompts = sorted(all_prompts)

    # Header
    print(f"\n{'Prompt':<25} | {'Detected':<10} | {'Avg Score':<10} | {'Rate':<10}")
    print("-" * 65)

    for prompt in all_prompts:
        n_detected = 0
        total_score = 0.0
        n_total = 0
        for r in results:
            pr = r["prompt_results"].get(prompt, {})
            n_total += 1
            if pr.get("detected", False):
                n_detected += 1
            total_score += pr.get("best_score", 0.0)

        avg_score = total_score / max(n_total, 1)
        rate = n_detected / max(n_total, 1)
        print(f"{prompt:<25} | {n_detected}/{n_total:<8} | {avg_score:<10.3f} | {rate:<10.1%}")

    # Overall: any detection at all?
    n_with_any = sum(1 for r in results if r["detections"])
    print(f"\nOverall: {n_with_any}/{len(results)} images had at least one detection")

    # Patch grid stats
    patch_counts = []
    for r in results:
        if r["patch_grid"] is not None:
            n = (r["patch_grid"] > 0).sum().item()
            patch_counts.append(n)
    if patch_counts:
        print(f"Patch grid: avg={np.mean(patch_counts):.1f} active patches, "
              f"range=[{min(patch_counts)}, {max(patch_counts)}]")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", default="/content/grounding_dino_test")
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--env-names", nargs="+", default=["PickCube-v1"])
    parser.add_argument("--box-threshold", type=float, default=0.15)
    parser.add_argument("--gdino-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--sam-model", default="facebook/sam-vit-base")
    args = parser.parse_args()

    run_full_test(
        save_dir=args.save_dir,
        num_seeds=args.num_seeds,
        env_names=args.env_names,
        box_threshold=args.box_threshold,
        gdino_model_id=args.gdino_model,
        sam_model_id=args.sam_model,
    )
