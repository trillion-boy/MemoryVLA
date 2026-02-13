"""
End-to-end test: Grounding DINO + SAM → Spatial Gating Control → Action Prediction.

Tests Method A (training-free spatial control) on ManiSkill2 PickCube-v1:
1. Render image from ManiSkill2
2. Resize to 224x224, run Grounding DINO + SAM → binary mask
3. Convert mask to 16x16 patch grid (target_grid)
4. Run MemoryVLA with SpatialGatingControl (replaces dead per_attn)
5. Compare actions: baseline vs spatial control at multiple scales

Key difference from test_control_dit.py:
- No optimization loop (training-free, instant)
- Replaces dead per_attn with manual spatial attention
- Tests multiple scale values to find optimal setting

Usage in Colab (after cells 1-4 have loaded the model):

    from evaluation.maniskill2.test_spatial_control import run_spatial_control_test
    results = run_spatial_control_test(vla=vla, save_dir="/content/spatial_control_test")
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

# Reuse grid generation and image rendering from test_control_dit
from evaluation.maniskill2.test_control_dit import (
    generate_target_grid,
    _render_images_224,
    _extract_rgb,
)


# ================================================================
# Visualization (adapted for spatial control)
# ================================================================

def visualize_spatial_control_result(
    image_224: Image.Image,
    sam_mask: np.ndarray,
    target_grid: torch.Tensor,
    actions_baseline: np.ndarray,
    actions_by_scale: dict,
    diagnostics: dict,
    title: str = "",
    save_path: Optional[str] = None,
):
    """Visualize: original, SAM mask, target grid, action comparison across scales."""
    n_scales = len(actions_by_scale)
    fig, axes = plt.subplots(1, 4 + 1, figsize=(25, 5))

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

    # 4: Focus diagnostics
    if diagnostics:
        labels = ["Random\nExpected", "Original\nper_attn", "Spatial\nOverride"]
        values = [
            diagnostics.get("expected_random_focus", 0),
            diagnostics.get("original_focus", 0),
            diagnostics.get("spatial_focus", 0),
        ]
        colors = ["gray", "orange", "green"]
        axes[3].bar(labels, values, color=colors, alpha=0.8)
        axes[3].set_ylabel("Focus on Target")
        axes[3].set_title("Attention Focus")
        axes[3].set_ylim(0, 1)
        axes[3].grid(True, alpha=0.3, axis="y")
    else:
        axes[3].text(0.5, 0.5, "No diagnostics", ha="center", va="center")
        axes[3].set_title("Diagnostics")

    # 5: Action comparison across scales
    action_labels = ["dx", "dy", "dz", "rx", "ry", "rz", "grip"]
    x_pos = np.arange(len(action_labels))
    n_bars = 1 + n_scales
    w = 0.8 / n_bars

    axes[4].bar(
        x_pos - 0.4 + w / 2,
        actions_baseline[0], w,
        label="Baseline", alpha=0.7, color="gray",
    )
    for i, (scale, actions) in enumerate(sorted(actions_by_scale.items())):
        axes[4].bar(
            x_pos - 0.4 + w * (i + 1.5),
            actions[0], w,
            label=f"scale={scale}", alpha=0.7,
        )
    axes[4].set_xticks(x_pos)
    axes[4].set_xticklabels(action_labels, fontsize=8)
    axes[4].set_title("Action Comparison (t=0)")
    axes[4].legend(fontsize=7)
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

def run_spatial_control_test(
    vla,
    save_dir: str = "/content/spatial_control_test",
    env_name: str = "PickCube-v1",
    num_seeds: int = 3,
    text_prompt: str = "red cube",
    task_instruction: str = "pick up the small red cube from the table",
    unnorm_key: str = "libero_object_no_noops",
    scales: Optional[list] = None,
    cfg_scale: float = 1.5,
    use_ddim: bool = True,
    num_ddim_steps: int = 10,
):
    """
    Test training-free spatial control at multiple scale values.

    Args:
        vla: Loaded MemoryVLA model
        save_dir: Where to save results
        scales: List of scale values to test (default: [0.1, 0.3, 0.5, 1.0])
        (other args same as run_control_test)
    """
    if scales is None:
        scales = [0.1, 0.3, 0.5, 1.0]

    os.makedirs(save_dir, exist_ok=True)

    # ====== Step 1: Render ManiSkill2 images ======
    print("=" * 60)
    print("STEP 1: Rendering ManiSkill2 images (224x224)")
    print("=" * 60)
    images_224 = _render_images_224(env_name, num_seeds, save_dir)

    if not images_224:
        print("ERROR: No images rendered!")
        return None

    # ====== Step 2: Load detection models ======
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

        # ====== Step 5: Spatial control at multiple scales ======
        actions_by_scale = {}
        diagnostics = None

        for scale in scales:
            print(f"\n--- Spatial control (scale={scale}) ---")
            t0 = time.time()
            actions_ctrl, norm_actions_ctrl = vla.predict_action_with_spatial_control(
                pil_224, task_instruction, target_grid,
                unnorm_key=unnorm_key, cfg_scale=cfg_scale,
                use_ddim=use_ddim, num_ddim_steps=num_ddim_steps,
                episode_first_frame='True',
                spatial_scale=scale,
            )
            t_ctrl = time.time() - t0
            actions_by_scale[scale] = actions_ctrl
            print(f"  Controlled action[0]: {actions_ctrl[0].round(4)}")
            print(f"  Time: {t_ctrl:.2f}s")

            diff = np.abs(actions_ctrl - actions_base)
            print(f"  Action diff (L1): {diff.mean():.4f} (max: {diff.max():.4f})")

        # ====== Step 6: Run diagnostics (once, at scale=0.5) ======
        print("\n--- Running diagnostics ---")
        from action_model.spatial_control import SpatialGatingControl
        # Need to re-extract tokens for diagnostics
        # Use a lightweight forward to get per_tokens and cog_tokens
        try:
            controller = SpatialGatingControl(vla.action_model.net, scale=0.5)
            # Get tokens from last predict call's internal state
            # For diagnostics, we need to manually extract tokens
            diagnostics = _run_diagnostics(
                vla, pil_224, task_instruction, target_grid, controller
            )
        except Exception as e:
            print(f"  Diagnostics failed: {e}")
            diagnostics = {}

        # ====== Step 7: Visualize ======
        visualize_spatial_control_result(
            pil_224, sam_mask, target_grid,
            actions_base, actions_by_scale, diagnostics,
            title=f"Spatial Control: {env_name} seed={seed}, prompt='{text_prompt}'",
            save_path=os.path.join(save_dir, f"spatial_seed{seed}.png"),
        )

        results.append({
            "seed": seed,
            "baseline_action": actions_base,
            "actions_by_scale": actions_by_scale,
            "diagnostics": diagnostics,
            "detection_score": grid_result["detection"]["score"],
            "time_baseline": t_base,
        })

    # ====== Summary ======
    if results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for r in results:
            print(f"\n  seed={r['seed']}: detect={r['detection_score']:.3f}")
            for scale, actions in sorted(r["actions_by_scale"].items()):
                diff = np.abs(actions - r["baseline_action"]).mean()
                print(f"    scale={scale}: action_diff={diff:.4f}")
            if r["diagnostics"]:
                d = r["diagnostics"]
                print(f"    focus: random={d.get('expected_random_focus', 0):.4f}, "
                      f"original={d.get('original_focus', 0):.4f}, "
                      f"spatial={d.get('spatial_focus', 0):.4f}")

    return results


def _run_diagnostics(vla, image, instruction, target_grid, controller):
    """Extract tokens and run spatial control diagnostics."""
    from transformers import LlamaTokenizerFast

    image_transform = vla.vlm.vision_backbone.image_transform
    tokenizer = vla.vlm.llm_backbone.tokenizer

    prompt_builder = vla.vlm.get_prompt_builder()
    prompt_builder.add_turn(
        role="human",
        message=f"What action should the robot take to {instruction.lower()}?",
    )
    prompt_text = prompt_builder.get_prompt()

    input_ids = tokenizer(
        prompt_text, truncation=True, return_tensors="pt"
    ).input_ids.to(vla.vlm.device)

    if isinstance(tokenizer, LlamaTokenizerFast):
        input_ids = torch.cat(
            (input_ids, torch.unsqueeze(
                torch.Tensor([29871, 2]).long(), dim=0
            ).to(vla.vlm.device)),
            dim=1,
        )

    model_dtype = next(vla.parameters()).dtype
    pixel_values = image_transform(image)
    if isinstance(pixel_values, torch.Tensor):
        pixel_values = pixel_values[None, ...].to(vla.vlm.device, dtype=model_dtype)
    elif isinstance(pixel_values, dict):
        pixel_values = {
            k: v[None, ...].to(vla.vlm.device, dtype=model_dtype)
            for k, v in pixel_values.items()
        }

    from prismatic.models.vlms.prismatic import PrismaticVLM

    autocast_dtype = torch.bfloat16 if model_dtype == torch.bfloat16 else torch.float32

    with torch.inference_mode():
        with torch.autocast(
            "cuda", dtype=autocast_dtype,
            enabled=(autocast_dtype == torch.bfloat16),
        ):
            output = super(PrismaticVLM, vla.vlm).generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        act_dtype = next(vla.action_model.net.parameters()).dtype
        cog_tokens = output.hidden_states[-1][-1][:, -1, :]
        cog_tokens = cog_tokens.unsqueeze(1).to(act_dtype)

        vision_feats = vla.vlm.vision_feats
        per_tokens = vla.per_compr(vision_feats)

    # Diagnostics need grad, so exit inference_mode
    return controller.diagnose(per_tokens, cog_tokens, target_grid)
