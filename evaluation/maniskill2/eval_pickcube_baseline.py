"""
MemoryVLA baseline + Depth-Guided Action Correction on ManiSkill2 PickCube.

Depth Action Correction: DA V2 depth → detect cube position → compute
correction vector → add to MemoryVLA action. Training-free, plug-and-play.

Usage (Colab):
    from evaluation.maniskill2.eval_pickcube_baseline import run_baseline
    # Pure baseline (no depth)
    results = run_baseline(vla, save_dir="/content/eval_baseline")
    # With depth action correction
    results = run_baseline(vla, depth_model=depth_model,
        depth_correction=True, save_dir="/content/eval_depth_correction")
"""

import cv2
import gymnasium as gym
try:
    import mani_skill.envs
except ModuleNotFoundError:
    import mani_skill2.envs
import numpy as np
from PIL import Image
import torch
import imageio
import os
from typing import Dict, Any, Optional, Tuple
from transforms3d.euler import euler2axangle

from evaluation.simpler_env.adaptive_ensemble import AdaptiveEnsembler


TASK_INSTRUCTION = (
    "Pick up the small cube from the table. Move the gripper directly above "
    "the cube, lower it down to grasp the cube, close the gripper firmly, "
    "and lift the cube upward off the table surface."
)


def get_rgb_from_obs(obs):
    """Extract RGB from ManiSkill2 observation."""
    if "sensor_data" in obs:
        for cam in ["base_camera", "hand_camera"]:
            if cam in obs["sensor_data"] and "rgb" in obs["sensor_data"][cam]:
                rgb = obs["sensor_data"][cam]["rgb"]
                if isinstance(rgb, torch.Tensor):
                    rgb = rgb.cpu().numpy()
                if rgb.ndim == 4:
                    rgb = rgb[0]
                if rgb.max() <= 1.0:
                    rgb = (rgb * 255).astype(np.uint8)
                return rgb
    raise ValueError("No RGB found in obs")


def convert_action_to_maniskill2(raw_action: np.ndarray, action_scale: float = 1.0) -> np.ndarray:
    """
    Convert MemoryVLA raw action (LIBERO format) to ManiSkill2 format.

    LIBERO:     [x, y, z, roll, pitch, yaw, gripper(0~1)]
    ManiSkill2: [dx, dy, dz, ax, ay, az, gripper(-1~+1)]
    """
    delta_pos = raw_action[:3] * action_scale

    roll, pitch, yaw = raw_action[3], raw_action[4], raw_action[5]
    axis, angle = euler2axangle(roll, pitch, yaw)
    delta_rot_axangle = axis * angle * action_scale

    gripper = raw_action[6]
    gripper_normalized = 2.0 * (gripper > 0.5) - 1.0

    return np.concatenate([delta_pos, delta_rot_axangle, [gripper_normalized]]).astype(np.float32)


def detect_cube_from_depth(
    depth_map: np.ndarray,
    debug: bool = False,
) -> Optional[Tuple[float, float]]:
    """
    Detect cube pixel position from depth map using 2-phase CCA.

    Returns (cx, cy) centroid in pixel coordinates, or None if not found.
    Uses progressive thresholds with background masking and center-proximity scoring.
    """
    depth = depth_map.astype(np.float32)
    h, w = depth.shape[:2]

    # === Mask background (sky) ===
    # DA V2 relative depth: far = low values (sky), close = high values (table, arm)
    # Mask out the far background to avoid sky-table boundary artifacts
    depth_p20 = np.percentile(depth, 20)
    bg_mask = (depth < depth_p20).astype(np.uint8)
    bg_mask = cv2.dilate(bg_mask, np.ones((11, 11), np.uint8))

    # Smooth background estimate
    smooth = cv2.GaussianBlur(depth, (31, 31), 0)
    abs_deviation = np.abs(smooth - depth)

    # Zero out background region in deviation
    abs_deviation[bg_mask > 0] = 0

    if debug:
        bg_pct = 100 * bg_mask.sum() / (h * w)
        print(f"    [depth] shape={depth.shape}, dev mean={abs_deviation.mean():.4f}, "
              f"std={abs_deviation.std():.4f}, max={abs_deviation.max():.4f}, bg_masked={bg_pct:.0f}%")

    # Phase 1: Find and remove robot arm (largest anomaly)
    phase1_thr = abs_deviation.mean() + abs_deviation.std() * 1.0
    phase1_mask = (abs_deviation > phase1_thr).astype(np.uint8)

    num_labels_p1, labels_p1, stats_p1, _ = cv2.connectedComponentsWithStats(
        phase1_mask, connectivity=8
    )
    arm_mask = np.zeros_like(phase1_mask)
    if num_labels_p1 > 1:
        areas_p1 = stats_p1[1:, cv2.CC_STAT_AREA]
        arm_label = int(np.argmax(areas_p1)) + 1
        arm_mask = (labels_p1 == arm_label).astype(np.uint8)
        arm_mask = cv2.dilate(arm_mask, np.ones((15, 15), np.uint8))
        if debug:
            print(f"    [phase1] arm area={areas_p1[arm_label-1]}, total components={num_labels_p1-1}")

    # Combine exclusion mask: background + arm
    exclude_mask = np.maximum(bg_mask, arm_mask)

    # Phase 2: Detect small objects on table with progressive thresholds
    table_deviation = abs_deviation.copy()
    table_deviation[exclude_mask > 0] = 0

    table_valid = abs_deviation[exclude_mask == 0]
    if table_valid.size == 0 or table_valid.std() < 1e-6:
        if debug:
            print(f"    [phase2] FAIL: table_valid empty or zero std")
        return None

    center_x, center_y = w / 2.0, h / 2.0
    max_dist = np.sqrt(center_x**2 + center_y**2)

    # Try progressively lower thresholds: 2.0, 1.5, 1.0, 0.5
    for multiplier in [2.0, 1.5, 1.0, 0.5]:
        phase2_thr = table_valid.mean() + table_valid.std() * multiplier

        target_candidates = (
            (table_deviation > phase2_thr) & (exclude_mask == 0)
        ).astype(np.uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            target_candidates, connectivity=8
        )

        # Score components: prefer small, center-close ones
        best_component = None
        best_score = -1
        all_info = []
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < 1 or area > 300:
                continue
            cx, cy = centroids[label_id]
            dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            # Score: closer to center = better (0~1), penalize very large areas
            proximity = 1.0 - (dist / max_dist)
            score = proximity
            all_info.append((label_id, area, cx, cy, dist, score))
            if score > best_score:
                best_score = score
                best_component = label_id

        if debug:
            areas = [info[1] for info in all_info]
            print(f"    [phase2] thr_mult={multiplier:.1f}, thr={phase2_thr:.4f}, "
                  f"candidates={len(all_info)}, areas={sorted(areas, reverse=True)[:5]}")
            for info in sorted(all_info, key=lambda x: -x[5])[:3]:
                lid, a, cx, cy, d, s = info
                print(f"      comp {lid}: area={a}, pos=({cx:.1f},{cy:.1f}), dist={d:.1f}, score={s:.2f}")

        if best_component is not None:
            cx = centroids[best_component][0]
            cy = centroids[best_component][1]
            if debug:
                print(f"    [phase2] SELECTED: centroid=({cx:.1f}, {cy:.1f}), score={best_score:.2f}")
            return (cx, cy)

    if debug:
        print(f"    [phase2] FAIL: no component found at any threshold")
    return None


def compute_depth_correction(
    cube_pos: Tuple[float, float],
    image_size: int,
    correction_scale: float = 0.005,
) -> np.ndarray:
    """
    Compute action correction vector to move gripper toward detected cube.

    Maps pixel offset (cube vs image center) to action delta.
    ManiSkill2 base_camera looks from behind/above the robot:
      - image horizontal (left-right) → action[1] (world y, left-right)
      - image vertical (top-bottom)   → action[0] (world x, forward-back)

    Args:
        cube_pos: (cx, cy) pixel coordinates of cube
        image_size: image width/height (128)
        correction_scale: magnitude of correction per unit offset

    Returns:
        correction: [dx, dy, dz, 0, 0, 0, 0] action correction
    """
    cx, cy = cube_pos
    center = image_size / 2.0

    # Normalized offset: [-1, 1]
    offset_x = (cx - center) / center  # positive = cube is to the right
    offset_y = (cy - center) / center  # positive = cube is below center

    # Map image offsets to robot action space
    # These signs may need tuning based on camera orientation
    correction = np.zeros(7, dtype=np.float32)
    correction[0] = correction_scale * offset_y   # image down → move forward
    correction[1] = correction_scale * offset_x   # image right → move right

    return correction


def run_baseline(
    vla,
    depth_model=None,
    num_trials: int = 4,
    max_steps: int = 100,
    task_instruction: str = TASK_INSTRUCTION,
    unnorm_key: str = "libero_object_no_noops",
    save_dir: str = "/content/eval_pure_baseline",
    cfg_scale: float = 1.5,
    save_videos: bool = True,
    sensor_resolution: int = 128,
    # Action ensemble
    action_ensemble: bool = True,
    action_ensemble_horizon: int = 7,
    adaptive_ensemble_alpha: float = 0.1,
    # Depth action correction
    depth_correction: bool = False,
    correction_scale: float = 0.005,
) -> Dict[str, Any]:
    """
    MemoryVLA baseline with optional depth-guided action correction.

    Args:
        depth_model: DA V2 model (required if depth_correction=True)
        depth_correction: Enable depth-guided action correction
        correction_scale: Strength of correction (tune this!)
    """
    os.makedirs(save_dir, exist_ok=True)

    env = gym.make(
        "PickCube-v1",
        obs_mode="rgbd",
        control_mode="pd_ee_delta_pose",
        robot_uids="panda",
        num_envs=1,
        render_mode="rgb_array",
        max_episode_steps=max_steps,
        sensor_configs=dict(
            base_camera=dict(width=sensor_resolution, height=sensor_resolution),
        ),
    )

    # Action ensemble
    ensembler = None
    if action_ensemble:
        ensembler = AdaptiveEnsembler(
            pred_action_horizon=action_ensemble_horizon,
            adaptive_ensemble_alpha=adaptive_ensemble_alpha,
        )

    ensemble_str = f"ON (horizon={action_ensemble_horizon}, alpha={adaptive_ensemble_alpha})" if action_ensemble else "OFF"
    depth_str = f"ON (scale={correction_scale})" if depth_correction else "OFF"
    mode_name = "DEPTH CORRECTION" if depth_correction else "PURE BASELINE"

    print(f"\n{'='*60}")
    print(f"{mode_name}: MemoryVLA + action conversion")
    print(f"{'='*60}")
    print(f"Task: {task_instruction}")
    print(f"Trials: {num_trials}, Max steps: {max_steps}")
    print(f"unnorm_key: {unnorm_key}")
    print(f"Action conversion: euler→axis-angle + gripper [-1,+1]")
    print(f"Action ensemble: {ensemble_str}")
    print(f"Depth correction: {depth_str}")
    print(f"{'='*60}\n")

    # Track cube detection for logging
    cube_detected_count = 0
    cube_total_steps = 0

    results = []

    for trial in range(num_trials):
        obs, _ = env.reset()
        frames = []
        total_reward = 0.0

        if ensembler is not None:
            ensembler.reset()

        # Cache cube position per trial (cube is stationary)
        cached_cube_pos = None

        for step in range(max_steps):
            rgb = get_rgb_from_obs(obs)
            pil_image = Image.fromarray(rgb)
            frames.append(rgb.copy())

            episode_first = "True" if step == 0 else "False"

            actions, _ = vla.predict_action(
                image=pil_image,
                instruction=task_instruction,
                unnorm_key=unnorm_key,
                cfg_scale=cfg_scale,
                use_ddim=True,
                num_ddim_steps=10,
                episode_first_frame=episode_first,
            )

            raw_action = actions[0]

            if ensembler is not None:
                raw_action = ensembler.ensemble_action(raw_action)

            action = convert_action_to_maniskill2(raw_action)

            # Depth-guided action correction
            if depth_correction and depth_model is not None:
                cube_total_steps += 1

                # Detect cube (use cache if available)
                if cached_cube_pos is None:
                    rgb_np = np.array(pil_image)
                    if hasattr(depth_model, 'infer_image'):
                        depth_map = depth_model.infer_image(rgb_np)
                    else:
                        result = depth_model(pil_image)
                        depth_map = np.array(result["depth"]).astype(np.float32)

                    # Debug on first attempt of each trial
                    is_first_attempt = (step == 0)
                    cube_pos = detect_cube_from_depth(depth_map, debug=is_first_attempt)
                    if cube_pos is not None:
                        cached_cube_pos = cube_pos
                        cube_detected_count += 1
                        print(f"  Trial {trial+1} step {step}: Cube detected at pixel ({cube_pos[0]:.1f}, {cube_pos[1]:.1f})")

                    # Save annotated debug image on first step
                    if is_first_attempt:
                        depth_debug_path = os.path.join(save_dir, f"trial_{trial:02d}_depth.npy")
                        np.save(depth_debug_path, depth_map)

                        # Create side-by-side: RGB | Depth with detection overlay
                        d_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
                        depth_rgb = np.stack([d_norm * 255] * 3, axis=-1).astype(np.uint8)

                        # Draw detected point on both
                        rgb_annotated = rgb_np.copy()
                        if cached_cube_pos is not None:
                            cx_i, cy_i = int(cached_cube_pos[0]), int(cached_cube_pos[1])
                            # Red cross on RGB
                            cv2.drawMarker(rgb_annotated, (cx_i, cy_i), (255, 0, 0),
                                           cv2.MARKER_CROSS, 15, 2)
                            # Red cross on depth
                            cv2.drawMarker(depth_rgb, (cx_i, cy_i), (255, 0, 0),
                                           cv2.MARKER_CROSS, 15, 2)
                            # Blue cross at center
                            c = sensor_resolution // 2
                            cv2.drawMarker(rgb_annotated, (c, c), (0, 0, 255),
                                           cv2.MARKER_CROSS, 10, 1)
                            cv2.drawMarker(depth_rgb, (c, c), (0, 0, 255),
                                           cv2.MARKER_CROSS, 10, 1)

                        # Concatenate side by side
                        combined = np.concatenate([rgb_annotated, depth_rgb], axis=1)
                        debug_path = os.path.join(save_dir, f"trial_{trial:02d}_debug.png")
                        Image.fromarray(combined).save(debug_path)
                        print(f"  Trial {trial+1}: Debug image saved → {debug_path}")

                # Apply correction if cube was found
                if cached_cube_pos is not None:
                    correction = compute_depth_correction(
                        cached_cube_pos, sensor_resolution, correction_scale
                    )
                    action[:3] += correction[:3]

                    if step == 0:
                        print(f"  Trial {trial+1}: Correction dx={correction[0]:.4f}, dy={correction[1]:.4f}")
                elif step == 0:
                    print(f"  Trial {trial+1}: Cube NOT detected, no correction")

            obs, reward, terminated, truncated, info = env.step(
                torch.tensor(action).unsqueeze(0)
            )

            if isinstance(reward, torch.Tensor):
                reward = reward.item()
            if isinstance(terminated, torch.Tensor):
                terminated = terminated.item()
            if isinstance(truncated, torch.Tensor):
                truncated = truncated.item()

            total_reward += reward

            if terminated or truncated:
                rgb = get_rgb_from_obs(obs)
                frames.append(rgb.copy())
                break

        success = info.get("success", False)
        if isinstance(success, torch.Tensor):
            success = success.item()

        result = {
            "trial": trial,
            "success": success,
            "total_reward": total_reward,
            "steps": step + 1,
        }
        results.append(result)

        if save_videos and frames:
            status = "success" if success else "failure"
            video_path = os.path.join(save_dir, f"trial_{trial:02d}_{status}.mp4")
            imageio.mimsave(video_path, frames, fps=10)

        sc = sum(r["success"] for r in results)
        status = "SUCCESS" if success else "failure"
        print(
            f"Trial {trial+1:2d}/{num_trials}: {status:10s} | "
            f"Reward: {total_reward:7.3f} | Steps: {step+1:3d} | "
            f"Success: {sc}/{trial+1}"
        )

    env.close()

    total_success = sum(r["success"] for r in results)
    avg_reward = np.mean([r["total_reward"] for r in results])

    if depth_correction and cube_total_steps > 0:
        det_rate = 100 * cube_detected_count / num_trials
        print(f"\nCube detection: {cube_detected_count}/{num_trials} trials ({det_rate:.0f}%)")

    print(f"\n{'='*60}")
    print(f"{mode_name} Summary")
    print(f"{'='*60}")
    print(f"Success Rate: {total_success}/{num_trials} ({100*total_success/num_trials:.1f}%)")
    print(f"Avg Reward:   {avg_reward:.3f}")
    print(f"{'='*60}")

    return {
        "pipeline": mode_name,
        "action_ensemble": action_ensemble,
        "depth_correction": depth_correction,
        "correction_scale": correction_scale,
        "success_rate": 100 * total_success / num_trials,
        "avg_reward": avg_reward,
        "results": results,
    }
