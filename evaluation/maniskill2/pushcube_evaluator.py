# ============================================
# PushCube 태스크 - ManiSkill 기본 카메라
# ============================================
import gymnasium as gym
import mani_skill.envs
import numpy as np
from PIL import Image
import torch
import imageio
import os

SAVE_DIR = "/content/eval_results_pushcube"
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_TRIALS = 8
MAX_STEPS = 100
UNNORM_KEY = "libero_object_no_noops"
TASK = "push the cube to the target"


def get_rgb_from_obs(obs):
    """
    ManiSkill 기본 카메라에서 RGB 이미지 추출.
    sensor_data 구조를 사용하며, 기본 카메라 우선순위:
    1. base_camera (third-person view)
    2. hand_camera (wrist camera)
    3. 첫 번째 사용 가능한 카메라
    """
    if "sensor_data" in obs:
        sensor_data = obs["sensor_data"]

        # 기본 카메라 우선순위
        preferred_cameras = ["base_camera", "hand_camera"]

        cam_name = None
        for pref_cam in preferred_cameras:
            if pref_cam in sensor_data:
                cam_name = pref_cam
                break

        # 우선순위 카메라가 없으면 첫 번째 카메라 사용
        if cam_name is None:
            cam_name = list(sensor_data.keys())[0]

        if "rgb" in sensor_data[cam_name]:
            rgb = sensor_data[cam_name]["rgb"]
            if isinstance(rgb, torch.Tensor):
                rgb = rgb.cpu().numpy()
            if rgb.ndim == 4:
                rgb = rgb[0]
            if rgb.max() <= 1.0:
                rgb = (rgb * 255).astype(np.uint8)
            return rgb, cam_name

    raise ValueError("No RGB data found in observation. Make sure obs_mode='rgbd'")


def run_episode(env, vla, task_instruction, max_steps):
    """
    단일 에피소드 실행.

    Returns:
        success: 성공 여부
        total_reward: 누적 보상
        frames: 비디오 프레임들
        num_steps: 실행된 스텝 수
        end_reason: 종료 이유 ('success', 'terminated', 'truncated', 'max_steps')
    """
    obs, info = env.reset()
    frames = []
    total_reward = 0
    end_reason = "max_steps"

    for step in range(max_steps):
        rgb, cam_name = get_rgb_from_obs(obs)
        pil_image = Image.fromarray(rgb)
        frames.append(rgb.copy())

        episode_first = 'True' if step == 0 else 'False'

        actions, _ = vla.predict_action(
            image=pil_image,
            instruction=task_instruction,
            unnorm_key=UNNORM_KEY,
            cfg_scale=1.5,
            use_ddim=True,
            num_ddim_steps=10,
            episode_first_frame=episode_first
        )

        action = actions[0]
        obs, reward, terminated, truncated, info = env.step(
            torch.tensor(action).unsqueeze(0)
        )

        if isinstance(reward, torch.Tensor):
            reward = reward.item()
        total_reward += reward

        # 성공 여부 확인
        success = info.get('success', False)
        if isinstance(success, torch.Tensor):
            success = success.item()

        # 종료 조건 확인 및 이유 기록
        if success:
            end_reason = "success"
            rgb, _ = get_rgb_from_obs(obs)
            frames.append(rgb.copy())
            break
        elif terminated:
            if isinstance(terminated, torch.Tensor):
                terminated = terminated.item()
            if terminated:
                end_reason = "terminated"
                rgb, _ = get_rgb_from_obs(obs)
                frames.append(rgb.copy())
                break
        elif truncated:
            if isinstance(truncated, torch.Tensor):
                truncated = truncated.item()
            if truncated:
                end_reason = "truncated"
                rgb, _ = get_rgb_from_obs(obs)
                frames.append(rgb.copy())
                break

    # 최종 성공 여부 다시 확인
    final_success = info.get('success', False)
    if isinstance(final_success, torch.Tensor):
        final_success = final_success.item()

    return final_success, total_reward, frames, step + 1, end_reason


def main(vla):
    """
    PushCube 평가 메인 함수.

    Args:
        vla: MemoryVLA 모델 인스턴스
    """
    # PushCube 환경 생성
    print("PushCube 환경 생성 중...")
    env = gym.make(
        "PushCube-v1",
        obs_mode="rgbd",
        control_mode="pd_ee_delta_pose",
        robot_uids="panda",
        num_envs=1,
        render_mode="rgb_array",
    )

    # 환경 확인
    obs, _ = env.reset()
    rgb, cam_name = get_rgb_from_obs(obs)
    print(f"✅ PushCube 환경 생성 완료!")
    print(f"   사용 카메라: {cam_name}")
    print(f"   이미지 크기: {rgb.shape}")

    # 시각화 (Colab 환경용)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 5))
        plt.imshow(rgb)
        plt.title(f"PushCube - Camera: {cam_name}")
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"시각화 스킵: {e}")

    # 평가 실행
    results = []
    print(f"\n{'='*60}")
    print(f"PushCube 평가: {NUM_TRIALS} trials, {MAX_STEPS} steps")
    print(f"Task: {TASK}")
    print(f"Camera: {cam_name} (ManiSkill default)")
    print(f"{'='*60}\n")

    for trial in range(NUM_TRIALS):
        success, total_reward, frames, steps, end_reason = run_episode(
            env, vla, TASK, MAX_STEPS
        )

        results.append({
            'trial': trial,
            'success': success,
            'reward': total_reward,
            'steps': steps,
            'end_reason': end_reason
        })

        # 상태 표시
        if success:
            status = "SUCCESS ✓"
        else:
            status = f"failure ({end_reason})"

        # 비디오 저장
        video_path = f"{SAVE_DIR}/trial_{trial:02d}_{'success' if success else 'failure'}_{end_reason}.mp4"
        imageio.mimsave(video_path, frames, fps=10)

        # 진행 상황 출력
        success_so_far = sum(r['success'] for r in results)
        print(f"Trial {trial+1:2d}/{NUM_TRIALS}: {status:25s} | "
              f"Reward: {total_reward:7.3f} | Steps: {steps:3d} | "
              f"Success: {success_so_far}/{trial+1}")

    env.close()

    # 결과 요약
    print(f"\n{'='*60}")
    print("결과 요약")
    print(f"{'='*60}")

    total_success = sum(r['success'] for r in results)
    print(f"🎯 PushCube Success Rate: {total_success}/{NUM_TRIALS} ({100*total_success/NUM_TRIALS:.1f}%)")

    # 종료 이유별 통계
    end_reasons = {}
    for r in results:
        reason = r['end_reason']
        end_reasons[reason] = end_reasons.get(reason, 0) + 1

    print(f"\n종료 이유별 통계:")
    for reason, count in sorted(end_reasons.items()):
        print(f"  - {reason}: {count}/{NUM_TRIALS} ({100*count/NUM_TRIALS:.1f}%)")

    # 평균 통계
    avg_reward = np.mean([r['reward'] for r in results])
    avg_steps = np.mean([r['steps'] for r in results])
    print(f"\n평균 보상: {avg_reward:.3f}")
    print(f"평균 스텝: {avg_steps:.1f}")
    print(f"{'='*60}")

    return results


# Colab에서 직접 실행할 때 사용
if __name__ == "__main__":
    print("이 스크립트는 Colab에서 vla 모델과 함께 사용하세요.")
    print("예시: results = main(vla)")
