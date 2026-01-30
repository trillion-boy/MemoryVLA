#!/usr/bin/env python3
"""
🔧 MetaWorld 평가 (개선판)
- 통계 데이터 문제 우회: Raw 출력값 사용
- 액션 스케일 자동 조정
- Verbose 디버깅 모드
"""

import os
import sys
import torch
import numpy as np
import metaworld
import random
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# =============================================================================
# 설정
# =============================================================================
PT_FILE = "/content/checkpoints/models--shihao1895--memvla-libero-spatial/snapshots/4d6572ce289736e459e38a48f8671b557a6fd078/checkpoints/memvla-libero-spatial.pt"
TASK_NAME = 'pick-place-v3'
INSTRUCTION = "pick up the puck and place it on the goal"
NUM_EPISODES = 5
MAX_STEPS = 200
RENDER_SIZE = 224

# 🔧 액션 스케일 설정 (실험적)
# MetaWorld의 액션 스페이스: [-1, 1] (각 축당 ~0.05m)
# LIBERO와 다를 수 있으므로 조정 필요
ACTION_SCALE = 1.0  # 필요시 2.0, 5.0으로 조정

# 디버깅 모드
DEBUG = True  # True로 설정하면 액션값 출력

# =============================================================================
# 1. 모델 로드 (강제 방식 - 하지만 개선)
# =============================================================================
print("🏗️ 모델 로딩 중...")

try:
    # OpenVLA 베이스 모델 로드
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b",
        trust_remote_code=True
    )

    # 가중치 로드
    print("💉 가중치 주입 중...")
    state_dict = torch.load(PT_FILE, map_location="cpu")

    # 키 정리
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]

    # module. 접두사 제거
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v

    # 로드 (일부 불일치 무시)
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

    if DEBUG:
        print(f"⚠️ Missing keys: {len(missing)}")
        print(f"⚠️ Unexpected keys: {len(unexpected)}")

    model = model.to("cuda")
    model.eval()
    print("✅ 모델 로드 완료")

except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    sys.exit(1)

# =============================================================================
# 2. 통계 데이터 문제 해결
# =============================================================================
print("\n🚑 통계 데이터 처리 중...")

# 방법 1: 기존 통계 복사 (불완전하지만 작동은 함)
if hasattr(model, 'norm_stats'):
    if "bridge_orig" in model.norm_stats:
        model.norm_stats["libero_spatial_no_noops"] = model.norm_stats["bridge_orig"]
        print("✅ bridge_orig 통계를 복사했습니다 (정확도 낮을 수 있음)")
    else:
        # 방법 2: 정규화 비활성화 (Mean=0, Std=1)
        print("⚠️ 정규화를 우회합니다 (Raw 출력값 사용)")
        model.norm_stats["libero_spatial_no_noops"] = {
            "action": {
                "mean": [0.0] * 7,
                "std": [1.0] * 7,
                "min": [-1.0] * 7,
                "max": [1.0] * 7
            }
        }
else:
    print("⚠️ norm_stats 속성이 없습니다. 모델 구조를 확인하세요")

# =============================================================================
# 3. MetaWorld 환경 설정
# =============================================================================
os.environ["MUJOCO_GL"] = "osmesa"
print(f"\n🛠️ MetaWorld 환경: {TASK_NAME}")

ml1 = metaworld.ML1(TASK_NAME)
env_cls = ml1.train_classes[TASK_NAME]
env = env_cls(render_mode='rgb_array', camera_name='corner2')

# 액션 스페이스 확인
print(f"📏 Action Space: {env.action_space}")
print(f"   - Low:  {env.action_space.low}")
print(f"   - High: {env.action_space.high}")

# =============================================================================
# 4. 평가 루프
# =============================================================================
success_count = 0
action_magnitudes = []  # 액션 크기 모니터링

print(f"\n🚀 평가 시작 ({NUM_EPISODES} episodes)")
print("="*60)

for episode in range(NUM_EPISODES):
    task = random.choice(ml1.train_tasks)
    env.set_task(task)
    obs = env.reset()

    print(f"\n🎬 Episode {episode+1}/{NUM_EPISODES}")
    episode_actions = []

    for step in range(MAX_STEPS):
        # 1. 이미지 캡처
        image_array = env.render()
        if image_array is None:
            print("❌ 렌더링 실패")
            break

        image = Image.fromarray(image_array).resize((RENDER_SIZE, RENDER_SIZE))

        # 2. 모델 추론
        prompt = f"In: <image> What action should the robot take to {INSTRUCTION}?\nOut:"
        inputs = processor(prompt, image).to("cuda", dtype=torch.bfloat16)

        try:
            with torch.no_grad():
                # 🔧 predict_action 대신 generate 사용 (더 안정적)
                output = model.predict_action(
                    **inputs,
                    unnorm_key="libero_spatial_no_noops",
                    do_sample=False
                )

                if isinstance(output, torch.Tensor):
                    action = output.cpu().numpy()
                else:
                    action = np.array(output)

        except Exception as e:
            print(f"⚠️ 추론 에러 (Step {step}): {e}")
            action = np.zeros(7)

        # 3. 액션 처리
        # 7-DoF → 4-DoF 변환
        # [x, y, z, roll, pitch, yaw, gripper] → [x, y, z, gripper]
        xyz = action[:3] * ACTION_SCALE
        gripper = action[-1]

        mw_action = np.concatenate([xyz, [gripper]])

        # Clipping (MetaWorld 범위 준수)
        mw_action = np.clip(mw_action, env.action_space.low, env.action_space.high)

        # 디버깅 출력
        if DEBUG and step < 5:
            print(f"   Step {step}:")
            print(f"     Raw Action: {action}")
            print(f"     XYZ: {xyz}")
            print(f"     Gripper: {gripper}")
            print(f"     Final: {mw_action}")

        episode_actions.append(np.linalg.norm(xyz))

        # 4. 환경 적용
        obs, reward, done, truncated, info = env.step(mw_action)

        # 5. 성공 체크
        if info.get('success', 0.0) > 0:
            print(f"   🎉 성공! (Step {step})")
            success_count += 1
            break

    # 에피소드 통계
    if episode_actions:
        avg_magnitude = np.mean(episode_actions)
        action_magnitudes.append(avg_magnitude)
        if DEBUG:
            print(f"   📊 평균 액션 크기: {avg_magnitude:.4f}")

    if info.get('success', 0.0) == 0:
        print(f"   ❌ 실패")

env.close()

# =============================================================================
# 5. 결과 분석
# =============================================================================
print("\n" + "="*60)
print("🏆 최종 결과")
print("="*60)
print(f"성공률: {success_count}/{NUM_EPISODES} ({success_count/NUM_EPISODES*100:.1f}%)")

if action_magnitudes:
    print(f"\n📊 액션 분석:")
    print(f"   평균 크기: {np.mean(action_magnitudes):.4f}")
    print(f"   최소/최대: {np.min(action_magnitudes):.4f} / {np.max(action_magnitudes):.4f}")

    # 진단
    avg_mag = np.mean(action_magnitudes)
    if avg_mag < 0.001:
        print("\n⚠️ 경고: 액션이 너무 작습니다!")
        print("   💡 해결책: ACTION_SCALE을 10.0 이상으로 설정하세요")
    elif avg_mag > 0.5:
        print("\n⚠️ 경고: 액션이 너무 큽니다!")
        print("   💡 해결책: ACTION_SCALE을 0.1로 줄이세요")
    else:
        print("\n✅ 액션 크기는 적절합니다")

print("="*60)
