#!/usr/bin/env python3
"""
✅ LIBERO 환경에서 MemoryVLA 평가 (권장)
- 모델이 실제로 학습한 환경이므로 가장 정확
- 통계 데이터 불일치 문제 최소화
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path

# MemoryVLA 임포트
try:
    from vla import load_vla
    print("✅ MemoryVLA 모듈 임포트 성공")
except ImportError as e:
    print(f"❌ MemoryVLA 임포트 실패: {e}")
    print("💡 pip install -e . 실행했는지 확인하세요")
    sys.exit(1)

# LIBERO 임포트
try:
    import libero
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    print("✅ LIBERO 임포트 성공")
except ImportError:
    print("❌ LIBERO가 설치되지 않았습니다")
    print("💡 설치 방법:")
    print("   git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git")
    print("   cd LIBERO && pip install -e .")
    sys.exit(1)

# =============================================================================
# 설정
# =============================================================================
CHECKPOINT_FILE = "/content/checkpoint_path_spatial.txt"
TASK_SUITE = "libero_spatial"  # 모델이 학습한 데이터셋
UNNORM_KEY = "libero_spatial_no_noops"
NUM_EPISODES = 10
MAX_STEPS = 300
RENDER_SIZE = 224

# =============================================================================
# 1. 체크포인트 로드
# =============================================================================
if not os.path.exists(CHECKPOINT_FILE):
    print(f"❌ {CHECKPOINT_FILE} 파일이 없습니다")
    sys.exit(1)

with open(CHECKPOINT_FILE, "r") as f:
    model_path = f.read().strip()

print(f"📂 모델 경로: {model_path}")

# =============================================================================
# 2. 모델 로드 (정상적인 방법)
# =============================================================================
print("🔄 모델 로딩 중...")
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_vla(model_path, device=device)
    model.eval()
    print(f"✅ 모델 로드 완료 (Device: {device})")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    print("\n💡 대안: 강제 로딩 방식을 사용하시겠습니까?")
    sys.exit(1)

# =============================================================================
# 3. LIBERO 환경 설정
# =============================================================================
print(f"\n🛠️ LIBERO 환경 설정: {TASK_SUITE}")
try:
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[TASK_SUITE]()

    # 첫 번째 태스크 가져오기
    task = task_suite.get_task(0)
    task_name = task.name
    task_description = task.language

    print(f"📋 Task: {task_name}")
    print(f"💬 Instruction: {task_description}")

    # 환경 생성
    env_args = {
        "bddl_file_name": task.problem_folder,
        "camera_heights": RENDER_SIZE,
        "camera_widths": RENDER_SIZE,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)

except Exception as e:
    print(f"❌ LIBERO 환경 설정 실패: {e}")
    sys.exit(1)

# =============================================================================
# 4. 평가 루프
# =============================================================================
success_count = 0
episode_lengths = []

print(f"\n🚀 평가 시작 ({NUM_EPISODES} episodes)")
print("="*60)

for episode in range(NUM_EPISODES):
    obs = env.reset()
    print(f"\n🎬 Episode {episode+1}/{NUM_EPISODES}")

    for step in range(MAX_STEPS):
        # 1. 이미지 추출
        # LIBERO는 보통 'agentview_image' 키를 사용
        if isinstance(obs, dict):
            if 'agentview_image' in obs:
                image_array = obs['agentview_image']
            elif 'image' in obs:
                image_array = obs['image']
            else:
                print(f"⚠️ 알 수 없는 observation 구조: {obs.keys()}")
                break
        else:
            image_array = obs

        # PIL Image로 변환
        if image_array.dtype == np.uint8:
            image = Image.fromarray(image_array)
        else:
            image = Image.fromarray((image_array * 255).astype(np.uint8))

        # 2. 모델 추론
        try:
            action = model.predict_action(
                image,
                task_description,
                unnorm_key=UNNORM_KEY,
                do_sample=False
            )
        except Exception as e:
            print(f"⚠️ 추론 에러 (Step {step}): {e}")
            break

        # 3. 환경 적용
        # LIBERO는 보통 7-DoF를 그대로 받음
        obs, reward, done, info = env.step(action)

        # 디버깅 출력 (처음 몇 스텝만)
        if step < 3:
            print(f"   Step {step}: Action = {action[:3]} ... (XYZ)")

        # 4. 성공 체크
        if done:
            if info.get('success', False):
                print(f"   🎉 성공! (Step {step})")
                success_count += 1
                episode_lengths.append(step)
            else:
                print(f"   ❌ 실패 (Step {step})")
            break
    else:
        print(f"   ⏱️ Timeout")

env.close()

# =============================================================================
# 5. 결과 출력
# =============================================================================
print("\n" + "="*60)
print("🏆 최종 결과")
print("="*60)
print(f"성공률: {success_count}/{NUM_EPISODES} ({success_count/NUM_EPISODES*100:.1f}%)")
if episode_lengths:
    print(f"평균 스텝 수: {np.mean(episode_lengths):.1f}")
print("="*60)
