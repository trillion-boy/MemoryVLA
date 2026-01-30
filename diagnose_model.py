#!/usr/bin/env python3
"""
🔍 모델 진단 스크립트
- 모델 출력값 확인
- 통계 데이터 검증
- 액션 스케일 분석
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
import json

# =============================================================================
# 설정
# =============================================================================
PT_FILE = "/content/checkpoints/models--shihao1895--memvla-libero-spatial/snapshots/4d6572ce289736e459e38a48f8671b557a6fd078/checkpoints/memvla-libero-spatial.pt"
STATS_FILE = "/content/checkpoints/models--shihao1895--memvla-libero-spatial/snapshots/4d6572ce289736e459e38a48f8671b557a6fd078/dataset_statistics.json"

print("="*80)
print("🔍 MemoryVLA 모델 진단")
print("="*80)

# =============================================================================
# 1. 통계 파일 검사
# =============================================================================
print("\n📊 1. 통계 파일 검사")
print("-"*80)

if os.path.exists(STATS_FILE):
    print(f"✅ 파일 존재: {STATS_FILE}")
    with open(STATS_FILE, 'r') as f:
        stats = json.load(f)

    print(f"\n사용 가능한 데이터셋 키:")
    for key in stats.keys():
        print(f"  - {key}")

    # libero_spatial_no_noops 확인
    if "libero_spatial_no_noops" in stats:
        print(f"\n✅ 'libero_spatial_no_noops' 키 존재")
        action_stats = stats["libero_spatial_no_noops"].get("action", {})
        print(f"\n   액션 통계:")
        print(f"     Mean: {action_stats.get('mean', 'N/A')}")
        print(f"     Std:  {action_stats.get('std', 'N/A')}")
        print(f"     Min:  {action_stats.get('min', 'N/A')}")
        print(f"     Max:  {action_stats.get('max', 'N/A')}")
    else:
        print(f"\n❌ 'libero_spatial_no_noops' 키 없음")
        print(f"   💡 이것이 문제의 원인입니다!")

        # bridge_orig 확인
        if "bridge_orig" in stats:
            print(f"\n   'bridge_orig' 키는 존재:")
            bridge_stats = stats["bridge_orig"].get("action", {})
            print(f"     Mean: {bridge_stats.get('mean', 'N/A')}")
            print(f"     Std:  {bridge_stats.get('std', 'N/A')}")
else:
    print(f"❌ 파일 없음: {STATS_FILE}")
    print(f"   💡 이것이 에러의 원인입니다!")

# =============================================================================
# 2. 모델 로드
# =============================================================================
print("\n\n🤖 2. 모델 로드")
print("-"*80)

try:
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
    state_dict = torch.load(PT_FILE, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]

    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

    print(f"✅ 모델 로드 완료")
    print(f"   Missing keys: {len(missing)}")
    print(f"   Unexpected keys: {len(unexpected)}")

    # GPU로 이동
    model = model.to("cuda")
    model.eval()

    # 통계 해킹 (테스트용)
    if hasattr(model, 'norm_stats'):
        print(f"\n✅ 모델이 norm_stats 속성을 가지고 있습니다")
        print(f"   기존 키: {list(model.norm_stats.keys())}")

        # 강제 주입
        if "libero_spatial_no_noops" not in model.norm_stats:
            print(f"\n⚠️ libero_spatial_no_noops 키가 없어서 생성합니다")
            if "bridge_orig" in model.norm_stats:
                model.norm_stats["libero_spatial_no_noops"] = model.norm_stats["bridge_orig"]
                print(f"   → bridge_orig를 복사했습니다")
            else:
                model.norm_stats["libero_spatial_no_noops"] = {
                    "action": {"mean": [0.0]*7, "std": [1.0]*7}
                }
                print(f"   → 더미 데이터를 생성했습니다")
    else:
        print(f"⚠️ 모델이 norm_stats 속성이 없습니다")

except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    sys.exit(1)

# =============================================================================
# 3. 테스트 추론
# =============================================================================
print("\n\n🧪 3. 테스트 추론 (더미 이미지)")
print("-"*80)

# 더미 이미지 생성 (224x224 RGB)
dummy_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
test_instruction = "pick up the object"

print(f"입력: {test_instruction}")

# 3가지 방법으로 테스트
test_cases = [
    ("정규화 없음", None),
    ("libero_spatial_no_noops", "libero_spatial_no_noops"),
    ("bridge_orig (비교용)", "bridge_orig")
]

results = []

for test_name, unnorm_key in test_cases:
    print(f"\n[{test_name}]")

    try:
        prompt = f"In: <image> What action should the robot take to {test_instruction}?\nOut:"
        inputs = processor(prompt, dummy_image).to("cuda", dtype=torch.bfloat16)

        if unnorm_key is None:
            # 정규화 없이 raw 출력
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False)
                # 토큰을 action으로 디코드 (간단히 마지막 7개 값 사용)
                # 실제로는 모델의 predict_action 내부 로직을 따라야 함
                print(f"  ⚠️ Raw output (token decoding 필요)")
        else:
            action = model.predict_action(
                **inputs,
                unnorm_key=unnorm_key,
                do_sample=False
            )

            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            else:
                action = np.array(action)

            print(f"  출력 Shape: {action.shape}")
            print(f"  XYZ: {action[:3]}")
            print(f"  Rotation: {action[3:6]}")
            print(f"  Gripper: {action[6]}")
            print(f"  L2 Norm: {np.linalg.norm(action[:3]):.6f}")

            results.append((test_name, action))

    except Exception as e:
        print(f"  ❌ 에러: {e}")

# =============================================================================
# 4. 비교 분석
# =============================================================================
if len(results) >= 2:
    print("\n\n📈 4. 액션 비교")
    print("-"*80)

    for i, (name, action) in enumerate(results):
        print(f"\n[{name}]")
        print(f"  XYZ Magnitude: {np.linalg.norm(action[:3]):.6f}")
        print(f"  Max Component: {np.max(np.abs(action)):.6f}")

        # 진단
        magnitude = np.linalg.norm(action[:3])
        if magnitude < 1e-4:
            print(f"  ⚠️ 움직임 거의 없음 (통계 불일치 의심)")
        elif magnitude < 0.01:
            print(f"  ⚠️ 움직임 매우 작음 (스케일 조정 필요)")
        elif magnitude < 0.1:
            print(f"  ✅ 적절한 크기 (MetaWorld 기준)")
        else:
            print(f"  ⚠️ 움직임 매우 큼 (Clipping 발생 가능)")

# =============================================================================
# 5. 권장사항
# =============================================================================
print("\n\n💡 5. 권장사항")
print("="*80)

print("""
현재 상황 분석:
1. MemoryVLA는 LIBERO 데이터로 학습됨
2. MetaWorld는 다른 환경 (Zero-shot transfer)
3. 통계 데이터 불일치로 액션 스케일 문제 발생

해결 방안 (우선순위 순):

[A] LIBERO 환경에서 평가 (가장 정확)
    → eval_libero_proper.py 사용
    → 모델이 학습한 환경이므로 제대로 작동해야 함

[B] MetaWorld에서 평가 (Zero-shot)
    → eval_metaworld_fixed.py 사용
    → ACTION_SCALE 조정 필요 (1.0 → 10.0 시도)
    → 성공률 낮을 수 있음 (환경 불일치)

[C] 통계 데이터 수정
    1. LIBERO 데이터셋에서 실제 통계 계산
    2. dataset_statistics.json에 올바른 값 입력
    3. 정규화 비활성화 (Mean=0, Std=1)

현재 사용 중인 "핵" 방식의 문제점:
- bridge_orig ≠ libero_spatial (액션 분포 다름)
- 회전 정보 삭제 (7-DoF → 4-DoF)
- 환경 불일치 (LIBERO vs MetaWorld)

👉 결론: LIBERO 환경에서 먼저 테스트하세요!
""")

print("="*80)
