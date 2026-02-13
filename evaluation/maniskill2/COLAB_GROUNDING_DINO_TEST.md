# Grounding DINO + SAM Detection Test (Colab)

기존 셀 1~4 실행 후 (모델 로딩까지 완료된 상태에서) 아래 셀들을 순서대로 실행.

---

## Cell 5: Grounding DINO + SAM 설치

```python
# ============================================
# [5] Grounding DINO + SAM 설치
# ============================================
# transformers 4.40.1에 GroundingDINO 내장되어 있음
# SAM도 transformers에 내장
# segment-anything-py는 별도 SAM 사용 시 필요 (여기선 transformers 내장 SAM 사용)

# 설치 확인
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import SamModel, SamProcessor
print("✅ Grounding DINO + SAM (transformers 내장) 사용 가능!")
```

## Cell 6: ManiSkill2 이미지 렌더링 + 자동 테스트

```python
# ============================================
# [6] Full Test: ManiSkill2 → Grounding DINO → SAM → Mask
# ============================================
from evaluation.maniskill2.test_grounding_dino_detection import run_full_test

results = run_full_test(
    save_dir="/content/grounding_dino_test",
    num_seeds=5,
    env_names=["PickCube-v0"],
    box_threshold=0.15,
)
```

## Cell 7: 결과 시각화 (이미 저장된 이미지 보기)

```python
# ============================================
# [7] 결과 확인
# ============================================
from IPython.display import display, Image as IPImage
import glob

detection_images = sorted(glob.glob("/content/grounding_dino_test/detections/*.png"))
for path in detection_images:
    print(f"\n{path}")
    display(IPImage(filename=path, width=900))
```

## Cell 6-alt: ManiSkill2 없이 단일 이미지 테스트

```python
# ============================================
# [6-alt] 단일 이미지 테스트 (ManiSkill2 없이)
# ============================================
import numpy as np
from PIL import Image
from evaluation.maniskill2.test_grounding_dino_detection import test_on_single_image

# 방법 1: 이미 저장된 이미지가 있는 경우
# img = np.array(Image.open("/content/some_maniskill_image.png"))

# 방법 2: ManiSkill2에서 직접 한 장 렌더
import gymnasium as gym
try:
    import mani_skill.envs
except:
    import mani_skill2.envs

env = gym.make("PickCube-v0", obs_mode="rgbd", control_mode="pd_ee_delta_pose")
obs, _ = env.reset(seed=42)

# RGB 추출
import torch
for cam in ["base_camera", "hand_camera"]:
    for key in ["sensor_data", "image"]:
        if key in obs and cam in obs[key] and "rgb" in obs[key][cam]:
            rgb = obs[key][cam]["rgb"]
            if isinstance(rgb, torch.Tensor):
                rgb = rgb.cpu().numpy()
            if rgb.ndim == 4:
                rgb = rgb[0]
            if rgb.max() <= 1.0:
                rgb = (rgb * 255).astype(np.uint8)
            img = rgb[:,:,:3]
            break
    else:
        continue
    break

env.close()
print(f"Image shape: {img.shape}")

# 테스트 실행
result = test_on_single_image(
    img,
    text_prompts=["red cube", "cube", "small red object", "red block", "object on table"],
    box_threshold=0.15,
    save_path="/content/grounding_dino_test/detections/single_test.png",
)

# 결과 출력
from IPython.display import display, Image as IPImage
display(IPImage(filename="/content/grounding_dino_test/detections/custom_seed0_detection.png", width=900))
```

## Cell 8: 다양한 threshold로 민감도 테스트

```python
# ============================================
# [8] Threshold 민감도 테스트
# ============================================
from evaluation.maniskill2.test_grounding_dino_detection import (
    load_grounding_dino, load_sam, detect_objects, segment_from_box,
    mask_to_patch_grid, visualize_detection
)
from PIL import Image
import numpy as np

# 모델 한 번만 로드
gdino_proc, gdino_model = load_grounding_dino()
sam_proc, sam_model = load_sam()

# 이미 렌더링된 이미지 사용
import glob
raw_images = sorted(glob.glob("/content/grounding_dino_test/raw_images/*.png"))

for threshold in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]:
    print(f"\n--- box_threshold = {threshold} ---")
    for img_path in raw_images[:3]:  # 처음 3장만
        pil = Image.open(img_path)
        dets = detect_objects(
            gdino_proc, gdino_model, pil,
            ["red cube"], box_threshold=threshold,
        )
        n = len(dets)
        best = dets[0]["score"] if dets else 0.0
        print(f"  {img_path.split('/')[-1]}: {n} detections, best={best:.3f}")
```
