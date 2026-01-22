# Google Colab에서 MemoryVLA + LIBERO 시뮬레이션 실행 가이드

> **목표**: 무료 Colab (14-15GB VRAM)에서 MemoryVLA를 사용하여 LIBERO 시뮬레이션 실행

---

## 📋 사전 확인사항

### Colab 기본 환경 (2024-2025)
- **Python**: 3.10.x ✅ (MemoryVLA 요구: ==3.10)
- **PyTorch**: 2.5.x + CUDA 12.4 ⚠️ (MemoryVLA 요구: 2.2.0 + CUDA 12.1)
- **TensorFlow**: 2.18.x ⚠️ (MemoryVLA 요구: 2.15.0)
- **NumPy**: 1.26.x ✅ (MemoryVLA 요구: 1.26.4)

### 버전 충돌 요소
1. **PyTorch**: 2.5.x → 2.2.0 다운그레이드 필수
2. **TensorFlow**: 2.18.x → 2.15.0 다운그레이드 필수
3. **Flash Attention**: 2.5.5 설치 필요 (prebuilt wheel 사용)
4. **CUDA Toolkit**: 12.4 → 12.1 호환성 확인 필요

---

## 🚀 단계별 실행 가이드

### **Step 1: 버전 호환성 확인 및 환경 초기화**

```python
# 1-1. 현재 Colab 환경 확인
!python --version
!nvcc --version
!nvidia-smi

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")

import numpy as np
print(f"NumPy version: {np.__version__}")
```

**예상 출력**:
```
Python version: 3.10.x
PyTorch version: 2.5.x+cu124
TensorFlow version: 2.18.x
NumPy version: 1.26.x
```

---

### **Step 2: 필수 패키지 다운그레이드 및 재설치**

```python
# 2-1. PyTorch 다운그레이드 (2.5.x → 2.2.0)
# CUDA 12.1용 PyTorch 2.2.0 설치
!pip uninstall -y torch torchvision torchaudio
!pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# 2-2. TensorFlow 다운그레이드 (2.18.x → 2.15.0)
!pip uninstall -y tensorflow tensorflow-datasets
!pip install tensorflow==2.15.0 tensorflow_datasets==4.9.3 tensorflow_graphics==2021.12.3

# 2-3. 버전 재확인
import torch
print(f"✅ PyTorch: {torch.__version__}")
import tensorflow as tf
print(f"✅ TensorFlow: {tf.__version__}")
```

---

### **Step 3: Flash Attention 설치 (Prebuilt Wheel)**

```python
# 3-1. Flash Attention 2.5.5 prebuilt wheel 다운로드 및 설치
# PyTorch 2.2.0 + CUDA 12.1 호환 버전
!wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
!pip install flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# 3-2. 설치 확인
try:
    import flash_attn
    print("✅ Flash Attention 설치 성공")
except ImportError:
    print("❌ Flash Attention 설치 실패")
```

---

### **Step 4: MemoryVLA 저장소 클론 및 설치**

```python
# 4-1. 저장소 클론
!git clone https://github.com/shihao1895/MemoryVLA.git
%cd MemoryVLA

# 4-2. MemoryVLA 패키지 설치 (editable mode)
!pip install -e .

# 4-3. 추가 의존성 확인
!pip list | grep -E "torch|transformers|peft|timm|accelerate"
```

---

### **Step 5: LIBERO 시뮬레이션 환경 설치**

```python
# 5-1. 시스템 패키지 설치 (MuJoCo headless 렌더링용)
!apt-get update
!apt-get install -y \
    libosmesa6-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libglfw3 \
    libglew-dev \
    patchelf \
    ffmpeg

# 5-2. LIBERO 저장소 클론
%cd /content
!git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_libs/LIBERO
%cd third_libs/LIBERO

# 5-3. LIBERO 설치
!pip install -e .

# 5-4. 환경 변수 설정 (headless 렌더링)
import os
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

print("✅ LIBERO 설치 완료")
```

---

### **Step 6: Pretrained Checkpoint 다운로드**

```python
# 6-1. Hugging Face Hub 로그인 (선택사항, public 모델은 불필요)
# from huggingface_hub import login
# login(token="YOUR_HF_TOKEN")

# 6-2. LIBERO Spatial 체크포인트 다운로드 (~14GB)
from huggingface_hub import snapshot_download

checkpoint_path = snapshot_download(
    repo_id="shihao1895/memvla-libero-spatial",
    cache_dir="/content/checkpoints",
    resume_download=True
)

print(f"✅ Checkpoint 다운로드 완료: {checkpoint_path}")

# 6-3. LIBERO 데이터셋 통계 다운로드 (unnorm_key용)
# LIBERO 학습 시 사용된 정규화 통계
!mkdir -p /content/datasets
%cd /content/datasets
!wget https://huggingface.co/datasets/shihao1895/libero-rlds/resolve/main/libero_spatial_no_noops_stats.json

print("✅ 데이터셋 통계 다운로드 완료")
```

**다운로드 가능한 체크포인트**:
- `shihao1895/memvla-libero-spatial` (Spatial 태스크)
- `shihao1895/memvla-libero-object` (Object 태스크)
- `shihao1895/memvla-libero-goal` (Goal 태스크)
- `shihao1895/memvla-libero-100` (Long-10, Long-90 태스크)

---

### **Step 7: VRAM 사용량 최적화 (중요!)**

```python
# 7-1. GPU 메모리 정리
import gc
import torch

gc.collect()
torch.cuda.empty_cache()

# 7-2. VRAM 체크
!nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# 7-3. bfloat16 추론 사용 (메모리 절약)
# deploy.py 실행 시 --use_bf16 플래그 추가
```

**메모리 예상 사용량**:
- 모델 (bf16): ~7GB
- LIBERO 환경: ~2GB
- 기타 오버헤드: ~2GB
- **총합**: ~11-12GB (14-15GB VRAM에서 여유 있음)

---

### **Step 8: 간소화된 평가 스크립트 작성**

Colab에서는 Flask 서버를 백그라운드로 실행하는 것보다 **직접 모델 호출**이 더 간단합니다.

```python
# 8-1. 간소화된 평가 스크립트 작성
%cd /content/MemoryVLA

# 파일 생성: evaluation/libero/eval_libero_colab.py
```

**`eval_libero_colab.py` 내용**:
```python
import os
os.environ['MUJOCO_GL'] = 'osmesa'

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from libero.libero import benchmark

from evaluation.libero.libero_utils import get_libero_env, get_libero_image, quat2axisangle, save_rollout_video
from evaluation.libero.robot_utils import set_seed_everywhere, DATE_TIME
from vla import load_vla

# TensorFlow CPU만 사용
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')


class DirectMemVLAPolicy:
    """Flask 서버 없이 직접 모델 호출"""
    def __init__(self, checkpoint_path, unnorm_key, use_bf16=True, action_chunking_window=8):
        self.vla = load_vla(
            model_id_or_path=checkpoint_path,
            load_for_training=False,
        )
        self.vla = self.vla.to("cuda").eval()

        if use_bf16:
            self.vla = self.vla.to(torch.bfloat16)
            print("✅ Using bfloat16 inference")

        self.unnorm_key = unnorm_key
        self.action_chunking_window = action_chunking_window

    def reset(self):
        """에피소드 초기화"""
        pass

    def predict(self, image, task_description, episode_first_frame='False'):
        """Action 예측"""
        # 이미지 전처리
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # 모델 예측
        unnormed_actions, _ = self.vla.predict_action(
            image=image,
            instruction=task_description,
            unnorm_key=self.unnorm_key,
            cfg_scale=1.5,
            use_ddim=True,
            num_ddim_steps=10,
            episode_first_frame=episode_first_frame,
        )

        # Action chunking
        actions = []
        for i in range(self.action_chunking_window):
            actions.append(unnormed_actions[i].cpu().numpy())

        return actions


def resize_image(image, size=(224, 224)):
    """이미지 리사이징"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    w, h = image.size
    left_margin = (w - h) // 2
    left_margin = min(max(left_margin, 0), w - h)
    image = image.crop((left_margin, 0, left_margin + h, h))
    image = image.resize(size, resample=Image.LANCZOS)

    # Center crop (random crop augmentation 대응)
    import math
    scale = 0.9
    new_w = int(w * math.sqrt(scale))
    new_h = int(h * math.sqrt(scale))
    margin_w = (w - new_w) // 2
    margin_h = (h - new_h) // 2
    image = image.crop((margin_w, margin_h, margin_w + new_w, margin_h + new_h))
    image = image.resize(size, resample=Image.LANCZOS)

    return image


def eval_libero_colab(
    checkpoint_path,
    task_suite_name="libero_spatial",
    unnorm_key="libero_spatial_no_noops",
    num_trials_per_task=10,  # Colab에서는 50회 대신 10회로 축소
    seed=7,
    use_bf16=True,
    action_chunking_window=8,
):
    """Colab용 LIBERO 평가"""

    # 시드 설정
    set_seed_everywhere(seed)

    # 로그 디렉토리
    log_dir = f"/content/logs/{task_suite_name}-{DATE_TIME}"
    os.makedirs(log_dir, exist_ok=True)

    # 정책 초기화
    print(f"Loading model from {checkpoint_path}...")
    policy = DirectMemVLAPolicy(
        checkpoint_path=checkpoint_path,
        unnorm_key=unnorm_key,
        use_bf16=use_bf16,
        action_chunking_window=action_chunking_window,
    )
    print("✅ Model loaded")

    # LIBERO 태스크 로드
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks = task_suite.n_tasks
    print(f"Task suite: {task_suite_name} ({num_tasks} tasks)")

    # 태스크별 최대 스텝 수
    max_steps_dict = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    max_steps = max_steps_dict.get(task_suite_name, 300)

    # 평가 시작
    total_episodes, total_successes = 0, 0

    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, resolution=256)

        print(f"\n{'='*60}")
        print(f"Task {task_id+1}/{num_tasks}: {task_description}")
        print(f"{'='*60}")

        task_successes = 0

        for episode_idx in tqdm(range(num_trials_per_task), desc=f"Task {task_id+1}"):
            # 환경 리셋
            env.reset()
            policy.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            episode_first_frame = 'True'
            replay_images = []
            t = 0
            done = False

            # 에피소드 실행
            while t < max_steps + 10:
                # 초기 안정화 대기
                if t < 10:
                    obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
                    t += 1
                    continue

                # 이미지 전처리
                img = get_libero_image(obs, resize_size=256)
                replay_images.append(img)
                resized_img = resize_image(img, size=(224, 224))

                # Action 예측
                actions = policy.predict(
                    image=resized_img,
                    task_description=task_description,
                    episode_first_frame=episode_first_frame,
                )
                episode_first_frame = 'False'

                # Action 실행 (chunking)
                for action in actions:
                    # Gripper 값 변환 (1.0 → -1.0, 0.0 → 1.0)
                    if action[6] == 1.0:
                        action[6] = -1.0
                    elif action[6] == 0.0:
                        action[6] = 1.0

                    obs, reward, done, info = env.step(action)
                    t += 1

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break

                if done:
                    break

            total_episodes += 1

            # 비디오 저장
            video_path = os.path.join(log_dir, f"task{task_id}_ep{episode_idx}_{'success' if done else 'fail'}.mp4")
            save_rollout_video(
                replay_images,
                total_episodes,
                success=done,
                task_description=task_description,
                log_file=None,
                rollout_dir=log_dir,
            )

            print(f"Episode {episode_idx+1}: {'✅ Success' if done else '❌ Fail'}")

        # 태스크 결과
        task_success_rate = task_successes / num_trials_per_task * 100
        print(f"\n📊 Task {task_id+1} Success Rate: {task_success_rate:.1f}% ({task_successes}/{num_trials_per_task})")

    # 최종 결과
    total_success_rate = total_successes / total_episodes * 100
    print(f"\n{'='*60}")
    print(f"🎯 FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Total Success Rate: {total_success_rate:.1f}% ({total_successes}/{total_episodes})")
    print(f"Logs saved to: {log_dir}")


if __name__ == "__main__":
    eval_libero_colab(
        checkpoint_path="/content/checkpoints/models--shihao1895--memvla-libero-spatial/snapshots/...",
        task_suite_name="libero_spatial",
        unnorm_key="libero_spatial_no_noops",
        num_trials_per_task=10,
        use_bf16=True,
    )
```

---

### **Step 9: 평가 실행**

```python
# 9-1. 체크포인트 경로 확인
import os
checkpoint_dir = "/content/checkpoints/models--shihao1895--memvla-libero-spatial/snapshots"
checkpoint_path = os.path.join(checkpoint_dir, os.listdir(checkpoint_dir)[0])
print(f"Checkpoint path: {checkpoint_path}")

# 9-2. 평가 실행
%cd /content/MemoryVLA

!python evaluation/libero/eval_libero_colab.py \
    --checkpoint_path {checkpoint_path} \
    --task_suite_name libero_spatial \
    --unnorm_key libero_spatial_no_noops \
    --num_trials_per_task 10 \
    --use_bf16
```

---

### **Step 10: 결과 확인 및 비디오 다운로드**

```python
# 10-1. 결과 로그 확인
!ls -lh /content/logs/

# 10-2. 성공/실패 비디오 확인
from IPython.display import Video
Video("/content/logs/libero_spatial-*/task0_ep0_success.mp4", width=400)

# 10-3. 결과 압축 및 다운로드
!zip -r /content/libero_results.zip /content/logs/
from google.colab import files
files.download('/content/libero_results.zip')
```

---

## 📊 예상 성능

| Task Suite | Expected Success Rate | Checkpoint |
|------------|----------------------|------------|
| LIBERO Spatial | ~98% | `memvla-libero-spatial` |
| LIBERO Object | ~98% | `memvla-libero-object` |
| LIBERO Goal | ~96% | `memvla-libero-goal` |
| LIBERO Long-10 | ~93% | `memvla-libero-100` |
| LIBERO Long-90 | ~96% | `memvla-libero-100` |

---

## ⚠️ 주의사항

### 1. **Colab 세션 타임아웃**
- 무료 Colab은 12시간 연속 사용 제한
- 50회 trials는 약 2-3시간 소요 → 10회로 축소 권장

### 2. **VRAM 부족 시**
```python
# Action chunking window 축소
action_chunking_window = 4  # 기본 8 → 4로 축소

# bfloat16 필수 사용
use_bf16 = True
```

### 3. **MuJoCo 렌더링 오류 시**
```bash
# osmesa 재설치
!apt-get install -y --reinstall libosmesa6-dev
```

### 4. **Flash Attention 컴파일 오류 시**
```python
# Prebuilt wheel 대신 pip 설치 (느림)
!pip install flash-attn --no-build-isolation
```

---

## 🎓 추가 팁

### A. 다른 태스크 시도
```python
# Object 태스크
checkpoint_path = snapshot_download("shihao1895/memvla-libero-object")
eval_libero_colab(
    checkpoint_path=checkpoint_path,
    task_suite_name="libero_object",
    unnorm_key="libero_object_no_noops",
)

# Goal 태스크
checkpoint_path = snapshot_download("shihao1895/memvla-libero-goal")
eval_libero_colab(
    checkpoint_path=checkpoint_path,
    task_suite_name="libero_goal",
    unnorm_key="libero_goal_no_noops",
)
```

### B. 단일 태스크만 평가
```python
# eval_libero_colab.py에서 task_id 필터링
# for task_id in range(num_tasks):
#     if task_id != 0:  # Task 0만 평가
#         continue
```

### C. 비디오 프레임 저장
```python
# replay_images를 PNG로 저장
import cv2
for i, img in enumerate(replay_images):
    cv2.imwrite(f"/content/logs/frame_{i:04d}.png", img)
```

---

## 📚 참고 자료

- [MemoryVLA GitHub](https://github.com/shihao1895/MemoryVLA)
- [LIBERO Benchmark](https://libero-project.github.io/)
- [OpenVLA Colab Tutorial](https://github.com/openvla/openvla)
- [Flash Attention Releases](https://github.com/Dao-AILab/flash-attention/releases)

---

**작성일**: 2025-01-22
**대상 환경**: Google Colab (무료 T4 GPU, 14-15GB VRAM)
**예상 소요 시간**: 설치 30분 + 평가 1-2시간
