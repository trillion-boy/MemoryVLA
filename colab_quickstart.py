"""
MemoryVLA LIBERO Simulation - Google Colab Quickstart
복사해서 Colab 노트북 셀에 붙여넣기

각 섹션을 순서대로 실행하세요.
"""

# ============================================================================
# 셀 1: 현재 환경 확인
# ============================================================================
print("="*70)
print("🔍 Colab 환경 확인")
print("="*70)

import sys
print(f"Python: {sys.version}")

# GPU 확인
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# 현재 패키지 버전
import torch
print(f"\n현재 PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

try:
    import tensorflow as tf
    print(f"TensorFlow: {tf.__version__}")
except:
    print("TensorFlow: 설치되지 않음")

import numpy as np
print(f"NumPy: {np.__version__}")


# ============================================================================
# 셀 2: PyTorch 다운그레이드 (2.2.0 + CUDA 12.1)
# ============================================================================
print("\n" + "="*70)
print("📦 PyTorch 다운그레이드 (2.5.x → 2.2.0)")
print("="*70)

!pip uninstall -y torch torchvision torchaudio -q
!pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121 -q

# 확인
import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA: {torch.version.cuda}")


# ============================================================================
# 셀 3: TensorFlow 다운그레이드 (2.15.0)
# ============================================================================
print("\n" + "="*70)
print("📦 TensorFlow 다운그레이드 (2.18.x → 2.15.0)")
print("="*70)

!pip uninstall -y tensorflow tensorflow-datasets -q
!pip install tensorflow==2.15.0 tensorflow_datasets==4.9.3 tensorflow_graphics==2021.12.3 -q

# 확인
import tensorflow as tf
print(f"✅ TensorFlow: {tf.__version__}")


# ============================================================================
# 셀 4: Flash Attention 설치
# ============================================================================
print("\n" + "="*70)
print("⚡ Flash Attention 2.5.5 설치")
print("="*70)

# Prebuilt wheel 다운로드
!wget -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# 설치
!pip install flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl -q

# 확인
try:
    import flash_attn
    print("✅ Flash Attention 설치 성공")
except ImportError:
    print("❌ Flash Attention 설치 실패")


# ============================================================================
# 셀 5: MemoryVLA 저장소 클론 및 설치
# ============================================================================
print("\n" + "="*70)
print("📥 MemoryVLA 저장소 클론")
print("="*70)

import os
os.chdir('/content')

# 저장소 클론
!git clone https://github.com/shihao1895/MemoryVLA.git -q
os.chdir('/content/MemoryVLA')

# 패키지 설치
!pip install -e . -q

print("✅ MemoryVLA 설치 완료")

# 주요 패키지 확인
!pip list | grep -E "transformers|accelerate|peft|timm"


# ============================================================================
# 셀 6: LIBERO 시뮬레이션 환경 설치
# ============================================================================
print("\n" + "="*70)
print("🤖 LIBERO 시뮬레이션 환경 설치")
print("="*70)

# 시스템 패키지 설치 (MuJoCo headless 렌더링)
!apt-get update -qq
!apt-get install -y -qq \
    libosmesa6-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libglfw3 \
    libglew-dev \
    patchelf \
    ffmpeg

# LIBERO 클론
os.chdir('/content')
!mkdir -p third_libs
!git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_libs/LIBERO -q
os.chdir('/content/third_libs/LIBERO')

# LIBERO 설치
!pip install -e . -q

# 환경 변수 설정
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

print("✅ LIBERO 설치 완료")


# ============================================================================
# 셀 7: Pretrained Checkpoint 다운로드
# ============================================================================
print("\n" + "="*70)
print("💾 Pretrained Checkpoint 다운로드")
print("="*70)

from huggingface_hub import snapshot_download

# LIBERO Spatial 체크포인트 (~14GB)
# 다른 태스크: memvla-libero-object, memvla-libero-goal, memvla-libero-100
checkpoint_path = snapshot_download(
    repo_id="shihao1895/memvla-libero-spatial",
    cache_dir="/content/checkpoints",
    resume_download=True,
)

print(f"✅ Checkpoint: {checkpoint_path}")

# 체크포인트 내부 구조 확인
!ls -lh {checkpoint_path}


# ============================================================================
# 셀 8: VRAM 확인 및 최적화
# ============================================================================
print("\n" + "="*70)
print("🧹 GPU 메모리 정리")
print("="*70)

import gc
import torch

gc.collect()
torch.cuda.empty_cache()

# VRAM 확인
!nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits

print("\n💡 Tip: 14-15GB VRAM에서 bfloat16 사용 시 충분히 실행 가능")


# ============================================================================
# 셀 9: 단일 태스크 빠른 테스트 (1회 trial)
# ============================================================================
print("\n" + "="*70)
print("🚀 빠른 테스트: Task 0, 1회 trial")
print("="*70)

os.chdir('/content/MemoryVLA')

# 단일 태스크만 1회 테스트
!python evaluation/libero/eval_libero_colab.py \
    --checkpoint_path {checkpoint_path} \
    --task_suite_name libero_spatial \
    --unnorm_key libero_spatial_no_noops \
    --num_trials_per_task 1 \
    --specific_task_ids 0 \
    --use_bf16 \
    --action_chunking_window 8

print("\n✅ 빠른 테스트 완료!")


# ============================================================================
# 셀 10: 전체 평가 실행 (10회 trials per task)
# ============================================================================
print("\n" + "="*70)
print("🎯 전체 평가 시작 (모든 태스크, 각 10회)")
print("="*70)

os.chdir('/content/MemoryVLA')

# 전체 태스크 평가
!python evaluation/libero/eval_libero_colab.py \
    --checkpoint_path {checkpoint_path} \
    --task_suite_name libero_spatial \
    --unnorm_key libero_spatial_no_noops \
    --num_trials_per_task 10 \
    --use_bf16 \
    --action_chunking_window 8 \
    --log_dir /content/logs/libero_spatial_full

print("\n✅ 평가 완료!")


# ============================================================================
# 셀 11: 결과 확인 및 비디오 시청
# ============================================================================
print("\n" + "="*70)
print("📊 결과 확인")
print("="*70)

# 결과 파일 읽기
results_file = "/content/logs/libero_spatial_full/results.txt"
if os.path.exists(results_file):
    with open(results_file, 'r') as f:
        print(f.read())

# 비디오 파일 목록
video_dir = "/content/logs/libero_spatial_full/videos"
if os.path.exists(video_dir):
    print(f"\n📁 비디오 파일 목록:")
    !ls -lh {video_dir}/*.mp4 | head -10

# 첫 번째 성공 비디오 재생
from IPython.display import Video
import glob

success_videos = glob.glob(f"{video_dir}/*success*.mp4")
if success_videos:
    print(f"\n🎥 성공 비디오 재생: {success_videos[0]}")
    display(Video(success_videos[0], width=600))


# ============================================================================
# 셀 12: 결과 다운로드
# ============================================================================
print("\n" + "="*70)
print("💾 결과 다운로드")
print("="*70)

# 로그 압축
!zip -r -q /content/libero_results.zip /content/logs/

# 다운로드
from google.colab import files
files.download('/content/libero_results.zip')

print("✅ 결과 다운로드 완료!")


# ============================================================================
# 선택: 다른 태스크 평가
# ============================================================================
"""
# LIBERO Object 평가
checkpoint_path_object = snapshot_download(
    repo_id="shihao1895/memvla-libero-object",
    cache_dir="/content/checkpoints",
)

!python evaluation/libero/eval_libero_colab.py \
    --checkpoint_path {checkpoint_path_object} \
    --task_suite_name libero_object \
    --unnorm_key libero_object_no_noops \
    --num_trials_per_task 10 \
    --use_bf16


# LIBERO Goal 평가
checkpoint_path_goal = snapshot_download(
    repo_id="shihao1895/memvla-libero-goal",
    cache_dir="/content/checkpoints",
)

!python evaluation/libero/eval_libero_colab.py \
    --checkpoint_path {checkpoint_path_goal} \
    --task_suite_name libero_goal \
    --unnorm_key libero_goal_no_noops \
    --num_trials_per_task 10 \
    --use_bf16


# LIBERO Long (Long-10, Long-90) 평가
checkpoint_path_100 = snapshot_download(
    repo_id="shihao1895/memvla-libero-100",
    cache_dir="/content/checkpoints",
)

!python evaluation/libero/eval_libero_colab.py \
    --checkpoint_path {checkpoint_path_100} \
    --task_suite_name libero_10 \
    --unnorm_key libero_100_no_noops \
    --num_trials_per_task 10 \
    --use_bf16
"""


# ============================================================================
# 트러블슈팅
# ============================================================================
"""
# 1. VRAM 부족 시
# - action_chunking_window를 8 → 4로 축소
# - num_trials_per_task를 10 → 5로 축소

!python evaluation/libero/eval_libero_colab.py \
    --checkpoint_path {checkpoint_path} \
    --task_suite_name libero_spatial \
    --unnorm_key libero_spatial_no_noops \
    --num_trials_per_task 5 \
    --use_bf16 \
    --action_chunking_window 4


# 2. MuJoCo 렌더링 오류 시
!apt-get install -y --reinstall libosmesa6-dev
os.environ['MUJOCO_GL'] = 'osmesa'


# 3. Flash Attention 오류 시
!pip install flash-attn==2.5.5 --no-build-isolation


# 4. 특정 태스크만 평가
!python evaluation/libero/eval_libero_colab.py \
    --checkpoint_path {checkpoint_path} \
    --task_suite_name libero_spatial \
    --unnorm_key libero_spatial_no_noops \
    --num_trials_per_task 10 \
    --specific_task_ids 0 1 2 \
    --use_bf16
"""


print("\n" + "="*70)
print("🎉 MemoryVLA LIBERO Simulation - Colab Quickstart 완료!")
print("="*70)
print("\n📚 도움말:")
print("  - 가이드: /content/MemoryVLA/COLAB_LIBERO_GUIDE.md")
print("  - 결과: /content/logs/")
print("  - 비디오: /content/logs/*/videos/")
print("\n💡 Tip:")
print("  - 빠른 테스트: num_trials_per_task=1, specific_task_ids=0")
print("  - 전체 평가: num_trials_per_task=10 (약 1-2시간 소요)")
print("  - VRAM 절약: use_bf16=True, action_chunking_window=4")
