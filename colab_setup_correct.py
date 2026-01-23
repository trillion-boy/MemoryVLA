"""
MemoryVLA LIBERO Colab 설치 스크립트 (정확한 버전)

현재 Colab 환경 (2026년 1월):
  - Python 3.12.12 → 3.10 필요 ❌
  - PyTorch 2.9.0+cu126 → 2.2.0+cu121 필요 ❌
  - TensorFlow 2.19.0 → 2.15.0 필요 ❌
  - NumPy 2.0.2 → 1.26.4 필요 ❌

해결: Conda로 Python 3.10 환경 생성
"""

# ============================================================================
# 셀 1: 현재 Colab 환경 확인
# ============================================================================
print("="*80)
print("🔍 현재 Colab 환경 확인")
print("="*80)

import sys
print(f"Python: {sys.version}")

!nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv

import torch
print(f"\n현재 PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")

import numpy as np
print(f"NumPy: {np.__version__}")

print("\n" + "="*80)
print("❌ 현재 환경은 MemoryVLA와 호환되지 않습니다!")
print("   MemoryVLA 요구사항:")
print("   - Python: ==3.10")
print("   - PyTorch: ==2.2.0")
print("   - TensorFlow: ==2.15.0")
print("   - NumPy: ==1.26.4")
print("="*80)


# ============================================================================
# 셀 2: Miniforge 설치 (TOS 문제 없음)
# ============================================================================
print("\n" + "="*80)
print("📦 Miniforge 설치 (conda-forge 전용)")
print("="*80)

import os
import sys

# Miniforge 다운로드 및 설치
!wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh
!bash /tmp/miniforge.sh -b -p /opt/conda
!rm /tmp/miniforge.sh

# PATH 업데이트
os.environ["PATH"] = f"/opt/conda/bin:{os.environ['PATH']}"

# Conda 초기화
!conda init bash > /dev/null 2>&1
!conda config --set auto_activate_base false

print("✅ Miniforge 설치 완료")
!conda --version


# ============================================================================
# 셀 3: Python 3.10 환경 생성 및 필수 패키지 설치
# ============================================================================
print("\n" + "="*80)
print("🐍 Python 3.10 환경 생성")
print("="*80)

# Python 3.10 환경 생성
!conda create -n memvla python=3.10 -y

print("✅ Python 3.10 환경 생성 완료")

# Conda 환경 활성화 및 패키지 설치를 위한 스크립트 작성
install_script = """
#!/bin/bash
set -e

source /opt/conda/etc/profile.d/conda.sh
conda activate memvla

echo "현재 Python 버전:"
python --version

# pip 업그레이드
pip install --upgrade pip

# PyTorch 2.2.0 + CUDA 12.1 설치
echo "PyTorch 2.2.0 설치 중..."
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# TensorFlow 2.15.0 설치
echo "TensorFlow 2.15.0 설치 중..."
pip install tensorflow==2.15.0 tensorflow_datasets==4.9.3 tensorflow_graphics==2021.12.3

# NumPy 1.26.4 설치
echo "NumPy 1.26.4 설치 중..."
pip install numpy==1.26.4

# Flash Attention 건너뜀 (inference에서 불필요, T4 GPU 컴파일 이슈)
echo ""
echo "⚠️  Flash Attention 건너뜀 (inference에서 불필요)"
echo "    PyTorch 2.2.0의 SDPA (Scaled Dot Product Attention) 자동 사용"

echo ""
echo "✅ 주요 패키지 설치 완료"

# 버전 확인
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import numpy as np; print(f'NumPy: {np.__version__}')"
"""

# 스크립트 파일로 저장
with open("/tmp/install_packages.sh", "w") as f:
    f.write(install_script)

# 실행
!bash /tmp/install_packages.sh

print("\n✅ 모든 주요 패키지 설치 완료!")


# ============================================================================
# 셀 4: MemoryVLA 저장소 클론 및 설치
# ============================================================================
print("\n" + "="*80)
print("📥 MemoryVLA 저장소 클론 및 설치")
print("="*80)

memvla_install_script = """
#!/bin/bash
set -e

source /opt/conda/etc/profile.d/conda.sh
conda activate memvla

cd /content

# MemoryVLA 클론 (수정된 버전)
if [ -d "MemoryVLA" ]; then
    rm -rf MemoryVLA
fi
echo "Cloning MemoryVLA (Python 3.10.x 지원 버전)..."
git clone -b claude/colab-memoryvla-simulation-ieGpy https://github.com/trillion-boy/MemoryVLA.git
cd MemoryVLA

# editable 모드로 설치
echo "Installing MemoryVLA..."
pip install -e .

echo "✅ MemoryVLA 설치 완료"

# 설치 확인
python -c "
try:
    from vla import load_vla
    print('✅ VLA 모듈 import 성공')
except Exception as e:
    print(f'❌ VLA 모듈 import 실패: {e}')
    exit(1)
"

# 설치된 주요 패키지 확인
pip list | grep -E "transformers|accelerate|peft|timm"
"""

with open("/tmp/install_memvla.sh", "w") as f:
    f.write(memvla_install_script)

!bash /tmp/install_memvla.sh


# ============================================================================
# 셀 5: LIBERO 시뮬레이션 환경 설치
# ============================================================================
print("\n" + "="*80)
print("🤖 LIBERO 시뮬레이션 환경 설치")
print("="*80)

# 시스템 패키지 설치
print("시스템 패키지 설치 중...")
!apt-get update
!apt-get install -y \
    libosmesa6-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libglfw3 \
    libglew-dev \
    patchelf \
    ffmpeg

libero_install_script = """
#!/bin/bash
set -e

source /opt/conda/etc/profile.d/conda.sh
conda activate memvla

cd /content

# LIBERO 클론
mkdir -p third_libs
if [ -d "third_libs/LIBERO" ]; then
    rm -rf third_libs/LIBERO
fi
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_libs/LIBERO
cd third_libs/LIBERO

# LIBERO 설치
pip install -e .

echo "✅ LIBERO 설치 완료"

# LIBERO config 미리 생성
mkdir -p /root/.libero
cat > /root/.libero/config.yaml << 'CONFIG_EOF'
benchmark_root: /content/third_libs/LIBERO/libero/libero
bddl_files: /content/third_libs/LIBERO/libero/libero/bddl_files
init_states: /content/third_libs/LIBERO/libero/libero/init_files
datasets: /content/third_libs/LIBERO/libero/datasets
assets: /content/third_libs/LIBERO/libero/libero/assets
CONFIG_EOF

# LIBERO 설치 확인
echo ""
echo "LIBERO 검증 중..."
export PYTHONPATH=/content/third_libs/LIBERO:$PYTHONPATH
/opt/conda/envs/memvla/bin/python -c "
from libero.libero import benchmark
benchmark_dict = benchmark.get_benchmark_dict()
task_suites = list(benchmark_dict.keys())
print('✅ LIBERO import 성공')
print(f'✅ Task suites: {task_suites}')
"
"""

with open("/tmp/install_libero.sh", "w") as f:
    f.write(libero_install_script)

!bash /tmp/install_libero.sh


# ============================================================================
# 셀 6: 환경 변수 설정 및 최종 확인
# ============================================================================
print("\n" + "="*80)
print("⚙️  환경 변수 설정 및 최종 확인")
print("="*80)

final_check_script = """
#!/bin/bash
set -e

source /opt/conda/etc/profile.d/conda.sh
conda activate memvla

# 환경 변수 설정
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export TOKENIZERS_PARALLELISM=false

echo "환경 변수 설정:"
echo "  MUJOCO_GL=$MUJOCO_GL"
echo "  PYOPENGL_PLATFORM=$PYOPENGL_PLATFORM"

echo ""
echo "="*80
echo "✅ 최종 패키지 버전 확인"
echo "="*80

python --version
python -c "import torch; print(f'PyTorch: {torch.__version__} (CUDA: {torch.version.cuda})')"
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import numpy as np; print(f'NumPy: {np.__version__}')"
python -c "import flash_attn; print('Flash Attention: ✅')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')"

echo ""
echo "="*80
echo "🎉 모든 설치 완료!"
echo "="*80
"""

with open("/tmp/final_check.sh", "w") as f:
    f.write(final_check_script)

!bash /tmp/final_check.sh


# ============================================================================
# 셀 7: Pretrained Checkpoint 다운로드
# ============================================================================
print("\n" + "="*80)
print("💾 Pretrained Checkpoint 다운로드")
print("="*80)

download_checkpoint_script = """
#!/bin/bash
set -e

source /opt/conda/etc/profile.d/conda.sh
conda activate memvla

cd /content

python << 'PYEOF'
from huggingface_hub import snapshot_download

print("LIBERO Spatial 체크포인트 다운로드 중... (~14GB)")
checkpoint_path = snapshot_download(
    repo_id="shihao1895/memvla-libero-spatial",
    cache_dir="/content/checkpoints",
    resume_download=True,
)

print(f"✅ Checkpoint: {checkpoint_path}")

# 체크포인트 구조 확인
import os
print("\\n체크포인트 파일:")
for root, dirs, files in os.walk(checkpoint_path):
    level = root.replace(checkpoint_path, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files[:5]:  # 처음 5개만 표시
        print(f'{subindent}{file}')
    if len(files) > 5:
        print(f'{subindent}... ({len(files)-5} more files)')
PYEOF
"""

with open("/tmp/download_checkpoint.sh", "w") as f:
    f.write(download_checkpoint_script)

!bash /tmp/download_checkpoint.sh


# ============================================================================
# 셀 8: GPU 메모리 정리 및 확인
# ============================================================================
print("\n" + "="*80)
print("🧹 GPU 메모리 확인")
print("="*80)

cleanup_script = """
#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate memvla

python << 'PYEOF'
import gc
import torch

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"✅ GPU 메모리 정리 완료")
    print(f"   사용 가능한 GPU: {torch.cuda.get_device_name(0)}")
    print(f"   총 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
PYEOF
"""

with open("/tmp/cleanup.sh", "w") as f:
    f.write(cleanup_script)

!bash /tmp/cleanup.sh

!nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits

print("\n💡 Tip: 14-15GB VRAM에서 bfloat16 사용 시 충분히 실행 가능")


# ============================================================================
# 셀 9: 빠른 테스트 (1 task, 1 trial)
# ============================================================================
print("\n" + "="*80)
print("🚀 빠른 테스트: Task 0, 1회 trial")
print("="*80)

quick_test_script = """
#!/bin/bash
set -e

source /opt/conda/etc/profile.d/conda.sh
conda activate memvla

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export TOKENIZERS_PARALLELISM=false

cd /content/MemoryVLA

# 체크포인트 경로 찾기
CHECKPOINT_PATH=$(find /content/checkpoints/models--shihao1895--memvla-libero-spatial/snapshots -type d -mindepth 1 -maxdepth 1 | head -1)

echo "Checkpoint path: $CHECKPOINT_PATH"

# 빠른 테스트 실행
python evaluation/libero/eval_libero_colab.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --task_suite_name libero_spatial \
    --unnorm_key libero_spatial_no_noops \
    --num_trials_per_task 1 \
    --specific_task_ids 0 \
    --use_bf16 \
    --action_chunking_window 8

echo "✅ 빠른 테스트 완료!"
"""

with open("/tmp/quick_test.sh", "w") as f:
    f.write(quick_test_script)

!bash /tmp/quick_test.sh


# ============================================================================
# 셀 10: 전체 평가 실행 (모든 태스크, 10회 trials)
# ============================================================================
print("\n" + "="*80)
print("🎯 전체 평가 시작")
print("="*80)

full_eval_script = """
#!/bin/bash
set -e

source /opt/conda/etc/profile.d/conda.sh
conda activate memvla

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export TOKENIZERS_PARALLELISM=false

cd /content/MemoryVLA

# 체크포인트 경로 찾기
CHECKPOINT_PATH=$(find /content/checkpoints/models--shihao1895--memvla-libero-spatial/snapshots -type d -mindepth 1 -maxdepth 1 | head -1)

echo "전체 평가 시작 (모든 태스크, 각 10회 trials)"
echo "예상 소요 시간: 1-2시간"
echo ""

# 전체 평가 실행
python evaluation/libero/eval_libero_colab.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --task_suite_name libero_spatial \
    --unnorm_key libero_spatial_no_noops \
    --num_trials_per_task 10 \
    --use_bf16 \
    --action_chunking_window 8 \
    --log_dir /content/logs/libero_spatial_full

echo "✅ 전체 평가 완료!"
"""

with open("/tmp/full_eval.sh", "w") as f:
    f.write(full_eval_script)

# 주의: 이 명령은 주석 처리해두고, 사용자가 필요할 때 실행하도록 함
print("전체 평가를 실행하려면 다음 명령을 실행하세요:")
print("!bash /tmp/full_eval.sh")

# 실제 실행하려면 아래 주석 해제
# !bash /tmp/full_eval.sh


# ============================================================================
# 셀 11: 결과 확인 및 비디오 시청
# ============================================================================
print("\n" + "="*80)
print("📊 결과 확인")
print("="*80)

view_results_script = """
#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate memvla

cd /content

python << 'PYEOF'
import os
import glob

# 결과 파일 확인
results_files = glob.glob("/content/logs/*/results.txt")
if results_files:
    print("결과 파일:")
    for rf in results_files:
        print(f"\n{'='*70}")
        print(f"📄 {rf}")
        print('='*70)
        with open(rf, 'r') as f:
            print(f.read())
else:
    print("⚠️  아직 결과 파일이 없습니다. 평가를 먼저 실행하세요.")

# 비디오 파일 목록
video_files = glob.glob("/content/logs/*/videos/*.mp4")
if video_files:
    print(f"\n📁 비디오 파일 개수: {len(video_files)}")
    print(f"   위치: /content/logs/*/videos/")

    # 성공/실패 비디오 분류
    success_videos = [v for v in video_files if 'success' in v]
    fail_videos = [v for v in video_files if 'fail' in v]
    print(f"   성공: {len(success_videos)}")
    print(f"   실패: {len(fail_videos)}")
else:
    print("\n⚠️  아직 비디오 파일이 없습니다.")
PYEOF
"""

with open("/tmp/view_results.sh", "w") as f:
    f.write(view_results_script)

!bash /tmp/view_results.sh

# 비디오 재생 (Jupyter/Colab에서만 작동)
print("\n비디오를 재생하려면 다음 코드를 실행하세요:")
print("""
from IPython.display import Video
import glob

success_videos = glob.glob("/content/logs/*/videos/*success*.mp4")
if success_videos:
    display(Video(success_videos[0], width=600))
""")


# ============================================================================
# 셀 12: 결과 다운로드
# ============================================================================
print("\n" + "="*80)
print("💾 결과 다운로드")
print("="*80)

print("결과를 다운로드하려면 다음 코드를 실행하세요:")
print("""
!zip -r -q /content/libero_results.zip /content/logs/
from google.colab import files
files.download('/content/libero_results.zip')
""")


# ============================================================================
# 유틸리티: Conda 환경에서 Python 실행하기
# ============================================================================
print("\n" + "="*80)
print("📚 유틸리티 함수")
print("="*80)

print("""
이후 셀에서 MemoryVLA를 사용하려면 다음과 같이 실행하세요:

방법 1: Shell 스크립트로 실행
```bash
%%bash
source /opt/conda/etc/profile.d/conda.sh
conda activate memvla
export MUJOCO_GL=osmesa

python your_script.py
```

방법 2: 래퍼 스크립트 사용
```python
def run_in_memvla_env(command):
    script = f'''
#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate memvla
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
{command}
'''
    with open('/tmp/run_cmd.sh', 'w') as f:
        f.write(script)
    !bash /tmp/run_cmd.sh

# 사용 예시
run_in_memvla_env('python evaluation/libero/eval_libero_colab.py --help')
```
""")


# ============================================================================
# 트러블슈팅
# ============================================================================
print("\n" + "="*80)
print("🔧 트러블슈팅")
print("="*80)

print("""
1. VRAM 부족 오류 시:
   - action_chunking_window를 8 → 4로 축소
   - num_trials_per_task를 10 → 5로 축소

2. MuJoCo 렌더링 오류 시:
   !apt-get install -y --reinstall libosmesa6-dev
   export MUJOCO_GL=osmesa

3. Flash Attention 오류 시:
   conda activate memvla
   pip uninstall flash-attn
   pip install flash-attn==2.5.5 --no-build-isolation

4. Python 버전 확인:
   source /opt/conda/etc/profile.d/conda.sh
   conda activate memvla
   python --version  # 반드시 Python 3.10.x여야 함

5. 환경 리셋이 필요한 경우:
   conda remove -n memvla --all -y
   # 그 다음 셀 3부터 다시 실행
""")


print("\n" + "="*80)
print("🎉 MemoryVLA LIBERO Colab 설정 완료!")
print("="*80)
print("\n📝 다음 단계:")
print("  1. 빠른 테스트 실행 (셀 9)")
print("  2. 전체 평가 실행 (셀 10)")
print("  3. 결과 확인 (셀 11)")
print("  4. 결과 다운로드 (셀 12)")
print("\n💡 Tip:")
print("  - 모든 Python 명령은 'conda activate memvla' 환경에서 실행")
print("  - MUJOCO_GL=osmesa 환경 변수 필수")
print("  - 예상 소요 시간: 설치 30-40분 + 평가 1-2시간")
