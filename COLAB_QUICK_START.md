# MemoryVLA LIBERO Colab 빠른 시작 가이드

> **중요**: 2026년 1월 Colab 환경과 MemoryVLA 요구사항 버전 차이로 인해 **Conda 환경 필수**

---

## 🔍 버전 호환성 문제

### 현재 Colab 환경 (2026년 1월)
```
Python: 3.12.12
PyTorch: 2.9.0+cu126
TensorFlow: 2.19.0
NumPy: 2.0.2
CUDA: 12.5/12.6
```

### MemoryVLA 요구사항 (pyproject.toml)
```
Python: ==3.10  ❌
PyTorch: ==2.2.0  ❌
TensorFlow: ==2.15.0  ❌
NumPy: ==1.26.4  ❌
Flash Attention: ==2.5.5  ❌
```

### 해결 방법
**Conda로 Python 3.10 환경 생성** → 모든 패키지를 정확한 버전으로 설치

---

## 🚀 5단계 빠른 설치 (Colab 노트북에서 실행)

각 코드 블록을 새로운 Colab 셀에 복사하여 **순서대로** 실행하세요.

---

### **Step 1: Miniforge 설치 및 Python 3.10 환경 생성**

```python
%%bash
# Miniforge 다운로드 및 설치 (conda-forge 전용, TOS 문제 없음)
wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh
bash /tmp/miniforge.sh -b -p /opt/conda
rm /tmp/miniforge.sh

# Conda 초기화
/opt/conda/bin/conda init bash > /dev/null 2>&1
/opt/conda/bin/conda config --set auto_activate_base false

# Python 3.10 환경 생성
/opt/conda/bin/conda create -n memvla python=3.10 -y

echo "✅ Python 3.10 환경 생성 완료"

# 환경 확인
source /opt/conda/etc/profile.d/conda.sh
conda activate memvla
python --version
```

**예상 시간**: 2-3분

---

### **Step 2: PyTorch, TensorFlow 설치 (Flash Attention 제외)**

```python
%%bash
source /opt/conda/etc/profile.d/conda.sh
conda activate memvla

pip install --upgrade pip -q

# PyTorch 2.2.0 + CUDA 12.1
echo "Installing PyTorch 2.2.0..."
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121 -q

# TensorFlow 2.15.0
echo "Installing TensorFlow 2.15.0..."
pip install tensorflow==2.15.0 tensorflow_datasets==4.9.3 \
    tensorflow_graphics==2021.12.3 -q

# NumPy 1.26.4
pip install numpy==1.26.4 -q

# Flash Attention 건너뜀 (inference에서 불필요, T4 GPU 컴파일 이슈)
echo ""
echo "⚠️  Flash Attention 건너뜀 (inference에서 불필요)"
echo "    PyTorch 2.2.0의 SDPA (Scaled Dot Product Attention) 자동 사용"

# 버전 확인
echo ""
echo "✅ 패키지 설치 완료:"
python -c "import sys; print(f'Python: {sys.version}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import numpy as np; print(f'NumPy: {np.__version__}')"
```

**예상 시간**: 3-5분 (Flash Attention 컴파일 없어서 빠름!)

> **Note**: Flash Attention은 training에서만 필요하며, inference에서는 PyTorch 2.0+의 내장 SDPA가 자동으로 사용됩니다. T4 GPU에서 Flash Attention 컴파일은 실패율이 높고 시간이 오래 걸리므로 생략합니다.

---

### **Step 3: MemoryVLA 및 LIBERO 설치**

```python
%%bash
source /opt/conda/etc/profile.d/conda.sh
conda activate memvla

cd /content

# MemoryVLA 클론 (수정된 버전 - Python 3.10.x 지원)
echo "Cloning MemoryVLA..."
git clone -b claude/colab-memoryvla-simulation-ieGpy https://github.com/trillion-boy/MemoryVLA.git -q
cd MemoryVLA

# MemoryVLA 설치
echo "Installing MemoryVLA..."
pip install -e . -q

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

# 시스템 패키지 설치 (LIBERO 의존성)
echo ""
echo "Installing system packages for LIBERO..."
apt-get update -qq 2>&1 | grep -v "Skipping acquire"
apt-get install -y -qq libosmesa6-dev libgl1-mesa-dev libglu1-mesa-dev \
    libglfw3 libglew-dev patchelf ffmpeg 2>&1 | grep -v "Skipping acquire"

# LIBERO 클론 및 설치
cd /content
mkdir -p third_libs
echo "Installing LIBERO..."
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_libs/LIBERO -q
cd third_libs/LIBERO
pip install -e . -q

echo "✅ LIBERO 설치 완료"

# LIBERO 설치 확인
python << 'LIBERO_CHECK'
try:
    from libero.libero import benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suites = list(benchmark_dict.keys())
    print(f"✅ LIBERO import 성공")
    print(f"✅ Task suites: {task_suites}")
except Exception as e:
    print(f"❌ LIBERO import 실패: {e}")
    exit(1)
LIBERO_CHECK
```

**예상 시간**: 5-10분

> **Note**: Flash Attention이 dependencies에서 제거되었습니다 (optional-dependencies[training]로 이동). Inference에서는 PyTorch 2.2.0의 내장 SDPA가 자동으로 사용됩니다.

---

### **Step 4: Checkpoint 다운로드**

```python
%%bash
source /opt/conda/etc/profile.d/conda.sh
conda activate memvla

cd /content

python << 'EOF'
from huggingface_hub import snapshot_download

print("📥 LIBERO Spatial 체크포인트 다운로드 중... (~14GB)")
checkpoint_path = snapshot_download(
    repo_id="shihao1895/memvla-libero-spatial",
    cache_dir="/content/checkpoints",
    resume_download=True,
)

print(f"✅ Checkpoint 경로: {checkpoint_path}")

# 경로를 파일에 저장 (다음 단계에서 사용)
with open("/content/checkpoint_path.txt", "w") as f:
    f.write(checkpoint_path)
EOF
```

**예상 시간**: 10-20분 (인터넷 속도에 따라 다름)

---

### **Step 5: 빠른 테스트 (1개 task, 1번 trial)**

```python
%%bash
source /opt/conda/etc/profile.d/conda.sh
conda activate memvla

# 환경 변수 설정
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export TOKENIZERS_PARALLELISM=false

cd /content/MemoryVLA

# 체크포인트 경로 읽기
CHECKPOINT_PATH=$(cat /content/checkpoint_path.txt)

echo "🚀 빠른 테스트 시작 (Task 0, 1회 trial)"

python evaluation/libero/eval_libero_colab.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --task_suite_name libero_spatial \
    --unnorm_key libero_spatial_no_noops \
    --num_trials_per_task 1 \
    --specific_task_ids 0 \
    --use_bf16 \
    --action_chunking_window 8

echo "✅ 테스트 완료!"
```

**예상 시간**: 3-5분

**예상 출력**:
```
Task 1/1: LIBERO_SPATIAL_0
Episode 1/1: ✅ Success

📊 Task 1 Results:
   Success Rate: 100.0% (1/1)

🎯 FINAL RESULTS
Overall Success Rate: 100.0% (1/1)
```

---

## 🎯 전체 평가 실행 (모든 task, 10회 trials)

빠른 테스트가 성공하면 전체 평가를 실행하세요:

```python
%%bash
source /opt/conda/etc/profile.d/conda.sh
conda activate memvla

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export TOKENIZERS_PARALLELISM=false

cd /content/MemoryVLA

CHECKPOINT_PATH=$(cat /content/checkpoint_path.txt)

echo "🎯 전체 평가 시작 (모든 태스크, 각 10회)"
echo "⏱️  예상 소요 시간: 1-2시간"

python evaluation/libero/eval_libero_colab.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --task_suite_name libero_spatial \
    --unnorm_key libero_spatial_no_noops \
    --num_trials_per_task 10 \
    --use_bf16 \
    --action_chunking_window 8 \
    --log_dir /content/logs/libero_spatial_full

echo "✅ 평가 완료!"
```

---

## 📊 결과 확인 및 다운로드

### 결과 텍스트 확인

```python
!cat /content/logs/libero_spatial_full/results.txt
```

### 비디오 재생

```python
from IPython.display import Video
import glob

# 성공 비디오 찾기
success_videos = glob.glob("/content/logs/*/videos/*success*.mp4")
if success_videos:
    print(f"✅ 성공 비디오 {len(success_videos)}개 발견")
    print(f"첫 번째 성공 비디오 재생:")
    display(Video(success_videos[0], width=600))
else:
    print("⚠️  성공 비디오가 없습니다.")
```

### 결과 다운로드

```python
!zip -r -q /content/libero_results.zip /content/logs/
from google.colab import files
files.download('/content/libero_results.zip')
```

---

## 🔧 트러블슈팅

### 1. VRAM 부족 오류

```python
# action_chunking_window 축소
--action_chunking_window 4  # 기본 8 → 4

# trials 축소
--num_trials_per_task 5  # 기본 10 → 5
```

### 2. MuJoCo 렌더링 오류

```bash
%%bash
apt-get install -y --reinstall libosmesa6-dev
export MUJOCO_GL=osmesa
```

### 3. Python 버전 확인

```bash
%%bash
source /opt/conda/etc/profile.d/conda.sh
conda activate memvla
python --version  # 반드시 Python 3.10.x
```

### 4. 환경 리셋

```bash
%%bash
/opt/conda/bin/conda remove -n memvla --all -y
# 그 다음 Step 1부터 다시 실행
```

---

## 📚 다른 Task Suite 평가

### LIBERO Object

```python
%%bash
source /opt/conda/etc/profile.d/conda.sh
conda activate memvla

cd /content

python << 'EOF'
from huggingface_hub import snapshot_download

checkpoint_path = snapshot_download(
    repo_id="shihao1895/memvla-libero-object",
    cache_dir="/content/checkpoints",
)
with open("/content/checkpoint_path_object.txt", "w") as f:
    f.write(checkpoint_path)
EOF
```

```bash
%%bash
source /opt/conda/etc/profile.d/conda.sh
conda activate memvla
export MUJOCO_GL=osmesa

cd /content/MemoryVLA

python evaluation/libero/eval_libero_colab.py \
    --checkpoint_path $(cat /content/checkpoint_path_object.txt) \
    --task_suite_name libero_object \
    --unnorm_key libero_object_no_noops \
    --num_trials_per_task 10 \
    --use_bf16
```

### LIBERO Goal

```python
checkpoint_path = snapshot_download(
    repo_id="shihao1895/memvla-libero-goal",
    cache_dir="/content/checkpoints",
)
```

```bash
--task_suite_name libero_goal \
--unnorm_key libero_goal_no_noops \
```

### LIBERO Long (10/90)

```python
checkpoint_path = snapshot_download(
    repo_id="shihao1895/memvla-libero-100",
    cache_dir="/content/checkpoints",
)
```

```bash
# Long-10
--task_suite_name libero_10 \
--unnorm_key libero_100_no_noops \

# Long-90
--task_suite_name libero_90 \
--unnorm_key libero_100_no_noops \
```

---

## 📊 예상 성공률 (논문 기준)

| Task Suite | Expected Success Rate | Trials |
|------------|----------------------|--------|
| LIBERO Spatial | ~98% | 50 |
| LIBERO Object | ~98% | 50 |
| LIBERO Goal | ~96% | 50 |
| LIBERO Long-10 | ~93% | 50 |
| LIBERO Long-90 | ~96% | 50 |

---

## ⏱️ 전체 소요 시간 요약

- **Step 1**: Miniconda 설치 (2-3분)
- **Step 2**: PyTorch/TensorFlow 설치 (5-7분)
- **Step 3**: MemoryVLA/LIBERO 설치 (10-15분)
- **Step 4**: Checkpoint 다운로드 (10-20분)
- **Step 5**: 빠른 테스트 (3-5분)
- **전체 평가**: (60-120분)

**총 설치 시간**: ~30-50분
**전체 소요 시간**: ~2-3시간

---

## 💡 중요 사항

1. **모든 Python 명령은 `conda activate memvla` 환경에서 실행**
2. **`MUJOCO_GL=osmesa` 환경 변수 필수** (headless 렌더링)
3. **`%%bash` 매직 명령 사용** (Colab 셀에서 bash 실행)
4. **GPU T4 (15GB VRAM) 권장** (무료 Colab 기본 GPU)

---

**작성일**: 2026-01-22
**대상 환경**: Google Colab (무료 T4 GPU)
**MemoryVLA 버전**: 0.0.1 (Python 3.10 전용)
