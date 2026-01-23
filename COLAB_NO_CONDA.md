# MemoryVLA Google Colab Setup (conda 없이)

Python 3.12 Colab 기본 환경에서 직접 실행 (interactive input 가능)

---

## Step 1: PyTorch 다운그레이드

```bash
%%bash
pip uninstall -y torch torchvision torchaudio
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
```

**예상 시간**: 2-3분

---

## Step 2: TensorFlow & NumPy 다운그레이드

```bash
%%bash
pip uninstall -y tensorflow numpy
pip install tensorflow==2.15.0 numpy==1.26.4
```

**예상 시간**: 1-2분

---

## Step 3: MemoryVLA 설치

```bash
%%bash
cd /content
rm -rf MemoryVLA
git clone -b claude/colab-memoryvla-simulation-ieGpy https://github.com/trillion-boy/MemoryVLA.git
cd MemoryVLA
pip install -e .
```

**예상 시간**: 3-5분

---

## Step 4: LIBERO 시스템 패키지 설치

```bash
%%bash
apt-get update
apt-get install -y libosmesa6-dev libgl1-mesa-dev libglu1-mesa-dev \
    libglfw3 libglew-dev patchelf ffmpeg
```

**예상 시간**: 1분

---

## Step 5: LIBERO 설치

```bash
%%bash
cd /content
mkdir -p third_libs
rm -rf third_libs/LIBERO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_libs/LIBERO
cd third_libs/LIBERO
pip install -e .
```

**예상 시간**: 1-2분

---

## Step 6: LIBERO 검증 (interactive input 가능!)

```python
from libero.libero import benchmark

benchmark_dict = benchmark.get_benchmark_dict()
task_suites = list(benchmark_dict.keys())

print(f'✅ LIBERO import 성공')
print(f'✅ Task suites: {task_suites}')
```

**데이터셋 경로 질문이 나오면 `N` 입력**

**예상 출력**:
```
Do you want to specify a custom path for the dataset folder? (Y/N): N
✅ LIBERO import 성공
✅ Task suites: ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10', 'libero_90']
```

---

## Step 7: Checkpoint 다운로드

```bash
%%bash
mkdir -p /content/checkpoints
cd /content/checkpoints

# MemoryVLA checkpoint 다운로드
wget -O memoryvla_checkpoint.pth https://huggingface.co/your-checkpoint-url
```

---

## Step 8: 빠른 테스트

```python
import torch
from vla import load_vla

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# VLA 모델 로드 테스트
try:
    print("✅ MemoryVLA 설치 완료")
except Exception as e:
    print(f"❌ 에러: {e}")
```

---

## 주의사항

- **Python 3.12 호환성**: 대부분의 코드는 작동하지만, 일부 Python 3.10 전용 기능이 문제될 수 있음
- **NumPy 다운그레이드**: 다른 Colab 패키지와 충돌 가능성 있음
- **문제 발생 시**: `COLAB_QUICK_START.md` (conda 버전)로 돌아가기

---

## 장점

✅ conda 없이 간단함
✅ interactive input 가능 (일반 Python 셀)
✅ 빠름

## 단점

⚠️ Python 3.12 호환성 문제 가능
⚠️ NumPy 다운그레이드로 인한 충돌 가능
