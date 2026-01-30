"""
Colab Setup Script for MemoryVLA + ManiSkill2 Evaluation

This script provides cell-by-cell code for Google Colab to:
1. Install MemoryVLA
2. Download Bridge checkpoint
3. Setup ManiSkill2 environment
4. Run zero-shot evaluation (Bridge checkpoint → ManiSkill2 tasks)

Usage: Copy each cell to Colab and run sequentially
"""

# =============================================================================
# CELL 1: Check GPU and Disk Space
# =============================================================================
CELL_1 = '''
%%bash
echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv

echo ""
echo "=== Disk Space ==="
df -h /content
df -h /root

echo ""
echo "=== Current Directory ==="
pwd
ls -la
'''

# =============================================================================
# CELL 2: Install MemoryVLA (after your conda setup)
# =============================================================================
CELL_2 = '''
%%bash
source /opt/conda/etc/profile.d/conda.sh
conda activate memvla

cd /content

# Clone MemoryVLA
if [ ! -d "MemoryVLA" ]; then
    git clone https://github.com/shihao1895/MemoryVLA.git
    echo "✅ MemoryVLA cloned"
else
    echo "MemoryVLA already exists"
fi

cd MemoryVLA

# Install MemoryVLA
pip install -e . --quiet

# Install additional dependencies for evaluation
pip install opencv-python pillow imageio --quiet

echo ""
echo "✅ MemoryVLA 설치 완료"
python -c "import vla; print('vla module imported successfully')"
'''

# =============================================================================
# CELL 3: Install ManiSkill2 and SimplerEnv
# =============================================================================
CELL_3 = '''
%%bash
source /opt/conda/etc/profile.d/conda.sh
conda activate memvla

cd /content/MemoryVLA

# Install ManiSkill2
echo "Installing ManiSkill2..."
pip install mani-skill2==0.5.3 --quiet

# Install SimplerEnv dependencies
pip install gymnasium sapien==2.2.1 --quiet

# Clone SimplerEnv
mkdir -p third_libs
cd third_libs

if [ ! -d "SimplerEnv" ]; then
    git clone https://github.com/simpler-env/SimplerEnv.git
    cd SimplerEnv
    pip install -e . --quiet
    echo "✅ SimplerEnv installed"
else
    echo "SimplerEnv already exists"
fi

# Download ManiSkill2 assets (필수!)
echo ""
echo "Downloading ManiSkill2 assets..."
python -m mani_skill2.utils.download_asset all -y

echo ""
echo "✅ ManiSkill2 설치 완료"
'''

# =============================================================================
# CELL 4: Download Bridge Checkpoint from HuggingFace
# =============================================================================
CELL_4 = '''
%%bash
source /opt/conda/etc/profile.d/conda.sh
conda activate memvla

cd /content/MemoryVLA

# Create checkpoints directory
mkdir -p checkpoints

# Install huggingface_hub if needed
pip install huggingface_hub --quiet

# Download MemoryVLA Bridge checkpoint
python << 'EOF'
from huggingface_hub import snapshot_download
import os

# MemoryVLA Bridge checkpoint
checkpoint_path = snapshot_download(
    repo_id="shihao1895/memvla-bridge",
    local_dir="./checkpoints/memvla-bridge",
    local_dir_use_symlinks=False
)

print(f"✅ Checkpoint downloaded to: {checkpoint_path}")

# List files
for root, dirs, files in os.walk(checkpoint_path):
    for file in files:
        filepath = os.path.join(root, file)
        size = os.path.getsize(filepath) / (1024**3)  # GB
        print(f"  {file}: {size:.2f} GB")
EOF

echo ""
echo "✅ Bridge checkpoint 다운로드 완료"
ls -la checkpoints/memvla-bridge/
'''

# =============================================================================
# CELL 5: Verify Installation
# =============================================================================
CELL_5 = '''
%%bash
source /opt/conda/etc/profile.d/conda.sh
conda activate memvla

cd /content/MemoryVLA

python << 'EOF'
print("=" * 60)
print("Installation Verification")
print("=" * 60)

# Check imports
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"❌ PyTorch: {e}")

try:
    import mani_skill2
    print(f"✅ ManiSkill2: {mani_skill2.__version__}")
except Exception as e:
    print(f"❌ ManiSkill2: {e}")

try:
    from vla.load import load_vla
    print("✅ MemoryVLA: vla.load imported")
except Exception as e:
    print(f"❌ MemoryVLA: {e}")

# Check checkpoint
import os
ckpt_path = "./checkpoints/memvla-bridge"
if os.path.exists(ckpt_path):
    files = os.listdir(ckpt_path)
    print(f"✅ Checkpoint: {len(files)} files found")
else:
    print(f"❌ Checkpoint not found at {ckpt_path}")

print("=" * 60)
EOF
'''

# =============================================================================
# CELL 6: Run ManiSkill2 Evaluation (Main Script)
# =============================================================================
CELL_6 = '''
%%bash
source /opt/conda/etc/profile.d/conda.sh
conda activate memvla

cd /content/MemoryVLA

# Run evaluation
python evaluation/maniskill2/eval_maniskill2_bridge.py \
    --checkpoint ./checkpoints/memvla-bridge \
    --num-episodes 20 \
    --save-videos \
    --log-dir ./logs/eval_maniskill2_bridge

echo ""
echo "✅ Evaluation 완료!"
echo "Results: ./logs/eval_maniskill2_bridge/"
'''

if __name__ == "__main__":
    print("=" * 70)
    print("Colab Setup Instructions")
    print("=" * 70)
    print("""
이 파일의 각 CELL을 Colab 노트북에 순서대로 복사하여 실행하세요.

순서:
1. CELL 1: GPU/디스크 확인
2. (당신의 conda/pip 셋업 코드 실행)
3. CELL 2: MemoryVLA 설치
4. CELL 3: ManiSkill2 설치
5. CELL 4: Bridge checkpoint 다운로드
6. CELL 5: 설치 확인
7. CELL 6: Evaluation 실행

예상 디스크 사용량:
- MemoryVLA repo: ~500MB
- ManiSkill2 assets: ~5GB
- Bridge checkpoint: ~15GB
- 총: ~20GB

예상 소요 시간:
- 설치: 10-15분
- 다운로드: 10-20분 (인터넷 속도에 따라)
- Evaluation: 30-60분 (에피소드 수에 따라)
""")
    print("=" * 70)
