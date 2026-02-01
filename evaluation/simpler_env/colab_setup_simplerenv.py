#!/usr/bin/env python3
"""
SimplerEnv-Bridge VA Evaluation - Colab Setup Guide

=== SETUP CELLS FOR GOOGLE COLAB ===

This file contains the setup code for running SimplerEnv-Bridge VA evaluation
on Google Colab. Copy each cell to your Colab notebook.

Key differences from ManiSkill2 setup:
1. Install SimplerEnv instead of ManiSkill2
2. Use SimplerEnv-Bridge environment (same WidowX robot as training)
3. Test VA (Visual Aggregation) without rgb_overlay

=== CELL 1: Miniforge (SAME AS BEFORE) ===
"""

CELL_1_MINIFORGE = '''
# Cell 1: Install Miniforge (required for conda)
!wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
!bash Miniforge3-Linux-x86_64.sh -b -p /opt/miniforge3
!rm Miniforge3-Linux-x86_64.sh

import os
os.environ["PATH"] = "/opt/miniforge3/bin:" + os.environ["PATH"]
os.environ["CONDA_PREFIX"] = "/opt/miniforge3"

# Verify
!conda --version
'''

"""
=== CELL 2: Create conda environment (SAME AS BEFORE) ===
"""

CELL_2_CONDA_ENV = '''
# Cell 2: Create memvla environment
!conda create -n memvla python=3.10 -y
'''

"""
=== CELL 3: PyTorch installation (SAME AS BEFORE) ===
"""

CELL_3_PYTORCH = '''
# Cell 3: Install PyTorch with CUDA
!conda run -n memvla pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
'''

"""
=== CELL 4: TensorFlow (SAME AS BEFORE) ===
"""

CELL_4_TENSORFLOW = '''
# Cell 4: Install TensorFlow
!conda run -n memvla pip install tensorflow==2.15.0
'''

"""
=== CELL 5: Clone MemoryVLA (SAME AS BEFORE) ===
"""

CELL_5_CLONE = '''
# Cell 5: Clone MemoryVLA
%cd /content
!git clone https://github.com/MemoryVLA/MemoryVLA.git
%cd MemoryVLA
'''

"""
=== CELL 6: Install MemoryVLA dependencies (SAME AS BEFORE) ===
"""

CELL_6_DEPS = '''
# Cell 6: Install MemoryVLA dependencies
%cd /content/MemoryVLA
!conda run -n memvla pip install -e .
!conda run -n memvla pip install imageio imageio-ffmpeg
'''

"""
=== CELL 7: Download Bridge checkpoint (SAME AS BEFORE) ===
Note: This takes time (~31GB). If already downloaded, skip this cell.
"""

CELL_7_CHECKPOINT = '''
# Cell 7: Download Bridge checkpoint
%cd /content/MemoryVLA

# Create directory
!mkdir -p checkpoints/memvla-bridge/checkpoints

# Download from HuggingFace (choose one method)
# Method 1: Using huggingface_hub
!conda run -n memvla pip install huggingface_hub

from huggingface_hub import hf_hub_download
import os

ckpt_path = hf_hub_download(
    repo_id="memory-vla/MemoryVLA-Bridge",
    filename="checkpoints/memvla-bridge.pt",
    local_dir="/content/MemoryVLA/checkpoints/memvla-bridge"
)
print(f"Checkpoint downloaded to: {ckpt_path}")
'''

"""
=== CELL 8: Install SimplerEnv (NEW - REPLACES MANISKILL2) ===

This is the key change: Install SimplerEnv instead of ManiSkill2.
SimplerEnv is built on ManiSkill2 but provides Bridge-specific environments.
"""

CELL_8_SIMPLERENV = '''
# Cell 8: Install SimplerEnv (NEW!)
%cd /content

# Clone SimplerEnv
!git clone https://github.com/simpler-env/SimplerEnv.git
%cd SimplerEnv

# Install SimplerEnv and dependencies
!conda run -n memvla pip install -e .
!conda run -n memvla pip install mani_skill2==0.5.3  # SimplerEnv uses ManiSkill2
!conda run -n memvla pip install transforms3d

# Download required assets for Bridge environments
!conda run -n memvla python -m mani_skill2.utils.download_asset all

# Setup Vulkan for rendering
!apt-get update && apt-get install -y libvulkan1 vulkan-utils

print("SimplerEnv installation complete!")
'''

"""
=== CELL 9: Run VA Evaluation (NEW!) ===

Run the Visual Aggregation evaluation.
VA tests visual generalization without cross-embodiment issues.
"""

CELL_9_EVAL = '''
# Cell 9: Run SimplerEnv-Bridge VA Evaluation

import os
import sys

# Add paths
os.environ["DISPLAY"] = ""
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

sys.path.insert(0, "/content/MemoryVLA")
sys.path.insert(0, "/content/SimplerEnv")

%cd /content/MemoryVLA

# Run evaluation
!conda run -n memvla python evaluation/simpler_env/eval_bridge_va.py \\
    --ckpt checkpoints/memvla-bridge/checkpoints/memvla-bridge.pt \\
    --episodes 3 \\
    --steps 80 \\
    --configs va_scene1
'''

"""
=== ALTERNATIVE: Run original SimplerEnv evaluation script ===

If the VA script has issues, use the original evaluation:
"""

CELL_ALT_EVAL = '''
# Alternative: Use original SimplerEnv evaluation
%cd /content/MemoryVLA

!conda run -n memvla python -m evaluation.simpler_env.simpler_env_inference \\
    --ckpt_path checkpoints/memvla-bridge/checkpoints/memvla-bridge.pt \\
    --robot widowx \\
    --policy_model MemoryVLA \\
    --env_name PutCarrotOnPlateInScene-v0 \\
    --scene_name bridge_table_1_v1 \\
    --rgb_overlay_path None \\
    --max_episode_steps 80
'''

"""
=== SUMMARY OF CHANGES ===

Cells to KEEP (same as before):
- Cell 1: Miniforge
- Cell 2: Conda environment
- Cell 3: PyTorch
- Cell 4: TensorFlow
- Cell 5: Clone MemoryVLA
- Cell 6: Dependencies
- Cell 7: Download checkpoint

Cells to REPLACE:
- Cell 8: ManiSkill2 -> SimplerEnv installation
- Cell 9: ManiSkill2 eval -> SimplerEnv-Bridge VA eval

Key Differences:
1. SimplerEnv provides Bridge-specific environments (WidowX robot)
2. VA testing: Set rgb_overlay_path=None to test visual generalization
3. Same robot, same action space - only visual domain changes
4. This isolates visual generalization from cross-embodiment challenges
"""

if __name__ == "__main__":
    print("SimplerEnv-Bridge VA Setup Guide")
    print("="*50)
    print("\nCopy the cells above to your Colab notebook.")
    print("\nKey changes from ManiSkill2 setup:")
    print("1. Cell 8: Install SimplerEnv instead of ManiSkill2")
    print("2. Cell 9: Run VA evaluation script")
    print("\nVA (Visual Aggregation) tests visual generalization")
    print("by removing rgb_overlay, creating domain gap while")
    print("keeping the same WidowX robot and action space.")
