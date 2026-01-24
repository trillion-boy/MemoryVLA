#!/bin/bash

# LIBERO Goal 평가 실행 스크립트 (5 tasks × 50 trials)
# Usage: bash run_libero_goal_5tasks.sh

set -e

echo "========================================"
echo "LIBERO Goal 평가 시작 (5 tasks × 50 trials)"
echo "========================================"

# HuggingFace 토큰 설정 (사용자가 미리 설정해야 함)
# Colab에서 먼저 실행: import os; os.environ['HF_TOKEN'] = 'your_token_here'
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  경고: HF_TOKEN 환경변수가 설정되지 않았습니다."
    echo "Colab에서 먼저 실행하세요: import os; os.environ['HF_TOKEN'] = 'your_hf_token'"
fi

# 체크포인트 경로 읽기
CHECKPOINT_PATH=$(cat /content/checkpoint_path_goal.txt)

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ 오류: 체크포인트 파일을 찾을 수 없습니다: $CHECKPOINT_PATH"
    echo "먼저 'bash download_checkpoint_libero_goal.sh'를 실행하세요."
    exit 1
fi

echo "📍 체크포인트: $CHECKPOINT_PATH"
echo "🎯 Task Suite: libero_goal"
echo "📊 평가 설정: Tasks 0-4 (5개), 각 50 trials"
echo "⏱️  예상 소요 시간: 약 2.5-3시간"
echo ""
echo "========================================"

# Conda 환경 활성화 및 평가 실행
cd /content/MemoryVLA

/opt/conda/envs/memvla/bin/python evaluation/libero/eval_libero_colab.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --task_suite_name "libero_goal" \
    --unnorm_key "libero_goal_no_noops" \
    --num_trials_per_task 50 \
    --seed 7 \
    --use_bf16 \
    --action_chunking_window 8 \
    --log_dir "/content/logs/libero_goal_5tasks" \
    --specific_task_ids 0 1 2 3 4

echo ""
echo "========================================"
echo "✅ LIBERO Goal 평가 완료!"
echo "📂 결과 저장 위치: /content/logs/libero_goal_5tasks/"
echo "========================================"
