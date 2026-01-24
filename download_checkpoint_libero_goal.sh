#!/bin/bash

# LIBERO Goal Checkpoint 다운로드 스크립트
# Usage: bash download_checkpoint_libero_goal.sh

set -e

echo "========================================"
echo "LIBERO Goal Checkpoint 다운로드 시작"
echo "========================================"

# 체크포인트 디렉토리 생성
CHECKPOINT_DIR="/content/checkpoints/libero_goal"
mkdir -p "$CHECKPOINT_DIR"
cd "$CHECKPOINT_DIR"

# 기존 파일 삭제
echo "📂 기존 파일 정리 중..."
rm -rf memoryvla_libero_goal.zip
rm -rf checkpoints/

# HuggingFace에서 체크포인트 다운로드
echo "⬇️  체크포인트 다운로드 중... (약 14GB, 5-10분 소요)"
wget -q --show-progress \
    https://huggingface.co/Hao1Zhang/MemoryVLA/resolve/main/ckpts/memoryvla_libero_goal.zip \
    -O memoryvla_libero_goal.zip

# 압축 해제
echo "📦 압축 해제 중..."
unzip -q memoryvla_libero_goal.zip

# .pt 파일 찾기
PT_FILE=$(find "$CHECKPOINT_DIR/checkpoints" -name "*.pt" | head -1)

if [ -z "$PT_FILE" ]; then
    echo "❌ 오류: .pt 체크포인트 파일을 찾을 수 없습니다."
    exit 1
fi

# 경로 저장
echo "$PT_FILE" > /content/checkpoint_path_goal.txt

echo ""
echo "✅ 체크포인트 다운로드 완료!"
echo "📍 체크포인트 경로: $PT_FILE"
echo "📄 경로 저장 위치: /content/checkpoint_path_goal.txt"
echo ""
echo "========================================"
echo "다음 단계: bash run_libero_goal_5tasks.sh"
echo "========================================"
