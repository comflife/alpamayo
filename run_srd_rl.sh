#!/bin/bash
################################################################################
# Self-Reflective Denoising RL (SRD-RL) Training Script
#
# 사용법:
#   bash run_srd_rl.sh [mode]
#
# mode:
#   basic      - 기본 학습 (권장)
#   aggressive - 초공격적 (노이즈 많은 데이터용)
#   conservative - 보수적 (깨끗한 데이터용)
################################################################################

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# GPU 설정
NUM_GPUS=2

# 데이터 경로 (수정 필요!)
DATA_PATH="/home/byounggun/alpamayo/src/alpamayo_r1/alignment/finetune_dataset/finetune_data.jsonl"

# 출력 디렉토리
OUTPUT_BASE="/home/byounggun/alpamayo/outputs"

# 기본 파라미터
BATCH_SIZE=1
GRAD_ACCUM=4
EPOCHS=10
LR=5e-6

# ============================================================================
# Mode Selection
# ============================================================================

MODE="${1:-basic}"

case $MODE in
  basic)
    echo "🚀 Running BASIC mode (balanced GT trust and safety)"
    OUTPUT_DIR="${OUTPUT_BASE}/alpamayo_srd_rl_basic"
    SAFETY_WEIGHT=1.5
    GT_WEIGHT=0.5
    REASONING_WEIGHT=0.3
    RL_WEIGHT=0.5
    NUM_SAMPLES=4
    DANGER_THRESHOLD=0.3
    GT_TRUST_MIN=0.1
    ;;

  aggressive)
    echo "⚡ Running AGGRESSIVE mode (low GT trust, high safety priority)"
    OUTPUT_DIR="${OUTPUT_BASE}/alpamayo_srd_rl_aggressive"
    SAFETY_WEIGHT=2.0
    GT_WEIGHT=0.3
    REASONING_WEIGHT=0.4
    RL_WEIGHT=0.7
    NUM_SAMPLES=6
    DANGER_THRESHOLD=0.4
    GT_TRUST_MIN=0.05
    ;;

  conservative)
    echo "🛡️  Running CONSERVATIVE mode (high GT trust, safety as backup)"
    OUTPUT_DIR="${OUTPUT_BASE}/alpamayo_srd_rl_conservative"
    SAFETY_WEIGHT=0.5
    GT_WEIGHT=1.0
    REASONING_WEIGHT=0.2
    RL_WEIGHT=0.2
    NUM_SAMPLES=4
    DANGER_THRESHOLD=0.2
    GT_TRUST_MIN=0.3
    ;;

  *)
    echo "❌ Unknown mode: $MODE"
    echo "Usage: bash run_srd_rl.sh [basic|aggressive|conservative]"
    exit 1
    ;;
esac

# ============================================================================
# Sanity Checks
# ============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SRD-RL Training Configuration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Mode:               $MODE"
echo "Data Path:          $DATA_PATH"
echo "Output Dir:         $OUTPUT_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Safety Weight:      $SAFETY_WEIGHT"
echo "GT Weight:          $GT_WEIGHT"
echo "Reasoning Weight:   $REASONING_WEIGHT"
echo "RL Weight:          $RL_WEIGHT"
echo "Num Samples:        $NUM_SAMPLES"
echo "Danger Threshold:   $DANGER_THRESHOLD"
echo "GT Trust Min:       $GT_TRUST_MIN"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ ERROR: Data file not found: $DATA_PATH"
    echo "Please update DATA_PATH in this script."
    exit 1
fi

# Check CUDA availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  WARNING: nvidia-smi not found. GPU training may not work."
fi

echo "✅ Sanity checks passed."
echo ""

# ============================================================================
# Training
# ============================================================================

cd /home/byounggun/alpamayo/src

echo "🏋️  Starting training..."
echo ""

torchrun --nproc_per_node=$NUM_GPUS \
    -m alpamayo_r1.alignment.finetune_consistency \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --num_train_epochs $EPOCHS \
    --learning_rate $LR \
    --save_steps 500 \
    --logging_steps 10 \
    --consistency_loss_weight 0.2 \
    --safety_reward_weight $SAFETY_WEIGHT \
    --gt_reward_weight $GT_WEIGHT \
    --reasoning_reward_weight $REASONING_WEIGHT \
    --num_trajectory_samples $NUM_SAMPLES \
    --rl_loss_weight $RL_WEIGHT \
    --danger_keyword_threshold $DANGER_THRESHOLD \
    --gt_trust_min $GT_TRUST_MIN \
    --gt_trust_max 1.0 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --gradient_checkpointing \
    --bf16

EXIT_CODE=$?

# ============================================================================
# Post-Training
# ============================================================================

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ✅ Training Completed Successfully!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Model saved to: $OUTPUT_DIR"
    echo ""
    echo "Next steps:"
    echo "1. Check logs for RL metrics (safety, gt_sim, reward)"
    echo "2. Run inference with the trained model"
    echo "3. Visualize cases where model distrusted GT"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
else
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ❌ Training Failed (Exit code: $EXIT_CODE)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Check error messages above."
    echo "Common issues:"
    echo "  - OOM: Reduce batch size or increase grad accumulation"
    echo "  - CUDA error: Check GPU availability"
    echo "  - Data error: Verify data_path and JSONL format"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
fi

exit $EXIT_CODE
