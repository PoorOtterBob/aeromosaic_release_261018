#!/bin/bash
# 消融实验 v2: 用降水文本(precip_text)作为 C 组件
# 7 个新实验 × 6 区域 × 3 seeds
# full_ABC 和 abl_AB 复用已有结果
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON=/root/work/anaconda3/envs/chatts311/bin/python3
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_balanced_gc_ablation.py"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_dagresidual_ablation.py"
MODEL=/data1/SEA-AD/ZZZ/MJM/ChatTS-main/ChatTS-main/ckpt
BASE=/data1/SEA-AD/ZZZ/MJM/proposal
OUT=$BASE/ablation_v2_20260416

SEEDS=(42 123 456)
REGIONS=(YRD BTH CYB ZHD SDP CJD)

# 固定超参 (和主实验一致)
ENV_W=0.8; ROOT_W=1.5; ALPHA=-3.0; HDIM=256

COMMON_TRAIN="--train_split train --eval_split val --bf16 \
  --root_aware_loss --root_label_extra 3:2.0,4:3.0 --max_root_sample_weight 4.0 \
  --root_region_env_fusion --root_region_exclude_self \
  --num_train_epochs 3 --learning_rate 1e-4 --gradient_accumulation_steps 2 \
  --logging_steps 10 --auto_num_labels \
  --root_weight $ROOT_W --root_region_env_weight $ENV_W"

# ── 数据路径 ──
# precip bundle (含降水文本, C=✓)
declare -A DATA_PRECIP=(
  [YRD]="$BASE/YRD_baseline/yrd_dag_group_bundle_precip"
  [BTH]="$BASE/BTH_baseline/bth_dag_group_bundle_precip"
  [CYB]="$BASE/CYB_baseline/cyb_dag_group_bundle_precip_short"
  [ZHD]="$BASE/ZHD_baseline/zhd_dag_group_bundle_precip"
  [SDP]="$BASE/SDP_baseline/sdp_dag_group_bundle_precip_short"
  [CJD]="$BASE/CJD_baseline/cjd_dag_group_bundle_precip"
)
# normal bundle (无降水文本, C=✗)
declare -A DATA_NORMAL=(
  [YRD]="$BASE/YRD_baseline/yrd_dag_group_bundle_v2_realn"
  [BTH]="$BASE/BTH_baseline/bth_dag_group_bundle_v2_realn"
  [CYB]="$BASE/CYB_baseline/cyb_dag_group_bundle_v2_realn"
  [ZHD]="$BASE/ZHD_baseline/zhd_dag_group_bundle_v2_realn"
  [SDP]="$BASE/SDP_baseline/sdp_dag_group_bundle_v2_realn"
  [CJD]="$BASE/CJD_baseline/cjd_dag_group_bundle_v2_realn"
)

# ── 消融实验配置 ──
# 格式: "data_type|dag_enable|ablation_mask|no_lora"
# A=PM2.5(col0), B=过去天气(col1-10), C=降水文本(prompt), T=时间(col11-12,始终保留)
declare -A ABL_CFG
ABL_CFG[abl_AC]="precip|true|0,11,12|false"
ABL_CFG[abl_BC]="precip|true|1,2,3,4,5,6,7,8,9,10,11,12|false"
ABL_CFG[abl_A]="normal|true|0,11,12|false"
ABL_CFG[abl_B]="normal|true|1,2,3,4,5,6,7,8,9,10,11,12|false"
ABL_CFG[abl_C]="precip|true|11,12|false"
ABL_CFG[no_graph]="precip|false|none|false"
ABL_CFG[no_lora]="precip|true|none|true"

run_ablation_experiment() {
  local ABL_ID=$1
  local GPU=$2
  export CUDA_VISIBLE_DEVICES=$GPU

  IFS='|' read -r DATA_TYPE DAG_ENABLE ABLATION_MASK NO_LORA <<< "${ABL_CFG[$ABL_ID]}"

  echo "[$(date '+%H:%M:%S')] [$ABL_ID] 开始 (GPU $GPU)"
  echo "  data_type=$DATA_TYPE dag=$DAG_ENABLE mask=$ABLATION_MASK no_lora=$NO_LORA"

  for REGION in "${REGIONS[@]}"; do
    local DATA
    if [ "$DATA_TYPE" = "precip" ]; then
      DATA="${DATA_PRECIP[$REGION]}"
    else
      DATA="${DATA_NORMAL[$REGION]}"
    fi

    for SEED in "${SEEDS[@]}"; do
      local SEED_DIR="$OUT/${ABL_ID}/${REGION}_seed${SEED}"
      local ADAPTER="$SEED_DIR/final_adapter"

      # --- 训练 ---
      if [ ! -d "$ADAPTER" ]; then
        mkdir -p "$SEED_DIR"
        echo "[$(date '+%H:%M:%S')] [$ABL_ID/$REGION] seed=$SEED 训练开始"

        local EXTRA_ARGS=""
        if [ "$DAG_ENABLE" = "true" ]; then
          EXTRA_ARGS="$EXTRA_ARGS --dag_residual_enable"
          EXTRA_ARGS="$EXTRA_ARGS --dag_residual_alpha_init $ALPHA --dag_residual_hidden_dim $HDIM"
        fi
        if [ "$ABLATION_MASK" != "none" ]; then
          EXTRA_ARGS="$EXTRA_ARGS --ablation_mask $ABLATION_MASK"
        fi
        if [ "$NO_LORA" = "true" ]; then
          EXTRA_ARGS="$EXTRA_ARGS --no_lora"
        fi

        $PYTHON $TRAIN_SCRIPT \
          --model_name_or_path $MODEL \
          --data_dir "$DATA" \
          --out_dir "$SEED_DIR" \
          $COMMON_TRAIN \
          $EXTRA_ARGS \
          --seed $SEED \
          2>&1 | tee "$SEED_DIR/train.log"
        echo "[$(date '+%H:%M:%S')] [$ABL_ID/$REGION] seed=$SEED 训练完成"
      else
        echo "[$(date '+%H:%M:%S')] [$ABL_ID/$REGION] seed=$SEED 已有 adapter，跳过"
      fi

      # --- 评估 ---
      local EVAL_DIR="$OUT/${ABL_ID}/${REGION}_seed${SEED}_eval_test"
      if [ -d "$ADAPTER" ] && [ ! -f "$EVAL_DIR/test/summary.json" ]; then
        mkdir -p "$EVAL_DIR"
        echo "[$(date '+%H:%M:%S')] [$ABL_ID/$REGION] seed=$SEED 评估开始"

        local EVAL_EXTRA=""
        if [ "$ABLATION_MASK" != "none" ]; then
          EVAL_EXTRA="$EVAL_EXTRA --ablation_mask $ABLATION_MASK"
        fi
        if [ "$NO_LORA" = "true" ]; then
          EVAL_EXTRA="$EVAL_EXTRA --no_lora"
        fi

        $PYTHON $EVAL_SCRIPT \
          --model_name_or_path $MODEL \
          --adapter_dir "$ADAPTER" \
          --data_dir "$DATA" \
          --out_dir "$EVAL_DIR" \
          --eval_splits test --bf16 --trust_remote_code \
          $EVAL_EXTRA \
          2>&1 | tee "$EVAL_DIR/eval.log"
        echo "[$(date '+%H:%M:%S')] [$ABL_ID/$REGION] seed=$SEED 评估完成"
      fi
    done
  done
  echo "[$(date '+%H:%M:%S')] [$ABL_ID] 全部完成"
}

# ============================================================
# 调度: 4 GPU 并行, 7 个实验分 2 批
# Batch 1: abl_AC(GPU1), abl_BC(GPU2), abl_A(GPU3), abl_B(GPU5)
# Batch 2: abl_C(GPU1), no_graph(GPU2), no_lora(GPU3)
# ============================================================
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  mkdir -p "$OUT"

  echo "=========================================="
  echo " 消融实验 v2 (降水文本版)"
  echo " 7 实验 × 6 区域 × 3 seeds"
  echo " $(date)"
  echo " 输出: $OUT"
  echo "=========================================="

  GPU_A=${GPU_A:-1}
  GPU_B=${GPU_B:-2}
  GPU_C=${GPU_C:-3}
  GPU_D=${GPU_D:-5}

  # ── Batch 1: 4 个实验并行 ──
  echo ">>> Batch 1: abl_AC, abl_BC, abl_A, abl_B"
  run_ablation_experiment abl_AC $GPU_A > "$OUT/abl_AC.log" 2>&1 &
  PID1=$!
  run_ablation_experiment abl_BC $GPU_B > "$OUT/abl_BC.log" 2>&1 &
  PID2=$!
  run_ablation_experiment abl_A  $GPU_C > "$OUT/abl_A.log" 2>&1 &
  PID3=$!
  run_ablation_experiment abl_B  $GPU_D > "$OUT/abl_B.log" 2>&1 &
  PID4=$!

  echo "GPU$GPU_A: abl_AC (PID=$PID1)"
  echo "GPU$GPU_B: abl_BC (PID=$PID2)"
  echo "GPU$GPU_C: abl_A  (PID=$PID3)"
  echo "GPU$GPU_D: abl_B  (PID=$PID4)"

  wait $PID1 $PID2 $PID3 $PID4
  echo ">>> Batch 1 完成 $(date)"

  # ── Batch 2: 3 个实验并行 ──
  echo ">>> Batch 2: abl_C, no_graph, no_lora"
  run_ablation_experiment abl_C    $GPU_A > "$OUT/abl_C.log" 2>&1 &
  PID5=$!
  run_ablation_experiment no_graph $GPU_B > "$OUT/no_graph.log" 2>&1 &
  PID6=$!
  run_ablation_experiment no_lora  $GPU_C > "$OUT/no_lora.log" 2>&1 &
  PID7=$!

  echo "GPU$GPU_A: abl_C    (PID=$PID5)"
  echo "GPU$GPU_B: no_graph (PID=$PID6)"
  echo "GPU$GPU_C: no_lora  (PID=$PID7)"

  wait $PID5 $PID6 $PID7
  echo ""
  echo "=========================================="
  echo " 全部消融实验完成 $(date)"
  echo "=========================================="
fi
