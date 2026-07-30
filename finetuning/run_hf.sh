#!/usr/bin/env bash
# HuggingFace Trainer-based SFT for Qwen3-TTS (Base).
#
# Fill in the paths below (or export them from your environment) before running.
#
# Usage:
#   bash finetuning/run_hf.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Make the in-repo qwen_tts package importable (mirrors run_grpo.sh).
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NUM_PROCESSES=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")

# ---- Paths (edit before running) ----
init_model_path=${INIT_MODEL_PATH:-"Qwen/Qwen3-TTS-12Hz-1.7B-Base"}
train_data_list=${TRAIN_LIST:-"<PATH_TO_TRAIN_LIST_OR_JSONL>"}
tokenizer_path=${TOKENIZER_PATH:-"Qwen/Qwen3-TTS-Tokenizer-12Hz"}

exp_name=${EXP_NAME:-"sft_hf_lora16_constant_lr2e-6"}
output_model_path=${OUTPUT_MODEL_PATH:-"${REPO_ROOT}/exp/${exp_name}"}
mkdir -p "$output_model_path"

torchrun \
  --nproc_per_node "$NUM_PROCESSES" \
  --master_port 29502 \
  ${SCRIPT_DIR}/sft_base_hf.py \
  --init_model_path $init_model_path \
  --output_model_path $output_model_path \
  --train_list $train_data_list \
  --tokenizer_path $tokenizer_path \
  --batch_size 1 \
  --gradient_accumulation_steps 2 \
  --max_steps -1 \
  --num_epochs 15 \
  --lr 2e-6 \
  --lr_scheduler_type constant \
  --warmup_steps 0 \
  --save_strategy epoch \
  --save_total_limit 100 \
  --logging_steps 10 \
  --sub_talker_loss_weight 0.2 \
  --no-icl_mode \
  --data_split_mode samples \
  --no-flatten \
  --use_lora \
  --lora_rank 16 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  2>&1 | tee $output_model_path/train.log
