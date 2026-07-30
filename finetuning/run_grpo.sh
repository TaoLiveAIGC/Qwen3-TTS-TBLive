#!/usr/bin/env bash
# GRPO/GDPO training entry (production defaults).
#
# Reward: GDPO 4-dim (CER + SSIM + CPS + SemiFL) with batch-level z-score
# aggregation, dead-zone and exponential penalty. Sub-talker uses REINFORCE
# mode (ratio=1) with trunk_grad_scale=1.0.
#
# Fill in the paths below before running.
#
# Usage:
#   bash finetuning/run_grpo.sh

set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export TRANSFORMERS_VERBOSITY=warning
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200
export TORCH_NCCL_DUMP_ON_TIMEOUT=0

JOB_NAME="grpo_gdpo_4dim"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Optional: put a local wespeaker install on PYTHONPATH.
# WESPEAKER_LIB=/path/to/wespeaker
# export PYTHONPATH="${REPO_ROOT}:${WESPEAKER_LIB}:${PYTHONPATH}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# ---- Paths (edit to point at your local checkpoints & data) ----
INIT_MODEL_PATH=${INIT_MODEL_PATH:-"<PATH_TO_SFT_CHECKPOINT>"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"Qwen/Qwen3-TTS-Tokenizer-12Hz"}
WESPEAKER_PATH=${WESPEAKER_PATH:-"<PATH_TO_WESPEAKER_CHECKPOINT>"}
ASR_MODEL_PATH=${ASR_MODEL_PATH:-"<PATH_TO_FUNASR_PARAFORMER>"}
TRAIN_LIST=${TRAIN_LIST:-"<PATH_TO_TRAIN_LIST_OR_JSONL>"}

OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/exp_grpo/${JOB_NAME}"}
mkdir -p "$OUTPUT_DIR"

NPROC=${NPROC:-$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)}

torchrun --nproc_per_node=$NPROC --master_port=29510 ${SCRIPT_DIR}/grpo_train.py \
    --init_model_path "$INIT_MODEL_PATH" \
    --output_model_path "$OUTPUT_DIR" \
    --train_list "$TRAIN_LIST" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --wespeaker_path "$WESPEAKER_PATH" \
    --asr_model_path "$ASR_MODEL_PATH" \
    --max_steps 100000 \
    --batch_size 6 \
    --gradient_accumulation_steps 2 \
    --num_generations 6 \
    --num_iterations 1 \
    --beta 0.05 \
    --importance_sampling_level sequence \
    --sub_talker_mode reinforce \
    --grpo_sub_talker_weight 0.3 \
    --sub_importance_sampling_level sequence \
    --sub_talker_trunk_grad_scale 1.0 \
    --w_cer 1.0 \
    --w_sim 1.0 \
    --w_cps 1.0 \
    --w_semi_fl 1.0 \
    --cer_deadzone 0.03 \
    --cer_exp_k 3.0 \
    --cps_deadzone_low 0.05 \
    --cps_deadzone_high 0.10 \
    --max_completion_length 600 \
    --gen_temperature 0.9 \
    --gen_top_k 50 \
    --gen_top_p 0.9 \
    --save_steps 50 \
    --logging_steps 1 \
    --warmup_steps 0 \
    --lr 2e-5 \
    --max_grad_norm 5.0 \
    --dataloader_num_workers 0 \
    --use_lora \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    2>&1 | tee "$OUTPUT_DIR/train.log"

# To resume from a saved LoRA checkpoint, add:
#   --resume_from_checkpoint $OUTPUT_DIR/checkpoint-<step>
#
# To disable ICL rollout (x-vector-only, requires SFT trained with --no-icl_mode), add:
#   --no-rollout_icl_mode
