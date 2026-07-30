#!/usr/bin/env bash
# Custom-voice TTS inference with post-selection (SSIM + CER).
#
# Uses generate_custom_voice(text, speaker) — a built-in speaker id from the
# model's supported speaker set. Fill in the paths below before running.
#
# Usage:
#   bash run_inference_custom_voice_postprocess.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ============================================================================
# Evaluate config
DEVICE=cuda:0
NUM_CANDIDATES=12           # candidates per sentence; single-shot: 1 + USE_POSTPROCESS=false
SSIM_KEEP=6                 # keep top-K by SSIM before CER re-ranking
SSIM_THRESHOLD=0.88
CER_THRESHOLD=0.08
CER_VERBOSE=false
CER_PICK_STRATEGY=medium    # medium | shorter
MAX_RETRIES=0
USE_POSTPROCESS=true        # false: return the 1st candidate directly (no selection)
SSIM_ONLY=false             # true: skip CER, return highest-SSIM candidate

WESPEAKER_PATH=${WESPEAKER_PATH:-"<PATH_TO_WESPEAKER_CHECKPOINT>"}
ASR_MODEL_PATH=${ASR_MODEL_PATH:-"<PATH_TO_FUNASR_PARAFORMER>"}
# ============================================================================

# ---- Model / inputs (edit before running) ----
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"}
TEXT_FILE=${TEXT_FILE:-"<PATH_TO_TEXT_FILE>"}

# Custom voice: speaker id (must be in model's supported speaker list)
SPEAKER=${SPEAKER:-"Vivian"}
INSTRUCT=${INSTRUCT:-""}    # optional: style/emotion instruction, e.g. "用开心的语气说"

# Optional: reference audio for SSIM evaluation only (not used for generation).
# Leave empty to disable SSIM filtering entirely.
REF_AUDIO=${REF_AUDIO:-""}

OUTPUT_DIR=${OUTPUT_DIR:-"./out_wavs/custom_voice_postprocess"}
mkdir -p "$OUTPUT_DIR"

CMD="python ${SCRIPT_DIR}/examples/inference_custom_voice_postprocess.py \
  --model-path $MODEL_PATH \
  --speaker $SPEAKER \
  --wespeaker-path $WESPEAKER_PATH \
  --asr-model-path $ASR_MODEL_PATH \
  --device $DEVICE \
  --num-candidates $NUM_CANDIDATES \
  --ssim-keep $SSIM_KEEP \
  --ssim-threshold $SSIM_THRESHOLD \
  --cer-threshold $CER_THRESHOLD \
  --cer-verbose $CER_VERBOSE \
  --cer-pick-strategy $CER_PICK_STRATEGY \
  --max-retries $MAX_RETRIES \
  --use-postprocess $USE_POSTPROCESS \
  --ssim-only $SSIM_ONLY \
  --output-dir $OUTPUT_DIR"

if [ -n "$REF_AUDIO" ]; then
  CMD="$CMD --ref-audio $REF_AUDIO"
fi

if [ -n "$INSTRUCT" ]; then
  CMD="$CMD --instruct \"$INSTRUCT\""
fi

if [ -n "$TEXT_FILE" ]; then
  CMD="$CMD --text-file $TEXT_FILE"
fi

echo "Running: $CMD"
eval $CMD
