#!/usr/bin/env bash
# Zero-shot TTS inference with post-selection (SSIM + CER).
#
# Fill in MODEL_PATH / REF_AUDIO / REF_TEXT / TEXT_FILE / OUTPUT_DIR
# (or export them from your environment) before running.
#
# Usage:
#   bash run_inference_postprocess.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ============================================================================
# Evaluate config
DEVICE=cuda:0
NUM_CANDIDATES=8            # candidates per sentence; single-shot: 1 + USE_POSTPROCESS=false
SSIM_KEEP=4                 # keep top-K by SSIM before CER re-ranking
SEGMENT_MAX_LENGTH=70
SEGMENT_MIN_LENGTH=35
SSIM_THRESHOLD=0.88
CER_THRESHOLD=0.08
CER_VERBOSE=false
CER_PICK_STRATEGY=medium    # medium | shorter
MAX_RETRIES=0
USE_POSTPROCESS=true        # false: return the 1st candidate directly (no selection)
SSIM_ONLY=false             # true: skip CER, return highest-SSIM candidate
X_VECTOR_ONLY=false         # true: prompt uses only speaker_encoder x-vector (matches SFT with --no-icl_mode)

# WeSpeaker + ASR model paths for post-selection scoring.
WESPEAKER_PATH=${WESPEAKER_PATH:-"<PATH_TO_WESPEAKER_CHECKPOINT>"}
ASR_MODEL_PATH=${ASR_MODEL_PATH:-"<PATH_TO_FUNASR_PARAFORMER>"}
# ============================================================================

# ---- Model / inputs (edit before running) ----
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-TTS-12Hz-1.7B-Base"}
REF_AUDIO=${REF_AUDIO:-"<PATH_TO_REFERENCE_WAV>"}
REF_TEXT=${REF_TEXT:-"<TRANSCRIPT_OF_REFERENCE_WAV>"}
TEXT_FILE=${TEXT_FILE:-"<PATH_TO_TEXT_FILE>"}
OUTPUT_DIR=${OUTPUT_DIR:-"./out_wavs/zero_shot_postprocess"}

mkdir -p "$OUTPUT_DIR"

CMD="python ${SCRIPT_DIR}/examples/inference_zero_shot_postprocess.py \
  --model-path $MODEL_PATH \
  --wespeaker-path $WESPEAKER_PATH \
  --asr-model-path $ASR_MODEL_PATH \
  --ref-audio $REF_AUDIO \
  --ref-text \"$REF_TEXT\" \
  --device $DEVICE \
  --num-candidates $NUM_CANDIDATES \
  --ssim-keep $SSIM_KEEP \
  --segment-max-length $SEGMENT_MAX_LENGTH \
  --segment-min-length $SEGMENT_MIN_LENGTH \
  --ssim-threshold $SSIM_THRESHOLD \
  --cer-threshold $CER_THRESHOLD \
  --cer-verbose $CER_VERBOSE \
  --cer-pick-strategy $CER_PICK_STRATEGY \
  --max-retries $MAX_RETRIES \
  --use-postprocess $USE_POSTPROCESS \
  --ssim-only $SSIM_ONLY \
  --x-vector-only $X_VECTOR_ONLY \
  --text-file $TEXT_FILE \
  --output-dir $OUTPUT_DIR"

echo "Running: $CMD"
eval $CMD
