#!/usr/bin/env python3
# coding=utf-8
"""
Zero-shot TTS inference with post-processing audio selection.

For each text, generates N=8 candidates in a single batch call,
then applies two-stage selection:
  Stage 1 (SSIM): Keep top 4 by speaker similarity to reference
  Stage 2 (CER):  Pick 1 best by lowest character error rate

Usage:
    python examples/inference_zero_shot_postprocess.py
"""

import os
import sys
import time
import random
import argparse

import numpy as np
import torch
import soundfile as sf

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

# Optional: if you have a local wespeaker checkout, expose it via env var.
WESPEAKER_LIB = os.environ.get("WESPEAKER_LIB", "")
if WESPEAKER_LIB:
    sys.path.insert(0, WESPEAKER_LIB)

from qwen_tts import Qwen3TTSModel
from evaluation import (
    SpeakerSimilaritySelector, SSIMConfig,
    CERSelector, CERConfig,
    auto_cut_llm,
)


# ============================================================================
# Config
# ============================================================================

DEFAULT_CONFIG = dict(
    model_path="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    wespeaker_path="",  # set via --wespeaker-path
    asr_model_path="",  # set via --asr-model-path
    ref_audio="",       # set via --ref-audio
    ref_text="",        # set via --ref-text
    device="cuda:0",
    num_candidates=8,       # total candidates per sentence (batch size)
    ssim_keep=4,            # keep top-K after SSIM
    sample_rate=24000,
)

GEN_KWARGS = dict(
    max_new_tokens=2048,
    do_sample=True,
    top_k=20,
    top_p=0.8,
    temperature=0.9,
    repetition_penalty=1.05,
    subtalker_dosample=True,
    subtalker_top_k=20,
    subtalker_top_p=0.8,
    subtalker_temperature=0.9,
)

# GEN_KWARGS = dict(
#     max_new_tokens=2048,
#     do_sample=True,
#     top_k=50,
#     top_p=1,
#     temperature=0.9,
#     repetition_penalty=1.05,
#     subtalker_dosample=True,
#     subtalker_top_k=50,
#     subtalker_top_p=1,
#     subtalker_temperature=0.9,
# )

# ============================================================================
# Core logic
# ============================================================================

class TTSWithPostSelection:
    """TTS inference with batch generation and two-stage audio selection.

    Directly uses SpeakerSimilaritySelector and CERSelector for single-sentence
    selection, so num_candidates and ssim_keep can be freely configured without
    integer divisibility constraints.
    """

    def __init__(
        self,
        model_path: str,
        wespeaker_path: str,
        asr_model_path: str,
        ref_audio: str,
        ref_text: str,
        device: str = "cuda:0",
        num_candidates: int = 8,
        ssim_keep: int = 4,
        sample_rate: int = 24000,
        cer_threshold: float = 0.08,
        cer_verbose: bool = False,
        cer_pick_strategy: str = "medium",
        ssim_only: bool = False,
        x_vector_only: bool = False,
        segment_max_length: int = 70,
        segment_min_length: int = 35,
        adapter_path: str = None,
    ):
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.device = device
        self.num_candidates = num_candidates
        self.ssim_keep = ssim_keep
        self.sample_rate = sample_rate
        self.ssim_only = ssim_only
        self.x_vector_only = x_vector_only
        self.segment_max_length = segment_max_length
        self.segment_min_length = segment_min_length

        # Load TTS model
        print("[Init] Loading TTS model...")
        t0 = time.time()
        self.tts = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        # Load LoRA adapter if specified
        if adapter_path:
            print(f"[Init] Loading LoRA adapter from {adapter_path}...")
            from peft import PeftModel
            self.tts.model.talker = PeftModel.from_pretrained(
                self.tts.model.talker,
                adapter_path,
                is_trainable=False
            )
            print(f"[Init] LoRA adapter loaded from {adapter_path}")

        # Pre-compute voice clone prompt once (avoids re-encoding ref audio every call).
        # x_vector_only=True: skip ref_text / ref_codes ICL prefix and only use the
        # speaker_encoder x-vector — matches sft training with --no-icl_mode.
        self.voice_clone_prompt = self.tts.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only,
        )
        print(f"[Init] TTS model loaded in {time.time() - t0:.1f}s")

        # Load SSIM selector (WeSpeaker)
        print("[Init] Loading SSIM selector (WeSpeaker)...")
        t0 = time.time()
        self.ssim_selector = SpeakerSimilaritySelector(
            SSIMConfig(model_path=wespeaker_path, device="cpu")
        )
        self.ref_embedding = self.ssim_selector.extract_reference_embedding(ref_audio)
        print(f"[Init] SSIM selector loaded in {time.time() - t0:.1f}s")

        # Load CER selector (FunASR) — skipped in ssim_only mode to save init time + GPU memory
        if ssim_only:
            self.cer_selector = None
            print("[Init] CER selector skipped (ssim_only mode).")
        else:
            print("[Init] Loading CER selector (FunASR)...")
            t0 = time.time()
            self.cer_selector = CERSelector(
                CERConfig(
                    asr_model_path=asr_model_path,
                    device=device,
                    batch_size=20,
                    source_sample_rate=sample_rate,
                    language="zh",
                    cer_threshold=cer_threshold,
                    verbose=cer_verbose,
                    pick_strategy=cer_pick_strategy,
                )
            )
            print(f"[Init] CER selector loaded in {time.time() - t0:.1f}s")

    @torch.no_grad()
    def generate_batch(self, text: str, language: str = "Chinese") -> list:
        """Generate num_candidates audio samples for a (possibly long) text.

        Pipeline:
          1. Split text into N segments via auto_cut_llm.
          2. Build a flat batch [seg0]*G + [seg1]*G + ... + [seg(N-1)]*G
             and run a single batched generate call (batch_size = N * G).
          3. Reshape outputs to [N, G] and concatenate across N per candidate,
             yielding G full-length wavs.

        Single-segment input degrades to one batched call of size G.

        Args:
            text: Text to synthesize (can be long).
            language: Language of the text.

        Returns:
            List of G torch.Tensor audio candidates, each spanning the full text.
        """
        segments = auto_cut_llm(
            text,
            max_length=self.segment_max_length,
            min_length=self.segment_min_length,
            return_type="list",
        )
        if not segments:
            segments = [text]

        if len(segments) > 1:
            print(f"  [Split] {len(segments)} segments:")
            for i, seg in enumerate(segments):
                print(f"    [{i}] {seg}")

        N = len(segments)
        G = self.num_candidates

        # Flat batch: [seg0, seg0, ..., seg1, seg1, ..., segN-1, segN-1]
        flat_texts = []
        for seg in segments:
            flat_texts.extend([seg] * G)
        flat_languages = [language] * (N * G)

        wavs, _sr = self.tts.generate_voice_clone(
            text=flat_texts,
            language=flat_languages,
            voice_clone_prompt=self.voice_clone_prompt,
            **GEN_KWARGS,
        )
        flat_tensors = [torch.from_numpy(w).float() for w in wavs]

        # Concatenate each candidate's segment outputs end-to-end:
        # candidate_i = concat(wavs[0*G + i], wavs[1*G + i], ..., wavs[(N-1)*G + i])
        candidates = []
        for cand_idx in range(G):
            chunks = [flat_tensors[seg_idx * G + cand_idx] for seg_idx in range(N)]
            candidates.append(torch.cat(chunks, dim=0))
        return candidates

    def select_best(self, candidates: list, top_entries: list,
                    ssim_scored: list, text: str) -> tuple:
        """Apply CER selection on SSIM-filtered candidates.

        Args:
            candidates: List of torch.Tensor audio candidates (already SSIM-qualified).
            top_entries: Parallel list of (ssim_score, global_idx) for `candidates`.
            ssim_scored: All (ssim_score, global_idx) tuples across the batch (incl. retries).
            text: Ground truth text.

        Returns:
            (best_wav, all_ssim, best_cer, best_ssim, cer_fallback)
              best_ssim:    SSIM of the chosen candidate (NaN if unavailable)
              cer_fallback: True if CER stage hit the min-CER fallback path
        """
        all_ssim = [s for s, _ in ssim_scored]

        # Stage 2: CER — pick best from filtered candidates
        filtered_texts = [text] * len(candidates)
        cer_result = self.cer_selector.select(
            candidates=candidates,
            texts=filtered_texts,
            repeat_count=len(candidates),
            sample_rate=self.sample_rate,
        )

        best_wav = cer_result.selected_wavs[0]
        best_cer = cer_result.cer_scores[0]
        # cer_result.selected_indices[0] is the position inside `candidates`
        best_local = cer_result.selected_indices[0]
        best_ssim = float(top_entries[best_local][0]) if best_local < len(top_entries) else float("nan")
        cer_fallback = cer_result.fallback_count > 0
        return best_wav, all_ssim, best_cer, best_ssim, cer_fallback

    def synthesize(self, text: str, language: str = "Chinese",
                   max_retries: int = 8, ssim_threshold: float = 0.88,
                   use_postprocess: bool = True) -> tuple:
        """Three modes, dispatched by (use_postprocess, self.ssim_only):

          1. use_postprocess=False
                → single batch, return the 1st candidate (no selection).
          2. use_postprocess=True, ssim_only=False  (default full pipeline)
                → batch + SSIM-resample → SSIM top-K → CER pick best.
          3. use_postprocess=True, ssim_only=True   (new)
                → batch + SSIM-resample → argmax(SSIM) across all candidates.
                  CER stage is skipped (and CER selector isn't even loaded).

        The resample loop's exit criterion is the same in modes 2 and 3:
        stop when `qualified ≥ ssim_keep` or after `max_retries` attempts.

        Args:
            text: Text to synthesize.
            language: Language.
            max_retries: Maximum additional generation attempts.
            ssim_threshold: Minimum SSIM score for a candidate to be qualified.
            use_postprocess: Whether to run the SSIM+CER selection pipeline.

        Returns:
            (best_wav_numpy: np.ndarray, sample_rate: int, info: dict)
        """
        t0 = time.time()

        # Fast path: single inference without post-selection.
        # We still compute SSIM (always) and CER (if cer_selector loaded) for *reporting*,
        # but don't filter / retry — the wav returned is always the first generated one.
        if not use_postprocess:
            candidates = self.generate_batch(text, language)
            wav = candidates[0]
            gen_time = time.time() - t0

            t1 = time.time()
            ssim_score = float(self.ssim_selector.compute_similarity(
                wav, self.ref_embedding, sample_rate=self.sample_rate))
            cer_score = float("nan")
            if self.cer_selector is not None:
                # Direct transcribe + compute_cer to skip select()'s [CER][sX] log line
                # (we're just reporting one sample, not picking among many).
                from evaluation.audio_utils import to_numpy_16k
                audios_16k = to_numpy_16k([wav], source_sr=self.sample_rate)
                asr_texts = self.cer_selector.transcribe(audios_16k)
                cer_score = self.cer_selector.compute_cer(asr_texts, [text])[0]
            sel_time = time.time() - t1

            return wav.numpy(), self.sample_rate, {
                "gen_time": gen_time,
                "sel_time": sel_time,
                "num_candidates": 1,
                "ssim_scores": [ssim_score],
                "best_cer": cer_score,
                "best_ssim": ssim_score,
                "retries": 0,
                "ssim_qualified": int(ssim_score >= ssim_threshold),
                "ssim_fallback": False,
                "cer_fallback": False,
            }

        # Post-selection pipeline
        # Collect all candidates and their SSIM scores across retries
        all_candidates = []
        all_ssim_scored = []  # [(score, global_idx), ...]
        total_generated = 0
        attempts_used = 0

        for attempt in range(1 + max_retries):
            attempts_used = attempt
            # Generate a batch
            new_candidates = self.generate_batch(text, language)
            base_idx = len(all_candidates)
            all_candidates.extend(new_candidates)
            total_generated += len(new_candidates)

            # Score new candidates
            for i, wav in enumerate(new_candidates):
                score = self.ssim_selector.compute_similarity(
                    wav, self.ref_embedding, sample_rate=self.sample_rate)
                all_ssim_scored.append((score, base_idx + i))

            # Check how many meet the threshold
            qualified = [(s, idx) for s, idx in all_ssim_scored if s >= ssim_threshold]

            if len(qualified) >= self.ssim_keep:
                # Enough qualified candidates
                if attempt > 0:
                    print(f"    [Resample] Got {len(qualified)} qualified candidates "
                          f"after {attempt + 1} attempts ({total_generated} total generated)")
                break
            else:
                if attempt < max_retries:
                    print(f"    [Resample] Only {len(qualified)}/{self.ssim_keep} candidates "
                          f"with SSIM>={ssim_threshold}, regenerating... "
                          f"(attempt {attempt + 2}/{1 + max_retries})")

        gen_time = time.time() - t0

        # ssim_only path: skip CER, just return the argmax-SSIM candidate.
        # Reuses the same resample loop above (same exit criterion: ssim_keep qualified),
        # so latency/quality stay comparable — we just swap the final selector.
        if self.ssim_only:
            best_score, best_idx = max(all_ssim_scored, key=lambda x: x[0])
            scores_desc = sorted((s for s, _ in all_ssim_scored), reverse=True)
            n_qual = sum(1 for s in scores_desc if s >= ssim_threshold)
            best_dur = all_candidates[best_idx].numel() / self.sample_rate
            delta_str = (f" (Δ2nd={best_score - scores_desc[1]:+.3f})"
                         if len(scores_desc) > 1 else "")
            print(f"[SSIM] qual={n_qual}/{len(scores_desc)} "
                  f"range={scores_desc[-1]:.3f}-{scores_desc[0]:.3f} "
                  f"pick idx={best_idx} SSIM={best_score:.3f}{delta_str} "
                  f"dur={best_dur:.2f}s")
            return all_candidates[best_idx].numpy(), self.sample_rate, {
                "gen_time": gen_time,
                "sel_time": 0.0,
                "num_candidates": len(all_candidates),
                "ssim_scores": [s for s, _ in all_ssim_scored],
                "best_cer": float("nan"),
                "best_ssim": float(best_score),
                "retries": attempts_used,
                "ssim_qualified": n_qual,
                "ssim_fallback": n_qual == 0,
                "cer_fallback": False,
            }

        # Stage 1: Select top-K by SSIM from all collected candidates
        t1 = time.time()

        # Prefer qualified candidates (>= threshold), sorted by score desc
        qualified = [(s, idx) for s, idx in all_ssim_scored if s >= ssim_threshold]
        qualified.sort(key=lambda x: x[0], reverse=True)
        ssim_qualified_n = len(qualified)
        ssim_fallback = False

        if len(qualified) >= self.ssim_keep:
            top_entries = qualified[:self.ssim_keep]
        elif len(qualified) > 0:
            # Not enough even after retries — only send qualified ones to CER
            top_entries = qualified
            ssim_fallback = True
            print(f"    [Resample] WARNING: Only {len(qualified)} candidates met "
                  f"SSIM>={ssim_threshold} after all retries, sending all qualified to CER")
        else:
            # No candidates met threshold — fallback to top ssim_keep by score
            all_ssim_scored_sorted = sorted(all_ssim_scored, key=lambda x: x[0], reverse=True)
            top_entries = all_ssim_scored_sorted[:self.ssim_keep]
            ssim_fallback = True
            print(f"    [Resample] WARNING: No candidates met SSIM>={ssim_threshold}, "
                  f"fallback to top-{len(top_entries)} by score")

        filtered_candidates = [all_candidates[idx] for _, idx in top_entries]

        # Stage 2: CER selection
        best_wav, all_ssim, best_cer, best_ssim, cer_fallback = self.select_best(
            filtered_candidates, top_entries, all_ssim_scored, text)
        sel_time = time.time() - t1

        info = {
            "gen_time": gen_time,
            "sel_time": sel_time,
            "num_candidates": len(all_candidates),
            "ssim_scores": all_ssim,
            "best_cer": best_cer,
            "best_ssim": best_ssim,
            "retries": attempts_used,
            "ssim_qualified": ssim_qualified_n,
            "ssim_fallback": ssim_fallback,
            "cer_fallback": cer_fallback,
        }
        return best_wav.numpy(), self.sample_rate, info


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot TTS with post-selection")
    parser.add_argument("--model-path", type=str, default=DEFAULT_CONFIG["model_path"])
    parser.add_argument("--wespeaker-path", type=str, default=DEFAULT_CONFIG["wespeaker_path"])
    parser.add_argument("--asr-model-path", type=str, default=DEFAULT_CONFIG["asr_model_path"])
    parser.add_argument("--ref-audio", type=str, default=DEFAULT_CONFIG["ref_audio"])
    parser.add_argument("--ref-text", type=str, default=DEFAULT_CONFIG["ref_text"])
    parser.add_argument("--device", type=str, default=DEFAULT_CONFIG["device"])
    parser.add_argument("--num-candidates", type=int, default=DEFAULT_CONFIG["num_candidates"],
                        help="Number of candidates to generate per sentence (batch size)")
    parser.add_argument("--ssim-keep", type=int, default=DEFAULT_CONFIG["ssim_keep"],
                        help="Number of candidates to keep after SSIM filtering")
    parser.add_argument("--segment-max-length", type=int, default=160,
                        help="Max word count per segment when auto-splitting long text (default: 70)")
    parser.add_argument("--segment-min-length", type=int, default=80,
                        help="Min word count per segment when auto-splitting long text (default: 35)")
    parser.add_argument("--ssim-threshold", type=float, default=0.88,
                        help="Minimum SSIM score for a candidate to be qualified (default: 0.88)")
    parser.add_argument("--use-postprocess", type=str, default="true",
                        help="Enable post-selection pipeline (true/false, default: true)")
    parser.add_argument("--ssim-only", type=str, default="false",
                        help="If true, skip CER and return the argmax-SSIM candidate across "
                             "all retries. Implies --use-postprocess true. Skips CER selector "
                             "init to save ~18s + GPU memory. (true/false, default: false)")
    parser.add_argument("--x-vector-only", type=str, default="false",
                        help="If true, build the voice-clone prompt with x_vector_only_mode=True: "
                             "only the speaker_encoder x-vector is used and ref_text/ref_codes "
                             "are NOT prefixed (no ICL). Matches sft training with --no-icl_mode. "
                             "(true/false, default: false)")
    parser.add_argument("--cer-threshold", type=float, default=0.08,
                        help="CER threshold below which a candidate is considered qualified (default: 0.08)")
    parser.add_argument("--cer-verbose", type=str, default="false",
                        help="Print per-candidate CER/ASR detail lines (true/false, default: false)")
    parser.add_argument("--max-retries", type=int, default=8,
                        help="Maximum number of additional batch generations when SSIM quota is unmet (default: 8)")
    parser.add_argument("--cer-pick-strategy", type=str, default="medium", choices=["medium", "shorter"],
                        help="Among CER-qualified candidates, pick the one with 'medium' (median) or 'shorter' (shortest) duration (default: medium)")
    parser.add_argument("--text-file", type=str, default=None,
                        help="Text file with one sentence per line")
    parser.add_argument("--texts", nargs="+", default=None,
                        help="Inline texts to synthesize")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(REPO_ROOT, "out_wavs", "zero_shot_postprocess"))
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Path to LoRA adapter checkpoint (optional). If specified, loads LoRA adapter on top of base model.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Determine text list
    if args.text_file:
        with open(args.text_file, "r") as f:
            text_list = [line.strip() for line in f if line.strip()]
    elif args.texts:
        text_list = args.texts
    else:
        # Default test texts
        text_list = [
            "今天天气真不错，适合出去走走。",
            "我们这款产品采用了全新的设计理念，让用户体验更加流畅。",
            "三月份的销售数据显示，整体业绩同比增长了百分之十五。",
            "请大家注意安全，不要在走廊里奔跑。",
            "这本书讲述了一个关于勇气和坚持的故事。",
            "下一站是人民广场，请需要下车的乘客提前做好准备。",
            "科技的发展让我们的生活变得越来越便利。",
            "今晚的晚餐我想吃火锅，你觉得怎么样？",
            "经过反复测试，这个方案终于通过了所有验证。",
            "春天来了，公园里的花都开了，特别漂亮。",
        ]

    # Clean texts
    text_list = [t.replace("@", "").replace("→", "").replace("¿", "") for t in text_list]

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  Zero-Shot TTS with Post-Selection")
    print("=" * 70)
    print(f"  Model:           {args.model_path}")
    print(f"  Ref audio:       {args.ref_audio}")
    print(f"  Texts:           {len(text_list)} sentences")
    use_postprocess = args.use_postprocess.lower() in ("true", "1", "yes")
    cer_verbose = args.cer_verbose.lower() in ("true", "1", "yes")
    ssim_only = args.ssim_only.lower() in ("true", "1", "yes")
    x_vector_only = args.x_vector_only.lower() in ("true", "1", "yes")
    # ssim_only implies post-selection (otherwise there's no resample loop to argmax over)
    if ssim_only and not use_postprocess:
        use_postprocess = True
        print("  [Note] --ssim-only=true forces --use-postprocess=true")

    print(f"  Candidates/sent: {args.num_candidates} (batch)")
    print(f"  SSIM keep:       {args.ssim_keep}")
    print(f"  SSIM threshold:  {args.ssim_threshold}")
    print(f"  CER threshold:   {args.cer_threshold}")
    print(f"  CER pick:        {args.cer_pick_strategy}")
    print(f"  Max retries:     {args.max_retries}")
    print(f"  Post-selection:  {use_postprocess}")
    print(f"  SSIM-only mode:  {ssim_only}  {'(skip CER, return argmax SSIM)' if ssim_only else ''}")
    print(f"  X-vector-only:   {x_vector_only}  {'(prompt uses speaker_encoder only, no ref_text/ref_codes)' if x_vector_only else '(ICL prompt: ref_text + ref_codes prefixed)'}")
    print(f"  CER verbose:     {cer_verbose}")
    print(f"  CER select:      {'(disabled)' if ssim_only else '1 (best)'}")
    print(f"  Seed:            {args.seed}")
    print(f"  Segment length:  [{args.segment_min_length}, {args.segment_max_length}] words")
    print(f"  Output:          {args.output_dir}")
    print("-" * 70)

    # Initialize
    engine = TTSWithPostSelection(
        model_path=args.model_path,
        wespeaker_path=args.wespeaker_path,
        asr_model_path=args.asr_model_path,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        device=args.device,
        num_candidates=args.num_candidates,
        ssim_keep=args.ssim_keep,
        cer_threshold=args.cer_threshold,
        cer_verbose=cer_verbose,
        cer_pick_strategy=args.cer_pick_strategy,
        ssim_only=ssim_only,
        x_vector_only=x_vector_only,
        segment_max_length=args.segment_max_length,
        segment_min_length=args.segment_min_length,
        adapter_path=args.adapter_path,
    )

    # Process each text
    total_gen_time = 0.0
    total_sel_time = 0.0
    all_cer = []
    all_ssim = []
    resample_count = 0
    ssim_fallback_count = 0
    cer_fallback_count = 0
    retry_dist = {}  # retries -> count
    n = len(text_list)

    for idx, text in enumerate(text_list):
        wav, sr, info = engine.synthesize(text, use_postprocess=use_postprocess,
                                          ssim_threshold=args.ssim_threshold,
                                          max_retries=args.max_retries)

        total_gen_time += info["gen_time"]
        total_sel_time += info["sel_time"]
        if not np.isnan(info["best_cer"]):
            all_cer.append(info["best_cer"])
        if not np.isnan(info["best_ssim"]):
            all_ssim.append(info["best_ssim"])
        if info["retries"] > 0:
            resample_count += 1
        if info["ssim_fallback"]:
            ssim_fallback_count += 1
        if info["cer_fallback"]:
            cer_fallback_count += 1
        retry_dist[info["retries"]] = retry_dist.get(info["retries"], 0) + 1

        # Save
        out_path = os.path.join(args.output_dir, f"{idx}.wav")
        sf.write(out_path, wav, sr)

        text_short = text[:30] + ".." if len(text) > 32 else text
        ssim_str = f"{info['best_ssim']:.3f}" if not np.isnan(info["best_ssim"]) else "  -  "
        cer_str = f"{info['best_cer']:.4f}" if not np.isnan(info["best_cer"]) else "  -  "
        flags = ""
        if info["ssim_fallback"]:
            flags += " [ssim-fb]"
        if info["cer_fallback"]:
            flags += " [cer-fb]"
        print(f"  [{idx+1}/{n}] gen={info['gen_time']:.1f}s sel={info['sel_time']:.1f}s "
              f"SSIM={ssim_str} CER={cer_str} cand={info['num_candidates']} "
              f"retry={info['retries']} qual={info['ssim_qualified']}{flags}  {text_short}")

    # Summary
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  Total sentences:     {n}")
    print(f"  Total gen time:      {total_gen_time:.1f}s (avg {total_gen_time/n:.1f}s/sent)")
    print(f"  Total select time:   {total_sel_time:.1f}s (avg {total_sel_time/n:.1f}s/sent)")
    if all_ssim:
        print(f"  Avg SSIM (selected): {np.mean(all_ssim):.4f}  "
              f"(min={np.min(all_ssim):.4f}, max={np.max(all_ssim):.4f})")
        print(f"  Min SSIM: {np.min(all_ssim):.4f}")
        print(f"  Max SSIM: {np.max(all_ssim):.4f}")
    if all_cer:
        print(f"  Avg CER  (selected): {np.mean(all_cer):.4f}  "
              f"(min={np.min(all_cer):.4f}, max={np.max(all_cer):.4f})")
        print(f"  Min CER: {np.min(all_cer):.4f}")
        print(f"  Max CER: {np.max(all_cer):.4f}")
    if use_postprocess:
        print(f"  Resample rate:       {resample_count}/{n} ({100*resample_count/n:.0f}%)")
        if ssim_only:
            print(f"  SSIM fallback:       {ssim_fallback_count}/{n} "
                  f"(no candidate >= {args.ssim_threshold} after retries; argmax used anyway)")
        else:
            print(f"  SSIM fallback:       {ssim_fallback_count}/{n} "
                  f"(< {args.ssim_keep} candidates >= {args.ssim_threshold} after retries)")
            print(f"  CER  fallback:       {cer_fallback_count}/{n} "
                  f"(no candidate < {args.cer_threshold} CER, used min-CER)")
        retry_str = ", ".join(f"{k}={retry_dist[k]}" for k in sorted(retry_dist))
        print(f"  Retry distribution:  {retry_str}")
    print(f"  Output:              {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
