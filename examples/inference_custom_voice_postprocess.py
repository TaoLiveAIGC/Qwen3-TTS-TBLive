# coding=utf-8
"""
Custom-voice TTS inference with post-processing audio selection (SSIM + CER).

Uses generate_custom_voice() API with a built-in speaker id:
  - No ref_audio / ref_text needed for generation
  - speaker (spk_id) selects the voice from the model's predefined speaker set
  - ref_audio is optional, only used as a comparison target for SSIM evaluation

Post-selection pipeline (same as inference_zero_shot_postprocess.py):
  Stage 1 (SSIM): Keep top-K by speaker similarity to ref_audio (if provided)
  Stage 2 (CER):  Pick 1 best by lowest character error rate

Usage:
    python examples/inference_custom_voice_postprocess.py \
        --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
        --speaker Vivian \
        --text-file <path> --output-dir <path>
"""

import os
import random
import sys
import time
import argparse

import numpy as np
import torch
import soundfile as sf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

WESPEAKER_LIB = os.environ.get("WESPEAKER_LIB", "")
if WESPEAKER_LIB:
    sys.path.insert(0, WESPEAKER_LIB)

from qwen_tts import Qwen3TTSModel
from evaluation import (
    SpeakerSimilaritySelector, SSIMConfig,
    CERSelector, CERConfig,
)


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Generation kwargs (matches test_model_12hz_base.py)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class CustomVoiceTTSWithPostSelection:
    """Custom-voice TTS inference with batch generation and two-stage selection.

    Unlike zero-shot voice cloning, this uses generate_custom_voice() with a
    built-in speaker id. No ref_audio/ref_text needed for generation.
    """

    def __init__(
        self,
        model_path: str,
        wespeaker_path: str,
        asr_model_path: str,
        speaker: str,
        ref_audio: str | None = None,
        device: str = "cuda:0",
        num_candidates: int = 8,
        ssim_keep: int = 4,
        sample_rate: int = 24000,
        cer_threshold: float = 0.08,
        cer_verbose: bool = False,
        cer_pick_strategy: str = "medium",
        ssim_only: bool = False,
    ):
        self.speaker = speaker
        self.ref_audio = ref_audio
        self.device = device
        self.num_candidates = num_candidates
        self.ssim_keep = ssim_keep
        self.sample_rate = sample_rate
        self.ssim_only = ssim_only
        self.has_ssim_ref = ref_audio is not None

        # --- TTS model ---
        print("[Init] Loading TTS model...")
        t0 = time.time()
        self.tts = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        supported = self.tts.get_supported_speakers()
        if supported is not None:
            print(f"[Init] Supported speakers: {supported}")
        print(f"[Init] TTS model loaded in {time.time() - t0:.1f}s (speaker={speaker})")

        # --- SSIM selector (WeSpeaker) — only loaded when ref_audio is provided ---
        if self.has_ssim_ref:
            print("[Init] Loading SSIM selector (WeSpeaker)...")
            t0 = time.time()
            self.ssim_selector = SpeakerSimilaritySelector(
                SSIMConfig(model_path=wespeaker_path, device="cpu")
            )
            self.ref_embedding = self.ssim_selector.extract_reference_embedding(ref_audio)
            print(f"[Init] SSIM selector loaded in {time.time() - t0:.1f}s")
        else:
            self.ssim_selector = None
            self.ref_embedding = None
            print("[Init] SSIM selector skipped (no ref_audio provided).")

        # --- CER selector (FunASR) — skipped in ssim_only mode ---
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
    def generate_batch(self, text: str, language: str = "Chinese",
                       instruct: str | None = None) -> list:
        """Generate num_candidates audio samples via generate_custom_voice."""
        texts = [text] * self.num_candidates
        languages = [language] * self.num_candidates
        speakers = [self.speaker] * self.num_candidates
        instructs = [instruct or ""] * self.num_candidates
        wavs, sr = self.tts.generate_custom_voice(
            text=texts,
            speaker=speakers,
            language=languages,
            instruct=instructs,
            **GEN_KWARGS,
        )
        return [torch.from_numpy(w).float() for w in wavs]

    def _compute_ssim(self, wav) -> float:
        if self.ssim_selector is None:
            return float("nan")
        return float(self.ssim_selector.compute_similarity(
            wav, self.ref_embedding, sample_rate=self.sample_rate))

    def _compute_cer(self, wav, text: str) -> float:
        if self.cer_selector is None:
            return float("nan")
        from evaluation.audio_utils import to_numpy_16k
        audios_16k = to_numpy_16k([wav], source_sr=self.sample_rate)
        asr_texts = self.cer_selector.transcribe(audios_16k)
        return self.cer_selector.compute_cer(asr_texts, [text])[0]

    def select_best(self, candidates: list, top_entries: list,
                    ssim_scored: list, text: str) -> tuple:
        """Apply CER selection on SSIM-filtered candidates."""
        all_ssim = [s for s, _ in ssim_scored]
        filtered_texts = [text] * len(candidates)
        cer_result = self.cer_selector.select(
            candidates=candidates,
            texts=filtered_texts,
            repeat_count=len(candidates),
            sample_rate=self.sample_rate,
        )
        best_wav = cer_result.selected_wavs[0]
        best_cer = cer_result.cer_scores[0]
        best_local = cer_result.selected_indices[0]
        best_ssim = float(top_entries[best_local][0]) if best_local < len(top_entries) else float("nan")
        cer_fallback = cer_result.fallback_count > 0
        return best_wav, all_ssim, best_cer, best_ssim, cer_fallback

    def select_best_cer_only(self, candidates: list, text: str) -> tuple:
        """CER-only selection (no SSIM pre-filter)."""
        texts = [text] * len(candidates)
        cer_result = self.cer_selector.select(
            candidates=candidates,
            texts=texts,
            repeat_count=len(candidates),
            sample_rate=self.sample_rate,
        )
        best_wav = cer_result.selected_wavs[0]
        best_cer = cer_result.cer_scores[0]
        cer_fallback = cer_result.fallback_count > 0
        return best_wav, best_cer, cer_fallback

    def synthesize(self, text: str, language: str = "Chinese",
                   instruct: str | None = None,
                   max_retries: int = 8, ssim_threshold: float = 0.88,
                   use_postprocess: bool = True) -> tuple:
        """Dispatched by (use_postprocess, ssim_only, has_ssim_ref).

        When has_ssim_ref=False: SSIM stage is skipped entirely, selection is
        CER-only (or random if both SSIM and CER are unavailable).
        """
        t0 = time.time()

        # --- Fast path: no post-selection ---
        if not use_postprocess:
            candidates = self.generate_batch(text, language, instruct)
            wav = candidates[0]
            gen_time = time.time() - t0

            t1 = time.time()
            ssim_score = self._compute_ssim(wav)
            cer_score = self._compute_cer(wav, text)
            sel_time = time.time() - t1

            return wav.numpy(), self.sample_rate, {
                "gen_time": gen_time, "sel_time": sel_time,
                "num_candidates": 1,
                "ssim_scores": [ssim_score],
                "best_cer": cer_score, "best_ssim": ssim_score,
                "retries": 0, "ssim_qualified": 0,
                "ssim_fallback": False, "cer_fallback": False,
            }

        # --- Post-selection pipeline ---
        all_candidates = []
        all_ssim_scored = []
        total_generated = 0
        attempts_used = 0

        for attempt in range(1 + max_retries):
            attempts_used = attempt
            new_candidates = self.generate_batch(text, language, instruct)
            base_idx = len(all_candidates)
            all_candidates.extend(new_candidates)
            total_generated += len(new_candidates)

            if self.has_ssim_ref:
                for i, wav in enumerate(new_candidates):
                    score = self._compute_ssim(wav)
                    all_ssim_scored.append((score, base_idx + i))

                qualified = [(s, idx) for s, idx in all_ssim_scored if s >= ssim_threshold]
                if len(qualified) >= self.ssim_keep:
                    if attempt > 0:
                        print(f"    [Resample] Got {len(qualified)} qualified candidates "
                              f"after {attempt + 1} attempts ({total_generated} total generated)")
                    break
                else:
                    if attempt < max_retries:
                        print(f"    [Resample] Only {len(qualified)}/{self.ssim_keep} candidates "
                              f"with SSIM>={ssim_threshold}, regenerating... "
                              f"(attempt {attempt + 2}/{1 + max_retries})")
            else:
                break

        gen_time = time.time() - t0

        # --- ssim_only: argmax(SSIM) ---
        if self.ssim_only:
            if not self.has_ssim_ref:
                best_idx = 0
                best_score = float("nan")
            else:
                best_score, best_idx = max(all_ssim_scored, key=lambda x: x[0])
            scores_desc = sorted((s for s, _ in all_ssim_scored), reverse=True) if all_ssim_scored else []
            n_qual = sum(1 for s in scores_desc if s >= ssim_threshold)
            best_dur = all_candidates[best_idx].numel() / self.sample_rate
            delta_str = (f" (delta2nd={best_score - scores_desc[1]:+.3f})"
                         if len(scores_desc) > 1 else "")
            print(f"[SSIM] qual={n_qual}/{len(scores_desc)} "
                  f"pick idx={best_idx} SSIM={best_score:.3f}{delta_str} "
                  f"dur={best_dur:.2f}s")
            return all_candidates[best_idx].numpy(), self.sample_rate, {
                "gen_time": gen_time, "sel_time": 0.0,
                "num_candidates": len(all_candidates),
                "ssim_scores": [s for s, _ in all_ssim_scored],
                "best_cer": float("nan"), "best_ssim": float(best_score),
                "retries": attempts_used, "ssim_qualified": n_qual,
                "ssim_fallback": n_qual == 0, "cer_fallback": False,
            }

        t1 = time.time()

        # --- No SSIM ref: CER-only selection across all candidates ---
        if not self.has_ssim_ref:
            if self.cer_selector is None:
                wav = all_candidates[0]
                sel_time = time.time() - t1
                return wav.numpy(), self.sample_rate, {
                    "gen_time": gen_time, "sel_time": sel_time,
                    "num_candidates": len(all_candidates),
                    "ssim_scores": [],
                    "best_cer": float("nan"), "best_ssim": float("nan"),
                    "retries": attempts_used, "ssim_qualified": 0,
                    "ssim_fallback": False, "cer_fallback": False,
                }
            best_wav, best_cer, cer_fallback = self.select_best_cer_only(
                all_candidates, text)
            sel_time = time.time() - t1
            return best_wav.numpy(), self.sample_rate, {
                "gen_time": gen_time, "sel_time": sel_time,
                "num_candidates": len(all_candidates),
                "ssim_scores": [],
                "best_cer": best_cer, "best_ssim": float("nan"),
                "retries": attempts_used, "ssim_qualified": 0,
                "ssim_fallback": False, "cer_fallback": cer_fallback,
            }

        # --- Stage 1: SSIM top-K ---
        qualified = [(s, idx) for s, idx in all_ssim_scored if s >= ssim_threshold]
        qualified.sort(key=lambda x: x[0], reverse=True)
        ssim_qualified_n = len(qualified)
        ssim_fallback = False

        if len(qualified) >= self.ssim_keep:
            top_entries = qualified[:self.ssim_keep]
        elif len(qualified) > 0:
            top_entries = qualified
            ssim_fallback = True
            print(f"    [Resample] WARNING: Only {len(qualified)} candidates met "
                  f"SSIM>={ssim_threshold} after all retries, sending all qualified to CER")
        else:
            all_ssim_scored_sorted = sorted(all_ssim_scored, key=lambda x: x[0], reverse=True)
            top_entries = all_ssim_scored_sorted[:self.ssim_keep]
            ssim_fallback = True
            print(f"    [Resample] WARNING: No candidates met SSIM>={ssim_threshold}, "
                  f"fallback to top-{len(top_entries)} by score")

        filtered_candidates = [all_candidates[idx] for _, idx in top_entries]

        # --- Stage 2: CER selection ---
        best_wav, all_ssim, best_cer, best_ssim, cer_fallback = self.select_best(
            filtered_candidates, top_entries, all_ssim_scored, text)
        sel_time = time.time() - t1

        return best_wav.numpy(), self.sample_rate, {
            "gen_time": gen_time, "sel_time": sel_time,
            "num_candidates": len(all_candidates),
            "ssim_scores": all_ssim,
            "best_cer": best_cer, "best_ssim": best_ssim,
            "retries": attempts_used, "ssim_qualified": ssim_qualified_n,
            "ssim_fallback": ssim_fallback, "cer_fallback": cer_fallback,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Custom-voice TTS with post-selection (SSIM + CER)")
    p.add_argument("--model-path", type=str,
                   default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    p.add_argument("--speaker", type=str, required=True,
                   help="Speaker id (e.g. Vivian). Must be in model's supported speaker list")
    p.add_argument("--instruct", type=str, default=None,
                   help="Optional instruction for style/emotion control")
    p.add_argument("--ref-audio", type=str, default=None,
                   help="Optional reference audio for SSIM evaluation (not used for generation)")
    p.add_argument("--wespeaker-path", type=str, default="",
                   help="Path to WeSpeaker checkpoint (required for SSIM scoring)")
    p.add_argument("--asr-model-path", type=str, default="",
                   help="Path to FunASR Paraformer checkpoint (required for CER scoring)")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num-candidates", type=int, default=8,
                   help="Number of candidates to generate per sentence (batch size)")
    p.add_argument("--ssim-keep", type=int, default=4,
                   help="Number of candidates to keep after SSIM filtering")
    p.add_argument("--ssim-threshold", type=float, default=0.88,
                   help="Minimum SSIM score for a candidate to be qualified")
    p.add_argument("--cer-threshold", type=float, default=0.08,
                   help="CER threshold below which a candidate is considered qualified")
    p.add_argument("--cer-verbose", type=str, default="false")
    p.add_argument("--cer-pick-strategy", type=str, default="medium",
                   choices=["medium", "shorter"],
                   help="Among CER-qualified candidates, pick by duration strategy")
    p.add_argument("--max-retries", type=int, default=8,
                   help="Maximum additional batch generations when SSIM quota is unmet")
    p.add_argument("--use-postprocess", type=str, default="true",
                   help="Enable post-selection pipeline (true/false)")
    p.add_argument("--ssim-only", type=str, default="false",
                   help="Skip CER, return argmax-SSIM candidate (true/false)")
    p.add_argument("--text-file", type=str, default=None,
                   help="Text file with one sentence per line")
    p.add_argument("--texts", nargs="+", default=None,
                   help="Inline texts to synthesize")
    p.add_argument("--output-dir", type=str,
                   default=os.path.join(REPO_ROOT, "out_wavs", "custom_voice_postprocess"))
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility (default: 42)")
    return p.parse_args()


def _str2bool(v: str) -> bool:
    return v.lower() in ("true", "1", "yes")


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

    # --- Text list ---
    if args.text_file:
        with open(args.text_file, "r") as f:
            text_list = [line.strip() for line in f if line.strip()]
    elif args.texts:
        text_list = args.texts
    else:
        text_list = [
            "今天天气真不错，适合出去走走。",
            "我们这款产品采用了全新的设计理念，让用户体验更加流畅。",
            "三月份的销售数据显示，整体业绩同比增长了百分之十五。",
            "请大家注意安全，不要在走廊里奔跑。",
            "这本书讲述了一个关于勇气和坚持的故事。",
        ]

    text_list = [t.replace("@", "").replace("→", "").replace("¿", "") for t in text_list]

    ensure_dir(args.output_dir)

    use_postprocess = _str2bool(args.use_postprocess)
    cer_verbose = _str2bool(args.cer_verbose)
    ssim_only = _str2bool(args.ssim_only)
    if ssim_only and not use_postprocess:
        use_postprocess = True
        print("[Note] --ssim-only=true forces --use-postprocess=true")

    has_ssim_ref = args.ref_audio is not None

    print("=" * 70)
    print("  Custom-Voice TTS Inference with Post-Selection")
    print("=" * 70)
    print(f"  Model:           {args.model_path}")
    print(f"  Speaker:         {args.speaker}")
    print(f"  Instruction:     {args.instruct or '(none)'}")
    print(f"  Ref audio (SSIM):{args.ref_audio or '(none — SSIM disabled)'}")
    print(f"  Texts:           {len(text_list)} sentences")
    print(f"  Candidates/sent: {args.num_candidates} (batch)")
    print(f"  SSIM keep:       {args.ssim_keep}")
    print(f"  SSIM threshold:  {args.ssim_threshold}")
    print(f"  CER threshold:   {args.cer_threshold}")
    print(f"  CER pick:        {args.cer_pick_strategy}")
    print(f"  Max retries:     {args.max_retries}")
    print(f"  Post-selection:  {use_postprocess}")
    print(f"  SSIM-only mode:  {ssim_only}  {'(skip CER, return argmax SSIM)' if ssim_only else ''}")
    print(f"  CER verbose:     {cer_verbose}")
    print(f"  Seed:            {args.seed}")
    print(f"  Output:          {args.output_dir}")
    print("-" * 70)

    # --- Initialize engine ---
    engine = CustomVoiceTTSWithPostSelection(
        model_path=args.model_path,
        wespeaker_path=args.wespeaker_path,
        asr_model_path=args.asr_model_path,
        speaker=args.speaker,
        ref_audio=args.ref_audio,
        device=args.device,
        num_candidates=args.num_candidates,
        ssim_keep=args.ssim_keep,
        cer_threshold=args.cer_threshold,
        cer_verbose=cer_verbose,
        cer_pick_strategy=args.cer_pick_strategy,
        ssim_only=ssim_only,
    )

    # --- Process each text ---
    total_gen_time = 0.0
    total_sel_time = 0.0
    all_cer = []
    all_ssim = []
    resample_count = 0
    ssim_fallback_count = 0
    cer_fallback_count = 0
    retry_dist = {}
    n = len(text_list)

    for idx, text in enumerate(text_list):
        wav, sr, info = engine.synthesize(
            text,
            instruct=args.instruct,
            use_postprocess=use_postprocess,
            ssim_threshold=args.ssim_threshold,
            max_retries=args.max_retries,
        )

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
              f"retry={info['retries']}{flags}  {text_short}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  Total sentences:     {n}")
    print(f"  Total gen time:      {total_gen_time:.1f}s (avg {total_gen_time/n:.1f}s/sent)")
    print(f"  Total select time:   {total_sel_time:.1f}s (avg {total_sel_time/n:.1f}s/sent)")
    if all_ssim:
        print(f"  Avg SSIM (selected): {np.mean(all_ssim):.4f}  "
              f"(min={np.min(all_ssim):.4f}, max={np.max(all_ssim):.4f})")
    if all_cer:
        print(f"  Avg CER  (selected): {np.mean(all_cer):.4f}  "
              f"(min={np.min(all_cer):.4f}, max={np.max(all_cer):.4f})")
    if use_postprocess:
        print(f"  Resample rate:       {resample_count}/{n} ({100*resample_count/n:.0f}%)")
        if has_ssim_ref:
            if ssim_only:
                print(f"  SSIM fallback:       {ssim_fallback_count}/{n} "
                      f"(no candidate >= {args.ssim_threshold}; argmax used anyway)")
            else:
                print(f"  SSIM fallback:       {ssim_fallback_count}/{n} "
                      f"(< {args.ssim_keep} candidates >= {args.ssim_threshold})")
                print(f"  CER  fallback:       {cer_fallback_count}/{n} "
                      f"(no candidate < {args.cer_threshold} CER, used min-CER)")
        else:
            print(f"  CER  fallback:       {cer_fallback_count}/{n} "
                  f"(no candidate < {args.cer_threshold} CER, used min-CER)")
        retry_str = ", ".join(f"{k}={retry_dist[k]}" for k in sorted(retry_dist))
        print(f"  Retry distribution:  {retry_str}")
    print(f"  Output:              {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
