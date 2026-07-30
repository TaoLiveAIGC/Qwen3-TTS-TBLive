"""CER-based audio selection using FunASR for transcription and jiwer for scoring.

Selects the best audio candidate per sentence group by Character Error Rate.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple

from .audio_utils import to_numpy_16k
from .text_normalize import TextNormalizer


@dataclass
class CERConfig:
    """Configuration for CER-based selection."""
    asr_model_path: str
    device: str = "cuda"
    batch_size: int = 20
    source_sample_rate: int = 24000
    language: str = "zh"
    cer_threshold: float = 0.08
    verbose: bool = False
    pick_strategy: str = "medium"  # "medium" (median duration) | "shorter" (shortest duration)


@dataclass
class CERResult:
    """Result of CER-based filtering."""
    selected_wavs: List[torch.Tensor]
    selected_texts: List[str]
    selected_indices: List[int]
    cer_scores: List[float]
    transcriptions: List[str]
    fallback_count: int = 0  # # of groups that used min-cer fallback (no candidate met threshold)


class CERSelector:
    """Selects the best audio candidate per sentence group based on Character Error Rate.

    Pipeline:
    1. Resample candidates to 16kHz for ASR.
    2. Run batch ASR (FunASR paraformer) to transcribe all candidates.
    3. Compute CER between transcriptions and ground-truth texts.
    4. For each sentence group, pick the candidate with lowest CER.

    Example:
        >>> config = CERConfig(asr_model_path="/path/to/paraformer-zh", device="cuda")
        >>> selector = CERSelector(config)
        >>> result = selector.select(
        ...     candidates=wav_list,
        ...     texts=ground_truth_list,
        ...     repeat_count=3,
        ...     sample_rate=24000,
        ... )
    """

    def __init__(self, config: CERConfig):
        from funasr import AutoModel
        self._config = config
        self._asr_model = AutoModel(
            model=config.asr_model_path,
            model_revision="v2.0.4",
            device=config.device,
            batch_size=config.batch_size,
            disable_pbar=True,
            disable_update=True,
        )
        self._normalizer = TextNormalizer()

    def transcribe(self, audios: List[np.ndarray]) -> List[str]:
        """Run ASR on a batch of numpy audio arrays (16kHz, float32).

        Args:
            audios: List of 1D float32 numpy arrays at 16kHz.

        Returns:
            List of transcribed text strings.
        """
        try:
            results = self._asr_model.generate(input=audios)
            return [item['text'] for item in results]
        except Exception as e:
            print(f"[CERSelector] ASR error: {e}")
            return [""] * len(audios)

    def compute_cer(
        self,
        hypotheses: List[str],
        references: List[str],
    ) -> List[float]:
        """Compute CER between hypothesis and reference text lists.

        Args:
            hypotheses: ASR transcriptions.
            references: Ground-truth texts.

        Returns:
            List of CER values (0.0 = perfect match).
        """
        import jiwer

        results = []
        for hyp, ref in zip(hypotheses, references):
            norm_hyp = self._normalizer.normalize(hyp, self._config.language)
            norm_ref = self._normalizer.normalize(ref, self._config.language)
            if not norm_ref:
                results.append(0.0)
                continue
            cer = jiwer.cer(norm_ref, norm_hyp)
            results.append(round(cer, 4))
        return results

    def select(
        self,
        candidates: List[torch.Tensor],
        texts: List[str],
        repeat_count: int,
        sample_rate: int = 24000,
    ) -> CERResult:
        """Select one best candidate per sentence group by lowest CER.

        The candidate list is interleaved as:
          [sent0_r0, sent1_r0, ..., sentN_r0, sent0_r1, ..., sentN_r(K-1)]
        with total = N_sentences * repeat_count.

        For each sentence position, evaluates CER across all repeat_count
        candidates and picks the one with minimum CER.

        Args:
            candidates: List of audio tensors.
            texts: Parallel list of ground-truth texts.
            repeat_count: Number of candidates per sentence to choose from.
            sample_rate: Sample rate of candidate audios.

        Returns:
            CERResult with selected wavs, texts, indices, CER scores, and transcriptions.
        """
        interval = len(candidates) // repeat_count

        # Resample all candidates to 16kHz numpy for ASR
        audios_16k = to_numpy_16k(candidates, source_sr=sample_rate)

        # Batch transcribe
        asr_texts = self.transcribe(audios_16k)

        # Compute CER for all candidates
        cer_scores = self.compute_cer(asr_texts, texts)

        # For each sentence position, pick the candidate with lowest CER
        selected_indices: List[int] = []
        selected_cer: List[float] = []
        selected_transcriptions: List[str] = []
        fallback_count = 0  # sentences where no candidate met cer_threshold

        cer_threshold = self._config.cer_threshold
        verbose = self._config.verbose
        pick_strategy = self._config.pick_strategy
        if pick_strategy not in ("medium", "shorter"):
            raise ValueError(f"Unsupported pick_strategy: {pick_strategy!r} (expected 'medium' or 'shorter')")

        if verbose:
            print(f"[CERSelector] Starting selection: {interval} sentences, "
                  f"{repeat_count} candidates each, CER threshold={cer_threshold}")
            print(f"[CERSelector] {'='*80}")

        for k in range(interval):
            group_indices = [r * interval + k for r in range(repeat_count)]
            group_cers = [cer_scores[idx] for idx in group_indices]
            group_durs = [candidates[idx].numel() / sample_rate for idx in group_indices]

            if verbose:
                print(f"[CERSelector] Sentence {k}: GT='{texts[group_indices[0]][:40]}'")
                for rank, idx in enumerate(group_indices):
                    cer_val = cer_scores[idx]
                    qualified = "✓" if cer_val < cer_threshold else "✗"
                    print(f"[CERSelector]   candidate {rank}: idx={idx}, "
                          f"CER={cer_val:.4f} [{qualified}], "
                          f"duration={group_durs[rank]:.2f}s ({candidates[idx].numel()} samples), "
                          f"ASR='{asr_texts[idx][:40]}'")

            # Find candidates below CER threshold
            below_thresh = [
                (idx, cer_scores[idx])
                for idx in group_indices
                if cer_scores[idx] < cer_threshold
            ]
            lowest_cer_idx = group_indices[group_cers.index(min(group_cers))]

            if below_thresh:
                # Among qualified candidates, pick by configured strategy
                below_thresh_sorted = sorted(below_thresh, key=lambda x: candidates[x[0]].numel())
                if pick_strategy == "shorter":
                    best_idx = below_thresh_sorted[0][0]
                    strategy = "short-dur"
                else:  # "medium"
                    pick_idx = len(below_thresh_sorted) // 2
                    best_idx = below_thresh_sorted[pick_idx][0]
                    strategy = "med-dur"
            else:
                # Fallback: pick lowest CER
                best_idx = lowest_cer_idx
                strategy = "min-cer-fallback"
                fallback_count += 1

            # Compact one-line summary (always printed)
            qual_n = len(below_thresh)
            cer_range = f"{min(group_cers):.3f}-{max(group_cers):.3f}"
            dur_range = f"{min(group_durs):.2f}-{max(group_durs):.2f}s"
            alt_note = ""
            if best_idx != lowest_cer_idx:
                alt_note = f" (vs min-CER idx={lowest_cer_idx} CER={cer_scores[lowest_cer_idx]:.4f})"
            print(f"[CER][s{k}] qual={qual_n}/{repeat_count} "
                  f"CER {cer_range} dur {dur_range} | "
                  f"pick idx={best_idx} CER={cer_scores[best_idx]:.4f} "
                  f"dur={candidates[best_idx].numel()/sample_rate:.2f}s [{strategy}]{alt_note}")

            if verbose:
                print(f"[CERSelector]   {'-'*60}")

            selected_indices.append(best_idx)
            selected_cer.append(cer_scores[best_idx])
            selected_transcriptions.append(asr_texts[best_idx])

        # Summary log
        if verbose:
            avg_cer = sum(selected_cer) / len(selected_cer) if selected_cer else 0.0
            avg_duration = sum(candidates[i].numel() for i in selected_indices) / len(selected_indices) / sample_rate if selected_indices else 0.0
            print(f"[CERSelector] {'='*80}")
            print(f"[CERSelector] Selection complete: {len(selected_indices)} sentences selected, "
                  f"fallback={fallback_count}")
            print(f"[CERSelector]   Avg CER: {avg_cer:.4f}, Avg duration: {avg_duration:.2f}s")

        selected_wavs = [candidates[i] for i in selected_indices]
        selected_texts = [texts[i] for i in selected_indices]

        return CERResult(
            selected_wavs=selected_wavs,
            selected_texts=selected_texts,
            selected_indices=selected_indices,
            cer_scores=selected_cer,
            transcriptions=selected_transcriptions,
            fallback_count=fallback_count,
        )
