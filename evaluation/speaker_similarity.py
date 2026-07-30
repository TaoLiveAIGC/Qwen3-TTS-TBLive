"""Speaker similarity selection using WeSpeaker embeddings.

Filters audio candidates by cosine similarity to a reference speaker embedding.
"""

import librosa
import torch
import torchaudio
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .audio_utils import ensure_2d, resample

# Compatibility shim for newer torchaudio versions
if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda x: None  # type: ignore[attr-defined]


@dataclass
class SSIMConfig:
    """Configuration for speaker similarity selection."""
    model_path: str
    device: str = "cpu"
    sample_rate: int = 16000  # WeSpeaker expects 16kHz


@dataclass
class SSIMResult:
    """Result of speaker similarity filtering."""
    selected_wavs: List[torch.Tensor]
    selected_texts: List[str]
    selected_indices: List[int]
    scores: List[float]


class SpeakerSimilaritySelector:
    """Filters audio candidates by speaker embedding similarity to a reference.

    Uses WeSpeaker to extract speaker embeddings and compute cosine similarity.
    Given candidates structured as N_sentences * repeats, keeps top-K most similar
    to the reference speaker per sentence group.

    Example:
        >>> config = SSIMConfig(model_path="/path/to/wespeaker/chinese", device="cuda")
        >>> selector = SpeakerSimilaritySelector(config)
        >>> result = selector.select(
        ...     candidates=wav_list,
        ...     ref_path="/path/to/ref.wav",
        ...     texts=text_list,
        ...     repeat_count=2,
        ...     group_count=2,
        ...     sample_rate=24000,
        ... )
    """

    def __init__(self, config: SSIMConfig):
        from wespeaker.cli.speaker import Speaker
        self._config = config
        self._model = Speaker(config.model_path)
        if config.device == "cuda":
            self._model.set_device(config.device)

    def extract_reference_embedding(self, ref_path: str) -> torch.Tensor:
        """Extract speaker embedding from a reference audio file.

        Uses librosa to load audio to avoid torchaudio/torchcodec GCC version
        compatibility issues in some environments.

        Args:
            ref_path: Path to reference WAV file.

        Returns:
            Speaker embedding tensor.
        """
        # Load with librosa at 16kHz (WeSpeaker expected sample rate) to bypass
        # torchaudio.load which sometimes fails due to torchcodec/GCC mismatch.
        wav, sr = librosa.load(ref_path, sr=self._config.sample_rate, mono=True)
        pcm = torch.from_numpy(wav).unsqueeze(0)  # [1, time]
        return self._model.extract_embedding_from_pcm(pcm, sample_rate=self._config.sample_rate)

    def compute_similarity(
        self,
        wav: torch.Tensor,
        ref_embedding: torch.Tensor,
        sample_rate: int = 16000,
    ) -> float:
        """Compute cosine similarity between a candidate and reference embedding.

        Args:
            wav: Audio tensor (1D or 2D).
            ref_embedding: Pre-computed reference speaker embedding.
            sample_rate: Sample rate of wav.

        Returns:
            Cosine similarity score.
        """
        wav_2d = ensure_2d(wav)
        wav_emb = self._model.extract_embedding_from_pcm(wav_2d, sample_rate=sample_rate)
        return float(self._model.cosine_similarity(wav_emb, ref_embedding))

    def select(
        self,
        candidates: List[torch.Tensor],
        ref_path: str,
        texts: List[str],
        repeat_count: int,
        group_count: int,
        sample_rate: int = 24000,
    ) -> SSIMResult:
        """Select top-K candidates per sentence group by speaker similarity.

        The candidate list is structured as interleaved:
          [sent0_r0, sent1_r0, ..., sentN_r0, sent0_r1, ..., sentN_r(M-1)]
        where total = N_sentences * repeat_count * group_count.

        After selection, keeps `repeat_count` best per sentence, re-interleaved.

        Args:
            candidates: List of audio tensors at source sample_rate.
            ref_path: Path to reference speaker audio.
            texts: Parallel text list aligned with candidates.
            repeat_count: Number of candidates to keep per sentence (top-K).
            group_count: Group multiplier (ssim_repeat).
            sample_rate: Sample rate of candidate audios.

        Returns:
            SSIMResult with selected wavs, texts, indices, and scores.
        """
        keep_k = repeat_count
        repeats = repeat_count * group_count
        interval = len(candidates) // repeats

        ref_emb = self.extract_reference_embedding(ref_path)

        selected_indices: List[int] = []
        selected_scores: List[float] = []

        for k in range(interval):
            group_indices = [r * interval + k for r in range(repeats)]
            scored: List[Tuple[float, int]] = []
            for orig_idx in group_indices:
                score = self.compute_similarity(
                    candidates[orig_idx], ref_emb, sample_rate=sample_rate
                )
                scored.append((score, orig_idx))
            scored.sort(key=lambda x: x[0], reverse=True)
            for score, idx in scored[:keep_k]:
                selected_indices.append(idx)
                selected_scores.append(score)

        # Gather in grouped order
        selected_wavs = [candidates[i] for i in selected_indices]
        selected_texts = [texts[i] for i in selected_indices]

        # Re-interleave: from [k0_top0..k0_topK, k1_top0..k1_topK, ...]
        # back to [sent0_r0, sent1_r0, ..., sentN_r0, sent0_r1, ...]
        perm: List[int] = []
        for r in range(keep_k):
            for k in range(interval):
                perm.append(k * keep_k + r)

        selected_wavs = [selected_wavs[i] for i in perm]
        selected_texts = [selected_texts[i] for i in perm]
        selected_indices = [selected_indices[i] for i in perm]
        selected_scores = [selected_scores[i] for i in perm]

        return SSIMResult(
            selected_wavs=selected_wavs,
            selected_texts=selected_texts,
            selected_indices=selected_indices,
            scores=selected_scores,
        )
