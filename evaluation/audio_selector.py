"""High-level audio post-selection orchestrator.

Combines speaker similarity (SSIM) and CER-based selection into a two-stage pipeline.
"""

import torch
from dataclasses import dataclass
from typing import List, Optional

from .speaker_similarity import SpeakerSimilaritySelector, SSIMConfig, SSIMResult
from .cer_selector import CERSelector, CERConfig, CERResult


@dataclass
class AudioSelectorConfig:
    """Full pipeline configuration.

    Args:
        ssim_model_path: Path to WeSpeaker model directory.
        asr_model_path: Path to FunASR paraformer model directory.
        ssim_device: Device for WeSpeaker model ("cpu" or "cuda").
        asr_device: Device for ASR model ("cpu" or "cuda").
        asr_batch_size: Batch size for ASR inference.
        source_sample_rate: Sample rate of input candidate audios.
        repeat_count: Candidates to keep per sentence after SSIM (top-K).
        group_count: SSIM group multiplier (ssim_repeat).
        language: Language for text normalization.
        enable_ssim: Whether to enable SSIM stage.
        enable_cer: Whether to enable CER stage.
    """
    ssim_model_path: str = ""
    asr_model_path: str = ""
    ssim_device: str = "cpu"
    asr_device: str = "cuda"
    asr_batch_size: int = 20
    source_sample_rate: int = 24000
    repeat_count: int = 2
    group_count: int = 2
    language: str = "zh"
    enable_ssim: bool = True
    enable_cer: bool = True


@dataclass
class SelectionResult:
    """Final result of the full selection pipeline."""
    selected_wavs: List[torch.Tensor]
    selected_texts: List[str]
    ssim_result: Optional[SSIMResult] = None
    cer_result: Optional[CERResult] = None


class AudioSelector:
    """Two-stage audio post-selection pipeline.

    Stage 1 (SSIM): From N * repeat_count * group_count candidates,
                    keep top repeat_count per sentence by speaker similarity.
    Stage 2 (CER):  From N * repeat_count candidates,
                    pick 1 best per sentence by lowest character error rate.

    Each stage can be independently disabled via config.

    Example:
        >>> config = AudioSelectorConfig(
        ...     ssim_model_path="/path/to/wespeaker/chinese",
        ...     asr_model_path="/path/to/paraformer-zh",
        ...     ssim_device="cuda",
        ...     asr_device="cuda",
        ...     repeat_count=2,
        ...     group_count=2,
        ... )
        >>> selector = AudioSelector(config)
        >>> result = selector.select_best(
        ...     candidates=all_candidate_wavs,
        ...     ref_audio_path="/path/to/ref.wav",
        ...     ground_truth_texts=texts,
        ... )
        >>> final_wavs = result.selected_wavs  # one per sentence
    """

    def __init__(self, config: AudioSelectorConfig):
        self._config = config
        self._ssim_selector: Optional[SpeakerSimilaritySelector] = None
        self._cer_selector: Optional[CERSelector] = None

        if config.enable_ssim and config.ssim_model_path:
            self._ssim_selector = SpeakerSimilaritySelector(
                SSIMConfig(
                    model_path=config.ssim_model_path,
                    device=config.ssim_device,
                )
            )

        if config.enable_cer and config.asr_model_path:
            self._cer_selector = CERSelector(
                CERConfig(
                    asr_model_path=config.asr_model_path,
                    device=config.asr_device,
                    batch_size=config.asr_batch_size,
                    source_sample_rate=config.source_sample_rate,
                    language=config.language,
                )
            )

    @property
    def ssim_selector(self) -> Optional[SpeakerSimilaritySelector]:
        """Access the SSIM selector directly for independent use."""
        return self._ssim_selector

    @property
    def cer_selector(self) -> Optional[CERSelector]:
        """Access the CER selector directly for independent use."""
        return self._cer_selector

    def select_best(
        self,
        candidates: List[torch.Tensor],
        ref_audio_path: str,
        ground_truth_texts: List[str],
        repeat_count: Optional[int] = None,
        group_count: Optional[int] = None,
    ) -> SelectionResult:
        """Run the full two-stage selection pipeline.

        Args:
            candidates: List of candidate audio tensors (at source_sample_rate).
                        Length = N_sentences * repeat_count * group_count (if both stages enabled)
                        or N_sentences * repeat_count (if only CER enabled).
            ref_audio_path: Path to reference speaker audio file.
            ground_truth_texts: Parallel text list aligned with candidates.
            repeat_count: Override config repeat_count.
            group_count: Override config group_count.

        Returns:
            SelectionResult with final selected wavs (one per sentence).
        """
        repeat_count = repeat_count or self._config.repeat_count
        group_count = group_count or self._config.group_count
        sample_rate = self._config.source_sample_rate

        current_wavs = candidates
        current_texts = ground_truth_texts
        ssim_result = None
        cer_result = None

        # Stage 1: SSIM filtering
        if self._ssim_selector and self._config.enable_ssim:
            ssim_result = self._ssim_selector.select(
                candidates=current_wavs,
                ref_path=ref_audio_path,
                texts=current_texts,
                repeat_count=repeat_count,
                group_count=group_count,
                sample_rate=sample_rate,
            )
            current_wavs = ssim_result.selected_wavs
            current_texts = ssim_result.selected_texts

        # Stage 2: CER filtering
        if self._cer_selector and self._config.enable_cer and repeat_count > 1:
            cer_result = self._cer_selector.select(
                candidates=current_wavs,
                texts=current_texts,
                repeat_count=repeat_count,
                sample_rate=sample_rate,
            )
            current_wavs = cer_result.selected_wavs
            current_texts = cer_result.selected_texts

        return SelectionResult(
            selected_wavs=current_wavs,
            selected_texts=current_texts,
            ssim_result=ssim_result,
            cer_result=cer_result,
        )
