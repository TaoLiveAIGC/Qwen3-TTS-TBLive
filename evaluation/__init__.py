"""Audio post-selection module for TTS quality optimization.

Provides a two-stage pipeline:
  1. Speaker Similarity (SSIM) - filter by voice similarity to reference
  2. Character Error Rate (CER) - pick most accurate pronunciation

Each stage can be used independently or combined via AudioSelector.
"""

from .audio_selector import AudioSelector, AudioSelectorConfig, SelectionResult
from .speaker_similarity import SpeakerSimilaritySelector, SSIMConfig, SSIMResult
from .cer_selector import CERSelector, CERConfig, CERResult
from .text_segmentation import auto_cut_llm

__all__ = [
    "AudioSelector",
    "AudioSelectorConfig",
    "SelectionResult",
    "SpeakerSimilaritySelector",
    "SSIMConfig",
    "SSIMResult",
    "CERSelector",
    "CERConfig",
    "CERResult",
    "auto_cut_llm",
]
