"""Shared audio processing utilities for the evaluation module."""

from typing import List

import numpy as np
import torch
import torchaudio.transforms as T


def ensure_2d(wav: torch.Tensor) -> torch.Tensor:
    """Ensure waveform tensor has shape (1, T) — mono, 2D.

    Handles:
      - 1D tensor (T,) -> (1, T)
      - 2D mono (1, T) -> unchanged
      - 2D multi-channel (C, T) -> averaged to (1, T)
    """
    if wav.dim() == 1:
        return wav.unsqueeze(0)
    if wav.dim() != 2:
        raise ValueError(f"wav must be 1D or 2D, got shape={tuple(wav.shape)}")
    if wav.size(0) == 1:
        return wav
    return wav.mean(dim=0, keepdim=True)


def resample(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Resample a waveform tensor from orig_sr to target_sr.

    Args:
        wav: Audio tensor, any shape supported by torchaudio Resample.
        orig_sr: Original sample rate.
        target_sr: Target sample rate.

    Returns:
        Resampled tensor.
    """
    if orig_sr == target_sr:
        return wav
    resampler = T.Resample(orig_freq=orig_sr, new_freq=target_sr).to(wav.device)
    return resampler(wav)


def to_numpy_16k(audio_list: List[torch.Tensor], source_sr: int) -> List[np.ndarray]:
    """Convert list of torch.Tensor audio at source_sr to float32 numpy at 16kHz.

    This is the format expected by FunASR for batch inference.

    Args:
        audio_list: List of audio tensors (1D or 2D).
        source_sr: Sample rate of the input tensors.

    Returns:
        List of 1D float32 numpy arrays at 16kHz.
    """
    target_sr = 16000
    if not audio_list:
        return []

    device = audio_list[0].device
    resampler = None
    if source_sr != target_sr:
        resampler = T.Resample(orig_freq=source_sr, new_freq=target_sr).to(device)

    processed = []
    for audio in audio_list:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if resampler is not None:
            audio = resampler(audio)
        arr = audio.cpu().numpy().squeeze().astype(np.float32)
        processed.append(arr)

    return processed
