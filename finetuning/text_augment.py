# coding=utf-8
"""Text augmentation helpers for the SFT training pipeline.

Wraps `g2pw_tool.G2PWTextProcessor.random_replace_with_phonetics` behind a
process-level lazy singleton so DataLoader workers only pay the model load
cost on first use. Also centralizes the symbol cleanup applied to every
training text source.

If `g2pw_tool` is not installed, phonetic augmentation is silently disabled
(a one-time warning is printed) and inputs pass through unchanged. This
allows the SFT pipeline to run without downloading the g2pw model.
"""

from __future__ import annotations

import random
from typing import Optional

_PROCESSOR = None
_G2PW_DISABLED = False  # sentinel: True once we know g2pw_tool is unavailable
_STRIP_CHARS = ("@", "→", "¿")


def _clean_text_symbols(text: str) -> str:
    """Strip auxiliary symbols that should never appear in training text."""
    if not text:
        return text
    for ch in _STRIP_CHARS:
        if ch in text:
            text = text.replace(ch, "")
    return text


def _get_processor():
    """Lazy-load a single G2PWTextProcessor per process.

    Device resolution follows g2pw_tool defaults (G2PW_DEVICE env →
    LOCAL_RANK/RANK → cuda → cpu), so each DDP rank pins to its own GPU
    automatically. Set G2PW_DEVICE=cpu to override.

    Returns None if g2pw_tool is not installed — caller must handle.
    """
    global _PROCESSOR, _G2PW_DISABLED
    if _G2PW_DISABLED:
        return None
    if _PROCESSOR is None:
        try:
            from g2pw_tool import G2PWTextProcessor
        except ImportError as e:
            _G2PW_DISABLED = True
            print(f"[text_augment] WARNING: g2pw_tool not available "
                  f"({e.__class__.__name__}: {e}); phonetic augmentation is DISABLED. "
                  f"Text will pass through unchanged.")
            return None
        _PROCESSOR = G2PWTextProcessor(use_bert=False)
    return _PROCESSOR


def maybe_replace_with_phonetics(
    text: str,
    *,
    max_n: int,
    prob: float,
    mode: str = "pinyin",
    seed: Optional[int] = None,
) -> str:
    """Probabilistically replace up to `max_n` Chinese chars with their phonetics.

    Two-stage gating:
      1. Outer gate `prob`: whether this sample enters the replacement
         pipeline at all. A draw of `random() < prob` enters; otherwise the
         original text is returned untouched (sample trains as plain text).
      2. Inner sampler: once inside the pipeline, `n = randint(1, max_n)`
         characters are replaced (always at least one — the outer gate
         already decided that this sample should be augmented).

    Args:
        text: Input text.
        max_n: Upper bound for the number of characters to replace per sample
            once the outer gate fires. `max_n <= 0` disables the augmentation.
        prob: Probability in [0, 1] that the sample enters the replacement
            pipeline. `prob <= 0` disables; `prob >= 1` always enters.
        mode: 'pinyin' (e.g. 因 → yin1) or 'phoneme' (e.g. 因 → y in1).
        seed: Optional RNG seed forwarded to the underlying replacer for
            reproducibility. Pass None for fresh randomness per call.

    Returns:
        Possibly-augmented text. The original string is returned unchanged
        when augmentation is disabled, the outer gate misses, the input
        has no Chinese characters, or `g2pw_tool` is not installed.
    """
    if max_n <= 0 or prob <= 0.0 or not text:
        return text
    if prob < 1.0 and random.random() >= prob:
        return text

    proc = _get_processor()
    if proc is None:  # g2pw_tool unavailable → gracefully skip augmentation
        return text

    n = random.randint(1, max_n)
    return proc.random_replace_with_phonetics(text, n=n, mode=mode, seed=seed)
