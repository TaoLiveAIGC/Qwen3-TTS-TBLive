# coding=utf-8
"""Prosody reward dimensions for GRPO training.

Provides:
  - CpsReward: chars per second (speaking rate)
  - SemiFlReward: semitone fluctuation percentage (pitch dynamics)
  - gdpo_4dim_aggregate: batch-level z-score aggregation over 4 dimensions
"""

import math
import re
from typing import Dict, List, Tuple

import numpy as np


def _count_pronounceable(text: str) -> int:
    """Count pronounceable characters (CJK + alphanumeric), stripping punctuation."""
    cleaned = re.sub(r'[^一-龥a-zA-Z0-9]', '', text)
    return len(cleaned)


class CpsReward:
    """Chars per second — per-sample speaking rate."""

    def compute(self, waveforms, texts: List[str], sample_rate: int) -> List[float]:
        """Returns list of CPS values (chars/sec).

        Args:
            waveforms: list of 1-D float numpy arrays (generated audio).
            texts: list of target texts (same text repeated G times).
            sample_rate: sampling rate of the waveforms.

        Returns:
            List of CPS values, one per sample.
        """
        results = []
        for wav, text in zip(waveforms, texts):
            dur = len(wav) / max(sample_rate, 1)
            if dur < 0.1:
                results.append(0.0)
            else:
                n_chars = _count_pronounceable(text)
                results.append(n_chars / dur)
        return results


class SemiFlReward:
    """Semitone fluctuation % — fraction of adjacent F0 frames with >1 semitone pitch change.

    Higher values indicate more pitch variation (more expressive/natural prosody).
    """

    def __init__(self, sr: int = 22050):
        self.sr = sr

    def compute(self, waveforms, sample_rate: int) -> List[float]:
        """Returns list of semi_fl% values.

        Args:
            waveforms: list of 1-D float numpy arrays (generated audio).
            sample_rate: sampling rate of the waveforms.

        Returns:
            List of semitone fluctuation percentages, one per sample.
        """
        import librosa

        results = []
        for wav in waveforms:
            try:
                if len(wav) < int(0.2 * sample_rate):
                    results.append(0.0)
                    continue

                # Resample to self.sr for consistent F0 extraction
                if sample_rate != self.sr:
                    y = librosa.resample(
                        wav.astype(np.float32),
                        orig_sr=sample_rate,
                        target_sr=self.sr,
                    )
                else:
                    y = wav.astype(np.float32)

                f0, _, _ = librosa.pyin(
                    y,
                    fmin=librosa.note_to_hz('C2'),
                    fmax=librosa.note_to_hz('C6'),
                    sr=self.sr,
                )
                if f0 is None:
                    results.append(0.0)
                    continue

                midi_f0 = librosa.hz_to_midi(f0)
                midi_diff = np.diff(midi_f0)
                valid_diffs = midi_diff[~np.isnan(midi_diff)]

                if len(valid_diffs) < 5:
                    results.append(0.0)
                    continue

                semi_fl = float(np.sum(np.abs(valid_diffs) > 1.0) / len(valid_diffs) * 100)
                results.append(semi_fl)
            except Exception:
                results.append(0.0)
        return results


def gdpo_4dim_aggregate(
    all_sample_details: List[dict],
    weights: Dict[str, float] = None,
    cps_deadzone_low: float = 0.05,
    cps_deadzone_high: float = 0.10,
    cer_deadzone: float = 0.03,
    cer_exp_k: float = 3.0,
    eps: float = 1e-8,
) -> Tuple[List[List[float]], Dict[str, Tuple[float, float]]]:
    """GDPO 4-dim: batch-level z-score over {cer, sim, cps, semi_fl}.

    Each dimension is converted to a "higher = better" raw reward, then
    z-scored across the entire batch, then weighted-summed into a single
    advantage per sample.

    Dimensions:
      cer:     dead-zone + exponential penalty (cer <= tau → 0, else -(exp(k*(cer-tau))-1))
      sim:     raw cosine similarity (higher = better)
      cps:     dead-zone around GT CPS (inside → 0, outside → linear penalty)
      semi_fl: raw semitone fluctuation % (higher = better)

    Args:
        all_sample_details: list of dicts per prompt, each with:
            cers: list of CER values [G]
            sim_rewards: list of SSIM values [G]
            cps_vals: list of CPS values [G]
            semi_fl_vals: list of SemiFL values [G]
            gt_cps: float, ground-truth CPS for this prompt
        weights: {"cer": w, "sim": w, "cps": w, "semi_fl": w}
        cps_deadzone_low: fractional tolerance below GT CPS (default 5%)
        cps_deadzone_high: fractional tolerance above GT CPS (default 10%)
        cer_deadzone: CER threshold below which reward = 0 (default 0.03)
        cer_exp_k: exponential growth rate for CER penalty (default 3.0)
        eps: numerical stability constant

    Returns:
        advantages: list[list[float]] — per-prompt groups of advantage values
        dim_stats: dict {dim: (batch_mean, batch_std)}
    """
    if weights is None:
        weights = {"cer": 1.0, "sim": 1.0, "cps": 1.0, "semi_fl": 1.0}

    # Flatten all samples, computing per-sample raw rewards
    flat = []
    for d in all_sample_details:
        G = len(d["cers"])
        gt_cps = d.get("gt_cps", 5.0)
        lower = gt_cps * (1.0 - cps_deadzone_low)
        upper = gt_cps * (1.0 + cps_deadzone_high)

        for i in range(G):
            # CER: dead-zone + exponential penalty
            cer_i = d["cers"][i]
            if cer_i <= cer_deadzone:
                cer_reward = 0.0
            else:
                cer_reward = -(math.exp(cer_exp_k * (cer_i - cer_deadzone)) - 1.0)

            # CPS: dead-zone around GT, linear penalty outside
            gen_cps = d["cps_vals"][i]
            if gen_cps < lower:
                cps_reward = -(lower - gen_cps) / max(gt_cps, 1e-6)
            elif gen_cps > upper:
                cps_reward = -(gen_cps - upper) / max(gt_cps, 1e-6)
            else:
                cps_reward = 0.0

            flat.append({
                "cer": cer_reward,
                "sim": d["sim_rewards"][i],
                "cps": cps_reward,
                "semi_fl": d["semi_fl_vals"][i],
            })

    N = len(flat)
    if N == 0:
        return [], {}

    # Batch-level z-score + weighted sum
    advantages_flat = np.zeros(N, dtype=np.float32)
    dim_stats: Dict[str, Tuple[float, float]] = {}

    active_dims = [k for k in ("cer", "sim", "cps", "semi_fl") if weights.get(k, 0) > 0]
    for k in active_dims:
        vals = np.array([e[k] for e in flat], dtype=np.float32)
        m, s = float(vals.mean()), float(vals.std())
        dim_stats[k] = (m, s)
        z = (vals - m) / (s + eps)
        w = weights.get(k, 1.0)
        advantages_flat += w * z

    # Reshape to per-prompt groups
    advantages = []
    cursor = 0
    for d in all_sample_details:
        G = len(d["cers"])
        advantages.append(advantages_flat[cursor:cursor + G].tolist())
        cursor += G

    return advantages, dim_stats
