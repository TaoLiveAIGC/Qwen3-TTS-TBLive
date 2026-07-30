# coding=utf-8

from __future__ import annotations

import json
import os
import random
import time
from typing import Any, List

import librosa
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from dataset_utils import add_special_tokens_to_tokenizer, resolve_path
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram


class TTSGRPODataset(Dataset):
    """Dataset producing per-sample prompts for GRPO rollout (ref + target only, no audio_codes).

    Supports two data formats:
      - Grouped (default): JSON list[list[dict]], each inner list is a same-speaker
        group. A target and ref are picked from different items within the group.
      - Flat (``flatten=True``): JSON list[dict], each item must carry its own ref
        audio (const_ref_audio / dynamic_ref_audio / ref_audio), or falls back to
        the item's own audio path as ref.
    """

    def __init__(
        self,
        data_list: List[str],
        processor: Any,
        config: Any,
        tokenizer_path: str | None = None,
        max_steps: float = 2e5,
        mode: str = "train",
        flatten: bool = False,
        data_split_mode: str = "auto",
        use_const_ref: bool = False,
    ):
        self.processor = processor
        self.config = config
        self.max_train_step = int(max_steps)
        self.mode = mode
        self.default_sr = 24000
        self.flatten = flatten
        self.use_const_ref = use_const_ref

        if data_split_mode not in ("auto", "files", "samples"):
            raise ValueError(f"data_split_mode must be auto/files/samples, got {data_split_mode!r}")
        self.data_split_mode = data_split_mode

        self._get_dist_info()
        add_special_tokens_to_tokenizer(self.processor.tokenizer)

        self.list_data_files = [str(p).strip() for p in data_list if str(p).strip()]
        self._sample_split = self._resolve_split_mode()
        self._get_local_data_files()
        self.data_queue: List = []
        self.read_file_idx = 0
        self._update_data_queue()

    # ------------------------------------------------------------------ #
    # Distributed helpers
    # ------------------------------------------------------------------ #

    def _get_dist_info(self) -> None:
        if "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.rank = int(os.environ["RANK"])
        elif dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0
        if "LOCAL_RANK" in os.environ:
            self.local_rank = int(os.environ["LOCAL_RANK"])
        elif torch.cuda.is_available():
            self.local_rank = torch.cuda.current_device()
        else:
            self.local_rank = 0
        print(
            f"[TTSGRPODataset] world_size={self.world_size} rank={self.rank} "
            f"local_rank={self.local_rank}"
        )

    def _resolve_split_mode(self) -> bool:
        """Return True if samples should be sliced within files; False to split files across ranks."""
        if self.data_split_mode == "samples":
            return True
        if self.data_split_mode == "files":
            return False
        # auto: switch to sample-split when there aren't enough files to cover all ranks
        return len(self.list_data_files) < self.world_size

    def _get_local_data_files(self) -> None:
        """Distribute data files across ranks (files mode) or share all files (samples mode)."""
        total = len(self.list_data_files)
        if total == 0:
            return
        if self._sample_split:
            print(f"[TTSGRPODataset] rank={self.rank} sample-split mode: shares all {total} file(s), "
                  f"per-rank slicing inside _load_data_file (stride={self.world_size})")
            return
        per_rank = (total + self.world_size - 1) // self.world_size
        pad = per_rank * self.world_size - total
        if pad > 0:
            self.list_data_files.extend(self.list_data_files[i % total] for i in range(pad))
        start = self.rank * per_rank
        self.list_data_files = self.list_data_files[start : start + per_rank]
        print(f"[TTSGRPODataset] rank={self.rank} files-split mode: files_count={len(self.list_data_files)}")

    # ------------------------------------------------------------------ #
    # Data loading
    # ------------------------------------------------------------------ #

    def _load_data_file(self, file_path: str) -> List:
        file_path = resolve_path(file_path)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)

        if self.flatten:
            print(f"[TTSGRPODataset] flatten dataset, origin len is: {len(data)}.")
            data_list = []
            for item in data:
                if isinstance(item, list):
                    data_list.extend(item)
                else:
                    data_list.append(item)
            print(f"[TTSGRPODataset] flatten dataset list[list] to list[dict], "
                  f"from {len(data)} items to {len(data_list)} dicts.")
            data = data_list

        if self._sample_split and self.world_size > 1:
            n_before = len(data)
            # Deterministic shuffle (same seed across ranks) for balanced stride
            rng = random.Random(42)
            rng.shuffle(data)
            data = data[self.rank :: self.world_size]
            print(f"[TTSGRPODataset] rank={self.rank} sample-split slice: {n_before} → {len(data)} "
                  f"(stride {self.world_size})")

        if not self.flatten:
            # Grouped mode: filter groups with <2 items, UNLESS the single item
            # carries its own ref_audio (const_ref_audio / dynamic_ref_audio / ref_audio).
            filtered = []
            for group in data:
                if len(group) >= 2:
                    filtered.append(group)
                elif len(group) == 1 and self._item_has_ref_audio(group[0]):
                    filtered.append(group)
            data = filtered

        return data

    def _update_data_queue(self) -> None:
        if len(self.data_queue) > 0:
            return
        if not self.list_data_files:
            return
        if self.read_file_idx >= len(self.list_data_files):
            self.read_file_idx = 0
        path = self.list_data_files[self.read_file_idx]
        self.data_queue.extend(self._load_data_file(path))
        self.read_file_idx += 1
        random.seed(int(time.time()) + self.rank)
        random.shuffle(self.data_queue)
        print(
            f"[TTSGRPODataset] rank={self.rank} file_idx={self.read_file_idx - 1} "
            f"queue_size={len(self.data_queue)}"
        )

    def __len__(self) -> int:
        if self.mode == "train":
            return self.max_train_step
        return len(self.data_queue)

    # ------------------------------------------------------------------ #
    # Audio / text helpers
    # ------------------------------------------------------------------ #

    def _load_audio_to_np(self, x: str):
        x = resolve_path(x)
        audio, sr = librosa.load(x, sr=self.default_sr, mono=True)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        return audio.astype(np.float32), int(sr)

    @torch.inference_mode()
    def _extract_mel(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        if sr != 24000:
            raise ValueError("Only 24kHz audio is supported.")
        mels = mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0),
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2)
        return mels

    def _resolve_target_text(self, item: dict) -> str:
        if "label_norm_text" in item:
            text = item.get("label_norm_text", "")
            text = text.replace("{", "").replace("}", "<end_ins>").replace(": ", "").replace("],[", "][")
        else:
            text = item.get("norm_text", item.get("text", ""))
        text = text.replace("@", "").replace("→", "").replace("¿", "")
        return text

    @staticmethod
    def _item_has_ref_audio(item: dict) -> bool:
        """Check if an item carries its own reference audio field."""
        return bool(
            item.get("const_ref_audio")
            or item.get("dynamic_ref_audio")
            or item.get("ref_audio")
        )

    def _extract_ref_from_item(self, item: dict):
        """Extract ref audio path and text from a single item (flat / self-contained mode).

        Ref audio resolution order:
          1. const_ref_audio (when use_const_ref=True)
          2. dynamic_ref_audio (when use_const_ref=False)
          3. ref_audio (generic fallback)
          4. item's own ``path`` (last resort — same audio as both ref and target)

        Ref text resolution: label_norm_text > ref_text > norm_text > text.
        """
        # --- ref audio ---
        if self.use_const_ref:
            ref_audio = item.get("const_ref_audio") or item.get("dynamic_ref_audio")
        else:
            ref_audio = item.get("dynamic_ref_audio") or item.get("const_ref_audio")

        if not ref_audio:
            ref_audio = item.get("ref_audio")
        if not ref_audio:
            ref_audio = item.get("path")
        if ref_audio:
            ref_audio = resolve_path(ref_audio)

        # --- ref text ---
        ref_text = (
            item.get("label_norm_text")
            or item.get("ref_text")
            or item.get("norm_text")
            or item.get("text")
            or ""
        )
        ref_text = ref_text.replace("@", "").replace("→", "").replace("¿", "")

        return ref_audio, ref_text

    # ------------------------------------------------------------------ #
    # Sample construction
    # ------------------------------------------------------------------ #

    def __getitem__(self, idx: int) -> dict:
        max_retries = 50
        for retry in range(max_retries):
            if len(self.data_queue) == 0:
                self._update_data_queue()
            entry = self.data_queue.pop()
            try:
                if self.flatten:
                    # Flat mode: each item carries its own ref audio
                    sample = self._build_sample_from_flat(entry)
                elif len(entry) == 1:
                    # Single-item group with self-contained ref_audio
                    sample = self._build_sample_from_flat(entry[0])
                else:
                    # Grouped mode: pick target + ref from different items
                    sample = self._build_sample_from_group(entry)

                if sample is not None:
                    return sample
            except Exception as e:
                print(
                    f"[TTSGRPODataset] skip sample (retry {retry + 1}/{max_retries}): "
                    f"{type(e).__name__}: {e}"
                )
                continue
        raise RuntimeError(f"[TTSGRPODataset] Failed after {max_retries} retries.")

    def _build_sample_from_flat(self, item: dict):
        """Build a training sample from a single flat dict with self-contained ref audio."""
        ref_audio_path, ref_text = self._extract_ref_from_item(item)
        target_text = self._resolve_target_text(item)

        if not target_text or not ref_audio_path or not ref_text:
            return None

        audio, sr = self._load_audio_to_np(ref_audio_path)
        ref_mel = self._extract_mel(audio, sr)

        # Add target audio path for GT CPS computation (GDPO 4-dim)
        target_audio_path = resolve_path(item.get("path"))

        return {
            "ref_audio_path": ref_audio_path,
            "ref_text": ref_text,
            "target_text": target_text,
            "target_audio_path": target_audio_path,
            "ref_mel": ref_mel,
        }

    def _build_sample_from_group(self, group: List[dict]):
        """Build a training sample from a same-speaker group (original behavior)."""
        target_item = random.choice(group)
        remaining = [item for item in group if item["path"] != target_item["path"]]
        if not remaining:
            return None
        ref_item = random.choice(remaining)

        ref_audio_path = resolve_path(ref_item["path"])
        ref_text = ref_item.get("norm_text", ref_item.get("text", ""))
        target_text = self._resolve_target_text(target_item)

        if not target_text or not ref_audio_path or not ref_text:
            return None

        audio, sr = self._load_audio_to_np(ref_audio_path)
        ref_mel = self._extract_mel(audio, sr)

        # Add target audio path for GT CPS computation (GDPO 4-dim)
        target_audio_path = resolve_path(target_item["path"])

        return {
            "ref_audio_path": ref_audio_path,
            "ref_text": ref_text,
            "target_text": target_text,
            "target_audio_path": target_audio_path,
            "ref_mel": ref_mel,
        }

    # ------------------------------------------------------------------ #
    # Collation
    # ------------------------------------------------------------------ #

    def collate_fn(self, batch):
        return {"items": list(batch)}
