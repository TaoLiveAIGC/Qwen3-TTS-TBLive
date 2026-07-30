# coding=utf-8
"""Shared helpers for checkpoint saving.

  * bundle_inference_aux:  copy/symlink config + tokenizer files into each ckpt dir
  * merge_peft_if_present: context manager that merges LoRA adapters in place
  * strip_peft_keys:       clean a state_dict so it loads directly into the base model
  * get_inner_talker:      drill into wrapper.model.talker
"""

import contextlib
import shutil
from pathlib import Path
from typing import Dict, Optional

from transformers.utils import logging

logger = logging.get_logger(__name__)

# Small text/JSON files copied per checkpoint.
_AUX_FILES = (
    "config.json",
    "configuration.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
)
# Large subdirs (codec weights, etc.) — symlinked instead of copied to avoid
# multiplying disk usage by save_total_limit.
_AUX_DIRS = ("speech_tokenizer",)


def bundle_inference_aux(init_model_path: Optional[str], ckpt_dir: str) -> None:
    """Copy small aux files and symlink large aux dirs from init_model_path → ckpt_dir.

    No-op if init_model_path is None, not a local path, or doesn't exist.
    """
    if not init_model_path:
        return
    src = Path(init_model_path)
    dst = Path(ckpt_dir)
    if not src.exists() or not dst.exists():
        return
    for fname in _AUX_FILES:
        src_f, dst_f = src / fname, dst / fname
        if src_f.exists() and not dst_f.exists():
            try:
                shutil.copy2(src_f, dst_f)
            except Exception as e:
                logger.warning(f"[bundle_inference_aux] copy {fname} failed: {e}")
    for dname in _AUX_DIRS:
        src_d, dst_d = src / dname, dst / dname
        if src_d.exists() and not dst_d.exists():
            try:
                dst_d.symlink_to(src_d.resolve(), target_is_directory=True)
            except Exception as e:
                logger.warning(f"[bundle_inference_aux] symlink {dname} failed: {e}")


# PEFT adapter key markers. Bracketed by dots so we don't accidentally match user keys.
_PEFT_KEY_MARKERS = (
    ".lora_A.",
    ".lora_B.",
    ".lora_embedding_A.",
    ".lora_embedding_B.",
    ".lora_magnitude_vector.",
)


def get_inner_talker(model):
    """Return wrapper.model.talker if accessible, else None."""
    inner = getattr(model, "model", model)
    return getattr(inner, "talker", None)


@contextlib.contextmanager
def merge_peft_if_present(talker):
    """Merge LoRA adapters in place for the lifetime of the block, then unmerge.

    Yields True if a merge happened, False otherwise. Safe to use on non-PEFT modules.
    """
    is_peft = (
        talker is not None
        and hasattr(talker, "merge_adapter")
        and hasattr(talker, "unmerge_adapter")
    )
    if is_peft:
        talker.merge_adapter()
    try:
        yield is_peft
    finally:
        if is_peft:
            talker.unmerge_adapter()


def strip_peft_keys(state_dict: Dict) -> Dict:
    """Strip ALL PEFT-injected naming so the result loads directly into the base model.

    Two transformations:
      1. Drop `*.lora_A./.lora_B./...` adapter params (already merged into base_layer).
      2. Rename `*.base_layer.{weight,bias}` → `*.{weight,bias}` (per-Linear LoRA wrap).
      3. Strip `.base_model.model.` wrapper prefix that PEFT inserts between the parent
         module and the original wrapped module (e.g. `talker.base_model.model.<X>`
         → `talker.<X>`). Without this, from_pretrained sees all talker keys as
         "unused" and silently re-initialises the full talker with random weights.

    Use AFTER merge_adapter() so base_layer.weight already holds merged W + B@A.
    """
    clean = {}
    for k, v in state_dict.items():
        if any(marker in k for marker in _PEFT_KEY_MARKERS):
            continue
        new_k = (
            k.replace(".base_layer.weight", ".weight")
             .replace(".base_layer.bias", ".bias")
             .replace(".base_model.model.", ".")
        )
        clean[new_k] = v
    return clean
