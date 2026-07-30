# coding=utf-8

from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import Trainer
from transformers.utils import logging

from ._save_utils import (
    bundle_inference_aux,
    get_inner_talker,
    merge_peft_if_present,
    strip_peft_keys,
)

logger = logging.get_logger(__name__)


class TTSTrainer(Trainer):
    """
    Custom Trainer for Qwen3-TTS SFT training.

    Handles:
    - Non-standard batch format (ref_mels is a list of tensors, not a stacked tensor)
    - Custom loss computation via the model wrapper's forward()
    - Logging of sub_talker_loss as an additional metric
    - Bundling inference-required aux files into each checkpoint dir
    - Skipping DistributedSampler when dataset already splits data per rank
    """

    def __init__(self, *args, init_model_path: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_model_path = init_model_path

    def get_train_dataloader(self) -> DataLoader:
        dataset = self.train_dataset
        if hasattr(dataset, '_sample_split') and dataset._sample_split:
            sampler = RandomSampler(dataset)
            return DataLoader(
                dataset,
                batch_size=self.args.per_device_train_batch_size,
                sampler=sampler,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                drop_last=self.args.dataloader_drop_last,
            )
        return super().get_train_dataloader()

    def compute_loss(
        self,
        model,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        outputs = model(**inputs)
        loss = outputs["loss"]

        if self.state.global_step % self.args.logging_steps == 0:
            self._tts_metrics = {
                "talker_loss": outputs["talker_loss"].item(),
                "sub_talker_loss": outputs["sub_talker_loss"].item(),
            }

        if return_outputs:
            return loss, outputs
        return loss

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        if hasattr(self, "_tts_metrics"):
            logs.update(self._tts_metrics)
            del self._tts_metrics
        super().log(logs, start_time)

    def _save(self, output_dir: Optional[str] = None, state_dict=None) -> None:
        full_model = self.accelerator.unwrap_model(self.model)
        talker = get_inner_talker(full_model)
        with merge_peft_if_present(talker) as is_peft:
            if is_peft and state_dict is None:
                state_dict = strip_peft_keys(full_model.state_dict())
            super()._save(output_dir=output_dir, state_dict=state_dict)
        bundle_inference_aux(self._init_model_path, output_dir or self.args.output_dir)
