# coding=utf-8

from typing import Dict, List

import torch
import torch.nn as nn
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration


class _SafeConfigProxy:
    """Proxy around Qwen3TTSConfig that prevents TensorBoard serialization crashes."""

    def __init__(self, config):
        self._config = config

    def __getattr__(self, name):
        return getattr(self._config, name)

    def to_diff_dict(self):
        return {"model_type": self._config.model_type}

    def to_json_string(self, use_diff=True):
        import json
        return json.dumps(self.to_diff_dict(), indent=2)


class Qwen3TTSForSFT(nn.Module):
    """
    Training wrapper for Qwen3-TTS that encapsulates the complete forward pass.

    DDP wraps this module, so its forward() is called through the DDP wrapper,
    ensuring proper gradient synchronization across GPUs.
    """

    def __init__(
        self,
        model: Qwen3TTSForConditionalGeneration,
        sub_talker_loss_weight: float = 0.1,
    ):
        super().__init__()
        self.model = model
        self.sub_talker_loss_weight = sub_talker_loss_weight

        if self.model.speaker_encoder is not None:
            for p in self.model.speaker_encoder.parameters():
                p.requires_grad = False

    @property
    def config(self):
        return _SafeConfigProxy(self.model.config)

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        # Use get_decoder() instead of `.model` so PEFT-wrapped talker resolves correctly:
        # under PEFT, `talker.model` falls through __getattr__ to the original talker
        # (one level too shallow), while get_decoder() reaches the inner text decoder.
        self.model.talker.get_decoder().gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        self.model.talker.get_decoder().gradient_checkpointing_disable()

    def forward(
        self,
        input_ids: torch.Tensor,
        codec_ids: torch.Tensor,
        ref_mels: List[torch.Tensor],
        text_embedding_mask: torch.Tensor,
        codec_embedding_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        codec_0_labels: torch.Tensor,
        codec_mask: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        m = self.model
        device = self.device
        dtype = self.dtype

        with torch.no_grad():
            speaker_embedding = []
            for mel in ref_mels:
                speaker_embedding.append(
                    m.speaker_encoder(mel.to(device).to(dtype)).detach()
                )
            speaker_embedding = torch.cat(speaker_embedding, dim=0)

        input_text_ids = input_ids[:, :, 0]
        input_codec_ids = input_ids[:, :, 1]

        # Use helpers (get_text_embeddings / get_input_embeddings) instead of `.model.<emb>`
        # so the access works both with and without PEFT wrapping on talker.
        input_text_embedding = m.talker.get_text_embeddings()(input_text_ids)
        input_text_embedding = (
            m.talker.text_projection(input_text_embedding) * text_embedding_mask
        )
        input_codec_embedding = (
            m.talker.get_input_embeddings()(input_codec_ids) * codec_embedding_mask
        )
        input_codec_embedding[:, 6, :] = speaker_embedding

        input_embeddings = input_text_embedding + input_codec_embedding

        for i in range(1, 16):
            codec_i_embedding = m.talker.code_predictor.get_input_embeddings()[i - 1](
                codec_ids[:, :, i]
            )
            codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
            input_embeddings = input_embeddings + codec_i_embedding

        outputs = m.talker(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            labels=codec_0_labels,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states[0][-1]
        target_codec_mask = codec_mask[:, 1:]
        talker_hidden_states = hidden_states[:, :-1][target_codec_mask]
        talker_codec_ids = codec_ids[:, 1:][target_codec_mask]

        _, sub_talker_loss = m.talker.forward_sub_talker_finetune(
            talker_codec_ids, talker_hidden_states
        )

        loss = outputs.loss + self.sub_talker_loss_weight * sub_talker_loss

        return {
            "loss": loss,
            "talker_loss": outputs.loss.detach(),
            "sub_talker_loss": sub_talker_loss.detach(),
        }
