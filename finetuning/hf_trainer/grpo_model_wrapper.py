# coding=utf-8

from typing import Optional, Tuple

import torch
import torch.nn as nn
from trl.trainer.utils import entropy_from_logits, selective_log_softmax

from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration

from .model_wrapper import _SafeConfigProxy


class _GradScale(torch.autograd.Function):
    """Identity in forward, scales gradient by `scale` in backward.

    Used to throttle (not block) sub_talker GRPO gradient flowing back through
    the shared hidden state into the talker trunk. scale=0.0 ≡ .detach();
    scale=1.0 ≡ no-op (full gradient flow). Intermediate values let trunk
    receive a fraction of the sub_talker signal — analogous to how the sibling
    LoRA-based implementation naturally limits trunk impact via small adapter
    capacity instead of explicit scaling.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output * ctx.scale, None


def _scale_or_detach(x: torch.Tensor, scale: float) -> torch.Tensor:
    if scale <= 0.0:
        return x.detach()
    if scale == 1.0:
        return x
    return _GradScale.apply(x, scale)


class Qwen3TTSForGRPO(nn.Module):
    """Training wrapper for Qwen3-TTS GRPO. DDP wraps this module."""

    def __init__(
        self,
        model: Qwen3TTSForConditionalGeneration,
        freeze_sub_talker: bool = True,
        freeze_speaker_encoder: bool = True,
        sub_talker_aux_loss_weight: float = 0.0,
        sub_talker_trunk_grad_scale: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.sub_talker_aux_loss_weight = sub_talker_aux_loss_weight
        # 0.0 = fully detach hidden before sub_talker forward (safe default,
        # matches v9 attempts 2-4 behavior); >0 lets a fraction of sub_talker
        # GRPO grad shape the trunk via hidden. Combine with C1 KL clamp
        # (±5 nats) and num_iterations=1 to bound risk. See run_grpo.sh
        # header for empirical range guidance.
        self.sub_talker_trunk_grad_scale = float(sub_talker_trunk_grad_scale)

        if freeze_speaker_encoder and self.model.speaker_encoder is not None:
            for p in self.model.speaker_encoder.parameters():
                p.requires_grad = False

        if freeze_sub_talker and getattr(self.model.talker, "code_predictor", None) is not None:
            for p in self.model.talker.code_predictor.parameters():
                p.requires_grad = False

        # HF Trainer expects this attribute for checkpoint loading
        self._keys_to_ignore_on_save = None

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
        # Use get_decoder() so PEFT-wrapped talker resolves correctly
        self.model.talker.get_decoder().gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        self.model.talker.get_decoder().gradient_checkpointing_disable()

    def _build_input_embeddings(
        self,
        input_ids: torch.Tensor,
        codec_ids: torch.Tensor,
        ref_mels,
        text_embedding_mask: torch.Tensor,
        codec_embedding_mask: torch.Tensor,
    ) -> torch.Tensor:
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

        # Use helper methods for PEFT compatibility
        input_text_embedding = m.talker.get_text_embeddings()(input_text_ids)
        input_text_embedding = (
            m.talker.text_projection(input_text_embedding) * text_embedding_mask
        )
        input_codec_embedding = (
            m.talker.get_input_embeddings()(input_codec_ids) * codec_embedding_mask
        )
        input_codec_embedding[:, 6, :] = speaker_embedding

        input_embeddings = input_text_embedding + input_codec_embedding

        num_layers = codec_ids.shape[-1]
        for i in range(1, num_layers):
            codec_i_embedding = m.talker.code_predictor.get_input_embeddings()[i - 1](
                codec_ids[:, :, i]
            )
            input_embeddings = input_embeddings + codec_i_embedding

        return input_embeddings

    def forward_logits(
        self,
        input_ids: torch.Tensor,
        codec_ids: torch.Tensor,
        ref_mels,
        text_embedding_mask: torch.Tensor,
        codec_embedding_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_start_idx: torch.Tensor,
        completion_length: int,
        compute_entropy: bool = False,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute per-token log probabilities on completion positions."""
        m = self.model

        input_embeddings = self._build_input_embeddings(
            input_ids=input_ids,
            codec_ids=codec_ids,
            ref_mels=ref_mels,
            text_embedding_mask=text_embedding_mask,
            codec_embedding_mask=codec_embedding_mask,
        )

        outputs = m.talker(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        logits = outputs.logits  # (B, T, V_codec0)
        batch_size = logits.shape[0]
        device = logits.device

        per_token_logps = []
        entropies = [] if compute_entropy else None
        hidden_last = outputs.hidden_states[0][-1] if return_hidden else None
        per_sample_hidden = [] if return_hidden else None

        codec_layer0 = codec_ids[:, :, 0]

        for b in range(batch_size):
            start = int(completion_start_idx[b].item()) if torch.is_tensor(completion_start_idx) else int(completion_start_idx[b])
            logit_slice = logits[b, start - 1 : start - 1 + completion_length, :]
            target_slice = codec_layer0[b, start : start + completion_length]
            per_token_logps.append(
                selective_log_softmax(logit_slice.unsqueeze(0), target_slice.unsqueeze(0)).squeeze(0)
            )
            if compute_entropy:
                entropies.append(entropy_from_logits(logit_slice.unsqueeze(0)).squeeze(0))
            if return_hidden:
                per_sample_hidden.append(hidden_last[b, start - 1 : start - 1 + completion_length, :])

        per_token_logps = torch.stack(per_token_logps, dim=0)
        if compute_entropy:
            entropies = torch.stack(entropies, dim=0)
        if return_hidden:
            per_sample_hidden = torch.stack(per_sample_hidden, dim=0)

        return per_token_logps, entropies, per_sample_hidden

    def compute_sub_talker_logps(
        self,
        hidden: torch.Tensor,
        codec_ids: torch.Tensor,
        completion_start_idx: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Per-frame summed log-prob over sub_talker layers 1..15.

        hidden: [B, L_max, D] talker last-hidden at completion positions.
        codec_ids: [B, T_full, 16] full codec ids (prompt + completion).
        completion_start_idx: [B] start of completion in T_full.
        completion_mask: [B, L_max] 1 where the frame is real.

        Returns: [B, L_max] per-frame sum_{l=1..15} log P(code_l | hidden, code_<l).
        """
        talker = self.model.talker
        B, L_max, _ = hidden.shape
        device = hidden.device

        hidden_chunks = []
        codec_chunks = []
        valid_per_b = []
        for b in range(B):
            s = int(completion_start_idx[b].item()) if torch.is_tensor(completion_start_idx) else int(completion_start_idx[b])
            cl = int(completion_mask[b].sum().item())
            valid_per_b.append(cl)
            if cl == 0:
                continue
            # Scale (or detach) the hidden gradient before sub_talker forward.
            # scale=0.0 → .detach() (talker trunk fully shielded; v9 attempts
            # 2-4 default). scale>0 → grad * scale flows back into trunk via
            # the shared hidden state — analogous to how the sibling LoRA impl
            # implicitly bounds trunk impact through adapter capacity.
            # Combine with C1 KL clamp + num_iterations=1 + bounded scale
            # (≤0.1 recommended for full-finetune) to keep risk bounded.
            hidden_chunks.append(
                _scale_or_detach(hidden[b, :cl, :], self.sub_talker_trunk_grad_scale)
            )
            codec_chunks.append(codec_ids[b, s : s + cl, :])

        if not hidden_chunks:
            return torch.zeros((B, L_max), device=device, dtype=hidden.dtype)

        hidden_flat = torch.cat(hidden_chunks, dim=0)  # [N, D]
        codec_flat = torch.cat(codec_chunks, dim=0)    # [N, 16]

        num_layers = talker.config.num_code_groups  # 16
        sub_inputs = [hidden_flat.unsqueeze(1)]
        for i in range(num_layers - 1):
            if i == 0:
                emb = talker.get_input_embeddings()(codec_flat[:, :1])
            else:
                emb = talker.code_predictor.get_input_embeddings()[i - 1](codec_flat[:, i : i + 1])
            sub_inputs.append(emb)
        sub_inputs_embeds = torch.cat(sub_inputs, dim=1)  # [N, 16, D]

        sub_out = talker.code_predictor.forward_finetune(inputs_embeds=sub_inputs_embeds)
        sub_logits = sub_out.logits  # [N, 15, V]
        sub_targets = codec_flat[:, 1:]  # [N, 15]
        log_probs = torch.log_softmax(sub_logits.float(), dim=-1)
        per_layer_lp = log_probs.gather(2, sub_targets.unsqueeze(2)).squeeze(2)  # [N, 15]
        per_frame_lp = per_layer_lp.sum(dim=1)  # [N]

        out = torch.zeros((B, L_max), device=device, dtype=per_frame_lp.dtype)
        offset = 0
        for b, cl in enumerate(valid_per_b):
            if cl > 0:
                out[b, :cl] = per_frame_lp[offset : offset + cl]
                offset += cl
        return out

    def forward(self, **kwargs):
        compute_entropy = kwargs.pop("compute_entropy", False)
        return_hidden = kwargs.pop("return_hidden", False)
        compute_sub_talker_logps = kwargs.pop("compute_sub_talker_logps", False)
        completion_mask = kwargs.pop("completion_mask", None)
        completion_start_idx = kwargs.pop("completion_start_idx")
        completion_length = kwargs.pop("completion_length")

        need_hidden = return_hidden or compute_sub_talker_logps
        per_token_logps, entropies, hidden = self.forward_logits(
            input_ids=kwargs["input_ids"],
            codec_ids=kwargs["codec_ids"],
            ref_mels=kwargs["ref_mels"],
            text_embedding_mask=kwargs["text_embedding_mask"],
            codec_embedding_mask=kwargs["codec_embedding_mask"],
            attention_mask=kwargs["attention_mask"],
            completion_start_idx=completion_start_idx,
            completion_length=completion_length,
            compute_entropy=compute_entropy,
            return_hidden=need_hidden,
        )

        sub_talker_logps = None
        if compute_sub_talker_logps:
            if completion_mask is None:
                raise ValueError("compute_sub_talker_logps=True requires completion_mask")
            sub_talker_logps = self.compute_sub_talker_logps(
                hidden=hidden,
                codec_ids=kwargs["codec_ids"],
                completion_start_idx=completion_start_idx,
                completion_mask=completion_mask,
            )

        return {
            "per_token_logps": per_token_logps,
            "entropies": entropies,
            "hidden": hidden if return_hidden else None,
            "sub_talker_logps": sub_talker_logps,
        }
