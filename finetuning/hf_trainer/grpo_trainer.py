# coding=utf-8

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Trainer
from transformers.utils import logging
from trl.trainer.utils import RepeatSampler

from ._save_utils import (
    bundle_inference_aux,
    get_inner_talker,
    merge_peft_if_present,
    strip_peft_keys,
)
from .grpo_config import GRPOTTSConfig
from .grpo_rollout import TTSGRPORolloutEngine
from .prosody_rewards import gdpo_4dim_aggregate

logger = logging.get_logger(__name__)


class Qwen3TTSGRPOTrainer(Trainer):
    """GRPO trainer for Qwen3-TTS using the SSIM-based reward."""

    def __init__(
        self,
        model,
        args: GRPOTTSConfig,
        train_dataset,
        rollout_engine: TTSGRPORolloutEngine,
        ref_model: Optional[nn.Module] = None,
        data_collator=None,
        init_model_path: Optional[str] = None,
        **kwargs,
    ):
        if args.per_device_train_batch_size % args.num_generations != 0:
            raise ValueError(
                f"per_device_train_batch_size ({args.per_device_train_batch_size}) must be a multiple of "
                f"num_generations ({args.num_generations}); each per-device batch must contain whole groups "
                f"of G completions so group-relative advantages can be computed."
            )
        if args.gradient_accumulation_steps % args.steps_per_generation != 0 and args.steps_per_generation > 1:
            raise ValueError(
                f"gradient_accumulation_steps ({args.gradient_accumulation_steps}) must be a multiple of "
                f"steps_per_generation ({args.steps_per_generation}) to keep generation/optimizer steps aligned."
            )
        self.rollout_engine = rollout_engine
        self.ref_model = ref_model
        self.num_generations = args.num_generations
        self.epsilon = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        self.beta = args.beta
        self.loss_type = args.loss_type
        self.scale_rewards = args.scale_rewards
        self.importance_sampling_level = args.importance_sampling_level
        self.steps_per_generation = args.steps_per_generation
        self.num_iterations = args.num_iterations
        self.max_completion_length = args.max_completion_length
        self.mask_truncated_completions = args.mask_truncated_completions
        self.sub_talker_aux_loss_weight = args.sub_talker_aux_loss_weight
        self.grpo_sub_talker_weight = args.grpo_sub_talker_weight
        self.sub_importance_sampling_level = args.sub_importance_sampling_level
        self.sub_talker_mode = getattr(args, 'sub_talker_mode', 'ppo')

        self._tts_metrics: Dict[str, float] = {}
        self._rollout_step = 0
        self._buffered_inputs: Optional[List[Dict[str, Any]]] = None
        self._buffer_index = 0
        self._init_model_path = init_model_path

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            **kwargs,
        )
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        self.rollout_engine.unwrap_fn = self.accelerator.unwrap_model

    def get_train_dataloader(self):
        per_device = self.args.per_device_train_batch_size
        generation_batch = per_device * self.steps_per_generation
        batch_size_unique = max(1, generation_batch // self.num_generations)

        sampler = RepeatSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=batch_size_unique,
            repeat_count=self.num_iterations * self.steps_per_generation,
            shuffle=True,
            seed=self.args.seed,
        )

        loader = DataLoader(
            self.train_dataset,
            batch_size=generation_batch,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            drop_last=True,
            pin_memory=self.args.dataloader_pin_memory,
        )
        return self.accelerator.prepare(loader)

    def _talker_config(self):
        return self.accelerator.unwrap_model(self.model).model.config

    def _assemble_inputs(self, rollout: Dict[str, Any]) -> Dict[str, Any]:
        """Convert rollout output into the padded batch for GRPO forward pass.

        Builds a batch of ICL-mode training sequences, one per candidate.
        Each sequence has the following layout (left-aligned, right-padded with zeros):

        Position  | 0-2       | 3-7         | 8 .. 8+tl-4       | 8+tl-3 | 8+tl-2 | 8+tl-1 .. 8+tl-1+N_ref-1 | 8+tl-1+N_ref .. 8+tl+cl-1 | 8+tl+cl
        Text ch   | role(3)   | tts_pad(4)+ | ref_text+         | tts_eos| tts_pad| tts_pad * N_ref         | tts_pad * N_gen           | tts_pad ...
                  |           | tts_bos     | target_text       |        |        |                         |                           |
        Codec ch  | 0,0,0     | nothink,    | codec_pad *       | codec_ | codec_ | ref_code[:,0] * N_ref  | gen_code[:,0] * N_gen     | codec_eos
                  |           | think_bos,  | (tl-3)            | pad    | bos    |                         |                           |
                  |           | think_eos,  |                   |        |        |                         |                           |
                  |           | SPK(0),     |                   |        |        |                         |                           |
                  |           | codec_pad   |                   |        |        |                         |                           |

        codec_ids[:, :, 1:16] stores code layers 1-15 at the same codec positions,
        used by sub_talker for teacher-forcing sum embedding.

        Key outputs:
            completion_start_idx: where generated codec starts (after ref_codec),
                used by forward_logits to extract log-probs for gen portion only.
            completion_mask: [B, max_comp_len], 1 at valid gen positions.
        """
        cfg = self._talker_config()
        text_ids_list = rollout["combined_text_ids"]   # [1, tl] tokenized "ref_text+target_text"
        completion_list = rollout["completion_codec_ids"]  # [N_gen, 16] generated codec frames
        ref_codes_list = rollout["ref_codes"]          # [N_ref, 16] reference audio codec
        ref_mels = rollout["ref_mels"]

        B = len(text_ids_list)
        text_lens = [int(t.shape[1]) for t in text_ids_list]       # tl: total text tokens (role + content)
        ref_lens = [int(r.shape[0]) for r in ref_codes_list]       # N_ref: reference codec frames
        comp_lens = [int(c.shape[0]) for c in completion_list]     # N_gen: generated codec frames
        codec_total = [r + c for r, c in zip(ref_lens, comp_lens)] # N_ref + N_gen
        sample_len = [8 + tl + cl for tl, cl in zip(text_lens, codec_total)]  # role(3)+prefix(5)+text+codec

        # Pad length T: must be large enough so that for every sample b,
        # start_b + completion_length <= T, where start_b = 8 + tl_b - 1 + N_ref_b.
        # Otherwise forward_logits' fixed-length slice [start-1 : start-1+L] gets
        # silently truncated to different lengths per b, breaking torch.stack.
        needed_T = 7 + max(t + r for t, r in zip(text_lens, ref_lens)) + max(comp_lens)
        T = max(max(sample_len), needed_T)

        # Allocate zero-padded tensors (left-aligned, right-padded)
        input_ids = torch.zeros((B, T, 2), dtype=torch.long)       # [:,:,0]=text channel, [:,:,1]=codec channel
        codec_ids = torch.zeros((B, T, 16), dtype=torch.long)      # full 16-layer codec for sub_talker embedding
        text_embedding_mask = torch.zeros((B, T), dtype=torch.bool)  # where text embedding is active
        codec_embedding_mask = torch.zeros((B, T), dtype=torch.bool) # where codec embedding is active
        attention_mask = torch.zeros((B, T), dtype=torch.long)       # 1 at valid positions
        completion_mask = torch.zeros((B, max(comp_lens)), dtype=torch.long)  # 1 at gen positions
        completion_start_idx = torch.zeros((B,), dtype=torch.long)   # offset where gen codec begins

        for i in range(B):
            text_ids = text_ids_list[i]   # [1, tl]: tokenized "role + ref_text + target_text"
            tl = text_lens[i]
            ref_code = ref_codes_list[i]  # [N_ref, 16]
            comp = completion_list[i]     # [N_gen, 16]
            # Concatenate ref + generated codec: ref_codec goes first (ICL conditioning),
            # then generated codec. Both participate in the forward pass as teacher-forcing input.
            audio_codes = torch.cat([ref_code, comp], dim=0)  # [N_ref+N_gen, 16]
            cl = audio_codes.shape[0]

            # ---- Text channel (input_ids[:,:,0]) ----
            # Positions 0-2: role tokens (im_start, assistant, \n) from tokenizer
            input_ids[i, :3, 0] = text_ids[0, :3]
            # Positions 3-6: tts_pad (aligns with codec prefix: nothink/think_bos/think_eos/SPK)
            input_ids[i, 3:7, 0] = cfg.tts_pad_token_id
            # Position 7: tts_bos (marks start of text content in codec-aligned position)
            input_ids[i, 7, 0] = cfg.tts_bos_token_id
            # Positions 8 .. 8+tl-4: text content (ref_text + target_text, role already placed)
            input_ids[i, 8 : 8 + tl - 3, 0] = text_ids[0, 3:]
            # Position 8+tl-3: tts_eos (marks end of text content)
            input_ids[i, 8 + tl - 3, 0] = cfg.tts_eos_token_id
            # Positions 8+tl-2 onwards: tts_pad (codec area — text channel is pad)
            input_ids[i, 8 + tl - 2 : 8 + tl + cl, 0] = cfg.tts_pad_token_id
            text_embedding_mask[i, : 8 + tl + cl] = True

            # ---- Codec channel (input_ids[:,:,1]) — code layer 0 ----
            # Positions 3-7: codec prefix [nothink, think_bos, think_eos, SPK_placeholder, codec_pad]
            # Position 6 (SPK) is a placeholder; speaker embedding is injected separately
            # in _build_input_embeddings via codec_embedding_mask[i, 6] = False.
            input_ids[i, 3:8, 1] = torch.tensor(
                [
                    cfg.talker_config.codec_nothink_id,
                    cfg.talker_config.codec_think_bos_id,
                    cfg.talker_config.codec_think_eos_id,
                    0,  # placeholder for speaker embedding (not a real token)
                    cfg.talker_config.codec_pad_id,
                ]
            )
            # Positions 8 .. 8+tl-4: codec_pad (aligned with text content)
            input_ids[i, 8 : 8 + tl - 3, 1] = cfg.talker_config.codec_pad_id
            # Position 8+tl-3: codec_pad (aligned with tts_eos)
            input_ids[i, 8 + tl - 3, 1] = cfg.talker_config.codec_pad_id
            # Position 8+tl-2: codec_bos (marks start of codec area)
            input_ids[i, 8 + tl - 2, 1] = cfg.talker_config.codec_bos_id
            # Positions 8+tl-1 .. 8+tl-1+cl-1: ref_codec + gen_codec (layer 0 tokens)
            input_ids[i, 8 + tl - 1 : 8 + tl - 1 + cl, 1] = audio_codes[:, 0]
            # Position 8+tl-1+cl: codec_eos
            input_ids[i, 8 + tl - 1 + cl, 1] = cfg.talker_config.codec_eos_token_id

            # ---- Full 16-layer codec ids for sub_talker embedding sum ----
            # Same positions as layer 0, but stores all 16 code layers.
            # In _build_input_embeddings, layers 1-15 are looked up via
            # code_predictor.get_input_embeddings()[i-1] and summed into the main embedding.
            codec_ids[i, 8 + tl - 1 : 8 + tl - 1 + cl, :] = audio_codes

            # ---- Masks ----
            codec_embedding_mask[i, 3 : 8 + tl + cl] = True
            codec_embedding_mask[i, 6] = False  # SPK position: embedding injected separately

            attention_mask[i, : 8 + tl + cl] = 1

            # ---- Completion tracking ----
            # completion_start_idx: where generated codec begins (= codec_bos + 1 + N_ref)
            # forward_logits uses this to extract log-probs for gen portion only,
            # skipping the ref_codec conditioning prefix.
            completion_start_idx[i] = 8 + tl - 1 + ref_lens[i]
            completion_mask[i, : comp_lens[i]] = 1

        return {
            "input_ids": input_ids,
            "codec_ids": codec_ids,
            "ref_mels": ref_mels,
            "text_embedding_mask": text_embedding_mask.unsqueeze(-1),
            "codec_embedding_mask": codec_embedding_mask.unsqueeze(-1),
            "attention_mask": attention_mask,
            "completion_start_idx": completion_start_idx,
            "completion_length": max(comp_lens),
            "completion_mask": completion_mask,
        }

    def _move_inputs_to_device(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        device = self.accelerator.device
        moved = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                moved[k] = v.to(device)
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                moved[k] = [t.to(device) for t in v]
            else:
                moved[k] = v
        return moved

    def _generate_and_score_completions(self, generation_batch) -> List[Dict[str, Any]]:
        items = generation_batch.get("items") if isinstance(generation_batch, dict) else generation_batch
        rollout = self.rollout_engine.rollout(items)

        # Detect 4-dim GDPO mode: when w_cps > 0 or w_semi_fl > 0
        use_4dim = (getattr(self.args, 'w_cps', 0) > 0 or
                    getattr(self.args, 'w_semi_fl', 0) > 0)

        if use_4dim:
            # 4-dim GDPO: batch-level z-score over {cer, sim, cps, semi_fl}
            G = self.num_generations
            B = len(rollout["target_text"])
            n_prompts = B // G

            # Build per-prompt sample details for gdpo_4dim_aggregate
            sample_details = []
            for p in range(n_prompts):
                start_idx = p * G
                end_idx = start_idx + G
                sample_details.append({
                    "cers": rollout["cer"][start_idx:end_idx].tolist(),
                    "sim_rewards": rollout["ssim"][start_idx:end_idx].tolist(),
                    "cps_vals": rollout["cps"][start_idx:end_idx].tolist(),
                    "semi_fl_vals": rollout["semi_fl"][start_idx:end_idx].tolist(),
                    "gt_cps": float(rollout["gt_cps"][start_idx]),
                })

            # Compute advantages via batch-level z-score
            weights = {
                "cer": getattr(self.args, 'w_cer', 1.0),
                "sim": getattr(self.args, 'w_sim', 1.0),
                "cps": getattr(self.args, 'w_cps', 0.0),
                "semi_fl": getattr(self.args, 'w_semi_fl', 0.0),
            }
            advantages_list, dim_stats = gdpo_4dim_aggregate(
                sample_details,
                weights=weights,
                cps_deadzone_low=getattr(self.args, 'cps_deadzone_low', 0.05),
                cps_deadzone_high=getattr(self.args, 'cps_deadzone_high', 0.10),
                cer_deadzone=getattr(self.args, 'cer_deadzone', 0.03),
                cer_exp_k=getattr(self.args, 'cer_exp_k', 3.0),
            )
            advantages = torch.tensor([adv for group in advantages_list for adv in group],
                                     dtype=torch.float32, device=self.accelerator.device)

            # For backward compatibility, compute a pseudo-rewards tensor (weighted sum of raw dims)
            # This is used for metrics logging only, not for advantage computation
            rewards = (
                weights["cer"] * (1.0 - rollout["cer"]) +
                weights["sim"] * rollout["ssim"] +
                weights["cps"] * rollout["cps"] +
                weights["semi_fl"] * rollout["semi_fl"]
            ).to(self.accelerator.device)
        else:
            # 2-dim backward-compatible mode: per-group standardization
            rewards = rollout["rewards"].to(self.accelerator.device)
            G = self.num_generations
            reshaped = rewards.view(-1, G)
            mean_g = reshaped.mean(dim=1, keepdim=True)
            if G > 1:
                std_g = reshaped.std(dim=1, keepdim=True)
            else:
                std_g = torch.zeros_like(mean_g)
            advantages = (reshaped - mean_g) / (std_g + 1e-8)
            advantages = advantages.view(-1)

        assembled = self._assemble_inputs(rollout)
        assembled["advantages"] = advantages

        moved = self._move_inputs_to_device(assembled)

        wrapper = self.model
        wrapper.eval()
        sub_grpo_on = self.grpo_sub_talker_weight > 0
        sub_is_ppo = sub_grpo_on and self.sub_talker_mode == 'ppo'
        with torch.no_grad():
            out = wrapper(
                input_ids=moved["input_ids"],
                codec_ids=moved["codec_ids"],
                ref_mels=moved["ref_mels"],
                text_embedding_mask=moved["text_embedding_mask"],
                codec_embedding_mask=moved["codec_embedding_mask"],
                attention_mask=moved["attention_mask"],
                completion_start_idx=moved["completion_start_idx"],
                completion_length=moved["completion_length"],
                completion_mask=moved["completion_mask"] if sub_grpo_on else None,
                compute_entropy=False,
                return_hidden=False,
                compute_sub_talker_logps=sub_grpo_on,
            )
            moved["old_per_token_logps"] = out["per_token_logps"].detach()
            if sub_is_ppo:
                moved["old_sub_talker_logps"] = out["sub_talker_logps"].detach()

            # Compute reference model log-probs
            if self.beta != 0.0:
                # Check if using LoRA (talker has disable_adapter_layers)
                inner_talker = self.accelerator.unwrap_model(wrapper).model.talker
                use_lora_for_ref = hasattr(inner_talker, 'disable_adapter_layers')

                if use_lora_for_ref:
                    # LoRA mode: disable adapter to get base model behavior
                    inner_talker.disable_adapter_layers()
                    try:
                        ref_out = wrapper(
                            input_ids=moved["input_ids"],
                            codec_ids=moved["codec_ids"],
                            ref_mels=moved["ref_mels"],
                            text_embedding_mask=moved["text_embedding_mask"],
                            codec_embedding_mask=moved["codec_embedding_mask"],
                            attention_mask=moved["attention_mask"],
                            completion_start_idx=moved["completion_start_idx"],
                            completion_length=moved["completion_length"],
                            completion_mask=moved["completion_mask"] if sub_grpo_on else None,
                            compute_entropy=False,
                            return_hidden=False,
                            compute_sub_talker_logps=sub_grpo_on,
                        )
                        moved["ref_per_token_logps"] = ref_out["per_token_logps"].detach()
                        if sub_grpo_on:
                            moved["ref_sub_talker_logps"] = ref_out["sub_talker_logps"].detach()
                    finally:
                        inner_talker.enable_adapter_layers()
                elif self.ref_model is not None:
                    # Full fine-tuning mode: use separate ref_model
                    ref_unwrapped = self.accelerator.unwrap_model(self.ref_model)
                    ref_out = self._ref_forward_logits(ref_unwrapped, moved, with_sub_talker=sub_grpo_on)
                    moved["ref_per_token_logps"] = ref_out["per_token_logps"].detach()
                    if sub_grpo_on:
                        moved["ref_sub_talker_logps"] = ref_out["sub_talker_logps"].detach()
        wrapper.train()

        ssim_arr = rollout["ssim"]
        cer_arr = rollout["cer"]

        rollout_metrics = {
            "reward/mean": float(rewards.mean().item()),
            "reward/std": float(rewards.std().item()) if rewards.numel() > 1 else 0.0,
            "ssim/mean": float(ssim_arr.mean().item()),
            "cer/mean": float(cer_arr.mean().item()),
            "avg_attempts": float(rollout["avg_attempts"]),
        }

        # CER drift early-warning indicators
        try:
            cer_np = cer_arr.float().numpy()
            rollout_metrics["cer/p50"] = float(np.percentile(cer_np, 50))
            rollout_metrics["cer/p95"] = float(np.percentile(cer_np, 95))
            rollout_metrics["cer/over_010"] = float((cer_arr > 0.10).float().mean().item())
        except Exception:
            pass

        # 4-dim GDPO extras: CPS / SemiFL raw stats + per-dim z-score stats
        if use_4dim:
            for name in ("cps", "semi_fl", "gt_cps"):
                arr = rollout.get(name)
                if arr is None or arr.numel() == 0:
                    continue
                rollout_metrics[f"{name}/mean"] = float(arr.mean().item())
                if arr.numel() > 1:
                    rollout_metrics[f"{name}/std"] = float(arr.std().item())
            for dim, (m, s) in dim_stats.items():
                rollout_metrics[f"dim_stats/{dim}_mean"] = float(m)
                rollout_metrics[f"dim_stats/{dim}_std"] = float(s)

        # Per-group advantage spread (how distinct is anchor from random)
        try:
            G = self.num_generations
            adv_grouped = advantages.detach().cpu().view(-1, G)
            adv_range = (adv_grouped.max(dim=1).values - adv_grouped.min(dim=1).values).mean()
            rollout_metrics["advantage/range"] = float(adv_range.item())
            rollout_metrics["advantage/max_mean"] = float(adv_grouped.max(dim=1).values.mean().item())
            rollout_metrics["advantage/min_mean"] = float(adv_grouped.min(dim=1).values.mean().item())
        except Exception:
            pass

        self._tts_metrics.update(rollout_metrics)

        # Log detailed per-prompt metrics for rank 0
        if self.accelerator.is_main_process and self.state.global_step % self.args.logging_steps == 0:
            try:
                G = self.num_generations
                B = len(rollout["target_text"])  # Total samples = prompts * G
                n_prompts = B // G

                print(f"\n[Rank 0] Step {self.state.global_step} - Detailed sampling metrics:")

                for p in range(min(n_prompts, 5)):  # Show first 5 prompts
                    start_idx = p * G
                    end_idx = start_idx + G

                    target_text = rollout['target_text'][start_idx]  # Full text
                    ssim_group = [f"{rollout['ssim'][i].item():.3f}" for i in range(start_idx, end_idx)]
                    cer_group = [f"{rollout['cer'][i].item():.3f}" for i in range(start_idx, end_idx)]
                    adv_group = [f"{advantages[i].item():+.3f}" for i in range(start_idx, end_idx)]

                    print(f"  [Prompt {p}] text: {target_text}")
                    print(f"    SSIM:      [{', '.join(ssim_group)}]")
                    print(f"    CER:       [{', '.join(cer_group)}]")
                    if use_4dim:
                        cps_group = [f"{rollout['cps'][i].item():.2f}" for i in range(start_idx, end_idx)]
                        semi_fl_group = [f"{rollout['semi_fl'][i].item():.2f}" for i in range(start_idx, end_idx)]
                        gt_cps_val = rollout['gt_cps'][start_idx].item()
                        print(f"    CPS:       [{', '.join(cps_group)}]  (gt={gt_cps_val:.2f})")
                        print(f"    SemiFL:    [{', '.join(semi_fl_group)}]")
                    print(f"    Advantage: [{', '.join(adv_group)}]")
                    print()

                if n_prompts > 5:
                    print(f"  ... ({n_prompts - 5} more prompts)")
                print()
            except Exception as e:
                print(f"[Rank 0] Failed to log detailed metrics: {e}")

        per_device = self.args.per_device_train_batch_size
        slices: List[Dict[str, Any]] = []
        for s in range(self.steps_per_generation):
            start = s * per_device
            end = start + per_device
            sliced: Dict[str, Any] = {}
            for k, v in moved.items():
                if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == rewards.shape[0]:
                    sliced[k] = v[start:end]
                elif isinstance(v, list) and len(v) == rewards.shape[0]:
                    sliced[k] = v[start:end]
                else:
                    sliced[k] = v
            slices.append(sliced)
        return slices

    def _ref_forward_logits(
        self,
        ref_model,
        inputs: Dict[str, Any],
        with_sub_talker: bool = False,
    ) -> Dict[str, torch.Tensor]:
        from .grpo_model_wrapper import Qwen3TTSForGRPO

        if not isinstance(ref_model, Qwen3TTSForGRPO):
            tmp = Qwen3TTSForGRPO(ref_model)
        else:
            tmp = ref_model
        out = tmp(
            input_ids=inputs["input_ids"],
            codec_ids=inputs["codec_ids"],
            ref_mels=inputs["ref_mels"],
            text_embedding_mask=inputs["text_embedding_mask"],
            codec_embedding_mask=inputs["codec_embedding_mask"],
            attention_mask=inputs["attention_mask"],
            completion_start_idx=inputs["completion_start_idx"],
            completion_length=inputs["completion_length"],
            completion_mask=inputs["completion_mask"] if with_sub_talker else None,
            compute_entropy=False,
            return_hidden=False,
            compute_sub_talker_logps=with_sub_talker,
        )
        return {
            "per_token_logps": out["per_token_logps"],
            "sub_talker_logps": out["sub_talker_logps"] if with_sub_talker else None,
        }

    def _prepare_inputs(self, generation_batch):
        period = self.steps_per_generation * self.num_iterations
        if self._buffered_inputs is None or self._buffer_index >= period:
            self._buffered_inputs = self._generate_and_score_completions(generation_batch)
            self._buffer_index = 0
            self._rollout_step += 1
        slice_idx = self._buffer_index % self.steps_per_generation
        cur = self._buffered_inputs[slice_idx]
        self._buffer_index += 1
        return cur

    def _reduce_masked_loss(self, masked_loss: torch.Tensor, completion_mask: torch.Tensor, num_items_in_batch=None) -> torch.Tensor:
        """Apply self.loss_type reduction to a [B, L] masked per-token loss."""
        if self.loss_type == "grpo":
            denom = completion_mask.sum(-1).clamp(min=1)
            return (masked_loss.sum(-1) / denom).mean()
        if self.loss_type == "bnpo":
            return masked_loss.sum() / completion_mask.sum().clamp(min=1)
        if self.loss_type == "dr_grpo":
            B = masked_loss.shape[0]
            return masked_loss.sum() / max(B * self.max_completion_length, 1)
        if self.loss_type == "dapo":
            denom = num_items_in_batch
            if denom is None or (isinstance(denom, torch.Tensor) and denom.numel() == 0):
                denom = completion_mask.sum().clamp(min=1)
            elif isinstance(denom, torch.Tensor):
                denom = denom.to(masked_loss.dtype).clamp(min=1)
            else:
                denom = max(float(denom), 1.0)
            return masked_loss.sum() / denom
        raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        sub_grpo_on = self.grpo_sub_talker_weight > 0
        return_hidden = self.sub_talker_aux_loss_weight > 0  # legacy aux-loss path
        out = model(
            input_ids=inputs["input_ids"],
            codec_ids=inputs["codec_ids"],
            ref_mels=inputs["ref_mels"],
            text_embedding_mask=inputs["text_embedding_mask"],
            codec_embedding_mask=inputs["codec_embedding_mask"],
            attention_mask=inputs["attention_mask"],
            completion_start_idx=inputs["completion_start_idx"],
            completion_length=inputs["completion_length"],
            completion_mask=inputs["completion_mask"] if sub_grpo_on else None,
            compute_entropy=True,
            return_hidden=return_hidden,
            compute_sub_talker_logps=sub_grpo_on,
        )
        per_token_logps = out["per_token_logps"]
        entropies = out["entropies"]

        completion_mask = inputs["completion_mask"].to(per_token_logps.dtype)
        advantages = inputs["advantages"].to(per_token_logps.dtype).unsqueeze(-1)
        old_logps = inputs.get("old_per_token_logps", per_token_logps.detach())
        old_logps = old_logps.to(per_token_logps.dtype)

        log_ratio = per_token_logps - old_logps

        if self.importance_sampling_level == "sequence":
            seq_log_ratio = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1)
            log_ratio = seq_log_ratio.unsqueeze(-1).expand_as(per_token_logps)

        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1.0 - self.epsilon, 1.0 + self.epsilon_high)
        loss_unclipped = coef_1 * advantages
        loss_clipped = coef_2 * advantages
        per_token_loss = -torch.min(loss_unclipped, loss_clipped)

        if self.beta != 0.0 and "ref_per_token_logps" in inputs:
            ref_logps = inputs["ref_per_token_logps"].to(per_token_logps.dtype)
            kl_diff = ref_logps - per_token_logps
            per_token_kl = torch.exp(kl_diff) - kl_diff - 1.0
            per_token_loss = per_token_loss + self.beta * per_token_kl
            kl_value = float(((per_token_kl * completion_mask).sum() / completion_mask.sum().clamp(min=1)).item())
        else:
            kl_value = 0.0

        masked_loss = per_token_loss * completion_mask
        loss = self._reduce_masked_loss(masked_loss, completion_mask, num_items_in_batch=inputs.get("num_items_in_batch", None))

        # ---- sub_talker GRPO branch (layers 1..15) ----
        sub_metrics = {}
        if sub_grpo_on and out.get("sub_talker_logps") is not None:
            sub_cur = out["sub_talker_logps"]  # [B, L]

            if self.sub_talker_mode == "reinforce":
                # REINFORCE: ratio is numerically 1 (exp(0)=1) but gradient flows
                # through sub_cur via the detach trick: d/dx (x - x.detach()) = 1,
                # so d/dx exp(x - x.detach()) = exp(0) * 1 = 1, giving
                # d(loss)/d(sub_cur) = -advantage — standard REINFORCE gradient.
                sub_log_ratio = sub_cur - sub_cur.detach()
                sub_coef_1 = torch.exp(sub_log_ratio)
            else:
                # PPO: use importance sampling ratio
                sub_old = inputs.get("old_sub_talker_logps", sub_cur.detach()).to(sub_cur.dtype)
                sub_log_ratio = sub_cur - sub_old
                # Sub_talker logp = sum over 15 layers per frame → per-frame ratios swing
                # very wide (saw clip_ratio p90=0.27, kl max=31 in v9 token-level run).
                # Sequence-level dampens by averaging log-ratio across all valid frames first.
                if self.sub_importance_sampling_level == "sequence":
                    sub_seq_log_ratio = (sub_log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1)
                    sub_log_ratio = sub_seq_log_ratio.unsqueeze(-1).expand_as(sub_cur)
                sub_coef_1 = torch.exp(sub_log_ratio)

            sub_coef_2 = torch.clamp(sub_coef_1, 1.0 - self.epsilon, 1.0 + self.epsilon_high)
            sub_unclipped = sub_coef_1 * advantages
            sub_clipped = sub_coef_2 * advantages
            sub_per_token_loss = -torch.min(sub_unclipped, sub_clipped)

            # KL penalty (both modes)
            sub_kl_value = 0.0
            if self.beta != 0.0 and "ref_sub_talker_logps" in inputs:
                sub_ref = inputs["ref_sub_talker_logps"].to(sub_cur.dtype)
                # Per-frame sub_talker logp = sum over 15 layers, so diff easily
                # reaches ±10 nats on outlier frames. Unclamped K3 KL exp(diff)
                # can hit 22k per frame and pollute the gradient even after seq-IS.
                # Clamp to ±5 nats (exp(5)≈148) caps the worst-case contribution
                # while preserving signal in the well-behaved range.
                sub_kl_diff = (sub_ref - sub_cur).clamp(-5.0, 5.0)
                sub_per_token_kl = torch.exp(sub_kl_diff) - sub_kl_diff - 1.0
                sub_per_token_loss = sub_per_token_loss + self.beta * sub_per_token_kl
                sub_kl_value = float(((sub_per_token_kl * completion_mask).sum() / completion_mask.sum().clamp(min=1)).item())
            sub_masked_loss = sub_per_token_loss * completion_mask
            sub_loss = self._reduce_masked_loss(sub_masked_loss, completion_mask, num_items_in_batch=inputs.get("num_items_in_batch", None))
            loss = loss + self.grpo_sub_talker_weight * sub_loss

            with torch.no_grad():
                mask_sum = completion_mask.sum().clamp(min=1)
                sub_clip_low = (sub_coef_1 < 1.0 - self.epsilon).float()
                sub_clip_high = (sub_coef_1 > 1.0 + self.epsilon_high).float()
                sub_metrics["sub_talker/policy_loss"] = float(sub_loss.detach().item())
                sub_metrics["sub_talker/clip_ratio/low"] = float(((sub_clip_low * completion_mask).sum() / mask_sum).item())
                sub_metrics["sub_talker/clip_ratio/high"] = float(((sub_clip_high * completion_mask).sum() / mask_sum).item())
                if self.beta != 0.0:
                    sub_metrics["sub_talker/kl"] = sub_kl_value
                sub_metrics["sub_talker/logp_mean"] = float(((sub_cur.detach() * completion_mask).sum() / mask_sum).item())

        if return_hidden and self.sub_talker_aux_loss_weight > 0 and out.get("hidden") is not None:
            try:
                raw = self.accelerator.unwrap_model(model).model
                hidden = out["hidden"]
                # gather hidden + codec_ids over completion positions only
                B = hidden.shape[0]
                start_idx = inputs["completion_start_idx"]
                hidden_flat = []
                codec_flat = []
                for b in range(B):
                    s = int(start_idx[b].item())
                    cl = int(inputs["completion_mask"][b].sum().item())
                    hidden_flat.append(hidden[b, : cl])
                    codec_flat.append(inputs["codec_ids"][b, s : s + cl, :])
                if hidden_flat:
                    hidden_cat = torch.cat(hidden_flat, dim=0)
                    codec_cat = torch.cat(codec_flat, dim=0)
                    _, sub_loss = raw.talker.forward_sub_talker_finetune(codec_cat, hidden_cat)
                    loss = loss + self.sub_talker_aux_loss_weight * sub_loss
            except Exception as e:
                logger.warning(f"sub_talker aux loss failed: {e}")

        with torch.no_grad():
            clip_low = (coef_1 < 1.0 - self.epsilon).float()
            clip_high = (coef_1 > 1.0 + self.epsilon_high).float()
            mask_sum = completion_mask.sum().clamp(min=1)
            self._tts_metrics["policy_loss"] = float(loss.detach().item())
            self._tts_metrics["entropy"] = float(((entropies * completion_mask).sum() / mask_sum).item()) if entropies is not None else 0.0
            self._tts_metrics["clip_ratio/low"] = float(((clip_low * completion_mask).sum() / mask_sum).item())
            self._tts_metrics["clip_ratio/high"] = float(((clip_high * completion_mask).sum() / mask_sum).item())
            # clip_ratio/total removed: identically = low + high, derivable.
            if self.beta != 0.0:
                self._tts_metrics["kl"] = kl_value
            if sub_metrics:
                self._tts_metrics.update(sub_metrics)

        if return_outputs:
            return loss, out
        return loss

    def _reduce_tts_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """All-reduce custom rollout/loss metrics across DDP ranks.

        Without this, rank 0 only sees its own per-device batch (1 unique prompt
        × G samples = 4 candidates). After AVG reduce over 4 ranks the effective
        sample size becomes 16 and continuous metric noise halves (std reduced by √4).

        step_time uses MAX because the DDP step is bottlenecked by the slowest
        rank — the mean understates true wall-clock cost. Everything else uses
        AVG. Standard HF metrics (loss, grad_norm, learning_rate, epoch) are
        added by HF Trainer directly to ``logs`` and never enter _tts_metrics,
        so they are skipped here.
        """
        if not (dist.is_initialized() and dist.get_world_size() > 1):
            return metrics
        device = self.accelerator.device
        # IMPORTANT: collective ops require identical call order across ranks.
        # Sorting keys guarantees this even if dict insertion order ever drifts.
        # Assumption: all ranks emit the same key set. If we add a mode where
        # keys can differ per rank, switch to all_gather_object on the key union first.
        reduced = {}
        for k in sorted(metrics.keys()):
            v = metrics[k]
            op = dist.ReduceOp.MAX if k == "step_time" else dist.ReduceOp.AVG
            t = torch.tensor(float(v), device=device, dtype=torch.float32)
            dist.all_reduce(t, op=op)
            reduced[k] = float(t.item())
        return reduced

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        if self._tts_metrics:
            reduced = self._reduce_tts_metrics(self._tts_metrics)
            logs.update(reduced)
            self._tts_metrics = {}
        super().log(logs, start_time)

    def training_step(self, model, inputs, num_items_in_batch=None):
        t0 = time.time()
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        self._tts_metrics["step_time"] = time.time() - t0
        return loss

    def _save_lora_adapter(self, output_dir: str) -> None:
        """Save LoRA adapter weights without merging.

        For PEFT models, saves only the LoRA adapter weights and config.
        This allows resuming training from the adapter checkpoint.
        """
        full_model = self.accelerator.unwrap_model(self.model)
        talker = get_inner_talker(full_model)

        # Check if talker is a PEFT model
        if not hasattr(talker, "save_pretrained") or not hasattr(talker, "peft_config"):
            logger.warning("Model is not a PEFT model, skipping LoRA adapter save")
            return

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Save LoRA adapter (adapter_model.safetensors + adapter_config.json)
        talker.save_pretrained(output_dir, safe_serialization=True)
        logger.info(f"Saved LoRA adapter to {output_dir}")

    def _save(self, output_dir: Optional[str] = None, state_dict=None) -> None:
        full_model = self.accelerator.unwrap_model(self.model)
        talker = get_inner_talker(full_model)

        # Check if this is a PEFT model
        is_peft = hasattr(talker, "save_pretrained") and hasattr(talker, "peft_config")

        if is_peft and state_dict is None:
            # Save LoRA adapter only (adapter_model.safetensors + adapter_config.json)
            self._save_lora_adapter(output_dir or self.args.output_dir)
        else:
            # Fallback to original save logic for non-PEFT models
            with merge_peft_if_present(talker) as is_peft_ctx:
                if is_peft_ctx and state_dict is None:
                    state_dict = strip_peft_keys(full_model.state_dict())
                super()._save(output_dir=output_dir, state_dict=state_dict)

        bundle_inference_aux(self._init_model_path, output_dir or self.args.output_dir)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None) -> None:
        """Override to handle LoRA adapter checkpoints nested inside Qwen3TTSForGRPO wrapper.

        HF Trainer's default _load_from_checkpoint checks _is_peft_model(self.model),
        but our self.model is Qwen3TTSForGRPO (not PeftModel). The PEFT model is nested
        inside wrapper.model.talker. We detect adapter files and load them directly,
        then load optimizer/scheduler/rng state separately (skipping HF's model loading).
        """
        if model is None:
            model = self.model

        # PEFT adapter filenames (hardcoded to be robust across peft versions —
        # the constants moved around several times: peft.ADAPTER_WEIGHTS_NAME,
        # peft.utils.ADAPTER_SAFE_WEIGHTS_NAME, peft.utils.constants.SAFETENSORS_WEIGHTS_NAME…)
        ADAPTER_WEIGHTS_NAME = "adapter_model.bin"
        ADAPTER_SAFE_WEIGHTS_NAME = "adapter_model.safetensors"
        adapter_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_WEIGHTS_NAME)
        adapter_safe_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)

        has_adapter = os.path.exists(adapter_weights_file) or os.path.exists(adapter_safe_weights_file)

        full_model = self.accelerator.unwrap_model(model)
        talker = get_inner_talker(full_model)
        is_peft = hasattr(talker, "load_adapter") and hasattr(talker, "peft_config")

        is_main = self.accelerator.is_main_process
        if is_main:
            print(f"\n[Resume] Loading checkpoint from: {resume_from_checkpoint}")

        if has_adapter and is_peft:
            active_adapter = getattr(talker, "active_adapter", "default")
            if is_main:
                print(f"[Resume]   detected LoRA adapter files (active_adapter={active_adapter})")
            talker.load_adapter(resume_from_checkpoint, active_adapter, is_trainable=True)
            if is_main:
                print(f"[Resume]   ✓ LoRA adapter weights restored")
                print(f"[Resume]   optimizer/scheduler/RNG will be restored later by HF Trainer")
        else:
            if is_main:
                print(f"[Resume]   no LoRA adapter detected → falling back to HF full-model load")
            super()._load_from_checkpoint(resume_from_checkpoint, model=model)

        # Report resumed step (trainer_state.json is loaded by HF Trainer separately before this call)
        if is_main:
            trainer_state_path = os.path.join(resume_from_checkpoint, "trainer_state.json")
            if os.path.exists(trainer_state_path):
                import json
                try:
                    with open(trainer_state_path, "r") as f:
                        st = json.load(f)
                    print(f"[Resume]   trainer_state: global_step={st.get('global_step')} epoch={st.get('epoch', 0):.4f}")
                except Exception as e:
                    print(f"[Resume]   (could not read trainer_state.json: {e})")
            print(f"[Resume] Ready — training will continue from the above step.\n")
