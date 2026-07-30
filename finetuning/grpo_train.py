# coding=utf-8
"""GRPO training entry for Qwen3-TTS.

Usage:
    torchrun --nproc_per_node=N finetuning/grpo_train.py \\
        --init_model_path ... --train_list ... --output_model_path ...
"""

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import torch  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402

from grpo_dataset import TTSGRPODataset  # noqa: E402
from hf_trainer import (  # noqa: E402
    GRPOTTSConfig,
    Qwen3TTSForGRPO,
    Qwen3TTSGRPOTrainer,
    TTSGRPORolloutEngine,
)
from qwen_tts import Qwen3TTSTokenizer  # noqa: E402
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig  # noqa: E402
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration  # noqa: E402
from qwen_tts.core.models.processing_qwen3_tts import Qwen3TTSProcessor  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Qwen3-TTS GRPO training")
    p.add_argument("--init_model_path", type=str, required=True)
    p.add_argument("--output_model_path", type=str, required=True)
    p.add_argument("--train_list", type=str, required=True)
    p.add_argument("--tokenizer_path", type=str, required=True)
    p.add_argument("--wespeaker_path", type=str, required=True)
    p.add_argument("--asr_model_path", type=str, required=True)

    p.add_argument("--max_steps", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--dataloader_num_workers", type=int, default=1)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--gradient_checkpointing", action="store_true")

    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument("--ssim_threshold", type=float, default=0.82)
    p.add_argument("--cer_threshold", type=float, default=0.08)
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--epsilon_high", type=float, default=None)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--loss_type", type=str, default="grpo")
    p.add_argument("--scale_rewards", type=str, default="none")
    p.add_argument("--importance_sampling_level", type=str, default="token")
    p.add_argument("--steps_per_generation", type=int, default=1)
    p.add_argument("--num_iterations", type=int, default=1)
    p.add_argument("--mask_truncated_completions", action="store_true")
    p.add_argument("--sub_talker_aux_loss_weight", type=float, default=0.0)
    p.add_argument("--grpo_sub_talker_weight", type=float, default=0.0,
                   help="GRPO loss weight on sub_talker (layers 1..15). 0 disables. "
                        "When >0, freeze_sub_talker is auto-disabled. Typical 0.1-0.5.")
    p.add_argument("--sub_importance_sampling_level", type=str, default="sequence",
                   choices=["token", "sequence"],
                   help="Sub_talker importance ratio granularity. Default 'sequence' "
                        "(stabler, dampens per-frame swing from 15-layer logp sum). "
                        "'token' matches sibling impl but very volatile in our setup.")
    p.add_argument("--sub_talker_trunk_grad_scale", type=float, default=0.0,
                   help="Scale for sub_talker GRPO grad flowing into talker trunk via hidden. "
                        "0.0=detach (safe default). 0.05-0.1 mimics sibling LoRA's natural bound. "
                        "1.0=no scale (v9-attempt-1 catastrophe in full-finetune; do not use).")
    p.add_argument("--freeze_sub_talker", action="store_true", default=True)
    p.add_argument("--freeze_speaker_encoder", action="store_true", default=True)
    p.add_argument("--max_completion_length", type=int, default=1500)
    p.add_argument("--rollout_icl_mode", action=argparse.BooleanOptionalAction, default=True,
                   help="Rollout generation mode: ICL (default) or x-vector-only. "
                        "Pass --no-rollout-icl-mode for x-vector-only (only ref_audio speaker "
                        "embedding conditions generation, no ref_text/ref_codes prefix).")
    p.add_argument("--gen_temperature", type=float, default=1.0)
    p.add_argument("--gen_top_k", type=int, default=50)
    p.add_argument("--gen_top_p", type=float, default=0.9)
    p.add_argument("--gen_repetition_penalty", type=float, default=1.0)
    p.add_argument("--subtalker_temperature", type=float, default=1.0)
    p.add_argument("--subtalker_top_k", type=int, default=50)
    p.add_argument("--subtalker_top_p", type=float, default=0.9)

    p.add_argument("--reward_cer_threshold", type=float, default=0.3)
    p.add_argument("--reward_cer_penalty_weight", type=float, default=0.5)
    p.add_argument("--reward_cer_quality_weight", type=float, default=0.0,
                   help="Weight of CER quality bonus when CER ≤ threshold. 0 = pure SSIM.")

    # GDPO 4-dim reward: batch-level z-score over {cer, sim, cps, semi_fl}
    # When w_cps > 0 or w_semi_fl > 0, switches from 2-dim gated reward to 4-dim GDPO.
    p.add_argument("--w_cer", type=float, default=1.0,
                   help="Weight for CER dimension in GDPO 4-dim advantage. 0 disables.")
    p.add_argument("--w_sim", type=float, default=1.0,
                   help="Weight for SSIM dimension in GDPO 4-dim advantage. 0 disables.")
    p.add_argument("--w_cps", type=float, default=0.0,
                   help="Weight for CPS (chars/sec) dimension. 0 disables (default, 2-dim mode). >0 enables 4-dim GDPO.")
    p.add_argument("--w_semi_fl", type=float, default=0.0,
                   help="Weight for SemiFL (semitone fluctuation %). 0 disables (default, 2-dim mode). >0 enables 4-dim GDPO.")
    p.add_argument("--cer_deadzone", type=float, default=0.03,
                   help="CER dead-zone: CER ≤ tau → reward=0; CER > tau → exp penalty. Default 0.03.")
    p.add_argument("--cer_exp_k", type=float, default=3.0,
                   help="Exponential growth rate for CER penalty above dead-zone. Default 3.0.")
    p.add_argument("--cps_deadzone_low", type=float, default=0.05,
                   help="Fractional tolerance below GT CPS (5%% slower allowed).")
    p.add_argument("--cps_deadzone_high", type=float, default=0.10,
                   help="Fractional tolerance above GT CPS (10%% faster allowed).")
    p.add_argument("--skip_semi_fl_compute", action="store_true",
                   help="Skip SemiFL computation (uses 0.0 placeholder). Useful when w_semi_fl=0.")

    p.add_argument("--sub_talker_mode", type=str, default="ppo", choices=["ppo", "reinforce"],
                   help="Sub_talker loss mode: 'ppo' (default) or 'reinforce' (ratio=1).")

    # LoRA
    p.add_argument("--use_lora", action="store_true",
                   help="Enable LoRA fine-tuning on talker (attention + MLP layers).")
    p.add_argument("--lora_rank", type=int, default=16,
                   help="LoRA rank (default: 16).")
    p.add_argument("--lora_alpha", type=int, default=32,
                   help="LoRA alpha (default: 32).")
    p.add_argument("--lora_dropout", type=float, default=0.05,
                   help="LoRA dropout (default: 0.05).")
    p.add_argument("--lora_target_modules", type=str, nargs="+", default=None,
                   help="LoRA target modules. Default: q/k/v/o/gate/up/down_proj.")

    # Dataset format options (mirror sft_base_hf.py)
    p.add_argument("--flatten", action=argparse.BooleanOptionalAction, default=False,
                   help="Flatten nested list[list[dict]] to list[dict]. When True, each "
                        "item must carry its own ref_audio (const_ref_audio / dynamic_ref_audio / ref_audio).")
    p.add_argument("--data_split_mode", type=str, default="auto",
                   choices=["auto", "files", "samples"],
                   help="DDP data distribution: 'files' splits train_list across ranks; "
                        "'samples' has every rank load all files and take stride-N slices.")
    p.add_argument("--use_const_ref", action="store_true",
                   help="Use const_ref_audio field instead of dynamic_ref_audio.")

    return p.parse_args()


def main():
    args = parse_args()

    # When GRPO sub_talker loss is enabled, sub_talker must be trainable
    if args.grpo_sub_talker_weight > 0 and args.freeze_sub_talker:
        print(f"[grpo_train] grpo_sub_talker_weight={args.grpo_sub_talker_weight} > 0, "
              "auto-disabling freeze_sub_talker so code_predictor receives gradients.")
        args.freeze_sub_talker = False

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    reward_device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        args.init_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    config = Qwen3TTSConfig.from_pretrained(args.init_model_path)
    processor = Qwen3TTSProcessor.from_pretrained(args.init_model_path, fix_mistral_regex=True)
    speech_tokenizer = Qwen3TTSTokenizer.from_pretrained(args.tokenizer_path, device_map=reward_device)
    model.load_speech_tokenizer(speech_tokenizer)

    wrapper = Qwen3TTSForGRPO(
        model=model,
        freeze_sub_talker=args.freeze_sub_talker,
        freeze_speaker_encoder=args.freeze_speaker_encoder,
        sub_talker_aux_loss_weight=args.sub_talker_aux_loss_weight,
        sub_talker_trunk_grad_scale=args.sub_talker_trunk_grad_scale,
    )

    # Apply LoRA if requested
    if args.use_lora:
        from peft import LoraConfig, get_peft_model

        target_modules = args.lora_target_modules or [
            # talker attention
            "q_proj", "k_proj", "v_proj", "o_proj",
            # talker MLP
            "gate_proj", "up_proj", "down_proj",
        ]
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
        )
        wrapper.model.talker = get_peft_model(wrapper.model.talker, lora_config)
        wrapper.model.talker.print_trainable_parameters()

    # Load ref_model only if needed and not using LoRA
    # With LoRA, we can disable adapter layers to get base model behavior
    ref_model = None
    if args.beta != 0.0 and not args.use_lora:
        ref_model = Qwen3TTSForConditionalGeneration.from_pretrained(
            args.init_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        for p in ref_model.parameters():
            p.requires_grad_(False)
        ref_model.eval()

    if args.train_list.endswith(".json"):
        # User passed the JSON data file directly (single-file dataset).
        train_data = [args.train_list]
    else:
        # Manifest file: one JSON path per line.
        train_data = [l.strip() for l in open(args.train_list, "r") if l.strip()]
    dataset = TTSGRPODataset(
        data_list=train_data,
        processor=processor,
        config=config,
        tokenizer_path=args.tokenizer_path,
        max_steps=args.max_steps,
        flatten=args.flatten,
        data_split_mode=args.data_split_mode,
        use_const_ref=args.use_const_ref,
    )

    rollout_engine = TTSGRPORolloutEngine(
        model=model,
        processor=processor,
        speech_tokenizer=speech_tokenizer,
        config=GRPOTTSConfig(
            output_dir=args.output_model_path,
            wespeaker_path=args.wespeaker_path,
            asr_model_path=args.asr_model_path,
            tokenizer_path=args.tokenizer_path,
            num_generations=args.num_generations,
            ssim_threshold=args.ssim_threshold,
            cer_threshold=args.cer_threshold,
            reward_device=reward_device,
            max_completion_length=args.max_completion_length,
            gen_temperature=args.gen_temperature,
            gen_top_k=args.gen_top_k,
            gen_top_p=args.gen_top_p,
            gen_repetition_penalty=args.gen_repetition_penalty,
            subtalker_temperature=args.subtalker_temperature,
            subtalker_top_k=args.subtalker_top_k,
            subtalker_top_p=args.subtalker_top_p,
            reward_cer_threshold=args.reward_cer_threshold,
            reward_cer_penalty_weight=args.reward_cer_penalty_weight,
            reward_cer_quality_weight=args.reward_cer_quality_weight,
            w_cer=args.w_cer,
            w_sim=args.w_sim,
            w_cps=args.w_cps,
            w_semi_fl=args.w_semi_fl,
            cer_deadzone=args.cer_deadzone,
            cer_exp_k=args.cer_exp_k,
            cps_deadzone_low=args.cps_deadzone_low,
            cps_deadzone_high=args.cps_deadzone_high,
            rollout_icl_mode=args.rollout_icl_mode,
        ),
        device=reward_device,
    )

    logging_dir = os.path.join(args.output_model_path, "runs")
    training_args = GRPOTTSConfig(
        output_dir=args.output_model_path,
        logging_dir=logging_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        save_steps=args.save_steps,
        save_total_limit=None,
        logging_steps=args.logging_steps,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        ddp_find_unused_parameters=True,
        report_to=["tensorboard"],
        lr_scheduler_type="cosine",
        num_generations=args.num_generations,
        ssim_threshold=args.ssim_threshold,
        cer_threshold=args.cer_threshold,
        wespeaker_path=args.wespeaker_path,
        asr_model_path=args.asr_model_path,
        tokenizer_path=args.tokenizer_path,
        reward_device=reward_device,
        epsilon=args.epsilon,
        epsilon_high=args.epsilon_high,
        beta=args.beta,
        loss_type=args.loss_type,
        scale_rewards=args.scale_rewards,
        importance_sampling_level=args.importance_sampling_level,
        steps_per_generation=args.steps_per_generation,
        num_iterations=args.num_iterations,
        mask_truncated_completions=args.mask_truncated_completions,
        sub_talker_aux_loss_weight=args.sub_talker_aux_loss_weight,
        grpo_sub_talker_weight=args.grpo_sub_talker_weight,
        sub_importance_sampling_level=args.sub_importance_sampling_level,
        freeze_sub_talker=args.freeze_sub_talker,
        freeze_speaker_encoder=args.freeze_speaker_encoder,
        max_completion_length=args.max_completion_length,
        gen_temperature=args.gen_temperature,
        gen_top_k=args.gen_top_k,
        gen_top_p=args.gen_top_p,
        gen_repetition_penalty=args.gen_repetition_penalty,
        subtalker_temperature=args.subtalker_temperature,
        subtalker_top_k=args.subtalker_top_k,
        subtalker_top_p=args.subtalker_top_p,
        reward_cer_threshold=args.reward_cer_threshold,
        reward_cer_penalty_weight=args.reward_cer_penalty_weight,
        reward_cer_quality_weight=args.reward_cer_quality_weight,
        sub_talker_mode=args.sub_talker_mode,
        sub_talker_trunk_grad_scale=args.sub_talker_trunk_grad_scale,
        w_cer=args.w_cer,
        w_sim=args.w_sim,
        w_cps=args.w_cps,
        w_semi_fl=args.w_semi_fl,
        cer_deadzone=args.cer_deadzone,
        cer_exp_k=args.cer_exp_k,
        cps_deadzone_low=args.cps_deadzone_low,
        cps_deadzone_high=args.cps_deadzone_high,
        rollout_icl_mode=args.rollout_icl_mode,
    )

    trainer = Qwen3TTSGRPOTrainer(
        model=wrapper,
        args=training_args,
        train_dataset=dataset,
        data_collator=dataset.collate_fn,
        rollout_engine=rollout_engine,
        ref_model=ref_model,
        init_model_path=args.init_model_path,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
