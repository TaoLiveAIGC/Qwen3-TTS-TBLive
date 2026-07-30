# coding=utf-8
"""HuggingFace Trainer-based SFT for Qwen3-TTS.

Wraps the model so that forward() is called through the DDP wrapper, which
fixes gradient sync issues in the plain-DDP SFT script.

Usage:
    python sft_base_hf.py --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base --train_list train_list.txt
    torchrun --nproc_per_node=4 sft_base_hf.py --init_model_path ... --train_list ...
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.multiprocessing as mp
from dataset_sep_file import TTSDataset
from hf_trainer import Qwen3TTSForSFT, TTSTrainer
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from transformers import TrainingArguments


def _log_run_config(args, training_args, lora_config=None):
    """Dump full training config to stdout (tee'd into train.log) at run start.

    Rank-0 only to keep multi-GPU logs clean. Covers cmdline, git state,
    distributed/GPU info, every argparse field, the load-bearing
    TrainingArguments fields, and the LoRA config when applicable —
    none of which HF Trainer prints by default.
    """
    if int(os.environ.get("RANK", "0")) != 0:
        return

    rule = "=" * 80
    lines = [
        "",
        rule,
        f"[Run Config] {datetime.datetime.now().isoformat(timespec='seconds')}",
        rule,
        f"cmdline: {' '.join(sys.argv)}",
    ]

    repo_root = os.path.dirname(os.path.abspath(__file__))
    try:
        sha = subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.run(
            ["git", "-C", repo_root, "diff", "--quiet"], stderr=subprocess.DEVNULL
        ).returncode != 0
        lines.append(f"git: {sha}{' (dirty)' if dirty else ''}")
    except Exception:
        pass

    lines.append(f"world_size: {os.environ.get('WORLD_SIZE', '1')}  "
                 f"rank: {os.environ.get('RANK', '0')}")
    if torch.cuda.is_available():
        lines.append(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '(default)')}")
        lines.append(f"GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")

    lines.append("")
    lines.append("--- argparse ---")
    lines.append(json.dumps(vars(args), indent=2, ensure_ascii=False, default=str, sort_keys=True))

    keep = [
        "output_dir", "logging_dir",
        "per_device_train_batch_size", "gradient_accumulation_steps",
        "learning_rate", "max_steps", "num_train_epochs",
        "warmup_steps", "weight_decay", "max_grad_norm", "lr_scheduler_type",
        "save_steps", "save_total_limit", "logging_steps",
        "bf16", "gradient_checkpointing", "ddp_find_unused_parameters",
        "report_to", "dataloader_num_workers", "remove_unused_columns",
    ]
    lines.append("")
    lines.append("--- TrainingArguments (key fields) ---")
    lines.append(json.dumps({k: getattr(training_args, k, None) for k in keep},
                            indent=2, ensure_ascii=False, default=str, sort_keys=True))

    if lora_config is not None:
        lines.append("")
        lines.append("--- LoRA config ---")
        cfg_dict = lora_config.to_dict() if hasattr(lora_config, "to_dict") else vars(lora_config)
        # PEFT stores `target_modules` / `exclude_modules` as Python sets, which json
        # can't serialize natively (falls back to str → ugly quoted set repr). Convert
        # any set values to sorted lists so they render as proper JSON arrays.
        cfg_dict = {k: (sorted(v) if isinstance(v, set) else v) for k, v in cfg_dict.items()}
        lines.append(json.dumps(cfg_dict, indent=2, ensure_ascii=False, default=str, sort_keys=True))

    lines.append(rule)
    lines.append("")
    print("\n".join(lines), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-TTS SFT with HuggingFace Trainer")

    # Model
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output_hf")

    # Data
    parser.add_argument("--train_list", type=str, required=True)
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="Qwen/Qwen3-TTS-Tokenizer-12Hz",
    )
    parser.add_argument("--max_steps", type=int, default=200000)

    # Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=5,
                        help="Max number of checkpoints to retain; HF rotates oldest.")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--logging_dir", type=str, default=None,
                        help="TensorBoard log directory. Defaults to <output_model_path>/runs.")
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        choices=["cosine", "constant", "linear", "constant_with_warmup"])
    parser.add_argument("--save_strategy", type=str, default="steps",
                        choices=["steps", "epoch", "no"])
    parser.add_argument("--sub_talker_loss_weight", type=float, default=0.1)
    parser.add_argument("--dataloader_num_workers", type=int, default=1)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--use_const_ref", action="store_true",
                        help="Use const_ref_audio field instead of dynamic_ref_audio in samples.")
    parser.add_argument("--icl_mode", action=argparse.BooleanOptionalAction, default=True,
                        help="When True (default), if a sample has ref_codes/ref_text they are "
                             "concatenated in front of target text/audio_codes (ICL training). "
                             "Pass --no-icl_mode to disable: only ref_audio is used (for "
                             "speaker_embedding via the speaker_encoder), matching the "
                             "x-vector-only inference mode.")
    parser.add_argument("--data_split_mode", type=str, default="auto",
                        choices=["auto", "files", "samples"],
                        help="DDP data distribution: 'files' splits train_list across ranks "
                             "(good when num files >= world_size); 'samples' has every rank "
                             "load all files and take a stride-N slice of samples (good for "
                             "small single-file datasets); 'auto' picks based on file count.")
    parser.add_argument("--flatten", action=argparse.BooleanOptionalAction, default=True,
                        help="Flatten nested list[list[dict]] to list[dict] in dataset loading. "
                             "Pass --no-flatten to keep original grouping.")

    # Text augmentation: pinyin/phoneme replacement
    parser.add_argument("--pinyin_replace_max", type=int, default=0,
                        help="Max number of chars to replace with pinyin/phoneme per sample. "
                             "0 disables replacement.")
    parser.add_argument("--pinyin_replace_prob", type=float, default=0.0,
                        help="Probability that a sample enters the pinyin replacement path. "
                             "0 disables replacement.")
    parser.add_argument("--pinyin_replace_mode", type=str, default="pinyin",
                        choices=["pinyin", "phoneme"],
                        help="Replacement mode: 'pinyin' or 'phoneme'.")

    # LoRA / PEFT
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=None,
        help="Target modules for LoRA. Defaults to all linear projection layers.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load model
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        args.init_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    config = Qwen3TTSConfig.from_pretrained(args.init_model_path)

    # Wrap model for training
    wrapper = Qwen3TTSForSFT(
        model=model,
        sub_talker_loss_weight=args.sub_talker_loss_weight,
    )

    # Apply LoRA if requested
    lora_config = None
    if args.use_lora:
        from peft import LoraConfig, get_peft_model

        target_modules = args.lora_target_modules or [
            # talker + code_predictor (sub_talker) attention
            "q_proj", "k_proj", "v_proj", "o_proj",
            # talker + code_predictor MLP
            "gate_proj", "up_proj", "down_proj",
            # code_predictor entry projection — sub_talker timbre path
            # "small_to_mtp_projection",
            # NOT wrapping output heads by default:
            #   - `talker.lm_head` does not exist (modeling defines `codec_head`).
            #   - `code_predictor.lm_head` is a ModuleList[Linear]×15 container; PEFT
            #     can't wrap containers, and suffix matching can't reach the .0..14
            #     children. To wrap the 15 sub-talker output heads, pass:
            #       --lora_target_modules '.*\.(q|k|v|o|gate|up|down)_proj$|.*\.small_to_mtp_projection$|code_predictor\.lm_head\.\d+$'
        ]
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            # "lm_head" suffix matches BOTH talker.lm_head (Linear, wrap-able) and
            # code_predictor.lm_head (ModuleList container, NOT wrap-able → PEFT raises).
            # Exclude the ModuleList; talker.lm_head still gets wrapped.
            exclude_modules=["code_predictor.lm_head"],
            lora_dropout=args.lora_dropout,
            bias="none",
        )
        wrapper.model.talker = get_peft_model(wrapper.model.talker, lora_config)
        wrapper.model.talker.print_trainable_parameters()

    # Build dataset
    from qwen_tts.core.models.processing_qwen3_tts import Qwen3TTSProcessor

    processor = Qwen3TTSProcessor.from_pretrained(
        args.init_model_path, fix_mistral_regex=True
    )

    if args.train_list.endswith(".json"):
        # User passed the JSON data file directly (single-file dataset).
        train_data = [args.train_list]
    else:
        # Manifest file: one JSON path per line.
        train_data = [l.strip() for l in open(args.train_list, "r") if l.strip()]
    dataset = TTSDataset(
        train_data,
        processor,
        config,
        tokenizer_path=args.tokenizer_path,
        max_steps=args.max_steps,
        use_const_ref=args.use_const_ref,
        data_split_mode=args.data_split_mode,
        icl_mode=args.icl_mode,
        flatten=args.flatten,
        pinyin_replace_max=args.pinyin_replace_max,
        pinyin_replace_prob=args.pinyin_replace_prob,
        pinyin_replace_mode=args.pinyin_replace_mode,
    )

    # Training arguments
    logging_dir = args.logging_dir or os.path.join(args.output_model_path, "runs")
    training_args = TrainingArguments(
        output_dir=args.output_model_path,
        logging_dir=logging_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        ddp_find_unused_parameters=True,
        report_to=["tensorboard"],
        lr_scheduler_type=args.lr_scheduler_type,
        save_strategy=args.save_strategy,
    )

    _log_run_config(args, training_args, lora_config)

    # Trainer
    trainer = TTSTrainer(
        model=wrapper,
        args=training_args,
        train_dataset=dataset,
        data_collator=dataset.collate_fn,
        init_model_path=args.init_model_path,
    )

    # Train
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save final model
    trainer.save_state()
    if args.use_lora:
        # save_embedding_layers=False: skip PEFT's auto-detect path which calls
        # talker.get_output_embeddings() → returns self.lm_head, but the modeling
        # code defines self.codec_head, not lm_head (get_output_embeddings is a
        # stale stub never triggered outside the PEFT save path). We don't target
        # any Embedding/lm_head in target_modules, so skipping is safe.
        wrapper.model.talker.save_pretrained(
            os.path.join(args.output_model_path, "lora_adapter"),
            save_embedding_layers=False,
        )
    else:
        trainer.save_model()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
