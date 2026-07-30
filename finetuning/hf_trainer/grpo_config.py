# coding=utf-8

from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class GRPOTTSConfig(TrainingArguments):
    """Training configuration for Qwen3-TTS GRPO."""

    num_generations: int = field(
        default=4,
        metadata={"help": "Number of completions (G) per prompt used for group-relative advantage."},
    )
    ssim_threshold: float = field(
        default=0.82,
        metadata={"help": "Minimum SSIM (speaker similarity) for a candidate to be considered qualified."},
    )
    cer_threshold: float = field(
        default=0.08,
        metadata={"help": "Maximum CER for a candidate to be considered qualified."},
    )
    wespeaker_path: str = field(
        default="",
        metadata={"help": "Path to the WeSpeaker model directory used for SSIM."},
    )
    asr_model_path: str = field(
        default="",
        metadata={"help": "Path to the FunASR paraformer model used for CER."},
    )
    tokenizer_path: str = field(
        default="",
        metadata={"help": "Path to the Qwen3 TTS speech tokenizer (codec encoder/decoder)."},
    )
    reward_device: str = field(
        default="cuda",
        metadata={"help": "Device for SSIM/CER models."},
    )
    epsilon: float = field(
        default=0.2,
        metadata={"help": "PPO/GRPO clipping epsilon (low side)."},
    )
    epsilon_high: Optional[float] = field(
        default=None,
        metadata={"help": "PPO/GRPO clipping epsilon (high side). Defaults to epsilon if None."},
    )
    beta: float = field(
        default=0.0,
        metadata={"help": "KL coefficient. 0 disables the reference model."},
    )
    loss_type: str = field(
        default="grpo",
        metadata={"help": "Reduction strategy: one of 'grpo' | 'bnpo' | 'dr_grpo' | 'dapo'."},
    )
    scale_rewards: str = field(
        default="none",
        metadata={"help": "Reward scaling: one of 'group' | 'batch' | 'none'."},
    )
    importance_sampling_level: str = field(
        default="token",
        metadata={"help": "Importance ratio granularity: 'token' or 'sequence'."},
    )
    steps_per_generation: int = field(
        default=1,
        metadata={"help": "Number of optimizer steps per rollout (>=1)."},
    )
    num_iterations: int = field(
        default=1,
        metadata={"help": "Number of optimization iterations over the same rollout (mu in GRPO paper)."},
    )
    mask_truncated_completions: bool = field(
        default=False,
        metadata={"help": "Mask completions that hit max length."},
    )
    sub_talker_aux_loss_weight: float = field(
        default=0.0,
        metadata={"help": "Optional auxiliary SFT loss weight on sub_talker."},
    )
    grpo_sub_talker_weight: float = field(
        default=0.0,
        metadata={
            "help": "GRPO loss weight on sub_talker (layers 1..15). 0 disables (talker-only mode, "
                    "default). Typical values 0.1-0.5. When >0, freeze_sub_talker is auto-disabled "
                    "and sub_talker per-frame log-probs participate in PPO clip + K3 KL exactly "
                    "like talker. Same epsilon / epsilon_high / beta apply to both."
        },
    )
    sub_importance_sampling_level: str = field(
        default="sequence",
        metadata={
            "help": "Importance ratio granularity for sub_talker: 'token' (per-frame, more volatile, "
                    "matches sibling impl) or 'sequence' (one ratio per sequence, default — much "
                    "stabler because per-frame sub_talker logp sums 15 layers and swings wide). "
                    "Per-frame KL is kept regardless (consistent with talker behavior)."
        },
    )
    sub_talker_mode: str = field(
        default="ppo",
        metadata={
            "help": "Sub_talker loss mode: 'ppo' (default, uses importance sampling ratio with clip, "
                    "same as before) or 'reinforce' (ratio=1, no IS, like the reference implementation). "
                    "Only used when grpo_sub_talker_weight > 0."
        },
    )
    sub_talker_trunk_grad_scale: float = field(
        default=0.0,
        metadata={
            "help": "Scale factor for the sub_talker GRPO gradient flowing back through the shared "
                    "hidden state into the talker trunk. 0.0 (default) detaches hidden — trunk is "
                    "fully shielded (v9 attempts 2-4 behavior, safest). >0.0 lets a fraction of the "
                    "sub_talker signal shape the trunk: 0.05-0.1 is roughly analogous to the sibling "
                    "LoRA implementation's natural bound via adapter capacity. 1.0 = no scaling "
                    "(matches v9 attempt 1 which catastrophically blew up — do not use in "
                    "full-finetune). Risk is bounded by C1 KL clamp (±5 nats) and num_iterations=1; "
                    "monitor grad_norm and sub_talker/kl when raising above 0."
        },
    )
    freeze_sub_talker: bool = field(
        default=True,
        metadata={"help": "Freeze the code_predictor (sub_talker) parameters."},
    )
    freeze_speaker_encoder: bool = field(
        default=True,
        metadata={"help": "Freeze the speaker_encoder parameters."},
    )
    max_completion_length: int = field(
        default=1500,
        metadata={"help": "Maximum codec frames per completion."},
    )
    rollout_icl_mode: bool = field(
        default=True,
        metadata={
            "help": "Rollout generation mode. True (default) = ICL: ref_text + ref_codes are "
                    "concatenated as in-context learning prefix. False = x-vector-only: only "
                    "ref_audio speaker embedding conditions the generation, matching "
                    "--no-icl_mode SFT training. Pass --no-rollout-icl-mode to disable."
        },
    )
    gen_temperature: float = field(
        default=1.0,
        metadata={"help": "Sampling temperature for rollout generation."},
    )
    gen_top_k: int = field(
        default=50,
        metadata={"help": "Top-k for rollout generation."},
    )
    gen_top_p: float = field(
        default=0.9,
        metadata={"help": "Top-p for rollout generation."},
    )
    gen_repetition_penalty: float = field(
        default=1.0,
        metadata={"help": "Repetition penalty for rollout generation."},
    )
    subtalker_temperature: float = field(
        default=1.0,
        metadata={"help": "Sub-talker sampling temperature."},
    )
    subtalker_top_k: int = field(
        default=50,
        metadata={"help": "Sub-talker top-k."},
    )
    subtalker_top_p: float = field(
        default=0.9,
        metadata={"help": "Sub-talker top-p."},
    )
    reward_cer_threshold: float = field(
        default=0.3,
        metadata={"help": "CER threshold for gated reward: candidates with CER > threshold get penalty reward."},
    )
    reward_cer_penalty_weight: float = field(
        default=0.5,
        metadata={"help": "Penalty weight when CER > threshold: reward = (1 - cer) * penalty_weight."},
    )
    reward_cer_quality_weight: float = field(
        default=0.0,
        metadata={
            "help": "Weight of CER quality bonus when CER ≤ threshold. "
                    "reward = (1 - w) * ssim + w * (1 - cer). 0 = pure SSIM (default, backward-compatible)."
        },
    )
    # ── GDPO 4-dim reward: batch-level z-score over {cer, sim, cps, semi_fl} ──
    # When w_cps > 0 or w_semi_fl > 0, switches from 2-dim gated reward to 4-dim GDPO.
    w_cer: float = field(
        default=1.0,
        metadata={"help": "Weight for CER dimension in GDPO 4-dim advantage. 0 disables."},
    )
    w_sim: float = field(
        default=1.0,
        metadata={"help": "Weight for SSIM dimension in GDPO 4-dim advantage. 0 disables."},
    )
    w_cps: float = field(
        default=0.0,
        metadata={
            "help": "Weight for CPS (chars/sec) dimension in GDPO 4-dim advantage. "
                    "0 disables (default, backward-compatible 2-dim mode). >0 enables 4-dim GDPO."
        },
    )
    w_semi_fl: float = field(
        default=0.0,
        metadata={
            "help": "Weight for SemiFL (semitone fluctuation %) dimension in GDPO 4-dim advantage. "
                    "0 disables (default, backward-compatible 2-dim mode). >0 enables 4-dim GDPO."
        },
    )
    cer_deadzone: float = field(
        default=0.03,
        metadata={
            "help": "CER dead-zone threshold for GDPO 4-dim. CER ≤ tau → reward=0; "
                    "CER > tau → -(exp(k*(cer-tau))-1). Default 0.03 (3%)."
        },
    )
    cer_exp_k: float = field(
        default=3.0,
        metadata={
            "help": "Exponential growth rate for CER penalty in GDPO 4-dim. "
                    "Higher = sharper penalty above dead-zone. Default 3.0."
        },
    )
    cps_deadzone_low: float = field(
        default=0.05,
        metadata={
            "help": "Fractional tolerance below GT CPS. Inside dead-zone → reward=0. "
                    "Default 0.05 (5% slower than GT allowed)."
        },
    )
    cps_deadzone_high: float = field(
        default=0.10,
        metadata={
            "help": "Fractional tolerance above GT CPS. Inside dead-zone → reward=0. "
                    "Default 0.10 (10% faster than GT allowed)."
        },
    )
    skip_semi_fl_compute: bool = field(
        default=False,
        metadata={
            "help": "Skip SemiFL computation (saves time, uses 0.0 as placeholder). "
                    "Useful for quick iteration when w_semi_fl=0 anyway."
        },
    )
