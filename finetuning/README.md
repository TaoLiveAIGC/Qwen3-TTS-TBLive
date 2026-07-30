# Fine-tuning Qwen3-TTS (Base) — SFT + GRPO/GDPO

> This document extends the upstream
> [Qwen3-TTS/finetuning/README.md](https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning)
> with two additional training pipelines used to produce the live-streaming
> checkpoints released in this repo:
>
> 1. **HF-Trainer-based SFT** (`sft_base_hf.py` + `run_hf.sh`) — multi-GPU
>    SFT with LoRA / ICL, replacing the plain-DDP single-speaker
>    `sft_12hz.py` from upstream.
> 2. **GRPO / GDPO reinforcement learning** (`grpo_train.py` +
>    `run_grpo.sh`) — 2-dim gated or 4-dim reward (CER + SSIM + CPS +
>    SemiFL) fine-tuning on top of an SFT checkpoint.
>
> The upstream single-speaker SFT script `sft_12hz.py` is still shipped
> unchanged for reference.

---

## 1. Data format

Both `sft_base_hf.py` and `grpo_train.py` share the same input format,
controlled by two arguments:

- `--train_list` — either a single `.json` file, **or** a manifest text
  file with one `.json` path per line (used when the corpus is sharded
  into many files).
- `--flatten` / `--no-flatten` — selects which of the two JSON structures
  below is expected inside each `.json` file.

The default differs by entry point: `sft_base_hf.py` defaults to
`--flatten` (Mode B), while `grpo_train.py` and the shipped `run_hf.sh`
run with `--no-flatten` (Mode A).

### Mode A — grouped, `list[list[dict]]` (`--no-flatten`)

Each inner list is one meta entry, holding a set of utterances that
share the same speaker. On each step the loader picks one utterance as
the training target and another utterance **from the same inner list**
as the reference, so every inner list must contain at least two items.
(One speaker's data can be split across multiple inner lists — the
grouping is per-entry, not per-speaker.)

Typical use cases:
- **Continual pre-training** of the base model on large multi-speaker
  corpora.
- **Large-scale multi-speaker SFT** (many speakers × many utterances).

```json
[
  [
    {"path": "./data/spk1/utt0001.wav", "text": "第一句话。"},
    {"path": "./data/spk1/utt0002.wav", "text": "第二句话。"}
  ],
  [
    {"path": "./data/spk2/utt0100.wav", "text": "hello world"},
    {"path": "./data/spk2/utt0101.wav", "text": "another one"}
  ]
]
```

### Mode B — flat, `list[dict]` (`--flatten`)

Every item carries its own reference audio. Best for single-speaker
fine-tuning, or any case where the reference audio is fixed / chosen
externally rather than sampled from a group.

```json
[
  {"path": "./data/utt0001.wav", "text": "第一句话。",
   "ref_audio": "./data/ref.wav"},
  {"path": "./data/utt0002.wav", "text": "第二句话。",
   "ref_audio": "./data/ref.wav"}
]
```

Recognized reference-audio keys (checked in order): `const_ref_audio`,
`dynamic_ref_audio`, `ref_audio`. `--use_const_ref` prefers
`const_ref_audio`.

### Sharded corpora via manifest

For large datasets split across many `.json` shards, pass a manifest
instead of a single file:

```
# train_list.txt
/data/shard_000.json
/data/shard_001.json
/data/shard_002.json
```

`--data_split_mode` controls how shards / samples are distributed across
DDP ranks (`files` = one shard per rank, `samples` = every rank reads all
shards and takes a stride-N slice, `auto` picks between them).

### `audio_codes` extraction

Both `sft_base_hf.py` and `grpo_train.py` extract speech `audio_codes`
online using the tokenizer at `--tokenizer_path`; no pre-processing step
is required.

---

## 2. HF-Trainer SFT (`sft_base_hf.py`)

Wraps the model through `transformers.Trainer`, which fixes gradient sync
issues seen with the vanilla DDP script and adds LoRA, ICL data mode, and
multi-file dataset loading.

### Quickstart

```bash
bash finetuning/run_hf.sh
```

Edit the environment defaults at the top of `run_hf.sh` (or export them
before invoking):

```bash
export INIT_MODEL_PATH=Qwen/Qwen3-TTS-12Hz-1.7B-Base
export TRAIN_LIST=/path/to/train_list.txt
export TOKENIZER_PATH=Qwen/Qwen3-TTS-Tokenizer-12Hz
export OUTPUT_MODEL_PATH=./exp/my_sft_run
bash finetuning/run_hf.sh
```

### Key flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--icl_mode` / `--no-icl_mode` | `True` | With ICL: prepend `ref_text` + `ref_codes` as prefix; recommended for multi-speaker corpora. Disable for pure x-vector single-speaker SFT. |
| `--use_lora` | off | Enable PEFT LoRA fine-tuning. |
| `--lora_rank` / `--lora_alpha` / `--lora_dropout` | 64 / 128 / 0.05 | LoRA hyper-parameters. `run_hf.sh` uses 16/16/0.05. |
| `--sub_talker_loss_weight` | 0.1 | Weight on sub-talker (code predictor, layers 1..15) NLL. |
| `--data_split_mode` | `auto` | `samples` = split samples across ranks, `files` = split JSONL files across ranks. |
| `--flatten` / `--no-flatten` | `True` | `--flatten` expects flat `list[dict]` with per-item ref audio (Mode B); `--no-flatten` keeps grouped entries (Mode A). See §1. |
| `--pinyin_replace_max` | 0 | Max chars per sample to rewrite as pinyin/phoneme for text augmentation; 0 disables. Needs the optional `g2pw_tool`. |

See `sft_base_hf.py --help` for the full list.

---

## 3. GRPO / GDPO training (`grpo_train.py`)

Reinforcement fine-tuning on top of an SFT checkpoint. Two reward regimes
are supported through the same script:

- **2-dim gated reward** (default when `w_cps == 0 and w_semi_fl == 0`):
  gate on CER threshold, optimize SSIM. Suitable for correctness-first
  tuning.
- **4-dim GDPO reward** (`w_cps > 0` or `w_semi_fl > 0`): batch-level
  z-score aggregation over `{CER, SSIM, CPS, SemiFL}` with dead-zone and
  exponential penalty on CER. Suitable for polishing prosody and pace on
  an already-competent SFT model.

### Quickstart

```bash
bash finetuning/run_grpo.sh
```

The default `run_grpo.sh` runs 4-dim GDPO with sub-talker REINFORCE.
Set these environment variables (or edit the script) to point at your
assets:

```bash
export INIT_MODEL_PATH=/path/to/sft/checkpoint
export TOKENIZER_PATH=Qwen/Qwen3-TTS-Tokenizer-12Hz
export WESPEAKER_PATH=/path/to/wespeaker_chinese
export ASR_MODEL_PATH=/path/to/paraformer-large-zh
export TRAIN_LIST=/path/to/train_list.txt
export OUTPUT_DIR=./exp_grpo/my_run
bash finetuning/run_grpo.sh
```

### Reward configuration

| Flag | Default | Meaning |
|------|---------|---------|
| `--w_cer` / `--w_sim` / `--w_cps` / `--w_semi_fl` | 1 / 1 / 0 / 0 | Weights of the 4 reward dimensions. Non-zero `w_cps` or `w_semi_fl` triggers GDPO 4-dim mode. |
| `--cer_deadzone` | 0.03 | CER below this contributes zero penalty. |
| `--cer_exp_k` | 3.0 | Exponential slope on CER beyond the dead zone. |
| `--cps_deadzone_low` / `--cps_deadzone_high` | 0.05 / 0.10 | Dead-zone bounds on relative chars-per-second deviation vs. reference. |
| `--ssim_threshold` | 0.82 | Rollouts with SSIM below this are penalized (2-dim mode) or heavily weighted (4-dim mode). |
| `--reward_cer_threshold` | 0.3 | Hard CER gate in 2-dim mode. |

`SemiFL` (semitone fluctuation) is computed from `librosa.pyin`; set
`--skip_semi_fl_compute` when `w_semi_fl == 0` to save rollout time.

### GRPO / RL flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--num_generations` | 4 | Rollouts per prompt (group size for GRPO). |
| `--num_iterations` | 1 | PPO-style importance-sampling iterations per rollout. |
| `--beta` | 0.0 | KL coefficient to reference model. `run_grpo.sh` uses 0.05. |
| `--importance_sampling_level` | `token` | Talker importance-ratio granularity. `sequence` is more stable. |
| `--sub_talker_mode` | `ppo` | `reinforce` uses ratio=1 for sub-talker (recommended when unfreezing sub-talker). |
| `--grpo_sub_talker_weight` | 0.0 | GRPO loss weight on sub-talker (layers 1..15). > 0 unfreezes it. |
| `--sub_talker_trunk_grad_scale` | 0.0 | Scale of sub-talker gradient flowing back into the talker trunk. |
| `--freeze_sub_talker` / `--freeze_speaker_encoder` | on / on | Toggle to unfreeze components. Auto-disabled when the corresponding GRPO weight is > 0. |
| `--rollout_icl_mode` / `--no-rollout_icl_mode` | on | Match the rollout conditioning to whichever SFT variant was trained. |
| `--use_lora` + `--lora_rank/alpha/dropout` | off | LoRA GRPO (recommended for stability). |

External assets required at training time:

- **WeSpeaker** speaker-verification checkpoint (`--wespeaker_path`) — used
  to compute SSIM against `ref_audio`.
- **FunASR Paraformer** model (`--asr_model_path`) — used to compute CER
  against `text`.

Both are used only for reward computation; no gradient flows through them.

### Resume from checkpoint

```bash
bash finetuning/run_grpo.sh
# then add to run_grpo.sh:
#   --resume_from_checkpoint $OUTPUT_DIR/checkpoint-<step>
```

---

## 4. Register speakers for `generate_custom_voice()`

Both `sft_base_hf.py` and `grpo_train.py` save a **base-type** checkpoint —
its `tts_model_type` stays as `"base"`, and inference is done through the
zero-shot cloning API (`Qwen3TTSModel.generate(..., ref_audio=...)`).

To call `generate_custom_voice(text=..., speaker="<name>")` you must first
bake one or more reference speakers into the checkpoint. This is a
one-time post-processing step per checkpoint:

1. Write a `speaker_info.json` describing each speaker and its reference
   audios (multiple refs are averaged into a single embedding):

   ```json
   {
     "speakers": [
       {
         "speaker_name": "Vivian",
         "speaker_id": 2800,
         "ref_audios": [
           {"audio": "/path/to/vivian_ref_01.wav", "text": "对应文本1"},
           {"audio": "/path/to/vivian_ref_02.wav", "text": "对应文本2"}
         ]
       },
       {
         "speaker_name": "Alex",
         "speaker_id": 2801,
         "ref_audios": [
           {"audio": "/path/to/alex_ref_01.wav", "text": "对应文本1"}
         ]
       }
     ]
   }
   ```

   Each `speaker_id` must be an integer in the range **[2158, 3071]**
   (reserved CustomVoice slots in the codec embedding table).

2. Run the registration script:

   ```bash
   python finetuning/register_speaker.py \
     --model-path   exp/my_sft_run/checkpoint-<step> \
     --speaker-info speaker_info.json \
     --output-path  exp/my_sft_run/checkpoint-<step>-customvoice \
     --device       cuda:0
   ```

   The script (`finetuning/register_speaker.py`):
   - loads the SFT checkpoint (base-type),
   - extracts a speaker embedding per name via the frozen
     `speaker_encoder` (averaging across the listed ref audios),
   - writes each embedding into the reserved slot of
     `talker.model.codec_embedding.weight`,
   - patches `config.json`: sets `tts_model_type="custom_voice"` and
     populates `talker_config.spk_id` / `spk_is_dialect`,
   - drops `speaker_encoder.*` weights (unused at CustomVoice inference),
   - saves everything to `--output-path`.

3. Load the registered checkpoint and call `generate_custom_voice`:

   ```python
   import torch, soundfile as sf
   from qwen_tts import Qwen3TTSModel

   tts = Qwen3TTSModel.from_pretrained(
       "exp/my_sft_run/checkpoint-<step>-customvoice",
       device_map="cuda:0",
       dtype=torch.bfloat16,
       attn_implementation="flash_attention_2",
   )
   wavs, sr = tts.generate_custom_voice(text="Hello world.", speaker="Vivian")
   sf.write("out.wav", wavs[0], sr)
   ```

---

## 5. Quick inference test after training

Before registration, the SFT checkpoint runs as a base-type (zero-shot)
model driven by `ref_audio`:

```python
import torch, soundfile as sf
from qwen_tts import Qwen3TTSModel

tts = Qwen3TTSModel.from_pretrained(
    "exp/my_sft_run/checkpoint-<step>",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
wavs, sr = tts.generate(text="Hello world.", ref_audio="./data/ref.wav")
sf.write("out.wav", wavs[0], sr)
```

For the full two-stage post-selection inference (SSIM filter + CER pick),
see the scripts under [`examples/`](../examples/) and
[`evaluation/README.md`](../evaluation/README.md).

---

## 6. Relation to upstream `sft_12hz.py`

The upstream single-speaker script is kept unchanged. It bakes a single
speaker embedding into the checkpoint inline during training (see the
tail of `sft_12hz.py`), so its output is already CustomVoice-ready
without the extra `register_speaker.py` step.

`sft_base_hf.py` supersedes it for multi-speaker, LoRA, and ICL training
and defers speaker registration to `register_speaker.py`, which allows
registering multiple speakers into one checkpoint from a shared
`speaker_info.json`.
