# Qwen3-TTS-TBLive

**🔊 [Live demo](https://taoliveaigc.github.io/Qwen3-TTS-TBLive/)** &nbsp;·&nbsp; TTS samples on live-streaming scripts.

An extension of [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) for the Chinese
live-streaming domain, built on the upstream `Qwen3-TTS-12Hz-1.7B-Base`
model. We release two checkpoints:

- `Qwen3-TTS-TBLive-Base` — a domain-adapted base model produced by **continual
  pre-training (CPT)** on Chinese live-streaming data, followed by GRPO
  reinforcement post-training. Recommended starting point for downstream fine-tuning.
- `Qwen3-TTS-TBLive-CustomVoice` — a **CustomVoice** checkpoint built on
  `Qwen3-TTS-TBLive-Base` with six built-in open-source speakers spanning three
  live-streaming styles (professional presentation / warm recommendation / energetic
  promotion), each in a female and a male voice.

See [Released Models](#released-models) for download links and the full speaker list.

We also provide the training pipelines behind these checkpoints — supervised fine-tuning
and GRPO reinforcement post-training — plus the supporting tools for inference
post-selection and CustomVoice speaker registration:

- SFT / continual pre-training via HuggingFace Trainer.
- GRPO fine-tuning with a composite reward over CER / SSIM and prosody signals.
- Two-stage SSIM → CER post-selection for inference and GRPO rollout scoring.
- CustomVoice speaker registration — bake new speakers into a fine-tuned checkpoint
  ([`finetuning/register_speaker.py`](finetuning/register_speaker.py)).

### Scope boundary

For upstream features (Voice Clone / Voice Design / CustomVoice APIs, tokenizer usage, vLLM
serving, DashScope API, deployment recipes, etc.) please refer to the
[upstream Qwen3-TTS README](https://github.com/QwenLM/Qwen3-TTS#readme); this repository
does not re-implement them.

---

## Contents

- [What's new vs. upstream Qwen3-TTS](#whats-new-vs-upstream-qwen3-tts)
- [Released Models](#released-models)
- [Instruction Control Tokens](#instruction-control-tokens)
- [Benchmark](#benchmark)
- [Demo](#demo)
- [Quickstart](#quickstart)
- [Repository layout](#repository-layout)
- [Documentation index](#documentation-index)
- [License](#license)

---

## What's new vs. upstream Qwen3-TTS

| Area | Upstream Qwen3-TTS | This repository |
|---|---|---|
| Supervised fine-tuning | Single-speaker reference script (`sft_12hz.py`) | HuggingFace Trainer + LoRA + ICL rollout ([`finetuning/sft_base_hf.py`](finetuning/sft_base_hf.py)) |
| Reinforcement post-training | — | GRPO with a 2-dim gated reward or a 4-dim GDPO composite reward (CER + SSIM + CPS + SemiFL), sub-talker REINFORCE, batch-level z-score normalization ([`finetuning/grpo_train.py`](finetuning/grpo_train.py)) |
| Inference quality | Single-shot decoding | Two-stage post-selection — WeSpeaker SSIM filtering followed by FunASR CER re-ranking — plus automatic long-text segmentation ([`examples/`](examples/), [`evaluation/`](evaluation/)) |
| Data pipeline | JSONL → codes → SFT | Same JSONL entry point extended with on-the-fly audio-code extraction and a shard-aware DDP dataset ([`finetuning/dataset_sep_file.py`](finetuning/dataset_sep_file.py)) |
| CustomVoice speakers | One speaker baked inline during single-speaker SFT | Register any number of speakers into a fine-tuned checkpoint from a shared `speaker_info.json` ([`finetuning/register_speaker.py`](finetuning/register_speaker.py)) |

Model definitions, tokenizers, and the voice-clone / voice-design / custom-voice APIs are
inherited verbatim from upstream.

---

## Released Models

We release the following checkpoints, all trained on top of `Qwen3-TTS-12Hz-1.7B-Base` using
the pipelines in this repository. They are hosted on HuggingFace under the
[TaoLiveAIGC](https://huggingface.co/TaoLiveAIGC) organization.

| Model | Description | Size | Language | HuggingFace |
|---|---|---|---|---|
| `Qwen3-TTS-TBLive-Base` | Domain-adapted base checkpoint (continual pre-training + GRPO on Chinese live-streaming data). Recommended starting point for downstream fine-tuning. | ~4.2 GB | Chinese (live-streaming) | [TaoLiveAIGC/Qwen3-TTS-TBLive-Base](https://huggingface.co/TaoLiveAIGC/Qwen3-TTS-TBLive-Base) |
| `Qwen3-TTS-TBLive-CustomVoice` | CustomVoice checkpoint built on `TBLive-Base`, with the six built-in speakers listed below. | ~4.2 GB | Chinese | [TaoLiveAIGC/Qwen3-TTS-TBLive-CustomVoice](https://huggingface.co/TaoLiveAIGC/Qwen3-TTS-TBLive-CustomVoice) |

Each release contains the main `model.safetensors` checkpoint, the 12 Hz speech tokenizer
under `speech_tokenizer/`, and the generation / tokenizer configs needed by the inference
scripts in this repository.

`Qwen3-TTS-TBLive-CustomVoice` ships six built-in speakers — three live-streaming styles,
each in a female and a male voice. Pass the speaker ID to
`generate_custom_voice(text, speaker=...)`:

| Speaker | Gender | Style | Speaker ID |
|---|---|---|---|
| 专业讲解女 | Female | Professional presentation (专业讲解) | 115 |
| 专业讲解男 | Male | Professional presentation (专业讲解) | 96 |
| 温柔女音 | Female | Warm recommendation (贴心推荐) | 26 |
| 温柔男音 | Male | Warm recommendation (贴心推荐) | 80 |
| 激情促销女 | Female | Energetic promotion (激情促销) | 4 |
| 激情促销男 | Male | Energetic promotion (激情促销) | 1 |

Download from HuggingFace:

```bash
# Base checkpoint (recommended for fine-tuning)
huggingface-cli download TaoLiveAIGC/Qwen3-TTS-TBLive-Base --local-dir ./Qwen3-TTS-TBLive-Base

# CustomVoice checkpoint (six built-in speakers)
huggingface-cli download TaoLiveAIGC/Qwen3-TTS-TBLive-CustomVoice --local-dir ./Qwen3-TTS-TBLive-CustomVoice
```

---

## Instruction Control Tokens

> Available only with `Qwen3-TTS-TBLive-CustomVoice` and its six built-in speakers.

Insert the following tokens directly into the input text to control pauses and speaking
rate; they are interpreted inline at the position where they appear.

**Silence / pause tokens** — each token synthesizes a pause within a fixed duration band:

| Token | Pause duration |
|---|---|
| `<sil_L2>` | 0.55 – 0.70 s |
| `<sil_L3>` | 0.80 – 0.95 s |
| `<sil_L4>` | 1.05 – 1.20 s |
| `<sil_L5>` | 1.30 – 1.45 s |
| `<sil_L6>` | 1.55 – 2.50 s |

**Speed tokens** — like silence tokens, they can be inserted anywhere in the text and
adjust the speaking rate of the speech that follows:

| Token | Effect |
|---|---|
| `[speed_e2]` | Faster speech |
| `[speed_d2]` | Slower speech |

Example:

```text
[speed_e2] 家人们，最后一波福利来了！<sil_L3> 库存不多，喜欢的抓紧下单！
```

```python
audio = model.generate_custom_voice(
    text="[speed_e2] 家人们，最后一波福利来了！<sil_L3> 库存不多，喜欢的抓紧下单！",
    speaker="4",  # 激情促销女
)
```

---

## Benchmark

Zero-shot voice-cloning results on the public
[seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval) test sets, compared
against open-source baselines. `/` marks numbers not reported.

| Model | test-en SIM-o ↑ | test-en WER ↓ | test-en UTMOS ↑ | test-zh SIM-o ↑ | test-zh WER ↓ | test-zh UTMOS ↑ |
|---|---|---|---|---|---|---|
| Ground-truth | 0.734 | 2.14 | 3.52 | 0.755 | 1.25 | 2.78 |
| IndexTTS2 | 0.706 | 2.33 | 3.65 | 0.764 | 1.05 | 3.00 |
| CosyVoice3 | 0.696 | 2.17 | 3.96 | 0.778 | 1.14 | 3.32 |
| VoxCPM | 0.731 | 1.92 | 3.77 | 0.772 | 0.99 | 2.94 |
| MossTTS Local | 0.732 | 1.93 | / | 0.796 | 1.44 | / |
| Qwen3-TTS | 0.708 | 1.54 | 4.16 | 0.766 | 1.15 | 3.46 |
| `Qwen3-TTS-TBLive-Base` | 0.732 | 1.63 | 4.11 | 0.782 | 1.28 | 3.38 |

---

## Demo

**▶ Online demo: [https://taoliveaigc.github.io/Qwen3-TTS-TBLive/](https://taoliveaigc.github.io/Qwen3-TTS-TBLive/)**

Base vs. `TBLive` A/B samples across three live-streaming styles, plus instruction-control and digital-human video demos.

---

## Quickstart

### 1. Environment setup

```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts

# Upstream runtime: qwen-tts wheel (pulls transformers / torch / etc.)
pip install -U qwen-tts

# This repository (training & evaluation extras)
git clone https://github.com/TaoLiveAIGC/Qwen3-TTS-TBLive.git
cd Qwen3-TTS-TBLive
pip install -r requirements.txt

# Optional: FlashAttention-2 for reduced memory footprint
pip install -U flash-attn --no-build-isolation
```

Additional external dependencies:

- **WeSpeaker** — required for SSIM scoring. Install from
  <https://github.com/wenet-e2e/wespeaker> and point `--wespeaker-path` at a downloaded
  Chinese speaker-embedding checkpoint.
- **FunASR** (auto-installed via `pip`) — provides the Paraformer checkpoint used for CER
  scoring; point `--asr-model-path` at the model directory.

### 2. Prepare training data

Refer to [`finetuning/README.md`](finetuning/README.md) for the JSONL schema (inherited from
the upstream single-speaker SFT) and the on-the-fly audio-code extraction path implemented in
`dataset_sep_file.py`.

### 3. Supervised fine-tuning

```bash
# Edit paths at the top of the script, then launch:
bash finetuning/run_hf.sh
```

### 4. GRPO / GDPO post-training

```bash
bash finetuning/run_grpo.sh
```

### 5. Inference with post-selection

```bash
# Zero-shot voice cloning with a 3-second reference + SSIM/CER post-selection
bash run_inference_postprocess.sh

# CustomVoice (built-in speaker name) + SSIM/CER post-selection
bash run_inference_custom_voice_postprocess.sh
```

Both scripts run the full two-stage pipeline by default — generate `NUM_CANDIDATES`
candidates per sentence, keep the top `SSIM_KEEP` by speaker similarity, then pick the
lowest-CER one (8 → 4 for zero-shot, 12 → 6 for CustomVoice). To fall back to single-shot
decoding, set `USE_POSTPROCESS=false` and `NUM_CANDIDATES=1` at the top of the script;
`SSIM_ONLY=true` skips the CER stage and returns the highest-SSIM candidate.

For plain single-shot inference (voice clone / voice design / custom voice / tokenizer usage),
please consult the
[upstream Quickstart](https://github.com/QwenLM/Qwen3-TTS#quickstart).

---

## Repository layout

```
.
├── qwen_tts/                       # Upstream model / tokenizer / inference (Alibaba Qwen team)
│   ├── core/                       # Model & tokenizer implementations
│   └── inference/                  # High-level Qwen3TTSModel + tokenizer API
├── finetuning/                     # Training pipelines
│   ├── sft_12hz.py                 # Upstream single-speaker SFT reference
│   ├── sft_base_hf.py              # HuggingFace-Trainer-based SFT (this repository)
│   ├── grpo_train.py               # GRPO / GDPO entry point (this repository)
│   ├── register_speaker.py         # Bake speakers into a checkpoint for generate_custom_voice
│   ├── hf_trainer/                 # Trainer / rollout / reward wrappers
│   ├── dataset*.py, prepare_data.py, text_augment.py
│   ├── run_hf.sh                   # SFT launcher
│   └── run_grpo.sh                 # GRPO launcher
├── evaluation/                     # SSIM + CER post-selection & long-text segmentation
├── examples/                       # Inference examples
│   ├── inference_zero_shot_postprocess.py     # Zero-shot + SSIM/CER
│   ├── inference_custom_voice_postprocess.py  # CustomVoice + SSIM/CER
│   └── test_model_12hz_*.py                   # Upstream reference examples
├── assets/                         # Demo samples + generation manifest (see Demo)
├── run_inference_postprocess.sh
└── run_inference_custom_voice_postprocess.sh
```

---

## Documentation index

- [Fine-tuning — HuggingFace-Trainer SFT + GRPO](finetuning/README.md)
- [Evaluation & post-selection](evaluation/README.md)
- [Upstream Qwen3-TTS README](https://github.com/QwenLM/Qwen3-TTS#readme) — consult for:
  - Voice-Clone / Voice-Design / CustomVoice API details
  - Audio-tokenizer usage (12 Hz / 25 Hz)
  - **vLLM serving and inference acceleration**
  - **DashScope managed inference API**
  - Deployment recipes (Docker, Gradio demo, model-card snippets)

---

## License

Released under the [Apache License 2.0](LICENSE), consistent with upstream
[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS). Files under `qwen_tts/` and
`finetuning/{prepare_data,dataset,sft_12hz}.py` originate from upstream and retain their
original copyright headers; all other files are new contributions under the same license.

### Acknowledgements

Our sincere thanks to the Qwen team for open-sourcing
[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — the base model, tokenizer, and
reference training / inference code that make this work possible.
