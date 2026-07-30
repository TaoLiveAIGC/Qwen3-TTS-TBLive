# Evaluation & post-selection utilities

Scoring and selection helpers shared by inference and GRPO training.

Modules:

| Module | Purpose |
|--------|---------|
| `speaker_similarity.py` | Cosine speaker similarity vs. a reference audio (WeSpeaker). |
| `cer_selector.py` | ASR + CER against ground-truth text (FunASR Paraformer). |
| `audio_selector.py` | Two-stage SSIM → CER selector over multiple TTS candidates. |
| `text_normalize.py` | Text normalization (CJK / ASCII / digit → `cn2an`) used before CER. |
| `text_segmentation.py` | Long-text splitter into utterance-sized chunks. |
| `audio_utils.py` | Resampling / tensor helpers. |

Used in two places:

- **Inference** (`examples/inference_*_postprocess.py`) — pick the best
  candidate out of N TTS rollouts per sentence, plus long-text
  segmentation.
- **Training** (`finetuning/`) — the same scorers provide the SSIM / CER
  signals in the GRPO reward.

External model dependencies (paths passed in at construction time, not
bundled): a WeSpeaker Chinese checkpoint for SSIM and a FunASR Paraformer
model for CER.
