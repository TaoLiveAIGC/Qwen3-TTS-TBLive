# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""
Register speakers into an SFT-trained base model for custom_voice inference.

Reads a speaker_info.json, extracts speaker embeddings via speaker_encoder,
bakes them into codec_embedding weights, updates config.json with speaker
mappings, and saves a ready-to-use checkpoint.

speaker_info.json format:
{
    "speakers": [
        {
            "speaker_name": "Vivian",
            "speaker_id": 2800,
            "ref_audios": [
                {"audio": "/path/to/audio1.wav", "text": "对应文本1"},
                {"audio": "/path/to/audio2.wav", "text": "对应文本2"}
            ]
        },
        ...
    ]
}

Usage:
    python finetuning/register_speaker.py \
        --model-path /path/to/sft_checkpoint \
        --speaker-info speaker_info.json \
        --output-path /path/to/output_model
"""

import argparse
import json
import os
import shutil
import sys

import librosa
import numpy as np
import torch
from safetensors.torch import load_file, save_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from qwen_tts import Qwen3TTSModel


from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram


def load_and_resample(audio_path: str, target_sr: int = 24000) -> np.ndarray:
    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


def extract_ref_mel(audio: np.ndarray) -> torch.Tensor:
    """Extract mel spectrogram from 24kHz audio, same as training dataset pipeline."""
    mels = mel_spectrogram(
        torch.from_numpy(audio).unsqueeze(0),
        n_fft=1024,
        num_mels=128,
        sampling_rate=24000,
        hop_size=256,
        win_size=1024,
        fmin=0,
        fmax=12000,
    ).transpose(1, 2)  # (1, T, 128)
    return mels


@torch.no_grad()
def extract_speaker_embeddings(model, speaker_info: dict, device: str) -> dict:
    """Extract averaged speaker embeddings using the same ref_mel → speaker_encoder
    flow as the training dataset (dataset_sep_file.py).

    Returns: {speaker_id: embedding_tensor (enc_dim,)}
    """
    speaker_encoder = model.model.speaker_encoder
    spk_dtype = next(speaker_encoder.parameters()).dtype
    embeddings = {}

    for speaker in speaker_info["speakers"]:
        speaker_name = speaker["speaker_name"]
        speaker_id = speaker["speaker_id"]
        ref_audios = speaker["ref_audios"]

        print(f"[{speaker_name}] Extracting embeddings from {len(ref_audios)} reference audios...")

        emb_list = []
        for ref in ref_audios:
            audio_path = ref["audio"]
            if not os.path.isfile(audio_path):
                print(f"  WARNING: {audio_path} not found, skipping")
                continue

            audio = load_and_resample(audio_path, target_sr=24000)
            ref_mel = extract_ref_mel(audio)  # (1, T, 128)
            emb = speaker_encoder(ref_mel.to(device).to(spk_dtype))  # (1, enc_dim)
            emb_list.append(emb.squeeze(0).cpu())

        if not emb_list:
            raise ValueError(f"No valid reference audios for speaker '{speaker_name}'")

        avg_emb = torch.stack(emb_list).mean(dim=0)
        embeddings[speaker_id] = avg_emb
        print(f"  -> speaker_id={speaker_id}, averaged from {len(emb_list)} audios, "
              f"emb norm={avg_emb.norm().item():.4f}")

    return embeddings


def register_speakers(
    model_path: str,
    speaker_info_path: str,
    output_path: str,
    device: str = "cuda:0",
):
    with open(speaker_info_path, "r", encoding="utf-8") as f:
        speaker_info = json.load(f)

    speakers = speaker_info["speakers"]
    print(f"Registering {len(speakers)} speaker(s)...")

    for spk in speakers:
        sid = spk["speaker_id"]
        if sid < 2158 or sid > 3071:
            raise ValueError(
                f"speaker_id={sid} for '{spk['speaker_name']}' out of range [2158, 3071]"
            )

    # Load model (base type with speaker_encoder)
    print(f"\nLoading model from {model_path}...")
    tts_model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    print("Model loaded.")

    # Extract speaker embeddings
    print("\nExtracting speaker embeddings...")
    speaker_embeddings = extract_speaker_embeddings(tts_model, speaker_info, device)

    # Copy base model to output
    print(f"\nCopying model files to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    shutil.copytree(model_path, output_path, dirs_exist_ok=True)

    # Update config.json
    config_path = os.path.join(output_path, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    config_dict["tts_model_type"] = "custom_voice"

    speaker_name_to_id = {spk["speaker_name"]: spk["speaker_id"] for spk in speakers}
    speaker_is_dialect = {spk["speaker_name"]: spk.get("is_dialect", False) for spk in speakers}

    talker_config = config_dict.get("talker_config", {})
    talker_config["spk_id"] = speaker_name_to_id
    talker_config["spk_is_dialect"] = speaker_is_dialect
    config_dict["talker_config"] = talker_config

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    print(f"Updated config.json: tts_model_type=custom_voice, spk_id={speaker_name_to_id}")

    # Modify model weights: bake speaker embeddings into codec_embedding
    weights_path = os.path.join(output_path, "model.safetensors")
    print(f"\nLoading weights from {weights_path}...")
    state_dict = load_file(weights_path)

    codec_emb_key = "model.talker.model.codec_embedding.weight"
    if codec_emb_key not in state_dict:
        codec_emb_key = "talker.model.codec_embedding.weight"
    if codec_emb_key not in state_dict:
        available = [k for k in state_dict if "codec_embedding" in k]
        raise KeyError(f"codec_embedding key not found. Available: {available}")

    codec_weight = state_dict[codec_emb_key]
    print(f"codec_embedding shape: {codec_weight.shape}")

    for spk_id, emb in speaker_embeddings.items():
        emb_target = emb.to(codec_weight.device).to(codec_weight.dtype)
        codec_weight[spk_id] = emb_target
        print(f"  Wrote speaker_id={spk_id} into codec_embedding[{spk_id}]")

    state_dict[codec_emb_key] = codec_weight

    # Remove speaker_encoder weights (not needed for custom_voice inference)
    keys_to_drop = [k for k in state_dict if "speaker_encoder" in k]
    for k in keys_to_drop:
        del state_dict[k]
    if keys_to_drop:
        print(f"  Removed {len(keys_to_drop)} speaker_encoder keys (not needed for custom_voice)")

    # Save
    print(f"\nSaving modified weights to {weights_path}...")
    save_file(state_dict, weights_path)

    print(f"\nDone! Output model saved to: {output_path}")
    print(f"Registered speakers: {list(speaker_name_to_id.keys())}")
    print(f"Inference usage: generate_custom_voice(text=..., speaker='<name>')")


def main():
    parser = argparse.ArgumentParser(
        description="Register speakers into a base/SFT model for custom_voice inference"
    )
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the SFT-trained (or base) model checkpoint")
    parser.add_argument("--speaker-info", type=str, required=True,
                        help="Path to speaker_info.json")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Output directory for the registered model")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for speaker_encoder inference")
    args = parser.parse_args()

    register_speakers(
        model_path=args.model_path,
        speaker_info_path=args.speaker_info,
        output_path=args.output_path,
        device=args.device,
    )


if __name__ == "__main__":
    main()
