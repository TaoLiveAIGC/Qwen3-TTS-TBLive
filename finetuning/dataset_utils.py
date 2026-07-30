# coding=utf-8

import json
import os
import random
import shutil
from typing import List

import torch
from qwen_tts import Qwen3TTSTokenizer
from safetensors.torch import load_file, save_file
from transformers import PreTrainedTokenizer

def _load_path_fallbacks() -> List[tuple]:
    """Parse PATH_FALLBACK_MAP env var into (src_prefix, dst_prefix) pairs.

    Format: "src1:dst1,src2:dst2". Empty / unset yields no fallbacks.
    """
    spec = os.environ.get("PATH_FALLBACK_MAP", "").strip()
    if not spec:
        return []
    pairs = []
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        src, dst = entry.split(":", 1)
        src, dst = src.strip(), dst.strip()
        if src and dst:
            pairs.append((src, dst))
    return pairs


def resolve_path(path: str) -> str:
    """Return `path` if it exists, otherwise apply PATH_FALLBACK_MAP substitutions.

    Useful when the same dataset is mounted under different prefixes across
    machines. Set `PATH_FALLBACK_MAP="/prefix_a:/prefix_b,/prefix_b:/prefix_a"`
    in the environment to enable bidirectional fallback.
    """
    path = path.strip()
    if os.path.exists(path):
        return path
    for src, dst in _load_path_fallbacks():
        if src in path:
            fallback = path.replace(src, dst, 1)
            if fallback != path and os.path.exists(fallback):
                return fallback
    return path

# 添加结束符号与第三等级标签符号
DEFAULT_SPECIAL_TOKENS=[
    "[energy_low]",  "[energy_mid]",  "[energy_high]",
    "[pitch_low]",   "[pitch_mid]",   "[pitch_high]",
    "[speed_low]",   "[speed_mid]",   "[speed_high]",
    "[energy_d1]", "[energy_d2]", "[energy_e1]", "[energy_e2]",
    "[pitch_rate_d1]",  "[pitch_rate_d2]",  "[pitch_rate_e1]",  "[pitch_rate_e2]",
    "[speed_d1]",  "[speed_d2]",  "[speed_e1]",  "[speed_e2]",
    "[speed_d3]", "[speed_e3]", "[pitch_rate_d3]", "[pitch_rate_e3]", "[energy_d3]", "[energy_e3]",
    "<end_ins>",
]

def natural_language_instruction(label_text):
    """
    TODO: 后续设置合理的自然语言指令即对模型进行微调以待改进
    """
    # replace_dict = {
    #     "[energy_e2]": "[能量增强]",
    #     "[energy_d2]": "[能量减弱]",
    #     "[speed_e2]": "[语速加快]",
    #     "[speed_d2]": "[语速变慢]",
    # }
    
    # # 遍历替换字典，逐个替换
    # for key, val in replace_dict.items():
    #     label_text = label_text.replace(key, val)
    
    # return label_text
    return label_text
    
def add_special_tokens_to_tokenizer(
    tokenizer: PreTrainedTokenizer,
    special_tokens: List[str] = DEFAULT_SPECIAL_TOKENS,
) -> int:
    """
    以 special token 方式添加 token 到 tokenizer
    
    Args:
        special_tokens: 要添加的特殊 token 列表
        tokenizer: 预训练的 tokenizer
    
    Returns:
        int: 实际添加的新 token 数量
    """
    # 过滤已存在的 token
    existing_tokens = set(tokenizer.get_vocab().keys())
    new_tokens = [token for token in special_tokens if token not in existing_tokens]
    
    if not new_tokens:
        print("所有 token 已存在于词表中，无需添加")
        return 0
    
    # 使用 add_special_tokens 添加
    # 需要将列表包装为 additional_special_tokens
    num_added = tokenizer.add_special_tokens({
        "additional_special_tokens": new_tokens
    })
    
    print(f"成功添加 {num_added} 个 special token: {new_tokens}")
    return num_added


class SpeechTokenExtractor:
    """
    Speech token extractor class: based on Qwen3TTSTokenizer, extract speech token from audio.

    Args:
        tokenizer: Qwen3TTSTokenizer instance, used to encode audio to discrete codes.
        batch_size: batch size for encoding, default 32.
        default_sr: default sample rate for audio, default 24000.
    """

    def __init__(self, tokenizer: Qwen3TTSTokenizer, batch_size: int = 32, default_sr: int = 24000):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.default_sr = default_sr

    def extract_speech_token(self, data: List[dict]) -> List[dict]:
        """
        Extract speech token from audio. Frees CUDA memory after each batch to reduce peak GPU usage.

        Args:
            data: list of dict, each dict must contain "path" field, representing the audio file path.

        Returns:
            list of dict, each dict contains "audio_codes" and "ref_audio".
            If the item already has a non-empty "ref_audio" field, it is preserved;
            otherwise ref_audio is randomly chosen from other items in the group.
        """
        all_paths = [resolve_path(d.get('path', d.get('audio_path'))) for d in data]
        if not all_paths:
            return []

        result = []
        batch_items = []
        batch_paths = []

        for item in data:
            batch_items.append(item)
            batch_paths.append(resolve_path(item.get('path', item.get('audio_path'))))

            if len(batch_items) >= self.batch_size:
                enc_res = self.tokenizer.encode(batch_paths, sr=self.default_sr)
                for code, line in zip(enc_res.audio_codes, batch_items):
                    out = dict(line)
                    out["audio_codes"] = code.cpu().tolist()
                    if not out.get("ref_audio"):
                        out["ref_audio"] = random.choice(all_paths)
                    result.append(out)
                del enc_res
                batch_items.clear()
                batch_paths.clear()

        if len(batch_paths) > 0:
            enc_res = self.tokenizer.encode(batch_paths, sr=self.default_sr)
            for code, line in zip(enc_res.audio_codes, batch_items):
                out = dict(line)
                out["audio_codes"] = code.cpu().tolist()
                if not out.get("ref_audio"):
                    out["ref_audio"] = random.choice(all_paths)
                result.append(out)
            del enc_res

        return result
    
    def extract_speech_token_with_ref(self, data: dict, ref_key: str = "const_ref_audio") -> dict:
        """
        Extract speech tokens for both target audio and reference audio from a single sample.

        Args:
            data: dict containing audio path ('path'/'audio_path'), reference audio path
                  (ref_key, e.g. 'const_ref_audio'), and reference text ('{ref_key}_text').
            ref_key: key for the reference audio path field.
                     If 'dynamic_ref_audio', randomly selects from meta.json candidates
                     in the speaker's ref_audios directory.

        Returns:
            dict: original data fields plus 'audio_codes', 'ref_audio', 'ref_text', 'ref_codes'.
        """
        target_path = resolve_path(data.get('path', data.get('audio_path')))

        if ref_key == "dynamic_ref_audio":
            # Use const_ref_audio's parent dir to locate meta.json (ref_audios/speaker_id/)
            const_ref_path = data.get('const_ref_audio', data.get('dynamic_ref_audio', ''))
            ref_path, ref_text = self._get_random_ref_from_meta(const_ref_path)
        else:
            ref_path = resolve_path(data.get(ref_key, ''))
            ref_text = data.get(f"{ref_key}_text", "")

        enc_res = self.tokenizer.encode([target_path, ref_path], sr=self.default_sr)

        out = dict(data)
        out["audio_codes"] = enc_res.audio_codes[0].cpu().tolist()
        out["ref_audio"] = ref_path
        out["ref_text"] = ref_text
        out["ref_codes"] = enc_res.audio_codes[1].cpu().tolist()
        del enc_res

        return out

    def _get_random_ref_from_meta(self, ref_audio_path: str) -> tuple:
        """
        Read meta.json from the parent directory of a reference audio path
        (i.e., ref_audios/speaker_id/), and randomly select one candidate.

        Args:
            ref_audio_path: path to any reference audio file (e.g. const_ref_audio),
                           used to locate the speaker's meta.json.

        Returns:
            (ref_path, ref_text): path and text of the randomly selected reference audio.
        """
        ref_audio_path = resolve_path(ref_audio_path)
        parent_dir = os.path.dirname(ref_audio_path)
        meta_path = os.path.join(parent_dir, "meta.json")
        meta_path = resolve_path(meta_path)

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        candidates = meta.get("candidates", [])
        if not candidates:
            raise ValueError(f"No candidates found in {meta_path}")

        chosen = random.choice(candidates)
        ref_path = resolve_path(chosen.get("path", chosen.get("audio_path", "")))
        ref_text = chosen.get("text", "")
        return ref_path, ref_text


def get_latest_checkpoint(output_dir: str) -> str | None:
    """Return path to checkpoint dir with largest step (checkpoint-{step}), or None."""
    if not os.path.isdir(output_dir):
        return None
    best_step = -1
    best_path = None
    for name in os.listdir(output_dir):
        if name.startswith("checkpoint-") and name != "checkpoint-epoch-":
            rest = name[len("checkpoint-"):]
            if rest.isdigit():
                step = int(rest)
                if step > best_step:
                    best_step = step
                    best_path = os.path.join(output_dir, name)
    return best_path


def load_latest_checkpoint(accelerator, model, optimizer, scheduler, output_dir: str):
    """
    Load model, optimizer, scheduler and training state from latest checkpoint under output_dir.
    Returns (global_step, start_epoch, loaded). loaded=True if at least model was loaded.
    """
    resume_dir = get_latest_checkpoint(output_dir)
    if not resume_dir or not os.path.isfile(os.path.join(resume_dir, "model.safetensors")):
        return 0, 0, False

    accelerator.print(f"Resuming from {resume_dir}")
    unwrapped = accelerator.unwrap_model(model)
    state = load_file(os.path.join(resume_dir, "model.safetensors"))
    unwrapped.load_state_dict(state, strict=True)

    global_step, start_epoch = 0, 0
    device = next(unwrapped.parameters()).device
    opt_path = os.path.join(resume_dir, "optimizer.pt")
    if os.path.isfile(opt_path):
        optimizer.load_state_dict(torch.load(opt_path, map_location=device))
    sched_path = os.path.join(resume_dir, "scheduler.pt")
    if os.path.isfile(sched_path):
        scheduler.load_state_dict(torch.load(sched_path, map_location=device))
    ts_path = os.path.join(resume_dir, "training_state.json")
    if os.path.isfile(ts_path):
        with open(ts_path) as f:
            ts = json.load(f)
        global_step = ts["global_step"]
        start_epoch = ts["epoch"]
        accelerator.print(f"Resumed at global_step={global_step}, epoch={start_epoch}")

    return global_step, start_epoch, True


def save_checkpoint_base(
    accelerator,
    model,
    output_dir: str,
    init_model_path: str,
    optimizer=None,
    scheduler=None,
    global_step=None,
    epoch=None,
):
    """Save full base model; optionally optimizer/scheduler/training_state for resume."""
    shutil.copytree(init_model_path, output_dir, dirs_exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)
    state_dict = {
        k: v.detach().to("cpu")
        for k, v in unwrapped_model.state_dict().items()
    }
    save_path = os.path.join(output_dir, "model.safetensors")
    save_file(state_dict, save_path)
    if optimizer is not None and global_step is not None and epoch is not None:
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        if scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        with open(os.path.join(output_dir, "training_state.json"), "w") as f:
            json.dump({"global_step": global_step, "epoch": epoch}, f)
    accelerator.print(f"Checkpoint saved to {output_dir}")



