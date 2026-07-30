# coding=utf-8

from __future__ import annotations

import json
import os
import random
import time
from typing import Any, List, Tuple, Union

import librosa
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from dataset_utils import SpeechTokenExtractor, add_special_tokens_to_tokenizer, natural_language_instruction, resolve_path
from qwen_tts import Qwen3TTSTokenizer
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from text_augment import _clean_text_symbols, maybe_replace_with_phonetics
 

AudioLike = Union[
    str,
    np.ndarray,
    Tuple[np.ndarray, int],
]

MaybeList = Union[Any, List[Any]]


class TTSDataset(Dataset):
    """
    TTS training dataset: accepts a list of file paths (str). Each file is a JSON
    array of samples. Data is loaded into a queue and streamed; supports
    multi-rank file distribution.
    """

    DEFAULT_TOKENIZER_PATH = "Qwen/Qwen3-TTS-Tokenizer-12Hz"

    def __init__(
        self,
        data_list: List[str],
        processor: Any,
        config: Qwen3TTSConfig,
        tokenizer_path: str | None = None,
        lag_num: int = -1,
        max_steps: float = 6e6,
        mode: str = "train",
        flatten: bool = True,
        use_const_ref: bool = False,
        data_split_mode: str = "files",
        icl_mode: bool = False,
        pinyin_replace_max: int = 0,
        pinyin_replace_prob: float = 0.0,
        pinyin_replace_mode: str = "pinyin",
    ) -> None:
        self.processor = processor
        self.config = config
        self.lag_num = lag_num
        self.max_train_step = int(max_steps)
        self.mode = mode
        self.default_sr = 24000
        # 440 会OOM
        # self.token_max_len = 420
        self.token_max_len = 10000 # 对于长度不再设置限制
        self.pinyin_replace_max = int(pinyin_replace_max)
        self.pinyin_replace_prob = float(pinyin_replace_prob)
        if pinyin_replace_mode not in ("pinyin", "phoneme"):
            raise ValueError(
                f"pinyin_replace_mode must be 'pinyin' or 'phoneme', "
                f"got {pinyin_replace_mode!r}"
            )
        self.pinyin_replace_mode = pinyin_replace_mode
        self.flatten = flatten
        self.use_const_ref = use_const_ref
        self.icl_mode = icl_mode
        if data_split_mode not in ("auto", "files", "samples"):
            raise ValueError(f"data_split_mode must be auto/files/samples, got {data_split_mode!r}")
        self.data_split_mode = data_split_mode

        self._get_dist_info()
        device_map = f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        path = tokenizer_path or self.DEFAULT_TOKENIZER_PATH
        print(f"[TTSDataset] rank={self.rank}, load speech tokenizer from {tokenizer_path}")
        self.tokenizer = Qwen3TTSTokenizer.from_pretrained(
            path, device_map=device_map
        )
        # add special token for self.processor
        num_added = add_special_tokens_to_tokenizer(self.processor.tokenizer)
        self.speech_token_extractor = SpeechTokenExtractor(self.tokenizer, 
                                                        default_sr=self.default_sr)

        self.list_data_files = [str(p).strip() for p in data_list if str(p).strip()]
        # Defensive check: catch the common mistake of passing raw JSON content
        # (e.g. `open(train.json).readlines()`) instead of a path list.
        if self.list_data_files:
            first = self.list_data_files[0]
            if first.startswith(("[", "{", "\"")):
                raise ValueError(
                    f"data_list looks like JSON content, not file paths "
                    f"(first entry starts with {first[:20]!r}). "
                    f"If your dataset is a single JSON file, pass its path as a "
                    f"one-element list: TTSDataset(data_list=[json_path], ...)."
                )
        self._sample_split = self._resolve_split_mode()
        self._get_local_data_files()
        self.data_queue: List[dict] = []
        self.read_file_idx = 0
        self._update_data_queue()

    def _resolve_split_mode(self) -> bool:
        """Return True if samples should be sliced within files; False to split files across ranks."""
        if self.data_split_mode == "samples":
            return True
        if self.data_split_mode == "files":
            return False
        # auto: switch to sample-split when there aren't enough files to cover all ranks
        return len(self.list_data_files) < self.world_size

    def _get_dist_info(self) -> None:
        if "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.rank = int(os.environ["RANK"])
        elif dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        if "LOCAL_RANK" in os.environ:
            self.local_rank = int(os.environ["LOCAL_RANK"])
        elif torch.cuda.is_available():
            self.local_rank = torch.cuda.current_device()
        else:
            self.local_rank = 0

        print(
            f"[TTSDataset] world_size={self.world_size} rank={self.rank} "
            f"local_rank={self.local_rank}"
        )

    def _get_local_data_files(self) -> None:
        """Distribute data files across ranks (files mode) or share all files (samples mode)."""
        total = len(self.list_data_files)
        if total == 0:
            return
        if self._sample_split:
            print(f"[TTSDataset] rank={self.rank} sample-split mode: shares all {total} file(s), "
                  f"per-rank slicing inside _load_data_file (stride={self.world_size})")
            return
        per_rank = (total + self.world_size - 1) // self.world_size
        pad = per_rank * self.world_size - total
        if pad > 0:
            self.list_data_files.extend(
                self.list_data_files[i % total] for i in range(pad)
            )
        start = self.rank * per_rank
        self.list_data_files = self.list_data_files[start : start + per_rank]
        print(f"[TTSDataset] rank={self.rank} files-split mode: files_count={len(self.list_data_files)}")

    def _load_data_file(self, file_path: str) -> List[dict]:
        file_path = resolve_path(file_path)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        if self.flatten:
            print(f"[TTSDataset] flatten dataset, origin len is: {len(data)}.")
            data_list = []
            for item in data:
                if isinstance(item, list):
                    data_list.extend(item)
                else:
                    data_list.append(item)
            print(f"[TTSDataset] flatten dataset list[list] dict to list[dict], from {len(data)} items to {len(data_list)} dicts.")
            data = data_list

        if self._sample_split and self.world_size > 1:
            n_before = len(data)
            # Deterministic shuffle (same seed across ranks) so the rank stride is
            # balanced and there is no overlap between ranks.
            rng = random.Random(42)
            rng.shuffle(data)
            data = data[self.rank :: self.world_size]
            print(f"[TTSDataset] rank={self.rank} sample-split slice: {n_before} → {len(data)} "
                  f"(stride {self.world_size})")
        return data

    def _update_data_queue(self) -> None:
        if len(self.data_queue) > 0:
            return
        if not self.list_data_files:
            return
        if self.read_file_idx < len(self.list_data_files):
            path = self.list_data_files[self.read_file_idx]
            self.data_queue.extend(self._load_data_file(path))
            self.read_file_idx += 1
        else:
            self.read_file_idx = 0
            path = self.list_data_files[self.read_file_idx]
            self.data_queue.extend(self._load_data_file(path))
            self.read_file_idx += 1
        random.seed(int(time.time()))
        random.shuffle(self.data_queue)
        print(
            f"[TTSDataset] rank={self.rank} file_idx={self.read_file_idx - 1} "
            f"queue_size={len(self.data_queue)}"
        )

    def __len__(self) -> int:
        if self.mode == "train" and self.max_train_step > 0:
            return self.max_train_step
        return len(self.data_queue)

    def _get_item_info(self, item: dict) -> str:
        """Extract basic info from a data item for error reporting."""
        audio_path = item.get('path', item.get('audio_path', 'N/A'))
        ref_audio = item.get('const_ref_audio', item.get('ref_audio', 'N/A'))
        text = item.get('label_norm_text', item.get('norm_text', item.get('text', 'N/A')))
        if isinstance(text, str) and len(text) > 100:
            text = text[:100] + "..."
        return f"audio_path={audio_path}, ref_audio={ref_audio}, text={text}"

    def __getitem__(self, idx: int) -> dict:
        max_retries = 50
        for retry in range(max_retries):
            if len(self.data_queue) == 0:
                self._update_data_queue()

            item = self.data_queue.pop()
            try:
                if not self.flatten:
                    sample = self.speech_token_extractor.extract_speech_token(item)
                    random_sample = random.choice(sample)
                else:
                    ref_key = "const_ref_audio" if self.use_const_ref else "dynamic_ref_audio"
                    random_sample = self.speech_token_extractor.extract_speech_token_with_ref(item, ref_key=ref_key)

                sample = self._format_input_from_sample(random_sample)

                if not self._sample_length_check(sample):
                    continue

                return sample

            except Exception as e:
                print(
                    f"[TTSDataset] WARNING: Error processing sample (retry {retry + 1}/{max_retries}). "
                    f"Info: {self._get_item_info(item)}. "
                    f"Error: {type(e).__name__}: {e}"
                )
                continue

        raise RuntimeError(
            f"[TTSDataset] Failed to get a valid sample after {max_retries} retries."
        )
    
    def _sample_length_check(self, sample):
        token_len = sample['text_ids'].shape[1] + sample['audio_codes'].shape[0] + 8
        return token_len <= self.token_max_len

    def _load_audio_to_np(self, x: str) -> Tuple[np.ndarray, int]:
        x = resolve_path(x)
        audio, sr = librosa.load(x, sr=self.default_sr, mono=True)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        return audio.astype(np.float32), int(sr)

    def _normalize_audio_inputs(
        self, audios: Union[AudioLike, List[AudioLike]]
    ) -> List[Tuple[np.ndarray, int]]:
        if isinstance(audios, list):
            items = audios
        else:
            items = [audios]
        out: List[Tuple[np.ndarray, int]] = []
        for a in items:
            if isinstance(a, str):
                out.append(self._load_audio_to_np(a))
            elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
                out.append((a[0].astype(np.float32), int(a[1])))
            elif isinstance(a, np.ndarray):
                raise ValueError("For numpy waveform input, pass (audio, sr).")
            else:
                raise TypeError(f"Unsupported audio type: {type(a)}")
        return out

    def _build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _ensure_list(self, x: MaybeList) -> List[Any]:
        return x if isinstance(x, list) else [x]

    def _tokenize_texts(self, text: str) -> torch.Tensor:
        out = self.processor(text=text, return_tensors="pt", padding=True)
        input_id = out["input_ids"]
        if input_id.dim() == 1:
            input_id = input_id.unsqueeze(0)
        return input_id

    @torch.inference_mode()
    def extract_mels(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        if sr != 24000:
            raise ValueError("Only 24kHz audio is supported.")
        mels = mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0),
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2)
        return mels

    def _format_input_from_sample(self, item: dict) -> dict:
        # 优先考虑带标签数据进行训练
        if "label_norm_text" in item:
            text = item.get("label_norm_text")
            # 将文本格式和SFT模型统一化，去除前后边界，以及连续标签的边界，"[][][]text<>" 的格式进行训练，不放置其余token
            text = text.replace("{", "").replace("}", "<end_ins>").replace(": ", "").replace("],[", "][")
        else:
            # 当 norm_text 与 text 同时存在时随机选一个，增加文本多样性；
            # 仅其中一个存在时退化为原行为。
            candidates = [
                item[k] for k in ("norm_text", "text")
                if k in item and item[k]
            ]
            if not candidates:
                text = ""
            elif len(candidates) == 1:
                text = candidates[0]
            else:
                text = random.choice(candidates)

        # 统一清洗辅助符号（两路文本源都要做）
        text = _clean_text_symbols(text)

        # 拼音/音素替换（覆盖所有中文字，由 G2PW 上下文消歧）。两段门控：
        # - prob: 该样本是否进入替换链路（外层），prob<=0 / max<=0 时为 no-op。
        # - max:  进入链路后从 randint(1, max) 采样实际替换字数。
        text = maybe_replace_with_phonetics(
            text,
            max_n=self.pinyin_replace_max,
            prob=self.pinyin_replace_prob,
            mode=self.pinyin_replace_mode,
        )
        
        # # 使用自然语言作为指令信息，对比使用embedding
        # text = natural_language_instruction(text)

        audio_codes = item["audio_codes"]
        ref_audio_path = item["ref_audio"]

        # ICL 模式：将 ref_text 拼到 text 前面，ref_codes 拼到 audio_codes 前面。
        # icl_mode=False 时跳过拼接，只让 ref_audio 走 speaker_encoder 得到
        # speaker_embedding（x-vector-only 模式），与推理侧只给 prompt 音频对齐。
        if self.icl_mode and "ref_codes" in item and item["ref_codes"] is not None:
            text = item["ref_text"] + text
            audio_codes = item["ref_codes"] + audio_codes

        text = self._build_assistant_text(text)
        text_ids = self._tokenize_texts(text)
        # import pdb; pdb.set_trace()

        audio_codes_t = torch.tensor(audio_codes, dtype=torch.long)

        ref_audio_list = self._ensure_list(ref_audio_path)
        normalized = self._normalize_audio_inputs(ref_audio_list)
        wav, sr = normalized[0]
        ref_mel = self.extract_mels(audio=wav, sr=sr)
        # remove "<|im_end|>\n<|im_start|>assistant\n" for text_ids[:, :-5]
        return {
            "text_ids": text_ids[:, :-5], 
            "audio_codes": audio_codes_t,
            "ref_mel": ref_mel,
        }

    def collate_fn(self, batch):
        assert self.lag_num == -1

        item_length = [b['text_ids'].shape[1] + b['audio_codes'].shape[0] for b in batch]
        max_length = max(item_length) + 8
        b,t = len(batch),max_length

        input_ids   = torch.zeros((b,t,2),dtype=torch.long)
        codec_ids   = torch.zeros((b,t,16),dtype=torch.long)
        text_embedding_mask     = torch.zeros((b,t),dtype=torch.bool)
        codec_embedding_mask    = torch.zeros((b,t),dtype=torch.bool)
        codec_mask      = torch.zeros((b,t),dtype=torch.bool)
        attention_mask  = torch.zeros((b,t),dtype=torch.long)
        codec_0_labels  = torch.full((b, t), -100, dtype=torch.long)

        for i,data in enumerate(batch):
            text_ids        = data['text_ids']
            audio_codec_0   = data['audio_codes'][:,0]
            audio_codecs    = data['audio_codes']

            text_ids_len = text_ids.shape[1]
            codec_ids_len = audio_codec_0.shape[0]
            
            # text channel
            input_ids[i,  :3, 0] = text_ids[0,:3] # <|im_start|>assistant\n
            input_ids[i, 3:7, 0] = self.config.tts_pad_token_id
            input_ids[i,   7, 0] = self.config.tts_bos_token_id
            input_ids[i, 8:8+text_ids_len-3, 0] = text_ids[0,3:]
            input_ids[i,   8+text_ids_len-3, 0] = self.config.tts_eos_token_id
            input_ids[i, 8+text_ids_len-2:8+text_ids_len+codec_ids_len , 0] = self.config.tts_pad_token_id
            text_embedding_mask[i,  :8+text_ids_len+codec_ids_len] = True

            # codec channel
            # input_ids[i,   :3, 1] = 0
            input_ids[i,    3:8 ,1] = torch.tensor(
                                        [
                                            self.config.talker_config.codec_nothink_id,
                                            self.config.talker_config.codec_think_bos_id,
                                            self.config.talker_config.codec_think_eos_id,
                                            0,     # for speaker embedding
                                            self.config.talker_config.codec_pad_id       
                                        ]
                                    )
            input_ids[i,    8:8+text_ids_len-3  ,1] = self.config.talker_config.codec_pad_id
            input_ids[i,    8+text_ids_len-3    ,1] = self.config.talker_config.codec_pad_id
            input_ids[i,    8+text_ids_len-2    ,1] = self.config.talker_config.codec_bos_id
            input_ids[i,    8+text_ids_len-1:8+text_ids_len-1+codec_ids_len,    1] = audio_codec_0
            input_ids[i,    8+text_ids_len-1+codec_ids_len,    1] = self.config.talker_config.codec_eos_token_id

            codec_0_labels[i,    8+text_ids_len-1:8+text_ids_len-1+codec_ids_len] = audio_codec_0
            codec_0_labels[i,    8+text_ids_len-1+codec_ids_len] = self.config.talker_config.codec_eos_token_id

            codec_ids[i, 8+text_ids_len-1:8+text_ids_len-1+codec_ids_len,:] = audio_codecs

            codec_embedding_mask[i, 3:8+text_ids_len+codec_ids_len] = True
            codec_embedding_mask[i, 6] = False       # for speaker embedding

            codec_mask[i,   8+text_ids_len-1:8+text_ids_len-1+codec_ids_len] = True
            attention_mask[i, :8+text_ids_len+codec_ids_len] = True
        
        ref_mels = [data['ref_mel'] for data in batch]
        # ref_mels = torch.cat(ref_mels,dim=0)
        

        return {
            'input_ids':input_ids,
            'ref_mels':ref_mels,
            'attention_mask':attention_mask,
            'text_embedding_mask':text_embedding_mask.unsqueeze(-1),
            'codec_embedding_mask':codec_embedding_mask.unsqueeze(-1),
            'codec_0_labels':codec_0_labels,
            'codec_ids': codec_ids,
            'codec_mask':codec_mask
        }