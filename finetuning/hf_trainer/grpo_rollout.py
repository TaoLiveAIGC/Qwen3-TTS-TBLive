# coding=utf-8

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import librosa
import numpy as np
import torch

from evaluation.cer_selector import CERConfig, CERSelector
from evaluation.speaker_similarity import SpeakerSimilaritySelector, SSIMConfig

from .grpo_config import GRPOTTSConfig
from .prosody_rewards import CpsReward, SemiFlReward, _count_pronounceable


class TTSGRPORolloutEngine:
    """Generates Qwen3-TTS candidates, filters via SSIM/CER, returns aligned arrays for GRPO."""

    def __init__(
        self,
        model,
        processor,
        speech_tokenizer,
        config: GRPOTTSConfig,
        device: str,
        unwrap_fn: Optional[Callable] = None,
    ):
        self.model = model
        self.processor = processor
        self.speech_tokenizer = speech_tokenizer
        self.config = config
        self.device = device
        self.unwrap_fn = unwrap_fn or (lambda m: m)

        self.ssim_selector = SpeakerSimilaritySelector(
            SSIMConfig(model_path=config.wespeaker_path, device=config.reward_device)
        )
        self.cer_selector = CERSelector(
            CERConfig(
                asr_model_path=config.asr_model_path,
                device=config.reward_device,
                batch_size=20,
                source_sample_rate=24000,
                language="zh",
                cer_threshold=config.cer_threshold,
                verbose=False,
                pick_strategy="medium",
            )
        )
        # Prosody reward dimensions (only initialized when enabled)
        self.cps_reward = CpsReward() if getattr(config, 'w_cps', 0) > 0 else None
        self.semi_fl_reward = SemiFlReward() if getattr(config, 'w_semi_fl', 0) > 0 else None
        self._ref_cache: Dict[str, Dict[str, Any]] = {}

    def _compose_reward(self, ssim_t: torch.Tensor, cer_t: torch.Tensor) -> torch.Tensor:
        """Gated reward with CER quality bonus.

        CER > threshold → penalty: reward = (1 - cer) * penalty_weight
        CER ≤ threshold → quality: reward = (1 - w) * ssim + w * (1 - cer)
            where w = reward_cer_quality_weight (0 = pure SSIM, backward-compatible).
        """
        cer_thresh = getattr(self.config, 'reward_cer_threshold', 0.3)
        cer_penalty_w = getattr(self.config, 'reward_cer_penalty_weight', 0.5)
        cer_quality_w = getattr(self.config, 'reward_cer_quality_weight', 0.0)

        normal_reward = (1.0 - cer_quality_w) * ssim_t + cer_quality_w * (1.0 - cer_t)
        penalty_reward = (1.0 - cer_t) * cer_penalty_w

        reward = torch.where(cer_t > cer_thresh, penalty_reward, normal_reward)
        return reward

    def _build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _build_ref_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n"

    def _tokenize_text(self, text: str) -> torch.Tensor:
        out = self.processor(text=text, return_tensors="pt", padding=True)
        input_id = out["input_ids"]
        if input_id.dim() == 1:
            input_id = input_id.unsqueeze(0)
        return input_id

    def _get_ref_features(self, ref_audio_path: str, ref_text: str) -> Dict[str, Any]:
        if ref_audio_path in self._ref_cache:
            return self._ref_cache[ref_audio_path]

        raw_model = self.unwrap_fn(self.model).model if hasattr(self.unwrap_fn(self.model), "model") else self.unwrap_fn(self.model)

        enc = self.speech_tokenizer.encode([ref_audio_path], sr=24000)
        ref_code = enc.audio_codes[0]

        audio, sr = librosa.load(ref_audio_path, sr=None, mono=True)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        target_sr = raw_model.speaker_encoder_sample_rate
        if sr != target_sr:
            audio = librosa.resample(y=audio.astype(np.float32), orig_sr=int(sr), target_sr=target_sr)
        spk_emb = raw_model.extract_speaker_embedding(audio=audio.astype(np.float32), sr=target_sr)

        ref_emb = self.ssim_selector.extract_reference_embedding(ref_audio_path)

        feats = {
            "ref_code": ref_code,
            "spk_emb": spk_emb,
            "ref_emb": ref_emb,
        }
        self._ref_cache[ref_audio_path] = feats
        return feats

    @torch.no_grad()
    def _generate_candidates(
        self,
        ref_audio_path: str,
        ref_text: str,
        target_text: str,
        ref_feats: Dict[str, Any],
        n: int,
        language: str = "Chinese",
    ):
        raw_model = self.unwrap_fn(self.model)
        if hasattr(raw_model, "model"):
            raw_model = raw_model.model

        use_icl = self.config.rollout_icl_mode

        input_text = self._build_assistant_text(target_text)
        target_input_id = self._tokenize_text(input_text).to(raw_model.device)

        if use_icl:
            # ICL: ref_text + target_text as prefix, pass ref_ids for generate()
            ref_text_full = self._build_ref_text(ref_text)
            ref_input_id = self._tokenize_text(ref_text_full).to(raw_model.device)
            combined_text = self._build_assistant_text(ref_text + target_text)
            combined_input_id = self._tokenize_text(combined_text)[:, :-5]
            ref_ids = [ref_input_id for _ in range(n)]
        else:
            # x-vector-only: only target_text, no ref_text prefix
            combined_text = self._build_assistant_text(target_text)
            combined_input_id = self._tokenize_text(combined_text)[:, :-5]
            ref_ids = None

        input_ids = [target_input_id for _ in range(n)]

        device = raw_model.device
        voice_clone_prompt = {
            "ref_code": [ref_feats["ref_code"].to(device) for _ in range(n)] if use_icl else [None] * n,
            "ref_spk_embedding": [ref_feats["spk_emb"].to(device) for _ in range(n)],
            "x_vector_only_mode": [not use_icl for _ in range(n)],
            "icl_mode": [use_icl for _ in range(n)],
        }
        languages = [language for _ in range(n)]

        generate_kwargs = dict(
            input_ids=input_ids,
            voice_clone_prompt=voice_clone_prompt,
            languages=languages,
            non_streaming_mode=False,
            max_new_tokens=self.config.max_completion_length,
            do_sample=True,
            top_k=self.config.gen_top_k,
            top_p=self.config.gen_top_p,
            temperature=self.config.gen_temperature,
            repetition_penalty=self.config.gen_repetition_penalty,
            subtalker_dosample=True,
            subtalker_top_k=self.config.subtalker_top_k,
            subtalker_top_p=self.config.subtalker_top_p,
            subtalker_temperature=self.config.subtalker_temperature,
        )
        if use_icl:
            generate_kwargs["ref_ids"] = ref_ids
        talker_codes_list, _ = raw_model.generate(**generate_kwargs)

        completions = []
        codes_for_decode = []
        if use_icl:
            # ICL: prepend ref_code for decoding, then trim ref audio prefix
            ref_code_t = ref_feats["ref_code"].to(device)
            for codes in talker_codes_list:
                completions.append(codes.detach().cpu().long())
                codes_for_decode.append(torch.cat([ref_code_t, codes], dim=0))
        else:
            # x-vector-only: decode completion directly, no trimming
            for codes in talker_codes_list:
                completions.append(codes.detach().cpu().long())
                codes_for_decode.append(codes)

        wavs_all, fs = raw_model.speech_tokenizer.decode(
            [{"audio_codes": c} for c in codes_for_decode]
        )

        wavs_out = []
        if use_icl:
            ref_code_t = ref_feats["ref_code"].to(device)
            for i, wav in enumerate(wavs_all):
                ref_len = int(ref_code_t.shape[0])
                total_len = int(codes_for_decode[i].shape[0])
                cut = int(ref_len / max(total_len, 1) * wav.shape[0])
                wav_t = torch.from_numpy(wav[cut:]).float()
                wavs_out.append(wav_t)
        else:
            for wav in wavs_all:
                wavs_out.append(torch.from_numpy(wav).float())

        return completions, wavs_out, fs, combined_input_id.detach().cpu()

    def _score_candidates(
        self,
        wavs: List[torch.Tensor],
        target_text: str,
        ref_emb: torch.Tensor,
        sample_rate: int,
    ):
        ssim_scores = []
        for wav in wavs:
            try:
                s = self.ssim_selector.compute_similarity(wav, ref_emb, sample_rate=sample_rate)
            except Exception:
                s = 0.0
            ssim_scores.append(float(s))

        from evaluation.audio_utils import to_numpy_16k

        try:
            audios_16k = to_numpy_16k(wavs, source_sr=sample_rate)
            asr_texts = self.cer_selector.transcribe(audios_16k)
            cers = self.cer_selector.compute_cer(asr_texts, [target_text] * len(wavs))
        except Exception:
            cers = [1.0] * len(wavs)

        # Prosody rewards (CPS + SemiFL)
        wav_np = [w.numpy() if isinstance(w, torch.Tensor) else w for w in wavs]
        if self.cps_reward is not None:
            cps_vals = self.cps_reward.compute(wav_np, [target_text] * len(wavs), sample_rate)
        else:
            cps_vals = [0.0] * len(wavs)

        if self.semi_fl_reward is not None and not getattr(self.config, 'skip_semi_fl_compute', False):
            semi_fl_vals = self.semi_fl_reward.compute(wav_np, sample_rate)
        else:
            semi_fl_vals = [0.0] * len(wavs)

        return ssim_scores, cers, cps_vals, semi_fl_vals

    def _compute_gt_cps(self, target_audio_path: str, target_text: str) -> float:
        """Compute ground-truth CPS from target audio duration and text."""
        try:
            import soundfile as sf
            info = sf.info(target_audio_path)
            gt_duration = info.duration
        except Exception:
            gt_duration = 0.0
        n_chars = _count_pronounceable(target_text)
        return n_chars / gt_duration if gt_duration > 0.1 else 5.0

    def rollout(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate G candidates per unique prompt and score with SSIM/CER/CPS/SemiFL."""
        G = self.config.num_generations
        assert len(batch) % G == 0, f"batch size {len(batch)} not divisible by num_generations {G}"
        n_prompts = len(batch) // G

        use_4dim = (getattr(self.config, 'w_cps', 0) > 0 or
                     getattr(self.config, 'w_semi_fl', 0) > 0)

        out_combined_text_ids: List[torch.Tensor] = []
        out_completion_codes: List[torch.Tensor] = []
        out_ref_codes: List[torch.Tensor] = []
        out_prompt_codec_len: List[int] = []
        out_ref_mels: List[torch.Tensor] = []
        out_ref_text: List[str] = []
        out_target_text: List[str] = []
        out_ssim: List[float] = []
        out_cer: List[float] = []
        out_cps: List[float] = []
        out_semi_fl: List[float] = []
        out_gt_cps: List[float] = []  # per-sample gt_cps (repeated G times per prompt)
        out_wavs: List[torch.Tensor] = []  # generated audio waveforms
        out_sample_rate: List[int] = []  # sample rate for each waveform

        for p_idx in range(n_prompts):
            sample = batch[p_idx * G]
            ref_audio_path = sample["ref_audio_path"]
            ref_text = sample["ref_text"]
            target_text = sample["target_text"]
            target_audio_path = sample.get("target_audio_path", ref_audio_path)
            ref_mel = sample["ref_mel"]

            ref_feats = self._get_ref_features(ref_audio_path, ref_text)
            gt_cps = self._compute_gt_cps(target_audio_path, target_text) if use_4dim else 0.0

            # 一次生成 G 个候选（不 retry，不 oversample）
            completions, wavs, fs, combined_input_id = self._generate_candidates(
                ref_audio_path=ref_audio_path,
                ref_text=ref_text,
                target_text=target_text,
                ref_feats=ref_feats,
                n=G,
            )
            ssim_scores, cers, cps_vals, semi_fl_vals = self._score_candidates(
                wavs=wavs,
                target_text=target_text,
                ref_emb=ref_feats["ref_emb"],
                sample_rate=fs,
            )

            # 全部 G 个都保留
            for comp, wav, s, c, cps_v, sfl_v in zip(completions, wavs, ssim_scores, cers, cps_vals, semi_fl_vals):
                out_combined_text_ids.append(combined_input_id)
                out_completion_codes.append(comp)
                out_wavs.append(wav)
                out_sample_rate.append(fs)
                if self.config.rollout_icl_mode:
                    out_ref_codes.append(ref_feats["ref_code"].detach().cpu().long())
                    out_prompt_codec_len.append(int(ref_feats["ref_code"].shape[0]))
                else:
                    out_ref_codes.append(torch.zeros(0, 16, dtype=torch.long))
                    out_prompt_codec_len.append(0)
                out_ref_mels.append(ref_mel)
                out_ref_text.append(ref_text)
                out_target_text.append(target_text)
                out_ssim.append(float(s))
                out_cer.append(float(c))
                out_cps.append(float(cps_v))
                out_semi_fl.append(float(sfl_v))
                out_gt_cps.append(float(gt_cps))

        ssim_t = torch.tensor(out_ssim, dtype=torch.float32)
        cer_t = torch.tensor(out_cer, dtype=torch.float32)
        cps_t = torch.tensor(out_cps, dtype=torch.float32)
        semi_fl_t = torch.tensor(out_semi_fl, dtype=torch.float32)
        gt_cps_t = torch.tensor(out_gt_cps, dtype=torch.float32)

        result = {
            "combined_text_ids": out_combined_text_ids,
            "completion_codec_ids": out_completion_codes,
            "ref_codes": out_ref_codes,
            "prompt_codec_len": out_prompt_codec_len,
            "ref_mels": out_ref_mels,
            "ref_text": out_ref_text,
            "target_text": out_target_text,
            "cer": cer_t,
            "ssim": ssim_t,
            "cps": cps_t,
            "semi_fl": semi_fl_t,
            "gt_cps": gt_cps_t,
            "wavs": out_wavs,
            "sample_rate": out_sample_rate[0] if out_sample_rate else 24000,
            "avg_attempts": 1.0,
        }

        if use_4dim:
            # 4-dim GDPO mode: trainer computes advantage from raw dimensions
            # rewards key is NOT included (trainer uses gdpo_4dim_aggregate)
            pass
        else:
            # 2-dim backward-compatible mode: compute gated reward here
            result["rewards"] = self._compose_reward(ssim_t, cer_t)

        return result
