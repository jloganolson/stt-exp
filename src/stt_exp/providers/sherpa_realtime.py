from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass

import numpy as np
import sherpa_onnx

from stt_exp.audio import AudioClip
from stt_exp.providers.base import ProviderResult, ProviderTraceEvent, RealtimeProvider


@dataclass(slots=True)
class SherpaRealtimeConfig:
    model_dir: str
    provider: str
    num_threads: int
    chunk_ms: int
    pace: str
    warmup: bool
    trailing_silence_ms: int


def _find_model_file(model_dir: str, prefix: str) -> str:
    candidates = glob.glob(os.path.join(model_dir, f"{prefix}*.int8.onnx"))
    if not candidates:
        candidates = glob.glob(os.path.join(model_dir, f"{prefix}*.onnx"))
    if not candidates:
        raise RuntimeError(f"Could not find {prefix} model in {model_dir}")
    return sorted(candidates)[0]


def _create_recognizer(config: SherpaRealtimeConfig):
    feature_dim = 128 if "nemotron" in config.model_dir.lower() else 80
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=os.path.join(config.model_dir, "tokens.txt"),
        encoder=_find_model_file(config.model_dir, "encoder"),
        decoder=_find_model_file(config.model_dir, "decoder"),
        joiner=_find_model_file(config.model_dir, "joiner"),
        num_threads=config.num_threads,
        sample_rate=16000,
        feature_dim=feature_dim,
        decoding_method="greedy_search",
        provider=config.provider,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=300,
    )


class SherpaRealtimeProvider(RealtimeProvider):
    name = "sherpa"

    def __init__(self, config: SherpaRealtimeConfig):
        self.config = config
        self._recognizer = _create_recognizer(config)
        if config.warmup:
            self._warmup()

    def _warmup(self) -> None:
        stream = self._recognizer.create_stream()
        stream.accept_waveform(16000, np.zeros(int(1.5 * 16000), dtype=np.float32))
        while self._recognizer.is_ready(stream):
            self._recognizer.decode_stream(stream)
        stream.input_finished()
        while self._recognizer.is_ready(stream):
            self._recognizer.decode_stream(stream)

    def transcribe(self, audio_clip: AudioClip, label: str) -> ProviderResult:
        stream = self._recognizer.create_stream()
        result = ProviderResult(
            provider=self.name,
            transcript_text="",
            meta={
                "label": label,
                "model_dir": self.config.model_dir,
                "provider": self.config.provider,
                "chunk_ms": self.config.chunk_ms,
                "pace": self.config.pace,
            },
        )
        last_text = ""

        samples_per_chunk = max(1, int(audio_clip.sample_rate * self.config.chunk_ms / 1000))
        chunks = [
            audio_clip.samples[i : i + samples_per_chunk]
            for i in range(0, len(audio_clip.samples), samples_per_chunk)
        ]

        for chunk in chunks:
            now = time.perf_counter()
            if result.first_audio_sent_at is None:
                result.first_audio_sent_at = now
            result.last_audio_sent_at = now
            result.events.append(
                ProviderTraceEvent(ts_s=now, type="audio.append", meta={"samples": int(len(chunk))})
            )
            stream.accept_waveform(audio_clip.sample_rate, chunk.astype(np.float32))
            while self._recognizer.is_ready(stream):
                self._recognizer.decode_stream(stream)
            text = self._recognizer.get_result(stream).strip()
            if text and text != last_text:
                last_text = text
                if result.first_text_at is None:
                    result.first_text_at = time.perf_counter()
                result.transcript_text = text
                result.events.append(
                    ProviderTraceEvent(ts_s=time.perf_counter(), type="transcription.delta", text=text)
                )
            if self.config.pace == "realtime":
                time.sleep(self.config.chunk_ms / 1000.0)

        if self.config.trailing_silence_ms > 0:
            silence = np.zeros(int(audio_clip.sample_rate * self.config.trailing_silence_ms / 1000), dtype=np.float32)
            stream.accept_waveform(audio_clip.sample_rate, silence)

        stream.input_finished()
        while self._recognizer.is_ready(stream):
            self._recognizer.decode_stream(stream)

        final_text = self._recognizer.get_result(stream).strip() or last_text
        result.transcript_text = final_text
        result.final_at = time.perf_counter()
        if result.first_text_at is None and final_text:
            result.first_text_at = result.final_at
        result.events.append(
            ProviderTraceEvent(ts_s=result.final_at, type="transcription.done", text=final_text)
        )
        return result
