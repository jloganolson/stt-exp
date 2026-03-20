from __future__ import annotations

import time
from dataclasses import dataclass

from moonshine_voice import (
    LineCompleted,
    LineTextChanged,
    ModelArch,
    TranscriptEventListener,
    Transcriber,
    get_model_for_language,
)

from stt_exp.audio import AudioClip
from stt_exp.providers.base import ProviderResult, ProviderTraceEvent, RealtimeProvider


@dataclass(slots=True)
class MoonshineRealtimeConfig:
    language: str
    model_arch: int | None
    chunk_ms: int
    pace: str
    update_interval_s: float
    trailing_silence_ms: int


class _MoonshineListener(TranscriptEventListener):
    def __init__(self, result: ProviderResult):
        self.result = result

    def on_line_text_changed(self, event: LineTextChanged) -> None:
        now = time.perf_counter()
        text = event.line.text.strip()
        if not text:
            return
        if self.result.first_text_at is None:
            self.result.first_text_at = now
        self.result.transcript_text = text
        self.result.events.append(
            ProviderTraceEvent(ts_s=now, type="transcription.delta", text=text)
        )

    def on_line_completed(self, event: LineCompleted) -> None:
        now = time.perf_counter()
        text = event.line.text.strip()
        if not text:
            return
        if self.result.first_text_at is None:
            self.result.first_text_at = now
        self.result.transcript_text = text
        self.result.final_at = now
        self.result.events.append(
            ProviderTraceEvent(ts_s=now, type="transcription.done", text=text)
        )


class MoonshineRealtimeProvider(RealtimeProvider):
    name = "moonshine"

    def __init__(self, config: MoonshineRealtimeConfig):
        self.config = config
        wanted_arch = None if config.model_arch is None else ModelArch(config.model_arch)
        model_path, model_arch = get_model_for_language(
            wanted_language=config.language,
            wanted_model_arch=wanted_arch,
        )
        self.model_path = model_path
        self.model_arch = model_arch

    def transcribe(self, audio_clip: AudioClip, label: str) -> ProviderResult:
        result = ProviderResult(
            provider=self.name,
            transcript_text="",
            meta={
                "label": label,
                "language": self.config.language,
                "model_path": self.model_path,
                "model_arch": int(self.model_arch),
                "chunk_ms": self.config.chunk_ms,
                "pace": self.config.pace,
            },
        )
        listener = _MoonshineListener(result)

        with Transcriber(
            model_path=self.model_path,
            model_arch=self.model_arch,
            update_interval=self.config.update_interval_s,
        ) as transcriber:
            stream = transcriber.create_stream(update_interval=self.config.update_interval_s)
            stream.add_listener(listener)
            stream.start()

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
                stream.add_audio(chunk.astype(float).tolist(), audio_clip.sample_rate)
                if self.config.pace == "realtime":
                    time.sleep(self.config.chunk_ms / 1000.0)

            if self.config.trailing_silence_ms > 0:
                silence = [0.0] * int(audio_clip.sample_rate * self.config.trailing_silence_ms / 1000)
                stream.add_audio(silence, audio_clip.sample_rate)
                if self.config.pace == "realtime":
                    time.sleep(self.config.trailing_silence_ms / 1000.0)

            final_transcript = stream.stop()
            if final_transcript and final_transcript.lines:
                final_text = " ".join(line.text.strip() for line in final_transcript.lines if line.text.strip()).strip()
                if final_text:
                    result.transcript_text = final_text
                    result.final_at = time.perf_counter()
                    result.events.append(
                        ProviderTraceEvent(ts_s=result.final_at, type="transcription.done", text=final_text)
                    )
            stream.close()

        return result
