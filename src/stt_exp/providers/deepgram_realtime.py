from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

from dotenv import load_dotenv

from stt_exp.audio import AudioClip, iter_pcm16_chunks, silence_pcm16
from stt_exp.providers.base import ProviderResult, ProviderTraceEvent, RealtimeProvider

load_dotenv()


@dataclass(slots=True)
class DeepgramRealtimeConfig:
    model: str
    chunk_ms: int
    pace: str
    endpointing_ms: int
    utterance_end_ms: int
    trailing_silence_ms: int
    connect_timeout_s: float
    final_timeout_s: float
    post_audio_idle_s: float


class DeepgramRealtimeProvider(RealtimeProvider):
    name = "deepgram"

    def __init__(self, config: DeepgramRealtimeConfig):
        self.config = config

    def transcribe(self, audio_clip: AudioClip, label: str) -> ProviderResult:
        from deepgram import DeepgramClient
        from deepgram.core.events import EventType

        api_key = os.environ.get("DEEPGRAM_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is not set")

        result = ProviderResult(
            provider=self.name,
            transcript_text="",
            meta={
                "label": label,
                "model": self.config.model,
                "chunk_ms": self.config.chunk_ms,
                "pace": self.config.pace,
                "endpointing_ms": self.config.endpointing_ms,
                "utterance_end_ms": self.config.utterance_end_ms,
            },
        )
        ws_open = threading.Event()
        message_event = threading.Event()
        pending_lock = threading.Lock()
        pending_transcript = ""
        final_segments: list[str] = []
        last_nonempty_transcript = ""
        error_holder: list[str] = []
        last_message_at: float | None = None

        client = DeepgramClient(api_key=api_key)
        ctx = client.listen.v1.connect(
            model=self.config.model,
            encoding="linear16",
            sample_rate="16000",
            channels="1",
            interim_results="true",
            vad_events="true",
            utterance_end_ms=str(self.config.utterance_end_ms),
            endpointing=str(self.config.endpointing_ms),
        )
        socket = ctx.__enter__()
        result.session_started_at = time.perf_counter()

        def finalize(text: str, now: float, event_type: str) -> None:
            cleaned = text.strip()
            if not cleaned:
                return
            final_segments.append(cleaned)
            result.transcript_text = " ".join(final_segments).strip()
            result.final_at = now
            result.events.append(ProviderTraceEvent(ts_s=now, type=event_type, text=cleaned))

        def on_open(_socket) -> None:
            ws_open.set()

        def on_message(message) -> None:
            nonlocal pending_transcript, last_nonempty_transcript, last_message_at
            now = time.perf_counter()
            last_message_at = now
            message_event.set()
            msg_type = getattr(message, "type", "") if not isinstance(message, dict) else message.get("type", "")

            if msg_type == "UtteranceEnd":
                with pending_lock:
                    text = pending_transcript
                    pending_transcript = ""
                finalize(text, now, "utterance_end")
                return

            if msg_type != "Results":
                return

            transcript = ""
            is_final = False
            speech_final = False
            try:
                if isinstance(message, dict):
                    channel = message.get("channel", {})
                    alternatives = channel.get("alternatives", [])
                    if alternatives:
                        transcript = alternatives[0].get("transcript", "")
                    is_final = bool(message.get("is_final", False))
                    speech_final = bool(message.get("speech_final", False))
                else:
                    if message.channel and message.channel.alternatives:
                        transcript = message.channel.alternatives[0].transcript
                    is_final = bool(message.is_final)
                    speech_final = bool(getattr(message, "speech_final", False))
            except (AttributeError, IndexError, KeyError):
                return

            transcript = transcript.strip()
            if not transcript:
                return

            last_nonempty_transcript = transcript
            if result.first_text_at is None:
                result.first_text_at = now
            result.events.append(
                ProviderTraceEvent(
                    ts_s=now,
                    type="results",
                    text=transcript,
                    meta={"is_final": is_final, "speech_final": speech_final},
                )
            )

            if is_final:
                with pending_lock:
                    pending_transcript = transcript

            if speech_final:
                with pending_lock:
                    text = pending_transcript or transcript
                    pending_transcript = ""
                finalize(text, now, "speech_final")

        def on_error(error) -> None:
            error_holder.append(str(error))
            message_event.set()

        socket.on(EventType.OPEN, on_open)
        socket.on(EventType.MESSAGE, on_message)
        socket.on(EventType.ERROR, on_error)

        listener = threading.Thread(target=socket.start_listening, daemon=True)
        listener.start()

        if not ws_open.wait(timeout=self.config.connect_timeout_s):
            raise RuntimeError("Deepgram WebSocket did not open in time")

        try:
            audio_chunks = iter_pcm16_chunks(audio_clip.samples, chunk_ms=self.config.chunk_ms)
            for chunk in audio_chunks:
                now = time.perf_counter()
                if result.first_audio_sent_at is None:
                    result.first_audio_sent_at = now
                result.last_audio_sent_at = now
                result.events.append(
                    ProviderTraceEvent(ts_s=now, type="audio.append", meta={"bytes": len(chunk)})
                )
                socket.send_media(chunk)
                if self.config.pace == "realtime":
                    time.sleep(self.config.chunk_ms / 1000.0)

            if self.config.trailing_silence_ms > 0:
                silence = silence_pcm16(self.config.trailing_silence_ms)
                socket.send_media(silence)
                if self.config.pace == "realtime":
                    time.sleep(self.config.trailing_silence_ms / 1000.0)

            deadline = time.perf_counter() + self.config.final_timeout_s
            while time.perf_counter() < deadline:
                if error_holder:
                    break
                message_event.wait(timeout=0.1)
                message_event.clear()
                if (
                    last_message_at is not None
                    and (final_segments or pending_transcript or last_nonempty_transcript)
                    and time.perf_counter() - last_message_at >= self.config.post_audio_idle_s
                ):
                    break
        finally:
            try:
                socket.send_close_stream()
            except Exception:
                pass
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
            listener.join(timeout=2)

        if error_holder:
            result.error = error_holder[0]

        if not result.transcript_text:
            with pending_lock:
                fallback = pending_transcript.strip()
            result.transcript_text = fallback or last_nonempty_transcript
            if result.transcript_text and result.final_at is None:
                result.final_at = time.perf_counter()

        return result
