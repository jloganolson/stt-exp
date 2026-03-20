from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass

import websockets

from stt_exp.audio import AudioClip, iter_pcm16_chunks, silence_pcm16
from stt_exp.providers.base import ProviderResult, ProviderTraceEvent, RealtimeProvider


@dataclass(slots=True)
class VoxtralRealtimeConfig:
    uri: str
    model: str
    chunk_ms: int
    pace: str
    receive_timeout_s: float
    open_timeout_s: float
    warmup_silence_ms: int


class VoxtralRealtimeProvider(RealtimeProvider):
    name = "voxtral"

    def __init__(self, config: VoxtralRealtimeConfig):
        self.config = config

    def warmup(self) -> None:
        if self.config.warmup_silence_ms <= 0:
            return
        silence = silence_pcm16(self.config.warmup_silence_ms)
        asyncio.run(self._run_session([silence], label="voxtral-warmup"))

    def transcribe(self, audio_clip: AudioClip, label: str) -> ProviderResult:
        chunks = iter_pcm16_chunks(audio_clip.samples, chunk_ms=self.config.chunk_ms)
        return asyncio.run(self._run_session(chunks, label=label))

    async def _run_session(self, chunks: list[bytes], label: str) -> ProviderResult:
        result = ProviderResult(
            provider=self.name,
            transcript_text="",
            meta={
                "label": label,
                "uri": self.config.uri,
                "model": self.config.model,
                "chunk_ms": self.config.chunk_ms,
                "pace": self.config.pace,
            },
        )
        transcript_parts: list[str] = []

        try:
            async with websockets.connect(
                self.config.uri,
                open_timeout=self.config.open_timeout_s,
                max_size=None,
            ) as ws:
                result.session_started_at = time.perf_counter()
                hello = json.loads(await asyncio.wait_for(ws.recv(), timeout=self.config.receive_timeout_s))
                if hello.get("type") != "session.created":
                    raise RuntimeError(f"Unexpected handshake: {hello}")

                await ws.send(json.dumps({"type": "session.update", "model": self.config.model}))
                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

                for chunk in chunks:
                    now = time.perf_counter()
                    if result.first_audio_sent_at is None:
                        result.first_audio_sent_at = now
                    result.last_audio_sent_at = now
                    result.events.append(
                        ProviderTraceEvent(ts_s=now, type="audio.append", meta={"bytes": len(chunk)})
                    )
                    await ws.send(
                        json.dumps(
                            {
                                "type": "input_audio_buffer.append",
                                "audio": base64.b64encode(chunk).decode("utf-8"),
                            }
                        )
                    )
                    if self.config.pace == "realtime":
                        await asyncio.sleep(self.config.chunk_ms / 1000.0)

                await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))

                while True:
                    raw_message = await asyncio.wait_for(ws.recv(), timeout=self.config.receive_timeout_s)
                    message = json.loads(raw_message)
                    now = time.perf_counter()

                    if message.get("type") == "transcription.delta":
                        delta = message.get("delta", "")
                        if delta:
                            if result.first_text_at is None:
                                result.first_text_at = now
                            transcript_parts.append(delta)
                            result.events.append(
                                ProviderTraceEvent(ts_s=now, type="transcription.delta", text=delta)
                            )
                    elif message.get("type") == "transcription.done":
                        result.final_at = now
                        result.transcript_text = message.get("text", "").strip() or "".join(transcript_parts).strip()
                        result.meta["usage"] = message.get("usage")
                        result.events.append(
                            ProviderTraceEvent(ts_s=now, type="transcription.done", text=result.transcript_text)
                        )
                        break
                    elif message.get("type") == "error":
                        raise RuntimeError(message.get("error", "Unknown Voxtral error"))
        except Exception as exc:
            result.error = str(exc)
            if not result.transcript_text:
                result.transcript_text = "".join(transcript_parts).strip()

        return result
