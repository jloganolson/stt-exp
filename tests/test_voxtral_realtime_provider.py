from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
import websockets

from stt_exp.audio import AudioClip
from stt_exp.providers.voxtral_realtime import VoxtralRealtimeConfig, VoxtralRealtimeProvider
from stt_exp.voxtral_eou import VoxtralEouConfig


class FakeDetector:
    def __init__(self, decisions: list[bool]):
        self.decisions = iter(decisions)
        self.config = VoxtralEouConfig(mode="silero", preset_name="test", min_utterance_ms=120, silero_min_silence_ms=180)
        self.enabled = True

    async def on_audio_chunk(self, _chunk: bytes) -> bool:
        return next(self.decisions)

    def reset(self) -> None:
        return None


@pytest.mark.anyio
async def test_voxtral_realtime_provider_with_eou_receives_done(tmp_path: Path) -> None:
    commits: list[dict[str, object]] = []

    async def handler(ws):
        await ws.send(json.dumps({"type": "session.created"}))
        async for raw in ws:
            message = json.loads(raw)
            if message.get("type") == "input_audio_buffer.commit":
                commits.append(message)
                if message.get("final"):
                    await ws.send(json.dumps({"type": "transcription.done", "text": "hello world", "usage": {}}))

    server = await websockets.serve(handler, "127.0.0.1", 8890)
    try:
        provider = VoxtralRealtimeProvider(
            VoxtralRealtimeConfig(
                uri="ws://127.0.0.1:8890/v1/realtime",
                model="test-model",
                chunk_ms=40,
                pace="burst",
                receive_timeout_s=2.0,
                open_timeout_s=2.0,
                warmup_silence_ms=0,
                eou=VoxtralEouConfig(mode="silero", preset_name="test", min_utterance_ms=120, silero_min_silence_ms=180),
            )
        )
        provider._detector_factory.build = lambda _config: FakeDetector([False, True, False])  # type: ignore[method-assign]
        clip = AudioClip(path=tmp_path / "fake.wav", samples=np.zeros(1_920, dtype=np.float32))

        result = await asyncio.to_thread(provider.transcribe, clip, "fake")

        assert result.error is None
        assert result.transcript_text == "hello world"
        assert any(event.type == "eou.finalize" for event in result.events)
        assert sum(1 for item in commits if item.get("final")) == 2
    finally:
        server.close()
        await server.wait_closed()
