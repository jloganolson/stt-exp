from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from stt_exp.providers.base import ProviderResult, ProviderTraceEvent, RealtimeProvider


@dataclass(slots=True)
class ParakeetExternalConfig:
    python_executable: str
    worker_script: str
    model_id: str
    pace: str
    silence_chunks: int
    device: str = "cuda"
    att_context_size: tuple[int, int] | None = None


class ParakeetExternalProvider(RealtimeProvider):
    name = "parakeet"

    def __init__(self, config: ParakeetExternalConfig):
        self.config = config

    def transcribe(self, audio_clip, label: str) -> ProviderResult:
        cmd = [
            self.config.python_executable,
            self.config.worker_script,
            "--audio",
            str(Path(audio_clip.path).resolve()),
            "--model-id",
            self.config.model_id,
            "--device",
            self.config.device,
            "--pace",
            self.config.pace,
            "--silence-chunks",
            str(self.config.silence_chunks),
        ]
        if self.config.att_context_size is not None:
            cmd.extend(
                [
                    "--att-context-size",
                    str(self.config.att_context_size[0]),
                    str(self.config.att_context_size[1]),
                ]
            )
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        marker = "PARAKEET_RESULT_JSON="
        payload = None
        for line in combined.splitlines():
            if line.startswith(marker):
                payload = json.loads(line[len(marker) :])
                break
        if proc.returncode != 0:
            raise RuntimeError(f"Parakeet worker failed ({proc.returncode}): {combined.strip()}")
        if payload is None:
            raise RuntimeError(f"Parakeet worker did not emit result payload: {combined.strip()}")

        result = ProviderResult(
            provider=self.name,
            transcript_text=payload["transcript_text"],
            session_started_at=payload.get("session_started_at_s"),
            first_audio_sent_at=payload.get("first_audio_sent_at_s"),
            last_audio_sent_at=payload.get("last_audio_sent_at_s"),
            first_text_at=payload.get("first_text_at_s"),
            final_at=payload.get("final_at_s"),
            meta={"label": label, **payload.get("meta", {})},
            events=[
                ProviderTraceEvent(
                    ts_s=event["wall_time_s"],
                    type="transcription.delta",
                    text=event["text"],
                    meta={"audio_pos_s": event["audio_pos_s"]},
                )
                for event in payload.get("events", [])
            ],
        )
        if result.transcript_text:
            result.events.append(
                ProviderTraceEvent(ts_s=result.final_at or 0.0, type="transcription.done", text=result.transcript_text)
            )
        return result
