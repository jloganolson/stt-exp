from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ProviderTraceEvent:
    ts_s: float
    type: str
    text: str = ""
    meta: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ProviderResult:
    provider: str
    transcript_text: str
    session_started_at: float | None = None
    first_audio_sent_at: float | None = None
    last_audio_sent_at: float | None = None
    first_text_at: float | None = None
    final_at: float | None = None
    error: str | None = None
    meta: dict[str, object] = field(default_factory=dict)
    events: list[ProviderTraceEvent] = field(default_factory=list)


class RealtimeProvider:
    name: str

    def warmup(self) -> None:
        return None

    def transcribe(self, audio_clip, label: str) -> ProviderResult:  # pragma: no cover - interface
        raise NotImplementedError
