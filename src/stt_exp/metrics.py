from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from jiwer import cer, wer


@dataclass(slots=True)
class QualityMetrics:
    normalized_reference: str | None
    normalized_hypothesis: str
    wer: float | None
    cer: float | None


@dataclass(slots=True)
class TimingMetrics:
    ttft_ms: float | None
    final_latency_ms: float | None
    tail_latency_ms: float | None
    realtime_factor: float | None


def normalize_transcript(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def score_transcript(reference_text: str | None, hypothesis_text: str) -> QualityMetrics:
    normalized_hypothesis = normalize_transcript(hypothesis_text)
    if not reference_text:
        return QualityMetrics(
            normalized_reference=None,
            normalized_hypothesis=normalized_hypothesis,
            wer=None,
            cer=None,
        )
    normalized_reference = normalize_transcript(reference_text)
    return QualityMetrics(
        normalized_reference=normalized_reference,
        normalized_hypothesis=normalized_hypothesis,
        wer=wer(normalized_reference, normalized_hypothesis),
        cer=cer(normalized_reference, normalized_hypothesis),
    )


def compute_timing(
    *,
    audio_duration_s: float,
    first_audio_sent_at: float | None,
    last_audio_sent_at: float | None,
    first_text_at: float | None,
    final_at: float | None,
) -> TimingMetrics:
    ttft_ms = None
    final_latency_ms = None
    tail_latency_ms = None
    realtime_factor = None

    if first_audio_sent_at is not None and first_text_at is not None:
        ttft_ms = (first_text_at - first_audio_sent_at) * 1000.0
    if first_audio_sent_at is not None and final_at is not None:
        final_latency_ms = (final_at - first_audio_sent_at) * 1000.0
    if last_audio_sent_at is not None and final_at is not None:
        tail_latency_ms = (final_at - last_audio_sent_at) * 1000.0
    if first_audio_sent_at is not None and final_at is not None and audio_duration_s > 0:
        realtime_factor = (final_at - first_audio_sent_at) / audio_duration_s

    return TimingMetrics(
        ttft_ms=ttft_ms,
        final_latency_ms=final_latency_ms,
        tail_latency_ms=tail_latency_ms,
        realtime_factor=realtime_factor,
    )
