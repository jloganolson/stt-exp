import pytest

from stt_exp.metrics import compute_timing, normalize_transcript, score_transcript


def test_normalize_transcript() -> None:
    assert normalize_transcript(" Hey,   Man! ") == "hey man"


def test_score_transcript() -> None:
    metrics = score_transcript("Hey man how are you", "hey man how are you")
    assert metrics.wer == 0.0
    assert metrics.cer == 0.0


def test_compute_timing() -> None:
    timing = compute_timing(
        audio_duration_s=1.0,
        first_audio_sent_at=10.0,
        last_audio_sent_at=10.8,
        first_text_at=10.3,
        final_at=11.0,
    )
    assert timing.ttft_ms == pytest.approx(300.0)
    assert timing.final_latency_ms == pytest.approx(1000.0)
    assert timing.tail_latency_ms == pytest.approx(200.0)
    assert timing.realtime_factor == pytest.approx(1.0)
