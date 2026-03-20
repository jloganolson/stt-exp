import asyncio

import numpy as np

from stt_exp.voxtral_eou import (
    VoxtralEouConfig,
    apply_voxtral_eou_preset,
    build_voxtral_eou_detector,
    cycle_voxtral_eou_mode,
    cycle_voxtral_eou_preset_name,
)


def _make_chunk(duration_ms: int = 40) -> bytes:
    sample_count = int(16_000 * duration_ms / 1000)
    return np.zeros(sample_count, dtype=np.int16).tobytes()


class FakeSileroRuntime:
    def __init__(self, decisions: list[bool]):
        self._decisions = iter(decisions)

    def is_speech(self, _chunk: bytes) -> bool:
        return next(self._decisions)


class FakeSmartTurnRuntime:
    def __init__(self, decisions: list[bool]):
        self.decisions = iter(decisions)
        self.appended: list[bool] = []
        self.reset_calls = 0

    def append_audio(self, _chunk: bytes, *, is_speech: bool) -> None:
        self.appended.append(is_speech)

    async def analyze_end_of_turn(self) -> bool:
        return next(self.decisions)

    def reset(self) -> None:
        self.reset_calls += 1


def test_build_voxtral_eou_detector_none_mode_is_noop() -> None:
    detector = build_voxtral_eou_detector(VoxtralEouConfig(mode="none"))

    assert asyncio.run(detector.on_audio_chunk(_make_chunk())) is False


def test_silero_detector_finalizes_after_silence() -> None:
    detector = build_voxtral_eou_detector(
        VoxtralEouConfig(
            mode="silero",
            min_utterance_ms=80,
            silero_min_silence_ms=80,
        ),
        silero_runtime=FakeSileroRuntime([True, True, True, False, False]),
    )

    results = [asyncio.run(detector.on_audio_chunk(_make_chunk())) for _ in range(5)]

    assert results == [False, False, False, False, True]


def test_silero_smart_turn_requires_semantic_completion() -> None:
    smart_turn = FakeSmartTurnRuntime([False, True])
    detector = build_voxtral_eou_detector(
        VoxtralEouConfig(
            mode="silero-smart-turn",
            min_utterance_ms=80,
            silero_min_silence_ms=80,
        ),
        silero_runtime=FakeSileroRuntime(
            [
                True,
                True,
                False,
                False,
                True,
                True,
                False,
                False,
            ]
        ),
        smart_turn_runtime=smart_turn,
    )

    results = [asyncio.run(detector.on_audio_chunk(_make_chunk())) for _ in range(8)]

    assert results == [False, False, False, False, False, False, False, True]
    assert smart_turn.appended == [True, True, False, False, True, True, False, False]
    assert smart_turn.reset_calls == 1


def test_cycle_voxtral_eou_mode_wraps() -> None:
    assert cycle_voxtral_eou_mode("none") == "silero"
    assert cycle_voxtral_eou_mode("silero") == "silero-smart-turn"
    assert cycle_voxtral_eou_mode("silero-smart-turn") == "none"


def test_apply_and_cycle_voxtral_eou_preset() -> None:
    config = VoxtralEouConfig(mode="silero")
    config = apply_voxtral_eou_preset(config, cycle_voxtral_eou_preset_name("balanced"))

    assert config.preset_name == "fast"
    assert config.min_utterance_ms == 120
    assert config.silero_min_silence_ms == 180
