from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from importlib.util import find_spec

import numpy as np

from stt_exp.audio import TARGET_SAMPLE_RATE


VOXTRAL_EOU_MODES = ("none", "silero", "silero-smart-turn")
VOXTRAL_EOU_CONTROL_CYCLE_MODE = "voxtral.eou.cycle_mode"
VOXTRAL_EOU_CONTROL_CYCLE_PRESET = "voxtral.eou.cycle_preset"


@dataclass(frozen=True, slots=True)
class VoxtralEouPreset:
    name: str
    min_utterance_ms: int
    silero_min_silence_ms: int


VOXTRAL_EOU_PRESETS = (
    VoxtralEouPreset(name="balanced", min_utterance_ms=300, silero_min_silence_ms=500),
    VoxtralEouPreset(name="fast", min_utterance_ms=120, silero_min_silence_ms=180),
    VoxtralEouPreset(name="faster", min_utterance_ms=80, silero_min_silence_ms=140),
    VoxtralEouPreset(name="hair", min_utterance_ms=60, silero_min_silence_ms=100),
)


@dataclass(slots=True)
class VoxtralEouConfig:
    mode: str = "none"
    preset_name: str = "balanced"
    min_utterance_ms: int = 300
    silero_threshold: float = 0.5
    silero_min_silence_ms: int = 500
    smart_turn_model_path: str | None = None
    smart_turn_cpu_count: int = 1


@dataclass(slots=True)
class SpeechSegmentUpdate:
    is_speech: bool
    speech_started: bool
    speech_ended: bool
    speech_duration_ms: float


class BaseVoxtralEouDetector:
    def __init__(self, config: VoxtralEouConfig):
        self.config = config

    @property
    def enabled(self) -> bool:
        return self.config.mode != "none"

    async def on_audio_chunk(self, chunk: bytes) -> bool:
        return False

    def reset(self) -> None:
        return None


class NoopVoxtralEouDetector(BaseVoxtralEouDetector):
    pass


class _SpeechSegmentTracker:
    def __init__(self, *, min_silence_ms: int):
        self.min_silence_ms = float(min_silence_ms)
        self.current_audio_ms = 0.0
        self.speech_started_ms: float | None = None
        self.pending_silence_ms = 0.0

    def update(self, *, is_speech: bool, chunk_ms: float) -> SpeechSegmentUpdate:
        speech_started = False
        speech_ended = False
        speech_duration_ms = 0.0
        prior_audio_ms = self.current_audio_ms
        self.current_audio_ms += chunk_ms

        if is_speech:
            self.pending_silence_ms = 0.0
            if self.speech_started_ms is None:
                self.speech_started_ms = prior_audio_ms
                speech_started = True
        elif self.speech_started_ms is not None:
            self.pending_silence_ms += chunk_ms
            if self.pending_silence_ms >= self.min_silence_ms:
                speech_ended = True
                speech_end_ms = self.current_audio_ms - self.pending_silence_ms
                speech_duration_ms = max(0.0, speech_end_ms - self.speech_started_ms)
                self.speech_started_ms = None
                self.pending_silence_ms = 0.0

        return SpeechSegmentUpdate(
            is_speech=is_speech,
            speech_started=speech_started,
            speech_ended=speech_ended,
            speech_duration_ms=speech_duration_ms,
        )

    def reset(self) -> None:
        self.current_audio_ms = 0.0
        self.speech_started_ms = None
        self.pending_silence_ms = 0.0


class _SileroRuntime:
    def __init__(self, *, threshold: float):
        try:
            import torch
            from silero_vad import load_silero_vad
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised via integration
            raise RuntimeError(
                "silero-vad is not installed; run `uv sync --extra voxtral-eou` to use Voxtral EOU modes"
            ) from exc

        self._torch = torch
        self._model = load_silero_vad()
        self._threshold = threshold
        self._window_size = 512
        self._pending = np.zeros(0, dtype=np.float32)

    def is_speech(self, chunk: bytes) -> bool:
        samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size == 0:
            return False
        self._pending = np.concatenate((self._pending, samples))
        speech_seen = False
        while self._pending.size >= self._window_size:
            window = self._pending[: self._window_size]
            self._pending = self._pending[self._window_size :]
            tensor = self._torch.from_numpy(window)
            score = float(self._model(tensor, TARGET_SAMPLE_RATE).item())
            if score >= self._threshold:
                speech_seen = True
        return speech_seen


class _SmartTurnRuntime:
    def __init__(self, *, model_path: str | None, cpu_count: int):
        try:
            from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
            from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
            from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised via integration
            raise RuntimeError(
                "pipecat smart-turn is not installed; run `uv sync --extra voxtral-eou --extra voxtral-smart-turn`"
            ) from exc

        self._complete_state = EndOfTurnState.COMPLETE
        self._analyzer = LocalSmartTurnAnalyzerV3(
            sample_rate=TARGET_SAMPLE_RATE,
            params=SmartTurnParams(stop_secs=3600.0),
            smart_turn_model_path=model_path,
            cpu_count=cpu_count,
        )

    def append_audio(self, chunk: bytes, *, is_speech: bool) -> None:
        self._analyzer.append_audio(chunk, is_speech=is_speech)

    async def analyze_end_of_turn(self) -> bool:
        state, _metrics = await self._analyzer.analyze_end_of_turn()
        return state == self._complete_state

    def reset(self) -> None:
        self._analyzer.clear()


class VoxtralEouDetectorFactory:
    def __init__(self):
        self._silero_runtimes: dict[float, _SileroRuntime] = {}
        self._smart_turn_runtimes: dict[tuple[str | None, int], _SmartTurnRuntime] = {}

    def build(self, config: VoxtralEouConfig) -> BaseVoxtralEouDetector:
        silero_runtime = None
        smart_turn_runtime = None

        if config.mode in {"silero", "silero-smart-turn"}:
            silero_runtime = self._silero_runtimes.get(config.silero_threshold)
            if silero_runtime is None:
                silero_runtime = _SileroRuntime(threshold=config.silero_threshold)
                self._silero_runtimes[config.silero_threshold] = silero_runtime

        if config.mode == "silero-smart-turn":
            key = (config.smart_turn_model_path, config.smart_turn_cpu_count)
            smart_turn_runtime = self._smart_turn_runtimes.get(key)
            if smart_turn_runtime is None:
                smart_turn_runtime = _SmartTurnRuntime(
                    model_path=config.smart_turn_model_path,
                    cpu_count=config.smart_turn_cpu_count,
                )
                self._smart_turn_runtimes[key] = smart_turn_runtime

        return build_voxtral_eou_detector(
            config,
            silero_runtime=silero_runtime,
            smart_turn_runtime=smart_turn_runtime,
        )


class SileroVoxtralEouDetector(BaseVoxtralEouDetector):
    def __init__(self, config: VoxtralEouConfig, *, silero_runtime: _SileroRuntime | None = None):
        super().__init__(config)
        self._silero = silero_runtime or _SileroRuntime(threshold=config.silero_threshold)
        self._tracker = _SpeechSegmentTracker(min_silence_ms=config.silero_min_silence_ms)

    async def on_audio_chunk(self, chunk: bytes) -> bool:
        is_speech = self._silero.is_speech(chunk)
        update = self._tracker.update(is_speech=is_speech, chunk_ms=_chunk_duration_ms(chunk))
        if not update.speech_ended:
            return False
        should_finalize = update.speech_duration_ms >= self.config.min_utterance_ms
        if should_finalize:
            self.reset()
        return should_finalize

    def reset(self) -> None:
        self._tracker.reset()


class SileroSmartTurnVoxtralEouDetector(SileroVoxtralEouDetector):
    def __init__(
        self,
        config: VoxtralEouConfig,
        *,
        silero_runtime: _SileroRuntime | None = None,
        smart_turn_runtime: _SmartTurnRuntime | None = None,
    ):
        super().__init__(config, silero_runtime=silero_runtime)
        self._smart_turn = smart_turn_runtime or _SmartTurnRuntime(
            model_path=config.smart_turn_model_path,
            cpu_count=config.smart_turn_cpu_count,
        )

    async def on_audio_chunk(self, chunk: bytes) -> bool:
        is_speech = self._silero.is_speech(chunk)
        self._smart_turn.append_audio(chunk, is_speech=is_speech)
        update = self._tracker.update(is_speech=is_speech, chunk_ms=_chunk_duration_ms(chunk))
        if not update.speech_ended:
            return False
        if update.speech_duration_ms < self.config.min_utterance_ms:
            return False
        should_finalize = await self._smart_turn.analyze_end_of_turn()
        if should_finalize:
            self.reset()
        return should_finalize

    def reset(self) -> None:
        super().reset()
        self._smart_turn.reset()


def build_voxtral_eou_detector(
    config: VoxtralEouConfig,
    *,
    silero_runtime: _SileroRuntime | None = None,
    smart_turn_runtime: _SmartTurnRuntime | None = None,
) -> BaseVoxtralEouDetector:
    if config.mode == "none":
        return NoopVoxtralEouDetector(config)
    if config.mode == "silero":
        return SileroVoxtralEouDetector(config, silero_runtime=silero_runtime)
    if config.mode == "silero-smart-turn":
        return SileroSmartTurnVoxtralEouDetector(
            config,
            silero_runtime=silero_runtime,
            smart_turn_runtime=smart_turn_runtime,
        )
    raise ValueError(f"Unsupported Voxtral EOU mode: {config.mode}")


def _chunk_duration_ms(chunk: bytes) -> float:
    sample_count = len(chunk) / 2
    return sample_count * 1000.0 / TARGET_SAMPLE_RATE


def cycle_voxtral_eou_mode(mode: str) -> str:
    return cycle_voxtral_eou_mode_with_available(mode, VOXTRAL_EOU_MODES)


def cycle_voxtral_eou_mode_with_available(mode: str, available_modes: tuple[str, ...]) -> str:
    if not available_modes:
        return "none"
    try:
        index = available_modes.index(mode)
    except ValueError:
        return available_modes[0]
    return available_modes[(index + 1) % len(available_modes)]


def cycle_voxtral_eou_preset_name(name: str) -> str:
    names = [preset.name for preset in VOXTRAL_EOU_PRESETS]
    try:
        index = names.index(name)
    except ValueError:
        return names[0]
    return names[(index + 1) % len(names)]


def apply_voxtral_eou_preset(config: VoxtralEouConfig, preset_name: str) -> VoxtralEouConfig:
    preset = next((item for item in VOXTRAL_EOU_PRESETS if item.name == preset_name), None)
    if preset is None:
        raise ValueError(f"Unknown Voxtral EOU preset: {preset_name}")
    return replace(
        config,
        preset_name=preset.name,
        min_utterance_ms=preset.min_utterance_ms,
        silero_min_silence_ms=preset.silero_min_silence_ms,
    )


def summarize_voxtral_eou_config(config: VoxtralEouConfig) -> str:
    return (
        f"{config.mode}/{config.preset_name} "
        f"utt={config.min_utterance_ms}ms silence={config.silero_min_silence_ms}ms "
        f"thr={config.silero_threshold:.2f}"
    )


def get_available_voxtral_eou_modes() -> tuple[str, ...]:
    modes = ["none"]
    has_silero = find_spec("silero_vad") is not None
    has_pipecat = find_spec("pipecat") is not None
    if has_silero:
        modes.append("silero")
    if has_silero and has_pipecat:
        modes.append("silero-smart-turn")
    return tuple(modes)
