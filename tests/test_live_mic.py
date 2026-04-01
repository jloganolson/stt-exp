from __future__ import annotations

import queue
import sys
import threading
import time
import types
from argparse import Namespace
from typing import Any

from stt_exp.cli import _default_parakeet_python, run_live_command
from stt_exp.live_mic import (
    PARAKEET_CONTROL_CYCLE_PRESET,
    LiveConfig,
    _append_voxtral_delta,
    _format_parakeet_confidence,
    _format_parakeet_text_event,
    _format_voxtral_connection_error,
    _run_deepgram_live,
    _run_parakeet_live,
    apply_parakeet_live_preset,
    cycle_parakeet_live_preset_name,
)


def test_format_voxtral_connection_error_local_uri() -> None:
    message = _format_voxtral_connection_error(
        "ws://127.0.0.1:8000/v1/realtime",
        ConnectionRefusedError(111, "Connect call failed"),
    )

    assert "127.0.0.1:8000" in message
    assert "./scripts/serve_voxtral_optimized.sh" in message


def test_format_voxtral_connection_error_remote_uri() -> None:
    message = _format_voxtral_connection_error(
        "wss://voxtral.example.com/v1/realtime",
        OSError("timed out"),
    )

    assert "voxtral.example.com:443" in message
    assert "--voxtral-uri" in message


def test_append_voxtral_delta_preserves_leading_space() -> None:
    current_text = _append_voxtral_delta("what's", " up")

    assert current_text == "what's up"


def test_format_parakeet_confidence_and_text_event() -> None:
    confidence = {"label": "medium", "avg": 0.74, "min": 0.42}

    assert _format_parakeet_confidence(confidence) == " [conf medium avg=0.74 min=0.42]"
    assert _format_parakeet_text_event({"text": "hello there", "confidence": confidence}) == (
        "hello there [conf medium avg=0.74 min=0.42]"
    )


def test_default_parakeet_python_falls_back_to_sibling_repo(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path / "stt-exp"
    sibling_python = tmp_path / "parakeet-exp" / ".venv" / "bin" / "python"
    sibling_python.parent.mkdir(parents=True)
    sibling_python.write_text("", encoding="utf-8")

    monkeypatch.delenv("PARAKEET_PYTHON", raising=False)
    monkeypatch.setattr("stt_exp.cli.REPO_ROOT", repo_root)

    assert _default_parakeet_python() == str(sibling_python)


def test_cycle_and_apply_parakeet_live_preset() -> None:
    config = LiveConfig(
        providers=["parakeet"],
        chunk_ms=40,
        device=None,
        deepgram_model="nova-3",
        deepgram_endpointing_ms=300,
        deepgram_utterance_end_ms=1000,
        sherpa_model_dir="",
        sherpa_provider="cuda",
        sherpa_num_threads=2,
        parakeet_python="/tmp/python",
        parakeet_worker_script="/tmp/worker.py",
        parakeet_model_id="fake-model",
        parakeet_device="cuda",
        parakeet_live_mode="tuned",
        parakeet_preset="balanced",
        parakeet_eou_silence_ms=240,
        parakeet_min_utterance_ms=60,
        parakeet_force_finalize_ms=400,
        parakeet_preroll_ms=160,
        parakeet_rms_threshold=0.008,
        voxtral_uri="ws://127.0.0.1:8000/v1/realtime",
        voxtral_model="voxtral",
        voxtral_eou_mode="none",
        voxtral_eou_min_utterance_ms=300,
        voxtral_eou_silero_threshold=0.5,
        voxtral_eou_silero_min_silence_ms=500,
        voxtral_eou_smart_turn_model_path=None,
        voxtral_eou_smart_turn_cpu_count=1,
        parakeet_att_context_size=(70, 1),
    )

    assert cycle_parakeet_live_preset_name("balanced") == "fast"

    updated = apply_parakeet_live_preset(config, "hair")

    assert updated.parakeet_preset == "hair"
    assert updated.parakeet_eou_silence_ms == 100
    assert updated.parakeet_min_utterance_ms == 30
    assert updated.parakeet_force_finalize_ms == 160
    assert cycle_parakeet_live_preset_name("hair") == "accurate"
    assert cycle_parakeet_live_preset_name("accurate") == "very-accurate"

    accurate = apply_parakeet_live_preset(config, "accurate")
    assert accurate.parakeet_preset == "accurate"
    assert accurate.parakeet_eou_silence_ms == 320
    assert accurate.parakeet_min_utterance_ms == 70
    assert accurate.parakeet_force_finalize_ms == 550
    assert accurate.parakeet_preroll_ms == 220
    assert accurate.parakeet_rms_threshold == 0.007

    very_accurate = apply_parakeet_live_preset(config, "very-accurate")
    assert very_accurate.parakeet_preset == "very-accurate"
    assert very_accurate.parakeet_eou_silence_ms == 420
    assert very_accurate.parakeet_min_utterance_ms == 80
    assert very_accurate.parakeet_force_finalize_ms == 750
    assert very_accurate.parakeet_preroll_ms == 280
    assert very_accurate.parakeet_rms_threshold == 0.006
    assert cycle_parakeet_live_preset_name("very-accurate") == "balanced"


def test_run_live_command_defaults_to_accurate_preset(monkeypatch) -> None:
    captured = {}

    def fake_run_live(config: LiveConfig) -> None:
        captured["config"] = config

    monkeypatch.setattr("stt_exp.cli.run_live", fake_run_live)

    run_live_command(
        Namespace(
            providers=["parakeet"],
            chunk_ms=40,
            device=None,
            deepgram_model="nova-3",
            deepgram_endpointing_ms=300,
            deepgram_utterance_end_ms=1000,
            sherpa_model_dir="",
            sherpa_provider="cuda",
            sherpa_num_threads=2,
            parakeet_python="/tmp/python",
            parakeet_worker_script="/tmp/worker.py",
            parakeet_model_id="fake-model",
            parakeet_device="cuda",
            parakeet_live_mode="tuned",
            parakeet_preset="accurate",
            parakeet_eou_silence_ms=None,
            parakeet_min_utterance_ms=None,
            parakeet_force_finalize_ms=None,
            parakeet_preroll_ms=None,
            parakeet_rms_threshold=None,
            voxtral_uri="ws://127.0.0.1:8000/v1/realtime",
            voxtral_model="voxtral",
            voxtral_eou_mode="none",
            voxtral_eou_min_utterance_ms=300,
            voxtral_eou_silero_threshold=0.5,
            voxtral_eou_silero_min_silence_ms=500,
            voxtral_eou_smart_turn_model_path=None,
            voxtral_eou_smart_turn_cpu_count=1,
        )
    )

    config = captured["config"]
    assert config.parakeet_preset == "accurate"
    assert config.parakeet_eou_silence_ms == 320
    assert config.parakeet_min_utterance_ms == 70
    assert config.parakeet_force_finalize_ms == 550
    assert config.parakeet_preroll_ms == 220
    assert config.parakeet_rms_threshold == 0.007


def test_run_live_command_applies_parakeet_preset(monkeypatch) -> None:
    captured = {}

    def fake_run_live(config: LiveConfig) -> None:
        captured["config"] = config

    monkeypatch.setattr("stt_exp.cli.run_live", fake_run_live)

    run_live_command(
        Namespace(
            providers=["parakeet"],
            chunk_ms=40,
            device=None,
            deepgram_model="nova-3",
            deepgram_endpointing_ms=300,
            deepgram_utterance_end_ms=1000,
            sherpa_model_dir="",
            sherpa_provider="cuda",
            sherpa_num_threads=2,
            parakeet_python="/tmp/python",
            parakeet_worker_script="/tmp/worker.py",
            parakeet_model_id="fake-model",
            parakeet_device="cuda",
            parakeet_live_mode="tuned",
            parakeet_preset="fast",
            parakeet_eou_silence_ms=None,
            parakeet_min_utterance_ms=None,
            parakeet_force_finalize_ms=None,
            parakeet_preroll_ms=None,
            parakeet_rms_threshold=None,
            voxtral_uri="ws://127.0.0.1:8000/v1/realtime",
            voxtral_model="voxtral",
            voxtral_eou_mode="none",
            voxtral_eou_min_utterance_ms=300,
            voxtral_eou_silero_threshold=0.5,
            voxtral_eou_silero_min_silence_ms=500,
            voxtral_eou_smart_turn_model_path=None,
            voxtral_eou_smart_turn_cpu_count=1,
        )
    )

    config = captured["config"]
    assert config.parakeet_preset == "fast"
    assert config.parakeet_live_mode == "tuned"
    assert config.parakeet_eou_silence_ms == 180
    assert config.parakeet_min_utterance_ms == 50
    assert config.parakeet_force_finalize_ms == 300


def test_deepgram_and_parakeet_live_can_run_together(tmp_path, monkeypatch) -> None:
    worker_script = tmp_path / "fake_parakeet_worker.py"
    worker_script.write_text(
        "\n".join(
            [
                "import json",
                "import sys",
                'MARKER = "PARAKEET_LIVE_EVENT_JSON="',
                'print(MARKER + json.dumps({"type": "status", "message": "connected"}), flush=True)',
                "seen_audio = False",
                "for raw_line in sys.stdin:",
                "    message = json.loads(raw_line)",
                '    if message.get("type") == "audio" and not seen_audio:',
                "        seen_audio = True",
                '        print(MARKER + json.dumps({"type": "partial", "text": "parakeet partial"}), flush=True)',
                '        print(MARKER + json.dumps({"type": "final", "text": "parakeet final"}), flush=True)',
                '    elif message.get("type") == "stop":',
                "        break",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeEventType:
        OPEN = "open"
        MESSAGE = "message"
        ERROR = "error"

    class FakeSocket:
        def __init__(self) -> None:
            self.callbacks: dict[str, object] = {}
            self.closed = threading.Event()

        def on(self, event_type, callback) -> None:
            self.callbacks[event_type] = callback

        def start_listening(self) -> None:
            self.callbacks[FakeEventType.OPEN](self)
            self.closed.wait(timeout=5)

        def send_media(self, _chunk: bytes) -> None:
            self.callbacks[FakeEventType.MESSAGE](
                {"type": "Results", "channel": {"alternatives": [{"transcript": "deepgram partial"}]}}
            )
            self.callbacks[FakeEventType.MESSAGE](
                {
                    "type": "Results",
                    "channel": {"alternatives": [{"transcript": "deepgram final"}]},
                    "is_final": True,
                }
            )

        def send_finalize(self) -> None:
            return None

        def send_close_stream(self) -> None:
            self.closed.set()

    class FakeContext:
        def __init__(self, socket: FakeSocket) -> None:
            self.socket = socket

        def __enter__(self) -> FakeSocket:
            return self.socket

        def __exit__(self, exc_type, exc, tb) -> None:
            self.socket.send_close_stream()
            return None

    class FakeClient:
        def __init__(self, *, api_key: str) -> None:
            assert api_key == "test-key"
            self.listen = types.SimpleNamespace(v1=self)

        def connect(self, **_kwargs) -> FakeContext:
            return FakeContext(FakeSocket())

    fake_deepgram = types.ModuleType("deepgram")
    fake_deepgram.DeepgramClient = FakeClient
    fake_events = types.ModuleType("deepgram.core.events")
    fake_events.EventType = FakeEventType
    monkeypatch.setitem(sys.modules, "deepgram", fake_deepgram)
    monkeypatch.setitem(sys.modules, "deepgram.core.events", fake_events)
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")

    emitted: list[tuple[str, str, str]] = []
    emit_lock = threading.Lock()

    def emit(provider: str, text: str, kind: str = "status") -> None:
        with emit_lock:
            emitted.append((provider, kind, text))

    config = LiveConfig(
        providers=["deepgram", "parakeet"],
        chunk_ms=40,
        device=None,
        deepgram_model="nova-3",
        deepgram_endpointing_ms=300,
        deepgram_utterance_end_ms=1000,
        sherpa_model_dir="",
        sherpa_provider="cuda",
        sherpa_num_threads=2,
        parakeet_python=sys.executable,
        parakeet_worker_script=str(worker_script),
        parakeet_model_id="fake-model",
        parakeet_device="cpu",
        parakeet_live_mode="tuned",
        parakeet_preset="balanced",
        parakeet_eou_silence_ms=240,
        parakeet_min_utterance_ms=60,
        parakeet_force_finalize_ms=400,
        parakeet_preroll_ms=160,
        parakeet_rms_threshold=0.008,
        voxtral_uri="ws://127.0.0.1:8000/v1/realtime",
        voxtral_model="voxtral",
        voxtral_eou_mode="none",
        voxtral_eou_min_utterance_ms=300,
        voxtral_eou_silero_threshold=0.5,
        voxtral_eou_silero_min_silence_ms=500,
        voxtral_eou_smart_turn_model_path=None,
        voxtral_eou_smart_turn_cpu_count=1,
        parakeet_att_context_size=(70, 1),
    )

    stop_event = threading.Event()
    deepgram_audio_queue: queue.Queue[bytes] = queue.Queue()
    deepgram_control_queue: queue.Queue[str] = queue.Queue()
    parakeet_audio_queue: queue.Queue[bytes] = queue.Queue()
    parakeet_control_queue: queue.Queue[str] = queue.Queue()

    deepgram_thread = threading.Thread(
        target=_run_deepgram_live,
        args=(config, deepgram_audio_queue, deepgram_control_queue, stop_event, emit),
        daemon=True,
    )
    parakeet_thread = threading.Thread(
        target=_run_parakeet_live,
        args=(config, parakeet_audio_queue, parakeet_control_queue, stop_event, emit, lambda _text: None),
        daemon=True,
    )

    deepgram_thread.start()
    parakeet_thread.start()
    deepgram_audio_queue.put(b"\x00\x00" * 160)
    parakeet_audio_queue.put(b"\x00\x00" * 160)

    deadline = time.time() + 5
    while time.time() < deadline:
        with emit_lock:
            saw_deepgram = ("deepgram", "final", "deepgram final") in emitted
            saw_parakeet = ("parakeet", "final", "parakeet final") in emitted
        if saw_deepgram and saw_parakeet:
            break
        time.sleep(0.05)

    stop_event.set()
    deepgram_thread.join(timeout=2)
    parakeet_thread.join(timeout=2)

    assert ("deepgram", "final", "deepgram final") in emitted
    assert ("parakeet", "final", "parakeet final") in emitted
    assert not any(provider == "parakeet" and "error:" in text for provider, _kind, text in emitted)
    assert not any(provider == "deepgram" and "error:" in text for provider, _kind, text in emitted)


def test_run_parakeet_live_passes_tuning_flags(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    stop_event = threading.Event()

    class FakeStdin:
        def write(self, text: str) -> None:
            captured.setdefault("writes", []).append(text.strip())
            if '"type": "set_config"' in text:
                stop_event.set()
            return None

        def flush(self) -> None:
            return None

        def close(self) -> None:
            return None

    class FakeProc:
        def __init__(self) -> None:
            self.stdin = FakeStdin()
            self.stdout: list[str] = []
            self.returncode = 0

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            return None

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return FakeProc()

    monkeypatch.setattr("stt_exp.live_mic.subprocess.Popen", fake_popen)

    config = LiveConfig(
        providers=["parakeet"],
        chunk_ms=40,
        device=None,
        deepgram_model="nova-3",
        deepgram_endpointing_ms=300,
        deepgram_utterance_end_ms=1000,
        sherpa_model_dir="",
        sherpa_provider="cuda",
        sherpa_num_threads=2,
        parakeet_python="/tmp/parakeet-python",
        parakeet_worker_script="/tmp/parakeet-worker.py",
        parakeet_model_id="fake-model",
        parakeet_device="cuda",
        parakeet_live_mode="tuned",
        parakeet_preset="balanced",
        parakeet_eou_silence_ms=180,
        parakeet_min_utterance_ms=50,
        parakeet_force_finalize_ms=300,
        parakeet_preroll_ms=120,
        parakeet_rms_threshold=0.02,
        voxtral_uri="ws://127.0.0.1:8000/v1/realtime",
        voxtral_model="voxtral",
        voxtral_eou_mode="none",
        voxtral_eou_min_utterance_ms=300,
        voxtral_eou_silero_threshold=0.5,
        voxtral_eou_silero_min_silence_ms=500,
        voxtral_eou_smart_turn_model_path=None,
        voxtral_eou_smart_turn_cpu_count=1,
        parakeet_att_context_size=(70, 1),
    )
    control_queue: queue.Queue[str] = queue.Queue()
    control_queue.put(PARAKEET_CONTROL_CYCLE_PRESET)

    _run_parakeet_live(
        config,
        queue.Queue(),
        control_queue,
        stop_event,
        lambda *_args, **_kwargs: None,
        lambda _text: None,
    )

    assert captured["cmd"] == [
        "/tmp/parakeet-python",
        "-u",
        "/tmp/parakeet-worker.py",
        "--model-id",
        "fake-model",
        "--device",
        "cuda",
        "--live-mode",
        "tuned",
        "--preset",
        "balanced",
        "--eou-silence-ms",
        "180",
        "--min-utterance-ms",
        "50",
        "--force-finalize-ms",
        "300",
        "--preroll-ms",
        "120",
        "--rms-threshold",
        "0.02",
        "--att-context-size",
        "70",
        "1",
    ]
    assert any('"preset": "fast"' in item for item in captured["writes"])


def test_run_live_command_supports_parakeet_legacy_mode(monkeypatch) -> None:
    captured = {}

    def fake_run_live(config: LiveConfig) -> None:
        captured["config"] = config

    monkeypatch.setattr("stt_exp.cli.run_live", fake_run_live)

    run_live_command(
        Namespace(
            providers=["parakeet"],
            chunk_ms=40,
            device=None,
            deepgram_model="nova-3",
            deepgram_endpointing_ms=300,
            deepgram_utterance_end_ms=1000,
            sherpa_model_dir="",
            sherpa_provider="cuda",
            sherpa_num_threads=2,
            parakeet_python="/tmp/python",
            parakeet_worker_script="/tmp/worker.py",
            parakeet_model_id="fake-model",
            parakeet_device="cuda",
            parakeet_live_mode="legacy",
            parakeet_preset="balanced",
            parakeet_eou_silence_ms=None,
            parakeet_min_utterance_ms=None,
            parakeet_force_finalize_ms=None,
            parakeet_preroll_ms=None,
            parakeet_rms_threshold=None,
            voxtral_uri="ws://127.0.0.1:8000/v1/realtime",
            voxtral_model="voxtral",
            voxtral_eou_mode="none",
            voxtral_eou_min_utterance_ms=300,
            voxtral_eou_silero_threshold=0.5,
            voxtral_eou_silero_min_silence_ms=500,
            voxtral_eou_smart_turn_model_path=None,
            voxtral_eou_smart_turn_cpu_count=1,
        )
    )

    config = captured["config"]
    assert config.parakeet_live_mode == "legacy"
