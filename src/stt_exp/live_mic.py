from __future__ import annotations

import asyncio
import base64
import json
import os
import queue
import signal
import select
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from contextlib import contextmanager
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()


SAMPLE_RATE = 16_000


@dataclass(slots=True)
class LiveConfig:
    providers: list[str]
    chunk_ms: int
    device: int | None
    deepgram_model: str
    deepgram_endpointing_ms: int
    deepgram_utterance_end_ms: int
    sherpa_model_dir: str
    sherpa_provider: str
    sherpa_num_threads: int
    moonshine_language: str
    moonshine_model_arch: int | None
    parakeet_python: str
    parakeet_worker_script: str
    parakeet_model_id: str
    parakeet_device: str
    voxtral_uri: str
    voxtral_model: str


@dataclass(slots=True)
class ProviderDisplayState:
    status: str = "starting"
    text: str = ""
    finalized: bool = False


class LiveDisplay:
    def __init__(self, providers: list[str]):
        self.providers = providers
        self.states = {provider: ProviderDisplayState() for provider in providers}
        self.lock = threading.Lock()
        self.dynamic = sys.stdout.isatty()
        self._rendered = False

    def init(self) -> None:
        with self.lock:
            if self.dynamic:
                for provider in self.providers:
                    print(self._format_line(provider), flush=True)
                self._rendered = True

    def emit(self, provider: str, text: str, kind: str = "status") -> None:
        with self.lock:
            state = self.states[provider]
            if kind == "status":
                state.status = text
            elif kind == "clear":
                state.text = ""
                state.finalized = False
                state.status = text or "listening"
            elif kind == "replace":
                state.text = text
                state.finalized = False
                if state.status in {"starting", "final"}:
                    state.status = "listening"
            elif kind == "append":
                state.text += text
                state.finalized = False
                if state.status in {"starting", "final"}:
                    state.status = "listening"
            elif kind == "final":
                state.text = text
                state.status = "final"
                state.finalized = True
            else:
                state.status = text
            self._render_locked(provider, text, kind)

    def clear_all(self, status: str = "cleared") -> None:
        with self.lock:
            for provider in self.providers:
                state = self.states[provider]
                state.text = ""
                state.finalized = False
                state.status = status
            self._render_locked("", status, "clear")

    def _format_line(self, provider: str) -> str:
        state = self.states[provider]
        text = state.text.strip()
        if state.finalized and text:
            text = f"{text} [FINAL]"
        line = f"{provider:8s} | {state.status:10s}"
        if text:
            line += f" | {text}"
        return line

    def _render_locked(self, provider: str, text: str, kind: str) -> None:
        if not self.dynamic:
            if kind == "status":
                print(f"[{provider}] {text}", flush=True)
            elif kind == "final":
                print(f"[{provider}] FINAL {text}", flush=True)
            elif kind == "clear":
                print("[system] Cleared current text.", flush=True)
            else:
                print(f"[{provider}] {text}", flush=True)
            return

        if self._rendered:
            sys.stdout.write(f"\x1b[{len(self.providers)}F")
        for name in self.providers:
            sys.stdout.write("\x1b[2K")
            sys.stdout.write(self._format_line(name) + "\n")
        sys.stdout.flush()


@contextmanager
def _cbreak_stdin():
    if not sys.stdin.isatty():
        yield
        return

    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def list_input_devices() -> list[dict[str, object]]:
    import sounddevice as sd

    devices = []
    default_input, _default_output = sd.default.device
    for index, device in enumerate(sd.query_devices()):
        if int(device["max_input_channels"]) <= 0:
            continue
        devices.append(
            {
                "index": index,
                "name": device["name"],
                "is_default": index == default_input,
                "channels": int(device["max_input_channels"]),
                "samplerate": int(device["default_samplerate"]),
            }
        )
    return devices


def _format_voxtral_connection_error(uri: str, exc: OSError) -> str:
    parsed = urlparse(uri)
    host = parsed.hostname or "unknown-host"
    port = parsed.port or (443 if parsed.scheme == "wss" else 80)
    detail = str(exc).strip() or exc.__class__.__name__
    if host in {"127.0.0.1", "localhost", "::1"}:
        return (
            f"offline at {host}:{port}; start ./scripts/serve_voxtral_optimized.sh "
            f"({detail})"
        )
    return f"offline at {host}:{port}; check --voxtral-uri ({detail})"


def _append_voxtral_delta(current_text: str, delta: str) -> str:
    if delta == "":
        return current_text
    return current_text + delta


def run_live(config: LiveConfig) -> None:
    import sounddevice as sd

    stop_event = threading.Event()
    display = LiveDisplay(config.providers)
    queues = {provider: queue.Queue(maxsize=256) for provider in config.providers}
    control_queues = {provider: queue.Queue(maxsize=16) for provider in config.providers}
    threads: list[threading.Thread] = []
    samples_per_chunk = max(1, int(SAMPLE_RATE * config.chunk_ms / 1000))

    def audio_callback(indata, frames, time_info, status) -> None:
        chunk = bytes(indata)
        for q in queues.values():
            try:
                q.put_nowait(chunk)
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(chunk)
                except queue.Full:
                    pass

    if "deepgram" in config.providers:
        threads.append(
            threading.Thread(
                target=_run_deepgram_live,
                args=(config, queues["deepgram"], control_queues["deepgram"], stop_event, display.emit),
                daemon=True,
            )
        )
    if "voxtral" in config.providers:
        threads.append(
            threading.Thread(
                target=_run_voxtral_live,
                args=(config, queues["voxtral"], control_queues["voxtral"], stop_event, display.emit),
                daemon=True,
            )
        )
    if "sherpa" in config.providers:
        threads.append(
            threading.Thread(
                target=_run_sherpa_live,
                args=(config, queues["sherpa"], control_queues["sherpa"], stop_event, display.emit),
                daemon=True,
            )
        )
    if "moonshine" in config.providers:
        threads.append(
            threading.Thread(
                target=_run_moonshine_live,
                args=(config, queues["moonshine"], control_queues["moonshine"], stop_event, display.emit),
                daemon=True,
            )
        )
    if "parakeet" in config.providers:
        threads.append(
            threading.Thread(
                target=_run_parakeet_live,
                args=(config, queues["parakeet"], control_queues["parakeet"], stop_event, display.emit),
                daemon=True,
            )
        )

    print(f"[system] Listening on device={config.device if config.device is not None else 'default'}", flush=True)
    print(f"[system] Providers: {', '.join(config.providers)}", flush=True)
    print("[system] Press Ctrl+C to stop. Press r to clear/reset current utterance.", flush=True)
    display.init()

    for thread in threads:
        thread.start()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda _signum, _frame: stop_event.set())

    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=samples_per_chunk,
        channels=1,
        dtype="int16",
        device=config.device,
        callback=audio_callback,
    )

    with _cbreak_stdin():
        with stream:
            while not stop_event.is_set():
                if sys.stdin.isatty():
                    ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if ready:
                        key = sys.stdin.read(1)
                        if key.lower() == "r":
                            display.clear_all()
                            for control_queue in control_queues.values():
                                try:
                                    control_queue.put_nowait("reset")
                                except queue.Full:
                                    pass
                            continue
                else:
                    time.sleep(0.1)

    for thread in threads:
        thread.join(timeout=2)


def _run_deepgram_live(
    config: LiveConfig,
    audio_queue: queue.Queue[bytes],
    control_queue: queue.Queue[str],
    stop_event: threading.Event,
    emit,
) -> None:
    from deepgram import DeepgramClient
    from deepgram.core.events import EventType

    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        emit("deepgram", "DEEPGRAM_API_KEY is not set")
        return

    client = DeepgramClient(api_key=api_key)

    while not stop_event.is_set():
        ctx = client.listen.v1.connect(
            model=config.deepgram_model,
            encoding="linear16",
            sample_rate=str(SAMPLE_RATE),
            channels="1",
            interim_results="true",
            vad_events="true",
            utterance_end_ms=str(config.deepgram_utterance_end_ms),
            endpointing=str(config.deepgram_endpointing_ms),
        )
        socket = ctx.__enter__()

        def on_open(_socket) -> None:
            emit("deepgram", "connected", kind="status")

        def on_message(message) -> None:
            msg_type = getattr(message, "type", "") if not isinstance(message, dict) else message.get("type", "")
            if msg_type == "SpeechStarted":
                emit("deepgram", "speech", kind="status")
                return
            if msg_type == "UtteranceEnd":
                emit("deepgram", "utterance end", kind="status")
                return
            if msg_type != "Results":
                return

            transcript = ""
            is_final = False
            try:
                if isinstance(message, dict):
                    channel = message.get("channel", {})
                    alternatives = channel.get("alternatives", [])
                    if alternatives:
                        transcript = alternatives[0].get("transcript", "")
                    is_final = bool(message.get("is_final", False))
                else:
                    if message.channel and message.channel.alternatives:
                        transcript = message.channel.alternatives[0].transcript
                    is_final = bool(message.is_final)
            except (AttributeError, IndexError, KeyError):
                return

            transcript = transcript.strip()
            if transcript:
                emit("deepgram", transcript, kind="final" if is_final else "replace")

        def on_error(error) -> None:
            emit("deepgram", f"error: {error}", kind="status")
            stop_event.set()

        socket.on(EventType.OPEN, on_open)
        socket.on(EventType.MESSAGE, on_message)
        socket.on(EventType.ERROR, on_error)

        listener = threading.Thread(target=socket.start_listening, daemon=True)
        listener.start()

        reset_requested = False
        try:
            while not stop_event.is_set():
                try:
                    action = control_queue.get_nowait()
                except queue.Empty:
                    action = None
                if action == "reset":
                    reset_requested = True
                    try:
                        socket.send_finalize()
                    except Exception:
                        pass
                    break

                try:
                    chunk = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                socket.send_media(chunk)
        finally:
            try:
                socket.send_close_stream()
            except Exception:
                pass
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
            listener.join(timeout=1)

        if not reset_requested:
            break


def _run_voxtral_live(
    config: LiveConfig,
    audio_queue: queue.Queue[bytes],
    control_queue: queue.Queue[str],
    stop_event: threading.Event,
    emit,
) -> None:
    asyncio.run(_run_voxtral_live_async(config, audio_queue, control_queue, stop_event, emit))


async def _run_voxtral_live_async(
    config: LiveConfig,
    audio_queue: queue.Queue[bytes],
    control_queue: queue.Queue[str],
    stop_event: threading.Event,
    emit,
) -> None:
    import websockets

    last_connect_error: str | None = None

    while not stop_event.is_set():
        try:
            async with websockets.connect(config.voxtral_uri, open_timeout=10.0, max_size=None) as ws:
                last_connect_error = None
                hello = json.loads(await ws.recv())
                if hello.get("type") != "session.created":
                    emit("voxtral", f"error: unexpected handshake", kind="status")
                    stop_event.set()
                    return

                emit("voxtral", "connected", kind="status")
                await ws.send(json.dumps({"type": "session.update", "model": config.voxtral_model}))
                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                current_text = ""
                reset_requested = False

                async def send_audio() -> None:
                    nonlocal reset_requested
                    while not stop_event.is_set():
                        try:
                            action = control_queue.get_nowait()
                        except queue.Empty:
                            action = None
                        if action == "reset":
                            reset_requested = True
                            await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
                            return
                        try:
                            chunk = await asyncio.to_thread(audio_queue.get, True, 0.1)
                        except queue.Empty:
                            continue
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "input_audio_buffer.append",
                                    "audio": base64.b64encode(chunk).decode("utf-8"),
                                }
                            )
                        )

                async def recv_text() -> None:
                    nonlocal current_text
                    async for raw_message in ws:
                        message = json.loads(raw_message)
                        if message.get("type") == "transcription.delta":
                            delta = message.get("delta", "")
                            if delta != "":
                                current_text = _append_voxtral_delta(current_text, delta)
                                emit("voxtral", current_text, kind="replace")
                        elif message.get("type") == "transcription.done":
                            text = message.get("text", "").strip() or current_text.strip()
                            if text and not reset_requested:
                                emit("voxtral", text, kind="final")
                            current_text = ""
                            if reset_requested:
                                return
                        elif message.get("type") == "error":
                            emit("voxtral", f"error: {message.get('error')}", kind="status")
                            stop_event.set()
                            return

                sender = asyncio.create_task(send_audio())
                receiver = asyncio.create_task(recv_text())
                while not stop_event.is_set() and not reset_requested:
                    await asyncio.sleep(0.1)
                sender.cancel()
                receiver.cancel()
                try:
                    await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
                except Exception:
                    pass
                if not reset_requested:
                    break
        except OSError as exc:
            message = _format_voxtral_connection_error(config.voxtral_uri, exc)
            if message != last_connect_error:
                emit("voxtral", message, kind="status")
                last_connect_error = message
            await asyncio.sleep(2.0)
        except Exception as exc:
            emit("voxtral", f"error: {exc}", kind="status")
            stop_event.set()
            break


def _run_sherpa_live(
    config: LiveConfig,
    audio_queue: queue.Queue[bytes],
    control_queue: queue.Queue[str],
    stop_event: threading.Event,
    emit,
) -> None:
    import glob
    import sherpa_onnx
    import numpy as np

    def find_model(prefix: str) -> str:
        candidates = glob.glob(os.path.join(config.sherpa_model_dir, f"{prefix}*.int8.onnx"))
        if not candidates:
            candidates = glob.glob(os.path.join(config.sherpa_model_dir, f"{prefix}*.onnx"))
        if not candidates:
            raise RuntimeError(f"Missing {prefix} model in {config.sherpa_model_dir}")
        return sorted(candidates)[0]

    feature_dim = 128 if "nemotron" in config.sherpa_model_dir.lower() else 80
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=os.path.join(config.sherpa_model_dir, "tokens.txt"),
        encoder=find_model("encoder"),
        decoder=find_model("decoder"),
        joiner=find_model("joiner"),
        num_threads=config.sherpa_num_threads,
        sample_rate=SAMPLE_RATE,
        feature_dim=feature_dim,
        decoding_method="greedy_search",
        provider=config.sherpa_provider,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=300,
    )
    stream = recognizer.create_stream()
    last_text = ""
    emit("sherpa", "connected", kind="status")
    while not stop_event.is_set():
        try:
            action = control_queue.get_nowait()
        except queue.Empty:
            action = None
        if action == "reset":
            recognizer.reset(stream)
            last_text = ""
            emit("sherpa", "manual reset", kind="status")
            continue
        try:
            chunk = audio_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        stream.accept_waveform(SAMPLE_RATE, samples)
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        text = recognizer.get_result(stream).strip()
        if text and text != last_text:
            last_text = text
            emit("sherpa", text, kind="replace")
        if recognizer.is_endpoint(stream) and text:
            emit("sherpa", text, kind="final")
            recognizer.reset(stream)
            last_text = ""


def _run_moonshine_live(
    config: LiveConfig,
    audio_queue: queue.Queue[bytes],
    control_queue: queue.Queue[str],
    stop_event: threading.Event,
    emit,
) -> None:
    import numpy as np
    from moonshine_voice import ModelArch, TranscriptEventListener, Transcriber, get_model_for_language

    wanted_arch = None if config.moonshine_model_arch is None else ModelArch(config.moonshine_model_arch)
    model_path, model_arch = get_model_for_language(
        wanted_language=config.moonshine_language,
        wanted_model_arch=wanted_arch,
    )

    class Listener(TranscriptEventListener):
        def on_line_text_changed(self, event) -> None:
            text = event.line.text.strip()
            if text:
                emit("moonshine", text, kind="replace")

        def on_line_completed(self, event) -> None:
            text = event.line.text.strip()
            if text:
                emit("moonshine", text, kind="final")

    with Transcriber(model_path=model_path, model_arch=model_arch, update_interval=config.chunk_ms / 1000.0) as transcriber:
        stream = transcriber.create_stream(update_interval=config.chunk_ms / 1000.0)
        stream.add_listener(Listener())
        stream.start()
        emit("moonshine", "connected", kind="status")
        while not stop_event.is_set():
            try:
                action = control_queue.get_nowait()
            except queue.Empty:
                action = None
            if action == "reset":
                stream.stop()
                stream.close()
                stream = transcriber.create_stream(update_interval=config.chunk_ms / 1000.0)
                stream.add_listener(Listener())
                stream.start()
                emit("moonshine", "manual reset", kind="status")
                continue
            try:
                chunk = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            stream.add_audio(samples.tolist(), SAMPLE_RATE)
        stream.stop()
        stream.close()


def _run_parakeet_live(
    config: LiveConfig,
    audio_queue: queue.Queue[bytes],
    control_queue: queue.Queue[str],
    stop_event: threading.Event,
    emit,
) -> None:
    marker = "PARAKEET_LIVE_EVENT_JSON="
    cmd = [
        config.parakeet_python,
        "-u",
        config.parakeet_worker_script,
        "--model-id",
        config.parakeet_model_id,
        "--device",
        config.parakeet_device,
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as exc:
        emit("parakeet", f"error: failed to start worker: {exc}", kind="status")
        stop_event.set()
        return

    def read_worker_output() -> None:
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            line = raw_line.strip()
            if not line:
                continue
            if not line.startswith(marker):
                continue
            try:
                message = json.loads(line[len(marker) :])
            except json.JSONDecodeError:
                continue
            msg_type = message.get("type")
            if msg_type == "status":
                emit("parakeet", message.get("message", "status"), kind="status")
            elif msg_type == "partial":
                text = message.get("text", "").strip()
                if text:
                    emit("parakeet", text, kind="replace")
            elif msg_type == "final":
                text = message.get("text", "").strip()
                if text:
                    emit("parakeet", text, kind="final")
            elif msg_type == "error":
                emit("parakeet", f"error: {message.get('message', 'unknown worker error')}", kind="status")
                stop_event.set()
                return

    reader = threading.Thread(target=read_worker_output, daemon=True)
    reader.start()

    try:
        assert proc.stdin is not None
        while not stop_event.is_set():
            if proc.poll() is not None:
                emit("parakeet", f"error: worker exited with code {proc.returncode}", kind="status")
                stop_event.set()
                break
            try:
                action = control_queue.get_nowait()
            except queue.Empty:
                action = None
            if action == "reset":
                proc.stdin.write(json.dumps({"type": "reset"}) + "\n")
                proc.stdin.flush()
                continue
            try:
                chunk = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            payload = {
                "type": "audio",
                "audio": base64.b64encode(chunk).decode("utf-8"),
            }
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
    except Exception as exc:
        emit("parakeet", f"error: {exc}", kind="status")
        stop_event.set()
    finally:
        try:
            if proc.stdin is not None:
                proc.stdin.write(json.dumps({"type": "stop"}) + "\n")
                proc.stdin.flush()
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        reader.join(timeout=1)
