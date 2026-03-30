from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

from stt_exp.audio import load_audio
from stt_exp.live_mic import (
    PARAKEET_LIVE_PRESET_NAMES,
    LiveConfig,
    apply_parakeet_live_preset,
    list_input_devices,
    run_live,
)
from stt_exp.manifest import BenchmarkItem, load_manifest, scan_audio_dir, write_manifest
from stt_exp.metrics import compute_timing, score_transcript
from stt_exp.providers.base import ProviderResult
from stt_exp.providers.deepgram_realtime import DeepgramRealtimeConfig, DeepgramRealtimeProvider
from stt_exp.providers.parakeet_external import ParakeetExternalConfig, ParakeetExternalProvider
from stt_exp.providers.sherpa_realtime import SherpaRealtimeConfig, SherpaRealtimeProvider
from stt_exp.providers.voxtral_realtime import VoxtralRealtimeConfig, VoxtralRealtimeProvider
from stt_exp.voxtral_model import DEFAULT_VOXTRAL_REPO, prepare_voxtral_model
from stt_exp.voxtral_eou import VOXTRAL_EOU_MODES, VoxtralEouConfig

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHERPA_MODEL_DIR = os.environ.get(
    "SHERPA_MODEL_DIR",
    str(REPO_ROOT / "models" / "sherpa-onnx-nemotron-speech-streaming-en-0.6b-int8-2026-01-14"),
)
DEFAULT_PARAKEET_BENCH_WORKER = str(REPO_ROOT / "scripts" / "parakeet_worker.py")
DEFAULT_PARAKEET_LIVE_WORKER = str(REPO_ROOT / "scripts" / "parakeet_live_worker.py")


def _default_parakeet_python() -> str:
    configured = os.environ.get("PARAKEET_PYTHON")
    if configured:
        return configured

    candidates = (
        REPO_ROOT / ".venv-parakeet" / "bin" / "python",
        REPO_ROOT.parent / "parakeet-exp" / ".venv" / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live and file-based STT harness.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bench = subparsers.add_parser("benchmark", help="Run benchmark sessions.")
    bench.add_argument("--manifest", type=Path, default=None)
    bench.add_argument("--audio", type=Path, default=None)
    bench.add_argument("--reference", type=str, default=None)
    bench.add_argument("--label", type=str, default=None)
    bench.add_argument(
        "--providers",
        nargs="+",
        choices=["deepgram", "voxtral", "sherpa", "parakeet"],
        default=["deepgram", "voxtral"],
    )
    bench.add_argument("--chunk-ms", type=int, default=40)
    bench.add_argument("--pace", choices=["realtime", "burst"], default="realtime")
    bench.add_argument("--repeats", type=int, default=1)
    bench.add_argument("--output", type=Path, default=None)
    bench.add_argument("--deepgram-model", type=str, default="nova-3")
    bench.add_argument("--deepgram-endpointing-ms", type=int, default=300)
    bench.add_argument("--deepgram-utterance-end-ms", type=int, default=1000)
    bench.add_argument("--deepgram-trailing-silence-ms", type=int, default=800)
    bench.add_argument(
        "--sherpa-model-dir",
        type=str,
        default=DEFAULT_SHERPA_MODEL_DIR,
    )
    bench.add_argument("--sherpa-provider", type=str, default="cuda")
    bench.add_argument("--sherpa-num-threads", type=int, default=2)
    bench.add_argument("--sherpa-trailing-silence-ms", type=int, default=800)
    bench.add_argument(
        "--parakeet-python",
        type=str,
        default=_default_parakeet_python(),
    )
    bench.add_argument(
        "--parakeet-worker-script",
        type=str,
        default=DEFAULT_PARAKEET_BENCH_WORKER,
    )
    bench.add_argument(
        "--parakeet-model-id",
        type=str,
        default="nvidia/parakeet_realtime_eou_120m-v1",
    )
    bench.add_argument("--parakeet-silence-chunks", type=int, default=2)
    bench.add_argument("--voxtral-uri", type=str, default="ws://127.0.0.1:8000/v1/realtime")
    bench.add_argument(
        "--voxtral-model",
        type=str,
        default=DEFAULT_VOXTRAL_REPO,
    )
    bench.add_argument("--voxtral-open-timeout-s", type=float, default=10.0)
    bench.add_argument("--voxtral-receive-timeout-s", type=float, default=30.0)
    bench.add_argument("--voxtral-warmup-silence-ms", type=int, default=1000)
    bench.add_argument("--voxtral-eou-mode", choices=VOXTRAL_EOU_MODES, default="none")
    bench.add_argument("--voxtral-eou-min-utterance-ms", type=int, default=300)
    bench.add_argument("--voxtral-eou-silero-threshold", type=float, default=0.5)
    bench.add_argument("--voxtral-eou-silero-min-silence-ms", type=int, default=500)
    bench.add_argument("--voxtral-eou-smart-turn-model-path", type=str, default=None)
    bench.add_argument("--voxtral-eou-smart-turn-cpu-count", type=int, default=1)
    bench.set_defaults(func=run_benchmark_command)

    prepare = subparsers.add_parser("prepare-voxtral-model", help="Download and patch a local Voxtral model copy.")
    prepare.add_argument("--repo-id", type=str, default=DEFAULT_VOXTRAL_REPO)
    prepare.add_argument("--out-dir", type=Path, required=True)
    prepare.add_argument("--transcription-delay-ms", type=int, default=None)
    prepare.add_argument("--streaming-look-ahead-ms", type=float, default=None)
    prepare.add_argument("--streaming-n-left-pad-tokens", type=int, default=0)
    prepare.set_defaults(func=prepare_voxtral_model_command)

    manifest = subparsers.add_parser("make-manifest", help="Create a CSV manifest from a directory of audio files.")
    manifest.add_argument("directory", type=Path)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--absolute-paths", action="store_true")
    manifest.add_argument("--no-infer-reference", action="store_true")
    manifest.set_defaults(func=make_manifest_command)

    live = subparsers.add_parser("live", help="Stream microphone audio to one or more realtime STT providers.")
    live.add_argument(
        "--providers",
        nargs="+",
        choices=["deepgram", "voxtral", "sherpa", "parakeet"],
        default=["deepgram", "voxtral"],
    )
    live.add_argument("--chunk-ms", type=int, default=40)
    live.add_argument("--device", type=int, default=None)
    live.add_argument("--deepgram-model", type=str, default="nova-3")
    live.add_argument("--deepgram-endpointing-ms", type=int, default=300)
    live.add_argument("--deepgram-utterance-end-ms", type=int, default=1000)
    live.add_argument(
        "--sherpa-model-dir",
        type=str,
        default=DEFAULT_SHERPA_MODEL_DIR,
    )
    live.add_argument("--sherpa-provider", type=str, default="cuda")
    live.add_argument("--sherpa-num-threads", type=int, default=2)
    live.add_argument(
        "--parakeet-python",
        type=str,
        default=_default_parakeet_python(),
    )
    live.add_argument(
        "--parakeet-worker-script",
        type=str,
        default=DEFAULT_PARAKEET_LIVE_WORKER,
    )
    live.add_argument(
        "--parakeet-model-id",
        type=str,
        default="nvidia/parakeet_realtime_eou_120m-v1",
    )
    live.add_argument("--parakeet-device", type=str, choices=["auto", "cuda", "cpu"], default="auto")
    live.add_argument("--parakeet-preset", choices=PARAKEET_LIVE_PRESET_NAMES, default="balanced")
    live.add_argument("--parakeet-eou-silence-ms", type=int, default=None)
    live.add_argument("--parakeet-min-utterance-ms", type=int, default=None)
    live.add_argument("--parakeet-force-finalize-ms", type=int, default=None)
    live.add_argument("--parakeet-preroll-ms", type=int, default=None)
    live.add_argument("--parakeet-rms-threshold", type=float, default=None)
    live.add_argument("--voxtral-uri", type=str, default="ws://127.0.0.1:8000/v1/realtime")
    live.add_argument("--voxtral-model", type=str, default=DEFAULT_VOXTRAL_REPO)
    live.add_argument("--voxtral-eou-mode", choices=VOXTRAL_EOU_MODES, default="none")
    live.add_argument("--voxtral-eou-min-utterance-ms", type=int, default=300)
    live.add_argument("--voxtral-eou-silero-threshold", type=float, default=0.5)
    live.add_argument("--voxtral-eou-silero-min-silence-ms", type=int, default=500)
    live.add_argument("--voxtral-eou-smart-turn-model-path", type=str, default=None)
    live.add_argument("--voxtral-eou-smart-turn-cpu-count", type=int, default=1)
    live.set_defaults(func=run_live_command)

    devices = subparsers.add_parser("devices", help="List available input audio devices.")
    devices.set_defaults(func=list_devices_command)

    return parser


def run_benchmark_command(args: argparse.Namespace) -> None:
    items = _resolve_items(args)
    providers = _build_providers(args)
    results: list[dict[str, object]] = []

    for provider in providers:
        if provider.name == "voxtral" and args.voxtral_warmup_silence_ms > 0:
            provider.warmup()

        for item in items:
            audio_clip = load_audio(item.audio_path)
            for repeat_index in range(args.repeats):
                try:
                    provider_result = provider.transcribe(audio_clip, label=item.label)
                except Exception as exc:
                    provider_result = ProviderResult(
                        provider=provider.name,
                        transcript_text="",
                        error=str(exc),
                        meta={"label": item.label},
                    )
                timing = compute_timing(
                    audio_duration_s=audio_clip.duration_s,
                    first_audio_sent_at=provider_result.first_audio_sent_at,
                    last_audio_sent_at=provider_result.last_audio_sent_at,
                    first_text_at=provider_result.first_text_at,
                    final_at=provider_result.final_at,
                )
                reference_text = item.reference_text if not provider_result.error else None
                quality = score_transcript(reference_text, provider_result.transcript_text)

                record = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "provider": provider_result.provider,
                    "label": item.label,
                    "repeat_index": repeat_index,
                    "audio_path": str(item.audio_path),
                    "audio_duration_s": audio_clip.duration_s,
                    "reference_text": item.reference_text,
                    "transcript_text": provider_result.transcript_text,
                    "error": provider_result.error,
                    "timing": asdict(timing),
                    "quality": asdict(quality),
                    "meta": provider_result.meta,
                    "events": [asdict(event) for event in provider_result.events],
                }
                results.append(record)
                _print_run(record)

    output_path = args.output or _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    _print_summary(results, output_path)


def prepare_voxtral_model_command(args: argparse.Namespace) -> None:
    output_dir, changes = prepare_voxtral_model(
        repo_id=args.repo_id,
        out_dir=args.out_dir,
        transcription_delay_ms=args.transcription_delay_ms,
        streaming_look_ahead_ms=args.streaming_look_ahead_ms,
        streaming_n_left_pad_tokens=args.streaming_n_left_pad_tokens,
    )
    print(json.dumps({"output_dir": str(output_dir), "changes": changes}, indent=2))


def make_manifest_command(args: argparse.Namespace) -> None:
    audio_paths = scan_audio_dir(args.directory)
    output = write_manifest(
        audio_paths,
        args.output,
        infer_reference=not args.no_infer_reference,
        absolute_paths=args.absolute_paths,
    )
    print(f"Wrote {len(audio_paths)} rows to {output}")


def run_live_command(args: argparse.Namespace) -> None:
    live_config = apply_parakeet_live_preset(
        LiveConfig(
            providers=args.providers,
            chunk_ms=args.chunk_ms,
            device=args.device,
            deepgram_model=args.deepgram_model,
            deepgram_endpointing_ms=args.deepgram_endpointing_ms,
            deepgram_utterance_end_ms=args.deepgram_utterance_end_ms,
            sherpa_model_dir=args.sherpa_model_dir,
            sherpa_provider=args.sherpa_provider,
            sherpa_num_threads=args.sherpa_num_threads,
            parakeet_python=args.parakeet_python,
            parakeet_worker_script=args.parakeet_worker_script,
            parakeet_model_id=args.parakeet_model_id,
            parakeet_device=args.parakeet_device,
            parakeet_preset=args.parakeet_preset,
            parakeet_eou_silence_ms=args.parakeet_eou_silence_ms,
            parakeet_min_utterance_ms=args.parakeet_min_utterance_ms,
            parakeet_force_finalize_ms=args.parakeet_force_finalize_ms,
            parakeet_preroll_ms=args.parakeet_preroll_ms,
            parakeet_rms_threshold=args.parakeet_rms_threshold,
            voxtral_uri=args.voxtral_uri,
            voxtral_model=args.voxtral_model,
            voxtral_eou_mode=args.voxtral_eou_mode,
            voxtral_eou_min_utterance_ms=args.voxtral_eou_min_utterance_ms,
            voxtral_eou_silero_threshold=args.voxtral_eou_silero_threshold,
            voxtral_eou_silero_min_silence_ms=args.voxtral_eou_silero_min_silence_ms,
            voxtral_eou_smart_turn_model_path=args.voxtral_eou_smart_turn_model_path,
            voxtral_eou_smart_turn_cpu_count=args.voxtral_eou_smart_turn_cpu_count,
        ),
        args.parakeet_preset,
    )
    # Explicit Parakeet args override the preset defaults at startup.
    overrides = {}
    if args.parakeet_eou_silence_ms is not None:
        overrides["parakeet_eou_silence_ms"] = args.parakeet_eou_silence_ms
    if args.parakeet_min_utterance_ms is not None:
        overrides["parakeet_min_utterance_ms"] = args.parakeet_min_utterance_ms
    if args.parakeet_force_finalize_ms is not None:
        overrides["parakeet_force_finalize_ms"] = args.parakeet_force_finalize_ms
    if args.parakeet_preroll_ms is not None:
        overrides["parakeet_preroll_ms"] = args.parakeet_preroll_ms
    if args.parakeet_rms_threshold is not None:
        overrides["parakeet_rms_threshold"] = args.parakeet_rms_threshold
    if overrides:
        live_config = replace(live_config, **overrides)
    run_live(live_config)


def list_devices_command(_args: argparse.Namespace) -> None:
    for device in list_input_devices():
        default_flag = " default" if device["is_default"] else ""
        print(
            f"{device['index']:>3}  {device['name']}  "
            f"channels={device['channels']}  samplerate={device['samplerate']}{default_flag}"
        )


def _resolve_items(args: argparse.Namespace) -> list[BenchmarkItem]:
    if args.manifest:
        return load_manifest(args.manifest)
    if args.audio:
        return [
            BenchmarkItem(
                audio_path=args.audio.expanduser().resolve(),
                reference_text=args.reference,
                label=args.label or args.audio.stem,
            )
        ]
    raise SystemExit("Provide either --manifest or --audio")


def _build_providers(args: argparse.Namespace):
    providers = []
    for name in args.providers:
        if name == "deepgram":
            providers.append(
                DeepgramRealtimeProvider(
                    DeepgramRealtimeConfig(
                        model=args.deepgram_model,
                        chunk_ms=args.chunk_ms,
                        pace=args.pace,
                        endpointing_ms=args.deepgram_endpointing_ms,
                        utterance_end_ms=args.deepgram_utterance_end_ms,
                        trailing_silence_ms=args.deepgram_trailing_silence_ms,
                        connect_timeout_s=5.0,
                        final_timeout_s=20.0,
                        post_audio_idle_s=1.0,
                    )
                )
            )
        elif name == "voxtral":
            providers.append(
                VoxtralRealtimeProvider(
                    VoxtralRealtimeConfig(
                        uri=args.voxtral_uri,
                        model=args.voxtral_model,
                        chunk_ms=args.chunk_ms,
                        pace=args.pace,
                        receive_timeout_s=args.voxtral_receive_timeout_s,
                        open_timeout_s=args.voxtral_open_timeout_s,
                        warmup_silence_ms=args.voxtral_warmup_silence_ms,
                        eou=VoxtralEouConfig(
                            mode=args.voxtral_eou_mode,
                            preset_name="cli",
                            min_utterance_ms=args.voxtral_eou_min_utterance_ms,
                            silero_threshold=args.voxtral_eou_silero_threshold,
                            silero_min_silence_ms=args.voxtral_eou_silero_min_silence_ms,
                            smart_turn_model_path=args.voxtral_eou_smart_turn_model_path,
                            smart_turn_cpu_count=args.voxtral_eou_smart_turn_cpu_count,
                        ),
                    )
                )
            )
        elif name == "sherpa":
            providers.append(
                SherpaRealtimeProvider(
                    SherpaRealtimeConfig(
                        model_dir=args.sherpa_model_dir,
                        provider=args.sherpa_provider,
                        num_threads=args.sherpa_num_threads,
                        chunk_ms=args.chunk_ms,
                        pace=args.pace,
                        warmup=True,
                        trailing_silence_ms=args.sherpa_trailing_silence_ms,
                    )
                )
            )
        elif name == "parakeet":
            providers.append(
                ParakeetExternalProvider(
                    ParakeetExternalConfig(
                        python_executable=args.parakeet_python,
                        worker_script=args.parakeet_worker_script,
                        model_id=args.parakeet_model_id,
                        pace=args.pace,
                        silence_chunks=args.parakeet_silence_chunks,
                    )
                )
            )
    return providers


def _default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("results") / f"benchmark-{timestamp}.json"


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}"


def _print_run(record: dict[str, object]) -> None:
    timing = record["timing"]
    quality = record["quality"]
    assert isinstance(timing, dict)
    assert isinstance(quality, dict)
    print(
        f"{record['provider']:>8s}  "
        f"{record['label']:<24s}  "
        f"ttft={_fmt(timing.get('ttft_ms'))}ms  "
        f"final={_fmt(timing.get('final_latency_ms'))}ms  "
        f"tail={_fmt(timing.get('tail_latency_ms'))}ms  "
        f"wer={_fmt((quality.get('wer') or 0.0) * 100 if quality.get('wer') is not None else None)}%  "
        f"error={record['error'] or '-'}"
    )


def _print_summary(results: list[dict[str, object]], output_path: Path) -> None:
    grouped: dict[str, list[dict[str, object]]] = {}
    for result in results:
        grouped.setdefault(str(result["provider"]), []).append(result)

    print("\nSummary")
    for provider, rows in grouped.items():
        ttfts = _collect_metric(rows, ("timing", "ttft_ms"))
        finals = _collect_metric(rows, ("timing", "final_latency_ms"))
        wers = _collect_metric(rows, ("quality", "wer"))
        success_count = sum(1 for row in rows if not row.get("error"))
        print(
            f"{provider:>8s}  "
            f"runs={len(rows)}  "
            f"ok={success_count}  "
            f"mean_ttft={_fmt(mean(ttfts) if ttfts else None)}ms  "
            f"mean_final={_fmt(mean(finals) if finals else None)}ms  "
            f"mean_wer={_fmt(mean(wers) * 100 if wers else None)}%"
        )
    print(f"\nSaved results to {output_path}")


def _collect_metric(rows: list[dict[str, object]], path: tuple[str, str]) -> list[float]:
    values: list[float] = []
    outer_key, inner_key = path
    for row in rows:
        outer = row.get(outer_key)
        if not isinstance(outer, dict):
            continue
        value = outer.get(inner_key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values
