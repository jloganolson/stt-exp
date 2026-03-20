from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_VOXTRAL_REPO = "mistralai/Voxtral-Mini-4B-Realtime-2602"


def prepare_voxtral_model(
    *,
    repo_id: str,
    out_dir: str | Path,
    transcription_delay_ms: int | None,
    streaming_look_ahead_ms: float | None,
    streaming_n_left_pad_tokens: int | None,
) -> tuple[Path, dict[str, object]]:
    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.json"
    if not config_path.exists():
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
        )

    tekken_path = output_dir / "tekken.json"
    with tekken_path.open("r", encoding="utf-8") as handle:
        tekken = json.load(handle)

    audio = tekken.setdefault("audio", {})
    changes: dict[str, object] = {}

    if transcription_delay_ms is not None:
        audio["transcription_delay_ms"] = transcription_delay_ms
        changes["transcription_delay_ms"] = transcription_delay_ms
    if streaming_look_ahead_ms is not None:
        audio["streaming_look_ahead_ms"] = streaming_look_ahead_ms
        changes["streaming_look_ahead_ms"] = streaming_look_ahead_ms
    if streaming_n_left_pad_tokens is not None:
        audio["streaming_n_left_pad_tokens"] = streaming_n_left_pad_tokens
        changes["streaming_n_left_pad_tokens"] = streaming_n_left_pad_tokens

    with tekken_path.open("w", encoding="utf-8") as handle:
        json.dump(tekken, handle, indent=2)
        handle.write("\n")

    return output_dir, changes
