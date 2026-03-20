import json
from pathlib import Path

from stt_exp.voxtral_model import prepare_voxtral_model


def test_prepare_voxtral_model_patches_existing_dir(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "tekken.json").write_text(
        json.dumps(
            {
                "audio": {
                    "transcription_delay_ms": 480,
                    "streaming_look_ahead_ms": 2.5,
                    "streaming_n_left_pad_tokens": 32,
                }
            }
        ),
        encoding="utf-8",
    )

    output_dir, changes = prepare_voxtral_model(
        repo_id="unused",
        out_dir=model_dir,
        transcription_delay_ms=80,
        streaming_look_ahead_ms=2.5,
        streaming_n_left_pad_tokens=0,
    )

    patched = json.loads((output_dir / "tekken.json").read_text(encoding="utf-8"))
    assert patched["audio"]["transcription_delay_ms"] == 80
    assert patched["audio"]["streaming_n_left_pad_tokens"] == 0
    assert changes["streaming_n_left_pad_tokens"] == 0
