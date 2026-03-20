from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_REPO_ID = "k2-fsa/sherpa-onnx-nemotron-speech-streaming-en-0.6b-int8-2026-01-14"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Sherpa streaming model into a local directory.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("models") / "sherpa-onnx-nemotron-speech-streaming-en-0.6b-int8-2026-01-14",
    )
    args = parser.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
    )
    print(path)


if __name__ == "__main__":
    main()
