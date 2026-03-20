from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}


@dataclass(slots=True)
class BenchmarkItem:
    audio_path: Path
    reference_text: str | None
    label: str


def infer_reference_from_filename(path: str | Path) -> str:
    stem = Path(path).stem
    stem = re.sub(r"^[0-9]+[_-]*", "", stem)
    return stem.replace("_", " ").replace("-", " ").strip()


def load_manifest(path: str | Path) -> list[BenchmarkItem]:
    manifest_path = Path(path).expanduser().resolve()
    if manifest_path.suffix == ".jsonl":
        items = []
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            items.append(_item_from_row(row, manifest_path.parent))
        return items
    if manifest_path.suffix != ".csv":
        raise ValueError(f"Unsupported manifest format: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [_item_from_row(row, manifest_path.parent) for row in reader]


def scan_audio_dir(directory: str | Path) -> list[Path]:
    root = Path(directory).expanduser().resolve()
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def write_manifest(
    audio_paths: list[Path],
    output_path: str | Path,
    *,
    infer_reference: bool,
    absolute_paths: bool,
) -> Path:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd().resolve()
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audio_path", "reference_text", "label"])
        writer.writeheader()
        for audio_path in audio_paths:
            path_value = str(audio_path)
            if not absolute_paths:
                try:
                    path_value = str(audio_path.relative_to(cwd))
                except ValueError:
                    path_value = str(audio_path)
            writer.writerow(
                {
                    "audio_path": path_value,
                    "reference_text": infer_reference_from_filename(audio_path) if infer_reference else "",
                    "label": audio_path.stem,
                }
            )
    return output


def _item_from_row(row: dict[str, object], manifest_dir: Path) -> BenchmarkItem:
    raw_audio_path = str(row["audio_path"]).strip()
    audio_path = Path(raw_audio_path).expanduser()
    if not audio_path.is_absolute():
        audio_path = (manifest_dir / audio_path).resolve()
    reference = str(row.get("reference_text") or "").strip() or None
    label = str(row.get("label") or audio_path.stem).strip()
    return BenchmarkItem(audio_path=audio_path, reference_text=reference, label=label)
