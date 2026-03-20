from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np


TARGET_SAMPLE_RATE = 16_000


@dataclass(slots=True)
class AudioClip:
    path: Path
    samples: np.ndarray
    sample_rate: int = TARGET_SAMPLE_RATE

    @property
    def duration_s(self) -> float:
        return float(len(self.samples) / self.sample_rate)


def load_audio(path: str | Path, sample_rate: int = TARGET_SAMPLE_RATE) -> AudioClip:
    audio_path = Path(path).expanduser().resolve()
    samples, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    if samples.size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")
    return AudioClip(path=audio_path, samples=samples.astype(np.float32), sample_rate=sample_rate)


def float_to_pcm16_bytes(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    return pcm16.tobytes()


def silence_pcm16(duration_ms: int, sample_rate: int = TARGET_SAMPLE_RATE) -> bytes:
    sample_count = int(sample_rate * duration_ms / 1000)
    silence = np.zeros(sample_count, dtype=np.float32)
    return float_to_pcm16_bytes(silence)


def iter_pcm16_chunks(
    samples: np.ndarray,
    chunk_ms: int,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> list[bytes]:
    samples_per_chunk = max(1, int(sample_rate * chunk_ms / 1000))
    chunks: list[bytes] = []
    for start in range(0, len(samples), samples_per_chunk):
        chunk = samples[start : start + samples_per_chunk]
        chunks.append(float_to_pcm16_bytes(chunk))
    return chunks
