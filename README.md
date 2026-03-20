# stt-exp

Realtime STT benchmark and live-CLI harness for:

- `deepgram`
- `voxtral`
- `sherpa`
- `parakeet`
- `moonshine` (optional, not part of the current main comparison)

The repo is built around one idea: feed the same audio to each engine with the
same pacing, then compare latency and transcript output.

## What This Repo Covers

- File-based realtime benchmarks with shared metrics
- Live microphone CLI that fans the same mic audio out to multiple providers in parallel
- Voxtral tuning based on Mistral's public discussion:
  - warmup before measurement
  - `streaming_n_left_pad_tokens=0`
- Repo-local scripts for serving Voxtral and downloading the Sherpa model

## Current Comparison Summary

Main file benchmark without Moonshine:

- Deepgram: `ttft 1081.3ms`, `final 2506.8ms`, `tail 115.7ms`
- Parakeet: `ttft 607.7ms`, `final 2664.0ms`, `tail 332.4ms`
- Sherpa: `ttft 1307.3ms`, `final 2633.0ms`, `tail 116.2ms`
- Voxtral: `ttft 2460.7ms`, `final 2649.3ms`, `tail 229.3ms`

Source: [results/compare-no-moonshine-s2.json](/home/logan/Projects/stt-exp/results/compare-no-moonshine-s2.json)

Important caveat: the quick test manifest derives references from filenames, so
WER is not trustworthy unless you replace those references with real transcripts.

## Requirements

- Linux
- `uv`
- NVIDIA GPU recommended
- CUDA-capable environment for `voxtral`, `sherpa`, and `parakeet`
- Deepgram API key if you want to run Deepgram

## Repo Setup

Install the main benchmark/live environment:

```bash
uv sync --extra dev
cp .env.example .env
```

Fill in `DEEPGRAM_API_KEY` in `.env` for Deepgram.

The main `uv` environment covers:

- benchmark CLI
- Deepgram client
- Sherpa runtime
- Moonshine runtime
- live microphone CLI

Voxtral serving uses the optional `voxtral` extra. Parakeet uses a separate
Python environment because NeMo has heavier and more CUDA-sensitive dependencies.

## Provider Setup

### Deepgram

Only needs:

```bash
cp .env.example .env
```

Then set:

```bash
DEEPGRAM_API_KEY=...
```

### Voxtral

Install the serve dependencies:

```bash
uv sync --extra dev --extra voxtral
```

Prepare the optimized local model copy:

```bash
uv run stt-exp prepare-voxtral-model \
  --out-dir models/voxtral-optimized \
  --streaming-n-left-pad-tokens 0 \
  --transcription-delay-ms 80
```

Serve it:

```bash
./scripts/serve_voxtral_optimized.sh
```

Baseline, without the local patch:

```bash
./scripts/serve_voxtral_baseline.sh
```

Notes:

- default server URL is `ws://127.0.0.1:8000/v1/realtime`
- the serve scripts now default to a lower-memory realtime profile: `GPU_MEMORY_UTILIZATION=0.55` and `MAX_MODEL_LEN=4096`
- adjust `GPU_MEMORY_UTILIZATION`, `MAX_MODEL_LEN`, `VLLM_HOST`, or `VLLM_PORT` via env vars if needed
- `./scripts/run_live_external_models.sh` now auto-starts `./scripts/serve_voxtral_optimized.sh` when Voxtral is selected and no local server is already listening

### Sherpa

Download the model into `models/`:

```bash
uv run python scripts/download_sherpa_model.py
```

Default target directory:

```text
models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-int8-2026-01-14
```

If you want a different location, either pass `--sherpa-model-dir` on the CLI or
set `SHERPA_MODEL_DIR` in `.env`.

### Parakeet

Parakeet is intentionally kept in a separate env. Recommended repo-local setup:

```bash
uv venv .venv-parakeet --python 3.10
uv pip install --python .venv-parakeet/bin/python torch torchvision torchaudio
uv pip install --python .venv-parakeet/bin/python nemo_toolkit[asr] soundfile librosa
```

Then either export:

```bash
PARAKEET_PYTHON=.venv-parakeet/bin/python
```

or put it in `.env`:

```bash
PARAKEET_PYTHON=.venv-parakeet/bin/python
```

The default model is:

```text
nvidia/parakeet_realtime_eou_120m-v1
```

It is downloaded automatically by NeMo on first use.

Note: the live Parakeet path is persistent and EOU-aware, but the frontend still
recomputes mel features over accumulated audio. Encoder/decoder state is streamed;
the feature extractor is not yet a fully incremental online frontend.

### Moonshine

Moonshine is installed in the main environment and downloads its model on first run.
It is available in the CLI but not part of the current main comparison.

## Quick Start

Create a manifest from a directory of WAV files:

```bash
uv run stt-exp make-manifest ../voxtral-realtime/test_wavs \
  --output data/test-wavs.csv
```

Run the current four-way comparison:

```bash
uv run stt-exp benchmark \
  --manifest data/test-wavs.csv \
  --providers deepgram voxtral sherpa parakeet \
  --chunk-ms 40 \
  --pace realtime \
  --repeats 1 \
  --parakeet-silence-chunks 2 \
  --output results/compare-no-moonshine-s2.json
```

Single-file smoke test:

```bash
uv run stt-exp benchmark \
  --audio ../voxtral-realtime/test_wavs/001_hey_man_how_are_you.wav \
  --reference "hey man how are you" \
  --providers deepgram voxtral sherpa parakeet \
  --chunk-ms 40 \
  --pace realtime
```

## Live CLI

List devices:

```bash
uv run stt-exp devices
```

Run all main providers in parallel on the default input device:

```bash
uv run stt-exp live \
  --providers deepgram voxtral sherpa parakeet \
  --chunk-ms 40
```

Pick a specific mic:

```bash
uv run stt-exp live \
  --providers deepgram voxtral sherpa parakeet \
  --device 11 \
  --chunk-ms 40
```

Behavior:

- all selected providers run in parallel
- the CLI prints partial text as each provider emits it
- it also prints `FINAL ...` when that provider detects end-of-utterance

## Metrics

Each benchmark run records:

- `ttft_ms`: first transcript text after the first audio chunk is sent
- `final_latency_ms`: first audio chunk to final transcript
- `tail_latency_ms`: last real audio chunk to final transcript
- `realtime_factor`: wall time divided by audio duration
- `wer` and `cer`: normalized quality metrics when a real reference is present

`tail_latency_ms` is effectively "how long after the last real speech chunk did
the final transcript show up", which includes endpointing behavior.

## Reproducing The Existing Results

1. `uv sync --extra dev --extra voxtral`
2. Set `DEEPGRAM_API_KEY` in `.env`
3. `uv run python scripts/download_sherpa_model.py`
4. Create `.venv-parakeet` and install NeMo
5. Prepare the optimized Voxtral model
6. Start `./scripts/serve_voxtral_optimized.sh`
7. Run:

```bash
uv run stt-exp benchmark \
  --manifest data/test-wavs.csv \
  --providers deepgram voxtral sherpa parakeet \
  --chunk-ms 40 \
  --pace realtime \
  --repeats 1 \
  --parakeet-silence-chunks 2 \
  --output results/compare-no-moonshine-s2.json
```

## Files

- [src/stt_exp/cli.py](/home/logan/Projects/stt-exp/src/stt_exp/cli.py)
- [src/stt_exp/live_mic.py](/home/logan/Projects/stt-exp/src/stt_exp/live_mic.py)
- [src/stt_exp/providers/deepgram_realtime.py](/home/logan/Projects/stt-exp/src/stt_exp/providers/deepgram_realtime.py)
- [src/stt_exp/providers/voxtral_realtime.py](/home/logan/Projects/stt-exp/src/stt_exp/providers/voxtral_realtime.py)
- [src/stt_exp/providers/sherpa_realtime.py](/home/logan/Projects/stt-exp/src/stt_exp/providers/sherpa_realtime.py)
- [src/stt_exp/providers/parakeet_external.py](/home/logan/Projects/stt-exp/src/stt_exp/providers/parakeet_external.py)
- [scripts/parakeet_live_worker.py](/home/logan/Projects/stt-exp/scripts/parakeet_live_worker.py)
- [scripts/download_sherpa_model.py](/home/logan/Projects/stt-exp/scripts/download_sherpa_model.py)

## Upstream References

- Voxtral model card: https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602
- Voxtral discussion: https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602/discussions/11
- vLLM realtime protocol: https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/realtime/protocol/
