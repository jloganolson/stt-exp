# stt-exp

Realtime speech-to-text harness for running the same microphone audio through:

- `deepgram`
- `voxtral`
- `sherpa`
- `parakeet`

The main supported entrypoint is:

```bash
./scripts/run_live_external_models.sh
```

It starts the live CLI, fans the mic audio out to all configured providers, and
auto-starts the local optimized Voxtral server when needed.

## What This Repo Is For

- running the four-provider live comparison quickly
- tuning Voxtral realtime behavior, especially EOU behavior
- running deterministic WAV-based checks when live behavior is unclear

If you only care about live testing, you can treat everything else in the repo
as support code for `run_live_external_models.sh`.

## Requirements

- Linux
- `uv`
- Python `3.12`
- NVIDIA GPU for local `voxtral`, `sherpa`, and usually `parakeet`
- `DEEPGRAM_API_KEY` if you want Deepgram enabled

## Install

Base environment:

```bash
uv sync --extra dev
```

If you want Voxtral serving from this repo too:

```bash
uv sync --extra dev --extra voxtral
```

If you want client-side Voxtral EOU with Silero:

```bash
uv sync --extra dev --extra voxtral-eou
```

If you want Silero + Smart Turn:

```bash
uv sync --extra dev --extra voxtral-eou --extra voxtral-smart-turn
```

Set Deepgram if you use it:

```bash
export DEEPGRAM_API_KEY=...
```

## External Model Setup

### Voxtral

Prepare the optimized local model once:

```bash
uv run stt-exp prepare-voxtral-model \
  --out-dir models/voxtral-optimized \
  --streaming-n-left-pad-tokens 0 \
  --transcription-delay-ms 80
```

The live launcher will auto-start [scripts/serve_voxtral_optimized.sh](/home/logan/Projects/stt-exp/scripts/serve_voxtral_optimized.sh)
when Voxtral is selected and nothing is already listening at
`ws://127.0.0.1:8000/v1/realtime`.

Useful env vars:

- `VOXTRAL_URI`
- `VOXTRAL_AUTOSTART=0`
- `VOXTRAL_SERVE_SCRIPT`
- `GPU_MEMORY_UTILIZATION`
- `MAX_MODEL_LEN`

The default serve profile is intentionally conservative:

- `GPU_MEMORY_UTILIZATION=0.55`
- `MAX_MODEL_LEN=4096`

### Sherpa

Download the model once:

```bash
uv run python scripts/download_sherpa_model.py
```

Default expected location:

```text
models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-int8-2026-01-14
```

Override with `SHERPA_MODEL_DIR` if needed.

### Parakeet

Parakeet is intentionally isolated in a separate Python environment because its
NeMo stack is heavier and touchier than the main repo env.

Example:

```bash
uv venv .venv-parakeet --python 3.10
uv pip install --python .venv-parakeet/bin/python torch torchvision torchaudio
uv pip install --python .venv-parakeet/bin/python nemo_toolkit[asr] soundfile librosa
```

Then point the launcher at it:

```bash
export PARAKEET_PYTHON=.venv-parakeet/bin/python
```

Optional:

```bash
export PARAKEET_DEVICE=auto
```

Live tuning flags for Parakeet:

- `--parakeet-preset`
- `--parakeet-eou-silence-ms`
- `--parakeet-min-utterance-ms`
- `--parakeet-force-finalize-ms`
- `--parakeet-preroll-ms`
- `--parakeet-rms-threshold`

## Live Usage

Run the main four-provider setup:

```bash
./scripts/run_live_external_models.sh
```

Run Voxtral only:

```bash
./scripts/run_live_external_models.sh --providers voxtral
```

Run without Deepgram:

```bash
./scripts/run_live_external_models.sh --providers voxtral sherpa parakeet
```

Pick a mic:

```bash
uv run stt-exp devices
./scripts/run_live_external_models.sh --device 11
```

The launcher passes through `stt-exp live` flags, so any live CLI option can be
added directly.

Example with more aggressive Parakeet finalization for short utterances:

```bash
./scripts/run_live_external_models.sh \
  --providers deepgram parakeet \
  --parakeet-preset fast \
  --parakeet-eou-silence-ms 180 \
  --parakeet-min-utterance-ms 50 \
  --parakeet-force-finalize-ms 300 \
  --parakeet-preroll-ms 120
```

Live hotkeys when Parakeet is active:

- `r` clears the current utterance for all providers
- `p` cycles the Parakeet live preset

## Voxtral EOU

Supported modes:

- `none`
- `silero`
- `silero-smart-turn`

Run with explicit flags:

```bash
./scripts/run_live_external_models.sh \
  --providers voxtral \
  --voxtral-eou-mode silero \
  --voxtral-eou-min-utterance-ms 300 \
  --voxtral-eou-silero-min-silence-ms 500
```

More aggressive:

```bash
./scripts/run_live_external_models.sh \
  --providers voxtral \
  --voxtral-eou-mode silero \
  --voxtral-eou-min-utterance-ms 120 \
  --voxtral-eou-silero-min-silence-ms 180
```

Live hotkeys when Voxtral is active:

- `r` clears the current utterance for all providers
- `v` cycles Voxtral EOU mode
- `g` cycles Voxtral EOU aggressiveness preset

The live UI also prints a `[system] Voxtral EOU: ...` line so it is obvious when
those keypresses are actually being seen.

## Deterministic Debugging

When live behavior is ambiguous, use the benchmark path with a WAV instead of
testing by hand.

Example:

```bash
uv run stt-exp benchmark \
  --audio ../voxtral-realtime/test_wavs/001_hey_man_how_are_you.wav \
  --reference "hey man how are you" \
  --providers voxtral \
  --chunk-ms 40 \
  --pace realtime \
  --voxtral-eou-mode silero \
  --voxtral-eou-min-utterance-ms 300 \
  --voxtral-eou-silero-min-silence-ms 500 \
  --output results/voxtral-eou-debug.json
```

That output includes timing, transcript text, and provider event traces so you
can see whether EOU actually fired, or whether the final came from end-of-audio.

## Useful Commands

List audio devices:

```bash
uv run stt-exp devices
```

Create a manifest from a WAV directory:

```bash
uv run stt-exp make-manifest ../voxtral-realtime/test_wavs \
  --output data/test-wavs.csv
```

Run a file benchmark across the four main providers:

```bash
uv run stt-exp benchmark \
  --manifest data/test-wavs.csv \
  --providers deepgram voxtral sherpa parakeet \
  --chunk-ms 40 \
  --pace realtime \
  --repeats 1
```

## Notes

- the live launcher is the primary supported interface
- the benchmark commands remain useful for repeatable debugging and regression checks
- Parakeet live uses a persistent worker process in a separate env
- Voxtral live now tolerates server startup/offline situations better and can be run by itself for focused testing
