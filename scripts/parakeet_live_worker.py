from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.utils.data


def _patched_sampler_init(self, *args, **kwargs):
    pass


torch.utils.data.Sampler.__init__ = _patched_sampler_init

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.submodules.rnnt_greedy_decoding import label_collate

MARKER = "PARAKEET_LIVE_EVENT_JSON="
EOU_TOKEN_ID = 1024
EOB_TOKEN_ID = 1025
SAMPLE_RATE = 16_000


@dataclass(slots=True)
class LiveParakeetConfig:
    model_id: str
    device: str
    eou_silence_ms: int
    min_utterance_ms: int
    force_finalize_ms: int
    preroll_ms: int
    rms_threshold: float


def emit(message: dict[str, object]) -> None:
    print(f"{MARKER}{json.dumps(message)}", flush=True)


class ParakeetLiveStreamer:
    def __init__(self, config: LiveParakeetConfig):
        self.config = config
        if config.device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device = config.device
        if resolved_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("cuda requested but CUDA is not available")
        self.device = torch.device(resolved_device)
        map_location = "cpu" if self.device.type == "cpu" else None
        self.model = nemo_asr.models.ASRModel.from_pretrained(config.model_id, map_location=map_location)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.greedy = self.model.decoding.decoding
        scfg = self.model.encoder.streaming_cfg
        self.step_size = scfg.shift_size[-1] if isinstance(scfg.shift_size, list) else scfg.shift_size
        self.pre_enc_cache_size = (
            scfg.pre_encode_cache_size[-1]
            if isinstance(scfg.pre_encode_cache_size, list)
            else scfg.pre_encode_cache_size
        )
        self.step_ms = int(self.step_size * 10)
        self._warmup()
        self._reset_stream_state()

    def _warmup(self) -> None:
        with torch.no_grad():
            dummy = torch.zeros((1, 128, self.step_size), device=self.device)
            dummy_len = torch.tensor([self.step_size], dtype=torch.long, device=self.device)
            wc, wt, wcl = self.model.encoder.get_initial_cache_state(batch_size=1, device=self.device)
            self.model.encoder(
                audio_signal=dummy,
                length=dummy_len,
                cache_last_channel=wc,
                cache_last_time=wt,
                cache_last_channel_len=wcl,
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _reset_stream_state(self) -> None:
        self.cache_ch, self.cache_t, self.cache_ch_len = self.model.encoder.get_initial_cache_state(
            batch_size=1,
            device=self.device,
        )
        self.last_token = None
        self.dec_state = None
        self.all_tokens: list[int] = []
        self.prev_text = ""
        self.audio_buffer = np.zeros((0,), dtype=np.float32)
        self.preroll_buffer = np.zeros((0,), dtype=np.float32)
        self.processed_mel_frames = 0
        self.pre_encode_cache = None
        self.utterance_started_at = None
        self.last_audio_at = None
        self.last_speech_at = None
        self.last_text_change_at = None
        self.last_audio_pos_s = 0.0

    def _current_text(self) -> str:
        if not self.all_tokens:
            return ""
        return self.model.tokenizer.ids_to_text(self.all_tokens).strip()

    def _utterance_duration_ms(self) -> float:
        return (self.audio_buffer.shape[0] / SAMPLE_RATE) * 1000.0

    def _store_preroll(self, chunk: np.ndarray) -> None:
        if self.config.preroll_ms <= 0:
            return
        max_samples = int(SAMPLE_RATE * self.config.preroll_ms / 1000)
        self.preroll_buffer = np.concatenate([self.preroll_buffer, chunk])[-max_samples:]

    def _run_decoder_step(self, enc_frames: torch.Tensor) -> list[int]:
        new_tokens: list[int] = []
        for idx in range(enc_frames.shape[0]):
            frame = enc_frames[idx : idx + 1]
            not_blank = True
            symbols_added = 0
            while not_blank and symbols_added < 10:
                if self.last_token is None and self.dec_state is None:
                    last_label = self.greedy._SOS
                else:
                    last_label = label_collate([[self.last_token]])
                g, hidden_prime = self.greedy._pred_step(last_label, self.dec_state)
                logp = self.greedy._joint_step(frame, g, log_normalize=None)[0, 0, 0, :]
                if logp.dtype != torch.float32:
                    logp = logp.float()
                token_id = logp.argmax().item()
                if token_id == self.greedy._blank_index:
                    not_blank = False
                else:
                    self.all_tokens.append(token_id)
                    new_tokens.append(token_id)
                    self.dec_state = hidden_prime
                    self.last_token = token_id
                    symbols_added += 1
        return new_tokens

    def _decode_buffer(self, *, allow_partial_tail: bool) -> bool:
        with torch.no_grad():
            waveform = torch.from_numpy(self.audio_buffer).unsqueeze(0).to(self.device)
            length = torch.tensor([self.audio_buffer.shape[0]], dtype=torch.long, device=self.device)
            mel, _ = self.model.preprocessor(input_signal=waveform, length=length)

            while True:
                start = self.processed_mel_frames
                remaining = mel.shape[2] - start
                if remaining <= 0:
                    return False
                if remaining < self.step_size and not allow_partial_tail:
                    return False

                actual_end = min(start + self.step_size, mel.shape[2])
                new_frames = mel[:, :, start:actual_end]
                if new_frames.shape[2] < self.step_size:
                    pad = torch.zeros(
                        (1, mel.shape[1], self.step_size - new_frames.shape[2]),
                        device=self.device,
                    )
                    new_frames = torch.cat([new_frames, pad], dim=2)
                if self.processed_mel_frames == 0:
                    chunk_input = new_frames
                else:
                    chunk_input = torch.cat([self.pre_encode_cache, new_frames], dim=2)
                self.pre_encode_cache = new_frames[:, :, -self.pre_enc_cache_size :]
                chunk_len = torch.tensor([chunk_input.shape[2]], dtype=torch.long, device=self.device)

                enc_out, _enc_len, self.cache_ch, self.cache_t, self.cache_ch_len = self.model.encoder(
                    audio_signal=chunk_input,
                    length=chunk_len,
                    cache_last_channel=self.cache_ch,
                    cache_last_time=self.cache_t,
                    cache_last_channel_len=self.cache_ch_len,
                )
                new_tokens = self._run_decoder_step(enc_out.permute(2, 0, 1))
                self.processed_mel_frames = actual_end

                current_text = self._current_text()
                if current_text and current_text != self.prev_text:
                    self.last_text_change_at = time.monotonic()
                    self.last_audio_pos_s = round(actual_end * 0.01, 3)
                    emit(
                        {
                            "type": "partial",
                            "text": current_text,
                            "audio_pos_s": self.last_audio_pos_s,
                        }
                    )
                    self.prev_text = current_text

                if any(token_id in (EOU_TOKEN_ID, EOB_TOKEN_ID) for token_id in new_tokens):
                    self.last_audio_pos_s = round(actual_end * 0.01, 3)
                    self._emit_final_and_reset(current_text, reason="model_eou")
                    return True

                if actual_end == mel.shape[2]:
                    return False

    def _emit_final_and_reset(self, text: str, *, reason: str) -> None:
        final_text = text.strip()
        if final_text:
            emit(
                {
                    "type": "final",
                    "text": final_text,
                    "audio_pos_s": self.last_audio_pos_s or round(self.audio_buffer.shape[0] / SAMPLE_RATE, 3),
                    "reason": reason,
                }
            )
        self._reset_stream_state()

    def _finalize_from_fallback(self, *, reason: str) -> None:
        self._decode_buffer(allow_partial_tail=True)
        current_text = self._current_text()
        if current_text and self._utterance_duration_ms() >= self.config.min_utterance_ms:
            self._emit_final_and_reset(current_text, reason=reason)
            return
        self._reset_stream_state()

    def _maybe_finalize_on_fallback(self) -> None:
        if self.utterance_started_at is None:
            return

        now = time.monotonic()
        silence_ms = None
        if self.last_speech_at is not None:
            silence_ms = (now - self.last_speech_at) * 1000.0
        stalled_ms = None
        if self.last_text_change_at is not None:
            stalled_ms = (now - self.last_text_change_at) * 1000.0

        enough_audio = self._utterance_duration_ms() >= self.config.min_utterance_ms
        if enough_audio and silence_ms is not None and silence_ms >= self.config.eou_silence_ms:
            self._finalize_from_fallback(reason="silence")
            return
        if (
            enough_audio
            and stalled_ms is not None
            and stalled_ms >= self.config.force_finalize_ms
            and silence_ms is not None
            and silence_ms >= min(100, self.config.eou_silence_ms)
        ):
            self._finalize_from_fallback(reason="stall")
            return
        if silence_ms is not None and silence_ms >= max(self.config.eou_silence_ms, self.config.force_finalize_ms):
            self._finalize_from_fallback(reason="empty_silence")

    def feed_audio(self, pcm16_chunk: bytes) -> None:
        chunk = np.frombuffer(pcm16_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        if chunk.size == 0:
            return

        now = time.monotonic()
        rms = float(np.sqrt(np.mean(np.square(chunk), dtype=np.float32)))
        is_speech = rms >= self.config.rms_threshold

        if self.utterance_started_at is None and not is_speech:
            self._store_preroll(chunk)
            return

        if self.utterance_started_at is None:
            if self.preroll_buffer.size:
                self.audio_buffer = np.concatenate([self.preroll_buffer, chunk])
                self.preroll_buffer = np.zeros((0,), dtype=np.float32)
            else:
                self.audio_buffer = chunk.copy()
            self.utterance_started_at = now
        else:
            self.audio_buffer = np.concatenate([self.audio_buffer, chunk])

        self.last_audio_at = now
        self.last_audio_pos_s = round(self.audio_buffer.shape[0] / SAMPLE_RATE, 3)
        if is_speech:
            self.last_speech_at = now

        did_finalize = self._decode_buffer(allow_partial_tail=False)
        if did_finalize:
            return
        self._maybe_finalize_on_fallback()

    def flush(self) -> None:
        if self.utterance_started_at is None:
            self._reset_stream_state()
            return
        self._finalize_from_fallback(reason="stop")

    def reset(self) -> None:
        self._reset_stream_state()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="nvidia/parakeet_realtime_eou_120m-v1")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--eou-silence-ms", type=int, default=240)
    parser.add_argument("--min-utterance-ms", type=int, default=60)
    parser.add_argument("--force-finalize-ms", type=int, default=400)
    parser.add_argument("--preroll-ms", type=int, default=160)
    parser.add_argument("--rms-threshold", type=float, default=0.008)
    args = parser.parse_args()

    try:
        streamer = ParakeetLiveStreamer(
            LiveParakeetConfig(
                model_id=args.model_id,
                device=args.device,
                eou_silence_ms=args.eou_silence_ms,
                min_utterance_ms=args.min_utterance_ms,
                force_finalize_ms=args.force_finalize_ms,
                preroll_ms=args.preroll_ms,
                rms_threshold=args.rms_threshold,
            )
        )
    except Exception as exc:
        emit({"type": "error", "message": f"failed to initialize model: {exc}"})
        raise

    emit(
        {
            "type": "status",
            "message": (
                f"connected model={args.model_id} step_ms={streamer.step_ms} device={streamer.device.type} "
                f"eou_silence_ms={args.eou_silence_ms} min_utterance_ms={args.min_utterance_ms} "
                f"force_finalize_ms={args.force_finalize_ms} preroll_ms={args.preroll_ms} "
                f"rms_threshold={args.rms_threshold}"
            ),
        }
    )

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            emit({"type": "error", "message": f"invalid control message: {exc}"})
            break
        msg_type = message.get("type")
        if msg_type == "audio":
            try:
                pcm = base64.b64decode(message["audio"])
            except Exception as exc:
                emit({"type": "error", "message": f"invalid audio payload: {exc}"})
                break
            streamer.feed_audio(pcm)
        elif msg_type == "reset":
            streamer.reset()
            emit({"type": "status", "message": "manual reset"})
        elif msg_type == "stop":
            streamer.flush()
            break
        else:
            emit({"type": "error", "message": f"unsupported message type: {msg_type}"})
            break


if __name__ == "__main__":
    main()
