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
SAMPLE_RATE = 16_000


@dataclass(slots=True)
class LiveParakeetConfig:
    model_id: str
    device: str
    att_context_size: tuple[int, int] | None
    live_mode: str
    preset: str
    eou_silence_ms: int
    min_utterance_ms: int
    force_finalize_ms: int
    preroll_ms: int
    rms_threshold: float
    tail_silence_chunks: int


def emit(message: dict[str, object]) -> None:
    print(f"{MARKER}{json.dumps(message)}", flush=True)


def _resolve_model_eou_token_ids(model_id: str, tokenizer_size: int) -> tuple[int, int] | None:
    # The older realtime EOU checkpoints append two explicit EOU/EOB tokens
    # immediately before the RNNT blank token. Other checkpoints, including the
    # multitalker streaming model, do not use that layout.
    if "realtime_eou" not in model_id.lower():
        return None
    if tokenizer_size < 2:
        return None
    return (tokenizer_size - 2, tokenizer_size - 1)


def _format_status(config: LiveParakeetConfig, *, step_ms: int, device_type: str) -> str:
    att_context = "default" if config.att_context_size is None else list(config.att_context_size)
    return (
        f"connected model={config.model_id} step_ms={step_ms} device={device_type} "
        f"att_context={att_context} "
        f"live_mode={config.live_mode} "
        f"preset={config.preset} eou_silence_ms={config.eou_silence_ms} "
        f"min_utterance_ms={config.min_utterance_ms} force_finalize_ms={config.force_finalize_ms} "
        f"preroll_ms={config.preroll_ms} rms_threshold={config.rms_threshold} "
        f"tail_silence_chunks={config.tail_silence_chunks}"
    )


def _set_att_context_size(model, att_context_size: tuple[int, int] | None) -> None:
    if att_context_size is None:
        return
    if not hasattr(model.encoder, "set_default_att_context_size"):
        raise RuntimeError("model does not support --att-context-size")
    model.encoder.set_default_att_context_size(att_context_size=list(att_context_size))


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
        _set_att_context_size(self.model, config.att_context_size)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model_eou_token_ids = _resolve_model_eou_token_ids(
            config.model_id,
            len(self.model.tokenizer.tokenizer),
        )

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
        self.token_confidences: list[float] = []
        self.token_margins: list[float] = []
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

    def _confidence_summary(self) -> dict[str, float | str] | None:
        if not self.token_confidences:
            return None
        avg_conf = float(sum(self.token_confidences) / len(self.token_confidences))
        min_conf = float(min(self.token_confidences))
        last_conf = float(self.token_confidences[-1])
        avg_margin = float(sum(self.token_margins) / len(self.token_margins)) if self.token_margins else 0.0
        if avg_conf >= 0.85 and min_conf >= 0.55:
            label = "high"
        elif avg_conf >= 0.65 and min_conf >= 0.35:
            label = "medium"
        else:
            label = "low"
        return {
            "label": label,
            "avg": round(avg_conf, 3),
            "min": round(min_conf, 3),
            "last": round(last_conf, 3),
            "avg_margin": round(avg_margin, 3),
            "tokens": len(self.token_confidences),
        }

    def _utterance_duration_ms(self) -> float:
        return (self.audio_buffer.shape[0] / SAMPLE_RATE) * 1000.0

    def _store_preroll(self, chunk: np.ndarray) -> None:
        if self.config.preroll_ms <= 0:
            return
        max_samples = int(SAMPLE_RATE * self.config.preroll_ms / 1000)
        self.preroll_buffer = np.concatenate([self.preroll_buffer, chunk])[-max_samples:]

    def _append_tail_silence(self, silence_chunks: int) -> None:
        if silence_chunks <= 0:
            return
        samples_per_chunk = self.step_size * (SAMPLE_RATE // 100)
        silence = np.zeros((samples_per_chunk * silence_chunks,), dtype=np.float32)
        self.audio_buffer = np.concatenate([self.audio_buffer, silence])

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
                probs = torch.softmax(logp, dim=0)
                token_id = logp.argmax().item()
                if token_id == self.greedy._blank_index:
                    not_blank = False
                else:
                    top_probs = torch.topk(probs, k=min(2, probs.shape[0]))
                    top1_prob = float(probs[token_id].item())
                    top2_prob = float(top_probs.values[1].item()) if top_probs.values.shape[0] > 1 else 0.0
                    self.all_tokens.append(token_id)
                    self.token_confidences.append(top1_prob)
                    self.token_margins.append(max(0.0, top1_prob - top2_prob))
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
                    confidence = self._confidence_summary()
                    emit(
                        {
                            "type": "partial",
                            "text": current_text,
                            "audio_pos_s": self.last_audio_pos_s,
                            "confidence": confidence,
                        }
                    )
                    self.prev_text = current_text

                if self.model_eou_token_ids and any(token_id in self.model_eou_token_ids for token_id in new_tokens):
                    self.last_audio_pos_s = round(actual_end * 0.01, 3)
                    self._emit_final_and_reset(current_text, reason="model_eou")
                    return True

                if actual_end == mel.shape[2]:
                    return False

    def _emit_final_and_reset(self, text: str, *, reason: str) -> None:
        final_text = text.strip()
        if final_text:
            confidence = self._confidence_summary()
            emit(
                {
                    "type": "final",
                    "text": final_text,
                    "audio_pos_s": self.last_audio_pos_s or round(self.audio_buffer.shape[0] / SAMPLE_RATE, 3),
                    "reason": reason,
                    "confidence": confidence,
                }
            )
        self._reset_stream_state()

    def _finalize_from_fallback(self, *, reason: str) -> None:
        original_audio_samples = self.audio_buffer.shape[0]
        self._append_tail_silence(self.config.tail_silence_chunks)
        self._decode_buffer(allow_partial_tail=True)
        current_text = self._current_text()
        original_utterance_ms = (original_audio_samples / SAMPLE_RATE) * 1000.0
        if current_text and original_utterance_ms >= self.config.min_utterance_ms:
            self.last_audio_pos_s = round(original_audio_samples / SAMPLE_RATE, 3)
            self._emit_final_and_reset(current_text, reason=reason)
            return
        if self.utterance_started_at is not None and original_utterance_ms >= self.config.min_utterance_ms:
            emit(
                {
                    "type": "no_transcript",
                    "reason": reason,
                    "audio_pos_s": round(original_audio_samples / SAMPLE_RATE, 3),
                    "utterance_ms": round(original_utterance_ms, 1),
                }
            )
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

        if self.config.live_mode == "legacy":
            self.audio_buffer = np.concatenate([self.audio_buffer, chunk])
            self.last_audio_pos_s = round(self.audio_buffer.shape[0] / SAMPLE_RATE, 3)
            self._decode_buffer(allow_partial_tail=False)
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
        if self.config.live_mode == "legacy":
            final_text = self._current_text()
            if final_text:
                emit({"type": "final", "text": final_text, "reason": "stop"})
            self._reset_stream_state()
            return
        if self.utterance_started_at is None:
            self._reset_stream_state()
            return
        self._finalize_from_fallback(reason="stop")

    def reset(self) -> None:
        self._reset_stream_state()

    def update_config(self, config: LiveParakeetConfig) -> None:
        self.config = config
        self._reset_stream_state()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="nvidia/parakeet_realtime_eou_120m-v1")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--att-context-size", type=int, nargs=2, metavar=("LEFT", "RIGHT"), default=None)
    parser.add_argument("--live-mode", choices=["legacy", "tuned"], default="tuned")
    parser.add_argument("--preset", default="accurate")
    parser.add_argument("--eou-silence-ms", type=int, default=240)
    parser.add_argument("--min-utterance-ms", type=int, default=60)
    parser.add_argument("--force-finalize-ms", type=int, default=400)
    parser.add_argument("--preroll-ms", type=int, default=160)
    parser.add_argument("--rms-threshold", type=float, default=0.008)
    parser.add_argument("--tail-silence-chunks", type=int, default=2)
    args = parser.parse_args()

    try:
        streamer = ParakeetLiveStreamer(
            LiveParakeetConfig(
                model_id=args.model_id,
                device=args.device,
                att_context_size=tuple(args.att_context_size) if args.att_context_size is not None else None,
                live_mode=args.live_mode,
                preset=args.preset,
                eou_silence_ms=args.eou_silence_ms,
                min_utterance_ms=args.min_utterance_ms,
                force_finalize_ms=args.force_finalize_ms,
                preroll_ms=args.preroll_ms,
                rms_threshold=args.rms_threshold,
                tail_silence_chunks=args.tail_silence_chunks,
            )
        )
    except Exception as exc:
        emit({"type": "error", "message": f"failed to initialize model: {exc}"})
        raise

    emit(
        {
            "type": "status",
            "message": _format_status(streamer.config, step_ms=streamer.step_ms, device_type=streamer.device.type),
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
        elif msg_type == "set_config":
            try:
                next_config = LiveParakeetConfig(
                    model_id=streamer.config.model_id,
                    device=streamer.config.device,
                    att_context_size=streamer.config.att_context_size,
                    live_mode=str(message.get("live_mode", streamer.config.live_mode)),
                    preset=str(message.get("preset", streamer.config.preset)),
                    eou_silence_ms=int(message.get("eou_silence_ms", streamer.config.eou_silence_ms)),
                    min_utterance_ms=int(message.get("min_utterance_ms", streamer.config.min_utterance_ms)),
                    force_finalize_ms=int(message.get("force_finalize_ms", streamer.config.force_finalize_ms)),
                    preroll_ms=int(message.get("preroll_ms", streamer.config.preroll_ms)),
                    rms_threshold=float(message.get("rms_threshold", streamer.config.rms_threshold)),
                    tail_silence_chunks=int(message.get("tail_silence_chunks", streamer.config.tail_silence_chunks)),
                )
            except (TypeError, ValueError) as exc:
                emit({"type": "error", "message": f"invalid config update: {exc}"})
                break
            streamer.update_config(next_config)
            emit(
                {
                    "type": "status",
                    "message": _format_status(streamer.config, step_ms=streamer.step_ms, device_type=streamer.device.type),
                }
            )
        elif msg_type == "stop":
            streamer.flush()
            break
        else:
            emit({"type": "error", "message": f"unsupported message type: {msg_type}"})
            break


if __name__ == "__main__":
    main()
