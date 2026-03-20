from __future__ import annotations

import argparse
import base64
import json
import sys

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


def emit(message: dict[str, object]) -> None:
    print(f"{MARKER}{json.dumps(message)}", flush=True)


class ParakeetLiveStreamer:
    def __init__(self, model_id: str, device_name: str):
        if device_name == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device = device_name
        if resolved_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("cuda requested but CUDA is not available")
        self.device = torch.device(resolved_device)
        map_location = "cpu" if self.device.type == "cpu" else None
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_id, map_location=map_location)
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
        self.processed_mel_frames = 0
        self.pre_encode_cache = None

    def _current_text(self) -> str:
        if not self.all_tokens:
            return ""
        return self.model.tokenizer.ids_to_text(self.all_tokens).strip()

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

    def feed_audio(self, pcm16_chunk: bytes) -> None:
        chunk = np.frombuffer(pcm16_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        if chunk.size == 0:
            return
        self.audio_buffer = np.concatenate([self.audio_buffer, chunk])

        with torch.no_grad():
            waveform = torch.from_numpy(self.audio_buffer).unsqueeze(0).to(self.device)
            length = torch.tensor([self.audio_buffer.shape[0]], dtype=torch.long, device=self.device)
            mel, _ = self.model.preprocessor(input_signal=waveform, length=length)

            while self.processed_mel_frames + self.step_size <= mel.shape[2]:
                start = self.processed_mel_frames
                end = start + self.step_size
                new_frames = mel[:, :, start:end]
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
                self.processed_mel_frames = end

                current_text = self._current_text()
                if current_text and current_text != self.prev_text:
                    emit(
                        {
                            "type": "partial",
                            "text": current_text,
                            "audio_pos_s": round(end * 0.01, 3),
                        }
                    )
                    self.prev_text = current_text

                if any(token_id in (EOU_TOKEN_ID, EOB_TOKEN_ID) for token_id in new_tokens):
                    final_text = current_text
                    if final_text:
                        emit(
                            {
                                "type": "final",
                                "text": final_text,
                                "audio_pos_s": round(end * 0.01, 3),
                            }
                        )
                    self._reset_stream_state()
                    break

    def flush(self) -> None:
        final_text = self._current_text()
        if final_text:
            emit({"type": "final", "text": final_text})
        self._reset_stream_state()

    def reset(self) -> None:
        self._reset_stream_state()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="nvidia/parakeet_realtime_eou_120m-v1")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    try:
        streamer = ParakeetLiveStreamer(args.model_id, args.device)
    except Exception as exc:
        emit({"type": "error", "message": f"failed to initialize model: {exc}"})
        raise

    emit(
        {
            "type": "status",
            "message": f"connected model={args.model_id} step_ms={streamer.step_ms} device={streamer.device.type}",
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
