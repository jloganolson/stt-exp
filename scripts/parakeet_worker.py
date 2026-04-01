from __future__ import annotations

import argparse
import json
import time

import soundfile as sf
import torch
import torch.utils.data


def _patched_sampler_init(self, *args, **kwargs):
    pass


torch.utils.data.Sampler.__init__ = _patched_sampler_init

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.submodules.rnnt_greedy_decoding import label_collate


def _resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but CUDA is not available")
    return torch.device(requested_device)


def _set_att_context_size(model, att_context_size: tuple[int, int] | None) -> None:
    if att_context_size is None:
        return
    if not hasattr(model.encoder, "set_default_att_context_size"):
        raise RuntimeError("model does not support --att-context-size")
    model.encoder.set_default_att_context_size(att_context_size=list(att_context_size))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model-id", default="nvidia/parakeet_realtime_eou_120m-v1")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--pace", choices=["realtime", "burst"], default="realtime")
    parser.add_argument("--silence-chunks", type=int, default=6)
    parser.add_argument("--att-context-size", type=int, nargs=2, metavar=("LEFT", "RIGHT"), default=None)
    args = parser.parse_args()

    device = _resolve_device(args.device)
    map_location = "cpu" if device.type == "cpu" else None
    t_load = time.perf_counter()
    model = nemo_asr.models.ASRModel.from_pretrained(args.model_id, map_location=map_location)
    _set_att_context_size(model, tuple(args.att_context_size) if args.att_context_size is not None else None)
    model = model.to(device)
    model.eval()
    load_time_s = time.perf_counter() - t_load

    greedy = model.decoding.decoding
    scfg = model.encoder.streaming_cfg
    step_size = scfg.shift_size[-1] if isinstance(scfg.shift_size, list) else scfg.shift_size
    pre_enc_cache_size = (
        scfg.pre_encode_cache_size[-1]
        if isinstance(scfg.pre_encode_cache_size, list)
        else scfg.pre_encode_cache_size
    )
    step_ms = int(step_size * 10)

    audio_data, sr = sf.read(args.audio, dtype="float32")
    if sr != 16000:
        import librosa

        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
        sr = 16000
    if getattr(audio_data, "ndim", 1) > 1:
        audio_data = audio_data.mean(axis=1)
    waveform = torch.from_numpy(audio_data).unsqueeze(0)
    audio_duration_s = waveform.shape[1] / sr

    waveform_gpu = waveform.to(device)
    length = torch.tensor([waveform.shape[1]], dtype=torch.long, device=device)
    with torch.no_grad():
        mel, _mel_len = model.preprocessor(input_signal=waveform_gpu, length=length)
    total_mel_frames = mel.shape[2]

    with torch.no_grad():
        dummy = torch.zeros((1, mel.shape[1], step_size), device=device)
        dummy_len = torch.tensor([step_size], dtype=torch.long, device=device)
        wc, wt, wcl = model.encoder.get_initial_cache_state(batch_size=1, device=device)
        model.encoder(
            audio_signal=dummy,
            length=dummy_len,
            cache_last_channel=wc,
            cache_last_time=wt,
            cache_last_channel_len=wcl,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()

    cache_ch, cache_t, cache_ch_len = model.encoder.get_initial_cache_state(batch_size=1, device=device)
    last_token = None
    dec_state = None
    all_tokens: list[int] = []
    events: list[dict[str, object]] = []
    prev_text = ""

    mel_pos = 0
    chunk_idx = 0
    total_chunks = -(-total_mel_frames // step_size) + args.silence_chunks
    real_audio_chunks = -(-total_mel_frames // step_size)

    t_start = time.perf_counter()
    first_audio_sent_at_s = 0.0
    last_audio_sent_at_s = 0.0

    with torch.no_grad():
        while chunk_idx < total_chunks:
            is_first = chunk_idx == 0
            if args.pace == "realtime" and chunk_idx > 0:
                target_time = chunk_idx * (step_ms / 1000.0)
                elapsed = time.perf_counter() - t_start
                sleep_time = target_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            send_time_s = time.perf_counter() - t_start
            if chunk_idx == 0:
                first_audio_sent_at_s = send_time_s
            if chunk_idx < real_audio_chunks:
                last_audio_sent_at_s = send_time_s

            if mel_pos < total_mel_frames:
                end_pos = min(mel_pos + step_size, total_mel_frames)
                new_frames = mel[:, :, mel_pos:end_pos]
                actual_new = new_frames.shape[2]
                if actual_new < step_size:
                    pad = torch.zeros((1, mel.shape[1], step_size - actual_new), device=device)
                    new_frames = torch.cat([new_frames, pad], dim=2)
            else:
                end_pos = mel_pos + step_size
                new_frames = torch.zeros((1, mel.shape[1], step_size), device=device)

            if is_first:
                chunk_input = new_frames
            else:
                chunk_input = torch.cat([pre_encode_cache, new_frames], dim=2)

            pre_encode_cache = new_frames[:, :, -pre_enc_cache_size:]
            chunk_len = torch.tensor([chunk_input.shape[2]], dtype=torch.long, device=device)

            enc_out, _enc_len, cache_ch, cache_t, cache_ch_len = model.encoder(
                audio_signal=chunk_input,
                length=chunk_len,
                cache_last_channel=cache_ch,
                cache_last_time=cache_t,
                cache_last_channel_len=cache_ch_len,
            )

            enc_frames = enc_out.permute(2, 0, 1)
            for t in range(enc_frames.shape[0]):
                f = enc_frames[t : t + 1]
                not_blank = True
                symbols_added = 0
                while not_blank and symbols_added < 10:
                    if last_token is None and dec_state is None:
                        last_label = greedy._SOS
                    else:
                        last_label = label_collate([[last_token]])
                    g, hidden_prime = greedy._pred_step(last_label, dec_state)
                    logp = greedy._joint_step(f, g, log_normalize=None)[0, 0, 0, :]
                    if logp.dtype != torch.float32:
                        logp = logp.float()
                    k = logp.argmax().item()
                    if k == greedy._blank_index:
                        not_blank = False
                    else:
                        all_tokens.append(k)
                        dec_state = hidden_prime
                        last_token = k
                        symbols_added += 1

            current_text = model.tokenizer.ids_to_text(all_tokens) if all_tokens else ""
            if current_text != prev_text:
                wall_time_s = time.perf_counter() - t_start
                events.append(
                    {
                        "audio_pos_s": min(end_pos, total_mel_frames) * 0.01,
                        "wall_time_s": wall_time_s,
                        "text": current_text,
                    }
                )
                prev_text = current_text

            mel_pos = end_pos
            chunk_idx += 1

    final_at_s = time.perf_counter() - t_start
    final_text = model.tokenizer.ids_to_text(all_tokens) if all_tokens else ""
    first_text_at_s = events[0]["wall_time_s"] if events else None

    payload = {
        "provider": "parakeet",
        "transcript_text": final_text,
        "audio_duration_s": audio_duration_s,
        "session_started_at_s": 0.0,
        "first_audio_sent_at_s": first_audio_sent_at_s,
        "last_audio_sent_at_s": last_audio_sent_at_s,
        "first_text_at_s": first_text_at_s,
        "final_at_s": final_at_s,
        "events": events,
        "meta": {
            "model_id": args.model_id,
            "device": device.type,
            "att_context_size": list(args.att_context_size) if args.att_context_size is not None else None,
            "load_time_s": load_time_s,
            "native_step_ms": step_ms,
            "pace": args.pace,
        },
    }
    print(f"PARAKEET_RESULT_JSON={json.dumps(payload)}")


if __name__ == "__main__":
    main()
