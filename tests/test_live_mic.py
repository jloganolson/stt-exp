from stt_exp.live_mic import _append_voxtral_delta, _format_voxtral_connection_error


def test_format_voxtral_connection_error_local_uri() -> None:
    message = _format_voxtral_connection_error(
        "ws://127.0.0.1:8000/v1/realtime",
        ConnectionRefusedError(111, "Connect call failed"),
    )

    assert "127.0.0.1:8000" in message
    assert "./scripts/serve_voxtral_optimized.sh" in message


def test_format_voxtral_connection_error_remote_uri() -> None:
    message = _format_voxtral_connection_error(
        "wss://voxtral.example.com/v1/realtime",
        OSError("timed out"),
    )

    assert "voxtral.example.com:443" in message
    assert "--voxtral-uri" in message


def test_append_voxtral_delta_preserves_leading_space() -> None:
    current_text = _append_voxtral_delta("what's", " up")

    assert current_text == "what's up"
