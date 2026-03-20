from pathlib import Path

from stt_exp.manifest import infer_reference_from_filename


def test_infer_reference_from_filename() -> None:
    assert infer_reference_from_filename("001_hey_man_how_are_you.wav") == "hey man how are you"
    assert infer_reference_from_filename(Path("wait-hold-on.flac")) == "wait hold on"
