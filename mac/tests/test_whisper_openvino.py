"""OpenVINO GenAI Whisper vs faster-whisper on one real (macOS `say`) speech clip.

Skips cleanly when openvino_genai isn't installed or the Whisper IR hasn't been
exported (mac/intel/export_whisper.py). Needs macOS `say`/`afconvert` to
synthesize the clip unless a cached clip already exists in the scratchpad.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import wave
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("openvino_genai")

from intel import WHISPER_OV_DIR  # noqa: E402
from perception.transcriber import (  # noqa: E402
    FasterWhisperTranscriber, OpenVinoWhisperTranscriber, make_transcriber,
)

SCRATCH = Path(os.environ.get(
    "VOICELINK_SCRATCH",
    "/private/tmp/claude-501/-Users-tanayvinaykya-VoiceLink/"
    "f03806ca-81fa-49f4-aa5a-2ef28f0f6222/scratchpad"))
SENTENCE = "Do you want some water or a cup of coffee?"
KEY_WORDS = ["water", "cup", "coffee"]

pytestmark = pytest.mark.skipif(
    not (WHISPER_OV_DIR / "openvino_encoder_model.xml").exists(),
    reason="Whisper IR missing — run intel/export_whisper.py")


def speech_clip() -> np.ndarray:
    """16 kHz mono float32 of SENTENCE, synthesized once with macOS `say`."""
    wav = SCRATCH / "clip.wav"
    if not wav.exists():
        if not (shutil.which("say") and shutil.which("afconvert")):
            pytest.skip("no speech clip and no macOS `say` to synthesize one")
        SCRATCH.mkdir(parents=True, exist_ok=True)
        aiff = SCRATCH / "clip.aiff"
        subprocess.run(["say", "-o", str(aiff), SENTENCE], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                        str(aiff), str(wav)], check=True)
    with wave.open(str(wav)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    return pcm.astype(np.float32) / 32768.0


def _check(name, tr, pcm):
    tr.transcribe(pcm, 16000)  # warm-up (model load / first-call graph compile)
    t0 = time.perf_counter()
    res = tr.transcribe(pcm, 16000)
    dt = time.perf_counter() - t0
    print(f"\n[{name}] {dt*1000:.0f} ms  text={res.text!r}  "
          f"avg_logprob={res.avg_logprob:.2f} no_speech={res.no_speech_prob:.2f}")
    low = res.text.lower()
    for kw in KEY_WORDS:
        assert kw in low, f"{name}: {kw!r} missing from {res.text!r}"
    assert res.no_speech_prob < 0.5
    return dt


def test_both_backends_transcribe_clip():
    pytest.importorskip("faster_whisper")
    pcm = speech_clip()
    fw = FasterWhisperTranscriber("base.en", "cpu", "int8")
    ov = OpenVinoWhisperTranscriber(WHISPER_OV_DIR)
    t_fw = _check("faster-whisper", fw, pcm)
    t_ov = _check("openvino-genai", ov, pcm)
    print(f"[wall] faster-whisper {t_fw*1000:.0f} ms | openvino {t_ov*1000:.0f} ms")


def test_openvino_silence_reports_no_speech():
    ov = OpenVinoWhisperTranscriber(WHISPER_OV_DIR)
    res = ov.transcribe(np.zeros(16000, dtype=np.float32), 16000)
    # Whisper may hallucinate on pure silence; what we guarantee is the contract
    # shape and that empty text maps to no_speech_prob=1.0.
    assert isinstance(res.text, str)
    if not res.text:
        assert res.no_speech_prob == 1.0


def test_factory_selects_openvino():
    class Cfg:
        TRANSCRIBER = "openvino"
        WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE = "base.en", "cpu", "int8"
    assert isinstance(make_transcriber(Cfg), OpenVinoWhisperTranscriber)
