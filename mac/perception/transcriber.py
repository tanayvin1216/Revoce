"""
Speech transcription, behind a clean swappable interface.

Default backend: faster-whisper (base.en). It accepts a float32 numpy PCM array
directly (no ffmpeg) and runs on CPU/int8 — a good fit for the Mac. Swap the
model size or the whole runtime (e.g. an OpenVINO Whisper for the Intel port)
with a ONE-LINE change in config.py; nothing else in the pipeline moves.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class TranscriptionResult:
    text: str
    avg_logprob: float    # higher = more confident (~ -0.2 good, < -1.2 poor)
    no_speech_prob: float  # higher = more likely silence/noise


class Transcriber(ABC):
    @abstractmethod
    def transcribe(self, pcm_f32: np.ndarray, sample_rate: int) -> TranscriptionResult:
        ...


class FasterWhisperTranscriber(Transcriber):
    def __init__(self, model_size: str, device: str, compute_type: str) -> None:
        from faster_whisper import WhisperModel  # lazy: heavy import / model load
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, pcm_f32, sample_rate):
        # We feed 16 kHz mono float32 in [-1, 1]; that's what the wearable streams.
        segments, _info = self._model.transcribe(
            pcm_f32, language="en", beam_size=1, vad_filter=False,
        )
        segs = list(segments)
        text = " ".join(s.text.strip() for s in segs).strip()
        if segs:
            avg_logprob = sum(s.avg_logprob for s in segs) / len(segs)
            no_speech = sum(s.no_speech_prob for s in segs) / len(segs)
        else:
            avg_logprob, no_speech = -10.0, 1.0
        return TranscriptionResult(text, avg_logprob, no_speech)


class MockTranscriber(Transcriber):
    """Scripted transcriber for dev without models. Cycles through a script that
    includes questions, so the 'addressed' flag visibly flips in the logs."""

    _SCRIPT = [
        ("The weather is nice today.", -0.20, 0.04),
        ("Do you want some water?", -0.15, 0.03),         # question -> addressed
        ("Okay.", -0.30, 0.10),
        ("Are you hungry?", -0.15, 0.03),                 # question -> addressed
        ("just mumbling in the background", -1.40, 0.25),  # low conf -> ambient
    ]

    def __init__(self) -> None:
        self._i = 0

    def transcribe(self, pcm_f32, sample_rate):
        text, lp, ns = self._SCRIPT[self._i % len(self._SCRIPT)]
        self._i += 1
        return TranscriptionResult(text, lp, ns)


def make_transcriber(cfg) -> Transcriber:
    """Factory — the one place the transcription runtime is selected."""
    name = cfg.TRANSCRIBER
    if name == "faster-whisper":
        try:
            return FasterWhisperTranscriber(
                cfg.WHISPER_MODEL, cfg.WHISPER_DEVICE, cfg.WHISPER_COMPUTE_TYPE
            )
        except Exception as e:  # lib/model missing -> degrade gracefully
            print(f"[perception] faster-whisper unavailable ({e}); using MockTranscriber")
            return MockTranscriber()
    if name == "mock":
        return MockTranscriber()
    # if name == "openvino": return OpenVinoWhisperTranscriber(cfg)   # Intel port, later
    raise ValueError(f"unknown TRANSCRIBER={name!r}")
