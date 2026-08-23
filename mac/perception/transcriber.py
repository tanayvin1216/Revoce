"""
Speech transcription, behind a clean swappable interface.

Default backend: faster-whisper (base.en). It accepts a float32 numpy PCM array
directly (no ffmpeg) and runs on CPU/int8 — a good fit for the Mac. Swap the
model size or the whole runtime (OpenVinoWhisperTranscriber for the Intel port)
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
        # errstate: silences benign divide/overflow warnings from faster-whisper's
        # mel step under numpy 2.x (output is unaffected).
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
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


class OpenVinoWhisperTranscriber(Transcriber):
    """Whisper base.en (INT8 IR) through openvino_genai.WhisperPipeline.

    Same audio contract as FasterWhisperTranscriber: 16 kHz mono float32 in
    [-1, 1]. The IR comes from mac/intel/export_whisper.py (optimum-cli). Device
    from VOICELINK_OV_DEVICE (CPU default, GPU/NPU on Intel boxes).

    CONFIDENCE PROXY (honest note): openvino_genai does not expose per-token
    logprobs or a no-speech probability. `result.scores` is the sequence score
    of the chosen beam — under greedy decoding it is a flat 1.0, so it carries
    no information. We therefore report avg_logprob=-0.3 (a "normal speech"
    value on the faster-whisper scale the downstream addressed/ambient gate
    expects) whenever text is produced, and no_speech_prob 0.05 / 1.0 for
    text / no-text. If a future GenAI release yields a real log-probability in
    `scores` (negative, not exactly 1.0) we use it as avg_logprob instead.
    Consumers must NOT treat this backend's confidence as calibrated.
    """

    # Whisper's mel front-end wants >= ~30 ms of audio; shorter blobs are padded.
    _MIN_SAMPLES = 480

    def __init__(self, model_dir, device: str | None = None) -> None:
        import os
        from pathlib import Path
        import openvino_genai  # lazy: optional dependency
        model_dir = Path(model_dir)
        if not (model_dir / "openvino_encoder_model.xml").exists():
            raise FileNotFoundError(
                f"Whisper IR missing under {model_dir} — run mac/intel/export_whisper.py")
        self.device = (device or os.environ.get("VOICELINK_OV_DEVICE", "CPU")).upper()
        # ARM pitfall (mac/intel/README.md): the CPU plugin defaults to f16 and the
        # decoder then degenerates into '!!!' and runs ~8x slower. f32 is correct
        # and fine on real Intel CPUs too. GPU/NPU plugins reject the property.
        props = {"INFERENCE_PRECISION_HINT": "f32"} if self.device == "CPU" else {}
        self._pipe = openvino_genai.WhisperPipeline(str(model_dir), self.device, **props)
        # base.en is English-only: GenAI rejects language/task on non-multilingual
        # models, so leave those unset. Timestamps off (we only want the text).
        self._cfg = self._pipe.get_generation_config()
        self._cfg.return_timestamps = False

    def transcribe(self, pcm_f32, sample_rate):
        if sample_rate != 16000:
            raise ValueError("OpenVinoWhisperTranscriber expects 16 kHz audio")
        pcm = np.asarray(pcm_f32, dtype=np.float32)
        if pcm.shape[0] < self._MIN_SAMPLES:
            pcm = np.pad(pcm, (0, self._MIN_SAMPLES - pcm.shape[0]))
        res = self._pipe.generate(pcm.tolist(), self._cfg)
        text = " ".join(t.strip() for t in res.texts).strip()
        if not text:
            return TranscriptionResult("", -10.0, 1.0)
        avg_logprob = -0.3
        scores = list(getattr(res, "scores", []) or [])
        if scores and scores[0] < 0.0:          # a real log-prob, not the greedy 1.0
            avg_logprob = float(scores[0])
        return TranscriptionResult(text, avg_logprob, 0.05)


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
    if name == "openvino":
        try:
            from intel import WHISPER_OV_DIR  # artifacts/whisper-base.en-ov-int8
            return OpenVinoWhisperTranscriber(WHISPER_OV_DIR)
        except Exception as e:  # openvino_genai or the IR missing -> degrade gracefully
            print(f"[perception] openvino whisper unavailable ({e}); falling back to faster-whisper")
            try:
                return FasterWhisperTranscriber(
                    cfg.WHISPER_MODEL, cfg.WHISPER_DEVICE, cfg.WHISPER_COMPUTE_TYPE
                )
            except Exception as e2:
                print(f"[perception] faster-whisper unavailable ({e2}); using MockTranscriber")
                return MockTranscriber()
    raise ValueError(f"unknown TRANSCRIBER={name!r}")
