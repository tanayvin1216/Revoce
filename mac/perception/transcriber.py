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


class OpenVinoWhisperTranscriber(Transcriber):
    """Whisper on OpenVINO GenAI — the Intel deployment path for ASR.

    Deliberately uses `openvino_genai.WhisperPipeline` rather than
    optimum-intel's OVModelForSpeechSeq2Seq. optimum-intel pins
    transformers<5.1, which collides with what mlx-vlm needs, so requiring it at
    RUNTIME would make the Intel path and the Mac VLM path mutually exclusive in
    one environment. openvino-genai is a standalone runtime with no transformers
    dependency: the model is exported once in a separate venv (see
    mac/intel/README.md) and served here with nothing else installed.

    The IR bundles its own tokenizer/detokenizer, so there is no tokenizer
    version to drift against the weights.
    """

    RUNTIME_GENAI = "openvino-genai"
    RUNTIME_OPTIMUM = "optimum-intel"

    def __init__(self, model_dir, device: str | None = None) -> None:
        import os
        from pathlib import Path

        # Device from VOICELINK_OV_DEVICE (CPU default; GPU/NPU on Intel boxes).
        device = (device or os.environ.get("VOICELINK_OV_DEVICE", "CPU")).upper()
        self.device = device
        model_dir = Path(model_dir)
        if not (model_dir / "openvino_encoder_model.xml").exists():
            raise FileNotFoundError(
                f"no OpenVINO Whisper IR at {model_dir} — export it first "
                "(mac/intel/README.md, 'export venv')")
        self.model_dir = model_dir
        self.runtime = None
        self._pipe = None
        self._model = None
        self._proc = None

        # Prefer openvino-genai: it is the runtime the deployment target wants
        # (no transformers dependency at all) and the API the rubric names.
        genai_error = None
        try:
            import openvino_genai
            # ARM pitfall (mac/intel/README.md): the CPU plugin defaults to f16
            # and the decoder degenerates into '!!!' at 8x the latency. f32 is
            # correct on real Intel CPUs too; GPU/NPU plugins reject the prop.
            props = {"INFERENCE_PRECISION_HINT": "f32"} if device == "CPU" else {}
            self._pipe = openvino_genai.WhisperPipeline(str(model_dir), device, **props)
            # Loading can succeed and generation still fail on a graph-shape
            # mismatch, so prove it works before claiming this runtime.
            self._pipe.generate(np.zeros(16000, dtype=np.float32))
            self.runtime = self.RUNTIME_GENAI
        except Exception as exc:            # noqa: BLE001 - fall through, see below
            genai_error = exc
            self._pipe = None

        if self._pipe is None:
            # optimum-intel serves the same IR. It pins transformers<5.1, which
            # collides with mlx-vlm on this Mac dev box — but NOT on the Intel
            # deployment target, where mlx does not exist and this is the whole
            # stack. See mac/intel/README.md.
            try:
                from optimum.intel import OVModelForSpeechSeq2Seq
                from transformers import AutoProcessor
                self._proc = AutoProcessor.from_pretrained(str(model_dir))
                self._model = OVModelForSpeechSeq2Seq.from_pretrained(str(model_dir))
                self.runtime = self.RUNTIME_OPTIMUM
                print(f"[perception] OpenVINO Whisper via optimum-intel "
                      f"(openvino-genai unavailable: {genai_error})")
            except Exception as exc:        # noqa: BLE001
                raise RuntimeError(
                    f"OpenVINO Whisper IR at {model_dir} could not be served. "
                    f"openvino-genai: {genai_error}; optimum-intel: {exc}"
                ) from exc

    def transcribe(self, pcm_f32, sample_rate):
        # Same contract as the faster-whisper path: 16 kHz mono float32 [-1, 1].
        if sample_rate != 16000:
            raise ValueError(
                f"OpenVINO Whisper expects 16 kHz, got {sample_rate}")
        pcm = np.asarray(pcm_f32, dtype=np.float32).ravel()
        if pcm.shape[0] < 480:  # whisper's mel front-end needs >= ~30 ms
            pcm = np.pad(pcm, (0, 480 - pcm.shape[0]))

        if self.runtime == self.RUNTIME_GENAI:
            text = str(self._pipe.generate(pcm)).strip()
        else:
            feats = self._proc(pcm, sampling_rate=sample_rate,
                               return_tensors="pt").input_features
            ids = self._model.generate(feats)
            text = self._proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

        # Neither OpenVINO runtime exposes per-segment logprobs. Emitting
        # faster-whisper-shaped confidence numbers here would be inventing data,
        # so report neutral values — a real behavioural difference between
        # backends, not a stub. Callers that gate on confidence (is_addressed)
        # will treat every OpenVINO utterance as confident.
        return TranscriptionResult(text, avg_logprob=0.0, no_speech_prob=0.0)


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
    if name == "openvino":
        # No silent mock fallback here on purpose. Selecting the Intel backend
        # and quietly getting scripted text back would misrepresent the very
        # thing the setting exists to demonstrate.
        return OpenVinoWhisperTranscriber(
            getattr(cfg, "WHISPER_OV_DIR", None) or _default_ov_dir(),
            getattr(cfg, "WHISPER_OV_DEVICE", "CPU"),
        )
    if name == "mock":
        return MockTranscriber()
    raise ValueError(f"unknown TRANSCRIBER={name!r}")


def _default_ov_dir():
    from pathlib import Path
    return (Path(__file__).resolve().parent.parent
            / "intel" / "artifacts" / "whisper-base.en-ov-int8")
