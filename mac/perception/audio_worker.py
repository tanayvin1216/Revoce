"""
Audio worker — continuous Whisper on the wearable's 0x02 PCM stream.

Runs on its OWN thread so it never stalls waiting on the VLM. It segments the
incoming audio into utterances with a cheap energy VAD, transcribes each
COMPLETE utterance, classifies the speaker, decides whether the user was
addressed, and writes the structured result into the shared ContextState.

Speaker + addressed are HEURISTICS (v1 — "build the simple version first").
Real speaker diarization / partner enrollment replaces them later.
"""

from __future__ import annotations

import queue
import threading

import numpy as np

from .transcriber import TranscriptionResult

# wh-words that start a question, and second-person interrogative phrases
_WH = ("what", "where", "when", "why", "who", "how", "which", "whose")
_SECOND_PERSON = (
    "do you", "are you", "can you", "could you", "would you", "will you",
    "did you", "have you", "you want", "you need", "you like", "want some",
    "ready to",
)


def classify_speaker(rms: float, result: TranscriptionResult, partner_rms_threshold: float) -> str:
    """Near-field + confident speech == the communication partner; quiet or
    low-confidence == ambient/other. Heuristic v1."""
    if result.no_speech_prob > 0.6 or result.avg_logprob < -1.2:
        return "ambient"
    return "partner" if rms >= partner_rms_threshold else "other"


def is_addressed(utterance: str, speaker: str) -> bool:
    """True when the partner just said something aimed at the user that likely
    wants a response (a question or a second-person prompt). Heuristic v1."""
    if speaker != "partner" or not utterance:
        return False
    u = utterance.strip().lower().rstrip(".!")
    if u.endswith("?"):
        return True
    first = u.split()[0] if u.split() else ""
    if first in _WH:
        return True
    return any(p in u for p in _SECOND_PERSON)


class AudioWorker(threading.Thread):
    def __init__(self, audio_q: "queue.Queue[bytes]", transcriber_factory, context_state, cfg) -> None:
        super().__init__(name="audio", daemon=True)
        self._q = audio_q
        self._make_transcriber = transcriber_factory
        self._ctx = context_state
        self._cfg = cfg
        self._stop = threading.Event()
        # current (possibly in-progress) utterance, as raw int16 bytes
        self._utt = bytearray()
        self._in_speech = False
        self._silence_ms = 0.0
        self._recent = ""

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        cfg = self._cfg
        tx = self._make_transcriber()   # build backend here (off the event loop)
        self._tx = tx
        frame_bytes = int(cfg.SAMPLE_RATE * cfg.VAD_FRAME_MS / 1000) * 2  # int16 = 2 bytes
        carry = bytearray()
        while not self._stop.is_set():
            try:
                chunk = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            carry.extend(chunk)
            while len(carry) >= frame_bytes:        # process fixed VAD-sized frames
                frame = bytes(carry[:frame_bytes])
                del carry[:frame_bytes]
                self._process_frame(frame)

    def _process_frame(self, frame_bytes: bytes) -> None:
        cfg = self._cfg
        samples = np.frombuffer(frame_bytes, dtype=np.int16)
        rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2))) if samples.size else 0.0
        if rms >= cfg.VAD_RMS_THRESHOLD:            # speech
            self._utt.extend(frame_bytes)
            self._in_speech = True
            self._silence_ms = 0.0
        elif self._in_speech:                       # trailing silence inside an utterance
            self._utt.extend(frame_bytes)
            self._silence_ms += cfg.VAD_FRAME_MS
            if self._silence_ms >= cfg.VAD_SILENCE_MS:
                self._finalize_utterance()
        # else: idle silence — ignore

    def _finalize_utterance(self) -> None:
        cfg = self._cfg
        pcm16 = np.frombuffer(bytes(self._utt), dtype=np.int16)
        self._utt = bytearray()
        self._in_speech = False
        self._silence_ms = 0.0

        dur_ms = len(pcm16) / cfg.SAMPLE_RATE * 1000.0
        if dur_ms < cfg.VAD_MIN_UTTERANCE_MS:
            return
        rms = float(np.sqrt(np.mean(pcm16.astype(np.float32) ** 2)))
        pcm_f32 = pcm16.astype(np.float32) / 32768.0

        result = self._tx.transcribe(pcm_f32, cfg.SAMPLE_RATE)
        text = result.text.strip()
        if not text:
            return

        speaker = classify_speaker(rms, result, cfg.PARTNER_RMS_THRESHOLD)
        addressed = is_addressed(text, speaker)
        self._recent = (self._recent + " " + text).strip()[-500:]
        self._ctx.update_audio(
            recent_transcript=self._recent,
            last_utterance=text,
            speaker=speaker,
            addressed=addressed,
        )
        print(f"[audio] {speaker:<7} addressed={str(addressed):<5} «{text}»")
