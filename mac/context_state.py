"""
Shared perception state — the ONE place perception and the server meet.

Perception workers (audio + vision) WRITE to this; the server READS it in
_make_board() to pass scene/transcript/context to the seam (rank_vocabulary).
This is the only coupling between perception and the server: perception never
calls rank_vocabulary or the UI directly (CLAUDE.md). Thread-safe, because the
perception workers run on their own threads off the asyncio event loop.
"""

from __future__ import annotations

import copy
import threading
import time

# How long the most recent addressed question stays the "active question" the
# board answers — long enough for a nonverbal user to compose a reply, short
# enough that a stale question doesn't keep driving the board forever.
ACTIVE_QUESTION_TTL_S = 90.0


class ContextState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        # latest structured audio understanding
        self._audio = {
            "recent_transcript": "",
            "last_utterance": None,   # last COMPLETE utterance (overwritten freely)
            "speaker": None,          # "partner" | "other" | "ambient"
            "addressed": False,       # partner just said something wanting a response
            # The active question is HELD: set only when an addressed question
            # lands, and NOT overwritten by later non-question chatter, so the
            # board keeps answering it until the user replies or it times out.
            "active_question": None,
            "active_question_ts": 0.0,
            "ts": 0.0,
        }
        # latest scene understanding
        self._vision = {
            "description": None,      # short scene description (what to talk about)
            "scene_changed": False,
            "ts": 0.0,
        }

    def update_audio(self, **fields) -> None:
        with self._lock:
            self._audio.update(fields)
            self._audio["ts"] = time.time()

    def set_active_question(self, question: str) -> None:
        """Hold a new addressed question as the one the board should answer."""
        with self._lock:
            self._audio["active_question"] = question
            self._audio["active_question_ts"] = time.time()

    def clear_active_question(self) -> None:
        """Drop the held question (e.g. the user tapped an answer)."""
        with self._lock:
            self._audio["active_question"] = None
            self._audio["active_question_ts"] = 0.0

    def update_vision(self, **fields) -> None:
        with self._lock:
            self._vision.update(fields)
            self._vision["ts"] = time.time()

    def snapshot(self) -> tuple[str | None, str | None, dict]:
        """Return (scene, transcript, context) for rank_vocabulary().

        scene/transcript are the convenience positional inputs; context carries
        the full structured understanding (speaker, addressed, last_utterance,
        scene_changed, timestamps) — this is what gets logged with every tap.
        """
        with self._lock:
            audio = copy.deepcopy(self._audio)
            vision = copy.deepcopy(self._vision)
        scene = vision["description"]
        transcript = audio["recent_transcript"] or None
        # Expire the held question once it's older than the TTL.
        active_question = audio["active_question"]
        if active_question and (time.time() - audio["active_question_ts"]) > ACTIVE_QUESTION_TTL_S:
            active_question = None
        context = {
            "speaker": audio["speaker"],
            "addressed": audio["addressed"],
            "last_utterance": audio["last_utterance"],
            "active_question": active_question,
            "recent_transcript": audio["recent_transcript"],
            "scene_changed": vision["scene_changed"],
            "audio_ts": audio["ts"],
            "vision_ts": vision["ts"],
        }
        return scene, transcript, context
