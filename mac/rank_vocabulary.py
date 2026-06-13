"""
================================================================================
THE SEAM — rank_vocabulary()
================================================================================
This module defines the SINGLE function through which EVERY model call in the
entire VoiceLink system passes, now and forever:

    rank_vocabulary(scene, transcript, context) -> ranked_words

Architectural rule (CLAUDE.md, "The seam"):
  * The wearable NEVER calls a model. The web app NEVER calls a model.
  * They reach intelligence ONLY by talking to the Mac server, and the Mac
    server reaches intelligence ONLY by calling THIS function.
  * NEVER wire a model into a route, the firmware, or the UI. It goes here.

Why the boundary exists: it lets the inference backend be swapped later
(plain prompt now -> trained model -> Intel + OpenVINO) without touching the
server routes, the firmware, or the web UI.

CURRENT STATE: hardcoded STUB. No models, no perception. Returns a fixed list.
Per CLAUDE.md ("build the simple version first"), the real version will ship as
a plain prompt and only be measured/replaced later — but it MUST keep this exact
signature and return shape, because the server and web app depend on them.
================================================================================
"""

from __future__ import annotations

from typing import Optional, TypedDict


class RankedWord(TypedDict):
    word: str    # the AAC vocabulary word/phrase shown on the board
    symbol: str  # PLACEHOLDER symbol token (emoji); a real symbol set is wired in later


# Fixed AAC "core vocabulary" stub. Core words are the small set of high-frequency
# words that carry most everyday communication — a sensible placeholder board.
# The emoji are PLACEHOLDERS standing in for real AAC symbols.
_STUB_WORDS: list[RankedWord] = [
    {"word": "yes",        "symbol": "✅"},
    {"word": "no",         "symbol": "❌"},
    {"word": "more",       "symbol": "➕"},
    {"word": "stop",       "symbol": "✋"},
    {"word": "help",       "symbol": "🆘"},
    {"word": "I want",     "symbol": "🙋"},
    {"word": "go",         "symbol": "➡️"},
    {"word": "like",       "symbol": "👍"},
    {"word": "don't like", "symbol": "👎"},
    {"word": "hello",      "symbol": "👋"},
    {"word": "thank you",  "symbol": "🙏"},
    {"word": "all done",   "symbol": "🏁"},
]


def rank_vocabulary(
    scene: Optional[str] = None,
    transcript: Optional[str] = None,
    context: Optional[dict] = None,
) -> list[RankedWord]:
    """Return the vocabulary to show on the AAC board, best-first.

    Inputs (all optional for now; the STUB ignores them):
      scene:      what the wearable camera sees (later: VLM scene description)
      transcript: recent speech heard around the user (later: Whisper output)
      context:    the user's personal context (people, places, routines, prefs)

    Returns: a ranked list of RankedWord dicts, highest-priority first.

    STUB BEHAVIOR: ignores all inputs and returns a fixed core-vocabulary list.
    When this becomes real, the prompt/model call goes HERE and nowhere else:
    scene/transcript/context get composed into the prompt inside this body.
    """
    return list(_STUB_WORDS)
