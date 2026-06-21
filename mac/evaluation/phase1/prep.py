"""
Phase-1 data prep: repair the scrambled transcript field and extract topics.

THE TRANSPOSE BUG (load-bearing): the AACConversations export wrote the transcript
column-major. Each line is "K: V"; reading the V side top-to-bottom spells the real
transcript, the K side spells the speaker list. Example raw lines:
    [: [        ': '        C: H        a: e   ...
-> value side = "['Hey, ..."   key side = "['Caregiver', ...".
Feeding the raw field to the model destroys the situation signal, so every consumer
calls fix_transcript() first.

Topic extraction (Axis B of the diverse pool): the cheapest thing that works is a
content-word frequency pass over scene+transcript (reuse metrics.content_tokens),
returning the most salient concrete words ("photosynthesis", "curry", "community").
"""

from __future__ import annotations

import ast
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # mac/
from evaluation.metrics import content_tokens  # noqa: E402

# Words that are scene-furniture, not conversational topics — never useful as a
# vocabulary anchor for what the USER would say.
_SCENE_STOP = {
    "aac", "user", "users", "caregiver", "device", "board", "communication",
    "room", "living", "evening", "morning", "afternoon", "midday", "night",
    "dawn", "light", "warm", "soft", "glow", "cozy", "chair", "nearby", "sits",
    "sitting", "comfortably", "atmosphere", "speech", "generating", "lamp",
    "sun", "window", "windows", "curtains", "home", "house", "background",
    "music", "engaged", "attentive", "person", "people", "filters", "filtering",
    "casting", "casts", "streams", "spills", "rising", "filled", "fills", "gaze",
    "glancing", "smiling", "crossed", "relaxed", "front", "over", "their",
    "what", "show", "into", "through", "while", "playing", "sounds", "discuss",
    "sarcasm", "shocker", "check", "tidying", "outburst",
    # generic conversational filler that is never a usable topic anchor
    "like", "good", "feeling", "today", "really", "think", "know", "want",
}


def fix_transcript(raw: str | None) -> str:
    """Undo the column-major scramble; return clean text (best-effort).

    Returns the value-side characters joined. If the field is already clean
    (no "K: V" structure) it is returned unchanged.
    """
    if not raw:
        return ""
    lines = raw.split("\n")
    looks_scrambled = sum(1 for ln in lines if ": " in ln or ln.endswith(":")) >= len(lines) * 0.6
    if not looks_scrambled:
        return raw.strip()
    chars = []
    for ln in lines:
        if ": " in ln:
            chars.append(ln.split(": ", 1)[1])
        elif ln.endswith(":"):
            chars.append("")
    return "".join(chars).strip()


def transcript_turns(raw: str | None) -> list[str]:
    """Fixed transcript parsed into partner utterances (it is a python-repr list)."""
    fixed = fix_transcript(raw)
    if not fixed:
        return []
    try:
        val = ast.literal_eval(fixed)
        if isinstance(val, list):
            return [str(x).strip() for x in val if str(x).strip()]
    except (ValueError, SyntaxError):
        pass
    return [fixed]


def transcript_text(raw: str | None) -> str:
    """Human-readable transcript: partner turns joined, newest last."""
    turns = transcript_turns(raw)
    return "\n".join(turns) if turns else fix_transcript(raw)


def extract_topics(scene: str | None, transcript: str | None, top_n: int = 6) -> list[str]:
    """Most salient concrete content words; the live transcript weighs 3x the
    scene furniture, so what people are actually talking about wins."""
    counts: Counter = Counter()
    for tok in content_tokens(transcript_text(transcript)):
        if tok not in _SCENE_STOP and len(tok) >= 4:
            counts[tok.strip("'")] += 3
    for tok in content_tokens(scene or ""):
        if tok not in _SCENE_STOP and len(tok) >= 4:
            counts[tok.strip("'")] += 1
    return [w for w, _ in counts.most_common(top_n) if w]
