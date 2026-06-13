"""
Personal context loader — pure JSON + stdlib, no model dependency.

load_profile(path)  -> dict  (safe: returns minimal stub on missing/invalid file)
slice_for_prompt()  -> dict  (relevance heuristic for the current moment)
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

_MINIMAL_PROFILE: dict[str, Any] = {
    "version": 1,
    "identity": {"name": "User", "pronouns": "", "aac_user": True, "primary_language": "en"},
    "relationships": [],
    "routines": [],
    "sensory_notes": {"prefers_quiet": False, "noise_sensitive": False, "dietary": [], "comfort_items": []},
    "todays_events": [],
}

_NOISE_KEYWORDS = {"loud", "crowd", "busy", "noise", "terminal", "airport"}


def load_profile(path: str | Path) -> dict:
    """Read and return the personal context profile.

    On FileNotFoundError or JSONDecodeError returns a minimal stub so callers
    never need to handle None.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
        return json.loads(text)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return dict(_MINIMAL_PROFILE)


def slice_for_prompt(
    profile: dict,
    scene: str | None,
    transcript: str | None,
    context: dict,
) -> dict:
    """Return the context slice relevant to the current moment.

    Returns:
        {
            "user_name": str,
            "matched_events": list[dict],
            "matched_relationships": list[dict],
            "matched_routines": list[dict],
            "sensory_notes": dict,
        }
    """
    search_parts = filter(None, [
        scene,
        transcript,
        context.get("last_utterance"),
        context.get("recent_transcript"),
    ])
    search_text = " ".join(search_parts).lower()

    today = date.today().isoformat()

    # Events: date guard + keyword match
    matched_events = [
        ev for ev in profile.get("todays_events", [])
        if _event_matches(ev, search_text, today)
    ]

    # Relationships: alias in speech AND (speaker==partner OR alias in last_utterance)
    last_utt = (context.get("last_utterance") or "").lower()
    speaker = context.get("speaker") or ""
    matched_relationships = [
        rel for rel in profile.get("relationships", [])
        if _relationship_matches(rel, search_text, last_utt, speaker)
    ]

    # Routines: any location_keyword in search_text
    matched_routines = [
        rt for rt in profile.get("routines", [])
        if _routine_matches(rt, search_text)
    ]

    sensory = profile.get("sensory_notes", _MINIMAL_PROFILE["sensory_notes"])
    identity = profile.get("identity", {})
    user_name = identity.get("name") or "the user"

    return {
        "user_name": user_name,
        "matched_events": matched_events,
        "matched_relationships": matched_relationships,
        "matched_routines": matched_routines,
        "sensory_notes": sensory,
    }


def _event_matches(event: dict, search_text: str, today: str) -> bool:
    event_date = event.get("date")
    if event_date and event_date != today:
        return False
    loc_kws = [k.lower() for k in event.get("location_keywords", [])]
    partner_kws = [k.lower() for k in event.get("partner_keywords", [])]
    all_kws = loc_kws + partner_kws
    return any(kw in search_text for kw in all_kws)


def _relationship_matches(rel: dict, search_text: str, last_utt: str, speaker: str) -> bool:
    aliases = [a.lower() for a in rel.get("aliases", [])]
    if not aliases:
        return False
    alias_in_speech = any(alias in search_text for alias in aliases)
    if not alias_in_speech:
        return False
    # Alias guard: speaker is "partner" OR alias appears in last utterance
    if speaker == "partner":
        return True
    return any(alias in last_utt for alias in aliases)


def _routine_matches(routine: dict, search_text: str) -> bool:
    loc_kws = [k.lower() for k in routine.get("location_keywords", [])]
    return any(kw in search_text for kw in loc_kws)
