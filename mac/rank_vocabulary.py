"""
================================================================================
THE SEAM — rank_vocabulary()
================================================================================
This module defines the SINGLE function through which EVERY model call in the
entire VoiceLink system passes, now and forever:

    rank_vocabulary(scene, transcript, context) -> ranked_words

Architectural rule:
  * The wearable NEVER calls a model. The web app NEVER calls a model.
  * They reach intelligence ONLY by talking to the Mac server, and the Mac
    server reaches intelligence ONLY by calling THIS function.

Signature and return shape are FIXED. The server and web app depend on them.

TASK 5: plain-prompt local LLM (mlx-lm, Qwen2.5-3B-Instruct-4bit) behind a
swappable Ranker ABC.  Falls back to a deterministic heuristic ranker when the
model is unavailable, and to _STUB_WORDS when fewer than 4 items survive parse.
Symbols are resolved from a locally-cached ARASAAC pictogram set.
================================================================================
"""

from __future__ import annotations

from typing import Optional, TypedDict


class RankedWord(TypedDict):
    word: str    # the AAC vocabulary word/phrase shown on the board
    symbol: str  # "/symbols/{id}.png" when cached; emoji string otherwise


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

_SYSTEM_PROMPT = """\
You are a vocabulary selection assistant for an AAC (Augmentative and Alternative Communication) device.
The device user communicates by selecting pre-built phrases. Your job is to predict which 8 to 12
first-person phrases the user MOST LIKELY NEEDS TO SAY RIGHT NOW, ranked best-first.

Rules:
- Output ONLY a valid JSON array. No explanation, no code fences, no other text.
- Each element is an object: {"word": "<phrase>", "symbol": "<keyword>"}
- "word" is a short first-person phrase the user would say (e.g. "I need help", "yes please", "where is the bathroom").
- "symbol" is a single lowercase English keyword for a pictogram lookup (e.g. "help", "yes", "bathroom", "water").
- Phrases must be immediately speakable BY THE USER — not descriptions of the scene.
- Always include at least 2 high-frequency core words from: yes, no, help, more, stop.
- Return 8 to 12 items.\
"""

_USER_TEMPLATE = """\
Context:
- User name: {user_name}
- Scene: {scene}
- Recent speech heard nearby: {transcript}
- Someone is talking TO the user right now: {addressed}
- Last utterance heard: {last_utterance}
- Relevant people nearby: {people}
- Today/now relevant: {events}
- Sensory needs: {sensory}
- Suggested useful words for this situation: {hints}

Return the JSON array now.\
"""

# Module-level lazy singleton — built on first call when no ranker is injected.
_ranker_singleton = None


def _default_ranker():
    global _ranker_singleton
    if _ranker_singleton is None:
        import ranker_config
        from ranker import make_ranker
        _ranker_singleton = make_ranker(ranker_config)
    return _ranker_singleton


def _build_user_prompt(scene, transcript, context, slice_data) -> str:
    sensory = slice_data.get("sensory_notes", {})
    last_utt = (context.get("last_utterance") or "").lower()
    noise_ctx = sensory.get("noise_sensitive") and any(
        kw in (scene or "").lower() + " " + last_utt
        for kw in ("loud", "crowd", "busy", "noise", "terminal", "airport")
    )
    sensory_str = (
        ", ".join(sensory.get("comfort_items", [])) if noise_ctx else "none"
    )

    people_parts = [
        f'{r["name"]} ({r["role"]})' for r in slice_data.get("matched_relationships", [])
    ]
    events_parts = [e["label"] for e in slice_data.get("matched_events", [])]

    all_hints: list[str] = []
    seen_hints: set[str] = set()
    for ev in slice_data.get("matched_events", []):
        for h in ev.get("vocab_hints", []):
            if h.lower() not in seen_hints:
                seen_hints.add(h.lower())
                all_hints.append(h)
    for rt in slice_data.get("matched_routines", []):
        for h in rt.get("vocab_hints", []):
            if h.lower() not in seen_hints:
                seen_hints.add(h.lower())
                all_hints.append(h)

    return _USER_TEMPLATE.format(
        user_name=slice_data.get("user_name") or "the user",
        scene=scene or "not available",
        transcript=transcript or "none",
        addressed="yes" if context.get("addressed") else "no",
        last_utterance=context.get("last_utterance") or "none",
        people="; ".join(people_parts) or "none",
        events="; ".join(events_parts) or "none",
        sensory=sensory_str,
        hints=", ".join(all_hints) or "none",
    )


def _heuristic_or_stub(slice_data: dict, context: dict) -> list[RankedWord]:
    """Build a heuristic board from slice data without calling the LLM."""
    from symbol_map import resolve_symbol, EMOJI_FALLBACK

    core = [
        ("yes", "yes"), ("no", "no"), ("I need help", "help"),
        ("more", "more"), ("stop", "stop"), ("wait", "wait"),
        ("go", "go"), ("thank you", "thank you"),
    ]
    addressed = context.get("addressed", False)
    if addressed:
        core = [("yes", "yes"), ("no", "no"), ("wait", "wait")] + core

    candidates: list[tuple[str, str]] = list(core)

    for ev in slice_data.get("matched_events", []):
        for h in ev.get("vocab_hints", []):
            candidates.append((h, h.lower()))
    for rt in slice_data.get("matched_routines", []):
        for h in rt.get("vocab_hints", []):
            candidates.append((h, h.lower()))

    seen: set[str] = set()
    items: list[RankedWord] = []
    for word, sym_key in candidates:
        key = word.lower()
        if key in seen:
            continue
        seen.add(key)
        emoji = EMOJI_FALLBACK.get(key, EMOJI_FALLBACK.get(sym_key, "▢"))
        items.append({"word": word, "symbol": resolve_symbol(sym_key, emoji)})
        if len(items) == 12:
            break

    if len(items) < 8:
        for stub in _STUB_WORDS:
            if stub["word"].lower() not in seen:
                items.append(stub)
            if len(items) == 8:
                break

    return items if len(items) >= 4 else list(_STUB_WORDS)


def rank_vocabulary(
    scene: Optional[str] = None,
    transcript: Optional[str] = None,
    context: Optional[dict] = None,
    ranker=None,
) -> list[RankedWord]:
    """Return the vocabulary to show on the AAC board, best-first.

    Args:
        scene:      VLM scene description (str | None)
        transcript: recent speech transcript (str | None)
        context:    structured context dict from ContextState.snapshot();
                    may also carry "personal_context" key injected by server
        ranker:     Ranker instance; if None, uses/builds the module singleton.
                    Pass explicitly to avoid re-loading the model per call.

    Returns:
        list of RankedWord with 8-12 items; never raises; never returns <4 items
        (falls back to _STUB_WORDS in the worst case).
    """
    from personal_context_loader import slice_for_prompt
    from symbol_map import resolve_symbol, EMOJI_FALLBACK

    ctx = context or {}
    profile = ctx.get("personal_context") or {}
    slice_data = slice_for_prompt(profile, scene, transcript, ctx)

    system_prompt = _SYSTEM_PROMPT
    user_prompt = _build_user_prompt(scene, transcript, ctx, slice_data)

    r = ranker if ranker is not None else _default_ranker()

    try:
        raw_items = r.rank(system_prompt, user_prompt)
    except Exception:
        return _heuristic_or_stub(slice_data, ctx)

    # Map raw items to RankedWord: resolve symbol keyword -> path or emoji
    valid: list[RankedWord] = []
    seen_words: set[str] = set()
    for item in raw_items:
        word = str(item.get("word") or "").strip()
        if not word:
            continue
        key = word.lower()
        if key in seen_words:
            continue
        seen_words.add(key)
        sym_key = str(item.get("symbol_keyword") or item.get("symbol") or word).strip().lower()
        emoji = EMOJI_FALLBACK.get(key, EMOJI_FALLBACK.get(sym_key, "▢"))
        symbol = resolve_symbol(sym_key, emoji)
        valid.append({"word": word, "symbol": symbol})
        if len(valid) == 12:
            break

    if len(valid) < 4:
        return list(_STUB_WORDS)

    if len(valid) < 8:
        for stub in _STUB_WORDS:
            if stub["word"].lower() not in seen_words:
                valid.append(stub)
            if len(valid) == 8:
                break

    return valid
