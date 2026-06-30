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

import re
from typing import Optional, TypedDict

BOARD_SIZE = 7  # the device shows this many tappable phrases at once

# The safety words that MUST always reach the board, in this fixed order. They
# are deliberately 4 DISTINCT communication functions (yes=comment, no=reject,
# "I need help"=request, wait=direct-attention), so pinning them into every
# board makes "never mode-collapsed" structurally true even when the LLM
# collapses onto one idea. These are pinned in select_board().
GUARANTEED_CORE: list[str] = ["yes", "no", "I need help", "wait"]


class RankedWord(TypedDict):
    word: str    # the AAC vocabulary word/phrase shown on the board
    symbol: str  # "/symbols/{id}.png" when cached; emoji string otherwise


# ---------------------------------------------------------------------------
# DIVERSE STRATEGY (Phase 1) — generate a wide pool, then SELECT for coverage.
#
# The board must surface ~7 DIFFERENT ways the user could respond, so that one
# of them is the one they want — NOT 7 paraphrases of the single most-likely
# answer. We over-generate along two axes and MMR-select for spread:
#   Axis A — 9 AAC communication functions (one candidate each)
#   Axis B — concrete topics in play (scene + transcript entities)
# plus a few deterministic core words that must always be reachable.
# ---------------------------------------------------------------------------

# AAC pragmatics: the communicative functions a board should be able to launch.
COMMUNICATION_FUNCTIONS = [
    "request", "reject", "comment", "question", "share-information",
    "express-feeling", "social", "direct-attention", "clarify",
]

# Always-reachable safety net; injected (not LLM-generated) so a bad generation
# can never strand the user without yes/no/help.
_CORE_POOL: list[dict] = [
    {"word": "yes", "symbol_keyword": "yes", "function": "comment"},
    {"word": "no", "symbol_keyword": "no", "function": "reject"},
    {"word": "I don't know", "symbol_keyword": "confused", "function": "clarify"},
    {"word": "wait", "symbol_keyword": "wait", "function": "direct-attention"},
    {"word": "I need help", "symbol_keyword": "help", "function": "request"},
]

_DIVERSE_SYSTEM_PROMPT = """\
You generate vocabulary for a NONVERBAL AAC user who speaks by tapping pre-built phrases.
Your job is NOT to guess the one thing they will say. It is to offer MANY DIFFERENT
directions they could take the conversation, so whichever one they mean is on the board.

Hard rules:
- Output ONLY a JSON array. No prose, no code fences.
- Each element: {"word": "<phrase>", "function": "<function>", "topic": "<topic>"}
- "word": something THIS USER could SAY, first person, 1 to 5 words. Never describe the
  scene. Never speak for the other person.
- Every phrase must be a DIFFERENT communicative move. Never repeat or paraphrase an
  earlier phrase. Two phrases that mean the same thing is a failure.\
"""

# Slot-based: one line per function FORCES the model to span the space instead of
# mode-collapsing onto the single likeliest reply (its default failure).
_DIVERSE_USER_TEMPLATE = """\
Situation:
- Scene: {scene}
- Conversation just now: {transcript}
- Most recent thing said to the user: {last_utterance}
- Topics in play: {topics}

Produce EXACTLY ONE short first-person phrase the user could say for EACH of these 9
communicative moves (different move each line), then 4 MORE phrases that each engage a
DIFFERENT topic in play. 13 phrases total, all distinct:

1. request something
2. reject / disagree / decline
3. comment or react
4. ask a question back
5. share a piece of information
6. express a feeling
7. be social (greet / thank / acknowledge)
8. direct attention (get someone to look / wait / come)
9. clarify or repair ("I mean...", "not that")
10-13. four phrases on different topics in play

Return the JSON array of 13 objects now.\
"""


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


# ---------------------------------------------------------------------------
# Diverse pool generation + MMR selection
# ---------------------------------------------------------------------------

def _speakable(word: str) -> bool:
    """Reject narration / second-person / over-long items (3-layer guard)."""
    w = (word or "").strip()
    if not w or len(w.split()) > 6:
        return False
    low = w.lower()
    if low.startswith(("you ", "the user", "they ", "he ", "she ")):
        return False
    return True


def generate_pool(
    scene: Optional[str],
    transcript: Optional[str],
    context: Optional[dict],
    ranker=None,
    topics: Optional[list[str]] = None,
) -> list[dict]:
    """ONE LLM call -> a wide, tagged candidate pool (the expensive half).

    Returns a list of {word, symbol_keyword, function, topic?} including the
    deterministic core words. Never raises: on any failure returns the core
    pool merged with the heuristic board so the user is never stranded.
    """
    ctx = context or {}
    people = "; ".join(
        f'{r.get("name")} ({r.get("role")})'
        for r in (ctx.get("personal_context", {}) or {}).get("relationships", [])
    ) or "none"
    user_prompt = _DIVERSE_USER_TEMPLATE.format(
        user_name=(ctx.get("personal_context", {}) or {}).get("name") or "the user",
        scene=scene or "not available",
        transcript=transcript or "none",
        addressed="yes" if ctx.get("addressed") else "no",
        last_utterance=ctx.get("last_utterance") or "none",
        topics=", ".join(topics or []) or "none",
        people=people,
    )

    r = ranker if ranker is not None else _default_ranker()
    try:
        raw = r.rank(_DIVERSE_SYSTEM_PROMPT, user_prompt, max_tokens=520)
    except TypeError:
        raw = r.rank(_DIVERSE_SYSTEM_PROMPT, user_prompt)  # ranker w/o max_tokens
    except Exception:
        raw = []

    pool: list[dict] = []
    seen: set[str] = set()
    for item in list(_CORE_POOL) + list(raw):
        word = str(item.get("word") or "").strip()
        if not word or not _speakable(word):
            continue
        key = word.lower()
        if key in seen:
            continue
        seen.add(key)
        pool.append({
            "word": word,
            "symbol_keyword": str(item.get("symbol_keyword") or word).strip().lower(),
            "function": item.get("function", ""),
            "topic": item.get("topic", ""),
        })

    if len(pool) < 4:  # generation collapsed -> safety net
        for stub in _STUB_WORDS:
            if stub["word"].lower() not in seen:
                pool.append({"word": stub["word"], "symbol_keyword": stub["word"].lower(),
                             "function": "", "topic": ""})
    return pool


# Words too generic to be a useful "topic in play" for the diverse prompt.
_TOPIC_STOPWORDS = frozenset({
    "the", "and", "with", "that", "this", "they", "them", "their", "there",
    "here", "have", "has", "had", "are", "was", "were", "will", "would",
    "your", "you", "yours", "what", "when", "where", "which", "who", "whom",
    "about", "into", "from", "then", "than", "some", "just", "like", "want",
    "need", "going", "gonna", "okay", "yeah", "right", "really", "very",
    "user", "scene", "none", "available", "said", "say", "says", "talking",
})


def topics_from_context(
    scene: Optional[str],
    transcript: Optional[str],
    context: Optional[dict] = None,
    max_topics: int = 6,
) -> list[str]:
    """Cheap, model-free salient-term extraction for the diverse prompt's topic
    slots. Pulls distinct content words from scene + transcript + last utterance.

    Pure and deterministic — NOT a model call (the server passes the result into
    rank_vocabulary's `topics=`; it must never block the audio path).
    """
    ctx = context or {}
    text = " ".join(
        s for s in (scene, transcript, ctx.get("last_utterance")) if s
    )
    topics: list[str] = []
    seen: set[str] = set()
    for tok in re.findall(r"[A-Za-z][A-Za-z'-]+", text.lower()):
        if len(tok) < 4 or tok in _TOPIC_STOPWORDS or tok in seen:
            continue
        seen.add(tok)
        topics.append(tok)
        if len(topics) >= max_topics:
            break
    return topics


def _to_ranked(items: list[dict]) -> list[RankedWord]:
    """Resolve symbol keywords to pictogram paths / emoji."""
    from symbol_map import resolve_symbol, EMOJI_FALLBACK
    out: list[RankedWord] = []
    for it in items:
        word = it["word"]
        sym_key = (it.get("symbol_keyword") or word).strip().lower()
        emoji = EMOJI_FALLBACK.get(word.lower(), EMOJI_FALLBACK.get(sym_key, "▢"))
        out.append({"word": word, "symbol": resolve_symbol(sym_key, emoji)})
    return out


def _split_core(pool: list[dict]) -> tuple[list[int], list[int]]:
    """Partition pool indices into (pinned core, everything else).

    Core indices are returned in GUARANTEED_CORE order so the safety strip is
    deterministic; every other item (including non-guaranteed core words like
    "I don't know") goes to the rest, which MMR selects from.
    """
    rank_of = {w.lower(): r for r, w in enumerate(GUARANTEED_CORE)}
    core_by_rank: dict[int, int] = {}
    rest: list[int] = []
    for idx, p in enumerate(pool):
        r = rank_of.get(str(p.get("word", "")).lower())
        if r is not None and r not in core_by_rank:
            core_by_rank[r] = idx
        else:
            rest.append(idx)
    core_idx = [core_by_rank[r] for r in sorted(core_by_rank)]
    return core_idx, rest


def _pad_to_k(items: list[RankedWord], k: int) -> list[RankedWord]:
    """Guarantee EXACTLY k items: truncate if long, top up from stubs if short.

    Ensures the board is never empty and never under-fills the device, no matter
    how badly generation degraded.
    """
    out = list(items[:k])
    seen = {it["word"].lower() for it in out}
    for stub in _STUB_WORDS:
        if len(out) >= k:
            break
        if stub["word"].lower() not in seen:
            out.append(stub)
            seen.add(stub["word"].lower())
    return out[:k]


def select_board(
    pool: list[dict],
    scene: Optional[str],
    transcript: Optional[str],
    context: Optional[dict],
    k: int = BOARD_SIZE,
    lam: float = 0.6,
) -> list[RankedWord]:
    """Pick exactly k items: PIN the core safety words, MMR-fill the rest.

    Two structural guarantees the board cannot lose:
      1. The GUARANTEED_CORE safety words are always present (pinned first).
      2. The result is always exactly k items (padded from stubs if short).
    The remaining slots are filled by MMR over MiniLM embeddings for coverage;
    if embeddings are unavailable it falls back to function-coverage ordering
    (still diverse). Pure-python after the one pool-generation call.
    """
    if not pool:
        return _pad_to_k([], k)

    core_idx, rest_idx = _split_core(pool)
    pinned = core_idx[:k]
    remaining = k - len(pinned)

    chosen_rest: list[int] = []
    if remaining > 0 and rest_idx:
        rest_words = [pool[i]["word"] for i in rest_idx]
        rest_functions = [pool[i].get("function", "") for i in rest_idx]
        try:
            from evaluation.embed import embed
            from selection import build_query_vector, mmr_select
            cand_vecs = embed(rest_words)
            query = build_query_vector(embed, scene, transcript, context)
            sel = mmr_select(cand_vecs, query, remaining, lam, rest_functions)
        except Exception:
            sel = _function_coverage_order(rest_functions, remaining)
        chosen_rest = [rest_idx[s] for s in sel]

    ordered = pinned + chosen_rest
    return _pad_to_k(_to_ranked([pool[i] for i in ordered]), k)


def _function_coverage_order(functions: list[str], k: int) -> list[int]:
    """Embedding-free fallback: one item per distinct function first."""
    order, used = [], set()
    for i, f in enumerate(functions):
        if f and f not in used:
            used.add(f)
            order.append(i)
        if len(order) == k:
            return order
    for i in range(len(functions)):
        if i not in order:
            order.append(i)
        if len(order) == k:
            break
    return order


def rank_vocabulary(
    scene: Optional[str] = None,
    transcript: Optional[str] = None,
    context: Optional[dict] = None,
    ranker=None,
    strategy: str = "topk",
    k: int = BOARD_SIZE,
    lam: float = 0.6,
    topics: Optional[list[str]] = None,
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

    # DIVERSE strategy: over-generate a wide pool, then MMR-select for coverage.
    if strategy == "diverse":
        pool = generate_pool(scene, transcript, ctx, ranker=ranker, topics=topics)
        return select_board(pool, scene, transcript, ctx, k=k, lam=lam)

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
