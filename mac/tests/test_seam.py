"""Seam invariant tests for the diverse strategy (Phase 1).

These pin down the SAFETY contract of rank_vocabulary()'s diverse path. Each
test drives the seam with a FakeRanker so it is deterministic and model-free
(no mlx, no network), and monkeypatches the embedder so MMR runs on numpy alone.

Invariants under test:
  1. The deterministic core (yes / no / I need help / wait) is ALWAYS present.
  2. The board is ALWAYS exactly BOARD_SIZE items.
  3. The board is NEVER empty, even on total generation failure.
  4. The board is NEVER mode-collapsed: >= 4 distinct communication functions.
  5. A regression guard: a mode-collapsed POOL still yields >= 4 functions.
"""

from __future__ import annotations

import zlib

import numpy as np
import pytest

import evaluation.embed as embed_mod
from rank_vocabulary import (
    BOARD_SIZE,
    GUARANTEED_CORE,
    _CORE_POOL,
    rank_vocabulary,
    topics_from_context,
)


class FakeRanker:
    """Returns a fixed list[dict], mimicking a parsed LLM response. Deterministic."""

    def __init__(self, items: list[dict]) -> None:
        self._items = items

    def rank(self, system_prompt, user_prompt, max_tokens=None) -> list[dict]:
        return list(self._items)


class BoomRanker:
    """Always raises — simulates a dead/unavailable model mid-inference."""

    def rank(self, system_prompt, user_prompt, max_tokens=None) -> list[dict]:
        raise RuntimeError("model down")


@pytest.fixture
def fast_embed(monkeypatch):
    """Deterministic, offline embedding so select_board's MMR path runs on numpy
    alone — no MiniLM load, no network, stable across runs (crc32, not hash())."""

    def _embed(texts):
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        vecs = []
        for t in texts:
            seed = zlib.crc32(t.encode("utf-8"))
            v = np.random.default_rng(seed).standard_normal(384).astype(np.float32)
            v /= np.linalg.norm(v) or 1.0
            vecs.append(v)
        return np.asarray(vecs, dtype=np.float32)

    monkeypatch.setattr(embed_mod, "embed", _embed)
    return _embed


# --- helpers ----------------------------------------------------------------

def _words(board) -> list[str]:
    return [b["word"].lower() for b in board]


def _functions_for(board, llm_items: list[dict]) -> list[str]:
    """Map each board word back to its communication function, using the known
    core map plus the items the FakeRanker emitted (the test owns both)."""
    fmap: dict[str, str] = {}
    for it in list(_CORE_POOL) + list(llm_items):
        fmap[str(it.get("word", "")).lower()] = it.get("function", "")
    return [fmap.get(w, "") for w in _words(board)]


# A healthy, varied pool spanning many communication functions.
_VARIED_ITEMS = [
    {"word": "can I have water", "function": "request", "topic": "water"},
    {"word": "I don't want that", "function": "reject", "topic": "food"},
    {"word": "that looks fun", "function": "comment", "topic": "game"},
    {"word": "what time is it", "function": "question", "topic": "time"},
    {"word": "my name is Sam", "function": "share-information", "topic": "self"},
    {"word": "I feel tired", "function": "express-feeling", "topic": "body"},
    {"word": "thank you so much", "function": "social", "topic": "manners"},
    {"word": "look over there", "function": "direct-attention", "topic": "room"},
    {"word": "not that one", "function": "clarify", "topic": "choice"},
]


# --- invariant 1: core always present ---------------------------------------

def test_core_always_present_even_when_generation_is_useless(fast_embed):
    board = rank_vocabulary(
        "a quiet room", "hello there", {}, FakeRanker([]), strategy="diverse"
    )
    words = _words(board)
    for core in GUARANTEED_CORE:
        assert core.lower() in words, f"core word {core!r} missing from board"


def test_core_survives_a_large_competing_pool(fast_embed):
    board = rank_vocabulary(
        "classroom", "let's talk about photosynthesis", {},
        FakeRanker(_VARIED_ITEMS), strategy="diverse",
    )
    words = _words(board)
    for core in GUARANTEED_CORE:
        assert core.lower() in words


# --- invariant 2: always exactly N ------------------------------------------

@pytest.mark.parametrize("items", [
    [],                                                        # empty generation
    [{"word": "water", "function": "request"}],                # single item
    _VARIED_ITEMS,                                             # full pool
    _VARIED_ITEMS * 3,                                         # oversized pool
])
def test_board_is_always_exactly_board_size(fast_embed, items):
    board = rank_vocabulary("scene", "talk", {}, FakeRanker(items), strategy="diverse")
    assert len(board) == BOARD_SIZE


# --- invariant 3: never empty -----------------------------------------------

def test_board_never_empty_on_total_failure(fast_embed):
    board = rank_vocabulary(None, None, None, BoomRanker(), strategy="diverse")
    assert board, "board must never be empty"
    assert len(board) == BOARD_SIZE


# --- invariant 4: never mode-collapsed --------------------------------------

def test_board_spans_at_least_four_distinct_functions(fast_embed):
    board = rank_vocabulary(
        "kitchen", "do you want dinner", {}, FakeRanker(_VARIED_ITEMS),
        strategy="diverse",
    )
    funcs = [f for f in _functions_for(board, _VARIED_ITEMS) if f]
    assert len(set(funcs)) >= 4


# --- invariant 5: REGRESSION — mode-collapsed pool still spreads -------------

def test_regression_mode_collapsed_pool_is_rescued_by_pinned_core(fast_embed):
    """If the LLM mode-collapses (13 near-duplicate 'request' phrases on one
    topic), the board MUST NOT collapse with it. The pinned core contributes 4
    distinct functions, so this asserts >= 4. Remove the core-pin and this fails.
    """
    collapsed = [
        {"word": f"I want the item number {i}", "function": "request", "topic": "stuff"}
        for i in range(13)
    ]
    board = rank_vocabulary("a store", "", {}, FakeRanker(collapsed), strategy="diverse")

    funcs = [f for f in _functions_for(board, collapsed) if f]
    assert len(set(funcs)) >= 4, f"board mode-collapsed: only {set(funcs)}"

    # And the safety net survived the collapse.
    for core in GUARANTEED_CORE:
        assert core.lower() in _words(board)


# --- topk fallback still works (signature default unchanged) -----------------

def test_topk_strategy_still_returns_a_usable_board(fast_embed):
    items = [
        {"word": "yes", "symbol_keyword": "yes"},
        {"word": "no", "symbol_keyword": "no"},
        {"word": "I want water", "symbol_keyword": "water"},
        {"word": "I need help", "symbol_keyword": "help"},
    ]
    board = rank_vocabulary("scene", "talk", {}, FakeRanker(items), strategy="topk")
    assert len(board) >= 4


# --- topics_from_context: pure, model-free helper ---------------------------

def test_topics_from_context_extracts_salient_terms():
    topics = topics_from_context(
        "a biology classroom with a whiteboard", "explain photosynthesis", {}
    )
    assert "photosynthesis" in topics
    assert "classroom" in topics
    assert "with" not in topics          # stopword filtered
    assert "a" not in topics             # too short
    assert len(topics) <= 6


def test_topics_from_context_handles_empty_inputs():
    assert topics_from_context(None, None, None) == []


# --- answer-mode: questions to the user surface real answers -----------------

def test_is_question_detection():
    from rank_vocabulary import _is_question
    # direct/console call: the transcript is the question
    assert _is_question("How do plants make sugar?", {})
    assert _is_question("are you cold?", {})
    # live path: the held active_question is the question
    assert _is_question(None, {"active_question": "Do you want water?"})
    # not questions
    assert not _is_question("the sky is blue today", {})
    assert not _is_question(None, None)
    # a cleared/expired held question must not register as a question (H1/H2)
    assert not _is_question(None, {"active_question": None, "last_utterance": "Do you want water?"})


def test_answer_mode_surfaces_direct_answers_for_a_question(fast_embed):
    # The (faked) LLM returns answer-tagged candidates plus the usual deflections.
    items = [
        {"word": "sunlight", "function": "answer", "topic": "answer"},
        {"word": "water", "function": "answer", "topic": "answer"},
        {"word": "carbon dioxide", "function": "answer", "topic": "answer"},
        {"word": "from the sun", "function": "answer", "topic": "answer"},
        {"word": "Can you explain more?", "function": "clarify", "topic": "photosynthesis"},
        {"word": "I feel confused.", "function": "express-feeling", "topic": "photosynthesis"},
    ]
    board = rank_vocabulary(
        "a biology classroom",
        "How do plants make sugar during photosynthesis?",
        {}, FakeRanker(items), strategy="diverse",
    )
    words = set(_words(board))
    answers = {"sunlight", "water", "carbon dioxide", "from the sun"}
    assert len(answers & words) >= 3, f"expected >=3 answers, got {answers & words}"
    # core safety net and exact size are still intact in answer-mode
    for core in GUARANTEED_CORE:
        assert core.lower() in words
    assert len(board) == BOARD_SIZE


def test_non_question_is_unaffected_by_answer_mode(fast_embed):
    # No question, no answer tags -> behaves exactly like the diverse path.
    board = rank_vocabulary(
        "a kitchen", "we are making dinner together", {},
        FakeRanker(_VARIED_ITEMS), strategy="diverse",
    )
    assert len(board) == BOARD_SIZE
    for core in GUARANTEED_CORE:
        assert core.lower() in _words(board)


# --- the question drives the board; the scene is only background -------------

def test_current_question_uses_held_question_only_on_live_path():
    from rank_vocabulary import _current_question
    # live path (active_question key present): only the held question counts
    ctx = {"active_question": "Where do I sit?", "last_utterance": "anyway it's late"}
    assert _current_question(None, ctx) == "Where do I sit?"
    # cleared/expired held question must NOT resurrect from last_utterance (H1/H2)
    assert _current_question(None, {"active_question": None, "last_utterance": "Do you want water?"}) == ""
    # direct/console call (no active_question key): the transcript is the question
    assert _current_question("typed question?", {}) == "typed question?"


def test_query_ignores_rolling_chatter_when_a_question_is_held():
    """H4: with a held question, later chatter in the transcript must not dilute
    the query signal."""
    import numpy as np
    from selection import build_query_vector

    def embed(texts):
        out = []
        for t in texts:
            v = np.zeros(384, dtype=np.float32)
            if "QWORD" in t:
                v[0] = 1.0
            if "CHATTER" in t:
                v[2] = 1.0
            n = np.linalg.norm(v)
            out.append(v / n if n else v)
        return np.array(out, dtype=np.float32)

    q = build_query_vector(embed, "a scene", "CHATTER CHATTER late", {"active_question": "QWORD"})
    assert q[0] > 0 and q[2] == 0, "query must be the held question, not the chatter"


def test_near_duplicate_answers_are_diversified_out(monkeypatch):
    """Six answers where five are 'seated' paraphrases must NOT all reach the
    board — MMR should keep ~one and surface the genuinely different answers."""
    import numpy as np
    import evaluation.embed as embed_mod

    def embed(texts):
        out = []
        for t in texts:
            tl = t.lower()
            v = np.zeros(384, dtype=np.float32)
            if "seated" in tl:
                v[0] = 1.0; v[1] = 0.02      # near-identical cluster
            elif "back" in tl:
                v[2] = 1.0
            elif "reservation" in tl:
                v[3] = 1.0
            elif "table" in tl:
                v[4] = 1.0
            else:
                v[5] = 1.0
            n = np.linalg.norm(v)
            out.append(v / n if n else v)
        return np.array(out, dtype=np.float32)

    monkeypatch.setattr(embed_mod, "embed", embed)
    items = [
        {"word": "I'm seated", "function": "answer", "topic": "answer"},
        {"word": "Yes, I'm seated", "function": "answer", "topic": "answer"},
        {"word": "Yes, seated here", "function": "answer", "topic": "answer"},
        {"word": "in the back", "function": "answer", "topic": "answer"},
        {"word": "I need a reservation", "function": "answer", "topic": "answer"},
        {"word": "I don't have a table", "function": "answer", "topic": "answer"},
    ]
    board = rank_vocabulary(
        "a restaurant", "where are you seated?", {}, FakeRanker(items), strategy="diverse",
    )
    words = _words(board)
    seated = [w for w in words if "seated" in w]
    assert len(seated) <= 2, f"too many near-duplicate seated answers: {seated}"
    # at least one genuinely different answer made it in
    assert any(k in " ".join(words) for k in ("back", "reservation", "table"))


def test_only_genuine_questions_are_held():
    from perception.audio_worker import is_question_to_user
    assert is_question_to_user("Where do I sit?")
    assert is_question_to_user("how are you")          # wh-word opener
    assert not is_question_to_user("you want some water")   # second-person, not a question
    assert not is_question_to_user("ready to go")
    assert not is_question_to_user("")


def test_answer_block_centers_the_question_over_the_scene():
    from rank_vocabulary import _answer_block
    block = _answer_block("Where were you supposed to sit?")
    assert "Where were you supposed to sit?" in block
    assert "ANSWER THIS QUESTION" in block
    assert "FOLLOW THE QUESTION" in block


def test_query_weights_favor_the_question_over_the_scene():
    """The query vector must lean toward the question, not the scene, so a vivid
    scene can't hijack relevance ranking."""
    import numpy as np
    from selection import build_query_vector

    def embed(texts):
        out = []
        for t in texts:
            v = np.zeros(384, dtype=np.float32)
            if "QWORD" in t:
                v[0] = 1.0          # question axis
            if "SWORD" in t:
                v[1] = 1.0          # scene axis
            n = np.linalg.norm(v)
            out.append(v / n if n else v)
        return np.array(out, dtype=np.float32)

    q = build_query_vector(embed, "SWORD a busy scene", "QWORD the question", {})
    assert q[0] > q[1], "question axis must outweigh scene axis in the query"


# --- active-question hold: ContextState ---------------------------------------

def test_active_question_held_across_later_chatter():
    from context_state import ContextState
    cs = ContextState()
    # an addressed question lands
    cs.update_audio(last_utterance="Where do I sit?", addressed=True)
    cs.set_active_question("Where do I sit?")
    # later, non-addressed chatter overwrites last_utterance...
    cs.update_audio(last_utterance="anyway, it's getting late", addressed=False)
    _, _, ctx = cs.snapshot()
    assert ctx["last_utterance"] == "anyway, it's getting late"
    assert ctx["active_question"] == "Where do I sit?"   # ...but the question is held


def test_active_question_clears_on_answer():
    from context_state import ContextState
    cs = ContextState()
    cs.set_active_question("Where do I sit?")
    cs.clear_active_question()
    _, _, ctx = cs.snapshot()
    assert ctx["active_question"] is None


def test_active_question_expires_after_ttl(monkeypatch):
    import context_state
    monkeypatch.setattr(context_state, "ACTIVE_QUESTION_TTL_S", -1.0)  # everything is stale
    cs = context_state.ContextState()
    cs.set_active_question("Where do I sit?")
    _, _, ctx = cs.snapshot()
    assert ctx["active_question"] is None
