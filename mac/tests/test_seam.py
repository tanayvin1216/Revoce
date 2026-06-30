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
