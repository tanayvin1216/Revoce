"""Track-2 honest-gate tests — synthetic boards, real local MiniLM.

The investigation (evaluation/phase4/INVESTIGATION_useless_boards.md §3) showed
the old gates certified a hedge-wall board: coverage-of-intent was propped up by
the 4 pinned core words, near_dup_rate averaged the hedge pairs away, and core
"no" lexically matched "now". These tests pin the fixed instruments:

  1. same_function_pair_rate: a wall of hedges FAILs the <=0.10 student gate,
     a genuinely diverse board PASSes.
  2. _lexical_hit word-boundary: "no" no longer matches inside "now".
  3. dynamic-vs-full coverage: a board whose only match is a pinned core word
     covers on the FULL window but NOT on the dynamic words — the two numbers
     must diverge, and the gate reads the dynamic one.
  4. is_hedge: hedge-labelled acceptable moves are dropped before scoring.

MiniLM is already local (same policy as test_phase4.py) — no fake embedder here
because the hedge/diverse distinction IS the semantic behaviour under test.
"""

from __future__ import annotations

from evaluation.coverage_metrics import _lexical_hit
from evaluation.metrics import content_tokens
from evaluation.phase3 import metrics3
from evaluation.phase4 import evaluate

HEDGE_WALL = ["I don't know", "I'm not sure", "i am not sure", "no idea",
              "I can't remember", "not sure", "maybe"]
DIVERSE_BOARD = ["what about you?", "a window seat", "I need help",
                 "I feel happy", "can you repeat that?", "hello", "stop"]
CORE_WORDS = ["yes", "no", "I need help", "wait"]


# --- 1. same_function_pair_rate gate ----------------------------------------

def test_hedge_wall_fails_same_function_gate():
    rate = metrics3.same_function_pair_rate(HEDGE_WALL)
    assert rate > evaluate.SFP_GATE
    assert rate >= 0.5  # a 7-hedge wall is mostly same-function pairs


def test_diverse_board_passes_same_function_gate():
    rate = metrics3.same_function_pair_rate(DIVERSE_BOARD)
    assert rate <= evaluate.SFP_GATE


def test_same_function_pair_rate_degenerate_boards():
    assert metrics3.same_function_pair_rate([]) == 0.0
    assert metrics3.same_function_pair_rate(["yes"]) == 0.0


def test_same_function_pair_rate_respects_k():
    # Only the first k items are judged: 2 hedges inside k, wall beyond it.
    board = ["I don't know", "I'm not sure"] + HEDGE_WALL
    rate_k2 = metrics3.same_function_pair_rate(board, k=2)
    assert rate_k2 == 1.0  # the single pair in window shares a function


# --- 2. word-boundary lexical matching ---------------------------------------

def test_no_does_not_match_now():
    # The investigation's leak: substring "no" in "now" counted as a hit.
    assert not _lexical_hit("no", content_tokens("now"), "now")


def test_exact_and_phrase_hits_still_work():
    assert _lexical_hit("no", content_tokens("no thanks"), "no thanks")
    assert _lexical_hit("no", content_tokens("no"), "no")
    assert _lexical_hit("window seat", content_tokens("a window seat please"),
                        "a window seat please")


def test_coverage_of_intent_no_now_boundary():
    # Board offers only "no"; the move only contains "now" — must not cover
    # lexically (and "no" vs this move is nowhere near the semantic threshold).
    assert not metrics3.coverage_of_intent(
        ["no"], ["now boarding at gate four"])


# --- 3. dynamic vs full coverage divergence -----------------------------------

def test_dynamic_vs_full_coverage_diverges_on_pinned_core_board():
    board = CORE_WORDS + ["helicopter", "algebra", "tuesday"]
    moves = ["Wait, I have a peanut allergy"]
    full = metrics3.coverage_of_intent(board, moves, metrics3.TOPK)
    dyn = evaluate.dynamic_words(board, CORE_WORDS)
    dynamic = metrics3.coverage_of_intent(dyn, moves, metrics3.TOPK)
    assert full and not dynamic  # old gate certified this board; new gate reads dyn


def test_dynamic_words_takes_non_core_inside_top7():
    board = CORE_WORDS + ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    assert evaluate.dynamic_words(board, CORE_WORDS) == ["alpha", "beta", "gamma"]


# --- 4. hedge-labelled acceptable moves are dropped ---------------------------

def test_is_hedge_variants():
    for phrase in ["i don't know", "I don't know.", "I’m not sure",
                   "I am not sure.", "not sure", "no idea"]:
        assert metrics3.is_hedge(phrase), phrase
    for phrase in ["a window seat", "carbon dioxide", "my aunt", ""]:
        assert not metrics3.is_hedge(phrase), phrase


def test_hedge_wall_board_fails_dynamic_coverage_when_hedge_moves_dropped():
    # Seating scenario in miniature: the only acceptable-move the hedge board
    # matches is itself a hedge; drop it and the board no longer covers.
    board = CORE_WORDS + HEDGE_WALL[:3]
    moves = ["a window seat", "next to my aunt", "I don't know"]
    non_hedge = [m for m in moves if not metrics3.is_hedge(m)]
    assert non_hedge == ["a window seat", "next to my aunt"]
    dyn = evaluate.dynamic_words(board, CORE_WORDS)
    assert metrics3.coverage_of_intent(dyn, moves, metrics3.TOPK)      # gamed
    assert not metrics3.coverage_of_intent(dyn, non_hedge, metrics3.TOPK)  # honest


def test_gate_constants():
    assert evaluate.SFP_GATE == 0.10
    assert evaluate.DYNAMIC_COVERAGE_GATE == 0.90
