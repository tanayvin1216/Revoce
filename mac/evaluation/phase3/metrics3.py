"""The Phase-3 metric reframe — four scores, computed per board.

The board must COVER the space of plausible replies (so one of ~7 buttons is the
one the user wants), not predict a single utterance. So we stop rewarding a board
for matching one logged answer and instead measure coverage and spread:

  1. distinct-functions@7  how many of the spec's 9 communicative functions the
                           top-7 board spans (taxonomy + embed). Higher = better.
  2. near-duplicate rate   fraction of board item-PAIRS with cosine >= 0.85.
                           Lower = better. (New: coverage_metrics only had a bool.)
  3. coverage-of-intent@7  did the top-7 board offer >=1 item matching ANY
                           acceptable communicative move for the scenario?
                           Lexical-OR-semantic at tau=0.50. Replaces exact-match.
  4. exact-match           DIAGNOSTIC ONLY (see exact_match docstring).

All cosine via the shared MiniLM (embed.py); vectors are normalized so dot==cos.
Lexical-OR-semantic matching reuses coverage_metrics so there is zero drift from
the Phase-1 hybrid matcher.
"""

from __future__ import annotations

import numpy as np

from evaluation.coverage_metrics import DUP_TAU, COVER_TAU, _lexical_hit
from evaluation.metrics import content_tokens
from evaluation.phase3 import taxonomy

TOPK = 7  # the board's "visible" window the metrics judge


def distinct_functions_at(board: list[str], k: int = TOPK, board_vecs=None) -> int:
    """Distinct spec-functions (0-9) spanned by the top-k board words.

    Pass board_vecs to reuse an embedding already computed for this board."""
    top = board[:k]
    if not top:
        return 0
    if board_vecs is not None:
        vecs = board_vecs[:k]
    else:
        from evaluation.embed import embed  # lazy so tests can monkeypatch it
        vecs = embed(top)
    count, _ = taxonomy.classify_functions(vecs)
    return count


def near_dup_rate(board_vecs: np.ndarray) -> float:
    """Fraction of board item-PAIRS that are near-duplicates (cosine >= DUP_TAU).

    0.0 for boards with fewer than 2 items (no pairs to be redundant). This is the
    continuous analogue of coverage_metrics' boolean `dup`: a board with one bad
    pair scores low, a board that is all paraphrases scores near 1.0."""
    n = board_vecs.shape[0]
    if n < 2:
        return 0.0
    bb = board_vecs @ board_vecs.T
    triu = bb[np.triu_indices(n, k=1)]
    return float((triu >= DUP_TAU).mean())


def coverage_of_intent(
    board: list[str],
    acceptable_moves: list[str],
    k: int = TOPK,
    board_vecs=None,
) -> bool:
    """Does the top-k board offer >=1 item matching ANY acceptable move?

    A board item "matches" a move if it is a lexical hit (shared content word /
    substring) OR a strong paraphrase (cosine >= COVER_TAU). This is the exact
    hybrid rule from coverage_metrics/metrics, so coverage-of-intent is directly
    comparable to the Phase-1 hybrid coverage — only the target changed from one
    ground-truth utterance to the SET of acceptable moves.
    """
    top = board[:k]
    moves = [m for m in (acceptable_moves or []) if m and m.strip()]
    if not top or not moves:
        return False

    # Lexical leg: any board item shares a content word / substring with any move.
    for move in moves:
        m_tokens = content_tokens(move)
        m_text = move.lower().strip()
        if any(_lexical_hit(b, m_tokens, m_text) for b in top):
            return True

    # Semantic leg: any (board item, move) pair at cosine >= COVER_TAU.
    from evaluation.embed import embed  # lazy so tests can monkeypatch it
    bvecs = board_vecs[:k] if board_vecs is not None else embed(top)
    mvecs = embed(moves)
    if bvecs.shape[0] == 0 or mvecs.shape[0] == 0:
        return False
    return bool((bvecs @ mvecs.T >= COVER_TAU).any())


def exact_match(board: list[str], gold: str) -> bool:
    """DIAGNOSTIC ONLY — do NOT gate on this.

    Was the single logged `gold` utterance reproduced (lexical/substring) on the
    board? This is the OLD target the Phase-3 reframe demotes. The independent
    plausible-set oracle (evaluation/phase1/oracle.py) measured that a board with
    full real-world communication value still exact-matches the one logged reply
    only ~19% of the time, because the user could plausibly have said many other
    things. We keep it solely to show that number moving INDEPENDENTLY of the
    coverage metrics; coverage-of-intent (#3) is the real recall guard."""
    g = (gold or "").lower().strip()
    if not g:
        return False
    g_tokens = content_tokens(g)
    return any(_lexical_hit(b, g_tokens, g) for b in board)
