"""
Diverse-coverage metrics — the Phase-1 measurement suite.

The reframe: the board should COVER the space of plausible replies (one of 7 fits),
not predict the single answer. So we measure both halves:

  Coverage@7  : is the real ground truth caught by >=1 board item?  (recall guard,
                directly comparable to the old ~33% number)
  ILD         : 1 - mean pairwise cosine over the board (diversity; higher = wider)
  dup_rate    : any board pair with cosine >= 0.85 (near-duplicate failure)
  SetCov@7    : fraction of an INDEPENDENT plausible-set the board covers (the
                diversity-aware analogue of recall; needs the oracle)
  Soft-Cov    : max cosine(board, gt) — continuous, threshold-free
  DCS         : SetCov * (0.5 + 0.5*clip(ILD/0.6)) — one summary, never alone

All cosine via the shared MiniLM (embed.py). Lexical-OR-semantic hybrid match
matches the existing harness so Coverage@7 == semantic recall@7, no drift.
"""

from __future__ import annotations

import re

import numpy as np

from evaluation.embed import embed
from evaluation.metrics import content_tokens

COVER_TAU = 0.50       # hybrid semantic threshold (matches existing harness)
SETCOV_TAU = 0.50      # plausible-set coverage threshold
DUP_TAU = 0.85         # near-duplicate threshold

# Communication-function prototypes — both arms' boards are classified against
# these the SAME way (embedding-nearest), so function coverage is a fair,
# generation-tag-free measure of how many DISTINCT moves a board offers.
FUNCTION_PROTOTYPES = {
    "request": "I want something. Please give me that. Can I have it.",
    "reject": "No. I don't want that. Stop. I disagree.",
    "comment": "That is nice. I like it. Good point. Interesting.",
    "question": "What is that? Why? Can you tell me. How about you?",
    "share-info": "I did this. It is mine. Here is a fact about me.",
    "express-feeling": "I feel happy. I am tired. I am annoyed. I am scared.",
    "social": "Hello. Thank you. Good morning. Nice to see you.",
    "direct-attention": "Look at this. Come here. Wait. Listen to me.",
    "clarify": "I mean this. Not that. Let me say it again. Can you repeat.",
}
_proto_vecs = None
_proto_names = list(FUNCTION_PROTOTYPES.keys())


def _prototypes():
    global _proto_vecs
    if _proto_vecs is None:
        _proto_vecs = embed(list(FUNCTION_PROTOTYPES.values()))
    return _proto_vecs


def function_coverage(board_vecs: np.ndarray) -> tuple[int, list[str]]:
    """Number of distinct communication functions a board spans (0-9), fairly
    classified by embedding-nearest prototype. Returns (count, per-item labels)."""
    if board_vecs.shape[0] == 0:
        return 0, []
    sims = board_vecs @ _prototypes().T          # (B, 9)
    idx = sims.argmax(axis=1)
    labels = [_proto_names[i] for i in idx]
    return len(set(labels)), labels


def _phrase_in(needle: str, haystack: str) -> bool:
    """Word-boundary containment. Plain substring let core "no" match "now"
    (INVESTIGATION_useless_boards.md §3); lookarounds keep multi-word and
    punctuation-edged phrases matching while killing the intra-word leak."""
    if not needle or not haystack:
        return False
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None


def _lexical_hit(phrase: str, gt_tokens: set[str], gt_text: str) -> bool:
    p = (phrase or "").lower().strip()
    if not p or not gt_tokens:
        return False
    if _phrase_in(p, gt_text) or _phrase_in(gt_text, p):
        return True
    return bool(content_tokens(p) & gt_tokens)


def board_metrics(board: list[str], gt: str, board_vecs=None, gt_vec=None) -> dict:
    """Coverage@7, Soft-Coverage, ILD, dup_rate for one board against one GT."""
    if board_vecs is None:
        board_vecs = embed(board)
    if gt_vec is None:
        gt_vec = embed([gt])[0] if gt else np.zeros(384, dtype=np.float32)

    sims = board_vecs @ gt_vec if len(board) else np.array([0.0])
    soft = float(sims.max()) if len(board) else 0.0

    gt_tokens = content_tokens(gt)
    gt_text = (gt or "").lower().strip()
    lex = any(_lexical_hit(b, gt_tokens, gt_text) for b in board)
    covered = bool(lex or soft >= COVER_TAU)

    # ILD + dup over the board
    if len(board) >= 2:
        bb = board_vecs @ board_vecs.T
        triu = bb[np.triu_indices(len(board), k=1)]
        ild = float(1.0 - triu.mean())
        dup = bool((triu >= DUP_TAU).any())
    else:
        ild, dup = 0.0, False

    fcov, _ = function_coverage(board_vecs) if len(board) else (0, [])
    return {"covered": covered, "soft": soft, "ild": ild, "dup": dup, "fcov": fcov}


def set_coverage(board_vecs: np.ndarray, plausible_vecs: np.ndarray) -> float:
    """Fraction of plausible responses with a board item within SETCOV_TAU."""
    if plausible_vecs.shape[0] == 0 or board_vecs.shape[0] == 0:
        return 0.0
    sims = plausible_vecs @ board_vecs.T          # (P, B)
    best = sims.max(axis=1)
    return float((best >= SETCOV_TAU).mean())


def dcs(setcov: float, ild: float) -> float:
    """Summary score: substance (SetCov) scaled by a diversity factor in [0.5,1]."""
    return float(setcov * (0.5 + 0.5 * min(max(ild / 0.6, 0.0), 1.0)))


# --------------------------------------------------------------------------
# Aggregation + significance
# --------------------------------------------------------------------------

def bootstrap_ci(deltas: list[float], n: int = 10000, seed: int = 0) -> tuple[float, float]:
    """95% CI of the mean of paired deltas via bootstrap resampling."""
    arr = np.asarray(deltas, dtype=np.float64)
    if arr.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, size=(n, arr.size))].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def mcnemar(base_flags: list[bool], new_flags: list[bool]) -> dict:
    """McNemar on paired Coverage@7 flips (new-only wins b vs base-only wins c)."""
    b = sum(1 for o, n in zip(base_flags, new_flags) if n and not o)  # new fixed
    c = sum(1 for o, n in zip(base_flags, new_flags) if o and not n)  # new broke
    denom = b + c
    stat = ((abs(b - c) - 1) ** 2) / denom if denom else 0.0
    return {"new_only": b, "base_only": c, "stat": float(stat)}
