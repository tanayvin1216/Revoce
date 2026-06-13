"""
Metrics for vocabulary prediction.

The model produces a ranked board (a list of phrases). Ground truth is the AAC
user's actual utterance. We ask: was a word the user actually needed available
on the board, and how near the top?

Matching is lexical and deliberately transparent (this is a judgment call, so
it must be inspectable, not hidden): a predicted phrase is RELEVANT to the
utterance if they share a content word (after stop-word removal) or one is a
substring of the other. We report:
  - recall@k : fraction of examples with >=1 relevant phrase in the top-k board
  - token_recall@k : fraction of the utterance's content words covered by top-k
  - MRR : mean reciprocal rank of the first relevant phrase
  - mean_first_rank : mean 1-indexed rank of the first relevant phrase (hits only)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Small stop list. Note we KEEP high-value AAC words (yes, no, go, help, want,
# need) — they are content here, not noise.
_STOP = {
    "a", "an", "the", "to", "of", "in", "on", "at", "is", "are", "am", "was",
    "were", "be", "been", "do", "does", "did", "i", "you", "he", "she", "it",
    "we", "they", "me", "my", "your", "this", "that", "and", "or", "but", "for",
    "with", "please", "would", "could", "can", "will", "just", "so", "okay",
}

_WORD = re.compile(r"[a-z0-9']+")


def content_tokens(text: str) -> set[str]:
    """Lowercase content words (stop-words removed, single chars dropped)."""
    return {
        t for t in _WORD.findall((text or "").lower())
        if len(t) >= 2 and t not in _STOP
    }


def is_relevant(phrase: str, target_tokens: set[str], target_text: str) -> bool:
    if not target_tokens:
        return False
    p = (phrase or "").lower().strip()
    if not p:
        return False
    if p in target_text or target_text in p:        # substring either way
        return True
    return bool(content_tokens(p) & target_tokens)   # shared content word


@dataclass
class ExampleScore:
    hits: dict[int, int]            # k -> 0/1 (any relevant in top-k)
    token_recall: dict[int, float]  # k -> covered content-word fraction
    first_rank: int | None          # 1-indexed rank of first relevant phrase


def score_example(predicted: list[str], ground_truth: str, ks: list[int]) -> ExampleScore:
    target = content_tokens(ground_truth)
    gt_text = (ground_truth or "").lower().strip()

    first_rank = None
    covered_by_rank: list[set[str]] = []
    running: set[str] = set()
    for idx, phrase in enumerate(predicted, start=1):
        if is_relevant(phrase, target, gt_text) and first_rank is None:
            first_rank = idx
        running = running | (content_tokens(phrase) & target)
        covered_by_rank.append(set(running))

    hits, token_recall = {}, {}
    for k in ks:
        hits[k] = 1 if (first_rank is not None and first_rank <= k) else 0
        covered = covered_by_rank[min(k, len(covered_by_rank)) - 1] if covered_by_rank else set()
        token_recall[k] = (len(covered) / len(target)) if target else 0.0
    return ExampleScore(hits=hits, token_recall=token_recall, first_rank=first_rank)


@dataclass
class Aggregate:
    n: int
    ks: list[int]
    recall: dict[int, float]
    token_recall: dict[int, float]
    mrr: float
    mean_first_rank: float | None


def aggregate(scores: list[ExampleScore], ks: list[int]) -> Aggregate:
    n = len(scores)
    if n == 0:
        return Aggregate(0, ks, {k: 0.0 for k in ks}, {k: 0.0 for k in ks}, 0.0, None)
    recall = {k: sum(s.hits[k] for s in scores) / n for k in ks}
    token_recall = {k: sum(s.token_recall[k] for s in scores) / n for k in ks}
    mrr = sum((1.0 / s.first_rank) if s.first_rank else 0.0 for s in scores) / n
    ranks = [s.first_rank for s in scores if s.first_rank]
    mean_first_rank = (sum(ranks) / len(ranks)) if ranks else None
    return Aggregate(n, ks, recall, token_recall, mrr, mean_first_rank)
