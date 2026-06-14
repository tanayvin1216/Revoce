"""
Metrics for vocabulary prediction.

The model produces a ranked board (a list of phrases). Ground truth is the AAC
user's actual utterance. We ask: was a word the user actually needed available
on the board, and how near the top?

Two interchangeable MATCHERS decide whether a predicted phrase is "relevant" to
the utterance — chosen at the command line, both transparent:

  - lexical  : shares a content word (stop-words removed) or substring overlap.
               Strict; under-credits correct paraphrases (a pessimistic floor).
  - semantic : cosine similarity of sentence embeddings >= threshold.
               Credits paraphrase ("good" for "Feeling good thanks").

Reported per matcher:
  - recall@k  : fraction of examples with >=1 relevant phrase in the top-k board
  - secondary@k : token coverage (lexical) or best similarity (semantic)
  - MRR + mean_first_rank : the "time-to-relevant-word" proxy
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_STOP = {
    "a", "an", "the", "to", "of", "in", "on", "at", "is", "are", "am", "was",
    "were", "be", "been", "do", "does", "did", "i", "you", "he", "she", "it",
    "we", "they", "me", "my", "your", "this", "that", "and", "or", "but", "for",
    "with", "please", "would", "could", "can", "will", "just", "so", "okay",
}
_WORD = re.compile(r"[a-z0-9']+")


def content_tokens(text: str) -> set[str]:
    """Lowercase content words (stop-words removed, single chars dropped)."""
    return {t for t in _WORD.findall((text or "").lower()) if len(t) >= 2 and t not in _STOP}


# --------------------------------------------------------------------------
# Matchers — each returns (relevant_flags, cumulative_secondary) per example.
# relevant_flags[i] : is predicted[i] relevant to the utterance?
# cumulative_secondary[i] : the secondary score using the top-(i+1) board.
# --------------------------------------------------------------------------
class Matcher:
    secondary_label = "secondary"

    def relevance(self, predicted: list[str], ground_truth: str) -> tuple[list[bool], list[float]]:
        raise NotImplementedError


class LexicalMatcher(Matcher):
    secondary_label = "token coverage"

    def relevance(self, predicted, ground_truth):
        target = content_tokens(ground_truth)
        gt_text = (ground_truth or "").lower().strip()
        flags, cum, running = [], [], set()
        for p in predicted:
            flags.append(self._relevant(p, target, gt_text))
            running |= content_tokens(p) & target
            cum.append(len(running) / len(target) if target else 0.0)
        return flags, cum

    @staticmethod
    def _relevant(phrase, target_tokens, target_text):
        if not target_tokens:
            return False
        p = (phrase or "").lower().strip()
        if not p:
            return False
        if p in target_text or target_text in p:
            return True
        return bool(content_tokens(p) & target_tokens)


class SemanticMatcher(Matcher):
    """HYBRID matcher: a phrase is relevant if it is a lexical match OR a strong
    paraphrase (embedding cosine >= threshold). This is strictly fairer than
    lexical (it can only ADD hits), and it credits paraphrases that share no
    surface words ("more coffee please" for "I would like a coffee"). Short
    exact-correct phrases ("yes") score low as embeddings vs a long sentence, so
    the lexical leg is what catches those — the two legs are complementary."""

    secondary_label = "best similarity"

    def __init__(self, threshold: float = 0.5,
                 model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # lazy heavy import
        self._model = SentenceTransformer(model)
        self._thr = threshold

    def relevance(self, predicted, ground_truth):
        if not predicted:
            return [], []
        target = content_tokens(ground_truth)
        gt_text = (ground_truth or "").lower().strip()
        embs = self._model.encode([ground_truth] + list(predicted), normalize_embeddings=True)
        gt_vec, phrase_vecs = embs[0], embs[1:]
        sims = [float(v @ gt_vec) for v in phrase_vecs]   # cosine (vectors normalized)

        flags, cum, running = [], [], -1.0
        for phrase, sim in zip(predicted, sims):
            lexical = LexicalMatcher._relevant(phrase, target, gt_text)
            flags.append(lexical or sim >= self._thr)
            running = max(running, sim)
            cum.append(running)
        return flags, cum


def make_matcher(name: str, threshold: float = 0.5) -> Matcher:
    if name == "semantic":
        return SemanticMatcher(threshold)
    return LexicalMatcher()


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
@dataclass
class ExampleScore:
    hits: dict[int, int]
    secondary: dict[int, float]
    first_rank: int | None


def score_example(predicted: list[str], ground_truth: str, ks: list[int], matcher: Matcher) -> ExampleScore:
    flags, cum = matcher.relevance(predicted, ground_truth)
    first_rank = next((i + 1 for i, f in enumerate(flags) if f), None)
    hits = {k: (1 if first_rank and first_rank <= k else 0) for k in ks}
    secondary = {k: (cum[min(k, len(cum)) - 1] if cum else 0.0) for k in ks}
    return ExampleScore(hits=hits, secondary=secondary, first_rank=first_rank)


@dataclass
class Aggregate:
    n: int
    ks: list[int]
    recall: dict[int, float]
    secondary: dict[int, float]
    secondary_label: str
    mrr: float
    mean_first_rank: float | None


def aggregate(scores: list[ExampleScore], ks: list[int], secondary_label: str) -> Aggregate:
    n = len(scores)
    if n == 0:
        return Aggregate(0, ks, {k: 0.0 for k in ks}, {k: 0.0 for k in ks}, secondary_label, 0.0, None)
    recall = {k: sum(s.hits[k] for s in scores) / n for k in ks}
    secondary = {k: sum(s.secondary[k] for s in scores) / n for k in ks}
    mrr = sum((1.0 / s.first_rank) if s.first_rank else 0.0 for s in scores) / n
    ranks = [s.first_rank for s in scores if s.first_rank]
    mean_first_rank = (sum(ranks) / len(ranks)) if ranks else None
    return Aggregate(n, ks, recall, secondary, secondary_label, mrr, mean_first_rank)
