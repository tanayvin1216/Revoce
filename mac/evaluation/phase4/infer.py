"""Student inference — MMR + answer-aware selection (Phase-5 Track 1, RC-1/RC-3).

Replaces the raw ``torch.topk`` board (INVESTIGATION_useless_boards.md RC-1) with
the same selection discipline the teacher seam applies in
rank_vocabulary._select_dynamic:

  1. over-generate a candidate pool (top pool_mult*k_dyn vocab items by sigmoid
     score) and UNION it with content terms extracted from the transcript+scene,
     so an answer literally present in the context can reach the board even when
     it sits below the vocab frequency floor (RC-3-lite);
  2. when the partner asked a question, reserve >=2 slots for non-hedge
     answer-like candidates picked by low-lambda MMR (the teacher's answer quota);
  3. MMR-fill the remaining slots for spread (selection.mmr_select over MiniLM
     embeddings — the exact production utilities);
  4. cap the hedge cluster ("i don't know" family) at <=1 phrase per board.

Pure inference-time policy: the trained model provides relevance candidates, the
selection provides the diversity the raw argmax threw away.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# The teacher's answer quota (rank_vocabulary._select_dynamic) runs
# diversity-heavy MMR (ANSWER_LAM=0.35) — safe there because its candidates are
# ALL function-tagged answers. Our pool is unfiltered (vocab + context terms),
# so a low lambda picks irrelevant-but-different phrases over the actual answer
# (measured: "no, thanks." beat "table"/"aunt" on the seating probe). The
# faithful mirror of "surface the most situation-relevant answers FIRST" is
# relevance order with a near-duplicate guard.
ANSWER_DUP_GUARD = 0.85
FILL_LAM = 0.6  # sweep: 0.5 was worse on every axis (sfp .191, cov .761); 0.6 is the operating point
ANSWER_SLOTS = 2
HEDGE_CAP = 1
# When an open answer is on the board, spend the hedge slot on it instead.
# A hedge cannot contribute dynamic coverage — the gate scores against non-hedge
# moves only (evaluate.held_out_eval) — but it does count against the
# same-function-pair rate. So the hedge slot is the one place an answer can be
# added without displacing a word the coverage gate credits.
HEDGE_CAP_WITH_OPEN = 0

# The hedge cluster lives in ONE shared module (review S3/L1) so the board cap
# and the same-function metric can never disagree about what a hedge is.
from evaluation.hedges import is_hedge, normalize as _normalize  # noqa: E402


@lru_cache(maxsize=1)
def _pool_mult() -> int:
    import yaml
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    return int(cfg["board"].get("pool_mult", 3))


# --- context candidate extraction (RC-3-lite) ---------------------------------
# Copied from rank_vocabulary.topics_from_context / _content_terms
# (mac/rank_vocabulary.py:402-451) rather than imported — PLAN.md Track 1 keeps
# phase4 file ownership disjoint from the seam module. Caps are widened
# (10 terms, 6 from the scene) because this feeds an MMR candidate pool, not a
# prompt: breadth is cheap here, MMR prunes.

_TOPIC_STOPWORDS = frozenset({
    "the", "and", "with", "that", "this", "they", "them", "their", "there",
    "here", "have", "has", "had", "are", "was", "were", "will", "would",
    "your", "you", "yours", "what", "when", "where", "which", "who", "whom",
    "about", "into", "from", "then", "than", "some", "just", "like", "want",
    "need", "going", "gonna", "okay", "yeah", "right", "really", "very",
    "user", "scene", "none", "available", "said", "say", "says", "talking",
})

_MAX_CONTEXT_TERMS = 10
_SCENE_TERM_CAP = 6


def _content_terms(text: str | None) -> list[str]:
    return [
        tok for tok in re.findall(r"[A-Za-z][A-Za-z'-]+", (text or "").lower())
        if len(tok) >= 4 and tok not in _TOPIC_STOPWORDS
    ]


def _context_candidates(scene: str | None, transcript: str | None) -> list[str]:
    """Salient content terms, transcript (the question) first, scene capped."""
    terms: list[str] = []
    seen: set[str] = set()

    def take(tokens: list[str], cap: int) -> None:
        for tok in tokens:
            if len(terms) >= _MAX_CONTEXT_TERMS or cap <= 0:
                return
            if tok not in seen:
                seen.add(tok)
                terms.append(tok)
                cap -= 1

    take(_content_terms(transcript), _MAX_CONTEXT_TERMS)
    take(_content_terms(scene), _SCENE_TERM_CAP)
    return terms


# --- selection -----------------------------------------------------------------

# Review-opus fix 2: the single-token context union injected question-echo nouns
# ("budget", "doorway") whose cosine-to-query is inflated by lexical self-overlap,
# and they displaced real vocab answers (ablation: disabling the union recovered
# dynamic coverage 0.563 -> 0.803). Off until a multi-word answer extractor exists.
CONTEXT_UNION = False

# I-02 ablation: opt in per call; normal callers keep the learned candidate head.
# Frequency counts must be fitted on training labels in this exact vocab order.
# Random callers own the Generator and row traversal; inference never reseeds it.
POOL_SOURCE = "learned"

# RC-3: that multi-word answer extractor now exists (answers.py). It fixes both
# defects that sank CONTEXT_UNION — it emits multi-word spans, and it never
# scores candidates against the question vector. Its picks enter the pool tagged
# as open answers and are reserved by their OWN ranking, not by cosine to a
# query that is 0.7 * the question. Kept as a flag so the ablation stays runnable.
OPEN_ANSWERS = True

# How many of the k_dyn slots the open path may claim, BY QUESTION TYPE —
# extraction confidence is not uniform, so the slot budget should not be either:
#   alternative ("the window seat or the aisle seat?") — the answers are stated
#     in the question, so both conjuncts are near-certainly correct;
#   wh ("where would you like to sit?") — the spans are inferred from the scene
#     and are speculative, so the open path takes one slot and leaves the rest
#     to the vocab, whose general moves are what coverage-of-intent credits.
# Measured: a flat 2 slots for both types costs 0.169 dynamic coverage.
OPEN_SLOTS = {"alternative": 2, "wh": 2}


def _open_answers(scene: str, transcript: str,
                  is_question: bool) -> tuple[list[str], int]:
    """(open answer spans, slot budget) for this context; ([], 0) when N/A."""
    if not (OPEN_ANSWERS and is_question):
        return [], 0
    from evaluation.phase4.answers import answer_candidates, question_kind
    budget = OPEN_SLOTS.get(question_kind(transcript), 0)
    if budget <= 0:
        return [], 0
    return answer_candidates(scene, transcript), budget


def _candidate_pool(model, vocab: list[str], scene: str, transcript: str,
                    is_question: bool, k_dyn: int, *, pool_source: str | None = None,
                    pool_frequency=None, pool_rng=None) -> list[str]:
    """Vocab top-N by sigmoid score (+ context terms when CONTEXT_UNION)."""
    from evaluation.phase4 import data as d
    source = POOL_SOURCE if pool_source is None else pool_source
    pool_n = min(len(vocab), _pool_mult() * k_dyn)
    if source != "learned":
        if source == "cosine":
            from evaluation.embed import embed
            from selection import build_query_vector
            scores = embed(vocab) @ build_query_vector(embed, scene, transcript, None)
            top = np.argsort(-scores, kind="quicksort")[:pool_n]
        elif source == "frequency":
            counts = np.asarray(pool_frequency)
            if (counts.shape != (len(vocab),) or
                    not np.issubdtype(counts.dtype, np.number) or
                    not np.isfinite(counts).all() or (counts < 0).any()):
                raise ValueError("pool_frequency requires finite nonnegative counts in vocab order")
            top = np.argsort(-counts, kind="quicksort")[:pool_n]
        elif source == "vocab_order":
            top = range(pool_n)
        elif source == "full":
            top = range(len(vocab))
        elif source == "random":
            if not isinstance(pool_rng, np.random.Generator):
                raise ValueError("random pool requires an explicit numpy Generator")
            top = pool_rng.choice(len(vocab), size=pool_n, replace=False)
        else:
            raise ValueError(f"Unknown POOL_SOURCE: {source!r}")
        pool = [vocab[int(i)] for i in top]
        if CONTEXT_UNION:
            seen = {_normalize(w) for w in pool}
            for term in _context_candidates(scene, transcript):
                if term not in seen and term not in d.CORE_SET:
                    seen.add(term)
                    pool.append(term)
        return pool
    feats = d.row_features(scene, transcript, is_question)[None, :]
    # Codex P1: torch is imported ONLY for the torch student, so a runtime-only
    # OpenVINO install (no torch) can still serve boards instead of silently
    # falling back to the LLM path when this module fails to import.
    if hasattr(model, "state_dict"):  # torch StudentRanker
        import torch
        with torch.no_grad():
            out = model(torch.from_numpy(feats))[0]
        logits = out.detach().cpu().numpy()
    else:  # OpenVinoStudent (numpy in, numpy out)
        logits = np.asarray(model(feats), dtype=np.float32)[0]
    pool_n = min(len(vocab), _pool_mult() * k_dyn)
    top = np.argsort(-logits)[:pool_n].tolist()  # sigmoid is monotone; rank logits
    pool = [vocab[i] for i in top]
    if CONTEXT_UNION:
        seen = {_normalize(w) for w in pool}
        for term in _context_candidates(scene, transcript):
            if term not in seen and term not in d.CORE_SET:
                seen.add(term)
                pool.append(term)
    return pool


def _reserve_answers(cand_vecs: np.ndarray, query: np.ndarray,
                     hedge_flags: list[bool], k_dyn: int) -> list[int]:
    """Teacher-style answer quota: >=2 non-hedge slots, most relevant first,
    near-duplicates (cosine >= ANSWER_DUP_GUARD) skipped."""
    quota = min(ANSWER_SLOTS, max(k_dyn - 1, 1))
    rel = cand_vecs @ query
    picked: list[int] = []
    for i in np.argsort(-rel):
        if len(picked) >= quota:
            break
        if hedge_flags[i]:
            continue
        if picked and float((cand_vecs[picked] @ cand_vecs[i]).max()) >= ANSWER_DUP_GUARD:
            continue
        picked.append(int(i))
    return picked


def _reserve_open(cand_vecs: np.ndarray, open_idx: list[int],
                  k_dyn: int, budget: int) -> list[int]:
    """Reserve answer slots for extracted open answers, in EXTRACTOR order.

    Deliberately not scored against the query vector: that vector is 0.7 * the
    question (selection._Q_WEIGHTS), and ranking answers by similarity to the
    question is exactly the defect that sank CONTEXT_UNION. answers.py has
    already ranked these against the answer-type prototype, so the only job here
    is to drop near-duplicates ("window seat" vs "the window seat").
    """
    quota = min(budget, ANSWER_SLOTS, max(k_dyn - 1, 1))
    picked: list[int] = []
    for i in open_idx:
        if len(picked) >= quota:
            break
        if picked and float((cand_vecs[picked] @ cand_vecs[i]).max()) >= ANSWER_DUP_GUARD:
            continue
        picked.append(i)
    return picked


def student_dynamic_words(model, vocab: list[str], scene: str, transcript: str,
                          is_question: bool, k_dyn: int, *, pool_source=None,
                          pool_frequency=None, pool_rng=None) -> list[str]:
    """The k_dyn dynamic board words: pooled, unioned, MMR-selected, hedge-capped."""
    from evaluation.embed import embed
    from selection import build_query_vector, mmr_select

    # Preserve the original call shape for existing adapters on the default path.
    options = {} if pool_source is None and pool_frequency is None and pool_rng is None else dict(
        pool_source=pool_source, pool_frequency=pool_frequency, pool_rng=pool_rng)
    candidates = _candidate_pool(model, vocab, scene, transcript,
                                 is_question, k_dyn, **options)
    if not candidates:
        return []
    cand_vecs = embed(candidates)
    query = build_query_vector(embed, scene, transcript, None)
    hedge_flags = [is_hedge(w) for w in candidates]

    # RC-3: union the open answer spans, remembering which pool slots they own so
    # they can be reserved by extractor rank instead of query cosine. answers.py
    # decides whether this context has open answers at all; when it returns
    # nothing the board is exactly the closed-vocab board it was before.
    open_start = len(candidates)
    open_idx: list[int] = []
    open_spans, open_budget = _open_answers(scene, transcript, is_question)
    seen = {_normalize(w) for w in candidates}
    new: list[str] = []
    for span in open_spans:
        if _normalize(span) not in seen:
            seen.add(_normalize(span))
            new.append(span)
    if new:
        candidates.extend(new)
        cand_vecs = np.vstack([cand_vecs, embed(new)])
        hedge_flags.extend(is_hedge(w) for w in new)
        open_idx = list(range(open_start, len(candidates)))

    if not is_question:
        chosen: list[int] = []
    elif open_idx:
        chosen = _reserve_open(cand_vecs, open_idx, k_dyn, open_budget)
        # a single extracted answer still leaves a quota slot for the vocab path
        if len(chosen) < min(ANSWER_SLOTS, max(k_dyn - 1, 1)):
            for i in _reserve_answers(cand_vecs, query, hedge_flags, k_dyn):
                if i not in chosen and len(chosen) < ANSWER_SLOTS:
                    chosen.append(i)
    else:
        chosen = _reserve_answers(cand_vecs, query, hedge_flags, k_dyn)
    n_hedges = sum(1 for i in chosen if hedge_flags[i])
    hedge_cap = HEDGE_CAP_WITH_OPEN if open_idx else HEDGE_CAP
    # Function-aware fill (review round 3): the teacher passes per-candidate
    # communication-function tags to mmr_select so the board SPANS functions
    # (rank_vocabulary.py:540); omitting them here left the all-vocab pool
    # stacking same-function filler (sfp 0.197 vs teacher 0.168).
    from evaluation.phase3 import taxonomy
    _, functions = taxonomy.classify_functions(cand_vecs)
    order = mmr_select(cand_vecs, query, len(candidates), FILL_LAM, functions)
    for i in order:
        if len(chosen) >= k_dyn:
            break
        if i in chosen:
            continue
        if hedge_flags[i]:
            if n_hedges >= hedge_cap:
                continue
            n_hedges += 1
        chosen.append(i)
    for i in order:  # backfill if the hedge cap starved a tiny pool
        if len(chosen) >= k_dyn:
            break
        if i not in chosen:
            chosen.append(i)
    return [candidates[i] for i in chosen[:k_dyn]]


def student_board(model, vocab: list[str], scene: str, transcript: str,
                  is_question: bool, k: int, k_dyn: int,
                  core_words: list[str], *, pool_source=None,
                  pool_frequency=None, pool_rng=None) -> list[str]:
    """Full board: core safety words pinned first, MMR-selected dynamic after.

    Same signature as evaluate.py's inline helper so the eval harness (Track 2)
    can import this directly.
    """
    from evaluation.phase4 import data as d
    dynamic = student_dynamic_words(model, vocab, scene, transcript,
                                    is_question, k_dyn, pool_source=pool_source,
                                    pool_frequency=pool_frequency, pool_rng=pool_rng)
    return d.assemble_board(core_words, dynamic, k)
