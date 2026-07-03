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


def _candidate_pool(model, vocab: list[str], scene: str, transcript: str,
                    is_question: bool, k_dyn: int) -> list[str]:
    """Vocab top-N by sigmoid score (+ context terms when CONTEXT_UNION)."""
    from evaluation.phase4 import data as d
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


def student_dynamic_words(model, vocab: list[str], scene: str, transcript: str,
                          is_question: bool, k_dyn: int) -> list[str]:
    """The k_dyn dynamic board words: pooled, unioned, MMR-selected, hedge-capped."""
    from evaluation.embed import embed
    from selection import build_query_vector, mmr_select

    candidates = _candidate_pool(model, vocab, scene, transcript,
                                 is_question, k_dyn)
    if not candidates:
        return []
    cand_vecs = embed(candidates)
    query = build_query_vector(embed, scene, transcript, None)
    hedge_flags = [is_hedge(w) for w in candidates]

    chosen = _reserve_answers(cand_vecs, query, hedge_flags, k_dyn) \
        if is_question else []
    n_hedges = sum(1 for i in chosen if hedge_flags[i])
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
            if n_hedges >= HEDGE_CAP:
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
                  core_words: list[str]) -> list[str]:
    """Full board: core safety words pinned first, MMR-selected dynamic after.

    Same signature as evaluate.py's inline helper so the eval harness (Track 2)
    can import this directly.
    """
    from evaluation.phase4 import data as d
    dynamic = student_dynamic_words(model, vocab, scene, transcript,
                                    is_question, k_dyn)
    return d.assemble_board(core_words, dynamic, k)
