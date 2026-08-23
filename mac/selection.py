"""
Diversity selection for the AAC board — the SELECT half of the pool/select split.

rank_vocabulary() over-generates a pool of candidate phrases, then this module
picks the k that best COVER the conversational space: relevant to the situation
yet spread across distinct communicative directions, so whichever way the user
wants to go, a launch point is on the board.

Algorithm: greedy Maximal Marginal Relevance (MMR).
    score(i) = lam * rel[i]  -  (1 - lam) * max_{j in selected} cos(cand_i, cand_j)
- rel[i]   = cosine(query, candidate_i)   (fit to the situation; no ground truth)
- penalty  = similarity to the most-similar already-picked item (kills near-dups)
- lam      = 1.0 -> pure relevance (mode-seeking, old behavior)
             0.0 -> pure spread.  Default 0.6 (the knee).

Pure numpy over MiniLM vectors: sub-millisecond on a ~20-item pool, so the board
can be re-selected for free without another model call (non-blocking guarantee).
"""

from __future__ import annotations

import numpy as np

DEFAULT_LAMBDA = 0.6
TIE_EPS = 0.02  # MMR scores within this -> break by unmet communication function

# Query weights: what the user was ASKED dominates; the scene is only background
# context. (A vivid scene must not pull the board away from the actual question —
# e.g. "where do I sit?" in a restaurant must not become reservation/menu vocab.)
_Q_WEIGHTS = {"transcript": 0.7, "scene": 0.2, "context": 0.1}


def build_query_vector(embed_fn, scene, transcript, context) -> np.ndarray:
    """One unit query vector summarizing the situation, weighted toward the
    question the user must answer (active_question / last utterance / transcript)."""
    ctx = context or {}
    personal = ctx.get("personal_context") or {}
    blurb = personal.get("about") or personal.get("name") or ""
    if "active_question" in ctx:
        # Live path: the held question alone drives the query — later chatter in
        # the rolling transcript must not dilute it. No held question -> transcript.
        question_text = (ctx.get("active_question") or transcript or "").strip()
    else:
        # Direct/console call: the transcript is the question.
        question_text = (transcript or "").strip()
    parts = {
        "transcript": question_text,
        "scene": scene or "",
        "context": str(blurb),
    }
    vec = np.zeros(384, dtype=np.float32)
    any_text = False
    for key, text in parts.items():
        if text and text.strip():
            vec = vec + _Q_WEIGHTS[key] * embed_fn([text])[0]
            any_text = True
    if not any_text:
        return np.zeros(384, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def mmr_select(
    cand_vecs: np.ndarray,
    query_vec: np.ndarray,
    k: int,
    lam: float = DEFAULT_LAMBDA,
    functions: list[str] | None = None,
    rel: np.ndarray | None = None,
) -> list[int]:
    """Return indices of the k MMR-selected candidates, best-first.

    cand_vecs : (N, 384) normalized candidate embeddings.
    query_vec : (384,) normalized query, or zeros (then relevance is uniform).
    functions : optional per-candidate communication-function tag for tie-break.
    rel       : optional (N,) relevance override, already scaled to [0, 1]. When
                given, it replaces cosine-to-query as the relevance term. The v3
                bi-encoder uses this to feed its OWN learned scores into MMR
                instead of having them discarded after the pool cut; leaving it
                None preserves the v2 behaviour exactly.
    """
    n = cand_vecs.shape[0]
    if n == 0:
        return []
    k = min(k, n)

    if rel is not None:
        rel = np.clip(np.asarray(rel, dtype=np.float32), 0.0, 1.0)
        if rel.shape[0] != n:
            raise ValueError(f"rel has {rel.shape[0]} entries, expected {n}")
    elif query_vec is not None and np.linalg.norm(query_vec) > 0:
        rel = cand_vecs @ query_vec            # cosine to situation
        rel = np.clip(rel, 0.0, 1.0)
    else:
        rel = np.zeros(n, dtype=np.float32)    # no signal -> diversity drives

    sim = cand_vecs @ cand_vecs.T              # candidate-candidate cosine
    selected: list[int] = [int(np.argmax(rel))]
    used_functions = {functions[selected[0]]} if functions else set()

    while len(selected) < k:
        penalty = sim[:, selected].max(axis=1)              # nearest picked
        score = lam * rel - (1.0 - lam) * penalty
        score[selected] = -np.inf

        best = int(np.argmax(score))
        # Function-coverage tie-break: among near-ties, prefer an unseen function.
        if functions:
            top = float(score[best])
            ties = [i for i in range(n) if score[i] >= top - TIE_EPS and i not in selected]
            unseen = [i for i in ties if functions[i] not in used_functions]
            if unseen:
                best = max(unseen, key=lambda i: score[i])
            used_functions.add(functions[best])
        selected.append(best)

    return selected


def intra_list_diversity(vecs: np.ndarray) -> float:
    """1 - mean pairwise cosine over a board (higher = more spread)."""
    n = vecs.shape[0]
    if n < 2:
        return 0.0
    sims = vecs @ vecs.T
    triu = sims[np.triu_indices(n, k=1)]
    return float(1.0 - triu.mean())
