"""Serving path for the v3 bi-encoder student.

Same shape as infer.py — pool, then MMR with an answer quota and a hedge cap —
and it deliberately reuses infer.py's selection helpers rather than forking
them, so board composition policy stays defined in exactly one place.

What changes is where candidates come from. v2 could only ever propose one of
94 vocabulary columns, so `CONTEXT_UNION` — appending nouns lifted from the
transcript — was a hack that had to be switched OFF: those terms had no score,
so they could only be bolted onto the pool, and the review ablation measured
them costing 0.24 dynamic coverage by crowding out scored candidates.

Here they have a score. A phrase pulled out of the live transcript, or a name
from the on-device personal-context store, is embedded and run through the same
phrase tower as everything in the bank, so it competes on the same axis instead
of jumping the queue. That is what makes "aunt" reachable: not a special case,
just a candidate the model is finally able to rank.

Cost note: the phrase tower runs ONCE offline over the bank (BankIndex caches
the projections). Serving is one context-tower pass plus a matmul, and the only
per-board phrase-tower work is over the handful of live context terms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evaluation.hedges import is_hedge
from evaluation.phase4.infer import (
    FILL_LAM,
    HEDGE_CAP,
    _content_terms,
    _normalize,
    _pool_mult,
    _reserve_answers,
)

# Cap on live context candidates so a long transcript can't flood the pool.
MAX_LIVE_CANDIDATES = 12


@dataclass
class BankIndex:
    """The phrase bank with its projections precomputed. Build once, reuse."""

    phrases: list[str]
    proj: np.ndarray            # (N, proj_dim)

    @property
    def size(self) -> int:
        return len(self.phrases)


def build_bank_index(model, phrases: list[str]) -> BankIndex:
    """Project the whole bank through the phrase tower, once."""
    from evaluation.phase4.bank import phrase_features
    return BankIndex(list(phrases), _project_phrases(model, phrase_features(phrases)))


def _project_phrases(model, feats: np.ndarray) -> np.ndarray:
    if hasattr(model, "state_dict"):          # torch BiEncoderRanker
        import torch
        with torch.no_grad():
            return model.encode_phrase(torch.from_numpy(feats)).numpy()
    return np.asarray(model.encode_phrase(feats), dtype=np.float32)


def _project_context(model, feats: np.ndarray) -> np.ndarray:
    if hasattr(model, "state_dict"):
        import torch
        with torch.no_grad():
            return model.encode_context(torch.from_numpy(feats)).numpy()
    return np.asarray(model.encode_context(feats), dtype=np.float32)


def live_candidates(scene: str, transcript: str,
                    personal_context: dict | None = None) -> list[str]:
    """Candidate phrases that exist only in THIS moment.

    Content terms from what was just said and seen, plus names and routines from
    the personal-context store. These are exactly the items a frequency floor
    can never admit, which is why v2 could not represent them at all.
    """
    out: list[str] = []
    seen: set[str] = set()

    for term in _content_terms(transcript) + _content_terms(scene):
        key = _normalize(term)
        if key and key not in seen:
            seen.add(key)
            out.append(term)

    for value in _personal_terms(personal_context):
        key = _normalize(value)
        if key and key not in seen:
            seen.add(key)
            out.append(value)

    return out[:MAX_LIVE_CANDIDATES]


def _personal_terms(profile: dict | None) -> list[str]:
    """Names and routine words from the on-device profile. Never leaves the box."""
    if not profile:
        return []
    out: list[str] = []
    people = profile.get("people") or profile.get("relationships") or []
    if isinstance(people, dict):
        people = list(people.values())
    for p in people:
        if isinstance(p, dict):
            name = p.get("name") or p.get("relation")
        else:
            name = p
        if isinstance(name, str) and name.strip():
            out.append(name.strip())
    return out


@dataclass
class Selection:
    """v3 selection knobs. Defaults come from config.yaml:selection_v3.

    rel_source decides what MMR treats as relevance:
      "cosine" — cosine to the situation vector (what v2 does; the model's
                 scores are used only for the pool cut and then discarded)
      "model"  — the bi-encoder's own scores, min-max scaled
      "blend"  — mean of the two

    See config.yaml for the validation sweep behind the defaults.
    """

    lam: float | None = None
    pool_mult: int | None = None
    rel_source: str | None = None

    def __post_init__(self) -> None:
        cfg = _selection_cfg()
        if self.lam is None:
            self.lam = float(cfg.get("lam", FILL_LAM))
        if self.pool_mult is None:
            self.pool_mult = int(cfg.get("pool_mult", _pool_mult()))
        if self.rel_source is None:
            self.rel_source = str(cfg.get("rel_source", "cosine"))

    def pool_size(self, k_dyn: int, n: int) -> int:
        return min(n, self.pool_mult * k_dyn)


def _selection_cfg() -> dict:
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load(
        (Path(__file__).resolve().parent / "config.yaml").read_text())
    return cfg.get("selection_v3") or {}


def _scale01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi - lo < 1e-9 else (x - lo) / (hi - lo)


def student_dynamic_words(model, bank: BankIndex, scene: str, transcript: str,
                          is_question: bool, k_dyn: int,
                          personal_context: dict | None = None,
                          sel: "Selection | None" = None) -> list[str]:
    """The k_dyn dynamic words: scored pool (bank + live), MMR, hedge-capped."""
    from evaluation.embed import embed
    from evaluation.phase4 import data as d
    from selection import build_query_vector, mmr_select

    sel = sel or Selection()

    ctx_vec = _project_context(
        model, d.row_features(scene, transcript, is_question)[None, :])[0]

    phrases = list(bank.phrases)
    proj = bank.proj

    # Live candidates go through the SAME phrase tower, so they are ranked
    # against the bank rather than appended past it.
    live = [w for w in live_candidates(scene, transcript, personal_context)
            if _normalize(w) not in d.CORE_SET]
    if live:
        live_proj = _project_phrases(model, np.asarray(embed(live), dtype=np.float32))
        phrases = phrases + live
        proj = np.concatenate([proj, live_proj], axis=0)

    if not phrases:
        return []

    scores = proj @ ctx_vec
    pool_n = sel.pool_size(k_dyn, len(phrases))
    top = np.argsort(-scores)[:pool_n].tolist()
    candidates = [phrases[i] for i in top]

    cand_vecs = embed(candidates)
    query = build_query_vector(embed, scene, transcript, None)
    hedge_flags = [is_hedge(w) for w in candidates]

    chosen = _reserve_answers(cand_vecs, query, hedge_flags, k_dyn) \
        if is_question else []
    n_hedges = sum(1 for i in chosen if hedge_flags[i])

    # The model ranked 1959 candidates down to this pool; throwing that ranking
    # away and re-scoring on raw cosine is what the coverage diagnostic caught
    # (19 of 20 misses had a satisfying phrase already IN the pool).
    rel = None
    if sel.rel_source != "cosine":
        model_rel = _scale01(scores[top])
        if sel.rel_source == "model":
            rel = model_rel
        elif sel.rel_source == "blend":
            rel = 0.5 * model_rel + 0.5 * _scale01(cand_vecs @ query)
        else:
            raise ValueError(f"unknown rel_source {sel.rel_source!r}")

    from evaluation.phase3 import taxonomy
    _, functions = taxonomy.classify_functions(cand_vecs)
    order = mmr_select(cand_vecs, query, len(candidates), sel.lam, functions,
                       rel=rel)

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
    for i in order:                       # backfill if the cap starved the pool
        if len(chosen) >= k_dyn:
            break
        if i not in chosen:
            chosen.append(i)

    return [candidates[i] for i in chosen[:k_dyn]]


def student_board(model, bank: BankIndex, scene: str, transcript: str,
                  is_question: bool, k: int, k_dyn: int,
                  core_words: list[str],
                  personal_context: dict | None = None,
                  sel: "Selection | None" = None) -> list[str]:
    """Full board: core safety words pinned first, selected dynamic after."""
    from evaluation.phase4 import data as d
    dynamic = student_dynamic_words(model, bank, scene, transcript,
                                    is_question, k_dyn, personal_context, sel)
    return d.assemble_board(core_words, dynamic, k)
