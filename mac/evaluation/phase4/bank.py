"""The phrase bank — the open candidate set the v3 bi-encoder scores against.

This is what replaces `vocab.json`. The v2 vocabulary was a closed list of
output columns, so an item had to clear `min_freq` to exist at all. The bank has
no such constraint: it is simply every dynamic phrase the teacher ever produced,
plus whatever else we want scorable, because the model scores (context, phrase)
pairs rather than owning a column per phrase.

Two consequences worth being precise about:

  * NO FREQUENCY FLOOR. A phrase the teacher said once is in the bank. Whether
    it ever reaches a board is decided at inference by its score, which is the
    correct place for that decision.
  * THE BANK IS EXTENSIBLE AT RUNTIME. Phrases extracted from the live
    transcript, or names from the on-device personal-context store, can be
    embedded and scored with the same weights. Nothing needs retraining. That
    is the whole point of moving the vocabulary out of the architecture.

Hedge dedup still applies. Collapsing "I don't know" and its paraphrases is a
board-quality fix (RC-2), not a capacity fix, and it is just as necessary here —
without it the bank offers seven ways to say nothing and MMR happily spreads
across them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from evaluation.phase4.data import CORE_SET, dedup_vocab

BANK_PATH = Path(__file__).resolve().parent / "phrase_bank.json"


def dedup_bank(ranked: list[str], tau: float,
               linkage: str = "complete") -> tuple[list[str], dict[str, str]]:
    """Semantic dedup for the OPEN bank. Same contract as data.dedup_vocab.

    Why this exists rather than reusing dedup_vocab: that function uses
    SINGLE-link connected components, and it says why — the hedge paraphrases
    only unite transitively ("i don't know" ~ "i am not sure" is 0.784, below
    any workable tau, but they link through "i'm not sure"). That is the right
    call for a 116-word closed vocabulary.

    It does not survive an open bank. Single-link chaining grows with the number
    of phrases: A~B~C merges A and C however dissimilar they are, and the more
    phrases there are the more chains there are. Swept on the real corpus
    (1358 distinct phrases), single-link tripped the oversize guard at EVERY tau
    from 0.80 to 0.90 — 13, 6, 3 and 2 oversized components respectively. So
    raising tau is a treadmill, not a fix, and by 0.95 the hedges stop merging
    at all, which is the failure dedup exists to prevent.

    COMPLETE linkage (the default) ends the treadmill by construction: a phrase
    joins a cluster only if it clears tau against EVERY member, so a chain can
    never drag in something the far end of the cluster disagrees with. It is the
    only linkage that passed at 0.85, and it collapsed the hedge cluster hardest
    there (14 variants -> 7, against 8 for average at 0.90). Because the
    constraint is on every pair rather than on the graph's connectivity, the
    setting stays valid as the bank grows instead of needing a re-sweep.

    "average" (mean similarity to the cluster) is kept as the intermediate
    option; on this corpus it needed tau=0.90 to pass.
    """
    from evaluation.embed import embed

    if len(ranked) <= 1:
        return list(ranked), {}

    vecs = np.asarray(embed(ranked), dtype=np.float32)
    # Greedy agglomeration in frequency order: the first-ranked member of a
    # cluster is its representative, matching dedup_vocab's rule.
    clusters: list[list[int]] = []
    for i in range(len(ranked)):
        placed = False
        for c in clusters:
            sims = vecs[c] @ vecs[i]
            agg = float(sims.mean()) if linkage == "average" else float(sims.min())
            if agg >= tau:
                c.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])

    kept = [ranked[c[0]] for c in clusters]
    canonical = {ranked[j]: ranked[c[0]] for c in clusters for j in c[1:]}

    _audit(canonical)
    return kept, canonical


def _audit(canonical: dict[str, str]) -> None:
    """Same loud guard as dedup_vocab: log non-hedge merges, fail on oversize."""
    from evaluation.hedges import is_hedge

    sizes: dict[str, int] = {}
    for dropped, rep in canonical.items():
        sizes[rep] = sizes.get(rep, 1) + 1
        if not (is_hedge(dropped) and is_hedge(rep)):
            print(f"[dedup_bank] non-hedge merge: {dropped!r} -> {rep!r} (audit)")
    oversized = {rep: n for rep, n in sizes.items()
                 if n > 4 and not is_hedge(rep)}
    if oversized:
        raise ValueError(
            f"dedup_bank: non-hedge component(s) exceed size 4: {oversized} — "
            f"raise bank_dedup_tau or switch linkage before retraining")


def build_phrase_bank(rows: list[dict], dedup_tau: float,
                      min_freq: int = 1,
                      linkage: str = "complete") -> tuple[list[str], dict[str, str]]:
    """Every dynamic phrase in the teacher corpus, hedge-deduped.

    min_freq defaults to 1 — i.e. no floor. It stays a parameter only so an
    ablation can put the floor back and measure what it costs.
    """
    freq: dict[str, int] = {}
    for r in rows:
        for w in r["board"]:
            key = w.strip().lower()
            if key and key not in CORE_SET:
                freq[key] = freq.get(key, 0) + 1

    ranked = sorted((w for w in freq if freq[w] >= min_freq),
                    key=lambda w: (-freq[w], w))
    return dedup_bank(ranked, dedup_tau, linkage)


def bank_sha(bank: list[str]) -> str:
    """Stable fingerprint. Same contract as vocab_sha: the served artifact must
    be able to prove which bank it was built against."""
    return hashlib.sha256("\n".join(bank).encode()).hexdigest()[:16]


def save_bank(bank: list[str], canonical: dict[str, str],
              path: Path = BANK_PATH) -> None:
    Path(path).write_text(
        json.dumps({"phrases": bank, "canonical": canonical,
                    "sha": bank_sha(bank)}, indent=2),
        encoding="utf-8",
    )


def load_bank(path: Path = BANK_PATH) -> tuple[list[str], dict[str, str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["phrases"], {k.lower(): v.lower()
                             for k, v in data.get("canonical", {}).items()}


def phrase_features(phrases: list[str]) -> np.ndarray:
    """(N, 384) MiniLM embeddings for the bank. Shared embedder, as everywhere."""
    from evaluation.embed import embed
    return np.asarray(embed(list(phrases)), dtype=np.float32)


def positives_for_row(row: dict, canonical: dict[str, str],
                      phrase2idx: dict[str, int]) -> list[int]:
    """Bank indices of the dynamic phrases on this teacher board.

    canonical redirects a dedup-dropped variant to the representative that
    survived, so the label mass follows the merge instead of being silently
    dropped — same reasoning as multihot() in the v2 pipeline.
    """
    out: list[int] = []
    for w in row["board"]:
        key = w.strip().lower()
        if not key or key in CORE_SET:
            continue
        key = canonical.get(key, key)
        idx = phrase2idx.get(key)
        if idx is not None and idx not in out:
            out.append(idx)
    return out
