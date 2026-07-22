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


def build_phrase_bank(rows: list[dict], dedup_tau: float,
                      min_freq: int = 1) -> tuple[list[str], dict[str, str]]:
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
    kept, canonical = dedup_vocab(ranked, dedup_tau)
    return kept, canonical


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
