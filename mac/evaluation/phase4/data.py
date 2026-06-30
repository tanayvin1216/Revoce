"""Phase-4 data plumbing — teacher rows -> fixed vocab -> (X embeddings, Y multi-hot).

A teacher row is one context and the diverse board the teacher produced for it:
    {"scene": str, "transcript": str, "board": [word, ...], "is_question": bool}

v2 change: the student learns to predict ONLY dynamic (non-core) words. The 4
always-pinned core words (GUARANTEED_CORE) are excluded from both the vocab and
the multihot targets, then pinned deterministically at inference via assemble_board.
This removes the 0.571 agreement floor that core-forcing created in v1 and frees
model capacity for situation-specific vocabulary.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VOCAB_PATH = HERE / "vocab.json"
SPLITS_PATH = HERE / "splits.json"

# Import here so CORE_SET is available at module level without loading heavy models.
from rank_vocabulary import GUARANTEED_CORE  # noqa: E402 (after stdlib)

CORE_SET: frozenset[str] = frozenset(w.lower() for w in GUARANTEED_CORE)


def load_teacher_rows(path: Path) -> list[dict]:
    """Read a teacher-data JSONL (or a Phase-3 boards.jsonl filtered to diverse)."""
    rows: list[dict] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("strategy") and r["strategy"] != "diverse":
            continue  # boards.jsonl carries both arms; the teacher is the diverse one
        board = [w for w in (r.get("board") or []) if w]
        if not board:
            continue
        rows.append({
            "scene": r.get("scene", ""),
            "transcript": r.get("transcript", ""),
            "board": board,
            "is_question": bool(r.get("is_question")),
            "category": str(r.get("category") or ""),
        })
    return rows


def context_text(row: dict) -> str:
    return f"{row.get('scene', '')} || {row.get('transcript', '')}".strip()


def build_vocab(rows: list[dict], min_freq: int,
                max_size: int) -> tuple[list[str], dict[str, int]]:
    """Dynamic vocabulary: frequent board words EXCLUDING core.

    Core words never enter the vocab — they are pinned deterministically at
    inference by assemble_board, so the student uses all capacity on dynamic words.
    """
    freq: dict[str, int] = {}
    for r in rows:
        for w in r["board"]:
            key = w.strip().lower()
            if key and key not in CORE_SET:
                freq[key] = freq.get(key, 0) + 1

    ranked = sorted((w for w in freq if freq[w] >= min_freq),
                    key=lambda w: (-freq[w], w))
    vocab = ranked[:max_size]
    word2idx = {w: i for i, w in enumerate(vocab)}
    return vocab, word2idx


def save_vocab(vocab: list[str], path: Path = VOCAB_PATH) -> None:
    path.write_text(json.dumps(vocab, indent=2))


def load_vocab(path: Path = VOCAB_PATH) -> tuple[list[str], dict[str, int]]:
    vocab = json.loads(Path(path).read_text())
    return vocab, {w.lower(): i for i, w in enumerate(vocab)}


def multihot(board: list[str], word2idx: dict[str, int]) -> np.ndarray:
    """Multi-hot over the DYNAMIC vocab only; core words are skipped."""
    y = np.zeros(len(word2idx), dtype=np.float32)
    for w in board:
        key = w.strip().lower()
        if key in CORE_SET:
            continue  # core is pinned at inference — not a learning target
        idx = word2idx.get(key)
        if idx is not None:
            y[idx] = 1.0
    return y


def assemble_board(core_words: list[str], dynamic_words: list[str],
                   k: int) -> list[str]:
    """Return exactly k words: core_words first (in order), then dynamic_words.

    core and dynamic are disjoint vocabs by construction (build_vocab excludes core).
    The result is truncated to k if core + dynamic > k.
    """
    out = list(core_words[:k])
    remaining = k - len(out)
    for w in dynamic_words:
        if remaining <= 0:
            break
        out.append(w)
        remaining -= 1
    return out[:k]


def cap_categories(rows: list[dict], max_frac: float = 0.15) -> list[dict]:
    """Downsample any single 'category' value to at most max_frac of total rows.

    Prevents a single over-represented scenario family from biasing the split.
    """
    max_per_cat = max(1, int(round(len(rows) * max_frac)))
    counts: dict[str, int] = {}
    out: list[dict] = []
    for r in rows:
        cat = r.get("category", "")
        n = counts.get(cat, 0)
        if n < max_per_cat:
            counts[cat] = n + 1
            out.append(r)
    return out


def dedup_rows(rows: list[dict]) -> list[dict]:
    """Greedy cosine dedup: drop a row whose context embedding cosine similarity
    to any already-kept row exceeds 0.95.

    Eliminates template twins that would straddle train/test and inflate
    held-out agreement metrics (split-leakage fix). Vectors from embed.py are
    already L2-normalised, so dot product == cosine similarity.
    """
    from evaluation.embed import embed  # lazy: do not load MiniLM at import time
    if len(rows) <= 1:
        return list(rows)
    vecs = embed([context_text(r) for r in rows])  # (N, 384), L2-normalised
    kept_indices: list[int] = []
    kept_vecs: list[np.ndarray] = []
    for i, v in enumerate(vecs):
        if kept_vecs:
            sims = np.stack(kept_vecs) @ v  # (K,) cosine sims to kept set
            if float(sims.max()) > 0.95:
                continue
        kept_indices.append(i)
        kept_vecs.append(v)
    return [rows[i] for i in kept_indices]


def embed_contexts(rows: list[dict]) -> np.ndarray:
    from evaluation.embed import embed  # lazy, shared MiniLM
    return embed([context_text(r) for r in rows])


def make_xy(rows: list[dict], word2idx: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    X = embed_contexts(rows)
    Y = np.stack([multihot(r["board"], word2idx) for r in rows]) if rows \
        else np.zeros((0, len(word2idx)), np.float32)
    return X, Y


def split_indices(n: int, val_frac: float, test_frac: float, seed: int) -> dict:
    """Deterministic shuffled split, persisted so re-runs reuse the same partition."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n).tolist()
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test = perm[:n_test]
    val = perm[n_test:n_test + n_val]
    train = perm[n_test + n_val:]
    return {"train": train, "val": val, "test": test}


def rows_fingerprint(rows: list[dict]) -> str:
    """Hash of the ORDERED context texts. Stored with the split so evaluate.py can
    detect if gather_rows() drifted (a regenerated source file) and refuse to score
    against a stale partition — preventing silent train/test leakage."""
    blob = "\n".join(context_text(r) for r in rows)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def save_splits(splits: dict, path: Path = SPLITS_PATH) -> None:
    path.write_text(json.dumps(splits, indent=2))
