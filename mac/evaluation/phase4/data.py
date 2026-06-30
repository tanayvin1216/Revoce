"""Phase-4 data plumbing — teacher rows -> fixed vocab -> (X embeddings, Y multi-hot).

A teacher row is one context and the diverse board the teacher produced for it:
    {"scene": str, "transcript": str, "board": [word, ...], "is_question": bool}

The student learns context_embedding -> multi-hot(board over the fixed vocab). The
context embedding is the shared MiniLM (evaluation/embed.py) of "<scene> || <transcript>"
— the exact same encoder the selector and every Phase-3 metric already use, so the
student sees the world in the same space its teacher's selection was scored in.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VOCAB_PATH = HERE / "vocab.json"
SPLITS_PATH = HERE / "splits.json"


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
            "scene": r.get("scene", ""), "transcript": r.get("transcript", ""),
            "board": board, "is_question": bool(r.get("is_question")),
        })
    return rows


def context_text(row: dict) -> str:
    return f"{row.get('scene', '')} || {row.get('transcript', '')}".strip()


def build_vocab(rows: list[dict], min_freq: int, max_size: int,
                core_words: list[str]) -> tuple[list[str], dict[str, int]]:
    """Fixed candidate vocabulary: the always-on core words first (always kept),
    then the most frequent remaining board words at or above min_freq."""
    freq: dict[str, int] = {}
    for r in rows:
        for w in r["board"]:
            key = w.strip()
            if key:
                freq[key.lower()] = freq.get(key.lower(), 0) + 1

    vocab: list[str] = []
    seen: set[str] = set()
    for w in core_words:                       # core is force-included, in order
        if w.lower() not in seen:
            seen.add(w.lower())
            vocab.append(w)
    ranked = sorted((w for w in freq if w not in seen),
                    key=lambda w: (-freq[w], w))
    for w in ranked:
        if len(vocab) >= max_size:
            break
        if freq[w] >= min_freq:
            seen.add(w)
            vocab.append(w)
    word2idx = {w.lower(): i for i, w in enumerate(vocab)}
    return vocab, word2idx


def save_vocab(vocab: list[str], path: Path = VOCAB_PATH) -> None:
    path.write_text(json.dumps(vocab, indent=2))


def load_vocab(path: Path = VOCAB_PATH) -> tuple[list[str], dict[str, int]]:
    vocab = json.loads(Path(path).read_text())
    return vocab, {w.lower(): i for i, w in enumerate(vocab)}


def multihot(board: list[str], word2idx: dict[str, int]) -> np.ndarray:
    y = np.zeros(len(word2idx), dtype=np.float32)
    for w in board:
        idx = word2idx.get(w.strip().lower())
        if idx is not None:
            y[idx] = 1.0
    return y


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
