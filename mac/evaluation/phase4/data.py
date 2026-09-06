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
import os

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# Artifact directory switch (2026-09-05 audit). Default "" keeps every path exactly
# as shipped. VOICELINK_PHASE4_ARTIFACTS=clean puts checkpoints, vocab, bank,
# splits and analysis outputs under evaluation/phase4/clean/ so a re-trained
# student can be evaluated without overwriting the served one.
ART = HERE / os.environ.get("VOICELINK_PHASE4_ARTIFACTS", "")
ART.mkdir(parents=True, exist_ok=True)
VOCAB_PATH = ART / "vocab.json"
SPLITS_PATH = ART / "splits.json"
CANONICAL_PATH = ART / "canonical.json"

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


def dedup_vocab(ranked: list[str], tau: float) -> tuple[list[str], dict[str, str]]:
    """Semantic dedup of a frequency-ranked vocab (RC-2 fix).

    Single-link connected components of the cosine>=tau graph — NOT greedy
    against-kept-only, because the 7 hedge paraphrases only unite transitively
    ("i don't know"~"i am not sure" = 0.784 < 0.80, but both link through
    "i'm not sure" at 0.841/0.943; see impl-track1.md). Each component keeps its
    highest-frequency (first-ranked) member; the canonical map sends every
    dropped member to its component representative so multihot can redirect the
    label mass instead of losing it.
    """
    from evaluation.embed import embed  # lazy: do not load MiniLM at import time
    if len(ranked) <= 1:
        return list(ranked), {}
    vecs = embed(ranked)
    linked = (vecs @ vecs.T) >= tau            # (N, N) adjacency, includes self
    component = list(range(len(ranked)))       # root = lowest rank in component

    def find(i: int) -> int:
        while component[i] != i:
            component[i] = component[component[i]]
            i = component[i]
        return i

    for i in range(len(ranked)):
        for j in range(i + 1, len(ranked)):
            if linked[i, j]:
                ri, rj = find(i), find(j)
                if ri != rj:
                    component[max(ri, rj)] = min(ri, rj)

    kept = [w for i, w in enumerate(ranked) if find(i) == i]
    canonical = {w: ranked[find(i)] for i, w in enumerate(ranked) if find(i) != i}

    # Review S4: single-link chaining can transitively swallow distinct answers
    # ("...nervous." -> "...anxious." is a real size-2 merge). Guard both ways:
    # every non-hedge merge is logged for manual audit, and an oversized
    # component (>4) fails loudly instead of silently eating vocabulary.
    from evaluation.hedges import is_hedge
    sizes: dict[str, int] = {}
    for dropped, rep in canonical.items():
        sizes[rep] = sizes.get(rep, 1) + 1
        if not (is_hedge(dropped) and is_hedge(rep)):
            print(f"[dedup_vocab] non-hedge merge: {dropped!r} -> {rep!r} (audit)")
    oversized = {rep: n for rep, n in sizes.items()
                 if n > 4 and not is_hedge(rep)}
    if oversized:
        raise ValueError(
            f"dedup_vocab: non-hedge component(s) exceed size 4: {oversized} — "
            f"lower tau or whitelist before retraining")
    return kept, canonical


def vocab_sha(vocab: list[str]) -> str:
    """Order-sensitive fingerprint of the vocab — stamped into the checkpoint
    and IR sidecars so a stale/reordered artifact fails loudly at load instead
    of silently serving wrong words (review S5)."""
    return hashlib.sha256("\n".join(vocab).encode()).hexdigest()[:16]


def save_vocab(vocab: list[str], path: Path = VOCAB_PATH) -> None:
    path.write_text(json.dumps(vocab, indent=2))


def save_canonical(canonical: dict[str, str], path: Path = CANONICAL_PATH) -> None:
    path.write_text(json.dumps(canonical, indent=2, sort_keys=True))


def load_canonical(path: Path = CANONICAL_PATH) -> dict[str, str]:
    if not Path(path).exists():
        return {}
    return {k.lower(): v.lower() for k, v in json.loads(Path(path).read_text()).items()}


def load_vocab(path: Path = VOCAB_PATH) -> tuple[list[str], dict[str, int]]:
    vocab = json.loads(Path(path).read_text())
    return vocab, {w.lower(): i for i, w in enumerate(vocab)}


def multihot(board: list[str], word2idx: dict[str, int],
             canonical: dict[str, str] | None = None) -> np.ndarray:
    """Multi-hot over the DYNAMIC vocab only; core words are skipped.

    canonical redirects dedup-dropped variants ("i am not sure") to their kept
    representative ("i don't know") so the label mass survives the vocab dedup.
    """
    y = np.zeros(len(word2idx), dtype=np.float32)
    for w in board:
        key = w.strip().lower()
        if key in CORE_SET:
            continue  # core is pinned at inference — not a learning target
        if canonical:
            key = canonical.get(key, key)
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


# v2.2 RICHER INPUT. v1/v2 fed ONE pooled MiniLM embedding of "<scene> || <transcript>"
# (384-d); with more data its dynamic_overlap got WORSE, which told us that single
# pooled vector cannot separate the teacher's specific dynamic choice. So the input is
# now the scene and transcript embedded SEPARATELY and concatenated (2*384=768), plus
# the is_question flag (1) -> 769-d. This gives the head a scene axis and a speech axis
# it can weight independently, instead of one averaged blur.
CONTEXT_DIM = 769


def row_features(scene: str, transcript: str, is_question: bool) -> np.ndarray:
    """The 769-d input for ONE context (used identically at train and inference)."""
    from evaluation.embed import embed
    v = embed([scene or "", transcript or ""])           # (2, 384)
    flag = np.array([1.0 if is_question else 0.0], dtype=np.float32)
    return np.concatenate([v[0], v[1], flag]).astype(np.float32)


def context_features(rows: list[dict]) -> np.ndarray:
    """Batched 769-d features for many rows (separate scene/transcript embeds + flag)."""
    from evaluation.embed import embed  # lazy, shared MiniLM
    if not rows:
        return np.zeros((0, CONTEXT_DIM), np.float32)
    # ONE batched call for scene+transcript: two calls cost two full encoder
    # round-trips (~5.5 ms each on OpenVINO) for what is a 2-row batch.
    both = embed([r.get("scene", "") for r in rows] +
                 [r.get("transcript", "") for r in rows])    # (2N, 384)
    scenes, trans = both[:len(rows)], both[len(rows):]
    flags = np.array([[1.0 if r.get("is_question") else 0.0] for r in rows],
                     dtype=np.float32)
    return np.concatenate([scenes, trans, flags], axis=1).astype(np.float32)


def make_xy(rows: list[dict], word2idx: dict[str, int],
            canonical: dict[str, str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    X = context_features(rows)
    Y = np.stack([multihot(r["board"], word2idx, canonical) for r in rows]) if rows \
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
