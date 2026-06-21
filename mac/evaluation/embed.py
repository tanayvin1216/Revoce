"""
Shared sentence-embedding helper — ONE MiniLM instance for the whole eval/runtime.

Both the production selector (selection.py) and every eval metric reach embeddings
through here, so the all-MiniLM-L6-v2 model is loaded exactly once per process.
Vectors are L2-normalized, so a dot product IS cosine similarity.
"""

from __future__ import annotations

import numpy as np

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # heavy, lazy
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Return an (N, 384) array of L2-normalized embeddings (cosine = dot)."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    vecs = _get_model().encode(list(texts), normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine for normalized inputs: (Na,Nb)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    return a @ b.T
