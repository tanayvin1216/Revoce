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
_backend_loaded: str | None = None  # which runtime the singleton is


def _backend() -> str:
    """'torch' (sentence-transformers, default) or 'openvino' (MiniLM IR).

    VOICELINK_EMBED_BACKEND wins; otherwise follow VOICELINK_STUDENT_BACKEND so
    the Intel student path is OpenVINO end to end with ONE flag. Read at call
    time (not import time) so tests can monkeypatch the environment.
    """
    import os
    explicit = os.environ.get("VOICELINK_EMBED_BACKEND")
    if explicit:
        return explicit
    return "openvino" if os.environ.get("VOICELINK_STUDENT_BACKEND") == "openvino" else "torch"


def _get_model():
    global _model, _backend_loaded
    want = _backend()
    if _model is None or _backend_loaded != want:
        if want == "openvino":
            try:
                from intel import OpenVinoEmbedder  # torch-free
                _model = OpenVinoEmbedder()
            except Exception as e:  # IR missing / openvino absent -> torch
                print(f"[embed] OpenVINO MiniLM unavailable ({e}); using torch")
                want = "torch"
        if want == "torch":
            from sentence_transformers import SentenceTransformer  # heavy, lazy
            _model = SentenceTransformer(_MODEL_NAME)
        _backend_loaded = want
    return _model


def backend_in_use() -> str | None:
    """Which runtime actually produced embeddings so far (None before first call)."""
    return _backend_loaded


def embed(texts: list[str]) -> np.ndarray:
    """Return an (N, 384) array of L2-normalized embeddings (cosine = dot)."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    m = _get_model()
    if hasattr(m, "encode"):  # sentence-transformers
        vecs = m.encode(list(texts), normalize_embeddings=True)
    else:                     # OpenVinoEmbedder (already normalised)
        vecs = m(list(texts))
    return np.asarray(vecs, dtype=np.float32)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine for normalized inputs: (Na,Nb)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    return a @ b.T
