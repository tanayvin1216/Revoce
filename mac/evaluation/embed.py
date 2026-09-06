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


_ov_failed = False  # remember an OpenVINO load failure so we warn ONCE, not per call


def _strict() -> bool:
    """VOICELINK_EMBED_STRICT=1 turns the permissive (warned) torch fallback into a
    hard error. Off by default so the live device keeps serving boards; benchmarks
    and quality evaluations that LABEL a run by its embedding backend must set it,
    otherwise a missing MiniLM IR silently produces a torch row under an OpenVINO
    label (I-12 / audit C-10: exactly what happened to the ov-full row)."""
    import os
    return os.environ.get("VOICELINK_EMBED_STRICT", "").strip() in ("1", "true", "yes")


class EmbedBackendError(RuntimeError):
    """The requested embedding backend is not the one actually serving vectors."""


def _get_model():
    global _model, _backend_loaded, _ov_failed
    want = _backend()
    if want == "openvino" and _ov_failed:
        if _strict():
            raise EmbedBackendError("OpenVINO MiniLM backend requested but it failed to "
                                    "load earlier in this process (VOICELINK_EMBED_STRICT=1)")
        want = "torch"  # already fell back; don't retry (and re-warn) every call
    if _model is None or _backend_loaded != want:
        if want == "openvino":
            try:
                from intel import OpenVinoEmbedder  # torch-free
                _model = OpenVinoEmbedder()
            except Exception as e:  # IR missing / openvino absent -> torch
                if _strict():
                    raise EmbedBackendError(
                        f"OpenVINO MiniLM backend requested but unavailable ({e}); refusing "
                        f"the torch fallback because VOICELINK_EMBED_STRICT=1 — run "
                        f"mac/intel/convert.py --skip-student to export the bundle") from e
                print(f"[embed] OpenVINO MiniLM unavailable ({e}); using torch "
                      "(this message prints once)")
                _ov_failed = True
                want = "torch"
        if want == "torch":
            from sentence_transformers import SentenceTransformer  # heavy, lazy
            _model = SentenceTransformer(_MODEL_NAME)
        _backend_loaded = want
    return _model


def backend_in_use() -> str | None:
    """Which runtime actually produced embeddings so far (None before first call)."""
    return _backend_loaded


def assert_backend(expected: str) -> str:
    """Hard check that `expected` ('torch'|'openvino') is the runtime that actually
    served embeddings. Loads the model if nothing has been embedded yet. Any run
    that reports a number under a backend label must call this (bench_e2e does;
    quality evaluations on the exported path should too), so a warned fallback
    can never survive into a labelled row."""
    _get_model()
    got = _backend_loaded
    if got != expected:
        raise EmbedBackendError(
            f"embedding backend mismatch: expected {expected!r} but {got!r} is serving "
            f"(VOICELINK_EMBED_BACKEND={__import__('os').environ.get('VOICELINK_EMBED_BACKEND')!r})")
    return got


# Memo of text -> vector, keyed by backend. The student path embeds the SAME
# ~94-word closed vocabulary on every board; measured 71 of 73 ms per board was
# that re-embedding (intel/bench_e2e.py, 2026-08-23). Vectors are pure functions
# of (backend, text), so caching is exact, not approximate. Bounded so a long
# transcript stream can't grow it without limit.
_CACHE_MAX = 20_000
_cache: dict[tuple[str, str], np.ndarray] = {}


def _encode(m, texts: list[str]) -> np.ndarray:
    if hasattr(m, "encode"):  # sentence-transformers
        vecs = m.encode(list(texts), normalize_embeddings=True)
    else:                     # OpenVinoEmbedder (already normalised)
        vecs = m(list(texts))
    return np.asarray(vecs, dtype=np.float32)


def embed(texts: list[str]) -> np.ndarray:
    """Return an (N, 384) array of L2-normalized embeddings (cosine = dot).

    Repeated texts are served from a process-wide memo (see _cache); only the
    unseen ones hit the model, in one batch."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    m = _get_model()
    be = _backend_loaded or "torch"
    out = np.empty((len(texts), 384), dtype=np.float32)
    missing: dict[str, list[int]] = {}
    for i, t in enumerate(texts):
        hit = _cache.get((be, t))
        if hit is not None:
            out[i] = hit
        else:
            missing.setdefault(t, []).append(i)
    if missing:
        fresh = _encode(m, list(missing))
        if len(_cache) + len(missing) > _CACHE_MAX:
            _cache.clear()
        for t, vec in zip(missing, fresh):
            _cache[(be, t)] = vec
            for i in missing[t]:
                out[i] = vec
    return out


def cache_clear() -> None:
    _cache.clear()


def reset_backend() -> None:
    """Forget a failed OpenVINO load (e.g. after running intel/convert.py)."""
    global _model, _backend_loaded, _ov_failed
    _model = None; _backend_loaded = None; _ov_failed = False


def cache_forget(texts: list[str]) -> None:
    """Evict specific texts (every backend). Benchmarks use this to keep the
    vocabulary warm but force the per-board scene/transcript to be re-encoded,
    which is what a live device sees: static vocab, never-before-seen context."""
    for be in ("torch", "openvino"):
        for t in texts:
            _cache.pop((be, t), None)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine for normalized inputs: (Na,Nb)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    return a @ b.T
