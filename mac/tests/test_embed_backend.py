"""The embedding helper can run MiniLM on OpenVINO (torch-free) and must agree
with the torch path — otherwise the MMR selector would pick different words on
Intel than on the Mac, silently."""
from __future__ import annotations

import importlib

import numpy as np
import pytest

pytest.importorskip("openvino")


def _fresh_embed(monkeypatch, backend: str):
    monkeypatch.setenv("VOICELINK_EMBED_BACKEND", backend)
    import evaluation.embed as em
    em = importlib.reload(em)
    return em


def test_openvino_embedder_matches_torch(monkeypatch):
    from intel import MINILM_OV_DIR
    if not (MINILM_OV_DIR / "openvino_model.bin").exists():
        pytest.skip("MiniLM IR not exported (run intel/convert.py)")
    texts = ["what do plants take in from the air", "I need help", "carbon dioxide"]
    torch_em = _fresh_embed(monkeypatch, "torch")
    ref = torch_em.embed(texts)
    ov_em = _fresh_embed(monkeypatch, "openvino")
    got = ov_em.embed(texts)
    assert ov_em.backend_in_use() == "openvino"
    assert got.shape == (3, 384)
    np.testing.assert_allclose(np.linalg.norm(got, axis=1), 1.0, atol=1e-5)
    cos = (ref * got).sum(1)
    assert cos.min() > 0.999, cos


def test_openvino_embedder_falls_back_when_ir_missing(monkeypatch, tmp_path):
    import intel
    monkeypatch.setattr(intel, "MINILM_OV_DIR", tmp_path / "nope")
    em = _fresh_embed(monkeypatch, "openvino")
    # OpenVinoEmbedder default arg was bound at import; force the miss via attr.
    monkeypatch.setattr(intel.OpenVinoEmbedder, "__init__",
                        lambda self, model_dir=None: (_ for _ in ()).throw(FileNotFoundError("x")))
    v = em.embed(["hello"])
    assert v.shape == (1, 384)
    assert em.backend_in_use() == "torch"
