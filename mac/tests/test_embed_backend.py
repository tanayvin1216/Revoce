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


def test_embed_memo_is_exact_and_order_preserving(monkeypatch):
    em = _fresh_embed(monkeypatch, "torch")
    em.cache_clear()
    a = em.embed(["yes", "no", "carbon dioxide"])
    b = em.embed(["carbon dioxide", "yes", "water", "yes"])  # mixed hits/misses/dups
    np.testing.assert_array_equal(b[0], a[2])
    np.testing.assert_array_equal(b[1], a[0])
    np.testing.assert_array_equal(b[3], a[0])
    ref = em._encode(em._get_model(), ["water"])[0]
    np.testing.assert_allclose(b[2], ref, atol=1e-6)
    assert b.shape == (4, 384)


def test_embed_strict_mode_refuses_torch_fallback(monkeypatch, tmp_path):
    """I-12: with VOICELINK_EMBED_STRICT=1 a missing MiniLM IR must raise, not
    silently serve torch vectors under an 'openvino' label."""
    import intel
    monkeypatch.setenv("VOICELINK_EMBED_STRICT", "1")
    em = _fresh_embed(monkeypatch, "openvino")
    monkeypatch.setattr(intel.OpenVinoEmbedder, "__init__",
                        lambda self, model_dir=None: (_ for _ in ()).throw(FileNotFoundError("x")))
    with pytest.raises(em.EmbedBackendError):
        em.embed(["hello"])
    with pytest.raises(em.EmbedBackendError):  # sticky: the retry path raises too
        em.assert_backend("openvino")
    assert em.backend_in_use() is None


def test_assert_backend_matches_actual_runtime(monkeypatch):
    em = _fresh_embed(monkeypatch, "torch")
    assert em.assert_backend("torch") == "torch"
    with pytest.raises(em.EmbedBackendError):
        em.assert_backend("openvino")


def test_minilm_bundle_status_reports_missing_files(tmp_path):
    import intel
    d = tmp_path / "minilm-ov"
    d.mkdir()
    (d / "openvino_model.xml").write_text("<net/>")  # the state the repo was found in
    s = intel.minilm_bundle_status(d)
    assert s["complete"] is False
    assert "openvino_model.bin" in s["missing"]
    with pytest.raises(FileNotFoundError, match="openvino_model.bin"):
        intel.OpenVinoEmbedder(d)
