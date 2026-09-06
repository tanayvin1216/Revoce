"""I-02 candidate-source contract; corpus regression is in the analysis runner."""
import numpy as np
import pytest

from evaluation.phase4 import infer, data


def test_no_head_controls_and_explicit_random_stream(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No-learning controls must not construct head features")
    monkeypatch.setattr(data, "row_features", forbidden)
    monkeypatch.setattr(infer, "_pool_mult", lambda: 3)
    vocab = [str(i) for i in range(30)]
    call = lambda **kw: infer._candidate_pool(None, vocab, "", "", False, 6, **kw)
    assert call(pool_source="full") == vocab
    assert call(pool_source="vocab_order") == vocab[:18]
    counts = np.arange(30, dtype=float)
    assert call(pool_source="frequency", pool_frequency=counts) == vocab[::-1][:18]
    rng = np.random.default_rng(0)
    expected = np.random.default_rng(0)
    for _ in range(2):
        assert call(pool_source="random", pool_rng=rng) == [vocab[i] for i in expected.choice(30, 18, replace=False)]
    monkeypatch.setattr(infer, "POOL_SOURCE", "full")
    assert call() == vocab


@pytest.mark.parametrize("kw", [
    {"pool_source": "unknown"}, {"pool_source": "random"},
    {"pool_source": "frequency"},
    {"pool_source": "frequency", "pool_frequency": [1]},
    {"pool_source": "frequency", "pool_frequency": [1, float("nan")]},
    {"pool_source": "frequency", "pool_frequency": [1, -1]},
])
def test_invalid_pool_configuration_fails_explicitly(kw):
    with pytest.raises(ValueError):
        infer._candidate_pool(None, ["a", "b"], "", "", False, 6, **kw)
