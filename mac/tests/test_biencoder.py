"""v3 bi-encoder: the open-vocabulary invariants.

The claim this model exists to make is NOT "names appear on boards" — that is a
ranking outcome and depends on training. The claim is that an out-of-bank phrase
is REPRESENTABLE: it can be embedded, scored on the same axis as the bank, and
can win a slot on merit. v2 could not do that for any phrase below the frequency
floor, no matter how relevant, because it had no output column to put it in.

These tests use a randomly-initialised model on purpose. The architecture either
admits the phrase or it does not; that property must not depend on a checkpoint.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from evaluation.phase4 import infer_biencoder as ib  # noqa: E402
from evaluation.phase4.student_biencoder import BiEncoderRanker  # noqa: E402

BANK = [
    "i want water", "i need to eat", "it is too loud", "i am tired",
    "can we go home", "i don't know", "that is funny", "i feel sick",
]
CORE = ["yes", "no", "I need help", "wait"]


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = BiEncoderRanker()
    m.eval()
    return m


@pytest.fixture(scope="module")
def bank(model):
    return ib.build_bank_index(model, BANK)


def test_bank_index_projects_every_phrase(model, bank):
    assert bank.size == len(BANK)
    assert bank.proj.shape == (len(BANK), model.proj_dim)
    assert np.isfinite(bank.proj).all()


def test_an_out_of_bank_phrase_is_scorable(model, bank):
    """The core capability. v2 had no way to assign a score to a phrase that
    was not one of its output columns; here it is just another vector."""
    novel = "aunt Marie"
    assert novel.lower() not in [p.lower() for p in BANK]

    from evaluation.embed import embed
    from evaluation.phase4 import data as d

    ctx = ib._project_context(
        model, d.row_features("a kitchen", "who is coming?", True)[None, :])[0]
    proj = ib._project_phrases(
        model, np.asarray(embed([novel]), dtype=np.float32))

    score = float(proj @ ctx)
    assert np.isfinite(score)


def test_bank_can_grow_without_retraining(model, bank):
    """Adding phrases must not require touching the weights — that is the whole
    reason the vocabulary left the architecture."""
    before = bank.size
    grown = ib.build_bank_index(model, BANK + ["my aunt", "sugar", "the bus"])

    assert grown.size == before + 3
    assert grown.proj.shape[1] == bank.proj.shape[1]
    # The original phrases project identically — growth is append-only.
    assert np.allclose(grown.proj[:before], bank.proj, atol=1e-6)


def test_live_candidates_pull_from_transcript_scene_and_profile():
    got = ib.live_candidates(
        "a kitchen table with sugar",
        "Who would you like to sit next to at dinner?",
        {"relationships": [{"name": "Sam", "role": "partner"}]},
    )
    lowered = [w.lower() for w in got]

    assert "dinner" in lowered          # from the transcript
    assert "sugar" in lowered           # from the scene
    assert "sam" in lowered             # from the on-device profile
    assert len(got) <= ib.MAX_LIVE_CANDIDATES


def test_live_candidates_tolerate_a_missing_or_odd_profile():
    for profile in (None, {}, {"relationships": []}, {"people": ["Ada"]}):
        assert isinstance(ib.live_candidates("a room", "hello", profile), list)


def test_board_is_exactly_k_with_core_pinned_first(model, bank):
    board = ib.student_board(
        model, bank, "a kitchen", "are you hungry?", True,
        k=10, k_dyn=6, core_words=CORE,
        personal_context={"relationships": [{"name": "Sam"}]},
    )

    assert len(board) == 10
    assert [w.lower() for w in board[:4]] == [c.lower() for c in CORE]
    assert len(board) == len(set(w.lower() for w in board)), "duplicate on board"


def test_mmr_rel_override_does_not_change_v2_behaviour():
    """selection.mmr_select gained a `rel` override for v3. v2 passes rel=None
    and must be bit-identical to before, so this pins that path."""
    from selection import mmr_select

    rng = np.random.default_rng(3)
    v = rng.standard_normal((12, 8)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    q = v[0]

    assert mmr_select(v, q, 5, 0.6) == mmr_select(v, q, 5, 0.6, None, None)


def test_mmr_rel_override_actually_drives_selection():
    """A relevance override must be able to change the pick, or it is a no-op
    dressed up as a feature."""
    from selection import mmr_select

    rng = np.random.default_rng(5)
    v = rng.standard_normal((10, 8)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    q = v[0]

    forced = np.zeros(10, dtype=np.float32)
    forced[7] = 1.0                       # candidate 7 is maximally relevant

    assert mmr_select(v, q, 3, 0.9, None, forced)[0] == 7


def test_mmr_rel_override_rejects_a_wrong_length_array():
    from selection import mmr_select

    v = np.eye(4, 8, dtype=np.float32)
    with pytest.raises(ValueError, match="expected 4"):
        mmr_select(v, v[0], 2, 0.6, None, np.zeros(3, dtype=np.float32))


def test_selection_defaults_come_from_config():
    """The tuned operating point must be config-driven, not hardcoded — and it
    must be the setting the validation sweep actually chose."""
    sel = ib.Selection()

    assert sel.rel_source in ("cosine", "model", "blend")
    assert sel.pool_mult >= 1
    assert 0.0 <= sel.lam <= 1.0
    # explicit args still win over config
    override = ib.Selection(lam=0.9, pool_mult=2, rel_source="model")
    assert (override.lam, override.pool_mult, override.rel_source) == (0.9, 2, "model")


def test_unknown_rel_source_fails_loudly(model, bank):
    with pytest.raises(ValueError, match="unknown rel_source"):
        ib.student_dynamic_words(model, bank, "a room", "hello?", True, 6,
                                 sel=ib.Selection(rel_source="nonsense"))


def test_board_never_stacks_hedges(model, bank):
    """RC-2: the hedge cap must survive the move to an open bank."""
    from evaluation.hedges import is_hedge

    hedge_heavy = BANK + [
        "i am not sure", "i really don't know", "hard to say", "no idea",
    ]
    idx = ib.build_bank_index(model, hedge_heavy)
    board = ib.student_board(model, idx, "a room", "what do you think?", True,
                             k=10, k_dyn=6, core_words=CORE)

    dynamic = board[4:]
    assert sum(1 for w in dynamic if is_hedge(w)) <= ib.HEDGE_CAP


# --- serving v3 through the seam -------------------------------------------

def test_seam_serves_v3_when_selected(monkeypatch, tmp_path):
    """VOICELINK_STUDENT_VERSION=v3 must route through infer_biencoder, and the
    safety contract must survive the swap."""
    from pathlib import Path

    import ranker_config
    import rank_vocabulary as rv

    ckpt = (Path(rv.__file__).resolve().parent
            / "evaluation" / "phase4" / "student_biencoder.pt")
    if not ckpt.exists():
        pytest.skip("v3 checkpoint not trained in this checkout")

    monkeypatch.setattr(ranker_config, "STUDENT_VERSION", "v3")
    monkeypatch.setattr(rv, "_student_singleton", None)

    board = rv.rank_vocabulary("a lunch room", "who do you want to sit with?",
                               {}, strategy="student", k=10)
    words = [b["word"].lower() for b in board]

    assert len(board) == 10
    for core in rv.GUARANTEED_CORE:
        assert core.lower() in words, f"v3 seam path dropped {core!r}"


def test_v3_is_not_the_default():
    """v3 loses the coverage gate under cross-validation (margin -0.0397), so
    it must stay opt-in. If this ever flips, it should be a deliberate decision
    backed by a passing coverage gate, not a drifted default."""
    import ranker_config
    import importlib
    importlib.reload(ranker_config)

    assert ranker_config.STUDENT_VERSION == "v2"
