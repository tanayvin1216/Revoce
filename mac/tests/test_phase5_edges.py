"""Phase-5 adversarial/edge-case tests — the seams tonight's review found bugs in,
plus classic AAC-safety properties (never fewer than k, core always pinned,
hedge cluster never stacks the board).

Run from mac/ with VOICELINK_PERCEPTION=0. Deterministic, no network, no model
download — MiniLM is already local (same policy as test_phase4.py); embed() is
monkeypatched wherever real-model semantics aren't the thing under test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

MAC = Path(__file__).resolve().parent.parent
if str(MAC) not in sys.path:
    sys.path.insert(0, str(MAC))

import rank_vocabulary                               # noqa: E402
from evaluation.phase4 import data as d               # noqa: E402
from evaluation.phase4 import infer                   # noqa: E402
from evaluation.phase4.student import StudentRanker   # noqa: E402
from evaluation.hedges import is_hedge, normalize     # noqa: E402

CORE = ["yes", "no", "I need help", "wait"]


def _bias_model(vocab_size: int, bias: list[float]) -> StudentRanker:
    """A deterministic student whose logits are a fixed bias regardless of
    input (final Linear weight zeroed) — scripts exactly which vocab items the
    raw model ranks on top, independent of the (real) MiniLM embedding."""
    m = StudentRanker(context_dim=769, hidden_dim=8, vocab_size=vocab_size, dropout=0.0)
    with torch.no_grad():
        m.net[-1].weight.zero_()
        m.net[-1].bias.copy_(torch.tensor(bias, dtype=torch.float32))
    m.eval()
    return m


HEDGY_VOCAB = ["i don't know", "i am not sure", "not sure", "i'm not sure",
               "window seat", "aisle seat", "my aunt", "my friend",
               "over there", "in the front"]
HEDGY_BIAS = [5.0, 4.5, 4.0, 3.5, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

ALL_HEDGE_VOCAB = ["i don't know", "i am not sure", "not sure", "i'm not sure",
                   "no idea", "maybe", "i can't remember", "unsure"]


# ---------------------------------------------------------------------------
# 1. empty / None / whitespace scene+transcript never raises, never <k, core first
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scene,transcript", [
    ("", ""),
    (None, None),
    ("   ", "\t\n "),
    (None, "Where would you like to sit today?"),
    ("A bus with open seats.", None),
])
def test_student_board_empty_inputs_never_raise_never_short(scene, transcript):
    m = _bias_model(len(HEDGY_VOCAB), HEDGY_BIAS)
    board = infer.student_board(m, HEDGY_VOCAB, scene, transcript,
                                is_question=False, k=10, k_dyn=6, core_words=CORE)
    assert len(board) == 10
    assert board[:4] == CORE


def test_student_board_empty_scene_and_transcript_is_question_true():
    """Answer-mode reservation must not choke on an empty question text."""
    m = _bias_model(len(HEDGY_VOCAB), HEDGY_BIAS)
    board = infer.student_board(m, HEDGY_VOCAB, "", "",
                                is_question=True, k=10, k_dyn=6, core_words=CORE)
    assert len(board) == 10
    assert board[:4] == CORE


# ---------------------------------------------------------------------------
# 2. non-question context: no answer reservation, still no hedge stacking
# ---------------------------------------------------------------------------

def test_non_question_never_invokes_answer_reservation(monkeypatch):
    """is_question=False must skip the answer-quota entirely — a structural
    check (not just an output check) that _reserve_answers is never called."""
    calls: list[int] = []
    original = infer._reserve_answers

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(infer, "_reserve_answers", spy)
    m = _bias_model(len(HEDGY_VOCAB), HEDGY_BIAS)
    words = infer.student_dynamic_words(
        m, HEDGY_VOCAB, scene="A bus with open seats.",
        transcript="Where would you like to sit today?", is_question=False, k_dyn=6)
    assert calls == []
    assert len(words) == 6
    assert sum(1 for w in words if is_hedge(w)) <= 1  # hedge cap still holds


def test_non_question_hedge_cap_holds_even_when_hedges_dominate_bias():
    """No answer reservation to protect the board — the hedge cap alone must
    keep it from becoming a wall of epistemic stalls."""
    m = _bias_model(len(HEDGY_VOCAB), HEDGY_BIAS)  # hedges get the top 4 logits
    words = infer.student_dynamic_words(
        m, HEDGY_VOCAB, scene="A bus with open seats.",
        transcript="", is_question=False, k_dyn=6)
    assert sum(1 for w in words if is_hedge(w)) <= 1


# ---------------------------------------------------------------------------
# 3. hedge-only vocab (pathological): cap must not starve below k_dyn
# ---------------------------------------------------------------------------

def test_all_hedge_vocab_backfills_to_k_dyn_despite_cap():
    """Every vocab item is a hedge; HEDGE_CAP=1 would otherwise starve the
    board to 1 item — the backfill loop (infer.py:187-191) must top it back
    up to k_dyn."""
    m = _bias_model(len(ALL_HEDGE_VOCAB), list(range(len(ALL_HEDGE_VOCAB), 0, -1)))
    words = infer.student_dynamic_words(
        m, ALL_HEDGE_VOCAB, scene="", transcript="", is_question=False, k_dyn=6)
    assert len(words) == 6                       # never short despite the cap
    assert all(is_hedge(w) for w in words)        # pathological vocab, all hedges
    assert len(set(words)) == 6                   # no duplicates from the backfill pass


def test_all_hedge_vocab_smaller_than_k_dyn_returns_all_without_raising():
    tiny = ALL_HEDGE_VOCAB[:3]
    m = _bias_model(len(tiny), [3.0, 2.0, 1.0])
    words = infer.student_dynamic_words(
        m, tiny, scene="", transcript="", is_question=False, k_dyn=6)
    assert len(words) == 3                        # pool smaller than k_dyn, no crash
    assert len(set(words)) == 3


# ---------------------------------------------------------------------------
# 4. unicode / apostrophe / punctuation variants through hedges.normalize/is_hedge
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("phrase", [
    "I don’t know",           # curly right single quote
    "I don't know!?",             # trailing punctuation pile-up
    "  I don't know  ",           # surrounding whitespace
    "I DON'T KNOW",                # shouted case
    "I don’t know...",       # curly quote + trailing dots
    "i don't know,",               # trailing comma
])
def test_is_hedge_recognizes_unicode_and_punctuation_variants(phrase):
    assert is_hedge(phrase), phrase


def test_normalize_strips_curly_quote_and_trailing_punctuation():
    assert normalize("I don’t know!?") == "i don't know"
    assert normalize("  Not Sure.  ") == "not sure"


@pytest.mark.parametrize("phrase", [
    "I don't know how to swim",    # superset phrase, not the hedge itself
    "a window seat",
    "",
    "don't know",                  # missing subject — different phrase
])
def test_is_hedge_rejects_near_miss_phrases(phrase):
    assert not is_hedge(phrase), phrase


# ---------------------------------------------------------------------------
# 5. stale-artifact guards on _load_student_torch (tmp_path only — never the
#    real artifacts)
# ---------------------------------------------------------------------------

def _write_fake_checkpoint(tmp_path: Path, *, vocab_words: list[str],
                          ckpt_vocab_size: int, vocab_sha_override=None) -> Path:
    """A real, loadable StudentRanker checkpoint whose declared vocab_size (and
    optionally vocab_sha) can be made to disagree with vocab.json, so
    _load_student_torch's stale-artifact guard fires."""
    context_dim, hidden_dim, dropout = 769, 8, 0.0
    model = StudentRanker(context_dim, hidden_dim, ckpt_vocab_size, dropout)
    torch.save({
        "state_dict": model.state_dict(),
        "context_dim": context_dim, "hidden_dim": hidden_dim,
        "dropout": dropout, "vocab_size": ckpt_vocab_size,
        "vocab_sha": vocab_sha_override if vocab_sha_override is not None
                     else d.vocab_sha(vocab_words),
        "guaranteed_core": CORE, "k_dyn": 6,
    }, tmp_path / "student.pt")
    (tmp_path / "vocab.json").write_text(json.dumps(vocab_words))
    return tmp_path


def test_load_student_torch_raises_on_vocab_size_mismatch(tmp_path):
    vocab_words = ["water", "snack", "more"]          # 3 words in vocab.json
    _write_fake_checkpoint(tmp_path, vocab_words=vocab_words, ckpt_vocab_size=5)
    with pytest.raises(RuntimeError, match="vocab.json has 3 words but"):
        rank_vocabulary._load_student_torch(tmp_path)


def test_load_student_torch_raises_on_vocab_sha_mismatch(tmp_path):
    vocab_words = ["water", "snack", "more"]
    _write_fake_checkpoint(tmp_path, vocab_words=vocab_words, ckpt_vocab_size=3,
                          vocab_sha_override="deadbeefdeadbeef")
    with pytest.raises(RuntimeError, match="vocab.json content drifted"):
        rank_vocabulary._load_student_torch(tmp_path)


def test_load_student_torch_succeeds_when_consistent(tmp_path):
    """Negative control: the guard does NOT fire on a genuinely consistent
    checkpoint/vocab pair — proves the two failure tests above are testing the
    guard, not an unrelated load error."""
    vocab_words = ["water", "snack", "more"]
    _write_fake_checkpoint(tmp_path, vocab_words=vocab_words, ckpt_vocab_size=3)
    model, vocab, core, k_dyn = rank_vocabulary._load_student_torch(tmp_path)
    assert vocab == vocab_words
    assert core == CORE
    assert k_dyn == 6


# ---------------------------------------------------------------------------
# 6. _student_board failure fallback: _load_student raising must yield [], not
#    propagate, so rank_vocabulary's caller falls back to the diverse path.
# ---------------------------------------------------------------------------

def test_student_board_returns_empty_list_when_load_student_raises(monkeypatch):
    def boom():
        raise RuntimeError("stale artifact — simulated for the fallback test")

    monkeypatch.setattr(rank_vocabulary, "_load_student", boom)
    result = rank_vocabulary._student_board(
        "A bus with open seats.", "Where would you like to sit today?", {}, 10)
    assert result == []


def test_student_board_returns_empty_list_when_infer_raises(monkeypatch):
    """Failure inside student_board() itself (not just _load_student) must
    also degrade to [] rather than raising through the seam."""
    monkeypatch.setattr(rank_vocabulary, "_load_student",
                        lambda: (None, ["water"], CORE, 6))

    def boom_student_board(*args, **kwargs):
        raise ValueError("simulated inference failure")

    import evaluation.phase4.infer as infer_mod
    monkeypatch.setattr(infer_mod, "student_board", boom_student_board)
    result = rank_vocabulary._student_board("scene", "transcript", {}, 10)
    assert result == []


# ---------------------------------------------------------------------------
# 7. the seam contract under strategy="student" — REAL checkpoint
# ---------------------------------------------------------------------------

CKPT = MAC / "evaluation" / "phase4" / "student.pt"
_needs_ckpt = pytest.mark.skipif(
    not CKPT.exists(), reason="student.pt not trained in this checkout")


@_needs_ckpt
def test_seam_student_strategy_contract(monkeypatch):
    """strategy='student' through the real seam: exactly k=10 RankedWord items,
    core pinned first, board never empty — the seam-level version of
    test_infer_student_board_pins_core_first (this one exercises
    rank_vocabulary.rank_vocabulary(), not infer.student_board() directly)."""
    monkeypatch.setattr(rank_vocabulary, "_student_singleton", None)
    board = rank_vocabulary.rank_vocabulary(
        scene="A bus with open seats near the window.",
        transcript="Where would you like to sit today?",
        strategy="student",
    )
    assert len(board) == rank_vocabulary.BOARD_SIZE == 10
    assert board  # never empty
    for item in board:
        assert set(item.keys()) == {"word", "symbol"}
        assert isinstance(item["word"], str) and item["word"]
        assert isinstance(item["symbol"], str) and item["symbol"]
    words_lower = [b["word"].lower() for b in board[:4]]
    assert words_lower == [w.lower() for w in rank_vocabulary.GUARANTEED_CORE]


# ---------------------------------------------------------------------------
# 8. dedup_vocab edge cases
# ---------------------------------------------------------------------------

def test_dedup_vocab_empty_list():
    kept, canonical = d.dedup_vocab([], tau=0.80)
    assert kept == [] and canonical == {}


def test_dedup_vocab_single_word():
    kept, canonical = d.dedup_vocab(["water"], tau=0.80)
    assert kept == ["water"] and canonical == {}


def test_dedup_vocab_all_identical_words(monkeypatch):
    """Three copies of the same word must collapse to one kept entry — a
    trivial single-link component of size 3 (kept under the >4 oversized
    threshold, so it doesn't need to be a hedge to survive)."""
    fake_vec = {"water": np.array([1.0, 0.0], dtype=np.float32)}
    monkeypatch.setattr("evaluation.embed.embed",
                        lambda words: np.stack([fake_vec[w] for w in words]))
    kept, canonical = d.dedup_vocab(["water", "water", "water"], tau=0.80)
    assert kept == ["water"]
    assert canonical == {"water": "water"}   # dropped copies map to the survivor


def test_dedup_vocab_oversized_non_hedge_component_raises(monkeypatch):
    """6 near-identical NON-hedge phrases (cosine=1 to each other via a
    controlled fake embed) must trip the oversized-component guard (>4,
    non-hedge) rather than silently eating vocabulary — deterministic, no
    real MiniLM needed."""
    phrases = [f"phrase-{i}" for i in range(6)]
    same_vec = np.array([1.0, 0.0], dtype=np.float32)
    other_vec = np.array([0.0, 1.0], dtype=np.float32)
    vecs = {p: same_vec for p in phrases}
    vecs["distinct-control"] = other_vec

    monkeypatch.setattr("evaluation.embed.embed",
                        lambda words: np.stack([vecs[w] for w in words]))
    with pytest.raises(ValueError, match="non-hedge component"):
        d.dedup_vocab(phrases + ["distinct-control"], tau=0.80)


# ---------------------------------------------------------------------------
# 9. OpenVINO path (pytest.importorskip skips cleanly when unavailable)
# ---------------------------------------------------------------------------

# Codex P1: importorskip at module level skipped ALL 33 tests (hedge, dedup,
# fallback, guards) on machines without openvino. Import lazily per-test.


def _openvino_student_cls():
    pytest.importorskip("openvino")
    from intel import OpenVinoStudent
    return OpenVinoStudent


ARTIFACTS = MAC / "intel" / "artifacts"
FP32_XML = ARTIFACTS / "student_fp32.xml"

_needs_fp32_ir = pytest.mark.skipif(
    not FP32_XML.exists(), reason="student FP32 IR missing — run intel/convert.py")


@_needs_fp32_ir
def test_openvino_student_zeros_embedding_returns_well_formed_logits():
    student = _openvino_student_cls()(FP32_XML)
    logits = student(np.zeros(769, dtype=np.float32))
    assert logits.shape == (1, student.vocab_size)
    assert np.all(np.isfinite(logits))


@_needs_fp32_ir
def test_openvino_student_wrong_dtype_float64_input_still_returns_logits():
    """The runtime must coerce dtype rather than raising or silently corrupting
    output — OpenVinoStudent.__call__ casts via np.asarray(..., dtype=float32)."""
    student = _openvino_student_cls()(FP32_XML)
    batch = np.zeros((3, 769), dtype=np.float64)
    logits = student(batch)
    assert logits.shape == (3, student.vocab_size)
    assert logits.dtype == np.float32
    assert np.all(np.isfinite(logits))


# ---------------------------------------------------------------------------
# 10. Review-round additions (Codex P2 + Sonnet cross-review, EDGE_CASES.md)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CKPT.exists(), reason="student.pt not trained here")
def test_seam_student_strategy_does_not_use_diverse_fallback(monkeypatch):
    """Codex P2 / Sonnet #15: the contract test could pass via the diverse
    fallback while the student path was broken. Prove the student built the
    board by making the fallback path raise if touched."""
    import rank_vocabulary as rv

    def _boom(*a, **k):  # pragma: no cover - failure mode
        raise AssertionError("diverse fallback was used - student path broken")

    monkeypatch.setattr(rv, "generate_pool", _boom)
    monkeypatch.setattr(rv, "select_board", _boom)
    monkeypatch.setattr(rv, "_student_singleton", None)
    board = rv.rank_vocabulary(scene="A kitchen at snack time.",
                               transcript="Do you want apple or crackers?",
                               context={}, strategy="student")
    assert len(board) == 10
    assert [w["word"] for w in board[:4]] == rv.GUARANTEED_CORE


def test_openvino_loader_rejects_missing_or_stale_sidecar(tmp_path, monkeypatch):
    """Sonnet missing-case #3: _load_student_openvino's vocab.sha sidecar guard
    (the same-size-reorder defense) had zero coverage."""
    pytest.importorskip("openvino")
    import shutil
    import rank_vocabulary as rv
    if not FP32_XML.exists():
        pytest.skip("student FP32 IR missing")
    # The loader reads vocab.json BEFORE it reaches the sidecar check, so
    # without it this raises FileNotFoundError and never exercises the guard
    # under test. Its sibling artifact tests already skip on this; this one
    # only checked the IR, which is why it went red the moment openvino was
    # installed in a checkout with no phase-4 artifacts.
    if not (Path(rv.__file__).resolve().parent
            / "evaluation" / "phase4" / "vocab.json").exists():
        pytest.skip("vocab.json not built in this checkout")
    for ext in (".xml", ".bin"):
        shutil.copy(FP32_XML.with_suffix(ext), tmp_path / ("student_fp32" + ext))
    # copy WITHOUT the .vocab.sha sidecar -> loader must refuse to serve
    import intel
    monkeypatch.setattr(intel, "student_ir_path",
                        lambda: tmp_path / "student_fp32.xml")
    monkeypatch.setattr(rv, "_student_singleton", None)
    with pytest.raises(RuntimeError, match="different vocab.json"):
        rv._load_student_openvino(
            Path(rv.__file__).resolve().parent / "evaluation" / "phase4")
    # stale sidecar (wrong sha) must also be rejected
    (tmp_path / "student_fp32.vocab.sha").write_text("deadbeefdeadbeef")
    with pytest.raises(RuntimeError, match="different vocab.json"):
        rv._load_student_openvino(
            Path(rv.__file__).resolve().parent / "evaluation" / "phase4")


@pytest.mark.skipif(not CKPT.exists(), reason="student.pt not trained here")
def test_infer_student_board_end_to_end_with_openvino_model():
    """Sonnet missing-case #4: prove infer.py's numpy branch drives the full
    MMR selection with an OpenVinoStudent, not just raw __call__."""
    pytest.importorskip("openvino")
    if not FP32_XML.exists():
        pytest.skip("student FP32 IR missing")
    import json
    from evaluation.phase4 import infer
    student = _openvino_student_cls()(FP32_XML)
    vocab = json.loads((MAC / "evaluation" / "phase4" / "vocab.json").read_text())
    board = infer.student_board(student, vocab, "A kitchen at snack time.",
                                "Do you want apple or crackers?",
                                is_question=True, k=10, k_dyn=6,
                                core_words=["yes", "no", "I need help", "wait"])
    assert len(board) == 10
    assert board[:4] == ["yes", "no", "I need help", "wait"]
    assert all(w in vocab or w in board[:4] for w in board)
