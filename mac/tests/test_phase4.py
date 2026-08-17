"""Phase-4 unit tests — fast, offline (no LLM, no MiniLM).

Cover the data plumbing, the student forward contract, and the ONNX export
round-trip (the export-first guarantee), which together gate the build order.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

MAC = Path(__file__).resolve().parent.parent
if str(MAC) not in sys.path:
    sys.path.insert(0, str(MAC))

from evaluation.phase4 import data as d              # noqa: E402
from evaluation.phase4.student import StudentRanker  # noqa: E402


# --- data plumbing -----------------------------------------------------------

def test_build_vocab_excludes_core_and_respects_freq():
    """Core words never enter the dynamic vocab; freq gate still applies."""
    rows = [
        {"board": ["yes", "water", "no", "water"]},   # yes/no are core
        {"board": ["yes", "no", "water", "snack"]},
    ]
    vocab, w2i = d.build_vocab(rows, min_freq=2, max_size=50)
    assert "yes" not in w2i            # core excluded even though frequent
    assert "no" not in w2i             # core excluded
    assert "i need help" not in w2i    # core excluded
    assert "wait" not in w2i           # core excluded
    assert "water" in w2i              # freq 3 >= min_freq, not core
    assert "snack" not in w2i          # freq 1 < min_freq


def test_multihot_excludes_core_marks_dynamic():
    """Core words on the board are NOT included in the multihot target."""
    rows = [{"board": ["water", "water"]}]
    _, w2i = d.build_vocab(rows, 1, 50)
    assert "yes" not in w2i            # core excluded from vocab
    y = d.multihot(["yes", "water", "unknown-phrase"], w2i)
    assert y[w2i["water"]] == 1.0
    assert y.sum() == 1.0              # only water; yes is core, unknown dropped


HEDGE_VARIANTS = ["i don't know", "i am not sure", "i'm not sure", "i'm not sure.",
                  "i am not sure.", "not sure", "i don't know."]


def test_dedup_vocab_collapses_hedges_to_one():
    """The 7 known hedge paraphrases (RC-2) collapse to exactly 1 representative;
    every dropped variant maps to that representative in the canonical map."""
    ranked = HEDGE_VARIANTS + ["water", "carbon dioxide", "my aunt"]
    kept, canonical = d.dedup_vocab(ranked, tau=0.80)
    kept_hedges = [w for w in kept if w in HEDGE_VARIANTS]
    assert kept_hedges == ["i don't know"]     # highest-ranked member represents
    for w in HEDGE_VARIANTS[1:]:
        assert canonical[w] == "i don't know"
    assert "water" in kept and "carbon dioxide" in kept and "my aunt" in kept
    assert "water" not in canonical            # kept words never appear as sources


def test_dedup_vocab_preserves_rank_order():
    """The kept list stays in frequency-rank order; the duplicate ("i don't
    know." ~ "i don't know" = 0.926) drops and maps to the earlier-ranked word."""
    kept, canonical = d.dedup_vocab(["i don't know", "water", "i don't know."], tau=0.80)
    assert kept == ["i don't know", "water"]
    assert canonical == {"i don't know.": "i don't know"}


def test_multihot_canonical_redirects_label_mass():
    """A dropped variant on a teacher board marks its kept representative's bit
    instead of silently losing the label."""
    w2i = {"i don't know": 0, "water": 1}
    canonical = {"i am not sure": "i don't know"}
    y = d.multihot(["I am not sure", "water"], w2i, canonical)
    assert y[0] == 1.0                         # redirected label mass
    assert y[1] == 1.0
    assert y.sum() == 2.0


def test_assemble_board_pins_core_exactly_k_disjoint():
    """Core is pinned first in order; result is exactly k; dynamic section has no core."""
    core = ["yes", "no", "I need help", "wait"]
    dynamic = ["drink", "eat", "more", "stop", "go", "home"]
    board = d.assemble_board(core, dynamic, k=10)
    assert board[:4] == core                    # core pinned in order
    assert len(board) == 10                     # exactly k
    core_lower = {w.lower() for w in core}
    assert not any(w.lower() in core_lower for w in board[4:])  # dynamic section is disjoint


def test_assemble_board_truncates_to_k():
    """assemble_board never exceeds k even if core + dynamic > k."""
    core = ["yes", "no", "I need help", "wait"]
    dynamic = ["a", "b", "c", "d", "e", "f", "g", "h"]
    board = d.assemble_board(core, dynamic, k=6)
    assert len(board) == 6
    assert board[:4] == core


def test_load_teacher_rows_filters_topk(tmp_path):
    p = tmp_path / "b.jsonl"
    p.write_text(
        '{"scene":"s","transcript":"t","strategy":"diverse","board":["a","b"]}\n'
        '{"scene":"s","transcript":"t","strategy":"topk","board":["c"]}\n')
    rows = d.load_teacher_rows(p)
    assert len(rows) == 1 and rows[0]["board"] == ["a", "b"]


def test_split_indices_partition_is_disjoint_and_complete():
    s = d.split_indices(100, val_frac=0.15, test_frac=0.15, seed=0)
    allidx = s["train"] + s["val"] + s["test"]
    assert sorted(allidx) == list(range(100))    # complete, no overlap
    assert len(s["test"]) == 15 and len(s["val"]) == 15


# --- student -----------------------------------------------------------------

def test_student_forward_shape():
    m = StudentRanker(context_dim=384, hidden_dim=32, vocab_size=40, dropout=0.0)
    out = m(torch.zeros(5, 384))
    assert out.shape == (5, 40)


def test_student_board_indices_topk():
    m = StudentRanker(384, 32, 40, 0.0)
    idxs = m.board_indices(torch.randn(384), k=7)
    assert len(idxs) == 7 and len(set(idxs)) == 7


# --- inference (infer.py — MMR + answer reservation + hedge cap) --------------

def _bias_model(vocab_size: int, bias: list[float]) -> StudentRanker:
    """A deterministic student whose logits are a fixed bias regardless of input
    (final Linear weight zeroed) — lets tests script exactly which vocab items
    the raw model would rank on top."""
    m = StudentRanker(context_dim=769, hidden_dim=8, vocab_size=vocab_size, dropout=0.0)
    with torch.no_grad():
        m.net[-1].weight.zero_()
        m.net[-1].bias.copy_(torch.tensor(bias, dtype=torch.float32))
    m.eval()
    return m


HEDGY_VOCAB = ["i don't know", "i am not sure", "not sure", "i'm not sure",
               "window seat", "aisle seat", "my aunt", "my friend",
               "over there", "in the front"]
# Hedges get the top logits — the exact RC-1 failure shape (raw topk would
# return 4 hedges); infer.py must not.
HEDGY_BIAS = [5.0, 4.5, 4.0, 3.5, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]


def test_infer_caps_hedges_at_one():
    from evaluation.phase4 import infer
    m = _bias_model(len(HEDGY_VOCAB), HEDGY_BIAS)
    words = infer.student_dynamic_words(
        m, HEDGY_VOCAB, scene="A bus with open seats.",
        transcript="Where would you like to sit today?", is_question=False, k_dyn=6)
    assert len(words) == 6
    assert sum(1 for w in words if infer.is_hedge(w)) <= 1


def test_infer_question_reserves_answer_slots():
    from evaluation.phase4 import infer
    m = _bias_model(len(HEDGY_VOCAB), HEDGY_BIAS)
    words = infer.student_dynamic_words(
        m, HEDGY_VOCAB, scene="A bus with open seats.",
        transcript="Where would you like to sit today?", is_question=True, k_dyn=6)
    assert not infer.is_hedge(words[0])        # >=2 answer slots come first
    assert not infer.is_hedge(words[1])
    assert sum(1 for w in words if infer.is_hedge(w)) <= 1


def test_single_token_context_union_stays_disabled():
    """Review-opus fix 2: the single-token union injected question-echo nouns
    that displaced real answers (ablation: 0.563 -> 0.803 coverage without it).
    It stays OFF — RC-3 is served by answers.py's multi-word extractor instead."""
    from evaluation.phase4 import infer
    assert infer.CONTEXT_UNION is False


def test_every_board_word_is_vocab_or_grounded_in_the_context():
    """The RC-3 invariant that replaced "nothing outside the trained vocab".

    Open answers are out-of-vocab BY DESIGN — that is the whole point, "aunt"
    can never clear the frequency floor. What must still hold is that nothing is
    invented: an out-of-vocab word has to be read out of the scene or the
    question the partner actually said.
    """
    from evaluation.phase4 import infer
    vocab = ["i don't know", "i am not sure", "my aunt", "my friend"]
    m = _bias_model(len(vocab), [5.0, 4.5, 1.0, 0.9])
    scene = "A train carriage with empty seats."
    transcript = "Do you want the window seat or the aisle seat?"
    words = infer.student_dynamic_words(
        m, vocab, scene=scene, transcript=transcript, is_question=True, k_dyn=6)

    context = f"{scene} {transcript}".lower()
    for w in words:
        assert w in vocab or all(tok in context for tok in w.lower().split()), w
    # and the open path did fire here — the answers are stated in the question
    assert any(w not in vocab for w in words), words


def test_context_union_still_works_when_enabled(monkeypatch):
    """The RC-3-lite machinery is kept behind the flag for the future
    multi-word extractor — verify it still functions when switched on."""
    from evaluation.phase4 import infer
    monkeypatch.setattr(infer, "CONTEXT_UNION", True)
    vocab = ["i don't know", "i am not sure", "my aunt", "my friend"]
    m = _bias_model(len(vocab), [5.0, 4.5, 1.0, 0.9])
    words = infer.student_dynamic_words(
        m, vocab, scene="A train carriage with empty seats.",
        transcript="Do you want the window seat or the aisle seat?",
        is_question=True, k_dyn=6)
    assert any(w in {"window", "aisle", "seat", "train", "carriage"} for w in words)


def test_infer_student_board_pins_core_first():
    from evaluation.phase4 import infer
    core = ["yes", "no", "I need help", "wait"]
    m = _bias_model(len(HEDGY_VOCAB), HEDGY_BIAS)
    board = infer.student_board(
        m, HEDGY_VOCAB, scene="A bus.", transcript="Where do you want to sit?",
        is_question=True, k=10, k_dyn=6, core_words=core)
    assert board[:4] == core
    assert len(board) == 10


# --- ONNX export round-trip (the export-first guarantee) ----------------------

def test_onnx_export_roundtrip(tmp_path):
    from evaluation.phase4.export_onnx import export_student, verify_onnx
    m = StudentRanker(384, 32, 40, 0.0)
    path = tmp_path / "s.onnx"
    export_student(m, path)
    assert path.exists()
    assert verify_onnx(m, path) < 1e-4           # onnxruntime matches eager


# --- seating probes on the REAL retrained checkpoint ---------------------------
# The two investigation scenarios that produced hedge walls (v2-32, v2-85).
# Skipped when the (gitignored) artifacts are absent; green after train.py runs.

import pytest  # noqa: E402

CKPT = MAC / "evaluation" / "phase4" / "student.pt"
_needs_ckpt = pytest.mark.skipif(
    not CKPT.exists(), reason="student.pt not trained in this checkout")

# Review S2: the old stems accepted generic scene nouns ("room", "table"), so
# the test passed on boards whose real answer sat outside top-7. Probes now
# demand the ACTUAL answer tokens.
#
# The two out-of-vocab probes were xfail while the board could only emit closed
# vocabulary: an answer the teacher never produced above the frequency floor was
# unrepresentable at any model quality. They are live assertions now that
# answers.py supplies open-vocabulary answers for deictic questions — "aunt" is
# reachable because it is read out of the scene, not because it entered the
# vocab. See answers.py for why the frequency floor cannot fix this on its own.
SEATING_PROBES = [
    pytest.param(
        "A train carriage with pairs of seats, some by the window and some by the aisle.",
        "Do you want the window seat or the aisle seat?",
        ("window seat", "aisle seat"),
        id="in-vocab-window-aisle"),
    pytest.param(
        "The teacher gestures around a room with several open desks as the user comes in.",
        "Where would you like to sit today?",
        ("sit", "seat", "desk", "front", "back"),
        id="oov-classroom"),
    pytest.param(
        "Crowded holiday dinner table, aunt pulling out an empty chair between two relatives.",
        "Who do you want to sit next to?",
        ("aunt", "friend", "sibling", "grandparent", "mom", "dad"),
        id="oov-dinner-person"),
]


@_needs_ckpt
@pytest.mark.parametrize("scene,transcript,concrete_stems", SEATING_PROBES)
def test_seating_probe_has_concrete_answer(scene, transcript, concrete_stems):
    """Top-7 must offer >=1 concrete answer (seat/person word), not a hedge wall."""
    from evaluation.phase4 import infer
    from evaluation.phase4.evaluate import load_student
    model, vocab, core_words, k_dyn = load_student()
    board = infer.student_board(model, vocab, scene, transcript,
                                is_question=True, k=10, k_dyn=k_dyn,
                                core_words=core_words)
    top7 = [w.lower() for w in board[:7]]
    assert sum(1 for w in top7 if infer.is_hedge(w)) <= 1
    assert any(stem in w for w in top7 for stem in concrete_stems), top7
    # a hedge-free board of generic filler is not a pass — demand the answer
    # tokens above, not merely "no hedge wall"
