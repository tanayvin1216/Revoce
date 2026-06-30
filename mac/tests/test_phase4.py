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


# --- ONNX export round-trip (the export-first guarantee) ----------------------

def test_onnx_export_roundtrip(tmp_path):
    from evaluation.phase4.export_onnx import export_student, verify_onnx
    m = StudentRanker(384, 32, 40, 0.0)
    path = tmp_path / "s.onnx"
    export_student(m, path)
    assert path.exists()
    assert verify_onnx(m, path) < 1e-4           # onnxruntime matches eager
