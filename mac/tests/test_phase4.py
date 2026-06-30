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

def test_build_vocab_forces_core_and_respects_freq():
    rows = [
        {"board": ["water", "more", "water"]},
        {"board": ["water", "snack"]},
    ]
    vocab, w2i = d.build_vocab(rows, min_freq=2, max_size=50, core_words=["yes", "no"])
    assert vocab[:2] == ["yes", "no"]          # core force-included, in order
    assert "water" in w2i                       # freq 3 >= min_freq
    assert "snack" not in w2i                   # freq 1 < min_freq


def test_multihot_marks_only_known_words():
    _, w2i = d.build_vocab([{"board": ["water", "water"]}], 1, 50, ["yes"])
    y = d.multihot(["yes", "water", "unknown-phrase"], w2i)
    assert y[w2i["yes"]] == 1.0 and y[w2i["water"]] == 1.0
    assert y.sum() == 2.0                        # unknown dropped


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
