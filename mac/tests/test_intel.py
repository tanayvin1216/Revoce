"""Track-3 tests: OpenVINO IR parity, INT8 quality, and the seam backend flag.

Skips cleanly when openvino isn't installed or the IR artifacts haven't been
built yet (mac/intel/convert.py + quantize.py generate them).
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("openvino")

MAC = Path(__file__).resolve().parents[1]
ARTIFACTS = MAC / "intel" / "artifacts"
FP32_XML = ARTIFACTS / "student_fp32.xml"
INT8_XML = ARTIFACTS / "student_int8.xml"
STUDENT_PT = MAC / "evaluation" / "phase4" / "student.pt"
SCENARIOS = MAC / "evaluation" / "phase4" / "v2_scenarios.jsonl"

needs_fp32 = pytest.mark.skipif(
    not (FP32_XML.exists() and STUDENT_PT.exists()),
    reason="student FP32 IR / checkpoint missing — run intel/convert.py",
)
needs_int8 = pytest.mark.skipif(
    not (FP32_XML.exists() and INT8_XML.exists()),
    reason="student INT8 IR missing — run intel/quantize.py",
)


def load_torch_student():
    import torch

    from evaluation.phase4.student import StudentRanker
    ck = torch.load(STUDENT_PT, map_location="cpu", weights_only=False)
    model = StudentRanker(ck["context_dim"], ck["hidden_dim"],
                          ck["vocab_size"], ck["dropout"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model


def real_feature_batch(n_rows: int = 20) -> np.ndarray:
    from evaluation.phase4 import data as d
    rows = [json.loads(ln) for ln in SCENARIOS.read_text().splitlines()[:n_rows]]
    return np.stack([
        d.row_features(r.get("scene", ""), r.get("transcript", ""),
                       bool(r.get("is_question")))
        for r in rows
    ])


@needs_fp32
def test_fp32_ir_matches_torch_logits():
    import torch

    from intel import OpenVinoStudent
    model = load_torch_student()
    student = OpenVinoStudent(FP32_XML)
    x = np.random.default_rng(7).standard_normal((10, 769)).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    assert float(np.abs(student(x) - ref).max()) < 1e-3


@needs_int8
def test_int8_top6_overlap_vs_fp32():
    from intel import OpenVinoStudent
    fp32, int8 = OpenVinoStudent(FP32_XML), OpenVinoStudent(INT8_XML)
    feats = real_feature_batch()
    k = 6
    overlaps = [
        len(set(np.argsort(-a)[:k]) & set(np.argsort(-b)[:k])) / k
        for a, b in zip(fp32(feats), int8(feats))
    ]
    assert float(np.mean(overlaps)) >= 0.8


def test_backend_flag_round_trip(monkeypatch):
    import ranker_config
    monkeypatch.setenv("VOICELINK_STUDENT_BACKEND", "openvino")
    importlib.reload(ranker_config)
    assert ranker_config.STUDENT_BACKEND == "openvino"
    monkeypatch.delenv("VOICELINK_STUDENT_BACKEND")
    importlib.reload(ranker_config)
    assert ranker_config.STUDENT_BACKEND == "torch"  # the default


@needs_fp32
def test_seam_serves_student_board_on_openvino_backend(monkeypatch):
    import rank_vocabulary
    import ranker_config

    from intel import OpenVinoStudent
    monkeypatch.setattr(ranker_config, "STUDENT_BACKEND", "openvino")
    monkeypatch.setattr(rank_vocabulary, "_student_singleton", None)
    model, vocab, core, k_dyn = rank_vocabulary._load_student()
    assert isinstance(model, OpenVinoStudent)

    board = rank_vocabulary.rank_vocabulary(
        scene="a school bus with empty seats near the window",
        transcript="Where would you like to sit today?",
        strategy="student",
    )
    assert len(board) == rank_vocabulary.BOARD_SIZE
    words = [b["word"].lower() for b in board]
    assert set(w.lower() for w in core) <= set(words)
    # leave no OpenVINO student cached for other tests
    monkeypatch.setattr(rank_vocabulary, "_student_singleton", None)
