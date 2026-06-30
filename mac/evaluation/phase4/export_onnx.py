"""EXPORT-FIRST CHECKPOINT (Phase-4 build order, step 1).

Before ANY training: build the UNTRAINED student, export it to ONNX, then load the
ONNX back with onnxruntime and run it on a dummy input. If this fails, the
architecture uses ops the Intel/OpenVINO path won't take — STOP and simplify the
layers now, not after hours of training. This is the cheapest place to catch the
most expensive Phase-5 failure.

    VOICELINK_PERCEPTION=0 python3 evaluation/phase4/export_onnx.py

Exits 0 only if the ONNX loads and its output matches the eager model's.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MAC = HERE.parent.parent
if str(MAC) not in sys.path:
    sys.path.insert(0, str(MAC))

from evaluation.phase4 import data as d        # noqa: E402
from evaluation.phase4.student import StudentRanker  # noqa: E402

ONNX_PATH = HERE / "student.onnx"
BOARDS = MAC / "evaluation" / "phase3" / "boards.jsonl"


def load_config() -> dict:
    import yaml
    return yaml.safe_load((HERE / "config.yaml").read_text())


def export_student(model: StudentRanker, path: Path = ONNX_PATH) -> Path:
    """Export to ONNX with a dynamic batch axis. Standard opset, no custom ops."""
    model.eval()
    dummy = torch.zeros(1, model.context_dim, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["context_emb"], output_names=["logits"],
        dynamic_axes={"context_emb": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    return path


def verify_onnx(model: StudentRanker, path: Path = ONNX_PATH) -> float:
    """Load with onnxruntime, run a random input, return max abs diff vs eager."""
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(str(path)))  # structural validity
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).standard_normal((3, model.context_dim), dtype=np.float32)
    ort_out = sess.run(["logits"], {"context_emb": x})[0]
    with torch.no_grad():
        eager = model(torch.from_numpy(x)).numpy()
    return float(np.abs(ort_out - eager).max())


def build_untrained(cfg: dict, vocab_size: int) -> StudentRanker:
    m = cfg["model"]
    torch.manual_seed(cfg["seed"])
    return StudentRanker(m["context_dim"], m["hidden_dim"], vocab_size, m["dropout"])


def main() -> int:
    cfg = load_config()
    from rank_vocabulary import GUARANTEED_CORE

    # Size the export to a realistic vocab so the proven artifact matches what we
    # will train. The vocab comes from the Phase-3 teacher boards we already have.
    if BOARDS.exists():
        rows = d.load_teacher_rows(BOARDS)
        vocab, _ = d.build_vocab(rows, cfg["vocab"]["min_freq"],
                                 cfg["vocab"]["max_size"], GUARANTEED_CORE)
    else:
        vocab = GUARANTEED_CORE
    vocab_size = max(len(vocab), len(GUARANTEED_CORE))
    print(f"untrained student: context_dim={cfg['model']['context_dim']} "
          f"hidden={cfg['model']['hidden_dim']} vocab={vocab_size}")

    model = build_untrained(cfg, vocab_size)
    path = export_student(model)
    diff = verify_onnx(model)
    print(f"exported -> {path}")
    print(f"onnxruntime vs eager max abs diff = {diff:.2e}")
    ok = diff < 1e-4
    print(f">>> EXPORT-FIRST CHECKPOINT: {'PASS' if ok else 'FAIL'} <<<")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
