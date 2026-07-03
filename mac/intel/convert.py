"""Convert VoiceLink models to OpenVINO IR (Phase 5, Track 3, step 2).

    VOICELINK_PERCEPTION=0 python3 intel/convert.py [--skip-minilm]

1. student.onnx (the export-first Phase-4 artifact) -> artifacts/student_fp32.xml
   via ov.convert_model + ov.save_model, then a parity check against onnxruntime.
2. MiniLM (sentence-transformers/all-MiniLM-L6-v2, the embedding model the seam
   already uses) -> artifacts/minilm-ov/ via optimum-cli with
   --library sentence_transformers. NOTE: this step needs optimum-intel, which
   pins transformers<5.1 (see README.md) — the runtime path does not.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MAC = HERE.parent
if str(MAC) not in sys.path:
    sys.path.insert(0, str(MAC))

from intel import ARTIFACTS_DIR, MINILM_OV_DIR, STUDENT_FP32_XML  # noqa: E402

STUDENT_ONNX = MAC / "evaluation" / "phase4" / "student.onnx"
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _write_vocab_sidecar(xml_path: Path) -> None:
    """Codex P2: stamp the vocab fingerprint next to the IR so a same-sized but
    reordered vocab.json fails loudly at load instead of serving wrong words."""
    import json as _json
    from evaluation.phase4.data import vocab_sha
    phase4 = Path(__file__).resolve().parent.parent / "evaluation" / "phase4"
    vocab = _json.loads((phase4 / "vocab.json").read_text())
    xml_path.with_suffix(".vocab.sha").write_text(vocab_sha(vocab))


def convert_student(onnx_path: Path = STUDENT_ONNX,
                    xml_path: Path = STUDENT_FP32_XML) -> Path:
    """student.onnx -> FP32 OpenVINO IR (.xml + .bin)."""
    import openvino as ov
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    ov_model = ov.convert_model(str(onnx_path))
    ov.save_model(ov_model, str(xml_path), compress_to_fp16=False)
    _write_vocab_sidecar(xml_path)
    return xml_path


def verify_student_ir(onnx_path: Path = STUDENT_ONNX,
                      xml_path: Path = STUDENT_FP32_XML) -> float:
    """Max |logit delta| between onnxruntime and the compiled IR on random input."""
    import onnxruntime as ort

    from intel import OpenVinoStudent

    x = np.random.default_rng(0).standard_normal((4, 769)).astype(np.float32)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ref = sess.run(["logits"], {"context_emb": x})[0]
    got = OpenVinoStudent(xml_path)(x)
    return float(np.abs(ref - got).max())


def export_minilm(out_dir: Path = MINILM_OV_DIR) -> subprocess.CompletedProcess:
    """MiniLM -> OpenVINO IR via optimum-cli (the documented sentence_transformers path)."""
    cmd = [
        sys.executable, "-m", "optimum.commands.optimum_cli", "export", "openvino",
        "--model", MINILM_MODEL,
        "--library", "sentence_transformers",
        "--task", "feature-extraction",
        str(out_dir),
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def main() -> int:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    xml = convert_student()
    diff = verify_student_ir()
    print(f"student FP32 IR -> {xml}")
    print(f"onnxruntime vs OV-FP32 max abs logit diff = {diff:.2e}")
    if diff >= 1e-4:
        print(">>> student conversion FAILED parity <<<")
        return 1

    if "--skip-minilm" not in sys.argv:
        r = export_minilm()
        if r.returncode == 0 and (MINILM_OV_DIR / "openvino_model.xml").exists():
            print(f"minilm IR -> {MINILM_OV_DIR}")
        else:
            print(f"minilm export FAILED (rc={r.returncode}):\n{r.stderr[-2000:]}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
