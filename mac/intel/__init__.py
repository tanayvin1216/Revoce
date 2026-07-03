"""Intel/OpenVINO lifecycle for the Phase-4 student ranker (Phase 5, Track 3).

convert.py  — student.onnx -> FP32 OpenVINO IR; MiniLM -> OpenVINO via optimum-cli
quantize.py — NNCF post-training INT8 quantization with REAL calibration embeddings
bench.py    — honest torch-CPU vs OV-FP32 vs OV-INT8 latency/size table

Runtime entry point is OpenVinoStudent, reached ONLY through the seam
(rank_vocabulary._load_student when VOICELINK_STUDENT_BACKEND=openvino).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
STUDENT_FP32_XML = ARTIFACTS_DIR / "student_fp32.xml"
STUDENT_INT8_XML = ARTIFACTS_DIR / "student_int8.xml"
MINILM_OV_DIR = ARTIFACTS_DIR / "minilm-ov"
WHISPER_OV_DIR = ARTIFACTS_DIR / "whisper-base.en-ov-int8"


def student_ir_path() -> Path:
    """The IR the runtime should serve: INT8 when present, FP32 fallback."""
    if STUDENT_INT8_XML.exists():
        return STUDENT_INT8_XML
    if STUDENT_FP32_XML.exists():
        return STUDENT_FP32_XML
    raise FileNotFoundError(
        f"no student IR under {ARTIFACTS_DIR} — run mac/intel/convert.py "
        "(and quantize.py) first"
    )


class OpenVinoStudent:
    """StudentRanker-compatible inference over an OpenVINO IR (CPU plugin).

    Exposes the same logits interface the torch path uses — __call__ returning
    (B, vocab) logits and board_indices() for top-k — so _student_board needs
    no backend-specific code.
    """

    def __init__(self, xml_path: Path) -> None:
        import openvino as ov  # lazy: package importable without openvino
        self.xml_path = Path(xml_path)
        # The ARM CPU plugin defaults to f16 inference (measured: 1.1e-2 logit drift
        # vs onnxruntime). Pin f32 so FP32 means FP32; INT8 quantization is explicit.
        self._compiled = ov.Core().compile_model(
            str(self.xml_path), "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
        self._output = self._compiled.output(0)

    def __call__(self, context_emb) -> np.ndarray:
        """(B, context_dim) -> (B, vocab_size) logits, as float32 ndarray."""
        x = np.asarray(context_emb, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        return self._compiled(x)[self._output]

    def board_indices(self, context_emb, k: int) -> list[int]:
        """Top-k dynamic vocab indices for ONE context embedding (1D or (1,D))."""
        logits = self(context_emb)[0]
        k = min(k, logits.shape[0])
        return np.argsort(-logits)[:k].tolist()
