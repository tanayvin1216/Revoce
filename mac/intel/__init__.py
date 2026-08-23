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


def ov_device() -> str:
    """OpenVINO target for the student head. VOICELINK_OV_DEVICE=CPU|GPU|NPU|AUTO.

    CPU is the default everywhere (the dev Mac has only CPU). On an Intel Core
    Ultra box set NPU to put the INT8 head on the AI engine; on a box with an
    Arc GPU set GPU. The IR is the same file either way — that is the point.
    """
    import os
    return os.environ.get("VOICELINK_OV_DEVICE", "CPU").upper()


def hardware_fingerprint() -> dict:
    """Honest machine label for every benchmark row: CPU brand, OS, OpenVINO
    version and the devices the OpenVINO runtime actually sees (so a Tiber
    run records 'Intel(R) Xeon(R) ...' + ['CPU'] and a Core Ultra laptop records
    ['CPU', 'GPU', 'NPU'] without anyone typing it by hand)."""
    import platform
    import subprocess
    cpu = platform.processor() or ""
    try:
        if platform.system() == "Darwin":
            cpu = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        elif platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    info = {"cpu": cpu, "arch": platform.machine(),
            "os": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(), "openvino": None,
            "ov_devices": [], "intel_hardware": "intel" in cpu.lower()}
    try:
        import openvino as ov
        core = ov.Core()
        info["openvino"] = ov.__version__
        info["ov_devices"] = list(core.available_devices)
        info["ov_device_names"] = {
            d: core.get_property(d, "FULL_DEVICE_NAME") for d in core.available_devices}
    except Exception as e:  # pragma: no cover
        info["openvino_error"] = str(e)
    return info


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

    def __init__(self, xml_path: Path, device: str | None = None) -> None:
        import openvino as ov  # lazy: package importable without openvino
        self.xml_path = Path(xml_path)
        self.device = device or ov_device()
        # The ARM CPU plugin defaults to f16 inference (measured: 1.1e-2 logit drift
        # vs onnxruntime). Pin f32 so FP32 means FP32; INT8 quantization is explicit.
        # (The hint is CPU-only; NPU/GPU plugins reject unknown properties.)
        cfg = {"INFERENCE_PRECISION_HINT": "f32"} if self.device == "CPU" else {}
        self._compiled = ov.Core().compile_model(str(self.xml_path), self.device, cfg)
        self._output = self._compiled.output(0)
        # Output dim of the head == vocab size; used by the seam's stale-artifact
        # guard (review S5).
        self.vocab_size = int(self._output.get_partial_shape()[-1].get_length())

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


class OpenVinoEmbedder:
    """all-MiniLM-L6-v2 sentence embeddings over OpenVINO (tokenizer + encoder
    + pooling all inside the IR, torch-free).

    Drop-in for evaluation.embed.embed(): returns (N, 384) L2-normalised float32.
    Measured on Apple Silicon (ARM CPU plugin): cosine 1.000 vs the torch
    sentence-transformers path on every probe, ~2.9x faster per single text.
    This is what makes VOICELINK_STUDENT_BACKEND=openvino a FULLY OpenVINO
    student path (embed -> MLP -> symbolic MMR), not just the MLP head.
    """

    DIM = 384

    def __init__(self, model_dir: Path = MINILM_OV_DIR) -> None:
        # openvino_tokenizers must be imported BEFORE Core() so its custom opset
        # (SpecialTokensSplit etc.) is registered; otherwise compile_model fails.
        import openvino_tokenizers  # noqa: F401
        import openvino as ov
        model_dir = Path(model_dir)
        xml = model_dir / "openvino_model.xml"
        if not (xml.exists() and xml.with_suffix(".bin").exists()):
            raise FileNotFoundError(
                f"MiniLM IR missing under {model_dir} — run mac/intel/convert.py")
        core = ov.Core()
        self._tok = core.compile_model(str(model_dir / "openvino_tokenizer.xml"), "CPU")
        self._enc = core.compile_model(str(xml), "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
        self._enc_inputs = {i.any_name for i in self._enc.inputs}
        # optimum's sentence_transformers export bakes mean-pooling in as the
        # 'sentence_embedding' output; fall back to pooling token_embeddings.
        outs = {o.any_name for o in self._enc.outputs}
        self._pooled_out = "sentence_embedding" if "sentence_embedding" in outs else None

    def __call__(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.DIM), dtype=np.float32)
        t = self._tok(np.array(list(texts)))
        feed = {k.any_name: t[k] for k in t if k.any_name in self._enc_inputs}
        res = self._enc(feed)
        if self._pooled_out is not None:
            vec = res[self._enc.output(self._pooled_out)]
        else:
            tok_emb = res[self._enc.output("token_embeddings")]
            mask = feed["attention_mask"][..., None].astype(np.float32)
            vec = (tok_emb * mask).sum(1) / np.maximum(mask.sum(1), 1e-9)
        vec = np.asarray(vec, dtype=np.float32)
        return vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12)
