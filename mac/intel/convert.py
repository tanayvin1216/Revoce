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

from intel import (ARTIFACTS_DIR, MINILM_OV_DIR, STUDENT_FP32_XML,  # noqa: E402
                   minilm_bundle_status)

STUDENT_ONNX = MAC / "evaluation" / "phase4" / "student.onnx"
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _load_vocab() -> list[str]:
    import json as _json
    phase4 = Path(__file__).resolve().parent.parent / "evaluation" / "phase4"
    return _json.loads((phase4 / "vocab.json").read_text())


def _write_vocab_sidecar(xml_path: Path) -> None:
    """Codex P2: stamp the vocab fingerprint next to the IR so a same-sized but
    reordered vocab.json fails loudly at load instead of serving wrong words."""
    from evaluation.phase4.data import vocab_sha
    xml_path.with_suffix(".vocab.sha").write_text(vocab_sha(_load_vocab()))


def _assert_head_matches_vocab(ov_model, n_vocab: int) -> None:
    """The sidecar fingerprints vocab.json, NOT the model — so an ONNX exported
    from a differently-sized student would get a *correct-looking* sidecar and
    only fail later at load, or worse, not at all.

    This is not hypothetical: the 2026-08-24 Core Ultra run committed a 59-class
    student_fp32.xml stamped with the 95-word vocab hash, because run_on_intel.ps1
    -Train had retrained student.pt/.onnx locally while vocab.json stayed at 95.
    Check the width here, at the only point where both are in hand.
    """
    width = ov_model.outputs[0].partial_shape[-1]
    if width.is_dynamic or width.get_length() != n_vocab:
        raise RuntimeError(
            f"student.onnx has a {width}-wide output but vocab.json has {n_vocab} "
            f"words — the ONNX was exported from a different student. Re-export "
            f"with evaluation/phase4/export_onnx.py against this vocab, or restore "
            f"the committed student.onnx (git checkout -- evaluation/phase4/student.onnx). "
            f"Refusing to write an IR that would serve the wrong words.")


def convert_student(onnx_path: Path = STUDENT_ONNX,
                    xml_path: Path = STUDENT_FP32_XML) -> Path:
    """student.onnx -> FP32 OpenVINO IR (.xml + .bin)."""
    import openvino as ov
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    ov_model = ov.convert_model(str(onnx_path))
    _assert_head_matches_vocab(ov_model, len(_load_vocab()))  # before any write
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


MINILM_PROBES = ["what do plants take in from the air", "I need help", "carbon dioxide",
                 "A teacher stands at a whiteboard in a classroom", "yes", "wait"]


def validate_minilm_bundle(out_dir: Path = MINILM_OV_DIR, min_cos: float = 0.999) -> dict:
    """Prove the exported bundle is usable BEFORE any benchmark labels a run with it
    (I-12): every file present, the IR compiles under openvino_tokenizers, an
    encoding succeeds, the seam's embed helper reports 'openvino' as the backend
    actually in use, the vectors are 384-d unit vectors and agree with the torch
    sentence-transformers path (cosine > min_cos). Returns the evidence as a dict;
    raises RuntimeError on the first failed check."""
    import importlib
    import os

    status = minilm_bundle_status(out_dir)
    if not status["complete"]:
        raise RuntimeError(f"MiniLM bundle incomplete: missing {status['missing']} in {out_dir}")
    from intel import OpenVinoEmbedder
    ov_vecs = OpenVinoEmbedder(out_dir)(MINILM_PROBES)
    if ov_vecs.shape != (len(MINILM_PROBES), OpenVinoEmbedder.DIM):
        raise RuntimeError(f"unexpected embedding shape {ov_vecs.shape}")
    norms = np.linalg.norm(ov_vecs, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise RuntimeError(f"embeddings not unit-norm: {norms}")
    # The seam's helper must pick the bundle up too (that is what bench_e2e and
    # rank_vocabulary use), not just the class above with an explicit path.
    prev = os.environ.get("VOICELINK_EMBED_BACKEND")
    os.environ["VOICELINK_EMBED_BACKEND"] = "openvino"
    try:
        import evaluation.embed as em
        em = importlib.reload(em)
        em.cache_clear()
        seam_vecs = em.embed(MINILM_PROBES)
        backend = em.backend_in_use()
    finally:
        if prev is None:
            os.environ.pop("VOICELINK_EMBED_BACKEND", None)
        else:
            os.environ["VOICELINK_EMBED_BACKEND"] = prev
        em.reset_backend()
    if backend != "openvino":
        raise RuntimeError(f"evaluation.embed did not use the bundle (backend_in_use={backend!r})")
    if Path(out_dir).resolve() == MINILM_OV_DIR.resolve() and not np.allclose(seam_vecs, ov_vecs, atol=1e-5):
        raise RuntimeError("seam embeddings differ from the direct OpenVinoEmbedder output")
    result = {"dir": str(out_dir), "files": status["files"], "backend_in_use": backend,
              "dim": int(ov_vecs.shape[1]), "n_probes": len(MINILM_PROBES)}
    try:
        from sentence_transformers import SentenceTransformer
        ref = SentenceTransformer(MINILM_MODEL).encode(MINILM_PROBES, normalize_embeddings=True)
        cos = (np.asarray(ref, dtype=np.float32) * ov_vecs).sum(1)
        result["cos_min_vs_torch"] = float(cos.min())
        result["max_abs_delta_vs_torch"] = float(np.abs(ref - ov_vecs).max())
        if cos.min() <= min_cos:
            raise RuntimeError(f"OpenVINO MiniLM disagrees with torch: cosine min {cos.min():.5f}")
    except ImportError:  # torch-free box: file/compile/encode checks still count
        result["cos_min_vs_torch"] = None
    return result


def main() -> int:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # --skip-student: build ONLY the MiniLM IR. The student IR is committed, so a
    # machine that just needs embeddings (the Intel benchmark box) has no reason to
    # re-export it from whatever student.onnx happens to be on disk.
    if "--skip-student" not in sys.argv:
        xml = convert_student()
        diff = verify_student_ir()
        print(f"student FP32 IR -> {xml}")
        print(f"onnxruntime vs OV-FP32 max abs logit diff = {diff:.2e}")
        if diff >= 1e-4:
            print(">>> student conversion FAILED parity <<<")
            return 1

    # --validate-minilm-only: re-check an existing bundle without re-exporting
    # (what a benchmark or quality run should do before trusting an ov-full label).
    if "--validate-minilm-only" in sys.argv:
        v = validate_minilm_bundle()
        print(f"minilm bundle OK: {v}")
        return 0

    if "--skip-minilm" not in sys.argv:
        r = export_minilm()
        if r.returncode == 0 and (MINILM_OV_DIR / "openvino_model.xml").exists():
            print(f"minilm IR -> {MINILM_OV_DIR}")
        else:
            print(f"minilm export FAILED (rc={r.returncode}):\n{r.stderr[-2000:]}")
            return 1
        # An xml on disk is not a usable bundle (the .bin alone is 90 MB and was
        # the file found missing): validate the whole thing before reporting success.
        try:
            v = validate_minilm_bundle()
        except Exception as e:
            print(f"minilm bundle validation FAILED: {type(e).__name__}: {e}")
            return 1
        print(f"minilm bundle validated: backend={v['backend_in_use']} dim={v['dim']} "
              f"cos_min_vs_torch={v['cos_min_vs_torch']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
