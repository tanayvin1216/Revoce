"""NNCF post-training INT8 quantization of the student IR (Phase 5, Track 3, step 3).

    VOICELINK_PERCEPTION=0 python3 intel/quantize.py

Calibration data is REAL: the 769-d feature vectors (data.row_features — the exact
train/inference featurizer) for every row of v2_scenarios.jsonl, not random noise.
Emits artifacts/int8_accuracy.json with the honest FP32-vs-INT8 numbers:
max |logit delta| over the calibration set and mean top-k_dyn set overlap.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MAC = HERE.parent
if str(MAC) not in sys.path:
    sys.path.insert(0, str(MAC))

from intel import STUDENT_FP32_XML, STUDENT_INT8_XML  # noqa: E402

SCENARIOS = MAC / "evaluation" / "phase4" / "v2_scenarios.jsonl"
ACCURACY_JSON = HERE / "artifacts" / "int8_accuracy.json"


def load_k_dyn() -> int:
    import yaml
    cfg = yaml.safe_load((MAC / "evaluation" / "phase4" / "config.yaml").read_text())
    return int(cfg["board"]["k_dyn"])


def calibration_embeddings(path: Path = SCENARIOS) -> np.ndarray:
    """(N, 769) real feature vectors, one per eval scenario, via data.row_features."""
    from evaluation.phase4 import data as d
    feats = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        feats.append(d.row_features(r.get("scene", ""), r.get("transcript", ""),
                                    bool(r.get("is_question"))))
    return np.stack(feats)


def output_head_matmul(model) -> str:
    """Friendly name of the MatMul that feeds the logits output.

    Found by walking back from the Result node so the name survives re-exports
    (torch.onnx node naming is not stable across versions).
    """
    node = model.get_results()[0]
    while node.get_type_name() != "MatMul":
        node = node.input_value(0).get_node()
    return node.get_friendly_name()


def quantize_student(calib: np.ndarray, fp32_xml: Path = STUDENT_FP32_XML,
                     int8_xml: Path = STUDENT_INT8_XML) -> Path:
    import nncf
    import openvino as ov
    model = ov.Core().read_model(str(fp32_xml))
    # Measured on the 360-row calibration set: quantizing the final vocab
    # projection collapses top-6 overlap to 0.73; excluding it (standard
    # last-layer exclusion) restores 0.95 while the rest of the net stays INT8.
    head = output_head_matmul(model)
    dataset = nncf.Dataset(calib, transform_func=lambda row: row[None, :])
    quantized = nncf.quantize(model, dataset, subset_size=len(calib),
                              ignored_scope=nncf.IgnoredScope(names=[head]))
    ov.save_model(quantized, str(int8_xml), compress_to_fp16=False)
    return int8_xml


def accuracy_note(calib: np.ndarray, k_dyn: int) -> dict:
    """FP32 vs INT8 on the calibration set: max |logit delta| + top-k_dyn overlap."""
    from intel import OpenVinoStudent
    fp32 = OpenVinoStudent(STUDENT_FP32_XML)
    int8 = OpenVinoStudent(STUDENT_INT8_XML)
    lf, li = fp32(calib), int8(calib)
    overlaps = [
        len(set(np.argsort(-a)[:k_dyn]) & set(np.argsort(-b)[:k_dyn])) / k_dyn
        for a, b in zip(lf, li)
    ]
    return {
        "n_calibration_rows": int(calib.shape[0]),
        "k_dyn": k_dyn,
        "max_abs_logit_delta": float(np.abs(lf - li).max()),
        "mean_abs_logit_delta": float(np.abs(lf - li).mean()),
        "mean_topk_overlap": float(np.mean(overlaps)),
        "min_topk_overlap": float(np.min(overlaps)),
    }


def main() -> int:
    calib = calibration_embeddings()
    print(f"calibration set: {calib.shape} from {SCENARIOS.name}")
    xml = quantize_student(calib)
    print(f"student INT8 IR -> {xml}")
    note = accuracy_note(calib, load_k_dyn())
    ACCURACY_JSON.write_text(json.dumps(note, indent=2))
    print(json.dumps(note, indent=2))
    # Review S6: the mean alone averages away a single scenario silently losing
    # a third of its board to quantization noise — gate the worst case too.
    ok = (note["mean_topk_overlap"] >= 0.8
          and note["min_topk_overlap"] >= 0.5)
    print(f">>> INT8 accuracy: {'PASS' if ok else 'FAIL'} "
          f"(mean top-{note['k_dyn']} overlap {note['mean_topk_overlap']:.3f}, "
          f"min {note['min_topk_overlap']:.3f}; need mean>=0.8 AND min>=0.5) <<<")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
