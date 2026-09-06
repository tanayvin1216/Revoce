"""I-07 torch-only verification for a checker without a working OpenVINO runtime.

Run from mac/:
    VOICELINK_PERCEPTION=0 /usr/local/bin/python3 ../paper/odi/revision/analyses/I-07_verify_from_dumps.py \
        --export ../paper/odi/revision/analyses/I-07_export_v1 [--out ../paper/odi/revision/analyses/I-07_verify_<tag>.json]

Checks, none of which need openvino:
  * every artifact hash in <export>/manifest.json matches the file on disk;
  * calibration_ids.json: 391 ids == the training split of the historical splits.json, disjoint from
    val/test, context hashes match the I-05 rows, feature hash matches the dumped features;
  * the 360 v2 scenarios join onto the split as 255/54/51 (the historical calibration overlap);
  * features_559x769.npy equals a fresh data.row_features() pass (torch MiniLM);
  * logits_torch.npy equals a fresh forward of the I-05 checkpoint on those features;
  * onnxruntime on student.onnx agrees with logits_torch.npy (onnxruntime needs no OpenVINO);
  * the agreement numbers in manifest.json (train/val/test/all, OV-FP32 vs OV-INT8, INT8 vs torch)
    are recomputed from the dumped logits;
  * the validation-only selection rule reproduces the chosen candidate from the candidate logits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

MAC = Path(__file__).resolve().parents[4] / "mac"
sys.path.insert(0, str(MAC))
os.environ.update(VOICELINK_PERCEPTION="0", HF_HUB_OFFLINE="1", VOICELINK_EMBED_BACKEND="torch",
                  VOICELINK_STUDENT_BACKEND="torch", VOICELINK_DEVICE="cpu")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from evaluation.phase4 import data as d  # noqa: E402
from evaluation.phase4.student import StudentRanker  # noqa: E402


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def topk_overlap(a, b, k):
    return np.array([len(set(np.argsort(-x)[:k]) & set(np.argsort(-y)[:k])) / k for x, y in zip(a, b)])


def agreement(ref, got, k=6):
    ov = topk_overlap(ref, got, k)
    return {"max_abs_logit_delta": float(np.abs(ref - got).max()), "mean_abs_logit_delta": float(np.abs(ref - got).mean()),
            "mean_topk_overlap": float(ov.mean()), "min_topk_overlap": float(ov.min()),
            "top1_agreement": float((ref.argmax(1) == got.argmax(1)).mean())}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--export", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    ex = a.export.resolve()
    man = json.loads((ex / "manifest.json").read_text())
    bundle = MAC.parent / man["bundle"]
    report: dict = {"export": str(ex), "checks": {}}
    ok = True

    def check(name, cond, detail=None):
        nonlocal ok
        report["checks"][name] = {"pass": bool(cond), **({"detail": detail} if detail is not None else {})}
        ok = ok and bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail is not None else ""))

    # artifact hashes
    bad = [n for n, h in man["artifact_sha256"].items() if sha(ex / n) != h]
    check("artifact_hashes", not bad, bad or f"{len(man['artifact_sha256'])} files")
    badin = [n for n, h in man["input_sha256"].items() if sha(MAC.parent / n) != h]
    check("input_hashes_unchanged", not badin, badin or f"{len(man['input_sha256'])} inputs")

    rows = json.loads((bundle / "rows.json").read_text())
    splits = json.loads(d.SPLITS_PATH.read_text())
    calib = json.loads((ex / "calibration_ids.json").read_text())
    ids = calib["row_indices"]
    check("calibration_is_exactly_training_split", ids == splits["train"] and len(ids) == 391)
    check("calibration_disjoint_from_val_test", not (set(ids) & set(splits["val"])) and not (set(ids) & set(splits["test"])))
    ctx = [d.context_text(rows[i]) for i in ids]
    check("calibration_context_hashes", [hashlib.sha256(c.encode()).hexdigest()[:16] for c in ctx] == calib["context_sha256"])
    held = {d.context_text(rows[i]) for i in splits["val"] + splits["test"]}
    check("calibration_contexts_not_in_heldout_text", not (set(ctx) & held))
    v2 = [json.loads(ln) for ln in (MAC / "evaluation/phase4/v2_scenarios.jsonl").read_text().splitlines() if ln.strip()]
    ctx2split = {d.context_text(rows[i]): s for s in ("train", "val", "test") for i in splits[s]}
    join = {"train": 0, "val": 0, "test": 0, "unmatched": 0}
    for s in v2:
        join[ctx2split.get(d.context_text({"scene": s.get("scene", ""), "transcript": s.get("transcript", "")}), "unmatched")] += 1
    check("historical_v2_join_255_54_51", join == {"train": 255, "val": 54, "test": 51, "unmatched": 0}, join)

    X = np.load(ex / "features_559x769.npy")
    check("calibration_feature_hash", hashlib.sha256(X[ids].tobytes()).hexdigest() == calib["feature_sha256"])
    X_fresh = np.stack([d.row_features(r["scene"], r["transcript"], bool(r["is_question"])) for r in rows])
    check("features_match_fresh_row_features", np.abs(X - X_fresh).max() < 1e-6, float(np.abs(X - X_fresh).max()))

    ck = torch.load(bundle / "student.pt", map_location="cpu", weights_only=False)
    model = StudentRanker(ck["context_dim"], ck["hidden_dim"], ck["vocab_size"], ck["dropout"])
    model.load_state_dict(ck["state_dict"]); model.eval(); torch.set_num_threads(1)
    with torch.no_grad():
        L_fresh = model(torch.from_numpy(X)).numpy()
    L_torch = np.load(ex / "logits_torch.npy")
    check("torch_logits_match_fresh_forward", np.abs(L_fresh - L_torch).max() < 1e-5, float(np.abs(L_fresh - L_torch).max()))
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(ex / "student.onnx"), providers=["CPUExecutionProvider"])
        L_ort = sess.run(["logits"], {"context_emb": X})[0]
        check("onnx_matches_torch", np.abs(L_ort - L_torch).max() < 1e-4, float(np.abs(L_ort - L_torch).max()))
        check("onnx_head_width_is_vocab", L_ort.shape[1] == ck["vocab_size"] == len(json.loads((bundle / "vocab.json").read_text())))
    except ImportError:
        report["checks"]["onnx_matches_torch"] = {"pass": None, "detail": "onnxruntime not installed"}

    L_fp32, L_int8 = np.load(ex / "logits_ov_fp32.npy"), np.load(ex / "logits_ov_int8.npy")
    check("ov_fp32_matches_torch", np.abs(L_fp32 - L_torch).max() < 1e-4, float(np.abs(L_fp32 - L_torch).max()))
    rep = man["steps"]["agreement_after_freeze"]
    for name, sel in (("train_calibration", splits["train"]), ("val_selection", splits["val"]), ("test_heldout", splits["test"]), ("all_559", list(range(559)))):
        got = agreement(L_fp32[sel], L_int8[sel])
        diffs = {k: abs(got[k] - rep[name][k]) for k in got}
        check(f"agreement_{name}", max(diffs.values()) < 1e-9, {k: round(got[k], 6) for k in got})
    got = agreement(L_torch[splits["test"]], L_int8[splits["test"]])
    check("agreement_int8_vs_torch_test", max(abs(got[k] - rep["int8_vs_torch_test_heldout"][k]) for k in got) < 1e-9, {k: round(got[k], 6) for k in got})

    # selection rule from the candidate logits
    cands = man["steps"]["quantisation_candidates"]
    val = splits["val"]
    scores = {}
    for name in cands:
        Lc = np.load(ex / "candidates" / f"{name}_logits.npy")
        ag = agreement(L_fp32[val], Lc[val])
        scores[name] = (ag["mean_topk_overlap"], ag["min_topk_overlap"], -cands[name]["bytes_bin"])
    passing = [n for n, s in scores.items() if s[0] >= 0.8 and s[1] >= 0.5]
    chosen = max(passing, key=lambda n: scores[n]) if passing else "head_excluded_performance"
    check("selection_rule_reproduces_choice", chosen == man["steps"]["selection"]["chosen"],
          {n: [round(s[0], 4), round(s[1], 4)] for n, s in scores.items()})
    check("chosen_logits_equal_promoted_int8", np.array_equal(np.load(ex / "candidates" / f"{chosen}_logits.npy"), L_int8))

    report["all_pass"] = ok
    if a.out:
        with a.out.open("x") as f:
            json.dump(report, f, indent=2)
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
