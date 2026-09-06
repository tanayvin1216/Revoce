"""I-07b — train-only INT8 calibration of the I-05b (matched 95-entry, train-only preprocessing) checkpoint.

This is I-07_requantise_implement.py applied to the I-05b bundle. The chain, calibration policy (391 training
contexts only), candidate set, validation-only selection rule and every measurement are unchanged; the only
edits are the issue label, an I-05b bundle identity check (95 entries, matched-vocab preprocessing tag) and the
recorded provenance of the I-07 script this file was copied from.

Run from mac/:
    VOICELINK_PERCEPTION=0 /usr/local/bin/python3 ../paper/odi/revision/analyses/I-07b_requantise_implement.py \
        --bundle ../paper/odi/revision/analyses/I-05b_split_first \
        --out ../paper/odi/revision/analyses/I-07b_artifacts

What it does, in order (every step is logged into <out>/manifest.json):
  1. verify the I-05 bundle hashes against its own manifest, load rows/splits/vocab/student.pt;
  2. export the checkpoint to ONNX (opset 17, dynamic batch) and check onnxruntime vs eager;
  3. convert to an FP32 OpenVINO IR, stamp the vocab sha sidecar, check IR vs onnxruntime;
  4. build the calibration set from the 391 TRAINING-split contexts only (row_features, the exact
     train/inference featuriser) and assert it is disjoint from validation and test; also record the
     join of the 360 v2 scenarios onto the split (the historical calibration set) for the checker;
  5. quantise several NNCF candidates (all calibrated on the training features) and choose the
     quantisation exclusion on the VALIDATION contexts only, with a pre-registered rule;
  6. save the chosen INT8 IR (+ sidecar), then measure FP32-vs-INT8 agreement on train / val / test
     and the board-level agreement of the exported heads with the torch checkpoint on the 84 test
     rows, through the production `infer.student_board` selector;
  7. dump torch / OV-FP32 / OV-INT8 logits for all 559 rows as .npy so a checker without OpenVINO
     can re-derive every agreement number.

Nothing under mac/ is written. The output directory must not exist. No historical artifact is touched.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

MAC = Path(__file__).resolve().parents[4] / "mac"
sys.path.insert(0, str(MAC))
os.environ.update(VOICELINK_PERCEPTION="0", HF_HUB_OFFLINE="1", VOICELINK_EMBED_BACKEND="torch",
                  VOICELINK_STUDENT_BACKEND="torch", VOICELINK_DEVICE="cpu")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from evaluation.phase4 import data as d  # noqa: E402
from evaluation.phase4 import infer  # noqa: E402
from evaluation.phase4.student import StudentRanker  # noqa: E402
from evaluation.embed import assert_backend  # noqa: E402
from rank_vocabulary import GUARANTEED_CORE  # noqa: E402
from intel import OpenVinoStudent, hardware_fingerprint  # noqa: E402
from intel.quantize import output_head_matmul  # noqa: E402

K_DYN = 6
K_BOARD = 10
# Pre-registered validation-only selection rule for the quantisation configuration:
#   gate: mean top-6 overlap >= 0.8 AND min top-6 overlap >= 0.5 on the VALIDATION contexts
#   (the repository's historical accuracy gate, quantize.py:main);
#   among candidates passing the gate: highest validation mean top-6 overlap, then highest min,
#   then smallest .bin. If no candidate passes, the head-excluded candidate is the fallback and
#   the failure is reported.
GATE_MEAN, GATE_MIN = 0.8, 0.5


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_json(p: Path, x) -> None:
    with p.open("x") as f:
        json.dump(x, f, indent=2)


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    return np.array([len(set(np.argsort(-x)[:k]) & set(np.argsort(-y)[:k])) / k
                     for x, y in zip(a, b)], dtype=np.float64)


def agreement(ref: np.ndarray, got: np.ndarray, k: int) -> dict:
    ov = topk_overlap(ref, got, k)
    delta = np.abs(ref - got)
    return {"n": int(ref.shape[0]), "k": k,
            "max_abs_logit_delta": float(delta.max()),
            "mean_abs_logit_delta": float(delta.mean()),
            "mean_topk_overlap": float(ov.mean()), "min_topk_overlap": float(ov.min()),
            "n_rows_below_0.5": int((ov < 0.5).sum()),
            "top1_agreement": float((ref.argmax(1) == got.argmax(1)).mean())}


def board_jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = {w.lower() for w in a}, {w.lower() for w in b}
    return len(sa & sb) / len(sa | sb)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    bundle, out = args.bundle.resolve(), args.out.resolve()
    out.mkdir(exist_ok=False)
    (out / "candidates").mkdir()
    t0 = time.perf_counter()
    log: dict = {"issue": "I-07b", "bundle": str(bundle.relative_to(MAC.parent)), "steps": {},
                 "derived_from_script": "paper/odi/revision/analyses/I-07_requantise_implement.py",
                 "derived_from_script_sha256": sha(Path(__file__).resolve().with_name("I-07_requantise_implement.py"))}

    import onnx
    import onnxruntime as ort
    import openvino as ov
    import nncf

    # ---- 1. bundle identity --------------------------------------------------------------
    man = json.loads((bundle / "manifest.json").read_text())
    for name, h in man["artifact_sha256"].items():
        assert sha(bundle / name) == h, f"I-05 bundle drifted: {name}"
    rows = json.loads((bundle / "rows.json").read_text())
    splits = json.loads((bundle / "splits.json").read_text())
    vocab = json.loads((bundle / "vocab.json").read_text())
    assert splits["fingerprint"] == d.rows_fingerprint(rows) == "ab51f7fba4c9c39e"
    assert [len(splits[k]) for k in ("train", "val", "test")] == [391, 84, 84]
    assert splits == json.loads(d.SPLITS_PATH.read_text()), "bundle split != historical splits.json"
    ck = torch.load(bundle / "student.pt", map_location="cpu", weights_only=False)
    assert ck["vocab_sha"] == d.vocab_sha(vocab) == man["vocab_sha"] and ck["vocab_size"] == len(vocab)
    # I-05b identity: the matched-vocabulary control has exactly 95 entries (historical width) and the
    # train-only matched-vocab preprocessing tag; refuse any other bundle so numbers cannot be mixed.
    cfg = json.loads((bundle / "config.json").read_text())
    assert man["issue"] == "I-05b" and len(vocab) == 95 and cfg["preprocessing"] == "train-only-v1-matched-vocab", \
        f"not the I-05b bundle: issue={man.get('issue')} vocab={len(vocab)} prep={cfg.get('preprocessing')}"
    assert man["config_hash"] == "b72bb3ec7254" and man["vocab_sha"] == "0f3bb094cd439111", "I-05b identity drifted"
    assert cfg["vocab_override"] == {"min_freq": 2, "max_size": 116, "dedup_tau": 0.8}
    torch.set_num_threads(1)
    model = StudentRanker(ck["context_dim"], ck["hidden_dim"], ck["vocab_size"], ck["dropout"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    log["steps"]["bundle"] = {"manifest_sha256": sha(bundle / "manifest.json"),
                              "student_pt_sha256": sha(bundle / "student.pt"),
                              "vocab_sha": ck["vocab_sha"], "vocab_size": len(vocab),
                              "i05b_config_hash": man["config_hash"], "i05b_vocab_setting": man["vocab_setting"],
                              "split_fingerprint": splits["fingerprint"]}

    # ---- 2. ONNX export --------------------------------------------------------------------
    onnx_path = out / "student.onnx"
    dummy = torch.zeros(1, model.context_dim, dtype=torch.float32)
    torch.onnx.export(model, dummy, str(onnx_path), input_names=["context_emb"], output_names=["logits"],
                      dynamic_axes={"context_emb": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x_rand = np.random.default_rng(0).standard_normal((4, model.context_dim)).astype(np.float32)
    with torch.no_grad():
        eager_rand = model(torch.from_numpy(x_rand)).numpy()
    ort_rand = sess.run(["logits"], {"context_emb": x_rand})[0]
    assert ort_rand.shape[-1] == len(vocab)

    # ---- features for all 559 rows (exact train/inference featuriser) -----------------------
    X = np.stack([d.row_features(r["scene"], r["transcript"], bool(r["is_question"])) for r in rows])
    assert X.shape == (559, 769)
    X_batched = d.context_features(rows)
    feat_delta = float(np.abs(X - X_batched).max())  # the two featurisers must agree
    with torch.no_grad():
        L_torch = model(torch.from_numpy(X)).numpy()
    L_ort = sess.run(["logits"], {"context_emb": X})[0]
    log["steps"]["onnx"] = {"opset": 17, "dynamo": False, "sha256": sha(onnx_path),
                            "bytes": onnx_path.stat().st_size,
                            "ort_vs_eager_max_abs_random4": float(np.abs(ort_rand - eager_rand).max()),
                            "ort_vs_eager_max_abs_559rows": float(np.abs(L_ort - L_torch).max()),
                            "row_features_vs_context_features_max_abs": feat_delta}

    # ---- 3. FP32 IR ------------------------------------------------------------------------
    fp32_xml = out / "student_fp32.xml"
    ov_model = ov.convert_model(str(onnx_path))
    width = ov_model.outputs[0].partial_shape[-1]
    assert (not width.is_dynamic) and width.get_length() == len(vocab), "IR head width != vocab"
    ov.save_model(ov_model, str(fp32_xml), compress_to_fp16=False)
    fp32_xml.with_suffix(".vocab.sha").write_text(d.vocab_sha(vocab))
    fp32 = OpenVinoStudent(fp32_xml, device="CPU")
    assert fp32.vocab_size == len(vocab)
    L_fp32 = fp32(X)
    log["steps"]["fp32_ir"] = {"xml_sha256": sha(fp32_xml), "bin_sha256": sha(out / "student_fp32.bin"),
                               "bytes_xml": fp32_xml.stat().st_size,
                               "bytes_bin": (out / "student_fp32.bin").stat().st_size,
                               "ov_fp32_vs_ort_max_abs_random4": float(np.abs(fp32(x_rand) - ort_rand).max()),
                               "ov_fp32_vs_torch_max_abs_559rows": float(np.abs(L_fp32 - L_torch).max()),
                               "ov_fp32_vs_torch_top6_overlap_559rows": agreement(L_torch, L_fp32, K_DYN)}

    # ---- 4. calibration set: TRAINING contexts only ----------------------------------------
    train_ids, val_ids, test_ids = splits["train"], splits["val"], splits["test"]
    assert not (set(train_ids) & set(val_ids)) and not (set(train_ids) & set(test_ids))
    calib_ids = list(train_ids)                     # order as in splits.json
    X_calib = X[calib_ids]
    heldout_ctx = {d.context_text(rows[i]) for i in val_ids + test_ids}
    calib_ctx = [d.context_text(rows[i]) for i in calib_ids]
    assert not (set(calib_ctx) & heldout_ctx), "calibration context appears in val/test"
    # historical calibration set (all 360 v2 scenarios) joined onto the split, for the checker
    v2 = [json.loads(ln) for ln in (MAC / "evaluation/phase4/v2_scenarios.jsonl").read_text().splitlines() if ln.strip()]
    ctx2split = {}
    for name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        for i in ids:
            ctx2split[d.context_text(rows[i])] = name
    v2_join = {"train": 0, "val": 0, "test": 0, "unmatched": 0}
    for s in v2:
        key = d.context_text({"scene": s.get("scene", ""), "transcript": s.get("transcript", "")})
        v2_join[ctx2split.get(key, "unmatched")] += 1
    calib_record = {
        "policy": "all 391 training-split row contexts; validation and test contexts excluded",
        "n_calibration": len(calib_ids), "row_indices": calib_ids,
        "context_sha256": [hashlib.sha256(c.encode()).hexdigest()[:16] for c in calib_ctx],
        "disjoint_from_val_and_test": True,
        "historical_v2_360_join_onto_split": v2_join,
        "n_calibration_rows_that_are_v2_scenarios": sum(1 for c in calib_ctx if c in {
            d.context_text({"scene": s.get("scene", ""), "transcript": s.get("transcript", "")}) for s in v2}),
        "feature_sha256": hashlib.sha256(X_calib.tobytes()).hexdigest(),
    }
    write_json(out / "calibration_ids.json", calib_record)
    log["steps"]["calibration"] = {k: v for k, v in calib_record.items()
                                   if k not in ("row_indices", "context_sha256")}

    # ---- 5. NNCF candidates, chosen on VALIDATION only ---------------------------------------
    core = ov.Core()
    base = core.read_model(str(fp32_xml))
    head = output_head_matmul(base)
    dataset = nncf.Dataset(X_calib, transform_func=lambda row: row[None, :])
    candidates = {
        "full_int8_performance": dict(preset=nncf.QuantizationPreset.PERFORMANCE, ignored=[]),
        "full_int8_mixed": dict(preset=nncf.QuantizationPreset.MIXED, ignored=[]),
        "head_excluded_performance": dict(preset=nncf.QuantizationPreset.PERFORMANCE, ignored=[head]),
        "head_excluded_mixed": dict(preset=nncf.QuantizationPreset.MIXED, ignored=[head]),
    }
    cand_results = {}
    L_val_fp32 = L_fp32[val_ids]
    for name, spec in candidates.items():
        tq = time.perf_counter()
        m = core.read_model(str(fp32_xml))
        q = nncf.quantize(m, dataset, preset=spec["preset"], subset_size=len(calib_ids),
                          ignored_scope=nncf.IgnoredScope(names=spec["ignored"]) if spec["ignored"] else None)
        xml = out / "candidates" / f"{name}.xml"
        ov.save_model(q, str(xml), compress_to_fp16=False)
        runner = OpenVinoStudent(xml, device="CPU")
        L_q = runner(X)
        np.save(out / "candidates" / f"{name}_logits.npy", L_q)
        cand_results[name] = {
            "preset": str(spec["preset"].value), "ignored_scope": spec["ignored"],
            "quantize_wall_s": time.perf_counter() - tq,
            "bytes_bin": xml.with_suffix(".bin").stat().st_size, "bytes_xml": xml.stat().st_size,
            "xml_sha256": sha(xml), "bin_sha256": sha(xml.with_suffix(".bin")),
            "validation_agreement_vs_ov_fp32": agreement(L_val_fp32, L_q[val_ids], K_DYN),
            "train_agreement_vs_ov_fp32": agreement(L_fp32[train_ids], L_q[train_ids], K_DYN),
        }
        print(f"[candidate] {name}: val mean/min top-6 overlap "
              f"{cand_results[name]['validation_agreement_vs_ov_fp32']['mean_topk_overlap']:.4f}/"
              f"{cand_results[name]['validation_agreement_vs_ov_fp32']['min_topk_overlap']:.4f}, "
              f"bin {cand_results[name]['bytes_bin']} B")

    def key(name):
        v = cand_results[name]["validation_agreement_vs_ov_fp32"]
        return (v["mean_topk_overlap"], v["min_topk_overlap"], -cand_results[name]["bytes_bin"])
    passing = [n for n, r in cand_results.items()
               if r["validation_agreement_vs_ov_fp32"]["mean_topk_overlap"] >= GATE_MEAN
               and r["validation_agreement_vs_ov_fp32"]["min_topk_overlap"] >= GATE_MIN]
    chosen = max(passing, key=key) if passing else "head_excluded_performance"
    selection = {"rule": "validation-only: gate mean>=0.8 & min>=0.5 top-6 overlap vs OV-FP32; then max mean, max min, min bytes",
                 "candidates_passing_gate": passing, "chosen": chosen,
                 "test_contexts_used_in_selection": False}
    log["steps"]["quantisation_candidates"] = cand_results
    log["steps"]["selection"] = selection
    # promote the chosen candidate to the canonical INT8 artifact names
    int8_xml = out / "student_int8.xml"
    cxml = out / "candidates" / f"{chosen}.xml"
    # the .xml references its .bin by its own file name, so re-save through OpenVINO instead of a raw copy
    q_model = core.read_model(str(cxml))
    ov.save_model(q_model, str(int8_xml), compress_to_fp16=False)
    int8_xml.with_suffix(".vocab.sha").write_text(d.vocab_sha(vocab))
    int8 = OpenVinoStudent(int8_xml, device="CPU")
    assert int8.vocab_size == len(vocab)
    L_int8 = int8(X)
    assert np.array_equal(L_int8, np.load(out / "candidates" / f"{chosen}_logits.npy")), "re-saved IR differs"

    # ---- 6. agreement after the configuration is frozen -------------------------------------
    agree = {name: agreement(L_fp32[ids], L_int8[ids], K_DYN)
             for name, ids in (("train_calibration", train_ids), ("val_selection", val_ids), ("test_heldout", test_ids))}
    agree["all_559"] = agreement(L_fp32, L_int8, K_DYN)
    agree["int8_vs_torch_test_heldout"] = agreement(L_torch[test_ids], L_int8[test_ids], K_DYN)
    # board-level agreement on the 84 test rows through the production selector
    boards = []
    jac_int8, jac_fp32, exact_int8, exact_fp32 = [], [], 0, 0
    ref_boards = {b["row_index"]: b["student_board"] for b in json.loads((bundle / "boards.json").read_text())}
    for i in test_ids:
        r = rows[i]
        bt = infer.student_board(model, vocab, r["scene"], r["transcript"], r["is_question"], K_BOARD, K_DYN, GUARANTEED_CORE)
        assert bt == ref_boards[i], f"torch board differs from I-05 boards.json at row {i}"
        bf = infer.student_board(fp32, vocab, r["scene"], r["transcript"], r["is_question"], K_BOARD, K_DYN, GUARANTEED_CORE)
        bi = infer.student_board(int8, vocab, r["scene"], r["transcript"], r["is_question"], K_BOARD, K_DYN, GUARANTEED_CORE)
        jac_int8.append(board_jaccard(bt, bi)); jac_fp32.append(board_jaccard(bt, bf))
        exact_int8 += bt == bi; exact_fp32 += bt == bf
        boards.append({"row_index": i, "torch_board": bt, "ov_fp32_board": bf, "ov_int8_board": bi,
                       "jaccard_int8_vs_torch": jac_int8[-1], "jaccard_fp32_vs_torch": jac_fp32[-1]})
    write_json(out / "boards_test_by_backend.json", boards)
    board_agree = {"n": len(test_ids), "embedding_backend": "torch", "selector": "infer.student_board (production)",
                   "ov_int8_vs_torch": {"mean_jaccard": float(np.mean(jac_int8)), "min_jaccard": float(np.min(jac_int8)),
                                        "identical_boards": int(exact_int8),
                                        "dynamic_jaccard_mean": float(np.mean([board_jaccard(b["torch_board"][4:], b["ov_int8_board"][4:]) for b in boards]))},
                   "ov_fp32_vs_torch": {"mean_jaccard": float(np.mean(jac_fp32)), "min_jaccard": float(np.min(jac_fp32)),
                                        "identical_boards": int(exact_fp32)}}
    log["steps"]["agreement_after_freeze"] = agree
    log["steps"]["board_agreement_test"] = board_agree
    assert assert_backend("torch") == "torch"

    # ---- 7. logits dump, sizes, manifest -----------------------------------------------------
    np.save(out / "features_559x769.npy", X)
    np.save(out / "logits_torch.npy", L_torch)
    np.save(out / "logits_ov_fp32.npy", L_fp32)
    np.save(out / "logits_ov_int8.npy", L_int8)
    sizes = {n: (out / n).stat().st_size for n in ("student.onnx", "student_fp32.xml", "student_fp32.bin",
                                                   "student_int8.xml", "student_int8.bin")}
    sizes["student_pt_i05b_bundle"] = (bundle / "student.pt").stat().st_size
    sizes["fp32_ir_total"] = sizes["student_fp32.xml"] + sizes["student_fp32.bin"]
    sizes["int8_ir_total"] = sizes["student_int8.xml"] + sizes["student_int8.bin"]
    sizes["historical_fp32_ir_total"] = sum((MAC / "intel/artifacts" / n).stat().st_size for n in ("student_fp32.xml", "student_fp32.bin"))
    sizes["historical_int8_ir_total"] = sum((MAC / "intel/artifacts" / n).stat().st_size for n in ("student_int8.xml", "student_int8.bin"))
    log["steps"]["sizes_bytes"] = sizes
    log["wall_seconds"] = time.perf_counter() - t0
    log["versions"] = {x: importlib.metadata.version(x) for x in ("torch", "numpy", "openvino", "nncf", "onnx", "onnxruntime", "sentence-transformers", "transformers")}
    log["hardware"] = hardware_fingerprint()
    log["python"] = platform.python_version()
    inputs = [bundle / n for n in ("manifest.json", "student.pt", "vocab.json", "canonical.json", "config.json", "splits.json", "rows.json", "boards.json")]
    inputs += [MAC / x for x in ("evaluation/phase4/data.py", "evaluation/phase4/infer.py", "evaluation/phase4/student.py",
                                 "evaluation/phase4/v2_scenarios.jsonl", "evaluation/phase4/splits.json", "intel/__init__.py",
                                 "intel/quantize.py", "evaluation/embed.py", "selection.py", "evaluation/phase3/taxonomy.py",
                                 "evaluation/phase4/answers.py", "evaluation/hedges.py")]
    inputs.append(Path(__file__).resolve())
    log["input_sha256"] = {str(x.relative_to(MAC.parent)): sha(x) for x in inputs}
    log["artifact_sha256"] = {str(x.relative_to(out)): sha(x) for x in sorted(out.rglob("*")) if x.is_file()}
    log["references_i05b_manifest"] = man["config_hash"]
    write_json(out / "manifest.json", log)
    print(json.dumps({k: v for k, v in log["steps"].items() if k != "quantisation_candidates"}, indent=2))
    print(f"wall {log['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
