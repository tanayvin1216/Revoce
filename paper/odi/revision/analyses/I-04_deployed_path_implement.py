"""I-04 — held-out Table 2 quality through the ACTUAL deployed backends, joined with latency.

Two phases, deliberately in separate processes so the embedding backend that GENERATES a board can
never leak into the fixed torch ruler that SCORES it:

  generate  — one (family, head, embedder) configuration per process. Loads the head (torch checkpoint,
              OpenVINO FP32 IR, or OpenVINO INT8 IR) and the embedder (torch sentence-transformers or the
              OpenVINO MiniLM IR, strict — a torch fallback is an error, never a mislabelled row), builds
              the 84 held-out boards BOTH through the seam (`rank_vocabulary(..., strategy="student")`)
              and through the production selector directly (`infer.student_board`), asserts the backends
              that actually served, and writes the boards + head logits + hashes to a JSON file.
  score     — torch embedder forced and asserted; every boards file is scored with the unedited
              `evaluate.held_out_eval` (I-03's explicit windows: full@10, dynamic-only@6, full@7),
              cross-checked against I-03 / I-05 / I-07 stored boards, compared with the family's
              torch/torch reference board-by-board (Jaccard, identical boards, paired dynamic-coverage
              difference with a bootstrap CI), and joined with the archived latency rows.

Families are never mixed: `hist` = the historical 95-entry checkpoint and the IRs under
mac/intel/artifacts (calibration overlaps 51/84 test contexts — I-07); `i05` = the train-only 45-entry
I-05 checkpoint and the I-07 train-only-calibrated IRs. Nothing under mac/ is written; every output is
opened exclusively under paper/odi/revision/analyses/.

Run from mac/ with /usr/local/bin/python3 (see I-04_result.md for the exact command list).
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

HERE = Path(__file__).resolve().parent            # paper/odi/revision/analyses
REPO = HERE.parents[3]
MAC = REPO / "mac"
if str(MAC) not in sys.path:
    sys.path.insert(0, str(MAC))

I05_BUNDLE = HERE / "I-05_split_first_v2"
I07_EXPORT = HERE / "I-07_export_v1"
I03_NUMBERS = HERE / "I-03_numbers_implement_v2.json"
E2E_MAC = MAC / "intel" / "artifacts" / "e2e_mac.json"
K_BOARD, K_DYN = 10, 6
WINDOWS = [("full10", 10, False), ("dyn6", 6, True), ("full7", 7, False)]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_json(p: Path, x) -> None:
    with p.open("x") as f:                      # exclusive: never overwrite
        json.dump(x, f, indent=1)


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = {w.lower() for w in a}, {w.lower() for w in b}
    return len(sa & sb) / max(len(sa | sb), 1)


# ----------------------------------------------------------------------------------------------
# phase 1: generate
# ----------------------------------------------------------------------------------------------

def cmd_generate(args) -> None:
    head_backend = "openvino" if args.head.startswith("ov") else "torch"
    # Env BEFORE any mac/ import: ranker_config and embed read it at import/call time.
    os.environ.update(VOICELINK_PERCEPTION="0", HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                      VOICELINK_DEVICE="cpu", VOICELINK_EMBED_BACKEND=args.embed,
                      VOICELINK_STUDENT_BACKEND=head_backend, VOICELINK_EMBED_STRICT="1")
    t_start = time.perf_counter()
    import numpy as np
    import torch
    torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(1)

    import ranker_config  # noqa: F401  (reads VOICELINK_STUDENT_BACKEND now)
    import rank_vocabulary as rv
    from evaluation import embed as em
    from evaluation.phase4 import data as d, evaluate as e, infer
    from evaluation.phase4.student import StudentRanker
    from intel import OpenVinoStudent, hardware_fingerprint, minilm_bundle_status
    import intel

    if args.embed == "openvino":
        st = minilm_bundle_status()
        assert st["complete"], f"MiniLM bundle incomplete: {st}"

    core = list(rv.GUARANTEED_CORE)
    base = MAC / "evaluation" / "phase4"
    inputs: dict[str, str] = {}
    loader = "injected"
    if args.family == "hist":
        vocab, _ = d.load_vocab()
        ck = torch.load(base / "student.pt", map_location="cpu", weights_only=False)
        assert ck["config_hash"] == "d6ce2e56d50b" and len(vocab) == 95 and ck["guaranteed_core"] == core
        inputs.update({f"mac/evaluation/phase4/{n}": sha(base / n)
                       for n in ("student.pt", "vocab.json", "canonical.json", "splits.json", "config.yaml")})
        if args.head == "torch":
            loader = "native rank_vocabulary._load_student (VOICELINK_STUDENT_BACKEND=torch)"
            model, vocab_l, core_l, k_dyn = rv._load_student()
            assert isinstance(model, StudentRanker) and vocab_l == vocab and core_l == core and k_dyn == K_DYN
            head_path = base / "student.pt"
        elif args.head == "ov-int8":
            loader = "native rank_vocabulary._load_student (VOICELINK_STUDENT_BACKEND=openvino -> student_ir_path())"
            model, vocab_l, core_l, k_dyn = rv._load_student()
            assert isinstance(model, OpenVinoStudent) and vocab_l == vocab and core_l == core and k_dyn == K_DYN
            head_path = model.xml_path
            assert head_path == intel.STUDENT_INT8_XML
        else:  # ov-fp32 control: student_ir_path() prefers INT8, so inject the FP32 IR explicitly
            head_path = intel.STUDENT_FP32_XML
            model = OpenVinoStudent(head_path, device="CPU")
        if args.head != "torch":
            side = head_path.with_suffix(".vocab.sha").read_text().strip()
            assert side == d.vocab_sha(vocab) and model.vocab_size == len(vocab), "IR/vocab mismatch"
            inputs[f"mac/intel/artifacts/{head_path.name}"] = sha(head_path)
            inputs[f"mac/intel/artifacts/{head_path.with_suffix('.bin').name}"] = sha(head_path.with_suffix(".bin"))
    else:  # i05 family: train-only checkpoint + I-07 train-only-calibrated IRs
        man05 = json.loads((I05_BUNDLE / "manifest.json").read_text())
        for n in ("student.pt", "vocab.json", "canonical.json", "splits.json"):
            assert sha(I05_BUNDLE / n) == man05["artifact_sha256"][n], f"I-05 bundle drifted: {n}"
            inputs[f"analyses/I-05_split_first_v2/{n}"] = man05["artifact_sha256"][n]
        vocab = json.loads((I05_BUNDLE / "vocab.json").read_text())
        ck = torch.load(I05_BUNDLE / "student.pt", map_location="cpu", weights_only=False)
        assert ck["config_hash"] == "b691659cecf3" and len(vocab) == 45 and ck["vocab_sha"] == d.vocab_sha(vocab)
        assert ck["guaranteed_core"] == core and ck["k_dyn"] == K_DYN
        if args.head == "torch":
            model = StudentRanker(ck["context_dim"], ck["hidden_dim"], ck["vocab_size"], ck["dropout"])
            model.load_state_dict(ck["state_dict"]); model.eval()
            head_path = I05_BUNDLE / "student.pt"
        else:
            man07 = json.loads((I07_EXPORT / "manifest.json").read_text())
            stem = "student_int8" if args.head == "ov-int8" else "student_fp32"
            head_path = I07_EXPORT / f"{stem}.xml"
            for n in (f"{stem}.xml", f"{stem}.bin"):
                assert sha(I07_EXPORT / n) == man07["artifact_sha256"][n], f"I-07 export drifted: {n}"
                inputs[f"analyses/I-07_export_v1/{n}"] = man07["artifact_sha256"][n]
            model = OpenVinoStudent(head_path, device="CPU")
            side = head_path.with_suffix(".vocab.sha").read_text().strip()
            assert side == d.vocab_sha(vocab) and model.vocab_size == len(vocab), "IR/vocab mismatch"
    if args.embed == "openvino":
        for n in ("openvino_model.xml", "openvino_model.bin", "openvino_tokenizer.xml", "openvino_tokenizer.bin"):
            inputs[f"mac/intel/artifacts/minilm-ov/{n}"] = sha(intel.MINILM_OV_DIR / n)
    for n in ("rank_vocabulary.py", "selection.py", "evaluation/embed.py", "evaluation/phase4/infer.py",
              "evaluation/phase4/data.py", "evaluation/phase4/answers.py", "evaluation/phase4/evaluate.py",
              "evaluation/phase3/taxonomy.py", "evaluation/phase3/metrics3.py", "intel/__init__.py"):
        inputs[f"mac/{n}"] = sha(MAC / n)

    # The seam serves whatever _load_student() returns; for the injected configurations put the
    # verified (model, vocab, core, k_dyn) tuple there so rank_vocabulary(strategy="student") uses it.
    rv._student_singleton = (model, vocab, core, K_DYN)
    assert rv._load_student()[0] is model

    held = e._held_out_rows()
    assert len(held) == 84
    lookup = e._acceptable_moves_lookup()
    rows_out = []
    for i, r in enumerate(held):
        ctx = {"last_utterance": r.get("transcript", ""), "personal_context": {}}
        seam = [w["word"] for w in rv.rank_vocabulary(r["scene"], r["transcript"], ctx, strategy="student", k=K_BOARD)]
        if i == 0:
            # Hard label check right after the first seam call: the embedder that ACTUALLY served.
            assert em.assert_backend(args.embed) == args.embed
        assert not rv._student_failure_reported, "seam fell back to the LLM path"
        assert len(seam) == K_BOARD and seam[:4] == core, seam
        direct = infer.student_board(model, vocab, r["scene"], r["transcript"], r.get("is_question", False),
                                     K_BOARD, K_DYN, core)
        assert len(direct) == K_BOARD and direct[:4] == core
        feats = d.row_features(r["scene"], r["transcript"], r.get("is_question", False))[None, :]
        if isinstance(model, StudentRanker):
            with torch.no_grad():
                logits = model(torch.from_numpy(feats))[0].numpy()
        else:
            logits = np.asarray(model(feats), dtype=np.float32)[0]
        rows_out.append(dict(
            index=i, scene=r["scene"], transcript=r["transcript"], category=r.get("category"),
            is_question_row=bool(r.get("is_question", False)),
            is_question_seam=bool(rv._is_question(r["transcript"], ctx)),
            teacher_board=r["board"],
            moves=lookup.get((r["scene"].strip(), r["transcript"].strip()), []),
            seam_board=seam, direct_board=direct,
            logits=[float(x) for x in logits]))
    assert em.assert_backend(args.embed) == args.embed
    assert not rv._student_failure_reported
    head_runtime = type(model).__name__
    if args.head == "torch":
        assert head_runtime == "StudentRanker"
    else:
        assert head_runtime == "OpenVinoStudent" and model.device == "CPU"
    out = dict(
        issue="I-04", phase="generate", family=args.family, head=args.head, embed=args.embed,
        env={k: os.environ[k] for k in ("VOICELINK_STUDENT_BACKEND", "VOICELINK_EMBED_BACKEND", "VOICELINK_EMBED_STRICT")},
        asserted=dict(embed_backend_used=em.backend_in_use(), head_runtime=head_runtime,
                      head_artifact=str(head_path.relative_to(REPO)), ov_device=getattr(model, "device", None),
                      loader=loader, seam_fallback_reported=bool(rv._student_failure_reported)),
        vocab_size=len(vocab), vocab_sha=d.vocab_sha(vocab), core=core, k=K_BOARD, k_dyn=K_DYN,
        n_rows=len(rows_out),
        seam_vs_direct_identical=sum(a["seam_board"] == a["direct_board"] for a in rows_out),
        is_question_seam_vs_row_agree=sum(a["is_question_seam"] == a["is_question_row"] for a in rows_out),
        input_sha256=inputs, hardware=hardware_fingerprint(),
        versions={x: importlib.metadata.version(x) for x in ("torch", "numpy", "openvino", "sentence-transformers", "transformers")},
        python=platform.python_version(), wall_seconds=round(time.perf_counter() - t_start, 2),
        rows=rows_out)
    write_json(args.out, out)
    print(f"[generate] {args.family}/{args.head}/{args.embed}: embed={out['asserted']['embed_backend_used']} "
          f"head={head_runtime} seam==direct {out['seam_vs_direct_identical']}/84 "
          f"is_q agree {out['is_question_seam_vs_row_agree']}/84 wall {out['wall_seconds']}s -> {args.out.name}")


# ----------------------------------------------------------------------------------------------
# phase 2: score (fixed torch ruler)
# ----------------------------------------------------------------------------------------------

def _score_file(path: Path, e, metrics3, np, bootstrap_ci) -> dict:
    g = json.loads(path.read_text())
    core = g["core"]
    strip = lambda b: [w for w in b if w.lower() not in {x.lower() for x in core}]  # noqa: E731
    held = e._held_out_rows()
    assert len(held) == len(g["rows"]) == 84
    for r, b in zip(held, g["rows"]):
        assert r["scene"] == b["scene"] and r["transcript"] == b["transcript"] and r["board"] == b["teacher_board"]
    res: dict = {k: g[k] for k in ("family", "head", "embed", "asserted", "vocab_size", "vocab_sha", "input_sha256",
                                   "seam_vs_direct_identical", "is_question_seam_vs_row_agree", "wall_seconds")}
    res["source_file"] = path.name
    res["windows"] = {}
    for source in ("seam", "direct"):
        res["windows"][source] = {}
        for name, k, dyn in WINDOWS:
            it = iter(g["rows"])

            def board_fn(scene, transcript, is_q, _it=it, _src=source):
                b = next(_it)
                assert b["scene"] == scene and b["transcript"] == transcript
                return b[f"{_src}_board"]

            h = e.held_out_eval(None, None, K_BOARD, K_DYN, core, board_fn=board_fn, metric_k=k, dynamic_only=dyn)
            # per-row dynamic coverage (the gated number) for cross-backend pairing
            per_row = []
            for b in g["rows"]:
                sb = b[f"{source}_board"]; tb = b["teacher_board"]
                if dyn:
                    sb, tb = strip(sb)[:k], strip(tb)[:k]
                else:
                    sb, tb = sb[:k], tb[:k]
                honest = [m for m in b["moves"] if not metrics3.is_hedge(m)]
                if b["moves"] and honest:
                    s_dc = int(metrics3.coverage_of_intent(e.dynamic_words(sb, core, k=k), honest, k))
                    t_dc = int(metrics3.coverage_of_intent(e.dynamic_words(tb, core, k=k), honest, k))
                else:
                    s_dc = t_dc = None
                per_row.append(dict(index=b["index"], student_dynamic_coverage=s_dc, teacher_dynamic_coverage=t_dc))
            s_list = [p["student_dynamic_coverage"] for p in per_row if p["student_dynamic_coverage"] is not None]
            t_list = [p["teacher_dynamic_coverage"] for p in per_row if p["teacher_dynamic_coverage"] is not None]
            assert len(s_list) == h["coverage_of_intent"]["n_scored_dynamic"]
            assert round(float(np.mean(s_list)), 3) == h["coverage_of_intent"]["student_dynamic"]
            diffs = [s - t for s, t in zip(s_list, t_list)]
            lo, hi = bootstrap_ci(diffs)
            dr = h["diversity_retention"]; co = h["coverage_of_intent"]
            res["windows"][source][name] = dict(
                harness=h,
                summary=dict(
                    df=[dr["student_distinct_functions"], dr["teacher_distinct_functions"]],
                    df_retention=dr["retention_ratio"], df_margin=dr["distinct_functions_margin"],
                    near_dup=[dr["student_near_dup"], dr["teacher_near_dup"]],
                    sfp=[dr["student_same_function_pair"], dr["teacher_same_function_pair"]],
                    sfp_margin=dr["same_function_pair_margin"], board_variety=dr["board_variety"],
                    coverage_all=[co["student"], co["teacher"]],
                    dynamic_coverage=[co["student_dynamic"], co["teacher_dynamic"]],
                    dynamic_coverage_counts=[int(sum(s_list)), int(sum(t_list)), len(s_list)],
                    dynamic_coverage_retention=(round(float(sum(s_list)) / float(sum(t_list)), 4) if sum(t_list) else None),
                    paired_counts=co["paired_counts_student_teacher"], coverage_margin=co["margin"],
                    coverage_point_pass=co["pass"], coverage_verdict=co["verdict"],
                    raw_dynamic_coverage_difference=dict(mean=round(float(np.mean(diffs)), 4), ci=[round(lo, 4), round(hi, 4)]),
                    dynamic_agreement=h["agreement"]["dynamic_overlap"]),
                per_row_dynamic_coverage=per_row)
    return res


def cmd_score(args) -> None:
    os.environ.update(VOICELINK_PERCEPTION="0", HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                      VOICELINK_DEVICE="cpu", VOICELINK_EMBED_BACKEND="torch",
                      VOICELINK_STUDENT_BACKEND="torch", VOICELINK_EMBED_STRICT="1")
    t_start = time.perf_counter()
    import numpy as np
    import torch
    torch.manual_seed(0); np.random.seed(0)
    from evaluation import embed as em
    from evaluation.coverage_metrics import bootstrap_ci
    from evaluation.phase3 import metrics3
    from evaluation.phase4 import evaluate as e
    assert em.assert_backend("torch") == "torch"

    files = sorted(args.boards)
    scored = {}
    for p in files:
        s = _score_file(p, e, metrics3, np, bootstrap_ci)
        scored[f"{s['family']}/{s['head']}/{s['embed']}"] = s
        print(f"[score] {p.name}: full10 dyn-cov {s['windows']['direct']['full10']['summary']['dynamic_coverage_counts']} "
              f"seam {s['windows']['seam']['full10']['summary']['dynamic_coverage_counts']}")
    assert em.assert_backend("torch") == "torch"

    # --- cross-backend comparisons inside each family (reference = torch head + torch embedder) ---
    gens = {k: json.loads(next(p for p in files if p.name == v["source_file"]).read_text()) for k, v in scored.items()}
    cross = {}
    for key, s in scored.items():
        fam = s["family"]
        ref_key = f"{fam}/torch/torch"
        if ref_key not in scored:
            continue
        ref = scored[ref_key]; gref = gens[ref_key]; g = gens[key]
        comp = {}
        for source in ("seam", "direct"):
            js = [jaccard(a[f"{source}_board"], b[f"{source}_board"]) for a, b in zip(g["rows"], gref["rows"])]
            dyn_js = [jaccard(a[f"{source}_board"][4:], b[f"{source}_board"][4:]) for a, b in zip(g["rows"], gref["rows"])]
            comp[source] = dict(board_jaccard_vs_reference=dict(mean=round(float(np.mean(js)), 4), min=round(float(min(js)), 4),
                                                              identical=int(sum(j == 1.0 for j in js)), n=len(js)),
                                dynamic_slot_jaccard_mean=round(float(np.mean(dyn_js)), 4))
            for name, _, _ in WINDOWS:
                a = s["windows"][source][name]["per_row_dynamic_coverage"]
                b = ref["windows"][source][name]["per_row_dynamic_coverage"]
                pairs = [(x["student_dynamic_coverage"], y["student_dynamic_coverage"]) for x, y in zip(a, b)
                         if x["student_dynamic_coverage"] is not None]
                diffs = [x - y for x, y in pairs]
                lo, hi = bootstrap_ci(diffs)
                comp[source][name] = dict(
                    config_minus_reference_dynamic_coverage=dict(mean=round(float(np.mean(diffs)), 4), ci=[round(lo, 4), round(hi, 4)]),
                    flips=dict(reference_covered_config_not=int(sum(1 for x, y in pairs if y == 1 and x == 0)),
                               config_covered_reference_not=int(sum(1 for x, y in pairs if x == 1 and y == 0)),
                               both=int(sum(1 for x, y in pairs if x == 1 and y == 1)),
                               neither=int(sum(1 for x, y in pairs if x == 0 and y == 0))))
        # head-level agreement from the saved logits (top-6 overlap, max |delta|) vs the reference head
        L = np.array([r["logits"] for r in g["rows"]]); R = np.array([r["logits"] for r in gref["rows"]])
        ov6 = [len(set(np.argsort(-x)[:6]) & set(np.argsort(-y)[:6])) / 6 for x, y in zip(L, R)]
        comp["head_vs_reference_logits"] = dict(top6_overlap_mean=round(float(np.mean(ov6)), 4), top6_overlap_min=round(float(min(ov6)), 4),
                                                max_abs_logit_delta=float(np.abs(L - R).max()), mean_abs_logit_delta=float(np.abs(L - R).mean()),
                                                note="features come from each configuration's own embedder; for embed=torch this isolates the head")
        cross[key] = comp

    # --- continuity checks against the dependency artifacts ---
    checks = {}
    if "hist/torch/torch" in gens:
        i03 = json.loads(I03_NUMBERS.read_text())
        stored = {r["index"]: r["student_board"] for r in i03["per_row_table2"] if r["window"] == "full10"}
        g = gens["hist/torch/torch"]
        checks["hist_torch_direct_boards_equal_I03"] = int(sum(stored[r["index"]] == r["direct_board"] for r in g["rows"]))
        s = scored["hist/torch/torch"]["windows"]["direct"]
        checks["hist_torch_direct_counts_equal_I03"] = dict(
            full10=s["full10"]["summary"]["paired_counts"] == {"11": 58, "01": 10, "10": 3, "00": 0},
            full7=s["full7"]["summary"]["paired_counts"] == {"11": 49, "01": 13, "10": 7, "00": 2},
            df_full7=s["full7"]["summary"]["df"] == [4.095, 4.298])
    if "i05/torch/torch" in gens:
        b05 = {b["row_index"]: b["student_board"] for b in json.loads((I05_BUNDLE / "boards.json").read_text())}
        splits = json.loads((I05_BUNDLE / "splits.json").read_text())
        g = gens["i05/torch/torch"]
        checks["i05_torch_direct_boards_equal_I05"] = int(sum(b05[splits["test"][r["index"]]] == r["direct_board"] for r in g["rows"]))
        checks["i05_torch_direct_counts_equal_I05"] = scored["i05/torch/torch"]["windows"]["direct"]["full10"]["summary"]["paired_counts"] == {"11": 51, "01": 17, "10": 3, "00": 0}
    b07p = I07_EXPORT / "boards_test_by_backend.json"
    if b07p.exists():
        b07 = json.loads(b07p.read_text())
        rows07 = b07["rows"] if isinstance(b07, dict) and "rows" in b07 else b07
        splits = json.loads((I05_BUNDLE / "splits.json").read_text())
        for head, k07 in (("ov-fp32", "ov_fp32"), ("ov-int8", "ov_int8")):
            key = f"i05/{head}/torch"
            if key in gens:
                by_row = {}
                for rr in rows07:
                    idx = rr.get("row_index", rr.get("row"))
                    by_row[idx] = rr
                n_eq = 0; n_seen = 0
                for r in gens[key]["rows"]:
                    rr = by_row.get(splits["test"][r["index"]])
                    if rr is None:
                        continue
                    cand = rr.get(k07) or rr.get(f"{k07}_board") or rr.get("boards", {}).get(k07)
                    if cand is None:
                        continue
                    n_seen += 1; n_eq += int(cand == r["direct_board"])
                checks[f"i05_{head}_torch_direct_boards_equal_I07"] = [n_eq, n_seen]

    # --- latency join: archived e2e_mac.json rows (historical 95-entry artefacts, 2026-08-23, Apple M4) ---
    e2e = json.loads(E2E_MAC.read_text())
    lat_rows = {r["backend"]: r for r in e2e["results"]}
    join_map = {"hist/torch/torch": "torch", "hist/ov-int8/torch": "ov-head", "hist/ov-int8/openvino": "ov-full"}
    latency = {}
    for key in scored:
        if key in join_map:
            r = lat_rows[join_map[key]]
            latency[key] = dict(source="mac/intel/artifacts/e2e_mac.json (archived 2026-08-23; I-10 owns any rerun)",
                                bench_row=join_map[key], p50_ms=round(r["p50_ms"], 2), p90_ms=round(r["p90_ms"], 2),
                                mean_ms=round(r["mean_ms"], 2), n=r["n"], repeats=r["repeats"],
                                embed_backend_used=r["embed_backend_used"], student_runtime=r["student_runtime"],
                                board_jaccard_vs_torch_archived=round(r["board_jaccard_vs_torch"], 4),
                                hardware=e2e["hardware"]["cpu"], sha256=sha(E2E_MAC))
        else:
            latency[key] = dict(source=None, note="no archived latency row for this configuration (I-10 owns the rerun)")

    out = dict(issue="I-04", phase="score", scoring_embedder=em.backend_in_use(), seed=0,
               bootstrap=dict(resamples=10000, seed=0, method="paired percentile, row resampling, 95%"),
               versions={x: importlib.metadata.version(x) for x in ("torch", "numpy", "openvino", "sentence-transformers", "transformers")},
               configurations=scored, cross_backend=cross, continuity_checks=checks, latency_join=latency,
               wall_seconds=round(time.perf_counter() - t_start, 2))
    write_json(args.out, out)
    print(f"[score] wrote {args.out} ({out['wall_seconds']}s); checks={json.dumps(checks)}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--family", choices=["hist", "i05"], required=True)
    g.add_argument("--head", choices=["torch", "ov-fp32", "ov-int8"], required=True)
    g.add_argument("--embed", choices=["torch", "openvino"], required=True)
    g.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--boards", type=Path, nargs="+", required=True)
    s.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        p.error(f"{args.out} exists; use a distinct suffix")
    (cmd_generate if args.cmd == "generate" else cmd_score)(args)


if __name__ == "__main__":
    main()
