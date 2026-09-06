"""I-10 — end-to-end board latency with repeats, raw samples, counterbalanced sessions and a
consistent cache/state protocol, on the historical head and on the two train-only-calibrated
heads (I-07: 45-entry I-05 checkpoint; I-07b: 95-entry I-05b checkpoint), with the validated
torch-free MiniLM bundle (I-12).

Three sub-commands:

  sessions   one PROCESS running one or more backend sessions in the given order for one
             artifact family. Every session starts from the same state (bench_e2e._reset_seam
             full=True), pre-warms the static set (vocabulary, core, taxonomy exemplars) and, with
             --evict board, evicts after every timed board every embedding key that board added, so
             each context is encoded fresh on every call. The head that is timed is the family's
             own artifact (native seam loader for `hist`; injected verified (model, vocab, core,
             k_dyn) tuple for `i05`/`i05b`, exactly as I-04 did). Raw per-round, per-scenario
             samples, boards, backend/device/hash/fallback status and a machine-load snapshot before
             and after each session are written to ONE exclusive JSON file.
  run        the orchestrator: for each family, three counterbalanced in-process orders (a Latin
             square: every backend once in every position), three isolated single-backend
             processes, and one historical-protocol control (torch,ov-head,ov-full order, per-round
             eviction, no full reset, 3 repeats — the e2e_mac.json recipe, but with samples kept).
  summarise  pooled and per-session percentiles, scenario-level bootstrap CIs, between-session
             variation, Jaccard against the family's torch boards (reference fixed to torch whatever
             the order), board determinism across sessions, the slowest scenarios, and the
             historical e2e_mac.json rows for comparison.

Nothing under mac/ is written; every output is opened exclusively (never overwritten).
Run from mac/ with /usr/local/bin/python3 (see I-10_result.md for the exact command list).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent            # paper/odi/revision/analyses
REPO = HERE.parents[3]
MAC = REPO / "mac"
if str(MAC) not in sys.path:
    sys.path.insert(0, str(MAC))

PY = sys.executable
K_BOARD, K_DYN = 10, 6
BACKEND_NAMES = ("torch", "ov-head", "ov-full")
# Latin square over the three backends: each backend appears once in every position.
ORDERS = (("torch", "ov-head", "ov-full"),
          ("ov-full", "torch", "ov-head"),
          ("ov-head", "ov-full", "torch"))
FAMILIES = {
    # family: (checkpoint dir, int8 IR dir, expected vocab size, expected config hash)
    "hist": (MAC / "evaluation" / "phase4", MAC / "intel" / "artifacts", 95, "d6ce2e56d50b"),
    "i05":  (HERE / "I-05_split_first_v2", HERE / "I-07_export_v1", 45, "b691659cecf3"),
    "i05b": (HERE / "I-05b_split_first", HERE / "I-07b_artifacts", 95, "b72bb3ec7254"),
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_json(p: Path, x) -> None:
    with p.open("x") as f:                      # exclusive: never overwrite
        json.dump(x, f, indent=1)


def system_snapshot() -> dict:
    """Machine-load evidence: load averages and every process above 3 % CPU other than
    this one (the 'machine otherwise idle' condition is recorded, not assumed)."""
    me = os.getpid()
    busy = []
    try:
        out = subprocess.run(["ps", "-eo", "pid,ppid,pcpu,pmem,etime,command"], capture_output=True,
                             text=True, timeout=10).stdout.splitlines()[1:]
        for line in out:
            parts = line.split(None, 5)
            if len(parts) < 6:
                continue
            pid, ppid, pcpu = int(parts[0]), int(parts[1]), float(parts[2])
            if pid in (me,) or ppid == me:
                continue
            if pcpu >= 3.0:
                busy.append({"pid": pid, "pcpu": pcpu, "pmem": float(parts[3]), "etime": parts[4],
                             "command": parts[5][:160]})
    except Exception as e:  # pragma: no cover — evidence, not a gate
        busy.append({"error": repr(e)})
    return {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "loadavg_1_5_15": list(os.getloadavg()),
            "busy_processes_ge3pct": busy}


# ----------------------------------------------------------------------------------------------
# family loaders (same verification discipline as I-04_deployed_path_implement.py)
# ----------------------------------------------------------------------------------------------

def load_family(family: str, backend: str) -> tuple[tuple | None, dict]:
    """Return (student tuple to inject or None for the native loader, provenance dict)."""
    import torch
    from evaluation.phase4 import data as d
    from evaluation.phase4.student import StudentRanker
    import intel
    from intel import OpenVinoStudent
    import rank_vocabulary as rv

    ck_dir, ir_dir, n_vocab, cfg_hash = FAMILIES[family]
    core = list(rv.GUARANTEED_CORE)
    head_is_ov = backend != "torch"
    prov: dict = {"family": family, "inputs_sha256": {}}
    if family == "hist":
        vocab, _ = d.load_vocab()
        ck = torch.load(ck_dir / "student.pt", map_location="cpu", weights_only=False)
        assert ck["config_hash"] == cfg_hash and len(vocab) == n_vocab and ck["guaranteed_core"] == core
        for n in ("student.pt", "vocab.json", "canonical.json", "splits.json", "config.yaml"):
            prov["inputs_sha256"][f"mac/evaluation/phase4/{n}"] = sha(ck_dir / n)
        if head_is_ov:
            for n in ("student_int8.xml", "student_int8.bin", "student_int8.vocab.sha"):
                prov["inputs_sha256"][f"mac/intel/artifacts/{n}"] = sha(ir_dir / n)
            prov["head_artifact_expected"] = str(intel.STUDENT_INT8_XML)
        else:
            prov["head_artifact_expected"] = str(ck_dir / "student.pt")
        prov["loader"] = "native rank_vocabulary._load_student"
        prov["vocab_size"], prov["vocab_sha"], prov["config_hash"] = len(vocab), d.vocab_sha(vocab), cfg_hash
        return None, prov

    man = json.loads((ck_dir / "manifest.json").read_text())
    assert man["config_hash"] == cfg_hash, (family, man["config_hash"])
    for n in ("student.pt", "vocab.json", "canonical.json", "splits.json"):
        assert sha(ck_dir / n) == man["artifact_sha256"][n], f"{family} bundle drifted: {n}"
        prov["inputs_sha256"][f"analyses/{ck_dir.name}/{n}"] = man["artifact_sha256"][n]
    vocab = json.loads((ck_dir / "vocab.json").read_text())
    ck = torch.load(ck_dir / "student.pt", map_location="cpu", weights_only=False)
    assert ck["config_hash"] == cfg_hash and len(vocab) == n_vocab and ck["vocab_sha"] == d.vocab_sha(vocab)
    assert ck["guaranteed_core"] == core and ck["k_dyn"] == K_DYN
    if not head_is_ov:
        model = StudentRanker(ck["context_dim"], ck["hidden_dim"], ck["vocab_size"], ck["dropout"])
        model.load_state_dict(ck["state_dict"]); model.eval()
        head_path = ck_dir / "student.pt"
    else:
        man_ir = json.loads((ir_dir / "manifest.json").read_text())
        head_path = ir_dir / "student_int8.xml"
        for n in ("student_int8.xml", "student_int8.bin"):
            assert sha(ir_dir / n) == man_ir["artifact_sha256"][n], f"{family} IR drifted: {n}"
            prov["inputs_sha256"][f"analyses/{ir_dir.name}/{n}"] = man_ir["artifact_sha256"][n]
        model = OpenVinoStudent(head_path, device="CPU")
        side = head_path.with_suffix(".vocab.sha").read_text().strip()
        assert side == d.vocab_sha(vocab) and model.vocab_size == len(vocab), "IR/vocab mismatch"
    prov["head_artifact_expected"] = str(head_path)
    prov["loader"] = "injected rank_vocabulary._student_singleton (verified tuple)"
    prov["vocab_size"], prov["vocab_sha"], prov["config_hash"] = len(vocab), d.vocab_sha(vocab), cfg_hash
    return (model, vocab, core, K_DYN), prov


# ----------------------------------------------------------------------------------------------
# sessions
# ----------------------------------------------------------------------------------------------

def cmd_sessions(args) -> None:
    os.environ.update(VOICELINK_PERCEPTION="0", HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                      VOICELINK_DEVICE="cpu", VOICELINK_EMBED_STRICT="1")
    out_path = Path(args.out)
    if out_path.exists():
        raise SystemExit(f"{out_path} exists — refusing to overwrite")
    t_proc = time.perf_counter()
    import numpy as np
    import torch
    from intel import bench_e2e, hardware_fingerprint, minilm_bundle_status, MINILM_OV_DIR
    from evaluation import embed as em
    from evaluation.phase4 import data as d
    import rank_vocabulary as rv

    hw = hardware_fingerprint()
    scenarios, source = bench_e2e.load_scenarios(args.n)
    assert source.startswith("phase4 held-out split"), source
    scenario_ids = [{"index": i, "ctx_sha8": hashlib.sha256((s["scene"] + "\x1f" + s["transcript"]).encode()).hexdigest()[:8],
                     "category": s.get("category"), "is_question": bool(s.get("is_question", False))}
                    for i, s in enumerate(scenarios)]
    st = minilm_bundle_status()
    embed_bundle = {n: sha(MINILM_OV_DIR / n) for n in ("openvino_model.xml", "openvino_model.bin",
                                                         "openvino_tokenizer.xml", "openvino_tokenizer.bin")} \
        if st["complete"] else {"incomplete": st}
    code_sha = {f"mac/{n}": sha(MAC / n) for n in
                ("rank_vocabulary.py", "selection.py", "evaluation/embed.py", "evaluation/phase4/infer.py",
                 "evaluation/phase4/data.py", "evaluation/phase4/answers.py", "evaluation/phase3/taxonomy.py",
                 "intel/__init__.py", "intel/bench_e2e.py")}
    code_sha["analyses/I-10_latency_implement.py"] = sha(Path(__file__))

    backends = args.backends.split(",")
    sessions = []
    for pos, backend in enumerate(backends):
        before = system_snapshot()
        student, prov = load_family(args.family, backend)
        t0 = time.perf_counter()
        r = bench_e2e.bench_backend(backend, scenarios, warmup=args.warmup, repeats=args.repeats,
                                    student=student, full_reset=not args.no_full_reset, evict=args.evict)
        wall = time.perf_counter() - t0
        after = system_snapshot()
        # Label checks: the runtime that ACTUALLY served, not the one requested.
        assert r["embed_backend_used"] == bench_e2e.BACKENDS[backend]["VOICELINK_EMBED_BACKEND"], r["embed_backend_used"]
        assert r["student_runtime"] == ("StudentRanker" if backend == "torch" else "OpenVinoStudent"), r["student_runtime"]
        assert r["vocab_size"] == prov["vocab_size"]
        if backend != "torch":
            assert r["head_artifact"] == prov["head_artifact_expected"], (r["head_artifact"], prov["head_artifact_expected"])
            assert r["ov_device"] == "CPU"
        assert not rv._student_failure_reported
        for b in r["boards"]:
            assert len(b) == K_BOARD and b[:4] == list(rv.GUARANTEED_CORE), b
        sessions.append({
            "session_id": f"{args.family}/{args.tag}/pos{pos}/{backend}",
            "family": args.family, "tag": args.tag, "position": pos, "order": backends,
            "backend": backend, "provenance": prov,
            "label_checks": {"embed_backend_used": r["embed_backend_used"], "student_runtime": r["student_runtime"],
                             "head_artifact": r["head_artifact"], "ov_device": r["ov_device"],
                             "seam_fallback_reported": bool(rv._student_failure_reported),
                             "embed_strict": os.environ.get("VOICELINK_EMBED_STRICT")},
            "protocol": {"n": r["n"], "repeats": r["repeats"], "warmup": r["warmup"], "evict": r["evict"],
                         "full_reset": r["full_reset"],
                         "eviction_rule": ("static set (vocabulary + core + taxonomy exemplars) pre-warmed; after every "
                                           "timed board every (backend, text) key that board added is evicted"
                                           if r["evict"] == "board" else
                                           "raw scene/transcript strings of all n scenarios forgotten once before each "
                                           "round; everything else stays memoised (historical bench_e2e rule)")},
            "first_call_incl_load_s": r["first_call_incl_load_s"], "session_wall_s": round(wall, 3),
            "p25_ms": r["p25_ms"], "p50_ms": r["p50_ms"], "p75_ms": r["p75_ms"], "p90_ms": r["p90_ms"],
            "mean_ms": r["mean_ms"], "min_ms": r["min_ms"], "max_ms": r["max_ms"],
            "samples_ms": r["samples_ms"], "boards": r["boards"],
            "load_before": before, "load_after": after,
        })
        print(f"[sessions] {args.family} {args.tag} pos{pos} {backend:8s} p50 {r['p50_ms']:.2f} p90 {r['p90_ms']:.2f} "
              f"(p25 {r['p25_ms']:.2f} p75 {r['p75_ms']:.2f}) embed={r['embed_backend_used']} head={r['student_runtime']} "
              f"wall {wall:.1f}s load {before['loadavg_1_5_15'][0]:.2f}->{after['loadavg_1_5_15'][0]:.2f}", flush=True)

    out = {
        "issue": "I-10", "phase": "sessions", "family": args.family, "tag": args.tag, "order": backends,
        "n_scenarios": len(scenarios), "scenario_source": source, "scenario_ids": scenario_ids,
        "hardware": hw, "python": platform.python_version(), "torch_threads": torch.get_num_threads(),
        "versions": {m: _ver(m) for m in ("torch", "openvino", "openvino-tokenizers", "sentence-transformers", "numpy")},
        "minilm_bundle_sha256": embed_bundle, "code_sha256": code_sha,
        "process_wall_s": round(time.perf_counter() - t_proc, 3),
        "sessions": sessions,
    }
    write_json(out_path, out)
    print(f"[sessions] wrote {out_path}")


def _ver(name: str) -> str | None:
    import importlib.metadata as md
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


# ----------------------------------------------------------------------------------------------
# run (orchestrator)
# ----------------------------------------------------------------------------------------------

def cmd_run(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    log = out_dir / "run.log"
    plan = []
    for fam in args.families.split(","):
        for oi, order in enumerate(ORDERS):
            plan.append((fam, f"order{oi}", ",".join(order), args.repeats, "board", False))
        for b in BACKEND_NAMES:
            plan.append((fam, "isolated", b, args.repeats, "board", False))
        # historical-protocol control: the e2e_mac.json recipe with samples kept
        plan.append((fam, "histproto", ",".join(ORDERS[0]), 3, "round", True))
    manifest = {"issue": "I-10", "phase": "run", "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                "n": args.n, "repeats": args.repeats, "plan": [], "load_at_start": system_snapshot()}
    t_all = time.perf_counter()
    for fam, tag, backends, repeats, evict, no_full in plan:
        name = f"{fam}_{tag}_{backends.replace(',', '-')}.json"
        cmd = [PY, str(Path(__file__)), "sessions", "--family", fam, "--tag", tag, "--backends", backends,
               "--n", str(args.n), "--repeats", str(repeats), "--warmup", str(args.warmup), "--evict", evict,
               "--out", str(out_dir / name)]
        if no_full:
            cmd.append("--no-full-reset")
        t0 = time.perf_counter()
        with log.open("a") as lf:
            lf.write(f"\n=== {' '.join(cmd)}\n"); lf.flush()
            rc = subprocess.run(cmd, cwd=str(MAC), stdout=lf, stderr=subprocess.STDOUT,
                                env={**os.environ, "VOICELINK_PERCEPTION": "0"}).returncode
        wall = time.perf_counter() - t0
        manifest["plan"].append({"family": fam, "tag": tag, "backends": backends, "repeats": repeats, "evict": evict,
                                 "full_reset": not no_full, "file": name, "rc": rc, "wall_s": round(wall, 1)})
        print(f"[run] {fam:5s} {tag:9s} {backends:22s} rc={rc} {wall:.1f}s", flush=True)
        if rc != 0:
            print(f"[run] FAILED — see {log}", flush=True)
    manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["total_wall_s"] = round(time.perf_counter() - t_all, 1)
    manifest["load_at_end"] = system_snapshot()
    write_json(out_dir / "run_manifest.json", manifest)
    print(f"[run] wrote {out_dir / 'run_manifest.json'} total {manifest['total_wall_s']} s")


# ----------------------------------------------------------------------------------------------
# summarise
# ----------------------------------------------------------------------------------------------

def _pct(v, q):
    import numpy as np
    return float(np.percentile(np.asarray(v, dtype=float), q))


def _stats(v) -> dict:
    import numpy as np
    a = np.asarray(v, dtype=float)
    return {"n_samples": int(a.size), "p25": _pct(a, 25), "p50": _pct(a, 50), "p75": _pct(a, 75),
            "p90": _pct(a, 90), "p95": _pct(a, 95), "mean": float(a.mean()), "min": float(a.min()),
            "max": float(a.max())}


def _bootstrap_scenario(samples_by_scenario: list[list[float]], q: float, reps: int = 2000, seed: int = 0) -> list[float]:
    """Percentile-bootstrap CI for a pooled percentile, resampling SCENARIOS (the unit of
    replication: repeats of one scenario are not independent draws)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    n = len(samples_by_scenario)
    arrs = [np.asarray(s, dtype=float) for s in samples_by_scenario]
    vals = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        vals.append(np.percentile(np.concatenate([arrs[i] for i in idx]), q))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = {w.lower() for w in a}, {w.lower() for w in b}
    return len(sa & sb) / max(len(sa | sb), 1)


def cmd_summarise(args) -> None:
    import numpy as np
    run_dirs = [Path(d).resolve() for d in args.run_dir]
    files = sorted(p for rd in run_dirs for p in rd.glob("*.json") if p.name != "run_manifest.json")
    sessions, meta = [], None
    for p in files:
        x = json.loads(p.read_text())
        if meta is None:
            meta = {k: x[k] for k in ("n_scenarios", "scenario_source", "scenario_ids", "hardware", "python",
                                       "torch_threads", "versions", "minilm_bundle_sha256", "code_sha256")}
        else:
            assert x["scenario_ids"] == meta["scenario_ids"] and x["minilm_bundle_sha256"] == meta["minilm_bundle_sha256"]
        for s in x["sessions"]:
            s["_file"] = f"{p.parent.name}/{p.name}"
            s["_run"] = p.parent.name
            # Contamination flag (objective, from the before/after snapshots only): any OTHER process at
            # >= 40 % CPU. The snapshots bracket the session; they cannot see a burst that started and
            # ended inside it, so 'clean' means 'no evidence of load', not 'proven idle'.
            hot = [b for snap in (s["load_before"], s["load_after"]) for b in snap["busy_processes_ge3pct"]
                   if b.get("pcpu", 0) >= 40.0]
            s["_contaminated"] = bool(hot)
            s["_hot_processes"] = [f"{b['pcpu']:.0f}% {b['command'][:60]}" for b in hot]
            sessions.append(s)
    # text lengths explain the scenario-driven tail (long transcripts = more encoder work)
    try:
        from evaluation.phase4.evaluate import _held_out_rows
        held = _held_out_rows()
        for i, r in enumerate(held[:meta["n_scenarios"]]):
            meta["scenario_ids"][i].update(scene_chars=len(r["scene"]), transcript_chars=len(r["transcript"]))
    except Exception as e:  # pragma: no cover
        meta["scenario_text_lengths_error"] = repr(e)
    hist = json.loads((MAC / "intel" / "artifacts" / "e2e_mac.json").read_text())
    hist_rows = {r["backend"]: r for r in hist["results"]}
    n = meta["n_scenarios"]
    summary = {"issue": "I-10", "phase": "summarise", "run_dirs": [str(d.relative_to(REPO)) for d in run_dirs], "meta": meta,
               "historical_e2e_mac": {b: {k: r.get(k) for k in ("p50_ms", "p90_ms", "mean_ms", "repeats", "n",
                                                                  "board_jaccard_vs_torch", "first_call_incl_load_s")}
                                      for b, r in hist_rows.items()},
               "families": {}}
    lines = []
    for fam in sorted({s["family"] for s in sessions}):
        fs = [s for s in sessions if s["family"] == fam]
        fam_out = {"per_session": [], "pooled": {}, "board_agreement": {}, "between_session": {},
                   "historical_protocol_control": {}, "slowest_scenarios": {}}
        # fixed reference: the torch boards of this family (identical across every torch session — checked)
        torch_boards = [s["boards"] for s in fs if s["backend"] == "torch"]
        ref = torch_boards[0]
        fam_out["board_agreement"]["torch_sessions_identical_boards"] = all(b == ref for b in torch_boards)
        for s in fs:
            flat = [v for rnd in s["samples_ms"] for v in rnd]
            fam_out["per_session"].append({
                "session_id": s["session_id"], "file": s["_file"], "run": s["_run"], "tag": s["tag"], "position": s["position"],
                "contaminated": s["_contaminated"], "hot_processes": s["_hot_processes"],
                "backend": s["backend"], "evict": s["protocol"]["evict"], "full_reset": s["protocol"]["full_reset"],
                "repeats": s["protocol"]["repeats"], **_stats(flat),
                "per_round_p50": [_pct(r, 50) for r in s["samples_ms"]],
                "per_round_p90": [_pct(r, 90) for r in s["samples_ms"]],
                "jaccard_vs_torch_ref": float(np.mean([jaccard(a, b) for a, b in zip(s["boards"], ref)])),
                "identical_boards_vs_torch_ref": int(sum(a == b for a, b in zip(s["boards"], ref))),
                "head_artifact": s["label_checks"]["head_artifact"], "embed_backend_used": s["label_checks"]["embed_backend_used"],
                "student_runtime": s["label_checks"]["student_runtime"],
                "load_before_1m": s["load_before"]["loadavg_1_5_15"][0], "load_after_1m": s["load_after"]["loadavg_1_5_15"][0],
                "busy_before": len(s["load_before"]["busy_processes_ge3pct"]), "busy_after": len(s["load_after"]["busy_processes_ge3pct"]),
                "first_call_incl_load_s": s["first_call_incl_load_s"],
            })
        for backend in BACKEND_NAMES:
            main = [s for s in fs if s["backend"] == backend and s["tag"].startswith("order")]
            iso = [s for s in fs if s["backend"] == backend and s["tag"] == "isolated"]
            hp = [s for s in fs if s["backend"] == backend and s["tag"] == "histproto"]
            boards_all = [s["boards"] for s in main + iso + hp]
            fam_out["board_agreement"][backend] = {
                "sessions": len(boards_all),
                "identical_across_sessions": all(b == boards_all[0] for b in boards_all),
                "jaccard_vs_torch_ref": float(np.mean([jaccard(a, b) for a, b in zip(boards_all[0], ref)])),
                "identical_boards_vs_torch_ref": int(sum(a == b for a, b in zip(boards_all[0], ref))),
                "n": n,
            }
            # pooled over the three counterbalanced in-process sessions (the pre-registered main estimate)
            by_scn = [[] for _ in range(n)]
            for s in main:
                for rnd in s["samples_ms"]:
                    for i, v in enumerate(rnd):
                        by_scn[i].append(v)
            flat = [v for sc in by_scn for v in sc]
            pooled = _stats(flat)
            pooled["ci95_p50_scenario_bootstrap"] = _bootstrap_scenario(by_scn, 50)
            pooled["ci95_p90_scenario_bootstrap"] = _bootstrap_scenario(by_scn, 90)
            pooled["n_sessions"] = len(main)
            pooled["sessions"] = [s["session_id"] + " @" + s["_run"] for s in main]
            # sensitivity: only counterbalanced sessions with no evidence of external load
            clean = [s for s in main if not s["_contaminated"]]
            by_scn_c = [[] for _ in range(n)]
            for s in clean:
                for rnd in s["samples_ms"]:
                    for i, v in enumerate(rnd):
                        by_scn_c[i].append(v)
            if clean:
                pooled_clean = _stats([v for sc in by_scn_c for v in sc])
                pooled_clean["ci95_p50_scenario_bootstrap"] = _bootstrap_scenario(by_scn_c, 50)
                pooled_clean["ci95_p90_scenario_bootstrap"] = _bootstrap_scenario(by_scn_c, 90)
            else:
                pooled_clean = {}
            pooled_clean["n_sessions"] = len(clean)
            pooled_clean["sessions"] = [s["session_id"] + " @" + s["_run"] for s in clean]
            # + isolated processes pooled in as well (secondary)
            by_scn_all = [list(x) for x in by_scn]
            for s in iso:
                for rnd in s["samples_ms"]:
                    for i, v in enumerate(rnd):
                        by_scn_all[i].append(v)
            flat_all = [v for sc in by_scn_all for v in sc]
            pooled_all = _stats(flat_all)
            pooled_all["n_sessions"] = len(main) + len(iso)
            fam_out["pooled"][backend] = {"counterbalanced_inprocess": pooled, "clean_counterbalanced": pooled_clean,
                                          "plus_isolated": pooled_all}
            # between-session variation
            p50s = [_pct([v for r in s["samples_ms"] for v in r], 50) for s in main + iso]
            p90s = [_pct([v for r in s["samples_ms"] for v in r], 90) for s in main + iso]
            fam_out["between_session"][backend] = {
                "session_p50s": p50s, "session_p90s": p90s,
                "p50_range": [min(p50s), max(p50s)], "p90_range": [min(p90s), max(p90s)],
                "p50_sd": float(np.std(p50s, ddof=1)) if len(p50s) > 1 else None,
                "p90_sd": float(np.std(p90s, ddof=1)) if len(p90s) > 1 else None,
                "by_position": {str(s["position"]): {"tag": s["tag"], "p50": _pct([v for r in s["samples_ms"] for v in r], 50),
                                                     "p90": _pct([v for r in s["samples_ms"] for v in r], 90)}
                                for s in main},
                "isolated": [{"p50": _pct([v for r in s["samples_ms"] for v in r], 50),
                              "p90": _pct([v for r in s["samples_ms"] for v in r], 90)} for s in iso],
            }
            if hp:
                hflat = [v for r in hp[0]["samples_ms"] for v in r]
                fam_out["historical_protocol_control"][backend] = {**_stats(hflat), "position": hp[0]["position"],
                                                                   "per_round_p50": [_pct(r, 50) for r in hp[0]["samples_ms"]],
                                                                   "per_round_p90": [_pct(r, 90) for r in hp[0]["samples_ms"]]}
            # slowest scenarios by median over all counterbalanced samples
            med = [float(np.median(sc)) for sc in by_scn]
            top = sorted(range(n), key=lambda i: -med[i])[:5]
            fam_out["slowest_scenarios"][backend] = [{"index": i, **meta["scenario_ids"][i], "median_ms": med[i],
                                                      "max_ms": float(max(by_scn[i]))} for i in top]
            fam_out["slowest_scenarios"][backend + "_share_of_samples_above_pooled_p90_from_top5"] = float(
                sum(1 for i in top for v in by_scn[i] if v > pooled["p90"]) / max(1, sum(1 for v in flat if v > pooled["p90"])))
        summary["families"][fam] = fam_out
        # markdown
        lines.append(f"\n### family `{fam}`  (n={n}, pooled over {len(main)} counterbalanced sessions x {main and main[0]['protocol']['repeats']} repeats)\n")
        lines.append("| backend | head | embed | p25 | p50 [95% CI] | p75 | p90 [95% CI] | mean | max | clean sessions p50 / p90 (k) | Jaccard vs torch | session p50 range | session p90 range | hist-protocol p50/p90 | e2e_mac.json p50/p90 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for backend in BACKEND_NAMES:
            P = fam_out["pooled"][backend]["counterbalanced_inprocess"]
            B = fam_out["between_session"][backend]
            A = fam_out["board_agreement"][backend]
            H = fam_out["historical_protocol_control"].get(backend, {})
            E = hist_rows[backend]
            C = fam_out["pooled"][backend]["clean_counterbalanced"]
            ps = [s for s in fam_out["per_session"] if s["backend"] == backend][0]
            lines.append(f"| {backend} | {ps['student_runtime']} | {ps['embed_backend_used']} | {P['p25']:.2f} | "
                         f"{P['p50']:.2f} [{P['ci95_p50_scenario_bootstrap'][0]:.2f}, {P['ci95_p50_scenario_bootstrap'][1]:.2f}] | "
                         f"{P['p75']:.2f} | {P['p90']:.2f} [{P['ci95_p90_scenario_bootstrap'][0]:.2f}, {P['ci95_p90_scenario_bootstrap'][1]:.2f}] | "
                         f"{P['mean']:.2f} | {P['max']:.1f} | "
                         + (f"{C['p50']:.2f} / {C['p90']:.2f} ({C['n_sessions']})" if C.get('p50') is not None else f"— (0)") + " | "
                         f"{A['jaccard_vs_torch_ref']:.3f} ({A['identical_boards_vs_torch_ref']}/{n}) | "
                         f"{B['p50_range'][0]:.2f}–{B['p50_range'][1]:.2f} | {B['p90_range'][0]:.2f}–{B['p90_range'][1]:.2f} | "
                         f"{H.get('p50', float('nan')):.2f} / {H.get('p90', float('nan')):.2f} | {E['p50_ms']:.2f} / {E['p90_ms']:.2f} |")
        lines.append("\nPer session (pos = position in its process; iso = own process; hp = historical protocol):\n")
        lines.append("| run | session | backend | pos | evict | ext. load ≥40% | p25 | p50 | p75 | p90 | mean | max | per-round p50 | load 1m before→after | busy procs |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for s in fam_out["per_session"]:
            lines.append(f"| {s['run']} | {s['tag']} | {s['backend']} | {s['position']} | {s['evict']} | {'YES' if s['contaminated'] else 'no'} | {s['p25']:.2f} | {s['p50']:.2f} | {s['p75']:.2f} | "
                         f"{s['p90']:.2f} | {s['mean']:.2f} | {s['max']:.1f} | {', '.join(f'{v:.1f}' for v in s['per_round_p50'])} | "
                         f"{s['load_before_1m']:.2f}→{s['load_after_1m']:.2f} | {s['busy_before']}/{s['busy_after']} |")
    md = "\n".join(lines)
    write_json(Path(args.out), summary)
    Path(args.out).with_suffix(".md").write_text(md + "\n")
    print(md)
    print(f"\n[summarise] wrote {args.out} and {Path(args.out).with_suffix('.md')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sessions")
    s.add_argument("--family", choices=tuple(FAMILIES), required=True)
    s.add_argument("--tag", required=True)
    s.add_argument("--backends", required=True, help="comma-separated, run in this order in this process")
    s.add_argument("--n", type=int, default=84)
    s.add_argument("--repeats", type=int, default=10)
    s.add_argument("--warmup", type=int, default=3)
    s.add_argument("--evict", choices=("round", "board"), default="board")
    s.add_argument("--no-full-reset", action="store_true")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_sessions)
    r = sub.add_parser("run")
    r.add_argument("--families", default="hist,i05,i05b")
    r.add_argument("--n", type=int, default=84)
    r.add_argument("--repeats", type=int, default=10)
    r.add_argument("--warmup", type=int, default=3)
    r.add_argument("--out-dir", required=True)
    r.set_defaults(fn=cmd_run)
    m = sub.add_parser("summarise")
    m.add_argument("--run-dir", required=True, nargs="+")
    m.add_argument("--out", required=True)
    m.set_defaults(fn=cmd_summarise)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
