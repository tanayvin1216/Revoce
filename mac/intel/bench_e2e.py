"""End-to-end BOARD latency through the seam, per backend — the number a paper
or the Intel entry may quote as "time to a board".

    VOICELINK_PERCEPTION=0 python3 intel/bench_e2e.py [--n 40] [--device CPU|NPU|GPU]
        [--repeats 3] [--warmup 3] [--evict round|board] [--full-reset] [--out PATH]

The bracketed flags (I-10) default to the historical protocol; every result row
now also carries p25/p75, min/max and the raw per-round `samples_ms` vector.

Unlike intel/bench.py (bare MLP forward pass, ~0.03 ms — NOT a board), this
times rank_vocabulary(strategy="student") on REAL held-out scenarios: MiniLM
embedding of the (never-seen) scene+transcript + student head + open-answer
extraction + symbolic MMR selection, with the static vocabulary embeddings
memoised as they are on a live device, for three runtime combinations:

    torch       : torch MiniLM + torch MLP          (the Mac dev path)
    ov-head     : torch MiniLM + OpenVINO INT8 MLP   (Phase-5 as shipped)
    ov-full     : OpenVINO MiniLM + OpenVINO INT8 MLP (torch-free; Intel target)

Every row carries a hardware fingerprint (intel.hardware_fingerprint) so a run
on Tiber / a Core Ultra box is self-labelling. Results go to
intel/artifacts/e2e_<hostname>.json and are appended as a section to
intel/BENCHMARKS.md. Boards from all three backends are compared for agreement
so a speedup is never reported over a board that silently changed.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MAC = HERE.parent
if str(MAC) not in sys.path:
    sys.path.insert(0, str(MAC))

SCENARIOS = MAC / "evaluation" / "phase4" / "v2_scenarios.jsonl"
SPLITS = MAC / "evaluation" / "phase4" / "splits.json"
BENCHMARKS_MD = HERE / "BENCHMARKS.md"

BACKENDS = {
    "torch":   {"VOICELINK_STUDENT_BACKEND": "torch",    "VOICELINK_EMBED_BACKEND": "torch"},
    "ov-head": {"VOICELINK_STUDENT_BACKEND": "openvino", "VOICELINK_EMBED_BACKEND": "torch"},
    "ov-full": {"VOICELINK_STUDENT_BACKEND": "openvino", "VOICELINK_EMBED_BACKEND": "openvino"},
}


def load_scenarios(n: int) -> tuple[list[dict], str]:
    """The Phase-4 HELD-OUT rows (same loader the eval gates use), so the latency
    figure is measured on contexts the student never trained on. Falls back to
    v2_scenarios.jsonl when the training corpus is unavailable.

    Returns (rows, source_label). The label is derived, never assumed: the written
    JSON used to hardcode "held-out split" even when this fell back, which would
    publish a latency number measured on the wrong scenarios under a label saying
    otherwise."""
    try:
        from evaluation.phase4.evaluate import _held_out_rows
        rows = _held_out_rows()
        source = "phase4 held-out split (evaluate._held_out_rows)"
    except Exception as e:
        print(f"[bench_e2e] held-out rows unavailable ({e}); using v2_scenarios")
        rows = [json.loads(l) for l in SCENARIOS.read_text().splitlines() if l.strip()]
        source = f"FALLBACK v2_scenarios.jsonl — held-out rows unavailable ({e})"
    return rows[:n], source


def _reset_seam(env: dict[str, str], full: bool = False) -> None:
    """Apply a backend env and drop every cached singleton so the next call
    actually loads the requested runtime.

    full=True (I-10) additionally puts the process in the SAME state every
    session starts from, whatever ran before it: the embedding memo, the
    remembered OpenVINO-load failure (`_ov_failed`), the seam's one-shot
    fallback flag, the taxonomy exemplar matrix and the answer-prototype memo
    are all cleared. Default off so the historical protocol is unchanged.
    """
    for k, v in env.items():
        os.environ[k] = v
    import ranker_config
    importlib.reload(ranker_config)
    import rank_vocabulary as rv
    rv._student_singleton = None
    import evaluation.embed as em
    em._model = None
    em._backend_loaded = None
    if full:
        em.reset_backend()          # also forgets _ov_failed
        em.cache_clear()
        rv._student_failure_reported = False
        from evaluation.phase3 import taxonomy
        taxonomy._proto_vecs = None
        taxonomy._proto_funcs = []
        from evaluation.phase4 import answers, infer
        answers._prototype_vec.cache_clear()
        infer._pool_mult.cache_clear()


def _static_texts(vocab: list[str], core: list[str]) -> list[str]:
    """Everything a live device has memoised before the first board: the closed
    vocabulary, the pinned core words, the taxonomy exemplars and the answer
    prototypes. Pre-warming these explicitly (evict='board') makes round 1 as
    warm as round N; the historical protocol warmed the vocabulary only as far
    as the first warmup+round happened to reach into it."""
    from evaluation.phase3 import taxonomy
    texts = list(vocab) + list(core)
    for exs in taxonomy.FUNCTION_EXEMPLARS_V3.values():
        texts.extend(exs)
    return texts


def bench_backend(name: str, scenarios: list[dict], warmup: int = 3, repeats: int = 3,
                  student: tuple | None = None, full_reset: bool = False,
                  evict: str = "round") -> dict:
    """One backend session.

    student     — optional (model, vocab, core, k_dyn) tuple injected into the
                  seam after the reset, so a head that is NOT under
                  mac/intel/artifacts (a revision export) can be timed through
                  the same rank_vocabulary path. None = the native loader.
    full_reset  — see _reset_seam(full=True).
    evict       — 'round' (historical): forget the raw scene/transcript strings
                  once before each round; texts a board encodes on the side
                  (stripped transcript, open-answer spans) stay memoised, and a
                  scene shared by two held-out rows is warm for the second.
                  'board': pre-warm the static set, then evict after EVERY
                  timed board every (backend, text) key that board added, so
                  each context is encoded fresh on every call.
    """
    if evict not in ("round", "board"):
        raise ValueError(f"evict must be 'round' or 'board', got {evict!r}")
    _reset_seam(BACKENDS[name], full=full_reset)
    import rank_vocabulary as rv
    from evaluation import embed as em
    if student is not None:
        rv._student_singleton = tuple(student)
    if BACKENDS[name]["VOICELINK_EMBED_BACKEND"] == "openvino":
        # Cheap file check BEFORE the first call: if the bundle is incomplete the
        # seam would otherwise catch the strict embed error inside _student_board
        # and serve the LLM path for one board before assert_backend fires.
        from intel import minilm_bundle_status
        st = minilm_bundle_status()
        if not st["complete"]:
            raise FileNotFoundError(
                f"MiniLM IR bundle incomplete ({st['missing']} missing in {st['dir']}); "
                f"'{name}' cannot be measured — run intel/convert.py --skip-student")

    def one(s):
        ctx = {"last_utterance": s.get("transcript", ""),
               "personal_context": s.get("profile", {})}
        return [w["word"] for w in rv.rank_vocabulary(
            s.get("scene"), s.get("transcript"), ctx, strategy="student")]

    t_load = time.perf_counter()
    one(scenarios[0])
    load_s = time.perf_counter() - t_load
    # Hard label check (I-12): the embedder that ACTUALLY served this row must be
    # the one the backend name promises. With VOICELINK_EMBED_STRICT set in main()
    # the fallback already raises at load; this catches the remaining case where
    # a stale singleton or an unset env slipped through. A mismatch becomes an
    # error row, never a mislabelled timing.
    em.assert_backend(BACKENDS[name]["VOICELINK_EMBED_BACKEND"])
    model, vocab, core, _k_dyn = rv._load_student()
    if student is not None and model is not student[0]:
        raise RuntimeError("injected student was not the one the seam served")
    static_keys: set = set()
    if evict == "board":
        static_texts = set(_static_texts(vocab, core))
        em.embed(sorted(static_texts))         # the live device's warm set
        # Only the static texts count as warm. (An earlier draft snapshotted every
        # key here, which silently kept the load call's scenario — scenarios[0] —
        # warm for the whole session: 1 of n boards then timed at ~0.3 ms.)
        static_keys = {key for key in em._cache if key[1] in static_texts}
    for s in scenarios[:warmup]:
        one(s)
    if evict == "board":                       # warmup contexts must not stay warm
        for key in set(em._cache.keys()) - static_keys:
            em._cache.pop(key, None)
    lat, boards = [], []
    rounds: list[list[float]] = []
    ctx_texts = [s.get("scene", "") for s in scenarios] + \
                [s.get("transcript", "") for s in scenarios]
    for _ in range(repeats):
        # Vocabulary embeddings stay memoised (they are static on a device);
        # the scene/transcript of each board must be encoded fresh every time.
        if evict == "round":
            em.cache_forget(ctx_texts)
        lat_round = []
        for s in scenarios:
            if evict == "board":
                before = set(em._cache.keys())
            t0 = time.perf_counter()
            b = one(s)
            lat_round.append((time.perf_counter() - t0) * 1000.0)
            if evict == "board":
                for key in set(em._cache.keys()) - before:
                    em._cache.pop(key, None)
            if len(boards) < len(scenarios):
                boards.append(b)
        lat.extend(lat_round)
        rounds.append(lat_round)
    if rv._student_failure_reported:
        raise RuntimeError("seam fell back to the LLM path during the session")
    em.assert_backend(BACKENDS[name]["VOICELINK_EMBED_BACKEND"])
    head_artifact = str(getattr(model, "xml_path", None) or "student.pt (torch checkpoint)")
    return {
        "backend": name, "env": BACKENDS[name],
        "embed_backend_used": em.backend_in_use(),
        "student_runtime": type(model).__name__,
        "ov_device": getattr(model, "device", None),
        "head_artifact": head_artifact, "vocab_size": len(vocab),
        "n": len(scenarios), "repeats": repeats, "warmup": warmup,
        "evict": evict, "full_reset": full_reset,
        "first_call_incl_load_s": round(load_s, 3),
        "p25_ms": float(np.percentile(lat, 25)),
        "p50_ms": float(np.percentile(lat, 50)),
        "p75_ms": float(np.percentile(lat, 75)),
        "p90_ms": float(np.percentile(lat, 90)),
        "mean_ms": float(np.mean(lat)),
        "min_ms": float(np.min(lat)), "max_ms": float(np.max(lat)),
        "samples_ms": rounds,          # [round][scenario], raw, unsorted
        "boards": boards,
    }


def board_agreement(a: list[list[str]], b: list[list[str]]) -> float:
    """Mean Jaccard over scenarios — 1.0 means the backend swap changed nothing."""
    js = []
    for x, y in zip(a, b):
        sx, sy = set(map(str.lower, x)), set(map(str.lower, y))
        js.append(len(sx & sy) / max(len(sx | sy), 1))
    return float(np.mean(js)) if js else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--device", default=None, help="OpenVINO device for the head")
    ap.add_argument("--backends", default=",".join(BACKENDS))
    ap.add_argument("--no-md", action="store_true", help="don't touch BENCHMARKS.md")
    # I-10 additions — every default reproduces the historical protocol exactly.
    ap.add_argument("--repeats", type=int, default=3, help="timed rounds per backend")
    ap.add_argument("--warmup", type=int, default=3, help="untimed boards before the rounds")
    ap.add_argument("--evict", choices=("round", "board"), default="round",
                    help="cache eviction rule (see bench_backend)")
    ap.add_argument("--full-reset", action="store_true",
                    help="clear memo/failure/taxonomy state before every backend session")
    ap.add_argument("--out", default=None,
                    help="write the JSON here instead of intel/artifacts/e2e_<host>.json "
                         "(refuses to overwrite an existing file)")
    args = ap.parse_args()
    if args.out and Path(args.out).exists():
        ap.error(f"--out {args.out} exists; choose a new path (artifacts are never overwritten)")
    if args.device:
        os.environ["VOICELINK_OV_DEVICE"] = args.device
    os.environ.setdefault("VOICELINK_PERCEPTION", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Never let a warned torch fallback be timed under an OpenVINO label (I-12).
    os.environ["VOICELINK_EMBED_STRICT"] = "1"

    from intel import hardware_fingerprint
    hw = hardware_fingerprint()
    scenarios, scenario_source = load_scenarios(args.n)
    if args.backends.split(",")[0] != "torch":
        print(f"[bench_e2e] WARNING: '{args.backends.split(',')[0]}' is first in "
              f"--backends, so board agreement is measured against IT, not torch. "
              f"Put torch first for a comparable Jaccard column.")
    results = []
    for name in args.backends.split(","):
        try:
            r = bench_backend(name, scenarios, warmup=args.warmup, repeats=args.repeats,
                              full_reset=args.full_reset, evict=args.evict)
        except Exception as e:  # an unavailable runtime is a row, not a crash
            r = {"backend": name, "error": f"{type(e).__name__}: {e}"}
        results.append(r)
        print(f"{name:8s} " + (f"p50 {r['p50_ms']:.1f} ms  p90 {r['p90_ms']:.1f} ms  "
                               f"embed={r['embed_backend_used']} head={r['student_runtime']}"
                               if "p50_ms" in r else r["error"]))
    ok = [r for r in results if "boards" in r]
    ref = ok[0]["boards"] if ok else []
    for r in ok:
        r["board_jaccard_vs_" + ok[0]["backend"]] = board_agreement(ref, r["boards"])
    out = {"hardware": hw, "n_scenarios": len(scenarios),
           "scenario_source": scenario_source,
           "ov_device_requested": args.device or "CPU (default)",
           "protocol": {"repeats": args.repeats, "warmup": args.warmup, "evict": args.evict,
                        "full_reset": args.full_reset, "backend_order": args.backends.split(",")},
           "results": [{k: v for k, v in r.items() if k != "boards"} for r in results]}
    host = socket.gethostname().split(".")[0]
    # Scope the filename by device: three --device runs on one box used to write
    # the same path, so CPU/GPU/NPU silently overwrote each other and only the
    # last survived.
    suffix = f"_{args.device.lower()}" if args.device else ""
    out_path = Path(args.out) if args.out else HERE / "artifacts" / f"e2e_{host}{suffix}.json"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")

    if not args.no_md:
        lines = ["", f"## End-to-end board latency — {hw['cpu'] or hw['arch']}", "",
                 f"`intel/bench_e2e.py --n {len(scenarios)}` | {hw['os']} | "
                 f"OpenVINO {hw['openvino']} | devices {hw['ov_devices']} | "
                 f"Intel hardware: {'YES' if hw['intel_hardware'] else 'no'}", "",
                 "| backend | embed | head | device | p50 (ms) | p90 (ms) | board Jaccard vs torch |",
                 "|---|---|---|---|---|---|---|"]
        for r in out["results"]:
            if "p50_ms" in r:
                jac = r.get("board_jaccard_vs_torch", r.get(f"board_jaccard_vs_{ok[0]['backend']}", float('nan')))
                lines.append(f"| {r['backend']} | {r['embed_backend_used']} | {r['student_runtime']} "
                             f"| {r['ov_device'] or 'cpu'} | {r['p50_ms']:.1f} | {r['p90_ms']:.1f} | {jac:.3f} |")
            else:
                lines.append(f"| {r['backend']} | — | — | — | error | {r['error']} | |")
        with BENCHMARKS_MD.open("a") as f:
            f.write("\n".join(lines) + "\n")
        print(f"appended section to {BENCHMARKS_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
