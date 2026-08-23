"""End-to-end BOARD latency through the seam, per backend — the number a paper
or the Intel entry may quote as "time to a board".

    VOICELINK_PERCEPTION=0 python3 intel/bench_e2e.py [--n 40] [--device CPU|NPU|GPU]

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


def load_scenarios(n: int) -> list[dict]:
    """The Phase-4 HELD-OUT rows (same loader the eval gates use), so the latency
    figure is measured on contexts the student never trained on. Falls back to
    v2_scenarios.jsonl when the training corpus is unavailable."""
    try:
        from evaluation.phase4.evaluate import _held_out_rows
        rows = _held_out_rows()
    except Exception as e:
        print(f"[bench_e2e] held-out rows unavailable ({e}); using v2_scenarios")
        rows = [json.loads(l) for l in SCENARIOS.read_text().splitlines() if l.strip()]
    return rows[:n]


def _reset_seam(env: dict[str, str]) -> None:
    """Apply a backend env and drop every cached singleton so the next call
    actually loads the requested runtime."""
    for k, v in env.items():
        os.environ[k] = v
    import ranker_config
    importlib.reload(ranker_config)
    import rank_vocabulary as rv
    rv._student_singleton = None
    import evaluation.embed as em
    em._model = None
    em._backend_loaded = None


def bench_backend(name: str, scenarios: list[dict], warmup: int = 3, repeats: int = 3) -> dict:
    _reset_seam(BACKENDS[name])
    import rank_vocabulary as rv
    from evaluation import embed as em

    def one(s):
        ctx = {"last_utterance": s.get("transcript", ""),
               "personal_context": s.get("profile", {})}
        return [w["word"] for w in rv.rank_vocabulary(
            s.get("scene"), s.get("transcript"), ctx, strategy="student")]

    t_load = time.perf_counter()
    one(scenarios[0])
    load_s = time.perf_counter() - t_load
    for s in scenarios[:warmup]:
        one(s)
    lat, boards = [], []
    ctx_texts = [s.get("scene", "") for s in scenarios] + \
                [s.get("transcript", "") for s in scenarios]
    for _ in range(repeats):
        # Vocabulary embeddings stay memoised (they are static on a device);
        # the scene/transcript of each board must be encoded fresh every time.
        em.cache_forget(ctx_texts)
        lat_round = []
        for s in scenarios:
            t0 = time.perf_counter()
            b = one(s)
            lat_round.append((time.perf_counter() - t0) * 1000.0)
            if len(boards) < len(scenarios):
                boards.append(b)
        lat.extend(lat_round)
    model, *_ = rv._load_student()
    return {
        "backend": name, "env": BACKENDS[name],
        "embed_backend_used": em.backend_in_use(),
        "student_runtime": type(model).__name__,
        "ov_device": getattr(model, "device", None),
        "n": len(scenarios), "repeats": repeats,
        "first_call_incl_load_s": round(load_s, 3),
        "p50_ms": float(np.percentile(lat, 50)), "p90_ms": float(np.percentile(lat, 90)),
        "mean_ms": float(np.mean(lat)), "boards": boards,
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
    args = ap.parse_args()
    if args.device:
        os.environ["VOICELINK_OV_DEVICE"] = args.device
    os.environ.setdefault("VOICELINK_PERCEPTION", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from intel import hardware_fingerprint
    hw = hardware_fingerprint()
    scenarios = load_scenarios(args.n)
    results = []
    for name in args.backends.split(","):
        try:
            r = bench_backend(name, scenarios)
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
           "scenario_source": "phase4 held-out split (evaluate._held_out_rows)",
           "results": [{k: v for k, v in r.items() if k != "boards"} for r in results]}
    host = socket.gethostname().split(".")[0]
    out_path = HERE / "artifacts" / f"e2e_{host}.json"
    out_path.parent.mkdir(exist_ok=True)
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
