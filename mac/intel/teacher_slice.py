"""The DATA stage on Intel hardware: generate teacher boards with the real
Qwen2.5-3B-Instruct teacher through transformers on this machine, for a slice
of the deictic set, and record device + latency per board.

    VOICELINK_PERCEPTION=0 python3 intel/teacher_slice.py [--n 10] [--device cpu|xpu]

Writes intel/artifacts/teacher_slice_<host>.jsonl (boards + timing) and a
hardware fingerprint, WITHOUT touching teacher_data.jsonl (the training corpus
is fingerprinted; regenerating rows into it would invalidate the held-out split).
On an Intel CPU the optional intel_extension_for_pytorch is applied if present.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAC = HERE.parent
if str(MAC) not in sys.path:
    sys.path.insert(0, str(MAC))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--device", default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    os.environ.setdefault("VOICELINK_PERCEPTION", "0")

    from intel import hardware_fingerprint
    from ranker import HfRanker
    import rank_vocabulary as rv

    rows = [json.loads(l) for l in
            (MAC / "evaluation/phase4/deictic_scenarios.jsonl").read_text().splitlines() if l.strip()]
    rows = rows[:args.n]
    t0 = time.perf_counter()
    r = HfRanker(args.model, temp=0.0, device=args.device)
    load_s = time.perf_counter() - t0
    try:  # IPEX inference optimisation on Intel CPUs; harmless no-op elsewhere
        if r.device == "cpu":
            import intel_extension_for_pytorch as ipex
            r._model = ipex.optimize(r._model)
            ipex_v = ipex.__version__
        else:
            ipex_v = None
    except Exception:
        ipex_v = None
    hw = hardware_fingerprint()
    out_path = HERE / "artifacts" / f"teacher_slice_{socket.gethostname().split('.')[0]}.jsonl"
    out_path.parent.mkdir(exist_ok=True)
    lat = []
    with out_path.open("w") as f:
        f.write(json.dumps({"_meta": True, "hardware": hw, "teacher_device": r.device,
                            "model": r.model_id, "ipex": ipex_v, "load_s": round(load_s, 1),
                            "n": len(rows)}) + "\n")
        for s in rows:
            ctx = {"last_utterance": s["transcript"], "addressed": True,
                   "active_question": s.get("active_question")}
            t1 = time.perf_counter()
            board = [w["word"] for w in rv.rank_vocabulary(s["scene"], s["transcript"], ctx,
                                                           ranker=r, strategy="diverse")]
            dt = time.perf_counter() - t1
            lat.append(dt)
            f.write(json.dumps({"id": s["id"], "scene": s["scene"], "transcript": s["transcript"],
                                "is_question": True, "board": board, "latency_s": round(dt, 2),
                                "expected_answers": s.get("expected_answers")}) + "\n")
            hit = any(a.lower() in " ".join(board).lower() for a in s.get("expected_answers", []))
            print(f"{s['id']:12s} {dt:5.1f}s hit={hit} {board}", flush=True)
    lat.sort()
    print(f"\nteacher on {r.device} ({hw['cpu']}): load {load_s:.0f}s, "
          f"p50 {lat[len(lat)//2]:.1f}s/board over {len(lat)} boards -> {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
