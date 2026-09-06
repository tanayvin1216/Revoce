"""I-12 smoke: run intel.bench_e2e.bench_backend for every backend on a few
held-out scenarios and record which embedder ACTUALLY served each row, without
writing to intel/artifacts (this is not a latency result — n is tiny and the
machine is not idle; I-10 owns the real rerun).

    cd mac && VOICELINK_PERCEPTION=0 /usr/local/bin/python3 \
        ../paper/odi/revision/analyses/I-12_bench_smoke.py [--n 10]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAC = HERE.parents[3] / "mac"
if str(MAC) not in sys.path:
    sys.path.insert(0, str(MAC))
os.environ.setdefault("VOICELINK_PERCEPTION", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["VOICELINK_EMBED_STRICT"] = "1"  # what bench_e2e.main() now sets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default=str(HERE / "I-12_bench_smoke.json"))
    args = ap.parse_args()
    from intel import bench_e2e, hardware_fingerprint
    scenarios, source = bench_e2e.load_scenarios(args.n)
    out = {"hardware": hardware_fingerprint(), "n": len(scenarios), "scenario_source": source,
           "strict": os.environ["VOICELINK_EMBED_STRICT"], "results": []}
    for name in bench_e2e.BACKENDS:
        t0 = time.perf_counter()
        try:
            r = bench_e2e.bench_backend(name, scenarios)
            r.pop("boards", None)
        except Exception as e:
            r = {"backend": name, "error": f"{type(e).__name__}: {e}"}
        r["wall_s"] = round(time.perf_counter() - t0, 2)
        out["results"].append(r)
        print(name, {k: r[k] for k in r if k in ("embed_backend_used", "student_runtime", "p50_ms", "error")})
    # Negative control: point the loader at an empty dir and prove ov-full becomes an
    # error row instead of a torch row under the ov-full label.
    import intel
    import evaluation.embed as em
    empty = HERE / "I-12_empty_bundle_dir"
    empty.mkdir(exist_ok=True)
    saved = intel.MINILM_OV_DIR
    intel.MINILM_OV_DIR = empty
    orig_init = intel.OpenVinoEmbedder.__init__
    intel.OpenVinoEmbedder.__init__ = lambda self, model_dir=empty: orig_init(self, model_dir)
    try:
        bench_e2e._reset_seam(bench_e2e.BACKENDS["ov-full"])
        em.reset_backend()
        r = bench_e2e.bench_backend("ov-full", scenarios[:2])
        out["negative_control"] = {"outcome": "MISLABELLED ROW SURVIVED", "row": {k: r[k] for k in r if k != "boards"}}
    except Exception as e:
        out["negative_control"] = {"outcome": "raised as required", "error": f"{type(e).__name__}: {e}"}
    finally:
        intel.OpenVinoEmbedder.__init__ = orig_init
        intel.MINILM_OV_DIR = saved
        em.reset_backend()
        try:
            empty.rmdir()
        except OSError:
            pass
    print("negative control:", out["negative_control"])
    Path(args.out).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
