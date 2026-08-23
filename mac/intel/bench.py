"""Benchmark torch-CPU vs OV-FP32 vs OV-INT8 for the student ranker (Track 3, step 4).

    VOICELINK_PERCEPTION=0 python3 intel/bench.py

Two measurements, both real:
  1. A Python-API timing loop (batch=1, the runtime shape) — p50/p90 wall latency.
  2. benchmark_app (OpenVINO's standard tool, latency hint) — subprocess, IRs only.
Writes mac/intel/BENCHMARKS.md. Machine is labeled honestly: Apple Silicon via the
OpenVINO ARM CPU plugin (dev stage); Intel-hardware numbers pending Tiber access.
"""

from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MAC = HERE.parent
if str(MAC) not in sys.path:
    sys.path.insert(0, str(MAC))

from intel import OpenVinoStudent, STUDENT_FP32_XML, STUDENT_INT8_XML  # noqa: E402

STUDENT_PT = MAC / "evaluation" / "phase4" / "student.pt"
STUDENT_ONNX = MAC / "evaluation" / "phase4" / "student.onnx"
ACCURACY_JSON = HERE / "artifacts" / "int8_accuracy.json"
BENCHMARKS_MD = HERE / "BENCHMARKS.md"

N_WARMUP = 50
N_ITERS = 500


def load_torch_student():
    import torch
    from evaluation.phase4.student import StudentRanker
    ck = torch.load(STUDENT_PT, map_location="cpu", weights_only=False)
    model = StudentRanker(ck["context_dim"], ck["hidden_dim"],
                          ck["vocab_size"], ck["dropout"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model


def time_loop(fn, x: np.ndarray) -> dict:
    """p50/p90 latency in ms for fn(x) over N_ITERS single-input calls."""
    for _ in range(N_WARMUP):
        fn(x)
    samples = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter()
        fn(x)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return {"p50_ms": float(np.percentile(samples, 50)),
            "p90_ms": float(np.percentile(samples, 90))}


def bench_torch(x: np.ndarray) -> dict:
    import torch
    model = load_torch_student()
    xt = torch.from_numpy(x)
    with torch.no_grad():
        return time_loop(lambda inp: model(xt), x)


def bench_openvino(xml_path: Path, x: np.ndarray) -> dict:
    student = OpenVinoStudent(xml_path)
    return time_loop(student, x)


def run_benchmark_app(xml_path: Path) -> str:
    """benchmark_app summary lines, or an honest note when unavailable."""
    exe = shutil.which("benchmark_app")
    if exe is None:
        return "benchmark_app not on PATH (not shipped by this pip wheel)"
    r = subprocess.run(
        [exe, "-m", str(xml_path), "-d", "CPU", "-hint", "latency", "-t", "5",
         "-data_shape", "context_emb[1,769]"],  # batch axis is dynamic in the IR
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return (f"benchmark_app failed (rc={r.returncode}): "
                f"{(r.stderr or r.stdout)[-500:]}")
    tail = [ln.strip() for ln in r.stdout.splitlines()
            if re.search(r"(Count|Duration|Median|Average|Min|Max|Throughput):", ln)]
    return "\n".join(tail) or r.stdout[-500:]


def size_kb(*paths: Path) -> float:
    return sum(p.stat().st_size for p in paths if p.exists()) / 1024.0


def machine_label() -> str:
    from intel import hardware_fingerprint
    hw = hardware_fingerprint()
    tag = ("Intel hardware" if hw["intel_hardware"]
           else "NOT Intel hardware — OpenVINO ARM CPU plugin (dev stage)")
    return f"{hw['cpu'] or hw['arch']} / {hw['os']} — {tag}; OV devices {hw['ov_devices']}"


def write_report(rows: list[dict], bench_app: dict[str, str], note: dict) -> None:
    import openvino as ov
    lines = [
        "# Student ranker benchmarks — torch CPU vs OpenVINO FP32 vs OpenVINO INT8",
        "",
        f"Machine: {machine_label()}",
        f"OpenVINO {ov.__version__} | batch=1, input (1, 769) float32 | "
        f"{N_ITERS} timed iterations after {N_WARMUP} warmup",
        "",
        "| backend | latency p50 (ms) | latency p90 (ms) | size on disk (KB) |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['name']} | {r['p50_ms']:.3f} | {r['p90_ms']:.3f} "
                     f"| {r['size_kb']:.0f} |")
    lines += ["", "## INT8 accuracy vs FP32 (real calibration scenarios)", "",
              "```json", json.dumps(note, indent=2), "```", "",
              "## benchmark_app (latency hint, 5 s each)", ""]
    for name, out in bench_app.items():
        lines += [f"### {name}", "```", out, "```", ""]
    BENCHMARKS_MD.write_text("\n".join(lines))


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-md", action="store_true",
                    help="print the table but do not rewrite BENCHMARKS.md")
    ap.add_argument("--skip-benchmark-app", action="store_true")
    args = ap.parse_args()
    x = np.random.default_rng(0).standard_normal((1, 769)).astype(np.float32)
    rows = [
        {"name": "torch CPU (student.pt)", **bench_torch(x),
         "size_kb": size_kb(STUDENT_PT)},
        {"name": "OpenVINO FP32 IR", **bench_openvino(STUDENT_FP32_XML, x),
         "size_kb": size_kb(STUDENT_FP32_XML, STUDENT_FP32_XML.with_suffix(".bin"))},
        {"name": "OpenVINO INT8 IR", **bench_openvino(STUDENT_INT8_XML, x),
         "size_kb": size_kb(STUDENT_INT8_XML, STUDENT_INT8_XML.with_suffix(".bin"))},
    ]
    bench_app = {} if args.skip_benchmark_app else {
        p.name: run_benchmark_app(p) for p in (STUDENT_FP32_XML, STUDENT_INT8_XML)}
    note = json.loads(ACCURACY_JSON.read_text()) if ACCURACY_JSON.exists() \
        else {"note": "run quantize.py first"}
    # The e2e section (bench_e2e.py) is appended AFTER this table; preserve it
    # across rewrites so a head-only re-run never silently drops a board number.
    existing = BENCHMARKS_MD.read_text() if BENCHMARKS_MD.exists() else ""
    tail = existing[existing.index("\n## End-to-end"):] if "\n## End-to-end" in existing else ""
    if args.no_md:
        print(json.dumps(rows, indent=2))
        return 0
    write_report(rows, bench_app, note)
    if tail:
        BENCHMARKS_MD.write_text(BENCHMARKS_MD.read_text() + tail)
    print(BENCHMARKS_MD.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
