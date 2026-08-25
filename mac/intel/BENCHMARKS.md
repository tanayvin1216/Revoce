# Student ranker benchmarks — torch CPU vs OpenVINO FP32 vs OpenVINO INT8

Two machines, measured with the same scripts and the same committed artifacts.

| | dev machine | Intel target |
|---|---|---|
| CPU | Apple M4 (arm64) | Intel(R) Core(TM) Ultra 9 185H |
| OS | Darwin 25.5.0 | Windows 11 |
| Python | 3.13.6 | 3.14.2 |
| OpenVINO | 2026.2.1-21919 | 2026.3.0-22451 |
| OV devices | `['CPU']` | `['CPU', 'GPU', 'NPU']` |
| Intel hardware | no (ARM CPU plugin) | **yes** |

The two machines differ in OS, Python and OpenVINO version, so **absolute
cross-machine latencies are not a controlled comparison**. What *is* controlled
is the torch-vs-OpenVINO ratio *within* each machine — same process, same
inputs, same artifacts, back to back.

---

## Student head, isolated (`intel/bench.py`)

batch=1, input `(1, 769)` float32 | 500 timed iterations after 50 warmup |
size on disk = `.xml` + `.bin`
Sources: `artifacts/head_bench_tanishlaptop.json`, and the M4 column below.

| backend | M4 p50 (ms) | M4 p90 | M4 size (KB) | Core Ultra p50 (ms) | Core Ultra p90 | Core Ultra size (KB) |
|---|---|---|---|---|---|---|
| torch CPU (`student.pt`) | 0.015 | 0.016 | 657 | 0.1075 | 0.1256 | 658 |
| OpenVINO FP32 IR | 0.025 | 0.028 | 661 | **0.0404** | 0.0522 | 662 |
| OpenVINO INT8 IR | 0.031 | 0.035 | 235 | **0.0426** | 0.1056 | 236 |

### The OpenVINO plugin inverts between the two CPUs

Against torch **on the same machine**, with the same 95-class head:

| | OpenVINO FP32 vs torch | OpenVINO INT8 vs torch |
|---|---|---|
| Apple M4 (ARM plugin) | **1.7× slower** | **2.1× slower** |
| Core Ultra 9 185H | **2.7× faster** | **2.5× faster** |

This is the cleanest Intel result in the project. It is a within-machine ratio,
so it is immune to the OS/Python/OpenVINO differences between the two boxes, and
the measurement is 500 iterations of a ~0.1 ms operation, so it is also
insensitive to machine load (observed p90/p50 spread: 1.17–1.29× on the torch
and FP32 rows). The OpenVINO CPU plugin is doing Intel-specific work; on ARM it
is a portability shim that costs performance.

**INT8 is not a speed win.** It is marginally *slower* than FP32 on both
machines (0.0426 vs 0.0404 on Intel; 0.031 vs 0.025 on ARM). Its result is the
**2.8× size reduction** (236 KB vs 662 KB) and the accuracy gate below. Quote it
as a toolchain and footprint result, never as a latency result.

## INT8 accuracy vs FP32 (real calibration scenarios)

```json
{
  "n_calibration_rows": 360,
  "k_dyn": 6,
  "max_abs_logit_delta": 1.0528192520141602,
  "mean_abs_logit_delta": 0.22787794470787048,
  "mean_topk_overlap": 0.9504629629629631,
  "min_topk_overlap": 0.6666666666666666
}
```

## benchmark_app (latency hint, 5 s each) — Apple M4

Not run on the Core Ultra (the run used `--skip-benchmark-app`; the
`head_bench_*.json` figures above cover the same ground).

### student_fp32.xml
```
[ INFO ] Count:            241769 iterations
[ INFO ] Duration:         5000.03 ms
[ INFO ]    Median:        0.02 ms
[ INFO ]    Average:       0.02 ms
[ INFO ]    Min:           0.01 ms
[ INFO ]    Max:           1.17 ms
[ INFO ] Throughput:   48353.55 FPS
```

### student_int8.xml
```
[ INFO ] Count:            184845 iterations
[ INFO ] Duration:         5000.02 ms
[ INFO ]    Median:        0.02 ms
[ INFO ]    Average:       0.02 ms
[ INFO ]    Min:           0.01 ms
[ INFO ]    Max:           0.50 ms
[ INFO ] Throughput:   36968.87 FPS
```

---

## End-to-end board latency (`intel/bench_e2e.py --n 84`)

84 scenarios from the phase-4 held-out split (`evaluate._held_out_rows`),
3 repeats. This is the number a user feels: a whole board through
`rank_vocabulary`, not an isolated matmul.

### Apple M4 — quiet machine, absolutes usable

Source: `artifacts/e2e_mac.json`.

| backend | embed | head | p50 (ms) | p90 (ms) | Jaccard vs torch |
|---|---|---|---|---|---|
| torch | torch | StudentRanker | 9.70 | 26.76 | 1.000 |
| ov-head | torch | OpenVinoStudent (INT8) | 9.65 | 13.58 | 0.913 |
| ov-full | **openvino** | OpenVinoStudent (INT8) | **5.51** | 9.12 | 0.913 |

### Core Ultra 9 185H — loaded machine, **ratios only**

Sources: `artifacts/e2e_tanishlaptop.json` (CPU), `_gpu.json`, `_npu.json`.

> **The absolute latencies in this table are withdrawn.** The 2026-08-25 run
> overlapped a 6 GB model download and a ~28-minute teacher slice on the same
> box. The identical torch configuration measured 40.06, 55.74 and 129.22 ms
> across the three runs, against 18.37 ms on an earlier quiet run, with p90s up
> to 208 ms. Do not quote any absolute number from this table. The
> **within-run ratios** are still meaningful, because the backends run
> back-to-back in one process on the same inputs.

| device | backend | embed | head | p50 (ms) | p90 (ms) | Jaccard vs torch |
|---|---|---|---|---|---|---|
| CPU | torch | torch | StudentRanker | ~~55.74~~ | ~~146.06~~ | 1.000 |
| CPU | ov-head | torch | OpenVinoStudent (**FP32**) | ~~91.34~~ | ~~156.09~~ | **1.000** |
| CPU | ov-full | **openvino** | OpenVinoStudent (**FP32**) | ~~27.62~~ | ~~63.83~~ | **1.000** |
| GPU | torch | torch | StudentRanker | ~~40.06~~ | ~~146.83~~ | 1.000 |
| GPU | ov-head | torch | OpenVinoStudent (INT8) | ~~47.78~~ | ~~121.51~~ | 0.920 |
| GPU | ov-full | **openvino** | OpenVinoStudent (INT8) | ~~21.71~~ | ~~48.27~~ | 0.920 |
| NPU | torch | torch | StudentRanker | ~~129.22~~ | ~~208.30~~ | 1.000 |
| NPU | ov-head | — | — | **compile error** | — | — |
| NPU | ov-full | — | — | **compile error** | — | — |

**The CPU rows serve the FP32 IR, not INT8.** The FP32 ablation ran after the
INT8 baseline and wrote to the same filename, overwriting it — so there is no
INT8-on-CPU end-to-end row for this machine. Fixed in
`docs/INTEL_LAPTOP_RUNBOOK.md` Block 4 for any future run.

### MiniLM through OpenVINO is the real win — now confirmed on Intel

Moving the sentence embedder from torch to OpenVINO (`ov-head` → `ov-full`) is
the largest OpenVINO effect anywhere in this project, and it reproduces on both
machines and on two independent Intel runs:

| machine / device | torch → ov-full | speedup |
|---|---|---|
| Apple M4 | 9.70 → 5.51 ms | **1.76×** |
| Core Ultra, CPU | ratio only | **2.02×** |
| Core Ultra, GPU | ratio only | **1.85×** |

MiniLM is a 22M-parameter transformer on the critical path; the student head is
~0.1 ms of the board. This is where the latency lives and where quantization and
the OpenVINO runtime actually pay. **Quote this as "~1.9× on Intel, 1.76× on the
dev Mac", never with absolute Intel milliseconds.**

Note the earlier Intel run reported `embed_backend_used: "torch"` for `ov-full`
and was therefore a duplicate of `ov-head`, not a third measurement. The cause
was a missing `optimum-intel` dependency, so the MiniLM export failed silently
and `embed.py` fell back. `run_on_intel.ps1` now installs it, hard-fails if the
IR is not built, and probes that the embedder actually resolves to OpenVINO
before benchmarking.

### The NPU cannot compile this IR (documented negative result)

`Intel(R) AI Boost` is present and enumerated, but both OpenVINO backends fail
to compile against it:

```
Exception from src\plugins\intel_npu\src\compiler_adapter\src\ze_graph_ext_wrappers.cpp:433:
Compilation failed. Level0 pfnCreate2 result: ZE_RESULT_ERROR_INVALID_NULL_POINTER
[NPU_VCL] Compiler returned msg: Missing upper bound for one or more nodes.
```

The NPU compiler requires **statically shaped** graphs. Our student IR is
exported with a dynamic batch axis (`?, 769`) so one artifact serves both
batch-1 runtime inference and batched evaluation. That design choice is what
blocks NPU compilation, and it is fixable — an additional batch-1 static IR
would very likely compile — but it is not done and must not be claimed.

**Honest statement:** the student runs on Intel CPU and on the Arc iGPU through
the same unmodified IR; the NPU is enumerated but rejects the dynamic-shape
graph.

## Whisper — Apple M4

`intel/bench_whisper.py` n=10 | committed clips `intel/clips/` | quiet machine | OpenVINO 2026.2.1-21919 | base.en INT8 both

| clip | audio (s) | faster-whisper p50 (ms) | p90 | openvino p50 (ms) | p90 | same text |
|---|---|---|---|---|---|---|
| bench_clip0.wav | 2.1 | 252 | 266 | 300 | 311 | yes |
| bench_clip1.wav | 4.0 | 407 | 509 | 403 | 548 | yes |

The Core Ultra Whisper run (`artifacts/whisper_bench_tanishlaptop.json`) is
**withdrawn**: it measured `faster_whisper` at 1280 ms p50 on the 2.1 s clip
against 252 ms on the Mac, i.e. it was taken on the same loaded machine as the
e2e run above.

## Teacher on Intel (`intel/teacher_slice.py`, n=10)

Source: `artifacts/teacher_slice_tanishlaptop.jsonl`. Median **170.93 s** per
board (min 73.06, max 456.10), CPU-only `transformers`, loaded machine.

This exists to show the teacher *can* run on Intel, and for nothing else. **Do
not build a speedup ratio from it** — pairing it with the student would produce
a ~6000× figure that compares an optimised student against an unoptimised
CPU-only teacher on a loaded box. The speedup ratio stays Mac-only
(`docs/VERIFIED_NUMBERS.md` §0b).

## Historical note (merge of the cloud line, 2026-08-23)

The cloud branch measured whisper INT8 via the optimum-intel runtime at 0.205 s
vs faster-whisper 0.324 s on one 6.25 s clip, and documented that
openvino_genai.WhisperPipeline could not serve its export ("Port for tensor
name cache_position was not found", genai <= 2025.3 with transformers <= 4.55
exports). That gap is version-bound and is gone: the export from
transformers 5.12 / optimum-intel 2.0 serves through genai 2026.2.1 (the tables
above). The transcriber still tries genai first, proves it can generate, and
falls back to optimum-intel, so either runtime works.

## Superseded

- An earlier Apple M4 end-to-end table (torch 16.8 / ov-head 15.7 / ov-full
  15.3) from `55306d8` was replaced by the re-measurement in `e27953e`. Both
  copies were left in this file by mistake; the stale one has been removed.
- The 2026-08-24 Core Ultra run reported an FP32 head at 0.062 ms / 631 KB. That
  was measured against a locally re-exported **59-class** IR (the repo's
  `student.onnx` was the pre-training export-first artifact). Both are fixed;
  `convert.py` now refuses to write an IR whose width disagrees with
  `vocab.json`.
- That same run's board Jaccard of 0.9155 was an artifact of the corrupted
  local state. On correct artifacts the OpenVINO head reproduces the torch board
  **exactly** on Intel CPU (1.000) and at 0.920 on the iGPU.
