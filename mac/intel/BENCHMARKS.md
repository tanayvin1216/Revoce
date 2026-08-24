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
inputs, same artifacts.

## Student head, isolated (`intel/bench.py`)

batch=1, input `(1, 769)` float32 | 500 timed iterations after 50 warmup |
size on disk = `.xml` + `.bin`

| backend | M4 p50 (ms) | M4 p90 | M4 size (KB) | Core Ultra p50 (ms) | Core Ultra p90 | Core Ultra size (KB) |
|---|---|---|---|---|---|---|
| torch CPU (`student.pt`) | 0.015 | 0.016 | 657 | 0.133 | 0.179 | 658 |
| OpenVINO FP32 IR | 0.025 | 0.028 | 661 | *(re-run pending — see note)* | | |
| OpenVINO INT8 IR | 0.031 | 0.035 | 235 | 0.073 | 0.099 | 236 |

**The OpenVINO plugin inverts between the two CPUs.** Against torch on the same
machine, OpenVINO INT8 is **2.1× slower on ARM** (0.031 vs 0.015) and
**1.8× faster on the Core Ultra** (0.073 vs 0.133). Both rows run the same
95-class head, so this ratio is a like-for-like comparison on each machine. It
is the clearest evidence in this file that the OpenVINO CPU plugin is doing
Intel-specific work, and it is why the ARM figures are a dev-stage proxy rather
than a result.

> **Note — Core Ultra FP32 row withdrawn.** The 2026-08-24 laptop run reported
> FP32 at 0.062 ms / 631 KB, but that measurement was taken against a
> locally re-exported `student_fp32.xml` with a **59-class** output head, not
> the committed 95-class artifact (its `.vocab.sha` sidecar was never
> regenerated, so the loader guard would have rejected it for serving). The
> number is not comparable to anything else in this table and has been removed.
> Re-run `intel/bench.py` on the Core Ultra against the committed IR to fill it
> in. The torch and INT8 rows are unaffected — both used the committed 95-class
> models.

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

Not yet re-run on the Core Ultra.

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

## End-to-end board latency (`intel/bench_e2e.py --n 84`)

84 scenarios from the phase-4 held-out split (`evaluate._held_out_rows`),
3 repeats. This is the number a user feels: a whole board through
`rank_vocabulary`, not an isolated matmul. Both runs serve the INT8 IR
(`_load_student_openvino` → `student_ir_path()`).

Sources: `artifacts/e2e_mac.json`, `artifacts/e2e_tanishlaptop.json`.

| backend | embed | head | Apple M4 p50 / p90 (ms) | Core Ultra p50 / p90 (ms) | board Jaccard vs torch |
|---|---|---|---|---|---|
| torch | torch | StudentRanker | 9.7 / 26.8 | 18.4 / 28.9 | 1.000 |
| ov-head | torch | OpenVinoStudent | 9.7 / 13.6 | 17.9 / 25.1 | 0.913 / 0.915 |
| ov-full | openvino | OpenVinoStudent | **5.5** / 9.1 | 18.4 / 25.2 ⚠ | 0.913 / 0.915 |

**⚠ The Core Ultra `ov-full` row did not run OpenVINO embeddings.** Its
`embed_backend_used` is `"torch"`: `mac/intel/artifacts/*` is gitignored, so the
MiniLM IR (`artifacts/minilm-ov/`) was never on the laptop, and `embed.py` did
what it is designed to do — warned once and fell back. That row is therefore a
duplicate of `ov-head`, not a third measurement, and the Core Ultra has **not**
yet been measured end-to-end through OpenVINO.

This matters, because on the M4 that step is the single largest win in the file:
9.7 ms → **5.5 ms, a 1.76× speedup**, entirely from moving MiniLM onto
OpenVINO. MiniLM is a 22M-parameter transformer on the critical path and is
where the board latency actually goes; the student head is ~0.1 ms of it. Build
the MiniLM IR on the Core Ultra and re-run before quoting any Intel
board-latency figure.

**Board Jaccard 0.913–0.915** means the OpenVINO student picks a different board
than the torch student on roughly every other scenario. On a 10-slot board with
4 pinned core words, all of that divergence sits in the 6 model-chosen slots.
Whether it is quantization or a bug in the OpenVINO path is untested — the FP32
ablation that would settle it is still to run.

## Whisper — Apple M4

`intel/bench_whisper.py` n=10 | committed clips `intel/clips/` | quiet machine | OpenVINO 2026.2.1-21919-ede283a88e3-releases/2026/2 | base.en INT8 both

| clip | audio (s) | faster-whisper p50 (ms) | p90 | openvino p50 (ms) | p90 | same text |
|---|---|---|---|---|---|---|
| bench_clip0.wav | 2.1 | 252 | 266 | 300 | 311 | yes |
| bench_clip1.wav | 4.0 | 407 | 509 | 403 | 548 | yes |

Not yet run on the Core Ultra.

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

An earlier Apple M4 end-to-end table (torch 16.8 / ov-head 15.7 / ov-full 15.3)
from `55306d8` was replaced by the re-measurement in `e27953e` against the
merged artifacts. Both copies were left in this file by mistake; the stale one
has been removed. Current M4 numbers are the ones above, and they match
`artifacts/e2e_mac.json`.
