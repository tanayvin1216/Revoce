# Student ranker benchmarks — torch CPU vs OpenVINO FP32 vs OpenVINO INT8

Machine: arm64 / Darwin 25.5.0 — Apple Silicon via OpenVINO ARM CPU plugin (dev stage); Intel-hardware numbers pending Tiber access
OpenVINO 2026.2.1-21919-ede283a88e3-releases/2026/2 | batch=1, input (1, 769) float32 | 500 timed iterations after 50 warmup

| backend | latency p50 (ms) | latency p90 (ms) | size on disk (KB) |
|---|---|---|---|
| torch CPU (student.pt) | 0.015 | 0.016 | 657 |
| OpenVINO FP32 IR | 0.025 | 0.028 | 661 |
| OpenVINO INT8 IR | 0.031 | 0.035 | 235 |

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

## benchmark_app (latency hint, 5 s each)

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

## End-to-end board latency — Apple M4

`intel/bench_e2e.py --n 84` | Darwin 25.5.0 | OpenVINO 2026.2.1-21919-ede283a88e3-releases/2026/2 | devices ['CPU'] | Intel hardware: no

| backend | embed | head | device | p50 (ms) | p90 (ms) | board Jaccard vs torch |
|---|---|---|---|---|---|---|
| torch | torch | StudentRanker | cpu | 16.8 | 56.6 | 1.000 |
| ov-head | torch | OpenVinoStudent | CPU | 15.7 | 19.9 | 0.914 |
| ov-full | openvino | OpenVinoStudent | CPU | 15.3 | 27.4 | 0.914 |

## Whisper — Apple M4

`intel/bench_whisper.py` n=10 | committed clips `intel/clips/` | quiet machine | OpenVINO 2026.2.1-21919-ede283a88e3-releases/2026/2 | base.en INT8 both

| clip | audio (s) | faster-whisper p50 (ms) | p90 | openvino p50 (ms) | p90 | same text |
|---|---|---|---|---|---|---|
| bench_clip0.wav | 2.1 | 252 | 266 | 300 | 311 | yes |
| bench_clip1.wav | 4.0 | 407 | 509 | 403 | 548 | yes |

## Historical note (merge of the cloud line, 2026-08-23)

The cloud branch measured whisper INT8 via the optimum-intel runtime at 0.205 s
vs faster-whisper 0.324 s on one 6.25 s clip, and documented that
openvino_genai.WhisperPipeline could not serve its export ("Port for tensor
name cache_position was not found", genai <= 2025.3 with transformers <= 4.55
exports). That gap is version-bound and is gone: the export from
transformers 5.12 / optimum-intel 2.0 serves through genai 2026.2.1 (the tables
above). The transcriber still tries genai first, proves it can generate, and
falls back to optimum-intel, so either runtime works.

## End-to-end board latency — Apple M4

`intel/bench_e2e.py --n 84` | Darwin 25.5.0 | OpenVINO 2026.2.1-21919-ede283a88e3-releases/2026/2 | devices ['CPU'] | Intel hardware: no

| backend | embed | head | device | p50 (ms) | p90 (ms) | board Jaccard vs torch |
|---|---|---|---|---|---|---|
| torch | torch | StudentRanker | cpu | 9.7 | 26.8 | 1.000 |
| ov-head | torch | OpenVinoStudent | CPU | 9.7 | 13.6 | 0.913 |
| ov-full | openvino | OpenVinoStudent | CPU | 5.5 | 9.1 | 0.913 |
