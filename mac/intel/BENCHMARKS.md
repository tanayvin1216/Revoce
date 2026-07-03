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
