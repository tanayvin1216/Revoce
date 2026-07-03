# Student ranker benchmarks — torch CPU vs OpenVINO FP32 vs OpenVINO INT8

Machine: arm64 / Darwin 25.5.0 — Apple Silicon via OpenVINO ARM CPU plugin (dev stage); Intel-hardware numbers pending Tiber access
OpenVINO 2026.2.1-21919-ede283a88e3-releases/2026/2 | batch=1, input (1, 769) float32 | 500 timed iterations after 50 warmup

| backend | latency p50 (ms) | latency p90 (ms) | size on disk (KB) |
|---|---|---|---|
| torch CPU (student.pt) | 0.015 | 0.015 | 674 |
| OpenVINO FP32 IR | 0.029 | 0.050 | 678 |
| OpenVINO INT8 IR | 0.032 | 0.044 | 251 |

## INT8 accuracy vs FP32 (real calibration scenarios)

```json
{
  "n_calibration_rows": 360,
  "k_dyn": 6,
  "max_abs_logit_delta": 1.0880393981933594,
  "mean_abs_logit_delta": 0.20365050435066223,
  "mean_topk_overlap": 0.9541666666666667,
  "min_topk_overlap": 0.6666666666666666
}
```

## benchmark_app (latency hint, 5 s each)

### student_fp32.xml
```
[ INFO ] Count:            194077 iterations
[ INFO ] Duration:         5000.01 ms
[ INFO ]    Median:        0.02 ms
[ INFO ]    Average:       0.02 ms
[ INFO ]    Min:           0.01 ms
[ INFO ]    Max:           0.52 ms
[ INFO ] Throughput:   38815.35 FPS
```

### student_int8.xml
```
[ INFO ] Count:            147284 iterations
[ INFO ] Duration:         5000.04 ms
[ INFO ]    Median:        0.02 ms
[ INFO ]    Average:       0.03 ms
[ INFO ]    Min:           0.01 ms
[ INFO ]    Max:           2.07 ms
[ INFO ] Throughput:   29456.58 FPS
```
