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

## Whisper base.en — faster-whisper vs OpenVINO INT8

Exported with `optimum-cli export openvino --weight-format int8` in a dedicated
export venv (see README.md). Measured on Apple M4 Pro via the OpenVINO **ARM CPU
plugin** — these are NOT Intel-hardware numbers. One 6.254 s clip of synthesized
speech, median of 5 runs after a warm-up, same machine, same process.

| backend | median | × realtime | on disk |
|---|---|---|---|
| **OpenVINO INT8** (optimum-intel runtime) | **0.205 s** | 30.5× | **81 MB** |
| faster-whisper base.en int8 | 0.324 s | 19.3× | 141 MB |

**1.58× faster and 1.74× smaller**, transcript identical apart from a trailing
period. Both are far inside the latency budget; the size difference is the more
meaningful one for an edge deployment.

n=1 clip. This is a smoke measurement, not a WER study — no accuracy claim is
made beyond "the transcript matched on this clip".

### Known gap: openvino-genai cannot serve this IR

`openvino_genai.WhisperPipeline` is the runtime we want (no transformers
dependency, and the GenAI API the 2026 rubric names). It currently fails on our
export with:

```
Port for tensor name cache_position was not found.
```

Tried and still failing: genai 2025.3 and 2025.2, against exports from
transformers 4.55.4 and 4.48.3 (optimum-intel 1.26.1). The exported decoder
exposes `beam_idx / encoder_hidden_states / input_ids`; genai expects a
`cache_position` port that this optimum-intel does not emit. No newer
openvino-genai exists than 2025.3.0.0.

So the serving path is optimum-intel for now. `OpenVinoWhisperTranscriber` tries
openvino-genai first, *proves* it can generate before accepting it, and falls
back to optimum-intel otherwise — so this fixes itself when the versions line up.

Worth being precise about the transformers pin this implies: optimum-intel wants
`transformers<5.1` and mlx-vlm wants `>=5.5`, so they cannot share one venv. That
is a **Mac dev-box** problem, not a deployment problem — the Intel target does not
have mlx on it at all, and there optimum-intel + OpenVINO is the entire stack.
