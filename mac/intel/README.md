# mac/intel — OpenVINO lifecycle for the student ranker

The Phase-4 student (ONNX, opset 17) converted to OpenVINO IR, quantized to INT8
with NNCF on REAL calibration embeddings, benchmarked, and served through the seam.

## Pipeline (run from `mac/`, `VOICELINK_PERCEPTION=0`)

```bash
python3 intel/convert.py    # student.onnx -> artifacts/student_fp32.xml (+ MiniLM IR)
python3 intel/quantize.py   # NNCF PTQ -> artifacts/student_int8.xml + accuracy note
python3 intel/bench.py      # -> BENCHMARKS.md (torch vs OV-FP32 vs OV-INT8)
```

## Serving through the seam

```bash
VOICELINK_STUDENT_BACKEND=openvino  # default: torch
```

`rank_vocabulary._load_student` compiles `student_int8.xml` (falls back to
`student_fp32.xml`) via the OpenVINO CPU plugin. Selection/assembly logic is
identical across backends — the flag swaps ONLY the logits producer, which is the
whole point of the seam.

## Dependencies

`requirements-intel.txt` (openvino, nncf, optimum[openvino]).

**Known conflict:** `optimum-intel 2.0.0` pins `transformers<5.1` while `mlx-vlm`
(the perception VLM runtime) needs `transformers>=5.5`. The pin only matters for
the EXPORT step (`convert.py`'s MiniLM export, the Whisper stretch export) — the
OpenVINO **runtime** never imports transformers. Workflow: downgrade transformers
for the one-off export, then restore (`pip install "transformers>=5.12"`); or run
exports in a scratch venv.

## Honesty labels

- All numbers in `BENCHMARKS.md` were measured on Apple Silicon through OpenVINO's
  **ARM CPU plugin** (the dev loop). They prove functional correctness and give a
  real FP32-vs-INT8 comparison, but are NOT Intel-hardware numbers — those are
  pending Tiber AI Cloud access.
- INT8 accuracy is reported against the actual eval scenarios
  (`v2_scenarios.jsonl` through `data.row_features`), not citations.
- Qwen2.5-VL export deliberately NOT attempted yet (7 GB download + version
  pitfalls, see research-intel-openvino.md) — documented next step.
