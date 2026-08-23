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

## Whisper (shipped)

```bash
python3 intel/export_whisper.py        # optimum-cli -> artifacts/whisper-base.en-ov-int8
python3 intel/bench_whisper.py --n 10  # faster-whisper vs WhisperPipeline, appends to BENCHMARKS.md
VOICELINK_TRANSCRIBER=openvino         # serve it (perception/config.py one-line swap)
```

- `intel/export_whisper.py` wraps `optimum-cli export openvino --model openai/whisper-base.en
  --task automatic-speech-recognition-with-past --weight-format int8`. That documented
  form works as-is with optimum-intel + transformers 5.12 (no downgrade needed); the
  script falls back to two alternative invocations if a future release renames the task,
  and records the winning command in `<dir>/export_invocation.json`.
- `perception.transcriber.OpenVinoWhisperTranscriber` runs the IR through
  `openvino_genai.WhisperPipeline` (device from `VOICELINK_OV_DEVICE`, default CPU, with
  `INFERENCE_PRECISION_HINT=f32` on CPU). `make_transcriber` selects it for
  `TRANSCRIBER="openvino"` and falls back to faster-whisper, then mock, if the package or
  IR is missing. base.en is English-only, so no language/task is set on the generation
  config (GenAI rejects it for non-multilingual models).
- **Confidence is a proxy, not calibrated.** GenAI exposes no logprobs / no-speech
  probability (`result.scores` is a flat 1.0 under greedy). The transcriber reports
  avg_logprob=-0.3 and no_speech_prob=0.05 when text is produced, -10/1.0 when empty.
  The addressed/ambient gate therefore cannot use confidence with this backend.
- Test: `tests/test_whisper_openvino.py` (skips without openvino_genai / the IR).

Measured (`bench_whisper.py`, Apple M4, ARM CPU plugin, N=10, identical transcripts):
faster-whisper int8 p50 **252 / 407 ms** vs WhisperPipeline INT8 p50 **300 / 403 ms** on the committed 2.1 / 4.0 s clips (`intel/clips/`, n=10, quiet M4): parity on the ARM plugin. An earlier run on a loaded machine (0.68 vs 1.16 s) is superseded.
clips. On this ARM box OpenVINO is ~1.5x SLOWER than CTranslate2 — the earlier one-off
"0.35 vs 0.42 s" smoke number did not reproduce. The point of this backend is the Intel
port (CPU/GPU/NPU via one env var), not ARM speed; Intel numbers pending.
**ARM pitfall (hit twice in this track):** the CPU plugin defaults to f16 inference — the
Whisper decoder degenerates into repeated `!` tokens and runs 8x slower. Always pass
`INFERENCE_PRECISION_HINT="f32"` on ARM (the transcriber does).

## Honesty labels

- All numbers in `BENCHMARKS.md` were measured on Apple Silicon through OpenVINO's
  **ARM CPU plugin** (the dev loop). They prove functional correctness and give a
  real FP32-vs-INT8 comparison, but are NOT Intel-hardware numbers — those are
  pending Tiber AI Cloud access.
- INT8 accuracy is reported against the actual eval scenarios
  (`v2_scenarios.jsonl` through `data.row_features`), not citations.
- Qwen2.5-VL export deliberately NOT attempted yet (7 GB download + version
  pitfalls, see research-intel-openvino.md) — documented next step.
