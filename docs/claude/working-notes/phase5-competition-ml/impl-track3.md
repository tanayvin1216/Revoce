# Track 3 — Intel/OpenVINO lifecycle — implementation log

Worktree: `.claude/worktrees/agent-a0fd2ca8f22f3ba3e`, branch `worktree-agent-a0fd2ca8f22f3ba3e`.

## Setup notes

- Worktree branch was stale (created at a4bed1a, phase-1 era, no `mac/evaluation/phase4/`).
  Fast-forwarded to 50f4f09 (phase4-distillation head) — clean `--ff-only`, no conflicts.
- Trained artifacts (`student.pt`, `student.onnx(+.data)`, `vocab.json`, `splits.json`)
  are gitignored, so they don't travel with the worktree checkout. Copied the current
  trained versions from the main checkout (`/Users/tanayvinaykya/VoiceLink/mac/evaluation/phase4/`,
  dated Jun 30 17:56) into the worktree. Per PLAN.md these are provisional — the
  orchestrator regenerates them post-merge and re-runs convert/quantize.
- No project venv found (`mac/.venv`, `.venv` absent). The repo runs on the framework
  Python: `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3` (3.13.6).
  Installing Intel packages there, same as every other project dep.

## Install

`python3 -m pip install openvino nncf "optimum[openvino]"` — succeeded, native arm64 wheels.

Exact versions installed (2026-07-02):

| package          | version    |
|------------------|------------|
| openvino         | 2026.2.1   |
| openvino-tokenizers | 2026.2.1.0 |
| nncf             | 3.2.0      |
| optimum          | 2.2.0      |
| optimum-intel    | 2.0.0      |
| transformers     | 5.0.0 (downgraded by pip from 5.12.0 — see conflict below) |
| torch            | 2.12.0 (unchanged) |
| onnx / onnxruntime | 1.22.0 / 1.26.0 (unchanged) |

**Dependency conflict (real, observed):** `optimum-intel 2.0.0` pins
`transformers<5.1,>=4.45`, while `mlx-vlm 0.6.3` / `mlx-audio 0.4.4` (the perception VLM
runtime) require `transformers>=5.5.0`. pip resolved by downgrading transformers to
5.0.0, leaving mlx-vlm formally broken. The two pins are irreconcilable in one env.
Plan: run the optimum-cli exports (the only step that needs optimum-intel) under
transformers 5.0.0, verify the OpenVINO runtime path (which does NOT import
transformers), then restore transformers 5.12.0 and re-verify perception imports.
Documented in mac/intel/README.md as an export-time-only dependency.

Sanity check under 5.0.0: `evaluation.embed.embed` and `data.row_features` both work
(sentence-transformers 5.5.1 tolerates it; one benign "UNEXPECTED weights" note).

- openvino-genai NOT installed yet — deferred to the whisper stretch step per plan.

## convert.py (step 2)

- `student.onnx -> artifacts/student_fp32.xml` via `ov.convert_model` + `ov.save_model`
  (compress_to_fp16=False).
- **Pitfall found (real, measured):** the ARM CPU plugin defaults to
  `INFERENCE_PRECISION_HINT=f16` — first parity check FAILED with max |logit delta|
  1.11e-02 vs onnxruntime. Compiling with an explicit `{"INFERENCE_PRECISION_HINT":
  "f32"}` gives 1.43e-06. `intel.OpenVinoStudent` pins f32 (comment in code).
- MiniLM export: `python3 -m optimum.commands.optimum_cli export openvino
  --model sentence-transformers/all-MiniLM-L6-v2 --library sentence_transformers
  --task feature-extraction` -> `artifacts/minilm-ov/` (openvino_model.xml + bin +
  tokenizer IRs). Succeeded under transformers 5.0.0.

Final convert.py output:
```
student FP32 IR -> .../mac/intel/artifacts/student_fp32.xml
onnxruntime vs OV-FP32 max abs logit diff = 1.43e-06
minilm IR -> .../mac/intel/artifacts/minilm-ov
```

## quantize.py (step 3)

Calibration = all 360 rows of v2_scenarios.jsonl through `data.row_features` (the
exact 769-d train/inference featurizer), wrapped in `nncf.Dataset`.

- **First attempt (default nncf.quantize): FAILED accuracy** — mean top-6 overlap
  0.728, max |logit delta| 17.7. Swept variants (scratchpad, all measured on the
  360-row set):
  | variant | max |d| | mean |d| | top-6 overlap |
  |---|---|---|---|
  | default (subset 360) | 17.74 | 10.07 | 0.728 |
  | model_type=TRANSFORMER | 21.09 | 14.30 | 0.513 |
  | preset=MIXED | 23.24 | 15.57 | 0.582 |
  | fast_bias_correction=False | 17.74 | 10.07 | 0.729 |
  | ignore FIRST MatMul | 17.73 | 10.09 | 0.729 |
  | **ignore LAST MatMul (vocab head)** | **1.09** | **0.20** | **0.954** |
  | weights-only int8 (compress_weights) | 0.03 | 0.005 | 0.997 |
- Root cause: quantizing the final 192->194 vocab projection destroys the ranking.
  Fix: standard last-layer exclusion via `nncf.IgnoredScope` — the head MatMul is
  found dynamically by walking back from the Result node (survives re-exports).
  Everything upstream (input, LayerNorm, hidden Linear+ReLU) stays INT8.

Final quantize.py output:
```
calibration set: (360, 769) from v2_scenarios.jsonl
student INT8 IR -> .../mac/intel/artifacts/student_int8.xml
{
  "n_calibration_rows": 360,
  "k_dyn": 6,
  "max_abs_logit_delta": 1.0880393981933594,
  "mean_abs_logit_delta": 0.20365050435066223,
  "mean_topk_overlap": 0.9541666666666667,
  "min_topk_overlap": 0.6666666666666666
}
>>> INT8 accuracy: PASS (mean top-6 overlap 0.954) <<<
```

## bench.py (step 4)

- `benchmark_app` IS shipped by the pip wheel (drift-check item resolved) but fails
  on the IR's dynamic batch axis — needed `-data_shape context_emb[1,769]`.
- Honest result: this MLP is so small that torch CPU is actually the fastest on ARM
  (0.015 ms p50); OV adds ~15 µs of API overhead per call and INT8 is NOT faster on
  the ARM plugin. The INT8 win here is 2.7x smaller disk/memory (678 -> 251 KB);
  latency wins are the Intel-hardware story (pending Tiber). All backends are far
  below the ~8 ms budget either way. Table in mac/intel/BENCHMARKS.md.

## Seam wiring (step 6) + tests (step 7)

- `ranker_config.STUDENT_BACKEND` (`VOICELINK_STUDENT_BACKEND=torch|openvino`,
  default torch).
- `rank_vocabulary._load_student` split into `_load_student_torch` /
  `_load_student_openvino` + a 10-line dispatcher; the OV branch compiles
  `student_int8.xml` (fallback fp32 via `intel.student_ir_path`) and is torch-free
  (core from GUARANTEED_CORE, k_dyn from config.yaml). `_student_board`,
  selection/assembly, and everything else in rank_vocabulary UNTOUCHED.
- TDD: wrote mac/tests/test_intel.py first — backend-flag tests RED
  (`AttributeError: STUDENT_BACKEND`), parity/INT8 tests green; then wired the seam.

Test results (real output):
```
tests/test_intel.py — 4 passed, 2 warnings in 5.26s
tests/ (full suite) — 59 passed, 5 warnings in 6.44s
```

Acceptance demo (fresh process, env-flag only):
```
$ VOICELINK_PERCEPTION=0 VOICELINK_STUDENT_BACKEND=openvino python3 -c "..."
backend object: OpenVinoStudent | IR: student_int8.xml
board: ['yes', 'no', 'I need help', 'wait', "i don't know", 'i am not sure',
        "i'm not sure", 'no thanks', "i'm not sure.", 'not now']
```
(The hedge-heavy dynamic words are the known RC-2 problem Track 1 is fixing — the
OpenVINO backend faithfully reproduces the current student, which is the point.)

