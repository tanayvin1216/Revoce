# VoiceLink — Verified Numbers

**Built 2026-08-23** per `docs/SUBMISSION_SPRINT.md` §9. One table of every number this
project quotes, each with the file that holds it, the command that (re)produced it, and a
status. Every downstream document (papers, the 150-word description, the video script,
the competition entry) should cite this file and nothing else.

Interpreter used for all re-runs: `/usr/local/bin/python3` (3.13.6; has torch 2.12.0,
sentence-transformers, OpenVINO 2026.2.1, pytest). All commands run from `mac/` with
`VOICELINK_PERCEPTION=0`. Scratch helper scripts live in the session scratchpad, not the
repo; what each one does is described inline so it can be reproduced in one line.

**Status legend**
- **verified-today** — recomputed 2026-08-23 with no teacher/VLM/LLM call; value matches the file.
- **verified-today (drift)** — recomputed today and the value *differs* from the committed file.
- **file-only** — read from the committed result file / note; re-running needs a model
  generation run (teacher LLM, VLM, Whisper) and was deliberately not launched.
- **superseded** — a number that was true for an earlier run and is now replaced.
- **UNSOURCED** — no file in the repo backs it.

All measurements are on Apple ARM silicon (M-series). **Zero Intel-hardware numbers
exist** as of today.


---

## 0a. Post-merge re-measurement (2026-08-23, evening) — CURRENT values

The cloud branch's regenerated corpus and retrained v2 student were merged (same
context fingerprint, so the held-out split is unchanged), and everything was
re-run against the merged artifacts (`config_hash d6ce2e56d50b`). **Quote these,
not §0/§1, where they differ.**

| Number | Value (current) | Source | Run date |
|---|---|---|---|
| Board, fully OpenVINO (MiniLM IR + INT8 head), 84 held-out | **5.5 ms p50 / 9.1 p90**; ov-head 9.7/13.6; torch 9.7/26.8; board Jaccard vs torch 0.913 | `mac/intel/artifacts/e2e_mac.json` | 2026-08-23 |
| Student board (torch, harness) vs teacher, same run | **10.1 ms vs 7.05 s = 695×** (teacher timing drifts run to run; quote ">400×") | `phase4_results.json` `speedup` | 2026-08-23 |
| Distinct-functions @7, student vs teacher | **4.095 vs 4.298**, retention **0.953** (PASS) | same | 2026-08-23 |
| Same-function-pair rate | **0.1893 vs 0.1735 — still FAIL** (bar: ≤ teacher) | same | 2026-08-23 |
| Coverage-of-intent, dynamic | **0.789 vs 0.873 = 0.904× (PASS ≥0.90×)** | same | 2026-08-23 |
| dynamic_overlap / full top-7 | **0.067 / 0.594** (report-only) | same | 2026-08-23 |
| Deictic set (n=60) | fire 0.883, answer-on-board **0.817 vs 0.017** closed-only | same, `deictic` | 2026-08-23 |
| Full suite | **213 passed** | `pytest` | 2026-08-23 |

The v3 bi-encoder results (cloud line) live in
`mac/evaluation/phase4/phase4_v3_results.json` and `V3_BIENCODER_REPORT.md`;
v3 is NOT the served default and its numbers are not interchangeable with v2's.

## 0b. Intel hardware (2026-08-24) — Core Ultra 9 185H

Source: `mac/intel/artifacts/e2e_tanishlaptop.json`, `mac/intel/BENCHMARKS.md`.
Machine: Intel(R) Core(TM) Ultra 9 185H / Windows 11 / Python 3.14.2 /
OpenVINO 2026.3.0. Devices seen: CPU, GPU (Arc iGPU), NPU (AI Boost).

| Number | Value | Status |
|---|---|---|
| Board latency, torch path, 84 held-out | **18.4 ms p50 / 28.9 p90** | verified |
| Board latency, ov-head (INT8 head, torch embed) | **17.9 / 25.1** | verified |
| Board latency, ov-full | **18.4 / 25.2** — but `embed_backend_used` is `"torch"` | **NOT a valid ov-full measurement**; the MiniLM IR was absent so it silently duplicated ov-head |
| Board Jaccard vs torch (OpenVINO student) | **0.915** | verified; divergence untested against FP32 |
| Student head, torch, isolated | **0.133 ms p50** | verified |
| Student head, OpenVINO INT8, isolated | **0.073 ms p50** | verified |
| Student head, OpenVINO FP32, isolated | *withdrawn* | measured against a 59-class IR; re-run needed |
| Speedup vs teacher on Intel | **not measured** | the teacher has never run on this machine; do NOT pair 18.4 ms with the Mac's 7.05 s teacher |
| GPU / NPU board latency | **not measured** | and note `OpenVinoEmbedder` pins MiniLM to CPU, so `VOICELINK_OV_DEVICE` moves only the ~0.1 ms head |

**Cross-machine claims.** The Mac and the Core Ultra differ in OS, Python and
OpenVINO version, so absolute latencies are not a controlled comparison. The
Core Ultra is *slower* end-to-end than the M4 (18.4 vs 9.7 ms torch), and the
claim "2.6× faster on Intel" that appears in `docs/INTEL_REMAINING.md` is **not
supported by any committed file** — it pairs a superseded 47.1 ms Mac figure
with the Intel number. Do not use it.

What IS controlled is the torch-vs-OpenVINO ratio within each machine, and it
inverts: OpenVINO INT8 is **2.1× slower than torch on ARM** (0.031 vs 0.015 ms)
and **1.8× faster on the Core Ultra** (0.073 vs 0.133 ms), same 95-class head
both sides. That is the defensible Intel performance claim.

## 0. Re-measured later on 2026-08-23 (supersedes rows below where noted)

After this inventory was built, the embedding memoisation landed (commit 55306d8) and
`evaluate.py --teacher-samples N` was re-run three times on the same M4. Gates are
unchanged to four decimals; only the latency rows moved.

| Number | Value (current) | Source file | Command | Run date | Status |
|---|---|---|---|---|---|
| Student end-to-end board latency (embed + MLP + MMR), torch path | **10.135 ms** (IQR 6.62–11.43, n=50). Supersedes the 12.3–14.8 ms and 47.081 ms rows, both of which this file previously quoted against a file that already said otherwise. | `mac/evaluation/phase4/phase4_results.json` `speedup` | `python3 evaluation/phase4/evaluate.py --teacher-samples 12` (quiet machine) | 2026-08-23 | verified-today |
| Student board, fully OpenVINO path (MiniLM IR + INT8 head), 84 held-out, context evicted per round | **5.51 ms p50 / 9.12 p90**; ov-head 9.65/13.58; torch 9.70/26.76; board Jaccard vs torch 0.913. (The 15.3/15.7/16.8 figures previously on this row were the superseded 55306d8 run, not the committed artifact.) | `mac/intel/artifacts/e2e_mac.json`, `mac/intel/BENCHMARKS.md` | `python3 intel/bench_e2e.py --n 84` | 2026-08-23 | verified-today |
| Teacher pipeline latency | **7.05 s (IQR 6.96–7.46, n=8)** as committed. Run-to-run drift is real (10.64 s and 11.68 s were observed on loaded machines, 15.7 s while the test suite ran); quote the committed figure and note the machine was quiet. | same | same | 2026-08-23 | verified-today (run-to-run drift is real; quote a range) |
| Speedup, student vs teacher | **695.5×** (10.135 ms vs 7.05 s), as committed in the cited file. Quote as "**~695×**" or conservatively "**>400×**". Never 841–864× (this file's own prior mis-transcription), never 142.6× (superseded), never 900×/7 ms (bare MLP). | same, `speedup_x` | same | 2026-08-23 | verified-today |
| Whisper base.en: OpenVINO GenAI INT8 vs faster-whisper int8, ARM plugin | **300 / 403 ms vs 252 / 407 ms p50** on the committed 2.1 s / 4.0 s clips (`mac/intel/clips/`), n=10, identical transcripts, quiet machine. An earlier same-day run (1162/1157 vs 681/795) overlapped three running agents and is discarded; the 0.35 vs 0.42 s row below is WITHDRAWN. | `mac/intel/artifacts/whisper_bench_mac.json` | `python3 intel/bench_whisper.py` | 2026-08-23 | verified-today |
| MiniLM over OpenVINO vs torch | cosine **1.000** on all probes; **5.6 ms vs 16.0 ms** per single text | `mac/tests/test_embed_backend.py` | `pytest tests/test_embed_backend.py` | 2026-08-23 | verified-today |
| Deictic question set (n=60, authored): open-answer fire rate / answer-on-board student / closed-only | **0.883 / 0.817 / 0.050**; "how many" 0/6 by design | `mac/evaluation/phase4/deictic_results.json` | `python3 evaluation/phase4/eval_deictic.py` | 2026-08-23 | verified-today (authored set, not independent) |
| Full test suite | **213 passed** | `mac/tests/` | `VOICELINK_PERCEPTION=0 python3 -m pytest -q` | 2026-08-23 | verified-today |
---

## 1. The table

### A. Phase 3 — diverse (MMR) selection vs naive top-k, 120 scenarios

| Number | Value (current) | Source file | Command | Run date | Status |
|---|---|---|---|---|---|
| Scenarios | 120 (5 categories × 24) | `mac/evaluation/phase3/selftest_results.json` (`n_scenarios`), `scenarios.jsonl` (120 lines) | `wc -l evaluation/phase3/scenarios.jsonl` | 2026-08-23 | verified-today |
| Distinct communicative functions @7, diverse vs top-k | **4.683 vs 3.875** (gap 0.808) | same, `overall.*.distinct_functions` | `python3 evaluation/phase3/selftest.py --from-boards` (re-scores cached `boards.jsonl`, no LLM) | 2026-08-23 | verified-today |
| Near-duplicate pair rate, diverse vs top-k | **0.0004 vs 0.0274** | same, `overall.*.near_dup` | same | 2026-08-23 | verified-today |
| Near-dup ratio | **69× lower** (0.02738 / 0.000397 = 69.0; "68×" comes from dividing the rounded 0.0274 / 0.0004 = 68.5) | derived | — | 2026-08-23 | verified-today (rounding note) |
| Coverage-of-intent @7, diverse vs top-k | **1.000 vs 0.775** on today's re-score. Committed file (dated 2026-06-30 04:26) says **1.000 vs 0.808**. The metric code was fixed on 2026-07-02/03 (word-boundary matching, hedge module — commits 146fee3, 3032854) *after* the file was written; the cached boards are unchanged, the scorer is not. | same, `overall.*.coverage_of_intent` | same | 2026-08-23 | **verified-today (drift)** — quote **1.00 vs 0.78** or re-commit the file |
| Exact-match to oracle word (diagnostic), diverse vs top-k | **0.525 vs 0.333** today; committed file says 0.567 vs 0.342 (same cause as above) | same, `overall.*.exact_match` | same | 2026-08-23 | **verified-today (drift)** |
| Phase-3 gate | PASS (scenarios ≥100; distinct-fn diverse ≥4.5, topk ≤4.0, gap ≥0.75; near-dup diverse ≤0.05 and ≥3× lower; judge proxy-labeled). Coverage is *not* gated. | same, `gates` | same | 2026-08-23 | verified-today |

The re-score output was captured to the scratchpad and `selftest_results.json` + the four PNGs were restored with `git checkout -- mac/evaluation/phase3/` so no committed result file changed in this task.

### B. Phase 4 — distilled student vs teacher (held-out split)

| Number | Value (current) | Source file | Command | Run date | Status |
|---|---|---|---|---|---|
| Teacher corpus rows | **600** (all boards exactly 10 words: 4 core + 6 dynamic) | `mac/evaluation/phase4/teacher_data.jsonl` | `wc -l` | 2026-08-23 | verified-today; file dated 2026-06-30 (teacher generation not re-run) |
| Student training rows after dedup / category cap | **559** | `phase4_results.json` `meta.n_rows`; `student.pt` | re-run of held-out eval (below) | 2026-08-23 | verified-today |
| Split sizes | train 391 / val 84 / **test 84**, fingerprint `ab51f7fba4c9c39e` | `mac/evaluation/phase4/splits.json` | `python3 -c "import json;print(json.load(open('evaluation/phase4/splits.json')))"` | 2026-08-23 | verified-today |
| Dynamic vocabulary size | **94** words (min_freq 3, max 200, dedup τ 0.80 → 22 canonical merges) | `vocab.json` (94 entries), `canonical.json` (22), `config.yaml` | `python3 -c "import json;print(len(json.load(open('evaluation/phase4/vocab.json'))))"` | 2026-08-23 | verified-today |
| Student input / architecture | 769-d (384 scene + 384 transcript + 1 is_question) → LayerNorm → Linear(769→192) → ReLU → Dropout 0.3 → Linear(192→94) | `config.yaml`, `student.py` | — | — | verified-today (read) |
| Board shape | k = 10 shown (4 pinned core + k_dyn 6); metrics judge top-7 | `config.yaml` | — | — | verified-today (read) |
| Config hash | `8151a7e34afd` | `phase4_results.json` | — | 2026-08-17 | verified-today |
| Distinct-functions @7, student vs teacher (held-out n=84) | **4.083 vs 4.333**, retention **0.942** (PASS ≥0.90) | `phase4_results.json` `diversity_retention` | `python3 evaluation/phase4/evaluate.py --teacher-samples 0` with `HERE`/`REPORT` redirected to scratch (helper `rerun_heldout.py`) — no LLM; MiniLM embeds only | 2026-08-23 | verified-today |
| Near-duplicate rate, student vs teacher | **0.0006 vs 0.0** (PASS ≤0.08) | same | same | 2026-08-23 | verified-today |
| Same-function-pair rate, student vs teacher | **0.1865 vs 0.1684** — **FAIL** (bar: ≤ teacher) | same | same | 2026-08-23 | verified-today — **stays reported as failing** |
| Board variety / dynamic variety | 77 distinct boards of 84 → **0.917** (PASS ≥0.5) | same | same | 2026-08-23 | verified-today |
| Combined diversity gate | **FAIL** (because sfp fails) | same, `diversity_retention.pass=false` | same | 2026-08-23 | verified-today |
| Teacher agreement — dynamic_overlap (report-only) | **0.056** (= "5.6%") | same, `agreement` | same | 2026-08-23 | verified-today |
| Teacher agreement — full top-7 overlap | **0.59** (inflated by the 4 pinned core words) | same | same | 2026-08-23 | verified-today |
| Coverage-of-intent, full board, student vs teacher (n=71) | **0.845 vs 0.944** (reported, not gated) | same, `coverage_of_intent` | same (requires `v2_scenarios.jsonl` next to the script — without it only 20 rows score) | 2026-08-23 | verified-today |
| Coverage-of-intent, DYNAMIC (the gated bar), student vs teacher (n=71) | **0.761 vs 0.845 = 0.901× (PASS ≥0.90×)** | same | same | 2026-08-23 | verified-today |
| Student end-to-end board latency (embed + MLP + MMR assemble) | File: **47.081 ms** (IQR 45.06–49.29, 50 samples). Today: 59.5 ms (IQR 55.9–62.4) and 69.3 ms on a second run — machine was also running another agent's benchmarks; treat the Aug-17 value as the quiet-machine figure and re-time before quoting a latency ≠ 47 ms | `phase4_results.json` `speedup` | same | file 2026-08-17; re-timed 2026-08-23 | verified-today (drift, load-dependent) |
| Teacher pipeline latency | **6.71 s** (IQR 6.63–6.97, 8 samples) | same | `evaluate.py --teacher-samples 8` — **not re-run** (LLM generation) | 2026-08-17 | file-only |
| Speedup, student vs teacher | **142.6×** (6.71 s / 47.081 ms) | same, `speedup_x` | not re-run (needs teacher timing) | 2026-08-17 | file-only |
| Open-answer path fires on | **2 of 84** held-out rows; **1 of 71** coverage-scored rows | `docs/claude/working-notes/rc3-open-answers/NOTES.md` §6 | loop `answers.answer_candidates(scene, transcript)` over `evaluate._held_out_rows()` | 2026-08-23 | verified-today |
| Hedge tokens in top-7 (Phase-5 student fix) | 2.107 → **0.012** (teacher 0.321); boards with ≥2 hedges 72.6% → 0% | `docs/claude/working-notes/phase5-competition-ml/FINAL_REVIEW.md` | — | 2026-07-03 | file-only |

### C. Closed-vocabulary ceiling (RC-3 root cause) — the AXIOM paper's numbers

All of these are pure counting over `teacher_data.jsonl` + `vocab.json` + `canonical.json`
(no model). Reproduced today by a 40-line scratch script `representability.py`; the logic
is: dynamic slots = board words not in `data.CORE_SET`; a slot is representable if the
lowercased word, or its `canonical.json` image, is in the 94-word vocab; filler = `evaluation.hedges.is_hedge`.
Vocab sweep = `data.build_vocab(rows, min_freq, max_size)` (frequency-built, no dedup).

| Number | Value (current) | Source file | Command | Run date | Status |
|---|---|---|---|---|---|
| Teacher dynamic word slots | **3600** (600 × 6) | `teacher_data.jsonl` | `representability.py` | 2026-08-23 | verified-today |
| Unique teacher dynamic words | **2362** today; NOTES.md / `answers.py` docstring say **2366** (a normalisation difference of 4 words; immaterial) | NOTES.md §1 | same | 2026-08-23 | verified-today (±4) |
| Unique words appearing < 3 times (the min_freq floor) | **2238 = 94.8%** of unique (NOTES.md says 94.5% — same ±4 denominator) | NOTES.md §1 | same | 2026-08-23 | verified-today |
| Teacher slots representable in the shipped 94-word vocab | **32.5%** | NOTES.md §1, `answers.py` docstring, `phase_status.json` phase 5 | same | 2026-08-23 | verified-today |
| Of representable slots, hedge/filler | **49.1%** | same | same | 2026-08-23 | verified-today |
| Rows with zero representable dynamic words | **73/600 = 12.2%** | same | same | 2026-08-23 | verified-today |
| Vocab sweep: min_freq=3, max 200 (shipped policy, pre-dedup) | size 124 → coverage **0.332** | NOTES.md §1 table | same | 2026-08-23 | verified-today |
| Vocab sweep: min_freq=2, max 400 | size 292 → **0.425** | same | same | 2026-08-23 | verified-today |
| Vocab sweep: min_freq=1, max 800 | size 800 → **0.566** (= "57%"; 800/94 = **8.5×** the shipped vocab) | NOTES.md §1 table; `answers.py` docstring ("growing the vocab to 800 words still only reaches 56.6%") | same | 2026-08-23 | verified-today — **this is the source of the "8× larger still only 57%" sentence** |
| Vocab sweep: exactly 8× (752 words, min_freq=1) | size 752 → **0.553** | — | `build_vocab(rows, 1, 752)` | 2026-08-23 | verified-today |
| Vocab sweep: 94 words, min_freq=1 (frequency-only, no dedup) | **0.307** | — | `build_vocab(rows, 1, 94)` | 2026-08-23 | verified-today |
| Vocab sweep: every word (2362) | **1.000** | — | `build_vocab(rows, 1, 10000)` | 2026-08-23 | verified-today |
| Achievable dynamic_overlap ceiling | 0.325; v2.1's 0.165 ≈ 51% of it | NOTES.md §1 | derived | — | verified-today (derived) |

Recommended phrasing: "a vocabulary 8.5× larger (800 words, every word the teacher ever
used at least once, frequency-ranked) still represents only 56.6% of teacher word choices."
"8×" → 55.3%, so "8× … 57%" slightly overstates; "8.5× … 57%" or "8× … 55%" are both exact.

### D. Phase 4 negative result (data-volume hypothesis)

| Number | Value (current) | Source file | Command | Run date | Status |
|---|---|---|---|---|---|
| dynamic_overlap v2 (authored 360-row corpus) | **0.232** | `phase_status.json` phase 4 notes; PHASE4 history | training run (not re-run) | ≤2026-07-03 | file-only |
| dynamic_overlap v2.1 (580-row corpus, 5× data, held-out n=87) | **0.165** (gate was ≥0.45 → FAIL) | `phase_status.json` phase 4 notes | training run (not re-run) | ≤2026-07-03 | file-only |
| Current model's dynamic_overlap (v2.2 richer input + MMR over logits) | **0.056** — note this is *lower* again and is now report-only | `phase4_results.json` | re-run above | 2026-08-23 | verified-today |

The 0.232 → 0.165 pair exists only in `phase_status.json` prose and working notes; no
JSON result file for v2/v2.1 is committed. It is quotable as a documented historical
run, not as a reproducible number.

### E. Intel / OpenVINO path

| Number | Value (current) | Source file | Command | Run date | Status |
|---|---|---|---|---|---|
| ONNX → OpenVINO IR parity (max abs logit delta, 4 random inputs) | **1.43e-06** | `mac/intel/convert.py` `verify_student_ir()`; `impl-track3.md` | `python3 -c "from intel import convert as c; print(c.verify_student_ir())"` | 2026-08-23 | verified-today |
| INT8 calibration rows | **360** (= `v2_scenarios.jsonl`) | `mac/intel/artifacts/int8_accuracy.json` | `quantize.accuracy_note(quantize.calibration_embeddings(), 6)` — recomputed from the committed FP32/INT8 IRs, no re-quantization | 2026-08-23 | verified-today |
| INT8 vs FP32 top-6 overlap, mean / min | **0.9505 / 0.6667** (gate: mean ≥0.8 AND min ≥0.5 → PASS) | same | same | 2026-08-23 | verified-today |
| INT8 vs FP32 logit delta, max / mean | **1.053 / 0.228** | same | same | 2026-08-23 | verified-today |
| Size on disk: torch / FP32 IR / INT8 IR | **657 / 661 / 235 KB** | `mac/intel/BENCHMARKS.md`; `ls -l artifacts` | `os.path.getsize` over `.bin`+`.xml` | 2026-08-23 | verified-today |
| INT8 size reduction | **2.8×** (661 / 235 = 2.81; `.bin` only 654 / 223 = 2.94). Documents say "2.7×". | derived | — | 2026-08-23 | verified-today (drift, minor) |
| Micro-latency p50, torch / OV FP32 / OV INT8 (batch 1, MLP only) | File: **0.015 / 0.025 / 0.031 ms**. Today: 0.037 / 0.063 / 0.069 ms (loaded machine). These are the bare MLP forward — **never quote as a board latency** | `BENCHMARKS.md` | `python3 intel/bench.py` (output redirected to scratch) | file 2026-07-03; re-run 2026-08-23 | verified-today (drift, load-dependent) |
| benchmark_app throughput FP32 / INT8 | 48,354 / 36,969 FPS | `BENCHMARKS.md` | `benchmark_app` (included in `bench.py`) | 2026-07-03 | file-only |
| Whisper base.en INT8 (openvino_genai, f32 hint) vs faster-whisper int8 | **0.35 s vs 0.42 s**, one synthesized 6.5 s clip, identical transcripts, **n = 1** | `mac/intel/README.md`; `docs/claude/working-notes/phase5-competition-ml/impl-track3.md`; `phase_status.json` phase 5 | not re-run (the OV whisper artifact is gitignored and absent; would need export + audio) | 2026-07-03 | file-only (single clip) |
| Whisper latency vs clip duration (faster-whisper base.en) | 2.5 s clip → 0.33 s (RTF 0.13); 5.2 s → 0.46 s; 10.6 s → 3.42 s; 21.5 s → 5.53 s; 32.3 s → 7.78 s | `mac/figures/results.json` | model run, not re-run | 2026-06-13 | file-only |
| VLM latency vs resolution (Qwen2.5-VL-3B, mlx) | 256 px 0.93 s; 512 px 2.05 s; 768 px 4.05 s; 1024 px 7.33 s; 1536 px 15.9 s; 2160 px 34.8 s | `mac/figures/results.json` | model run, not re-run | 2026-06-13 | file-only |
| Vision change-gate cost | **0.214 ms / frame** | `mac/figures/results.json` | not re-run | 2026-06-13 | file-only |
| Intel-hardware latency (Xeon / Core Ultra, Tiber) | **none** | — | `intel/bench.py` on Tiber — pending account | — | UNSOURCED (does not exist) |

**Concurrent, uncommitted work noticed during this task (not mine, not verified):**
`mac/intel/bench_e2e.py` and a new "End-to-end board latency — Apple M4" section appended
to `mac/intel/BENCHMARKS.md` (torch 74.2 ms p50 / ov-full 79.9 ms p50, n=40, board
Jaccard 0.923 vs torch). If that lands, it becomes a fourth latency figure and must be
reconciled with the 47.081 ms in `phase4_results.json` before either is quoted.

### F. Tests

| Number | Value (current) | Source file | Command | Run date | Status |
|---|---|---|---|---|---|
| Full suite | **144 passed, 0 xfailed, 0 failed** (5 warnings, 32.8 s) | `mac/tests/` | `VOICELINK_PERCEPTION=0 python3 -m pytest tests -q` | 2026-08-23 | verified-today |
| Previous suite state (pre-RC-3) | 116 passed, 2 xfailed | HANDOFF §11 | — | 2026-08-17 | superseded |
| `tests/test_answers.py` | **25** tests | — | `pytest tests/test_answers.py --co` | 2026-08-23 | verified-today |
| `tests/test_seam.py` | **24** tests collected (phase_status phase 1 says "12 seam tests" — true when written, the file has grown) | — | `pytest tests/test_seam.py --co` | 2026-08-23 | verified-today (drift, harmless) |

### G. Data / logging

| Number | Value (current) | Source file | Command | Run date | Status |
|---|---|---|---|---|---|
| Logged vocabulary taps | **121** rows | `mac/taps.jsonl` | `wc -l mac/taps.jsonl` | 2026-08-23 | verified-today |
| Authored v2 scenarios | 360 | `mac/evaluation/phase4/v2_scenarios.jsonl` | `wc -l` | 2026-08-23 | verified-today |
| Phase-3 cached boards | 240 (120 scenarios × 2 arms) | `mac/evaluation/phase3/boards.jsonl` | `wc -l` | 2026-08-23 | verified-today |
| Dedup threshold | cosine 0.95 (rows); vocab dedup τ 0.80; category cap 15% | `config.yaml`, `data.py` | — | — | verified-today (read) |

### H. Product / hardware claims

| Number | Value (current) | Source file | Command | Run date | Status |
|---|---|---|---|---|---|
| Wearable parts cost | "~$15" / "≈ $15" | none — no BOM, receipt, or parts list in the repo. The board is a Freenove ESP32-S3-WROOM CAM + SPH0645 mic; typical retail for that board alone is above $15 | — | — | **UNSOURCED** — write a BOM with prices or soften to "tens of dollars" |
| Board UI | 10 tap targets; 4 pinned safety words (yes / no / I need help / wait) | `config.yaml` `board.k=10`; `data.CORE_SET` | — | — | verified-today (read) |
| Firmware frame cadence / VAD constants | VGA JPEG 1 frame / 2000 ms; VAD 30 ms frames, RMS ≥350, 600 ms trailing silence, <300 ms discarded; 64×64 gate, threshold 12/255 | `PATENT_NOVELTY_BRIEF.md` §3/§8 (quoting firmware/server constants) | not checked against source in this pass | — | file-only |
| External statistics (CDC 1-in-31, 26.7% profound, 4–5 M AAC users, $4,000–$12,000 devices, 30–35% abandonment, 10–18 wpm) | as cited | `COMPETITION_EVIDENCE.md` with URLs | external — not verifiable in-repo | — | external citation |
| Festival 2025 participation figures | conflicting sources | — | — | — | **do not quote** (SPRINT §7) |

---

## 2. Discrepancies

Each line: file, line number, quoted text, correct value. "Correct" means the current
value in section 1. Documents are listed in the order SPRINT §9 names them; paper and
description files follow. **None of these files were edited in this task.**

### `docs/COMPETITION_EVIDENCE.md`

| Line | Quoted | Correct |
|---|---|---|
| 71–72 | "a distilled 769-d MLP student reproduces that policy at ~7 ms (~900× faster than the teacher pipeline)" | Student **does not** reproduce the policy (dynamic_overlap 0.056, sfp gate FAIL). End-to-end board **47.081 ms vs 6.71 s = 142.6×**. The 7 ms / 900× figure is the superseded v2.1 run's bare-MLP-ish number (`phase_status.json` phase 4: "7.7ms vs 6.97s = 905x"). Use SPRINT §7's replacement sentence. |
| 45 | "ESP32-S3 ≈ $15 wearable" | UNSOURCED (no BOM). |
| 123 | "120-scenario self-test harness with published metrics" | fine — but if the coverage number is quoted alongside it, use 1.00 vs 0.78 (see HANDOFF below). |

### `docs/PATENT_NOVELTY_BRIEF.md`

| Line | Quoted | Correct |
|---|---|---|
| 175–176 | "coverage-of-intent 1.00 vs 0.81" | **1.00 vs 0.78** (0.775) on the current scorer; 0.81 is the June-30 file value. |
| 236 | "Our v1 measured a 0.571 agreement" | historical v1 — not in any committed result file; label as historical or drop. |
| 321 | "same-function-pair rate from 0.168 (teacher) to 0.197" | current student sfp is **0.1865** vs teacher 0.1684 (0.197 was an intermediate run). |
| 337 | "(`dynamic_overlap` 0.056; full top-7 overlap 0.59)" | correct. |
| 750 | "coverage-of-intent 1.00 vs 0.81" | 1.00 vs 0.78. |
| 751 | "43.6 ms (IQR 39.4–45.8) vs 6.82 s (IQR 6.77–7.02) → 156×" | **47.081 ms (IQR 45.06–49.29) vs 6.71 s (IQR 6.63–6.97) → 142.6×**. |
| 752 | "distinct-fn 4.048 vs teacher 4.333 (ratio 0.934)" | **4.083 vs 4.333 (0.942)**; near-dup 0.0006 is correct. |
| 753 | "0.859 vs teacher 0.944; dynamic 0.775 vs 0.845" | **0.845 vs 0.944; dynamic 0.761 vs 0.845**. |
| 755 | "same-function-pair 0.1882 vs teacher 0.1684" | **0.1865 vs 0.1684** (still FAIL). |
| 758 | "NNCF INT8 2.7× smaller" | **2.8×** (661 → 235 KB). |
| 759 | "0.35 s vs 0.42 s" | correct, but it is n=1 on one synthesized 6.5 s clip — say so in a specification. |
| 763–767 | "`phase4_results.json` measures 43.6 ms / 156×. … Use 156× and the end-to-end number." | the file now measures **47.081 ms / 142.6×**; use 142.6×. |

### `docs/HANDOFF.md`

| Line | Quoted | Correct |
|---|---|---|
| 21 | "ESP32-S3 clip (~$15)" | UNSOURCED. |
| 58 | "taps.jsonl … (121 rows)" | correct (121). |
| 84 | "coverage-of-intent 1.00 vs 0.81" | 1.00 vs 0.78 on the current scorer. |
| 85–86 | "~7 ms vs ~6.97 s teacher pipeline. Honest MMR-path speedup 156x. Diversity retained (distinct-fn 96%)." | **47.081 ms vs 6.71 s, 142.6×; distinct-fn retention 94.2%** (4.083/4.333). 7 ms / 6.97 s / 96% are the v2.1 run; 156× is the Aug-8-era run. |
| 88 | "dynamic_overlap 0.232 → 0.165" | correct as a historical pair (file-only; current model is 0.056). |
| 93 | "2.7x smaller" | 2.8×. |
| 96 | "same-function-pair rate 0.188 vs teacher 0.168" | **0.1865 vs 0.1684**. |
| 203 | "116 passed, 2 xfailed" | superseded by line 221's "144 passed, 0 xfailed" — which is correct today. |
| 215 | "2366 unique words across 3600 slots" | 2362 on today's count (±4, immaterial). |
| 227 | "0.1865 vs teacher 0.1684" | correct. |

### `docs/DESKTOP_BRIEF.md`

| Line | Quoted | Correct |
|---|---|---|
| 49, 161 | "~$15 of parts" / "Roughly $15 of parts" | UNSOURCED. |
| 152–153 | "how often the board let you express your actual intent, 1.00 vs 0.81" | 1.00 vs 0.78. |
| 154–155 | "~143x faster … 47 milliseconds versus 6.7 seconds — while keeping 94% of the variety" | correct (142.6×, 47.081 ms, 6.71 s, 0.942). |
| 157 | "compresses to about a third of its size" | correct (235/661 = 0.355). |
| 158–159 | "(0.35s vs 0.42s)" | correct; n=1 clip. |
| 171 | "matches the teacher's word choices 5.6% of the time" | correct (0.056). |
| 176 | "0.761 against the teacher's 0.845" | correct. |
| 181 | "0.1865 vs 0.1684" | correct. |
| 199 | "only 32.5% … and 49% of what it can express is filler" | correct (32.5%, 49.1%). |
| 200 | "Making the list eight times bigger still only reaches 57%" | **Now sourced**: NOTES.md §1 / `answers.py` docstring, reproduced today — 800 words (8.5× the 94) → 0.566. Exactly 8× (752 words) → 0.553. Say "8.5× … 57%" or "8× … 55%". |

### `docs/SUBMISSION_SPRINT.md` (the document that commissioned this file)

| Line | Quoted | Correct |
|---|---|---|
| 162 | "coverage-of-intent 1.00 vs 0.81" | 1.00 vs 0.78. |
| 183 | "An 8× larger vocabulary still only reaches 57%" | 8.5× → 56.6% (8× → 55.3%). |
| 164–165 | "NNCF INT8 (2.7× smaller, top-6 overlap 0.950 mean / 0.667 min)" | 2.8×; overlaps correct. |
| 329–333 | "'…eight times bigger still only reaches 57%' appears in `DESKTOP_BRIEF.md` §5 and nowhere else — no measurement file backs it" | It is backed: `docs/claude/working-notes/rc3-open-answers/NOTES.md` §1 vocab-policy table (0.332 / 0.425 / 0.566) and the `answers.py` module docstring; reproduced today from `teacher_data.jsonl` with `data.build_vocab`. |
| 290 | "`phase4_results.json` (current, re-run Aug 17) 47.081 ms, 142.6×" | correct. |

### `phase_status.json`

| Line | Quoted | Correct |
|---|---|---|
| 17 | "12 seam tests pass" | 24 tests now in `test_seam.py` (grown; harmless). |
| 31 | "coverage-of-intent 1.00 vs 0.81" | 1.00 vs 0.78 on the current scorer. |
| 38 | "speedup PASS 7.7ms vs 6.97s = 905x … dynamic_overlap 0.165 … distinct-fn 96%, near-dup 0.0099, board-variety 0.563 … held-out n=87" | the whole phase-4 note describes the superseded **v2.1** run. Current: 47.081 ms / 6.71 s / 142.6×; dynamic_overlap 0.056; retention 0.942; near-dup 0.0006; variety 0.917; n=84. Phase 5's note (line 45) already carries the current values. |
| 45 | "2.7x smaller" | 2.8×. |

### `mac/evaluation/phase4/PHASE4_REPORT.md`

Matches `phase4_results.json` and today's re-run exactly (generated 2026-08-17 03:22). No discrepancy.

### `mac/intel/BENCHMARKS.md`

Committed values reproduce (INT8 accuracy exact; micro-latencies are load-dependent). The
uncommitted "End-to-end board latency — Apple M4" section is another agent's in-progress
work and is not verified here.

### `paper/rtca/README.md` and `paper/rtca/voicelink_rtca.tex`

| File:line | Quoted | Correct |
|---|---|---|
| README.md:33 | "1.00 / 0.81 coverage, 0.567 / 0.342 exact" | 1.00 / 0.78 coverage, 0.525 / 0.333 exact on the current scorer (or re-commit the Phase-3 file). |
| README.md:37 | "INT8 2.7×" | 2.8×. |
| voicelink_rtca.tex:65 | "coverage-of-intent from 0.81 to 1.00" | 0.78 → 1.00. |
| voicelink_rtca.tex:214 | "Coverage-of-intent 1.00 & 0.81" | 1.00 & 0.78. |
| voicelink_rtca.tex:215 | "Exact match to oracle word 0.567 & 0.342" | 0.525 & 0.333. |
| voicelink_rtca.tex:177 | "6.71 s (IQR 6.63–6.97)" | correct. |

### `docs/PROJECT_DESCRIPTION.md` (not in the §9 list, but quotes the same numbers)

| Line | Quoted | Correct |
|---|---|---|
| 67–68 | "coverage-of-intent (1.00 vs 0.81)" | 1.00 vs 0.78. |
| 73–74 | "Measured 156× faster … (43.6 ms vs 6.8 s)" | 142.6× (47.081 ms vs 6.71 s). |
| 100–102 | "same-function-pair 0.188 vs teacher 0.168 … two tests are marked `xfail`" | 0.1865 vs 0.1684; **0 xfail** now (144 passed). |

---

## 3. Summary of what is unsourced or could not be re-run

1. **"~$15" wearable cost** — no BOM anywhere in the repo. Either add one or soften.
2. **0.232 → 0.165** — exists only in `phase_status.json` prose and working notes (no v2/v2.1
   result JSON committed). Quotable as a documented historical run only.
3. **Whisper 0.35 s vs 0.42 s** — a single synthesized clip, artifact gitignored and absent;
   cannot be re-run without an export. Say "one 6.5 s clip".
4. **Teacher latency 6.71 s and therefore 142.6×** — needs an LLM run; not re-run today.
   Student-side latency re-timed at 59–69 ms on a loaded machine vs 47 ms in the file, so
   the speedup should be re-measured once, quietly, before it goes into a paper
   (`cd mac && VOICELINK_PERCEPTION=0 python3 evaluation/phase4/evaluate.py --teacher-samples 8`).
5. **Phase-3 coverage 1.00 vs 0.81** is the one widely-quoted number that the current
   code no longer reproduces (0.78). Decide: re-commit `selftest_results.json` from
   `selftest.py --from-boards` and quote 0.78, or keep quoting 0.81 and say "as scored
   in June" — not both.
6. **Intel hardware** — nothing.
