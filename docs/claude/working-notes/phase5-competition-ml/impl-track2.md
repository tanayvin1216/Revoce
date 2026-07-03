# Track 2 — Honest evaluation gates (implementer log)

Worktree: `.claude/worktrees/agent-af6b6fcd35bdc78c0`, branch fast-forwarded from
a4bed1a to phase4-distillation tip 50f4f09 (the worktree was cut from a stale
phase-1 commit and had no phase3/phase4 code; ff-only merge, no conflicts).

## Setup
- Copied gitignored artifacts from the main checkout into the worktree so the
  CURRENT (broken) checkpoint can be evaluated here: `teacher_data.jsonl`,
  `student.pt`, `vocab.json`, `splits.json`. They stay gitignored.
- Import sanity: `VOICELINK_PERCEPTION=0 python3 -c "import torch, yaml; from
  evaluation.phase3 import metrics3, taxonomy"` → ok (torch 2.12.0).
- Baseline `pytest tests/test_phase3.py` → 18 passed.
- `classify_functions` confirmed at `mac/evaluation/phase3/taxonomy.py:82` —
  matches the plan's assumption. GUARANTEED_CORE = ['yes','no','I need help','wait'].

## Pre-implementation probe (pin the definitions)
The investigation reports dynamic coverage 0.803 and same-function-pair 0.1689
"over the dynamic top-7". Before coding, measuring both candidate definitions
(dynamic words INSIDE top-7 vs the full k_dyn dynamic tail) on the current
checkpoint to implement the one that reproduces the investigation.

Probe output (real, current checkpoint, held-out n=84, scored n=71):

```
coverage full top7            0.958
coverage dyn-in-top7          0.803   <- matches investigation 0.803 exactly
coverage dyn-all              0.93
coverage dyn-in-top7 nohedge  0.718 n= 71
sfp full top7                 0.1689  <- matches investigation 0.1689 exactly
sfp dyn-in-top7 (student)     0.5397
sfp dyn-all (student)         0.3675
sfp dyn-in-top7 (teacher)     0.1865
```

Pinned definitions (implemented accordingly):
- DYNAMIC coverage = coverage_of_intent over the non-core words INSIDE the top-7
  window (not the full k_dyn tail — that gives 0.93 and is not what the
  investigation measured). Gated number additionally drops hedge-labelled
  acceptable moves → 0.718 on the current checkpoint.
- same_function_pair_rate = over the FULL top-7 ("over the same dynamic top-7" in
  the investigation §3 meant "the same top-7 the metrics judge" — the 0.1689
  reproduces only on the full top-7; dynamic-only gives 0.54).
- Deviation from the plan text flagged: plan says "score coverage on sb_dynamic
  (built at evaluate.py:132)" — that variable holds ALL non-core board words
  (k_dyn=6), which scores 0.93 and does NOT reproduce the investigation's 0.803.
  Implemented the investigation's definition (dynamic-within-top-7) since the
  acceptance criterion is reproducing the honest numbers.

## Implementation (TDD: tests written first, RED confirmed 11/12 failing)

1. `mac/evaluation/coverage_metrics.py` — `_phrase_in` word-boundary helper
   (`(?<!\w)…(?!\w)` lookarounds around the escaped phrase); `_lexical_hit` now
   uses it instead of plain substring containment. Signature unchanged.
2. `mac/evaluation/phase3/metrics3.py` — ADDITIVE only:
   - `same_function_pair_rate(board, k=TOPK, board_vecs=None)` — fraction of
     top-k pairs sharing a taxonomy function label (uses
     `taxonomy.classify_functions`, confirmed at taxonomy.py:82).
   - `HEDGE_PHRASES` + `is_hedge(text)` — normalized lookup (lowercase, curly
     apostrophe, trailing punctuation) over the investigation's hedge cluster.
   No existing signature touched.
3. `mac/evaluation/phase4/evaluate.py`:
   - `SFP_GATE = 0.10`, `DYNAMIC_COVERAGE_GATE = 0.90` constants.
   - `student_board` now `from evaluation.phase4.infer import student_board`
     inside try/except ImportError with the previous inline raw-topk builder as
     the fallback (worktree self-contained; resolves to Track-1's after merge).
   - `dynamic_words(board, core_words, k=7)` pure helper (testable without a
     checkpoint).
   - `held_out_eval` scores BOTH coverages (full top-7 reported un-gated;
     dynamic-within-top-7 vs non-hedge moves gated >=0.90) and both arms'
     `same_function_pair_rate` (`_df_and_dup` -> `_df_dup_sfp`, one embed per
     board). sfp <=0.10 added to the ① diversity_pass conjunction.
   - Report table: new same-function-pair row; ④ split into full (reported,
     not gated) and DYNAMIC (gated) rows; prose updated.
4. NEW `mac/tests/test_eval_gates.py` — 12 synthetic-board tests (real local
   MiniLM, same policy as test_phase4): hedge-wall FAILs sfp gate / diverse
   board PASSes; "no" vs "now" boundary at `_lexical_hit` and
   `coverage_of_intent` level; dynamic-vs-full divergence on a pinned-core
   board; hedge-move dropping flips a gamed cover to honest FAIL; gate
   constants pinned.

## Real output

`pytest tests/test_eval_gates.py` → 12 passed. `pytest tests/test_phase3.py` →
18 passed. Full `pytest tests/` → **67 passed** (7.68s).

`VOICELINK_PERCEPTION=0 python3 evaluation/phase4/evaluate.py` (default
--teacher-samples 8, real teacher timed) on the CURRENT checkpoint
(n_heldout=84, n_scored=71):

```
student_same_function_pair  0.1689   gate <=0.10  -> FAIL   (investigation: 0.169)
teacher_same_function_pair  0.1684
coverage_of_intent student           0.944 (full, reported un-gated; was 0.958
                                            before the no/now boundary fix)
coverage_of_intent student_dynamic   0.676  gate >=0.90 -> FAIL
coverage_of_intent teacher_dynamic   0.845
speedup 892.4x (8.284 ms vs 7.39 s)  -> PASS
>>> WRAPPER CRITIQUE DEAD: NO (①=FAIL ③=PASS) <<<   exit code 1
```

Both acceptance regimes hit: dynamic coverage clearly below 0.90 (0.676; the
raw un-hedge-filtered dynamic number is 0.803 exactly per the probe; hedge-drop
plus the word-boundary fix take the gated number to 0.676) and sfp clearly
above 0.10 (0.1689 exactly). The gates now catch what they previously certified
(old report: coverage 0.958 PASS).

## Flags for the orchestrator

- **Teacher sfp ~0.03 did NOT reproduce.** The plan says "teacher measures
  ~0.03"; measured teacher full-top-7 sfp = 0.1684 (and 0.1865 on
  dynamic-in-top-7). The pinned core alone contributes 1/21 = 0.048 ("yes" and
  "wait" both classify `comment`), so <=0.10 is a strict bar that the TEACHER
  currently fails too. The student number and gate are implemented exactly per
  plan/investigation; post-Track-1 the student may need the same scrutiny as
  the teacher on this bar.
- PHASE4_REPORT.md is regenerated by evaluate.py; the hand-edited "v3 verdict"
  prose from the 2026-06-30 run is overwritten by the auto verdict (inherent to
  running the script — the orchestrator regenerates post-merge anyway).
- board-variety FAIL (0.333) predates this track (visible in the old committed
  report) — not introduced here.
- Worktree was cut from a stale phase-1 commit; fast-forwarded to
  phase4-distillation tip 50f4f09 before starting.

