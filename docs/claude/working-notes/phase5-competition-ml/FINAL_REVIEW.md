# Phase 5 — FINAL REVIEW (for your morning sign-off)

Branch `phase4-distillation`, commits `52c667b..aff0607` (9 commits, overnight
autonomous run you authorized). Read this, then say "good to go" to merge to
main — or flag anything and I'll rework it.

## What the plan promised vs what shipped

**Track 1 — student quality (RC-1/2/3).** SHIPPED, exceeded on hedges.
Vocab semantically deduped 116→94 (the 7 "I don't know" paraphrases → 1, with a
canonical map preserving label mass); new `infer.py` does MMR + answer-quota +
hedge-cap selection instead of raw topk. Hedge tokens in top-7: **2.107 →
0.012** (teacher 0.321); boards with ≥2 hedges: 72.6% → 0%. Two honest drifts
documented and review-approved (single-link dedup; relevance-first answer
reservation). The transcript-candidate union (RC-3-lite) shipped but is
**disabled**: review ablation proved its question-echo nouns cost 0.24 dynamic
coverage. Out-of-vocab answers ("aunt") remain unrepresentable — that is
teacher-side answer-coverage work (PLAN item, next phase), marked by 2 honest
xfail tests.

**Track 2 — honest gates.** SHIPPED. Coverage is now gated on the words the
MODEL contributes (dynamic-within-top-7, hedge moves excluded); word-boundary
matching killed the "no"-matches-"now" leak; same-function-pair metric added.
The old gates certified the broken model at 0.958; the new ones correctly
failed it (0.563) and correctly pass the fixed one.

**Track 3 — Intel/OpenVINO.** SHIPPED. openvino 2026.2.1 / nncf 3.2.0 /
optimum-intel 2.0.0 (arm64-native). student.onnx → FP32 IR (parity 1.4e-06) →
NNCF INT8 (2.7× smaller, top-6 overlap mean 0.950 / min 0.667, gated on both);
MiniLM exported; Whisper base.en INT8 exported and **beats faster-whisper**
(0.35s vs 0.42s, identical transcript). `VOICELINK_STUDENT_BACKEND=openvino`
serves boards through the seam; IRs carry vocab-sha sidecars. `mac/intel/BENCHMARKS.md`
has the honest table (torch wins raw latency on this tiny MLP on ARM — the INT8
win is size; Intel-hardware numbers pending Tiber access).

## Final measured numbers (held-out n=84, the gates as recalibrated by review)

| gate | student | teacher | bar | verdict |
|---|---|---|---|---|
| distinct-functions retention | 4.05→ | 4.33 | ≥90% of teacher | **PASS** (0.934) |
| near-dup rate | 0.0006 | 0.0 | ≤0.08 | **PASS** |
| board-variety | 0.917 | — | ≥0.5 | **PASS** |
| same-function-pair | 0.188 | 0.168 | ≤ teacher | **FAIL — known gap** |
| coverage-of-intent (dynamic) | 0.775 | 0.845 | ≥90% of teacher | **PASS** (0.917×) |
| speedup (honest MMR-path timing) | 43.6 ms | 6.8 s | ≥10× | **PASS** (156×) |
| INT8 top-6 overlap | 0.950 mean / 0.667 min | — | ≥0.8 / ≥0.5 | **PASS** |

Tests: **116 passed, 2 xfailed** (the xfails are deliberate RC-3 markers).

## Review loop — findings and resolutions (all documented in review-opus.md)

- **Opus C1 (blocker, the catch of the night):** eval measured the new MMR
  boards while the DEVICE still served raw-topk hedge walls through
  `_student_board`. Fixed — production seam now runs `infer.student_board`;
  a new test proves the diverse fallback is NOT what builds the board.
- **Opus adjudications:** sfp + coverage gates made relative-to-teacher (the
  absolute bars were mis-estimated — teacher itself failed them);
  dynamic_overlap demoted to report-only (word-parroting contradicts the
  coverage thesis). Sigmoid-into-MMR was tested and REJECTED by measurement
  (0.634–0.775 vs 0.789 cosine).
- **Sonnet findings (all adjudicated REAL):** hedge lists unified into
  `evaluation/hedges.py`; loud vocab/artifact staleness guards (checkpoint sha
  + IR output-dim + IR sha sidecar); dedup component guard + non-hedge merge
  audit; seating-probe stems tightened (they had accepted "table" as an answer
  to "who do you want to sit next to?").
- **Codex findings (all applied):** torch-free OpenVINO inference path (a
  runtime-only install would have silently fallen back to the LLM); IR
  vocab-sha sidecars (same-size reorder defense); speedup now times the real
  MMR path (43.6 ms honest, was 8 ms timing a path the device no longer runs).
- **Edge cases:** 36 adversarial tests, 3-way reviewed (Haiku PASS; Sonnet+Codex
  convergent on 3 fixes, applied). See EDGE_CASES.md.

## Known gaps / open items (need you)

1. **sfp 0.188 vs teacher 0.168** — the one failing gate. The 94-word closed
   vocab is filler-heavy; the student stacks ~0.02 more same-function pairs than
   the open-vocab teacher. Function-aware MMR + λ sweep both applied; converged
   here. Next lever is teacher answer-coverage (below), which grows the
   concrete-answer vocab and dilutes filler. NOT gamed away — reported as FAIL.
2. **RC-3 teacher answer-coverage** — "sugar"/"aunt"-type answers below the
   frequency floor are still unrepresentable. Highest-leverage ML work
   remaining; makes gap 1 easier too.
3. **Tiber AI Cloud signup** (browser + account, ~10 min) → run
   `mac/intel/bench.py` on Xeon/Core Ultra for the Intel-hardware column.
4. **IDR participation** (3 rubric points) — email the festival organizers;
   enrollment mechanics unverifiable online.
5. **transformers venv conflict** — optimum-intel pins <5.1 vs mlx-vlm ≥5.5;
   exports were done under a temporary downgrade and re-verified. A dedicated
   export venv is the clean fix (documented in mac/intel/README.md).

## Rubric position after tonight

- **Metric 02 (AI Innovation):** two-stage teacher→student distillation with
  literature-grounded selection (MMR), anti-gaming evaluation, and honest
  failure reporting — plus `docs/COMPETITION_EVIDENCE.md` mapping every claim
  to citations and every guardrail to a named ethics framework.
- **Metric 03 (Intel):** OpenVINO + NNCF + optimum-intel now demonstrably in
  the lifecycle with our own measured tables — the same narrative shape as
  last year's global winner, minus the Intel-hardware numbers (item 3).
- **Metric 01:** evidence dossier done (CDC 1-in-31, 26.7% profound autism,
  10–18 vs 125–185 wpm, SDG 10.2/4.5 verbatim).
