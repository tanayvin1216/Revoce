# RC-3 — open-vocabulary answers

Closes the item carried as "teacher answer-coverage (RC-3)" in the Phase-5 final
review and as open item 6 in `docs/HANDOFF.md`. Written 2026-08-17.

The short version: RC-3 was misdiagnosed. It was filed as *teacher* answer
coverage — the teacher supposedly not producing concrete answers often enough.
Measurement says the teacher produces them constantly; the closed vocabulary
cannot hold them, and no amount of teacher data changes that. The fix is
architectural, and it is now in.

---

## 1. What the measurement actually said

Run over the 600-row teacher corpus (`teacher_data.jsonl`) against the shipped
94-word vocab:

| quantity | value |
|---|---|
| teacher dynamic word slots | 3600 |
| unique teacher words | 2366 |
| unique words appearing < 3 times (the `min_freq` floor) | 2238 (94.5%) |
| teacher word slots representable in the vocab | **32.5%** |
| of what IS representable, hedge/filler | **49.1%** |
| rows with ZERO representable dynamic words | 12.2% (73/600) |

Two consequences follow directly.

**The student was never being asked to match the teacher.** It could only ever
reach 32.5% of the teacher's dynamic choices, and half of that reachable half is
filler. Reported `dynamic_overlap` of 0.165 against a ceiling of 0.325 is ~51%
of the achievable maximum, not the 16% failure it reads as.

**The Phase-4 negative result is explained.** "5x-ing the training data made
teacher-agreement *worse*" was recorded as falsifying the data-volume
hypothesis. It does, but for a specific reason: concrete answers are
inherently long-tail — "right arm" is correct for exactly one scene — so new
rows contribute new one-off words rather than repeats. More data pushes the
frequency ranking further toward generic filler while the ceiling stays flat.

Growing the vocabulary does not rescue this. Measured, holding the corpus fixed:

| vocab policy | coverage of teacher slots |
|---|---|
| min_freq=3, max 200 (shipped) | 0.332 |
| min_freq=2, max 400 | 0.425 |
| min_freq=1, max 800 | 0.566 |

An 8.5x larger output layer buys 0.23 coverage and is still nowhere near
complete. **A closed-vocabulary classifier is the wrong instrument for the
concrete-answer half of this problem.** That is the real RC-3.

## 2. What shipped

`mac/evaluation/phase4/answers.py` — an open-vocabulary answer extractor. The
closed-vocab student keeps doing what it is good at (fast, diverse, general
communicative functions); concrete answers are read out of the context at
inference time, so "aunt" is reachable without ever entering the vocab.

Question type decides where the answer lives:

- **alternative** ("the window seat or the aisle seat?") — the answers are
  stated in the question. Split the coordination, take both conjuncts as
  multi-word spans. No embedding needed to find them.
- **deictic wh** (`where` / `who` / `whom` / `which`) — the answer is in the
  scene. Generate multi-word spans from the scene, drop any span echoing a
  question token, rank against an answer-type prototype blended with scene
  salience.
- **everything else** — stand down; the board is exactly the closed-vocab board
  it was before.

No new dependency: spans are built with n-grams plus determiner stripping and
function-word boundary rejection, and MiniLM (already in the pipeline) does the
semantic filtering. spaCy's model is deliberately not pulled into the Intel
runtime for this.

## 3. Why the previous attempt failed, and what was different this time

`infer.CONTEXT_UNION` (RC-3-lite) was built and disabled because it cost 0.24
dynamic coverage. Two defects:

1. it unioned single **tokens** — "window", "seat", "aisle" — not "window seat";
2. it scored answers against the standard query vector, which is **0.7 × the
   question** (`selection._Q_WEIGHTS`), so question-echo nouns had inflated
   cosine and displaced real answers.

Defect 2 is the deep one. An answer to "Where would you like to sit?" is not
similar to that question — it is similar to the answer space the question opens.
So `answers.py` never ranks candidates by question similarity, and
`infer._reserve_open` reserves them by extractor rank rather than query cosine.

## 4. The three iterations that were wrong first

Recorded because each was rejected by measurement, not by argument.

**(a) Flat 2 slots on every question.** Fixed all three probes; cost dynamic
coverage 0.775 → 0.606. Rejected.

**(b) A cosine(span, question) admission floor.** Behaves as a clean monotone
dial between coverage and same-function-pair rate, but there is no setting where
both gates pass — and it fails on exactly the case RC-3 is about:
cosine("aunt", "who do you want to sit next to?") is 0.233, so the floor that
protected coverage also deleted the person answer. Rejected as the primary
mechanism; survives only as a weak 0.10 backstop.

**(c) A vocab-sufficiency stand-down** ("skip the open path when the vocab
already has a relevant answer"). Measured **inert** — identical gate numbers at
every threshold including "never stand down". Deleted rather than shipped as an
unmeasured knob. Its first version was also silently a no-op for a different
reason: it dropped the *reservation* but left the spans in the pool, where the
MMR fill selected them anyway.

What worked was a **grammatical** discriminator, not a topical one. Every
coverage loss from an over-eager open path was a `what`/`how` question whose
answer is a state, a quantity, or an abstraction:

| question | real answer | scene offers |
|---|---|---|
| "How was your day?" | "it was good" | dinner, dishes, home |
| "How many people in your party?" | "four people" | restaurant clatter |
| "What are you going to wish for?" | "it's a secret" | cake, candles |

`where`/`who`/`which` ask the user to select among referents physically present.
`what`/`how`/`when`/`why` do not. Restricting the scene path to the deictic
wh-words, and requiring second-person volition ("do you want", "would you like"),
recovered the coverage while keeping the answers.

## 5. Results

Held-out n=84, gates as recalibrated by the Phase-5 review. `--teacher-samples 8`.

| gate | before | after | bar | verdict |
|---|---|---|---|---|
| distinct-functions retention | 0.934 | 0.942 | ≥90% of teacher | PASS |
| near-duplicate rate | 0.0006 | 0.0006 | ≤0.08 | PASS |
| board variety | 0.917 | 0.917 | ≥0.5 | PASS |
| same-function-pair | 0.1882 | 0.1865 | ≤ teacher (0.1684) | **FAIL — still open** |
| coverage-of-intent (dynamic) | 0.775 | 0.761 (0.901×) | ≥90% of teacher | PASS |
| speedup | — | 142.6× (47.1 ms vs 6.71 s) | ≥10× | PASS |
| dynamic_overlap (report-only) | 0.056 | 0.056 | — | unchanged |

Tests: **144 passed, 0 xfailed** (was 116 passed + 2 xfailed). The two RC-3
xfail markers in `test_phase4.py` are live assertions now — the "aunt" board the
Phase-5 review called unrepresentable:

```
['yes', 'no', 'i need help', 'wait', 'relatives', 'aunt', 'chair']
```

`tests/test_answers.py` adds 25 tests, weighted toward the stand-down cases,
since offering scene props to a question whose answer is not in the room is what
costs coverage.

## 6. Honest limits — read before quoting any of this

**The held-out corpus cannot measure this feature.** The open path fires on
**2 of 84** held-out rows, and only **1 of 71** coverage-scored rows. The
before/after gate deltas above are therefore one- and two-row noise, and the
0.1882 → 0.1865 same-function-pair movement must **not** be reported as an
improvement. What the numbers do support is the negative claim that matters for
shipping: **no gate regressed**, and coverage still clears its bar at 0.901×.

The demonstrated evidence is the probe set and the mechanism, not the corpus
statistics. Building a deictic-question evaluation set is the obvious next step
and is the only way this feature becomes measurable.

**The same-function-pair gate is still FAILING (0.1865 vs teacher 0.1684).** The
Phase-5 review predicted RC-3 would make that gap easier. At full aggression the
open path did reach exact teacher parity (0.1684) — but only at coverage 0.690,
well under the bar. The two remain in tension at k_dyn=6, and the failing gate
stays reported as failing.

**Concrete answers reduce word-overlap with the teacher by construction**, since
an extracted span is not a teacher vocab word. `dynamic_overlap` happens to be
unchanged here, but it is the wrong metric to defend this work with; coverage is.
