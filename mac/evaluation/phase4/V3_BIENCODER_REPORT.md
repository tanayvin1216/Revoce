# Phase 4 v3 — open-vocabulary bi-encoder student

Corpus: 600 teacher boards regenerated 2026-07-22 (Qwen2.5-3B-Instruct-4bit,
temp 0.0), 559 rows after exact/category/embedding dedup. Held-out n=84 — the
same split discipline and the same held-out size as v2, so the two are
comparable.

## Why v2 was replaced

v2 was `Linear(769 -> V)`: one output column per vocabulary item. A closed
output space has to be rationed by frequency, so an item needed `min_freq=3`
teacher boards to get a column at all.

Measured on this corpus: **1959 distinct dynamic phrases, of which the floor
keeps 103. It discards 94.7%.**

And it discards the wrong 94.7%. Situation-specific language is rare *by
definition* — a name, "sugar", "my neighbour" — so the phrases most likely to
actually answer the question in front of the user were exactly the ones the
architecture could not hold. That is what `dynamic_overlap = 0.056` was telling
us: the student had learned the teacher's generic function skeleton and almost
none of its concrete word choice. We read that as a capacity problem for a long
time. It was an output-space problem.

## The model

Two towers into a shared 128-d space; the score is their dot product.

```
context: LayerNorm(769) -> Linear(769,256) -> ReLU -> Dropout -> Linear(256,128)
phrase : LayerNorm(384) -> Linear(384,256) -> ReLU -> Dropout -> Linear(256,128)
score(c, p) = <f_ctx(c), f_phrase(p)>
```

364K params total, 232K on the serving hot path (v2: ~165K, closed at 94 words).
Nothing in the architecture depends on the size or membership of the candidate
set, so the frequency floor simply stops existing. Training is multi-label BCE
over a sampled candidate set — the batch's own positives as hard negatives, plus
128 random draws from the bank.

Serving cost stays low because the phrase tower runs **once, offline** over the
bank and the projections are cached. A board is one context-tower pass plus a
matmul. Only live context terms pay per-board phrase-tower cost.

## What the open vocabulary actually buys

Three question scenarios, dynamic slots only, asking of each word: *could this
have existed under v2's `min_freq=3` floor?*

| question | board (dynamic 6) | representable in v2 |
|---|---|---|
| Who would you like to sit next to at dinner? | dinner, i mean dinner, my sibling's child, i want a salad, i'd like to play video games, mexican | **0/6** |
| Do you want sugar in your tea? | i want chocolate, i need a snack, no not wheat, just water thanks, i am not sure, yes please | 3/6 |
| Who did you sit with at lunch today? | i need to sit down., **my neighbor**, school, where's the milk?, i'm waiting for the bus, **Sam** | **0/6** |

**3/18 overall.** 83% of what v3 chose was unreachable under the old
architecture at any score.

`Sam` is the one to look at. It is a name from the on-device personal-context
store — it appears in no teacher board, has no frequency, and is embedded and
scored at serve time through the same phrase tower as everything else. v2 had
no mechanism that could ever put it on a board. This is also why v2's
`CONTEXT_UNION` had to be disabled: unscored terms could only be bolted onto the
pool past scored ones, and the ablation measured that costing 0.24 dynamic
coverage. Here they compete on the same axis instead of queue-jumping.

## Head-to-head against v2 on the gates

v2 was **retrained from scratch on this same corpus** so the comparison is
apples-to-apples: same 559 rows, same held-out n=84, same teacher boards, same
gate code (`evaluate.held_out_eval`, with the board producer injected). The
previously published v2 numbers came from a different corpus and are not used
here.

| gate | v2 | v3 (first cut) | **v3 (tuned)** | teacher | bar |
|---|---|---|---|---|---|
| distinct-functions | 4.071 (0.947×) | 4.357 (1.014×) | **4.690 (1.091×)** | 4.298 | ≥90% of teacher |
| near-duplicate rate | 0.0 | 0.0 | 0.0006 | 0.0006 | ≤0.08 |
| same-function-pair | 0.1899 **FAIL** | 0.1621 | **0.1378** | 0.1735 | ≤ teacher |
| board-variety | 0.952 | 1.000 | 0.988 | — | ≥0.5 |
| coverage-of-intent (dyn) | 0.803 (0.920×) | 0.718 **FAIL** | **0.789 (0.904×)** | 0.873 | ≥90% of teacher |
| dynamic_overlap (report) | 0.067 | 0.091 | **0.139** | — | — |
| **diversity gate** | **FAIL** | PASS | **PASS** | | |
| **coverage gate** | PASS | **FAIL** | **PASS** | | |

**v3 tuned passes both gates. v2 never did.** It also fixes the
same-function-pair failure that stood through all of phase 5 as a known gap
(0.1378 against the teacher's 0.1735), and scores 9% *higher* distinct-functions
than the teacher itself. Word-level agreement doubled v2's (0.139 vs 0.067).

### Which of these verdicts survive resampling

Every phase-4 gate is a relative claim (`student >= 0.90 * teacher`, or
`student <= teacher`). So the honest question is not the point estimate but
whether the *paired per-scenario margin* is reliably on the right side of zero.
95% bootstrap CI over scenarios, both models, same corpus, same gates:

| paired margin vs bar | v2 | v3 |
|---|---|---|
| coverage-of-intent | +0.0169, CI [−0.1014, +0.1282] → **INCONCLUSIVE** | +0.0028, CI [−0.1042, +0.1042] → **INCONCLUSIVE** |
| distinct-functions | +0.2036, CI [−0.0179, +0.4250] → **INCONCLUSIVE** | +0.8226, CI [+0.6107, +1.0238] → **PASS** |
| same-function-pair | −0.0164, CI [−0.0414, +0.0091] → **INCONCLUSIVE** | +0.0357, CI [+0.0164, +0.0556] → **PASS** |

Three things follow, and two of them are uncomfortable.

**1. v3's diversity wins are real.** distinct-functions and same-function-pair
both clear zero by a wide margin with intervals nowhere near it. These are the
only robustly-supported model comparisons anywhere in the phase-4 evaluation.

**2. Neither model's coverage verdict means anything at this n.** v3's margin is
+0.0028 with a CI of ±0.104 — the interval is **37× wider than the effect**.
v2's is the same story. `coverage-of-intent` simply cannot resolve a difference
between these models on 71 scenarios. v3's coverage gate is reported as PASS by
the point estimate and INCONCLUSIVE by the interval; the interval is the honest
answer. This is a property of the instrument, not of either model, and the fix
is more scored scenarios, not a better student.

**3. The phase-5 "known gap" was never a gap.** v2's same-function-pair FAIL —
carried through FINAL_REVIEW as the one honest failure, with a paragraph
explaining why it was not gamed away — has a CI of [−0.0414, +0.0091]. It
straddles zero. It was not statistically distinguishable from passing, and the
project spent real effort reasoning about a number that was noise. That is worth
recording precisely because the reporting around it was scrupulous; being honest
about a point estimate is not the same as the point estimate meaning something.

**What may be claimed:** v3 beats v2 robustly on board diversity, and reaches
parity with the teacher on coverage. **What may not:** that v3 "passes the
coverage gate", or that v2 "fails same-function-pair".

### What actually fixed it

The first cut failed coverage at 0.718. The diagnostic that mattered: of 20
held-out misses, **19 had a satisfying phrase already inside the top-18 pool.**
So the model's ranking was fine and the pool cut was fine — MMR was discarding
the right answer.

The cause was structural. `mmr_select` computes relevance as cosine-to-situation
and the bi-encoder's own scores were used only to cut the pool, then thrown
away. For v2 that barely mattered (18 of 95 candidates, so the model's ranking
did little work). For v3 the model ranks 1959 candidates down to 18 and then had
its judgement ignored.

Fix: feed the model's scores into MMR as the relevance term, blended 50/50 with
cosine, and double the pool (`pool_mult` 3 → 6). Swept `rel_source` ×
`lam` × `pool_mult` on the **validation** split; `blend / 0.6 / 6` was the only
setting passing both gates. The test split was scored **once**, after the config
was fixed. Model-scores-alone gives the best diversity of anything measured
(df 4.69, sfp 0.139) but does not recover coverage; cosine-alone recovers
neither.

Sweep table is in `config.yaml:selection_v3`.

### Honest caveat on board quality

Selection is visibly imperfect: "mexican" and "i'd like to play video games" are
poor answers to a seating question, and "i mean dinner" nearly duplicates
"dinner". The coverage failure above is the measured version of that
observation.

## Quantization: sweep per tower, not per model

The two towers have different cost profiles, so quantizing the model as one blob
hides the real decision. The context tower is the hot path — size and speed both
matter. The phrase tower never runs at serve time, so its latency is free and
only the quality of the cached projections matters.

Nine combinations of FP32 / INT8 (NNCF PTQ) / INT4 (NNCF weight compression),
scored by top-6 board overlap against the FP32 boards:

| context | phrase | IR size | board overlap vs FP32 | hot-path ms | Pareto |
|---|---|---|---|---|---|
| fp32 | fp32 | 1082 KB | 1.000 | 0.032 | yes |
| fp32 | int8 | 804 KB | 0.960 | 0.032 | yes |
| int4 | fp32 | 513 KB | 0.954 | 0.030 | yes |
| int4 | int8 | 235 KB | 0.936 | 0.030 | yes |
| fp32 | int4 | 766 KB | 0.930 | 0.032 | |
| int4 | int4 | 197 KB | 0.922 | 0.030 | yes |
| int8 | fp32 | 587 KB | 0.920 | 0.041 | |
| int8 | int8 | 308 KB | 0.909 | 0.041 | |
| int8 | int4 | 271 KB | 0.897 | 0.041 | |

**INT4 beats INT8 on the context tower on all three axes** — smaller (513 vs
587 KB), more accurate (0.954 vs 0.920) and faster (0.030 vs 0.041 ms). NNCF
INT8 PTQ quantizes activations as well as weights, and on a 769→256→128 MLP
that costs more than dropping weights to 4 bits and leaving activations in
float. "Fewer bits is worse" does not hold here, and quantizing the model as a
unit would never have surfaced it.

Best compression on the front: **int4/int4, 197 KB — 5.5× smaller than FP32 at
0.922 board overlap.**

Latency is flat within noise across variants (0.030–0.041 ms). Consistent with
what `intel/BENCHMARKS.md` already says about the v2 student: at this size the
win is memory, not speed, and claiming otherwise would be dishonest.

Measured on Apple Silicon (M4 Pro) via the OpenVINO ARM CPU plugin,
openvino 2025.3.0 / nncf 2.18.0. **These are not Intel-hardware numbers.**

## Reproduce

```bash
VOICELINK_PERCEPTION=0 python -m evaluation.phase4.gen_teacher --target 600
VOICELINK_PERCEPTION=0 python -m evaluation.phase4.train_biencoder
VOICELINK_PERCEPTION=0 python intel/biencoder_pareto.py
```

`teacher_data.jsonl`, `phrase_bank.json`, `splits_biencoder.json` and
`student_biencoder.pt` are committed. The v2 artifacts were not, and were lost
with the machine that made them.

## Open

1. **Grow the coverage-scored set well past n=71.** This is now the single
   highest-leverage evaluation task: at this n the metric cannot separate any
   two models, so no amount of student work can be validated against it. It
   needs more scenarios carrying authored `acceptable_moves`, not more teacher
   boards. Re-tuning selection against the test split is NOT a lever — that
   split has been scored.
2. **Decide what the gates mean now that they have intervals.** A gate whose CI
   straddles zero should probably report INCONCLUSIVE rather than PASS/FAIL in
   the verdict line, instead of only in the margin block underneath it.
3. **Serve v3 through the seam.** `rank_vocabulary` still loads v2; v3 has an
   inference path, gates and tests but is not wired to the device.
4. Neither v2 nor v3 has been evaluated against a single real AAC user. Every
   number here is model-vs-model, on a synthetic corpus, scored by instruments
   built from the same embedder the models use.
