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

### Honest caveat on board quality

The *selection* is visibly imperfect — "mexican" and "i'd like to play video
games" are poor answers to a seating question, and "i mean dinner" is close to a
duplicate of "dinner". Representability is demonstrated; ranking quality on 559
rows is not yet where v2's was on a 94-word space, which is unsurprising: v3 is
ranking 1959 candidates instead of 94. The phase-3 gates have not been re-run
against v3 and **no v3 number should be quoted as beating v2 until they are.**
That is the next piece of work, not a claim being made here.

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

1. **Run the phase-3 gates against v3.** Until then v3 is unproven on
   distinct-functions, same-function-pair and coverage-of-intent.
2. **Ranking quality at 1959 candidates.** More teacher data, and probably a
   relevance signal for answer-mode, since "who" questions should prefer people.
3. **Serve it through the seam.** `rank_vocabulary` still loads the v2 student;
   v3 has an inference path and tests but is not yet wired to the device.
4. Neither v2 nor v3 has been evaluated against a single real AAC user. Every
   number here is model-vs-model.
