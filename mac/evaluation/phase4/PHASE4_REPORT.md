# Phase-4 distillation v2 — evaluation report

Generated: 2026-06-30 13:25:24  |  student rows=580 vocab=132 config_hash=2cc4de961af3

## The three numbers (① and ② on the HELD-OUT split, n=87)

| # | metric | student | teacher | bar | verdict |
|---|--------|---------|---------|-----|---------|
| ① | distinct-functions@7 (held-out) | 4.287 | 4.552 | ≥90% retention | PASS (ret 0.942) |
| ① | near-duplicate rate (held-out) | 0.0115 | 0.0005 | ≤0.08 | PASS |
| ① | board-variety (degeneracy guard) | 49 distinct of 87 (variety 0.563) | — | ≥0.5 | PASS |
| ① | dynamic-variety (distinct triples) | 0.563 | — | — | — |
| ① | **combined** | — | — | all three above | **PASS** |
| ② | dynamic_overlap (gate) | 0.176 | — | ≥0.45 | FAIL (n=87) |
| ② | full_overlap_top7 (continuity) | 0.624 | — | — | — |
| ③ | speedup | 7.915 ms | 6.23 s | ≥10× | PASS (787.4×) |

## Verdict: literal rule says "dead" — honest read: fast + diverse, NOT a faithful policy match

The literal rule (① AND ③) passes. We do **not** claim the wrapper critique is dead.

> A small student that is **94% of teacher diversity, board-variety 0.563, ~787× faster** —
> but `dynamic_overlap = 0.176`, far below 0.45.

### Two levers tried, neither moved agreement — so we stop iterating (per the design's rule)
| run | data | input | dynamic_overlap (n) |
|-----|------|-------|---------------------|
| v2  | 221 rows | pooled "scene\|\|transcript" (384-d) | 0.232 (n=33) |
| v2.1| 580 rows | pooled (384-d) | 0.165 (n=87) |
| v2.2| 580 rows | **richer: separate scene+transcript+is_question (769-d)** | 0.176 (n=87) |

More data made it *worse*; richer input did *not* recover it. And a **semantic** check (does a
student dynamic word land within cos≥0.5 of a teacher dynamic word?) is only **~0.19** — so this
is **not** a harsh-word-match artifact; the student genuinely picks different dynamic words.

### Why (the real diagnosis)
The teacher's board for a context is **ONE diverse SAMPLE** from a large space of equally-valid
diverse boards (LLM pool + MMR over many good options). A deterministic student trained with BCE
converges to the *average* diverse board, which cannot match any *specific* teacher sample —
so agreement-with-one-teacher-sample is **inherently low and is the wrong target**. This is a
one-to-many distillation property, not a capacity or input-richness problem; more architecture
won't fix it.

### What is solid vs not
- **Solid:** the distillation *path* (ONNX export → train in <1s → eval), a tiny **~800× faster**
  student that is genuinely diverse and not collapsed. Retraining is ~7s on the dataset we hold.
- **Not proven (and not the right bar):** exact reproduction of the teacher's specific picks.

### The right next step (reframe, not more iteration)
Validate the student on **coverage-of-intent** (does its board contain an acceptable communicative
move for the scenario — the Phase-3 metric, against `acceptable_moves`) and ultimately on **human/
pilot taps**, NOT on word-match to one teacher board. That measures whether the student is *good*,
which is the actual product question.

## Honesty note
v2 is validated against TEACHER behaviour on SYNTHETIC scenarios; the teacher is itself an LLM (Qwen) over curated/synthetic contexts. A student that passes here may be imitating the teacher imitating GPT. Human/pilot validation on consented interaction logs is the next step and the number that ultimately matters.
