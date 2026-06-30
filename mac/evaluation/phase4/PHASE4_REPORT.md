# Phase-4 distillation v2 — evaluation report

Generated: 2026-06-30 13:09:00  |  student rows=580 vocab=132 config_hash=6b90f08bac04

## The three numbers (① and ② on the HELD-OUT split, n=87)

| # | metric | student | teacher | bar | verdict |
|---|--------|---------|---------|-----|---------|
| ① | distinct-functions@7 (held-out) | 4.368 | 4.552 | ≥90% retention | PASS (ret 0.96) |
| ① | near-duplicate rate (held-out) | 0.0099 | 0.0005 | ≤0.08 | PASS |
| ① | board-variety (degeneracy guard) | 49 distinct of 87 (variety 0.563) | — | ≥0.5 | PASS |
| ① | dynamic-variety (distinct triples) | 0.563 | — | — | — |
| ① | **combined** | — | — | all three above | **PASS** |
| ② | dynamic_overlap (gate) | 0.165 | — | ≥0.45 | FAIL (n=87) |
| ② | full_overlap_top7 (continuity) | 0.619 | — | — | — |
| ③ | speedup | 7.691 ms | 6.97 s | ≥10× | PASS (905.6×) |

## Verdict: literal rule says "dead" — but the honest read is NOT a faithful distillation

The literal rule (① AND ③) passes, so the auto-checker prints YES. We do **not** claim the
wrapper critique is dead, because the honest agreement metric is failing AND getting worse:

> A fast, genuinely DIVERSE student (96% function retention, board-variety 0.563, 905× faster)
> — but `dynamic_overlap = 0.165`, far below 0.45, and it FELL from 0.232 (v2, n=33) to 0.165
> (v2.1, n=87) when we 5×'d the data. More data did not help; it made agreement worse on a
> larger, more reliable held-out set.

**This trips the design's falsification path.** Dynamic agreement degrading as data grows
means the bottleneck is not data volume — the pooled `"<scene> || <transcript>"` MiniLM
vector does not linearly separate the teacher's *specific* dynamic choice. The student
settles into a globally-diverse board policy that is largely context-agnostic in its exact
picks. So:

- **What is real:** the distillation *path* works end-to-end (export-first ONNX → train →
  eval), the student is tiny + ~900× faster, and it is genuinely diverse / not collapsed.
- **What is NOT proven:** that this student reproduces the teacher's *policy*. It does not
  (0.165). A fast model that picks different words than the teacher is not the teacher.

**Next (per the design, not "more data"):** richer input before more capacity — concatenate
SEPARATE scene and transcript embeddings (+ the structured context features) instead of one
pooled vector, or a retrieve-then-rerank head over the dynamic vocab. Then re-measure
`dynamic_overlap`. Capacity growth is explicitly the wrong lever here.

## Honesty note
v2 is validated against TEACHER behaviour on SYNTHETIC scenarios; the teacher is itself an LLM (Qwen) over curated/synthetic contexts. A student that passes here may be imitating the teacher imitating GPT. Human/pilot validation on consented interaction logs is the next step and the number that ultimately matters.
