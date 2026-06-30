# Phase-4 distillation v2 — evaluation report

Generated: 2026-06-30 12:00:05  |  student rows=221 vocab=71 config_hash=6b90f08bac04

## The three numbers (① and ② on the HELD-OUT split, n=33)

| # | metric | student | teacher | bar | verdict |
|---|--------|---------|---------|-----|---------|
| ① | distinct-functions@7 (held-out) | 4.576 | 4.727 | ≥90% retention | PASS (ret 0.968) |
| ① | near-duplicate rate (held-out) | 0.0115 | 0.0014 | ≤0.08 | PASS |
| ① | board-variety (degeneracy guard) | 22 distinct of 33 (variety 0.667) | — | ≥0.5 | PASS |
| ① | dynamic-variety (distinct triples) | 0.667 | — | — | — |
| ① | **combined** | — | — | all three above | **PASS** |
| ② | dynamic_overlap (gate) | 0.232 | — | ≥0.45 | FAIL (n=33) |
| ② | full_overlap_top7 (continuity) | 0.623 | — | — | — |
| ③ | speedup | 10.724 ms | 15.39 s | ≥10× | PASS (1434.8×) |

## Verdict: ①+③ pass by the literal rule — but DO NOT over-claim (see caveat)

Rule: dead IFF ① diversity retention AND ③ speedup both pass; ② is supporting.

> We distilled a slow LLM+MMR diverse-ranking pipeline into a small trained model that retains 97% of its communicative diversity at a 1434.8× speedup.

## Caveat — the honest agreement number is short
`dynamic_overlap = 0.232` is well below its 0.45 bar. Stripped of the 4 pinned core words
(which match trivially), the student reproduces only ~23% of the teacher's specific dynamic
choices. So v2 is a **fast, genuinely diverse ranker that makes its OWN context-varied
boards** — not yet a *faithful* distillation of the teacher's exact policy. We do not claim
the wrapper critique is fully dead on this run.

What v2 DID prove: the dynamic-only architecture **fixed the v1 board-variety failure
(0.459 → 0.667)** and lifted true dynamic agreement (~0.15 → 0.232) — on essentially v1's
data (only 11 of the 360 newly-authored scenarios had teacher boards in time for this run).
The remaining gap is agreement/data, not collapse.

**Next: v2.1** — retrain on the full ~600-row corpus (the 360 authored scenarios finish
generating in the background) to push `dynamic_overlap` toward 0.45. If two bounded
iterations don't move it, the design's falsification path applies (pooled MiniLM input may
not linearly separate the dynamic decision → richer input before more capacity).

## Honesty note
v2 is validated against TEACHER behaviour on SYNTHETIC scenarios; the teacher is itself an LLM (Qwen) over curated/synthetic contexts. A student that passes here may be imitating the teacher imitating GPT. Human/pilot validation on consented interaction logs is the next step and the number that ultimately matters.
