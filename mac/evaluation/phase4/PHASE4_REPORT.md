# Phase-4 distillation v2 — evaluation report

Generated: 2026-07-22 15:24:52  |  student rows=559 vocab=95 config_hash=d6ce2e56d50b

## The three numbers (① and ② on the HELD-OUT split, n=84)

| # | metric | student | teacher | bar | verdict |
|---|--------|---------|---------|-----|---------|
| ① | distinct-functions@7 (held-out) | 4.071 | 4.298 | ≥90% retention | PASS (ret 0.947) |
| ① | near-duplicate rate (held-out) | 0.0 | 0.0006 | ≤0.08 | PASS |
| ① | same-function-pair rate (held-out) | 0.1899 | 0.1735 | ≤ teacher | FAIL |
| ① | board-variety (degeneracy guard) | 80 distinct of 84 (variety 0.952) | — | ≥0.5 | PASS |
| ① | dynamic-variety (distinct triples) | 0.952 | — | — | — |
| ① | **combined** | — | — | all three above | **FAIL** |
| ② | dynamic_overlap (report-only) | 0.067 | — | — | — (n=84) |
| ② | full_overlap_top7 (continuity) | 0.594 | — | — | — |
| ③ | speedup | 25.968 ms | None s | ≥10× | FAIL (None×) |
| ④ | coverage-of-intent, full board (reported, NOT gated) | 0.887 | 0.944 | — | — (n=71) |
| ④ | **coverage-of-intent, DYNAMIC (the real bar)** | **0.803** | 0.873 | ≥90% of teacher | PASS (n=71) |

## The number that matters: DYNAMIC coverage-of-intent

Does the board the MODEL built give the user a way to express their intent? Student **0.803** vs teacher **0.873** (held-out, n=71; non-core words in the top-7, scored against non-hedge acceptable moves). The full-board number (0.887) is reported for continuity but is inflated by the 4 pinned core words self-certifying (INVESTIGATION_useless_boards.md §3). Coverage — not `dynamic_overlap` word-parroting — is the right product question; the dynamic slice is the honest version of it.

## Verdict (literal ①+③ rule): NOT crossed

Rule: dead IFF ① diversity retention AND ③ speedup both pass; ② is supporting. We read ④ coverage-of-intent as the honest 'is the board good' signal.

> v2 did not clear the bars; see numbers above. Honest finding, iterate.

## Honesty note
v2 is validated against TEACHER behaviour on SYNTHETIC scenarios; the teacher is itself an LLM (Qwen) over curated/synthetic contexts. A student that passes here may be imitating the teacher imitating GPT. Human/pilot validation on consented interaction logs is the next step and the number that ultimately matters.
