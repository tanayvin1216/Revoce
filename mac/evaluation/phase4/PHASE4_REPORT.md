# Phase-4 distillation v2 — evaluation report

Generated: 2026-08-23 16:14:32  |  student rows=559 vocab=95 config_hash=d6ce2e56d50b

## The three numbers (① and ② on the HELD-OUT split, n=84)

| # | metric | student | teacher | bar | verdict |
|---|--------|---------|---------|-----|---------|
| ① | distinct-functions@7 (held-out) | 4.095 | 4.298 | ≥90% retention | PASS (ret 0.953) |
| ① | near-duplicate rate (held-out) | 0.0 | 0.0006 | ≤0.08 | PASS |
| ① | same-function-pair rate (held-out) | 0.1893 | 0.1735 | ≤ teacher | FAIL |
| ① | board-variety (degeneracy guard) | 80 distinct of 84 (variety 0.952) | — | ≥0.5 | PASS |
| ① | dynamic-variety (distinct triples) | 0.952 | — | — | — |
| ① | **combined** | — | — | all three above | **FAIL** |
| ② | dynamic_overlap (report-only) | 0.067 | — | — | — (n=84) |
| ② | full_overlap_top7 (continuity) | 0.594 | — | — | — |
| ③ | speedup | 10.135 ms | 7.05 s | ≥10× | PASS (695.5×) |
| ④ | coverage-of-intent, full board (reported, NOT gated) | 0.873 | 0.944 | — | — (n=71) |
| ④ | **coverage-of-intent, DYNAMIC (the real bar)** | **0.789** | 0.873 | ≥90% of teacher | PASS (n=71) |

## The number that matters: DYNAMIC coverage-of-intent

Does the board the MODEL built give the user a way to express their intent? Student **0.789** vs teacher **0.873** (held-out, n=71; non-core words in the top-7, scored against non-hedge acceptable moves). The full-board number (0.873) is reported for continuity but is inflated by the 4 pinned core words self-certifying (INVESTIGATION_useless_boards.md §3). Coverage — not `dynamic_overlap` word-parroting — is the right product question; the dynamic slice is the honest version of it.

## Verdict (literal ①+③ rule): NOT crossed

Rule: dead IFF ① diversity retention AND ③ speedup both pass; ② is supporting. We read ④ coverage-of-intent as the honest 'is the board good' signal.

> The v2 student achieved the speedup but did NOT retain teacher diversity (mode-collapse). Iterating on diversity loss / more teacher data. Not a win.

## Deictic-question set (informational, NOT gated; n=60)

RC-3's open answer path fires on only 2/84 held-out rows, so its effect is measured on the authored `deictic_scenarios.jsonl` instead (eval_deictic.py). hit@board = any expected concrete answer appears on the board.

| slice | n | open fire | hit@board (student) | hit@board (closed-only) |
|---|---|---|---|---|
| overall | 60 | 0.883 | 0.817 | 0.017 |
| alternative | 8 | 1.000 | 1.000 | 0.000 |
| colour | 6 | 0.833 | 0.833 | 0.000 |
| count | 6 | 0.000 | 0.000 | 0.000 |
| where | 16 | 1.000 | 0.875 | 0.000 |
| which | 12 | 1.000 | 0.833 | 0.000 |
| who | 12 | 1.000 | 1.000 | 0.083 |

## Honesty note
v2 is validated against TEACHER behaviour on SYNTHETIC scenarios; the teacher is itself an LLM (Qwen) over curated/synthetic contexts. A student that passes here may be imitating the teacher imitating GPT. Human/pilot validation on consented interaction logs is the next step and the number that ultimately matters.
