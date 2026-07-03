# Phase-4 distillation v2 — evaluation report

Generated: 2026-07-02 23:57:55  |  student rows=559 vocab=116 config_hash=2cc4de961af3

## The three numbers (① and ② on the HELD-OUT split, n=84)

| # | metric | student | teacher | bar | verdict |
|---|--------|---------|---------|-----|---------|
| ① | distinct-functions@7 (held-out) | 4.071 | 4.333 | ≥90% retention | PASS (ret 0.94) |
| ① | near-duplicate rate (held-out) | 0.0198 | 0.0 | ≤0.08 | PASS |
| ① | same-function-pair rate (held-out) | 0.1689 | 0.1684 | ≤0.1 | FAIL |
| ① | board-variety (degeneracy guard) | 28 distinct of 84 (variety 0.333) | — | ≥0.5 | FAIL |
| ① | dynamic-variety (distinct triples) | 0.333 | — | — | — |
| ① | **combined** | — | — | all three above | **FAIL** |
| ② | dynamic_overlap (gate) | 0.143 | — | ≥0.45 | FAIL (n=84) |
| ② | full_overlap_top7 (continuity) | 0.616 | — | — | — |
| ③ | speedup | 8.284 ms | 7.39 s | ≥10× | PASS (892.4×) |
| ④ | coverage-of-intent, full board (reported, NOT gated) | 0.944 | 0.944 | — | — (n=71) |
| ④ | **coverage-of-intent, DYNAMIC (the real bar)** | **0.676** | 0.845 | ≥0.9 | FAIL (n=71) |

## The number that matters: DYNAMIC coverage-of-intent

Does the board the MODEL built give the user a way to express their intent? Student **0.676** vs teacher **0.845** (held-out, n=71; non-core words in the top-7, scored against non-hedge acceptable moves). The full-board number (0.944) is reported for continuity but is inflated by the 4 pinned core words self-certifying (INVESTIGATION_useless_boards.md §3). Coverage — not `dynamic_overlap` word-parroting — is the right product question; the dynamic slice is the honest version of it.

## Verdict (literal ①+③ rule): NOT crossed

Rule: dead IFF ① diversity retention AND ③ speedup both pass; ② is supporting. We read ④ coverage-of-intent as the honest 'is the board good' signal.

> The v2 student achieved the speedup but did NOT retain teacher diversity (mode-collapse). Iterating on diversity loss / more teacher data. Not a win.

## Honesty note
v2 is validated against TEACHER behaviour on SYNTHETIC scenarios; the teacher is itself an LLM (Qwen) over curated/synthetic contexts. A student that passes here may be imitating the teacher imitating GPT. Human/pilot validation on consented interaction logs is the next step and the number that ultimately matters.
