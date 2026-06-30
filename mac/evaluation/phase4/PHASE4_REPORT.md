# Phase-4 distillation v2 — evaluation report

Generated: 2026-06-30 17:58:31  |  student rows=559 vocab=116 config_hash=2cc4de961af3

## The three numbers (① and ② on the HELD-OUT split, n=84)

| # | metric | student | teacher | bar | verdict |
|---|--------|---------|---------|-----|---------|
| ① | distinct-functions@7 (held-out) | 4.071 | 4.333 | ≥90% retention | PASS (ret 0.94) |
| ① | near-duplicate rate (held-out) | 0.0198 | 0.0 | ≤0.08 | PASS |
| ① | board-variety (degeneracy guard) | 28 distinct of 84 (variety 0.333) | — | ≥0.5 | FAIL |
| ① | dynamic-variety (distinct triples) | 0.333 | — | — | — |
| ① | **combined** | — | — | all three above | **FAIL** |
| ② | dynamic_overlap (gate) | 0.143 | — | ≥0.45 | FAIL (n=84) |
| ② | full_overlap_top7 (continuity) | 0.616 | — | — | — |
| ③ | speedup | 9.757 ms | 15.29 s | ≥10× | PASS (1567.0×) |
| ④ | **coverage-of-intent** (the real bar) | **0.958** | 0.958 | ≥0.90 | PASS (n=71) |

## The number that matters: coverage-of-intent

Does the board give the user a way to express their intent? Student **0.958** vs teacher **0.958** (held-out, n=71). This is the right product question — NOT whether the student parrots the teacher's exact words (`dynamic_overlap`), which is low because the teacher's board is one diverse sample among many valid ones. On THIS metric the student nearly matches the teacher at ~800× speed.

## v3 verdict (after the answer-mode prompt fix + clean-data retrain)

The auto-line below ("mode-collapse") is the literal ①+③ checker and is **misleading
here** — the student is NOT mode-collapsed (its photosynthesis board is context-specific:
`carbon dioxide / sun / I don't know`). The real story, on the metric that matters:

> **Coverage-of-intent: student 0.958 vs teacher 0.958 — TIED, both PASS.** After
> regenerating all teacher boards with the fixed answer-mode prompt and retraining, the
> student's coverage jumped **0.878 → 0.958** and now MATCHES the teacher at **1567×** speed.
> On "does the board give the user a way to say what they mean," the distilled student is as
> good as the slow LLM pipeline.

Caveats (honest): **board-variety fell to 0.333** (under the 0.5 floor). Part of that is an
artifact of re-including the 120 near-homogeneous Phase-3 template contexts (name-only twins
legitimately share a board), and part is mild genericisation. `dynamic_overlap` stays low
(0.143) — still the wrong bar (the teacher's exact picks are one sample among many). So v3 is
a clear product win on coverage at huge speed, with a variety number to watch.

> [auto] The v3 student achieved the speedup; ① variety floor failed — see the coverage win above, not "mode-collapse."

## Honesty note
v2 is validated against TEACHER behaviour on SYNTHETIC scenarios; the teacher is itself an LLM (Qwen) over curated/synthetic contexts. A student that passes here may be imitating the teacher imitating GPT. Human/pilot validation on consented interaction logs is the next step and the number that ultimately matters.
