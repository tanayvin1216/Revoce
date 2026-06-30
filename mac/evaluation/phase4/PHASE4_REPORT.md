# Phase-4 distillation v1 — evaluation report

Generated: 2026-06-30 05:26:49  |  student rows=245 vocab=157 config_hash=aca31bc68c1c

## The three numbers (① and ② on the HELD-OUT split, n=37)

| # | metric | student | teacher | bar | verdict |
|---|--------|---------|---------|-----|---------|
| ① | distinct-functions@7 (held-out) | 4.649 | 4.838 | ≥90% retention | PASS (ret 0.961) |
| ① | near-duplicate rate (held-out) | 0.0013 | 0.0 | ≤0.08 | PASS |
| ① | board-variety (degeneracy guard) | 17 distinct of 37 (variety 0.459) | — | ≥0.5 | FAIL |
| ① | **combined** | — | — | all three above | **FAIL** |
| ② | top-7 set-overlap (held-out) | 0.637 | — | ≥0.70 | FAIL (n=37) |
| ③ | speedup | 9.436 ms | 8.01 s | ≥10× | PASS (848.8×) |

## Verdict: NOT crossed — iterate

Rule: dead IFF ① diversity retention AND ③ speedup both pass; ② is supporting.

> The v1 student achieved the speedup but did NOT retain teacher diversity (mode-collapse). Iterating on diversity loss / more teacher data. Not a win.

## Reading the numbers (and a correction)
- **① and ② are HELD-OUT** (the test split, n=37) — NOT the in-sample Phase-3 set. An
  earlier run of this report measured ① over the full Phase-3 scenario set, ~85% of which
  the student trained on, and declared "the wrapper critique is dead." That was in-sample
  and is **retracted**: on held-out contexts the verdict is **NOT crossed**.
- **What the student DOES retain**: distinct-functions 96.1% of teacher, near-dup 0.0013.
  So it is **not** mode-collapsed on the function axis (the bad first v1, trained on the
  homogeneous templates, emitted ONE board for every context — variety 0.008; this is far
  from that).
- **Where it falls short**: board-variety 0.459 (just under the 0.5 anti-collapse floor)
  and agreement 0.637 (<0.70). Caveat: the held-out set mixes Phase-3 name-only variants
  (which legitimately SHARE a board) with Phase-1 turns, so some shared boards are correct,
  not collapse — the floor is deliberately conservative and left unchanged rather than
  tuned to pass.
- **v2 lever**: more VARIED teacher contexts (the Phase-4 gen extends past 245). Variety and
  agreement are the two numbers that should move; distinct-functions and speedup are in hand.

## Honesty note
v1 is validated against TEACHER behaviour on SYNTHETIC scenarios; the teacher is itself an LLM (Qwen) over curated/synthetic contexts. A student that passes here may be imitating the teacher imitating GPT. Human/pilot validation on consented interaction logs is the next step and the number that ultimately matters.

## Next steps (v2)
- More teacher data (the Phase-4 gen extends past 245) to lift board-variety past 0.5 and agreement past 0.70.
- Phase 5: convert student.onnx → OpenVINO IR, INT8 via NNCF, re-check the bars.
