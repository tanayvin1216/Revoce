# Phase-5 edge-case tests — implementation log

Tester subagent, run 2026-07-03. Scope: NEW `mac/tests/test_phase5_edges.py` only,
adversarial/edge-case coverage of the Phase-5 surface (infer.py, hedges.py,
rank_vocabulary._student_board/_load_student_torch/_load_student_openvino,
data.dedup_vocab, intel.OpenVinoStudent) per the review-opus.md findings and
classic AAC-safety properties.

## Worktree sync (pre-requisite)

This worktree (`worktree-agent-aa5a3afdb7ebce239`) was checked out at `a4bed1a`
(tip of `main`, phase-1 era) — it did not yet contain the Phase-5 surface
(`mac/tests/`, `mac/intel/`, `mac/evaluation/phase4/infer.py`, `mac/evaluation/hedges.py`,
`mac/conftest.py`, etc). `phase4-distillation` (the branch the task's context files
live on) is a pure 34-commit fast-forward ahead of this worktree's HEAD with no
divergence, so:

    git merge --ff-only phase4-distillation

was run to bring the worktree's own branch up to that tip before writing any
tests. This is a pointer move to already-committed work (not new implementation
code), required so the tests have a production surface to import and exercise.
Confirmed clean fast-forward (no conflicts, no rewrite of history).

Gitignored artifacts (`student.pt`, `vocab.json`, `canonical.json`, `splits.json`)
are untracked and therefore NOT present in this fresh worktree checkout — same as
any fresh clone. The OpenVINO IR artifacts (`student_fp32.xml/.bin`,
`student_int8.xml/.bin`) ARE git-tracked (per the Track-3 plan) and are present.
openvino 2026.2.1 is installed in this environment.

## Edge cases written (`mac/tests/test_phase5_edges.py`)

1. **Empty/None/whitespace scene+transcript** through `infer.student_board` —
   parametrized over `("","")`, `(None,None)`, whitespace-only, and mixed
   None/real-text combos; both `is_question=False` and `=True` paths. Asserts
   never raises, always exactly k=10, core pinned first 4.
2. **Non-question context (`is_question=False`)** — a spy on `infer._reserve_answers`
   proves the answer-quota function is never invoked (structural, not just
   output-based); a second test proves the hedge cap (`HEDGE_CAP=1`) still holds
   even when hedges dominate the raw bias ranking, with no reservation to help it.
3. **Hedge-only vocab (pathological)** — an 8-word vocab that is ALL hedges:
   proves the backfill loop (`infer.py:187-191`) tops the board back up to
   `k_dyn=6` despite the cap wanting to reject every candidate after the first;
   also a smaller-than-`k_dyn` all-hedge vocab (pool exhaustion, no crash).
4. **Unicode/apostrophe/punctuation variants** through `hedges.normalize`/`is_hedge`
   — curly right single quote (’), trailing `!?`, surrounding whitespace, shouted
   case, curly-quote+ellipsis, trailing comma — all recognized as the same hedge;
   plus a negative set (superset phrase, unrelated phrase, empty string, phrase
   missing the subject) that must NOT be flagged, to prove `is_hedge` isn't just
   matching on a loose substring.
5. **Stale-artifact guards on `rank_vocabulary._load_student_torch`** — three
   `tmp_path`-only tests (never touching real artifacts): (a) checkpoint
   `vocab_size` disagrees with `vocab.json` length → `RuntimeError` matching
   "vocab.json has 3 words but"; (b) `vocab_sha` in the checkpoint disagrees with
   the freshly-hashed `vocab.json` → `RuntimeError` matching "vocab.json content
   drifted"; (c) a negative control — a genuinely consistent checkpoint/vocab
   pair loads cleanly, proving (a)/(b) test the guard and not some unrelated
   load failure.
6. **`rank_vocabulary._student_board` failure fallback** — (a) `_load_student`
   monkeypatched to raise → `_student_board(...)` returns `[]` (never propagates,
   so the caller degrades to the diverse strategy); (b) failure surfacing from
   inside `infer.student_board` itself (not just the loader) is caught by the
   same `try/except` and also returns `[]`.
7. **Seam contract under `strategy="student"`** with the REAL checkpoint —
   `rank_vocabulary.rank_vocabulary(..., strategy="student")` returns exactly
   `BOARD_SIZE=10` well-formed `RankedWord` dicts (`{"word", "symbol"}` keys,
   non-empty strings), core pinned first, board never empty. Guarded by
   `skipif(not student.pt.exists())` — SKIPPED in this checkout (artifact
   untracked/gitignored, matches the pre-existing pattern in `test_phase4.py`).
8. **`dedup_vocab` edge cases** — empty list (`[] -> ([], {})`), single word
   (no embed call needed, short-circuits), three identical-word copies collapsing
   to one survivor via a monkeypatched deterministic `evaluation.embed.embed`,
   and the oversized-non-hedge-component `ValueError` — 6 near-identical
   NON-hedge phrases forced to cosine=1.0 via a controlled fake embed (no real
   MiniLM dependency, fully deterministic) correctly trips the `>4` oversized
   guard in `data.dedup_vocab`.
9. **OpenVINO path** (`pytest.importorskip("openvino")`, further skipped if the
   FP32 IR artifact is missing) — `OpenVinoStudent` on the real IR: a
   zeros-embedding input returns a well-formed `(1, vocab_size)` finite-logit
   array; a wrong-dtype `float64` batch input is coerced to `float32` internally
   (`np.asarray(..., dtype=np.float32)`) and still returns well-formed,
   finite logits at the right shape — both ran against the REAL committed IR,
   not a mock.

## Run results

`tests/test_phase5_edges.py` alone (from `mac/`, `VOICELINK_PERCEPTION=0`):

```
32 passed, 1 skipped, 2 warnings in 4.80s
```
(the 1 skip is `test_seam_student_strategy_contract` — student.pt absent in this
fresh worktree, expected.)

Full suite (`pytest tests/`, same env):

```
109 passed, 6 skipped, 5 warnings in 7.83s
```

The 6 skips are all pre-existing patterns for artifacts this fresh worktree
doesn't have checked in (gitignored `student.pt` and the OpenVINO-backend seam
test that depends on it) — none are new skips introduced by this test file, and
none are silently-skipped tests without a reason string.

## Broken/flaky tests

None. No flakiness observed across the run (single run, deterministic fixtures/
monkeypatches throughout except the two OpenVINO tests and the real-checkpoint
seam test which depend on committed/generated artifacts, both gated by explicit
skip conditions rather than randomness).

## EXPOSES BUG flags

**None.** Every edge case in the target list passed against the current
production code on first write — no `xfail(strict=False)` markers were needed.
Specifically:
- The C1 seam-wiring fix (`_student_board` calling `infer.student_board`) holds
  under empty/None/whitespace inputs.
- The hedge cap + backfill (`infer.py:178-191`) correctly handles the
  pathological all-hedge vocab without starving the board.
- The stale-artifact guards (`_assert_vocab_matches`, `vocab_sha` check) fire
  exactly as designed.
- The `_student_board` try/except correctly degrades to `[]` on both
  loader-failure and inference-failure.
- `dedup_vocab`'s oversized-non-hedge-component guard fires correctly under a
  fully controlled (non-MiniLM) embedding.
- `OpenVinoStudent` handles zeros and wrong-dtype input cleanly.

## Verdict

PASS.
