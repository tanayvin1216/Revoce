# Phase 5 — Edge-Case Findings (Stage 5)

One Sonnet tester wrote `mac/tests/test_phase5_edges.py` (originally 33 cases, 9
classes: empty inputs, non-question mode, hedge-saturated vocab, unicode hedge
variants, stale-artifact guards, seam fallback, seam contract, dedup edges,
OpenVINO runtime). Three independent reviewers then walked every case.

## Verdicts

**Haiku (qa-checker, fresh context):** PASS — all 21 test functions SOUND; no
vacuous assertions; monkeypatch scoping verified correct; negative controls
present (consistent-checkpoint, non-question, non-hedge rejection).

**Sonnet (code-reviewer):** 20/21 KEEPS-YOU-SAFE, 1 MISLEADING —
`test_seam_student_strategy_contract` could pass via the diverse-fallback path
while the student path was silently broken (its assertions hold for both
paths). Also flagged test 4's docstring (the hedge bias never competes for
pool inclusion since the pool admits the whole 10-word vocab) and listed 5
missing cases.

**Codex (fresh non-interactive review):** 3 findings — (P1) module-level
`pytest.importorskip("openvino")` skipped the ENTIRE module on machines
without OpenVINO, silently deleting all hedge/dedup/guard coverage; (P2) the
seam contract test cannot distinguish student path from fallback (convergent
with Sonnet); (P2) `_load_student_openvino`'s vocab-sha/size guards had zero
coverage.

## Fixes applied (same session, commit follows)

1. importorskip made lazy/per-test — the 33 core cases now run everywhere.
2. New `test_seam_student_strategy_does_not_use_diverse_fallback`: monkeypatches
   `generate_pool`/`select_board` to raise-if-called, proving the student path
   built the board.
3. New `test_openvino_loader_rejects_missing_or_stale_sidecar`: covers the
   same-size-reorder defense (missing sidecar AND wrong-sha sidecar both
   rejected loudly).
4. New `test_infer_student_board_end_to_end_with_openvino_model`: drives the
   full MMR selection through an OpenVinoStudent (the numpy branch), not just
   raw `__call__`.

Post-fix suite: **116 passed, 2 xfailed** (the xfails are the two honest RC-3
out-of-vocab seating probes, documented in test_phase4.py).

## Accepted gaps (documented, not implemented tonight)

- Singleton/backend-switch reuse: `_student_singleton` is never invalidated if
  `VOICELINK_STUDENT_BACKEND` changes mid-process; no concurrent first-call
  test. Low risk (env var is read once at startup in practice).
- `k_dyn` 0/1 extremes and `len(vocab) == k_dyn` boundary; `assemble_board`
  truncation when core + dynamic > k.
- Test 4's docstring overstates the bias mechanism (Sonnet note) — the cap is
  exercised through real embedding semantics instead; acceptable but worth a
  deterministic reconstruction later.
- OpenVINO tests still skip where no IR/openvino exists (intentional).
