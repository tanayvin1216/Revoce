# Stage-4 Review — Phase 5 Competition-Grade ML Push

Reviewer: Opus 4.8 (Stage-4 lead). Scope: `git diff 50f4f09..HEAD` (three worktree
merges + retrain + fresh IRs). Read-only review; all probes run with
`VOICELINK_PERCEPTION=0` from `mac/` against the real retrained 94-word student and
the real held-out split (n=84, scored n=71).

## Verdict

**SHIP-AFTER-FIXES** — with **C1 as a mandatory blocker before merge to main.**

The three tracks are individually well-built and the retrain is honest, but the
headline claim ("fixed the useless student boards") is true only inside the *eval
harness*: the production **seam still serves the raw-topk hedge/generic wall**. Merging
a report that advertises hedge@7=0.012 on top of a product that still serves a
no-answer board is a competition-integrity problem. Fix C1 + the two gate adjudications,
re-run, then ship.

---

## Adjudication 1 — the sfp ≤0.10 gate

**Pick (i): make it relative-to-teacher** — gate `student_sfp ≤ teacher_sfp` (same
full-top-7 definition, same core-pinning), mirroring exactly how ① already gates
distinct-functions at ≥90% of teacher. Justification: the ≤0.10 absolute bar was set
from a wrong estimate (plan assumed teacher ~0.03; measured teacher = 0.1684), and
~0.048 of every board's sfp is a fixed artifact of the core-pinning policy
("yes" and "wait" both classify `comment`, 1/21 = 0.048) that the model cannot control —
gating the model on a number it does not own is the same category error the
investigation exposed for pinned-core coverage. Option (ii) dynamic-only is
statistically dead (3 dynamic words in top-7 ⇒ minimum non-zero sfp is 1/3 = 0.33, so a
≤0.10 bar is unreachable by construction); option (iii) re-splitting the taxonomy
re-litigates a Phase-3 decision with broad blast radius and Phase-3 tests to keep green.

**Exact change** (`mac/evaluation/phase4/evaluate.py`):
- line 51: delete `SFP_GATE = 0.10` (or repurpose as an epsilon).
- lines 201-203 `diversity_pass`: replace `and ssfp_mean <= SFP_GATE` with
  `and ssfp_mean <= tsfp_mean` where `tsfp_mean = float(np.mean(t_sfp))`.
- report lines 315-317 + `need` string 218-220: state the bar as "≤ teacher sfp".

Result at current numbers: student 0.1508 ≤ teacher 0.1684 → **PASS**, and a genuine
hedge wall (which pushes student well above teacher) still FAILs.

## Adjudication 2 — the dynamic coverage-of-intent regression (0.563 vs teacher 0.845)

**Diagnosis (proven by ablation, not argued): the primary cause is (b) — the RC-3-lite
transcript/scene candidate union injects question-echo nouns that displace the answer
vocab**, amplified by Track-1's relevance-first answer reservation (its Drift 4), which
sorts by cosine-to-query and therefore *prefers* tokens lifted verbatim from the query.
Held-out ablations (n=71):

| variant | dynamic coverage |
|---|---|
| baseline (as shipped) | **0.563** |
| reserve from vocab-only, union kept in fill | 0.676 |
| union minus question-echo tokens | 0.648 |
| **no context union at all** | **0.803** |

Removing the union alone recovers +0.240 and lands exactly on the investigation's
raw-topk dynamic number (0.803). The failing boards are unmistakable — for
"Where are we on the budget numbers?" the dynamic slots are `['budget','numbers',
'doorway']`; for "confirm the last four digits" they are `['digits','number','file']`.
These are the *question's* nouns echoed back, never the *answers* coverage scores
against. (c) shrunken vocab is not the driver; (d) is real but secondary — the 0.90 bar
sits **above the teacher's own 0.845**, so even a perfect distiller cannot reach it
(same error as adjudication 1). Track 1 correctly flagged the union as noisy
("gestures/crowded… future work") but under-weighted that it *net-tanks* coverage by 24
points, because the single-token regex (`infer.py:89-93`) can only extract nouns from
the question — it can never assemble the multi-word answer phrases ("a window seat")
that `acceptable_moves` actually contains.

**Minimal ordered fix:**
1. Exclude raw context-extracted tokens from `_reserve_answers` (reserve only from vocab
   candidates the model actually scored) — `infer.py:136-151`. [0.563 → 0.676]
2. Drop question-echo tokens (any extracted term appearing in the transcript) from the
   pool entirely — they can never be an answer — `infer.py:96-112`. [→ ~0.72]
3. Given single-token extraction provides *no measured coverage benefit* and −24pts,
   **disable the union** (or gate it behind a real multi-word answer-phrase extractor)
   until that extractor exists. [→ 0.803]
4. Re-gate coverage relative-to-teacher: `student_dynamic ≥ 0.90 × teacher_dynamic`
   (`evaluate.py:239`), consistent with ①. After #3: 0.803 / 0.845 = 0.95 → **PASS**.

## dynamic_overlap ≥0.45 — demote to report-only: YES

The design already declares coverage-of-intent the real bar and calls dynamic_overlap
"word-parroting" in the report prose (`evaluate.py:346-347`), and the verdict rule
(`wrapper_dead = ① and ③`) does not include ②. But it is still emitted with a
pass/FAIL gate (`evaluate.py:226`) that now reads 0.056 vs 0.45 — a scary red herring
that contradicts the doc's own thesis, and it collapsed *because* the student now
legitimately diverges (context-echo words + MMR spread). Demote it to a reported
diagnostic exactly like `full_overlap_top7` already is: drop the `pass`/`need` keys or
relabel as "diagnostic, not gated".

---

## Findings

### CRITICAL

**C1 — the useless-board fix is NOT in the production seam; eval measures a different
board builder than the seam serves.**
`mac/rank_vocabulary.py:661` — `_student_board` still calls
`model.board_indices(feat, k_dyn)` (raw topk), while `evaluate.py:69` imports
`infer.student_board` (MMR + answer-aware). They have diverged. Measured live on the
exact investigation scenario ("Where would you like to sit today?", crowded-bus scene):

```
SEAM (rank_vocabulary, strategy="student"): ['yes','no','I need help','wait',
    "i don't know",'i need to use the restroom.','no, thanks.','yes, i am',
    'yes, please','more']            <-- Mode-A hedge/generic wall, NO seat answer
EVAL (infer.student_board, MMR):     ['yes','no','I need help','wait','seats',
    'crowded','i need to use the restroom.','today','school','yes, please']
```

Consequences: (1) Phase-5 objective #1 ("fix the useless student boards") is unmet in
the shipped product — the device still shows the no-answer board the investigation
opened on. (2) Every headline number (hedge@7 0.012, near-dup 0, dynamic-variety 1.0)
describes `infer.py`, not what the seam serves, so the report does not describe the
product. Root cause is a **plan gap**: Track 1 was forbidden to touch
`rank_vocabulary.py`, Track 3 touched only `_load_student`, and the "Known merge point"
names only `evaluate.py`'s helper — nobody was assigned to rewire the seam. **Prescribed
fix:** in `_student_board`, replace the raw `board_indices` block with a call to
`infer.student_board(model, vocab, scene, transcript, is_q, k, k_dyn, core)` (guard with
the same try/except-ImportError fallback evaluate.py uses; keep the existing `return []`
→ diverse-fallback on exception). This is orchestrator-level integration by design, not
a Track boundary violation. Note the OpenVINO path (`OpenVinoStudent`) exposes the same
logits interface, so infer.py works over either backend unchanged.

### HIGH

**H1 — dynamic coverage regressed 0.676 (old checkpoint) → 0.563 (new), FAIL.** See
Adjudication 2. `infer.py:96-133` (context union) + `infer.py:136-151` (reservation).
Ablation-proven; fix ordered above. The Phase-5 changes made the *gated* metric worse,
not better — this must be fixed before the report can claim a quality win.

**H2 — coverage gate 0.90 is set above the teacher's own 0.845.** `evaluate.py:50`. A
distilled student cannot be expected to exceed its teacher on coverage; the absolute bar
is structurally unreachable. Convert to relative-to-teacher (Adjudication 2, fix #4).

**H3 — sfp ≤0.10 gate fails the teacher itself (0.1684).** `evaluate.py:51,201-203`.
~0.048 is an unavoidable core-pinning artifact. Convert to relative-to-teacher
(Adjudication 1).

### MEDIUM

**M1 — dynamic_overlap ≥0.45 still gated despite the design demoting it.**
`evaluate.py:45,222-228`. Reads 0.056 vs 0.45 — a misleading FAIL that contradicts the
report's own thesis. Demote to report-only.

**M2 — transformers pin is irreconcilable across export and perception.**
`impl-track3.md` + `mac/requirements-intel.txt`: `optimum-intel 2.0.0` pins
`transformers<5.1`, `mlx-vlm 0.6.3` needs `≥5.5`. One env cannot run both the
optimum-cli export and the perception VLM. Documented and out of the serving path (the
OV runtime is transformers-free), but it is a real reproducibility hazard for anyone
re-running convert/quantize on a machine with perception installed. Keep the export
tooling in a separate venv and say so in `requirements-intel.txt`, not only in the log.

**M3 — the RC-3-lite union extracts single tokens only, so it cannot surface the
multi-word answers it was built for.** `infer.py:89-93` regex yields lone nouns
("window","seat"), never "a window seat"; those lone nouns then match the *question*,
not the answer set. As built the mechanism adds noise with no measured coverage upside
(see Adjudication 2 table). Either replace with a noun-phrase extractor or disable.

### LOW

**L1 — two divergent hedge lists.** `metrics3.HEDGE_PHRASES` (includes "maybe",
"no idea", "i don't care", "i need more time") vs `infer._HEDGE_PHRASES` (includes
"unsure", "i don't know yet"). Different membership for the same concept invites drift;
consider one shared source.

**L2 — PHASE4_REPORT.md verdict prose is auto-overwritten by evaluate.py** (Track 2
flagged). Any hand-curated verdict note is lost on the next run; keep curated prose out
of the generated file.

**L3 — function length** (`ecc` <50-line guideline): `held_out_eval` (~55 lines) and to
a lesser extent `dedup_vocab` / `student_dynamic_words` exceed the budget. Non-blocking
for ML code but `held_out_eval` would read better split into per-metric accumulators.

---

## Clean (keep)

- **Vocab dedup (single-link components, Track-1 Drift 1)** — justified against the plan's
  greedy phrasing by the measured cosine graph ("i don't know"~"i am not sure" = 0.784
  bridges only transitively); collapses 116→94, exactly the investigation's 22 variants,
  hedges → 1, with a canonical map that redirects label mass in `multihot`. Correct.
- **NNCF INT8 last-layer IgnoredScope + f32 precision hint (Track-3 drifts)** — standard,
  well-measured (top-6 overlap 0.728 → 0.954; ARM f16 drift 1.1e-2 → 1.4e-6). Sound.
- **Fresh IRs verified**: both `student_fp32.xml` and `student_int8.xml` have output
  dim 94 (match the retrained vocab); gitignored artifacts (`vocab.json`, `student.pt`,
  `canonical.json`) are untracked, only the small IRs committed — matches the plan.
- **Word-boundary coverage fix** (`coverage_metrics._phrase_in`) correctly kills the
  "no"→"now" leak while preserving multi-word/punctuation-edged matches. Signature kept.
- **Seam boundary itself is intact**: wearable/web/UI still reach models only through
  `rank_vocabulary`; the OpenVINO backend flag round-trips and `_student_board` still
  returns `[]` on failure → diverse fallback (user never stranded). (The problem in C1 is
  that the *fix* isn't in the seam, not that anything new bypasses it.)
- **Guardrails**: core words pinned first (`assemble_board`), audio path untouched,
  vocabulary-tap logging untouched, all-local inference preserved.

---

## Ordered fix list (do in this order, then re-run `evaluate.py` and regenerate the report)

1. **[C1, blocker]** Wire `infer.student_board` into `rank_vocabulary._student_board`
   so the seam serves the MMR/answer-aware board (try/except fallback; `[]`→diverse on
   error). Verify the seam board for the seating scenario now carries a concrete answer.
2. **[H1/M3]** Keep raw context tokens out of `_reserve_answers`; drop question-echo
   tokens; disable the single-token union pending a multi-word extractor. Target dynamic
   coverage ≈ 0.80.
3. **[H3, adj-1]** sfp gate → `student_sfp ≤ teacher_sfp` (`evaluate.py:201-203,51`).
4. **[H2, adj-2 #4]** coverage gate → `student_dynamic ≥ 0.90 × teacher_dynamic`
   (`evaluate.py:239,50`).
5. **[M1]** Demote `dynamic_overlap` to report-only (`evaluate.py:222-228`).
6. Re-run `VOICELINK_PERCEPTION=0 python3 evaluation/phase4/evaluate.py`; confirm ①/④
   PASS on the honest relative gates; regenerate PHASE4_REPORT.md; then FINAL_REVIEW.
7. **[M2]** Move export tooling to a dedicated venv; note the transformers conflict in
   `requirements-intel.txt`.
