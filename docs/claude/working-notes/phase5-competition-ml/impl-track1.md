# Track 1 — Student quality (RC-1, RC-2, RC-3-lite) — implementation log

Worktree: `.claude/worktrees/agent-a7ef0d9b63c0243df`, branch fast-forwarded to
`phase4-distillation` tip (50f4f09). Gitignored inputs copied from the main working
dir: `teacher_data.jsonl` (600 rows), `phase3/boards.jsonl`, `phase3/scenarios.jsonl`.
Environment: system python3.13 with torch 2.12.0 + sentence-transformers (verified by
import). All commands run from `mac/` with `VOICELINK_PERCEPTION=0`.

## Decision 1 — dedup must be single-link components, not greedy-vs-kept (DRIFT)

The plan says "greedily drop any candidate whose MiniLM cosine to an already-kept
word >= dedup_tau (0.80)". Measured live (MiniLM, worktree):

```
"i don't know"       ~ 'i am not sure'      = 0.784   <- BELOW 0.80
"i don't know"       ~ "i'm not sure"       = 0.841
'i am not sure'      ~ "i'm not sure"       = 0.943
'i am not sure'      ~ 'not sure'           = 0.868
"i'm not sure."      ~ 'i am not sure.'     = 0.960
"i don't know"       ~ "i don't know."      = 0.926
'not sure'           ~ "i don't know."      = 0.633
```

Greedy-vs-kept in frequency order keeps rank-0 "i don't know", then KEEPS rank-1
"i am not sure" (0.784 < 0.80) — two hedge representatives survive, violating the
acceptance criterion ("collapses the 7 known hedge variants to exactly 1").
The investigation's own measurement was *single-link clustering* at cosine >= 0.80
("largest cluster being exactly the 7 hedges"). Single-link connected components at
0.80 DO unite all 7 (the "i'm not sure" hub bridges the don't-know and not-sure
families at 0.841/0.943). Minimal faithful adaptation: `dedup_vocab` builds
connected components of the cosine>=tau graph and keeps the highest-frequency member
of each component; every dropped member maps to its component representative in the
canonical map. Flagged as plan drift.

## Decision 2 — API shape

- `build_vocab` keeps its exact 2-tuple signature (export_onnx.py:79, which Track 1
  does NOT own, calls it). Dedup is a separate `data.dedup_vocab(vocab, tau)`
  composed by train.py right after build_vocab — same pipeline position the plan
  asks for.
- Canonical map persisted to a NEW artifact `mac/evaluation/phase4/canonical.json`
  (vocab.json must stay a plain list — `rank_vocabulary._load_student:622` indexes
  it directly and Track 1 must not touch that file). `canonical.json` added to the
  phase4 .gitignore artifact list.
- `multihot(board, word2idx, canonical=None)` / `make_xy(rows, word2idx,
  canonical=None)` — backward-compatible optional arg; dropped variants redirect
  their label mass to the kept representative.

## Decision 3 — infer.py

- Pool = top-(pool_mult x k_dyn) vocab items by sigmoid score (config
  `board.pool_mult: 3`), unioned with content terms extracted from
  transcript+scene (topics_from_context logic copied per plan, adapted caps:
  candidate pools want breadth, so max_terms=10 / scene cap 6 instead of the
  prompt-oriented 6/3).
- MMR via the exact production utilities `selection.mmr_select` +
  `selection.build_query_vector` (relevance = cosine to query).
- is_question => reserve 2 slots picked by MMR(lambda=0.35, mirroring
  rank_vocabulary.ANSWER_LAM) over non-hedge candidates only.
- Hedge cluster capped at <= 1 member per board; hedge = normalized membership in
  the known don't-know/not-sure phrase family.

## Decision 4 — answer reservation is relevance-first, not MMR(0.35) (DRIFT)

First implementation mirrored the teacher literally: `mmr_select(non_hedge, query,
2, ANSWER_LAM=0.35)`. Measured on the aunt/dinner seating probe, the diversity-heavy
second pick chose "no, thanks." / "i need to use the restroom." over "table"
(rel 0.369) and "aunt" (rel 0.336) — because the teacher's 0.35 lambda is only safe
when every candidate is a function-tagged ANSWER; our pool is unfiltered. Swept
ANSWER_LAM 0.5/0.6/0.7/0.8: never surfaced a seat/person word. The faithful mirror
of `_select_dynamic`'s stated contract ("surface the most situation-relevant answers
FIRST") over an unfiltered pool is relevance order + near-duplicate guard
(cosine >= 0.85 to an already-reserved answer skips). Measured with that shape:

```
probe 1 reserved: ['room', 'i need to use the restroom.']
probe 2 reserved: ['dinner', 'table']
```

## Results (all measured live, worktree, VOICELINK_PERCEPTION=0)

### Retrain (seeded, config-hashed)

```
vocab dedup tau=0.8: 116 -> 94 (22 variants remapped)   <- exactly the investigation's 22/116
config_hash=8151a7e34afd  rows=559  vocab=94  train=391 val=84 test=84
  early stop epoch 24  best_val=1.0541
saved student.pt + student.onnx  best_val=1.0541
```

### Acceptance 1 — hedge collapse

```
hedges kept in vocab: ["i don't know"]           <- exactly 1 of the 7
'i am not sure'  -> "i don't know"     "i'm not sure"   -> "i don't know"
"i'm not sure."  -> "i don't know"    'i am not sure.'  -> "i don't know"
'not sure'       -> "i don't know"    "i don't know."   -> "i don't know"
```

canonical.json carries 22 remapped variants total; multihot redirects their label
mass (test_multihot_canonical_redirects_label_mass).

### Acceptance 2 — held-out mean hedge tokens in top-7 (gate <= 0.6, was 2.107)

```
held-out n=84  vocab=94  k_dyn=6
  new model + infer.py MMR : 0.012  (>=2 hedges: 0/84)
  new model + raw topk     : 0.988  (>=2 hedges: 0/84)   <- dedup alone
  teacher                  : 0.321
```

0.012 — below the 0.6 gate AND below the teacher's own 0.321. Attribution: vocab
dedup alone gets 2.107 -> 0.988; the MMR + hedge-cap inference path takes it to 0.012.

### Acceptance 3 — seating probes (the investigation's hedge-wall scenarios)

```
Where would you like to sit today?  (v2-32 scene)
   ['yes', 'no', 'I need help', 'wait', 'room', 'i need to use the restroom.',
    'today', 'teacher', 'gestures', 'yes, please']
Who do you want to sit next to?  (v2-85 scene)
   ['yes', 'no', 'I need help', 'wait', 'dinner', 'table',
    'i need to use the restroom.', 'crowded', 'yes, please', 'aunt']
```

Zero hedge walls; concrete answers in top-7 ("room" / "table"+"dinner"; "aunt"
reaches the board at slot 10 via the transcript/scene candidate union — it is below
the vocab frequency floor and was previously unrepresentable). Honest note: the
boards are not teacher-quality — extracted raw tokens like "gestures"/"crowded" are
noise the RC-3-lite token union admits; a noun-phrase extractor would clean this,
flagged as future work, out of tonight's scope.

### Acceptance 4 — tests

```
tests/test_phase4.py: 18 passed          (6 new data tests, 4 new infer tests,
                                          2 real-checkpoint seating probes)
tests/ (entire mac suite): 64 passed
```

## Progress

- [x] Context read (PLAN.md, investigation, research brief, all owned+read-only files)
- [x] Hedge cosine structure measured (above)
- [x] data.py dedup + canonical map + tests
- [x] infer.py + tests
- [x] retrain + ONNX re-export (train.py prints above)
- [x] held-out hedge measurement: 0.012 (target <= 0.6, was 2.107)
- [x] seating probes: concrete answers, no hedge walls
- [x] full mac test suite green (64 passed)
