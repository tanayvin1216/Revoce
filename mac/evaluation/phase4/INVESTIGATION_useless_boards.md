# Why the distilled student board is useless despite passing metrics

Investigation of the seating and photosynthesis scenarios. Every number below was
measured live (`VOICELINK_PERCEPTION=0`, cwd `mac/`, the real `student.pt`, the real
held-out split n=84). Claims that did not reproduce are called out and discarded.

---

## 1. The two failure modes, stated plainly

**Mode A — the seating board is a wall of hedges with no answer (mode collapse).**
For "Where would you like to sit today?" the student board is:

```
['yes','no','I need help','wait', "i don't know",'i am not sure',"i'm not sure","i'm not sure.",'more','yes, i am']
```

Four paraphrases of "I don't know" fill the dynamic slots; there is **no seat answer at
all**. "Who do you want to sit next to?" is the same:

```
['yes','no','I need help','wait', "i don't know","i'm not sure",'i am not sure',"i'm not sure.","i don't care",'no thanks']
```

Four hedges, zero "my aunt / my friend". On the window/aisle variants the seat words DO
survive but get shoved behind the hedges to board positions 7–8, e.g. `... 'i am not
sure', "i don't know", 'aisle seat', 'window seat'` — "window seat" lands at position 8,
outside the top-7 the metrics judge. The teacher answered all of these with real seat
choices.

**Mode B — the photosynthesis board is actually fine; the "sugar" expectation is
out-of-corpus.** For "What do plants take in from the air?" the student board is:

```
['yes','no','I need help','wait', "i don't know",'sun','carbon dioxide','i am not sure','i need more time','more about volcanoes?']
```

The gold answer for this scene **is** "carbon dioxide", and the student produced it (vocab
index 46). "sugar" is the answer to a *different* question ("what do plants make?") that
**does not exist anywhere in the data** (`photosynthesis` literal = 0 contexts,
`plants + make/produce` = 0, `sugar` in any prompt = 0). And "sugar" as a board word never
occurs except as medical "blood sugar". So Mode B is not a hedge wall — it is the
**representability ceiling**: an answer the teacher never emits above the frequency floor
can never be a column the student outputs. The seating failure is the real defect; the
photosynthesis case is the honest demonstration of *why* some answers are structurally
impossible.

---

## 2. Root causes, ranked, each with file:line and the number that proves it

### RC-1 (primary) — the student inference path is raw `torch.topk`, so near-synonym hedges co-occupy every slot
`evaluation/phase4/student.py:46` — `return torch.topk(logits, k).indices.tolist()`.
No MMR, no dedup, no answer reservation. The teacher seam does the opposite:
`rank_vocabulary.py` `_select_dynamic` reserves an answer quota and MMR-selects for
spread. Measured amplification on the held-out split:

| metric (held-out n=84) | student | teacher |
|---|---|---|
| mean hedge tokens in top-7 | **2.107** | 0.321 |
| boards with ≥2 hedges in top-7 | **72.6%** | 1.2% |
| mean hedge tokens, full board (max) | **3.179 (6)** | 0.893 |

The teacher's mild ~0.3-hedge/board tendency becomes a 2.1-hedge/board collapse — a ~60×
increase in stacked-hedge boards — purely because argmax has no redundancy penalty.

### RC-2 (primary) — `build_vocab` does no semantic dedup, so one concept "I don't know" occupies 7 of 116 columns and the top-3 ranks
`evaluation/phase4/data.py:58-76` — frequency + core filter only. Measured on the actual
training corpus (`train.gather_rows()`, 559 rows, which is what built `vocab.json`):

```
rank 0  freq=278  "i don't know"     <-- HEDGE
rank 1  freq=102  'i am not sure'    <-- HEDGE
rank 2  freq= 54  "i'm not sure"     <-- HEDGE
rank 8  freq= 20  "i'm not sure."    <-- HEDGE
rank 9  freq= 17  'i am not sure.'   <-- HEDGE
rank10  freq= 14  'not sure'         <-- HEDGE
rank16  ...       "i don't know."    <-- HEDGE
```

Hedge variants = **496 / 3354 = 14.8%** of all dynamic training labels; "i don't know"
alone is on **49.7%** of teacher boards, any hedge on **69.9%**. Single-link clustering of
the 116-word vocab at cosine ≥ 0.80 gives **22/116 = 19.0%** redundant slots, the largest
cluster being exactly the 7 hedges. Under multi-label BCE these 7 correlated, frequently-
positive labels all learn high baseline logits — which is the fuel RC-1 then ignites.

### RC-3 (primary) — concrete answers below `min_freq=3` are *unrepresentable*, not merely unranked
`evaluation/phase4/config.yaml:17` `min_freq: 3` + `data.py:72`. Of **2362** distinct
non-core board phrases in the 600-row teacher data, only **124 survive freq≥3 → 2238
(94.8%) are dropped** and can never be a student column. Measured phrase frequencies:

```
'sugar'=0   'over there'=0   'in the back'=0   'glucose'=0   'sunlight'=0
'oxygen'=1  'water'=2        'carbon dioxide'=4  (survives -> vocab idx 46)
```

"carbon dioxide" cleared the floor and the student uses it (Mode B works). "sugar",
"oxygen", "water", "over there" did not, so no student-side change could ever surface
them. This is upstream of the student: the teacher rarely emits these, so they never reach
freq≥3.

---

## 3. Which "passing" metrics are gamed, and why

**coverage_of_intent@7 = 0.958 (PASS, gate ≥0.90) is gamed by the pinned core words.**
`metrics3.py:82-85` iterates the full top-7 — which is 4 always-pinned core words + only 3
student words (`data.py:108` pins core first). Measured:

- full top-7 coverage = **0.958**
- coverage of the **dynamic words inside that top-7** (core removed) = **0.803** — below
  the 0.90 gate, a FAIL.
- **11/71 (15.5%)** scenarios pass *solely* because a pinned core word lexically matched an
  acceptable move (e.g. core "wait" hits "Wait, I have a peanut allergy"). Remove core and
  those 11 flip to fail.

So the gate certifies the *presence of the safety words it pins itself* — not the model's
situational output. (Note: `coverage_metrics.py` substring matching also lets core "no"
match "now"; word-boundary matching would remove that leak.)

**near_dup_rate@7 = 0.0198 (looks clean) misses the hedge wall.** `metrics3.py:48-59`
counts board *pairs* with cosine ≥ `DUP_TAU=0.85`. Measured same-communicative-function
pair fraction over the same dynamic top-7 = **0.1689** — 8.5× what near_dup reports. The
threshold is the reason: of the 10 hedge pairs, the closest four **do** clear 0.85 (e.g.
"i am not sure"~"i'm not sure" = 0.943), but the semantically-identical "i don't
know"~"i am not sure" = **0.784** and "i don't know"~"not sure" = **0.719** sit below it,
and a board is mostly non-hedge pairs, so the *mean* dilutes to 0.02. near_dup is not
blind, but it is far too permissive to flag a board that is 3–4 paraphrases of one move.
(Discarded the stronger claim that near_dup misses hedges entirely — it catches the
closest pairs; it just averages them away.)

---

## 4. Remediation plan, ordered by leverage

Each item independently improves the seating-class board; none alone fixes everything.

1. **Vocab canonicalization + answer retention** — `evaluation/phase4/data.py:72-74`
   (build_vocab). After frequency ranking, greedily drop any candidate whose MiniLM cosine
   to an already-kept word is ≥ 0.80 (reuse `evaluation/embed.py`), collapsing the 7 hedges
   to one and freeing ~19% of slots. Highest leverage because it fixes the training target,
   not just inference. Retrain after.
2. **Student-inference MMR + answer slot** — `evaluation/phase4/student.py:40-46`. Replace
   `torch.topk` with: take a logit-top-N pool (~3×k_dyn), MMR down to k_dyn over MiniLM
   embeddings (the util `rank_vocabulary.py` `_select_dynamic` already uses), and when
   `is_question` reserve ≥1 non-hedge slot. This is what stops 4 hedges co-occupying the
   seating board even before retraining. `is_question` is already threaded to
   `student_board` (`evaluate.py:60`), so it is available.
3. **Metric fix that would actually fail these boards** — `evaluation/phase4/evaluate.py:128`.
   Score coverage on `sb_dynamic` (already built at line 132), not `sb` → student **0.803**,
   FAIL. Add an answer-presence check that drops hedge-labelled acceptable moves before
   scoring. Replace/augment `near_dup_rate` with the same-function-pair fraction
   (`taxonomy.classify_functions`) → **0.169** vs the reported 0.020. Any one of these fails
   the seating board today.
4. **Teacher-generation answer-mode (lowest leverage here)** — the teacher is *not* the
   seating culprit: it kept window/aisle/"My aunt"/"I want to sit here" and stacks ≥2
   hedges in top-7 on only 1.2% of boards. The teacher-side gap is only RC-3: answers below
   the frequency floor. Fix is to make answer-mode scenes emit concrete answers reliably (so
   "over there", "in the back" clear freq≥3) and/or lower/condition `min_freq` for content-
   noun answers (`config.yaml:17`). Without this, "sugar"-type answers stay unrepresentable
   regardless of student changes — but this does not need a rewrite of `rank_vocabulary`'s
   selection logic, which is sound.

---

## 5. Honest verdict — salvageable or is distillation the limiter?

The seating failure is **salvageable and is not a distillation-capability problem** — it is
two missing lines of inference discipline (RC-1) on top of an un-deduped vocab (RC-2), both
in `student.py`/`data.py`, not in the model's representational power. The teacher already
solves this scenario; the student threw the solution away at `topk`. Items 1–2 will
recover the seating-class boards, and item 3 will make the gates honest enough to prove it.
The genuine ceiling is narrower: the student can only ever output answers that cleared
`min_freq=3` in teacher boards, so any answer the teacher rarely emits ("sugar", "over
there") is structurally unreachable no matter how good the distillation is — but that is a
data/teacher-coverage limit (RC-3, item 4), not a limit of the small-MLP approach. Net:
fix the student inference path and vocab first (cheap, high-impact), fix the metrics so
they stop certifying pinned-core boards, and only then invest in teacher answer-coverage.
The distillation approach itself is fine; the current student is under-engineered, not
under-powered.
