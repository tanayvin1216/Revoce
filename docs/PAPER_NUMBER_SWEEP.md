# Paper number sweep — 2026-08-25

Three papers: `paper/axiom`, `paper/odi`, `paper/rtca` (`paper/common` is only the
style file). Both NeurIPS workshop deadlines are **Aug 29 AoE**.

Every claim in all three `.tex` files was checked against the **committed
artifact**, not against `VERIFIED_NUMBERS.md`. That matters: the papers' READMEs
pin their numbers to `VERIFIED_NUMBERS.md` §B, but `phase4_results.json` has been
re-run since §B was written, so the papers are two generations behind the file
they cite.

**Rule going forward: the committed JSON is the authority. `VERIFIED_NUMBERS.md`
is a summary of it and has been wrong twice.**

---

## A. Phase-3 numbers — source is committed and healthy

`mac/evaluation/phase3/selftest_results.json` (n=120) is committed and its values
are stable.

| Quantity | File | axiom | odi | rtca |
|---|---|---|---|---|
| distinct functions, diverse / top-k | 4.683 / 3.875 | — | 4.68 / 3.88 ✅ | 4.68 / 3.88 ✅ |
| near-dup, diverse / top-k | 0.000397 / 0.02738 (**69.0×**) | — | 69× ✅ | **68× ❌** |
| coverage of intent, diverse / top-k | 1.00 / **0.775** | — | 0.78 ✅ | **0.81 ❌** |
| exact match, diverse / top-k | 0.525 / 0.333 | — | 0.525 / 0.333 ✅ | **0.567 / 0.342 ❌** |

**rtca has three wrong phase-3 numbers.** Its README cites the same file odi does,
so these are transcription errors, not a different run. The 0.81 coverage figure
is the one `VERIFIED_NUMBERS.md` §3 item 5 already flagged as "the June scorer";
the committed file says 0.775.

---

## B. Phase-4 numbers — the papers are stale across the board

`mac/evaluation/phase4/phase4_results.json`, config hash `d6ce2e56d50b`.

| Quantity | **Committed file** | What the papers say |
|---|---|---|
| Vocabulary size | **95** | **94** — all three papers, and the README tables ❌ |
| Rows after dedup | 559 | 559 ✅ |
| Student board latency | **10.135 ms** | 12–15 ms (axiom, rtca) ❌ · 47.081 ms (odi §5) ❌ |
| Teacher latency | **7.05 s** (n=8) | 6.7–11.7 s ❌ · 6.71 s ❌ |
| Speedup | **695.5×** | >400× (safe under-claim) · **142.6×** (odi §5) ❌ |
| Distinct functions, student / teacher | **4.095 / 4.298** | 4.083 / 4.333 ❌ |
| Retention ratio | **0.953** | 0.942 / "94%" ❌ |
| Near-dup, student | **0.0** (teacher 0.0006) | 0.0006 quoted as the student's ❌ |
| Same-function pair, student / teacher | **0.1893 / 0.1735** | 0.1865 / 0.1684 ❌ |
| Board variety | **0.952** | 0.917 ❌ |
| Coverage of intent, dynamic | **0.789 / 0.873** | 0.761 / 0.845 ❌ |
| Coverage of intent, overall | **0.873 / 0.944** | 0.845 / 0.944 ❌ |
| Dynamic overlap (agreement) | **0.067** | 0.056 ❌ |
| Full top-7 overlap | 0.594 | 0.59 ✅ |
| Deictic open-answer fire rate | 0.8833 | 0.88 ✅ |
| Deictic closed-vocab hit | **0.0167** | 0.050 ❌ |

Nearly every Phase-4 number moved. The direction is mostly **favourable** —
retention is up (0.942 → 0.953), coverage is up, variety is up, latency is down
(47 → 10 ms) and the speedup is up (142.6× → 695.5×).

### ⚠ The gate verdicts are the serious problem

```
diversity_retention.pass    = False        verdict = INCONCLUSIVE
coverage_of_intent.pass     = True         verdict = INCONCLUSIVE   margin = 0.0028
same_function_pair_margin                  verdict = INCONCLUSIVE   margin = -0.0159
```

`axiom` line 398 states **"PASS (0.901× ≥ 0.90×)"** for dynamic coverage of
intent. The committed file reports that gate's margin test as **INCONCLUSIVE**,
with a margin of 0.0028 — i.e. the result clears the bar by a quarter of a
percentage point, well inside noise at n=71.

Claiming PASS on an INCONCLUSIVE margin is the single highest-risk statement in
any of the three papers. A reviewer who asks for the gate output will find it
disagrees with the sentence. The honest phrasing is that the student reaches
**0.904× of the teacher's dynamic coverage (n=71), clearing the 0.90 bar by a
margin the harness itself scores as inconclusive.**

---

## C. Intel / OpenVINO numbers

| Quantity | Truth | Papers |
|---|---|---|
| ONNX↔IR logit parity | 1.43e-6 | 1.4e-6 ✅ |
| INT8 vs FP32 size | 236 / 662 KB = **2.80×** | 2.8× (odi) ✅ · **2.7× (rtca) ❌** |
| INT8 top-6 overlap mean / min | 0.9505 / 0.6667 | ✅ |
| Mac head p50 torch/FP32/INT8 | 0.015 / 0.025 / 0.031 ms | ✅ |
| Mac end-to-end | **9.70 / 9.65 / 5.51 ms**, Jaccard 0.913 | **16.8 / 15.7 / 15.3, 0.914 ❌** — this is the superseded `55306d8` table that has now been deleted from `BENCHMARKS.md` |
| Whisper (Mac, quiet) | 252 / 407 vs 300 / 403 ms | odi quotes **681/795 vs 1162/1157** ❌ (withdrawn — loaded machine) and **0.35 vs 0.42 s** ❌ (withdrawn) |
| Test count | **213** | 144 ❌ |

### 🟢 The big opportunity: odi's Limitations section is now wrong in your favour

`paper/odi` has an open `\todo` on a Limitations paragraph stating **"zero
Intel-hardware measurements."** That is no longer true, and ODI is literally the
*On-Device Intelligence* workshop. You can now state:

- the same unmodified IR serves on a **Core Ultra 9 185H CPU and its Arc iGPU**
- the OpenVINO runtime makes the student head **2.7× faster than torch on Intel**
  while being **1.7× slower on ARM** — the toolchain is doing Intel-specific work
- moving MiniLM to OpenVINO speeds the whole board **~1.9× on Intel** (2.02× CPU,
  1.85× GPU, two independent runs) and 1.76× on the Mac
- the **NPU is enumerated but rejects the graph**: `Missing upper bound for one or
  more nodes` — it requires static shapes, and the IR carries a dynamic batch axis

Do **not** quote any absolute Core Ultra millisecond figure — that run overlapped
a 6 GB download and a 28-minute teacher slice. See `VERIFIED_NUMBERS.md` §0b.

---

## D. Numbers with no committed source

These appear in the papers but no artifact in the repo produces them:

| Number | Only source |
|---|---|
| 32.5% representable, 49.1% filler, 12.2% no representable word, 8.5×→56.6% | `docs/claude/working-notes/rc3-open-answers/NOTES.md` (analysis notes, no script) |
| 0.232 → 0.165 agreement, the "5× data" multiplier | `phase_status.json` prose only. `axiom` has an open `\todo` admitting the 5× cannot be reconciled against the committed 360 / 559 / 600-row corpora |
| "~$15 wearable" | no BOM. Already on the do-not-use list |

The RC-3 ceiling numbers are the intellectual core of `axiom`, so they need
either a committed script that regenerates them or an explicit "derived in
analysis notes" footnote. They are not wrong — they are unreproducible as
committed.

---

## E. Recommended actions, in order

1. **`axiom`: fix the PASS claim** on dynamic coverage of intent. Highest risk,
   smallest edit.
2. **All three: 94 → 95** for the vocabulary size, including the `769→192→94`
   architecture line.
3. **All three: refresh the Phase-4 table** to the committed values. Most move in
   your favour.
4. **`odi`: delete `142.6×` / `47.081 ms`** from §5 and use 10.135 ms / 7.05 s.
   Keep **">400×"** as the abstract-level claim — 695.5× rests on n=8 teacher
   samples and the teacher's run-to-run spread is large (7.05 s committed,
   10.64 s and 11.68 s observed on loaded machines).
5. **`odi`: replace the "zero Intel-hardware measurements" Limitations paragraph**
   with the §C results. Biggest single upgrade available, and it is a
   *strengthening*, not a correction.
6. **`odi`: fix the Mac end-to-end row** (16.8/15.7/15.3 → 9.70/9.65/5.51) and
   drop the two withdrawn Whisper figures.
7. **`rtca`: fix four transcription errors** — 68× → 69×, 0.81 → 0.775, exact
   match 0.567/0.342 → 0.525/0.333, INT8 2.7× → 2.8×.
8. **`odi`: 144 tests → 213.**
9. **Optional, and the only one needing a re-run:** one quiet
   `evaluate.py --teacher-samples 12` on the Mac would tighten the speedup and
   possibly move the two INCONCLUSIVE verdicts. ~10 minutes, no Intel hardware
   needed.
