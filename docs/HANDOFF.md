# VoiceLink — Context Handoff

Written 2026-08-06. Branch `phase4-distillation`, HEAD `b57cbaf`.
Read this first if you are picking up VoiceLink with no prior context.

---

## 1. What the product is

A wearable communication aid for people who cannot speak (nonverbal / minimally
verbal AAC users — autism-first, generalizing to aphasia, ALS, Parkinson's).

Existing AAC boards are static: a caregiver programs words in advance and hopes
they match the moment. VoiceLink senses the actual moment and puts the right
words on the board in real time, with no pre-programming.

Three devices:

| Device | Role |
|---|---|
| ESP32-S3 clip (~$15), camera + SPH0645 I2S mic | Captures and streams. Does **no** processing. |
| A computer (Mac today, Intel mini-PC as the deployment target) | Runs all inference. |
| Any tablet/phone | Opens a plain HTTP page served by that computer, shows the board, user taps. |

The clip talks only to the computer. The tablet talks only to the computer.
All inference is local — nothing leaves the machine.

## 2. The seam — the one architectural rule

Every model call in the whole system goes through one function:

```
rank_vocabulary(scene, transcript, context) -> ranked_words
```

Signature is fixed. The firmware never calls a model. The web UI never calls a
model. The console never calls a model. They reach intelligence only through the
server, and the server only through this function.

This is what lets the inference backend swap (Mac → Intel/OpenVINO) by config
flag without touching anything else. Never wire a model into a route, the
firmware, or the UI.

## 3. Repo layout

```
firmware/voicelink_capture/   Arduino C++ for the ESP32-S3
mac/
  server.py                   aiohttp server: board page, console, WebSocket, tap log
  rank_vocabulary.py          THE SEAM
  ranker.py / selection.py    candidate generation + MMR diverse selection
  ranker_config.py            BOARD_STRATEGY flag (diverse | topk)
  perception/                 whisper transcriber, Qwen2.5-VL scene describer, workers
  evaluation/                 phase1/3/4 eval harnesses, metrics, datasets
  intel/                      OpenVINO convert + NNCF quantize + bench + BENCHMARKS.md
  web/index.html              the AAC board (tablet)
  web/console.html            the dev console
  taps.jsonl                  every vocabulary tap + context snapshot (121 rows)
docs/
  COMPETITION_EVIDENCE.md     cited dossier mapped to the competition rubric
  PROJECT_DESCRIPTION.md      plain-language project description
  WEBSITE_CONTENT.md          site copy (untracked at time of writing)
  claude/working-notes/       verbose per-phase logs
phase_status.json             phase dashboard (the console reads this)
PLAN.md / CLAUDE.md / AGENTS.md   the operating plan and the two agent briefs
```

## 4. Phase status (from `phase_status.json`)

| Phase | Name | Status |
|---|---|---|
| 0 | Setup & guardrails | pending (work appears done; never flipped) |
| 1 | Make it live & correct | **green** |
| 2 | Console + device sim + dashboard | pending (console exists and works; never flipped) |
| 3 | Self-test loop + metric reframe | **green** |
| 4 | Distillation v1 | **green** (with a documented failure, see below) |
| 5 | Intel readiness | **green** (with a documented gap, see below) |
| 6 | Integration, demo, commit | **not started** |

## 5. Measured results worth knowing

- **Diverse selection beats naive top-k** (120-scenario self-test): distinct
  communicative functions 4.68 vs 3.88; near-duplicate rate 0.0004 vs 0.0274
  (68x lower); coverage-of-intent 1.00 vs 0.81.
- **Distilled student ranker**: ~7 ms vs ~6.97 s teacher pipeline. Honest
  MMR-path speedup 156x. Diversity retained (distinct-fn 96%).
- **Honest negative result**: 5x-ing the training data made teacher-agreement
  *worse* (dynamic_overlap 0.232 → 0.165), falsifying the data-volume
  hypothesis. The student is fast and diverse but is **not** a faithful policy
  match to the teacher. This is documented, not hidden — it is an asset in the
  submission, not a liability.
- **OpenVINO lifecycle is real**: student ONNX → IR (parity 1.4e-6) → NNCF INT8
  (2.7x smaller, top-6 overlap 0.950 mean). MiniLM exported. whisper-base.en
  INT8 beats faster-whisper (0.35 s vs 0.42 s). `VOICELINK_STUDENT_BACKEND=openvino`
  serves through the seam with vocab-sha staleness guards.
- **Known open gap**: same-function-pair rate 0.188 vs teacher 0.168 — a FAIL
  against the gate, reported rather than papered over.
- **All numbers above were measured on Apple ARM silicon.** There are currently
  zero measurements on Intel hardware.

## 6. Guardrails that must not be broken

- **The seam** (§2). Non-negotiable.
- **Simple before sophisticated.** `rank_vocabulary` shipped as a plain prompt
  and was only augmented after being measured against it.
- **Log every tap**, with the context snapshot that produced the board. This
  interaction data cannot be recovered retroactively.
- **Privacy is architectural**: local-only inference, ephemeral buffers (no
  persistent recording), hardware kill switch, personal-context store never
  leaves the device.
- **Team-only testing** until written proxy consent and pseudonym handling exist
  — the wearable captures bystanders who did not consent.
- The audio path must never block on the network or a slow model call.

## 7. The competition

Intel AI Global Impact Festival. Rubric: Enriching Lives (30) + AI Innovation
(30) + Use of Intel Technologies (30) + IDR participation (3).

`docs/COMPETITION_EVIDENCE.md` is the cited dossier, already mapped section by
section to that rubric. It is roughly 90% of the submission text.

### Where each metric stands

- **Enriching Lives** — strong and essentially written. Cited prevalence data,
  cited device-cost and abandonment data, SDG targets named specifically.
- **AI Innovation** — strong on substance (ASR + VLM + embeddings +
  distillation + MMR + INT8 quantization, anti-gaming slate metrics, published
  failures). Weak on demonstration: no demo video exists.
- **Use of Intel Technologies** — the software work is correct, but every
  benchmark came off a Mac. **This is the largest scoring hole.**
- **IDR (3 pts)** — unclaimed, blocked on one email to organizers.

## 8. Open items, highest leverage first

**Updated 2026-08-17.**

0. **THE DEADLINE — VERIFY THIS FIRST, TODAY.** Two sources disagree and the
   official Intel page blocks automated fetches, so a human has to confirm it:
   - CodeWeek states, verbatim: *"Global nominations open: 1 July 2026"* and
     *"Last date for global nominations: 30 August 2026"* →
     that is **13 days from 2026-08-17**.
   - Wrocław Univ. of Science & Technology states *"Submission of nominations
     for the national selection stage: by July 1, 2026"* — already past.

   These are probably two different stages (national vs global), but under
   either reading the window is closing or closed. **Confirm on the official
   Intel festival page and set the real date before doing any more engineering.**

   Submission requirements found alongside the dates, which change the plan:
   project title ≤10 words; description ≤150 words; **video ≤2 minutes with the
   team speaking directly to camera**; signed consent forms; proof of active
   student status; **AI-generated voiceover is prohibited**; non-English video
   needs English subtitles. Teams are max 3 students.

1. **Phase 6: the demo video.** Now the critical path — it is a hard submission
   requirement, not a nice-to-have, and the "team speaking to camera" and
   "no AI voiceover" rules mean it cannot be assembled from screen recordings
   alone. Must contain at least one moment where the surfaced vocabulary is
   obviously right in a way no pre-programmed board could be.
2. **Intel Tiber AI Cloud** — sign up (needs a browser + human account), run
   `mac/intel/bench.py` on a Xeon / Core Ultra instance, add the Intel-hardware
   column to `mac/intel/BENCHMARKS.md`. Converts the weakest metric into the
   strongest. Human-blocked.
3. **Email organizers about IDR** — ask whether Intel AI for Youth or a public
   Intel course counts for an independent team. Human-blocked. 3 free points.
4. **Flip Phases 0 and 2** — both gates were re-run on 2026-08-17 and pass
   (see §11); left unflipped only because the anti-cascade protocol in §9 says
   the human flips the dashboard, not the agent.
5. **Reconcile the festival's 2025 scale numbers** before quoting them publicly
   — sources conflict.
6. ~~Teacher answer-coverage (RC-3)~~ — **done 2026-08-17**, and the diagnosis
   was wrong: it was never teacher-side. See §11 and
   `docs/claude/working-notes/rc3-open-answers/NOTES.md`.

## 9. Working conventions

- Commits are plain descriptive sentences, **not** Conventional Commits, and
  carry **no AI co-author attribution**.
- Each task touches a scoped set of files and ends in a runnable test.
- L/XL phases run through the `fable-pipeline` skill
  (`.claude/skills/fable-pipeline/SKILL.md`) with verbose logs under
  `docs/claude/working-notes/<phase>/`. That skill depends on Fable 5, which
  sunsets ~2026-07-07 — the process needs redesigning with the user at that point.
- Anti-cascade rule from `PLAN.md`: an agent reporting a phase done is not the
  signal to advance. The human runs the eval gate, reads the output, eyeballs
  the console on the photosynthesis scenario, then commits and flips the
  dashboard. Red means stop and fix in place.

## 10. What has NOT been verified in this handoff

This document was assembled by reading the repository on 2026-08-06. The test
suite, the server, and the eval harnesses were **not** re-run while writing it.
All "green" statuses are as recorded in `phase_status.json` by the run that
produced them. Re-verify before quoting any number in a submission.

§11 below supersedes this for everything it names — those were re-run.

## 11. Session of 2026-08-17 — what was verified and what changed

Everything here was executed, not read.

**Re-verified as-is.** Full suite green before any change (116 passed,
2 xfailed). Server imports and the console route serves. Phase-2's eval gate
(fire the photosynthesis case) produces a genuinely good board through the seam:

```
yes · no · I need help · wait · oxygen · sun · carbon dioxide · water · air · sunlight
```

"carbon dioxide" — the actual answer — is on the board. Phase-0's gate needs
`git status` clean, which it now is.

**RC-3 closed, and re-diagnosed.** It was filed as *teacher* answer-coverage.
Measurement says the teacher emits concrete answers constantly (2366 unique
words across 3600 slots); the 94-word closed vocab can represent only **32.5%**
of them, and **49.1%** of that is filler. Concrete answers are long-tail by
nature, so more teacher data cannot fix it — which also explains Phase 4's
"more data made agreement worse" negative result. Shipped an open-vocabulary
answer path (`mac/evaluation/phase4/answers.py`) that reads answers out of the
context for alternative and deictic questions. Suite now **144 passed,
0 xfailed**; both RC-3 xfail markers are live assertions.

**Do not over-read the RC-3 numbers.** The open path fires on only 2 of 84
held-out rows, so the gate deltas are noise. The defensible claim is that no
gate regressed and the mechanism demonstrably works on the probes — not that
same-function-pair improved. That gate **still fails** (0.1865 vs teacher
0.1684). Full detail and the three rejected approaches:
`docs/claude/working-notes/rc3-open-answers/NOTES.md`.

**Found the deadline** (see §8 item 0) — and it is close enough that it should
reorder everything.
