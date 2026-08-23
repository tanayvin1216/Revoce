# VoiceLink — 9-Day Submission Sprint

**Written 2026-08-20.** Supersedes the deadline sections of `DESKTOP_BRIEF.md` §3
and `HANDOFF.md` §8 item 0. Everything in §2 below was verified against primary
sources today; everything in §7 is a hazard found while verifying.

---

## 1. The one-paragraph version

The Intel deadline is real and confirmed: **30 August 2026**. All four NeurIPS
workshops are real, all four land **29–30 August**. That is five deadlines in a
nine-day window, and it is not possible to hit all five at quality. The
recommendation is **two papers, not four** — ODI and AXIOM — because they are
genuinely different papers rather than one paper submitted twice, plus the Intel
entry. RTCA and LIGHT get dropped unless the first two land early.

But none of that is the first decision. **The first decision is the patent**,
because nothing has been filed, the demo video and the competition entry are
both public disclosures, and once either goes out, foreign patent rights are
gone permanently and the US clock starts. That decision has to be made in the
next day or two, not on the 29th.

---

## 2. Verified deadlines

Confirmed 2026-08-20 from official CFP pages and the NeurIPS 2026 workshop
announcement (Aug 10, 2026). Times converted to America/New_York.

| Venue | Official deadline | In your timezone | Pages | Archival | Blind |
|---|---|---|---|---|---|
| **AXIOM** (Paris) | Aug 29, 23:59 **UTC** | **Sat Aug 29, 7:59 PM** | 4 (excl. refs/appendix) | Non-archival | Double-blind |
| **LIGHT** (Paris) | Aug 29 — **no time published** ⚠️ | treat as **Fri Aug 28 EOD** | **not published** ⚠️ | Non-archival | **not published** ⚠️ |
| **RTCA** (Sydney) | Aug 29, 23:59 **AoE** | **Sun Aug 30, 7:59 AM** | 8 full / **4 short** / 2 demo | Non-archival | Double-blind |
| **ODI** (Sydney) | Aug 29, 23:59 **AoE** | **Sun Aug 30, 7:59 AM** | **5** (excl. refs/appendix) | Non-archival | Double-blind |
| **Intel festival** | Aug 30 | treat as **Sat Aug 29 EOD** | — | — | — |

Notification for all four workshops: **Sep 29, 2026** (NeurIPS-mandated, which
is why they're identical). Workshops themselves: Dec 11–12 Sydney, Dec 12–13
Paris.

**Corrections to what you had:**

- Your ODI entry said "Aug 30, 11:59 UTC." The official page says "Aug 29,
  23:59 AoE." **These are the same instant.** Do not "fix" it.
- **AXIOM is 12 hours earlier than it looks.** 23:59 UTC, not AoE. It is the
  real hard stop of the week — Saturday evening, not Sunday morning.
- ODI's full name is "On-Device Intelligence: **Foundation Models** under
  Real-World Constraints." RTCA's is "Real-Time Conversational Agents: Toward
  Natural Multimodal Interaction."
- There is a *second* real-time conversational workshop in Paris (**ReMuCAI**)
  that is easy to confuse with RTCA.

**Unresolved, needs a human email:** LIGHT publishes no submission time, no page
limit, and no anonymity policy. Do not assume 4-page double-blind defaults.

**Cross-workshop submission:** no workshop forbids it, but only ODI explicitly
permits it ("non-archival and may be submitted elsewhere"). NeurIPS's own
Workshops Guidance bars *previously published* work but is silent on
simultaneous workshop submissions. **Absence of a ban is not permission.** If
you submit to both ODI and AXIOM, email both sets of chairs first — one line,
today.

Sources: [NeurIPS 2026 workshop announcement](https://blog.neurips.cc/2026/08/10/announcing-the-neurips-2026-workshops/) ·
[Workshops Guidance](https://neurips.cc/Conferences/2026/WorkshopsGuidance) ·
[RTCA](https://rtcaneurips26.github.io/) · [LIGHT](https://almaai-disi-unibo.github.io/neurips2026-light-smallModels/) ·
[ODI](https://odi2026.github.io/) · [AXIOM](https://axiom-neurips2026.github.io/) ·
[Intel festival requirements](https://codeweek.eu/blog/intel-ai-global-impact-festival-2026-students/)

---

## 3. The patent decision — make this first

**Status: nothing filed.** `PATENT_NOVELTY_BRIEF.md` was written 2026-08-08 as a
*request* for a prior-art search. No results document exists in the repo, so as
far as I can tell that search was never run. You are nine days from publishing.

*I am not a lawyer and this is not legal advice. What follows is the shape of
the decision so you can take it to someone who is.*

### What counts as disclosure and what doesn't

- **Submitting to a double-blind workshop is generally not a public
  disclosure.** Reviewers are under confidentiality. This is the standard
  understanding, but confirm it with a practitioner before relying on it.
- **Posting the demo video publicly IS a disclosure.**
- **The Intel festival entry is probably a disclosure** — and worse, competition
  terms often carry publication rights or IP-assignment clauses. Your own brief
  flags this (§7.2). **Nobody has read the terms.** That is a 20-minute human
  task and it gates everything else.
- Accepted workshop papers go public around **Sep 29**. So even the papers have
  a fuse on them; it's just a longer one.

### Why a provisional, and why now

Your brief §7.4 already argues this against itself and lands in the right place.
The case is stronger now than it was on Aug 8:

- A provisional costs **$65 (micro entity) / $130 (small entity) / $325
  (undiscounted)** at current USPTO rates — negligible.
- It secures a priority date in days, permits "patent pending," and buys **12
  months** to refine claims.
- `PATENT_NOVELTY_BRIEF.md` is *already substantially a specification*: §3 gives
  the architecture, §4 gives ten described concepts, §8 gives claim scaffolds,
  §9 gives enablement data. A provisional needs no claims and is not examined.
  You are closer to a filing than you probably think.
- Several concepts are still moving (a gate is failing, no Intel-hardware
  numbers exist). Twelve months to firm those up before the non-provisional is
  exactly what a provisional is for.

**What I would do:** file the provisional before the video goes public and
before the Intel entry is submitted. Target **Mon Aug 24**. That preserves
foreign rights, starts nothing you can't finish, and unblocks the video, the
website, the papers, and the grant conversations all at once.

**What I would not do:** submit the Intel entry or post the video first and
decide the patent question afterward. That order is irreversible.

### Two things to preserve regardless

- **Confirm the GitHub repo has been private for its entire history.** A repo
  that was public even briefly is a publication (brief §7.2).
- **Inventorship evidence.** Brief §7.3 is right that AI-assisted inventions are
  patentable but a natural person must have made a significant contribution to
  each claim. Your working notes document who decided what — preserve them as
  they are, unedited, and do not clean them up.

---

## 4. What to cut

Four workshop papers + a two-minute on-camera video + a patent filing + Intel
benchmarks in nine days is not a plan, it's a wish. Here is the cut:

| | Verdict | Why |
|---|---|---|
| **ODI** | **Submit** — primary | 5 pages, and "Foundation Models under Real-World Constraints" is literally a description of your system. The only venue where the seam, MMR selection, distillation and INT8 all belong in one narrative. |
| **AXIOM** | **Submit** — second, distinct paper | 4 pages, explicitly welcomes "preliminary results, novel ideas, emerging research directions." The negative result is the strongest thing you have and this is the venue that wants it. |
| **RTCA** | **Drop** (revisit only if both above are done by Aug 27) | Turn-taking and latency fit, but efficiency and distillation are off-center here. Also carries a dataset ethics statement requirement. |
| **LIGHT** | **Drop** | Thematically on-target but the least-specified CFP of the four — no time, no page limit, no anonymity policy. Nine days is the wrong week to be guessing at submission requirements. |

The trigger to add RTCA back: both papers submitted, video cut, and it is still
Aug 27. Not before.

---

## 5. The two papers

These are **different papers with different core claims**, which is what makes
submitting both defensible rather than dual submission. Say so in the emails to
the chairs.

### Paper A — ODI (5 pages, Sydney)
**The system paper.** Context-aware AAC vocabulary selection on local hardware.

- The three-node topology and the architectural seam — one fixed-signature
  function through which every model call passes, which is what makes the
  backend swap a config flag.
- Diversity-first slate composition: MMR selection beating naive top-k over 120
  scenarios — distinct communicative functions 4.68 vs 3.88, near-duplicate rate
  0.0004 vs 0.0274 (68× lower), coverage-of-intent 1.00 vs 0.81.
- The deployment path: ONNX → OpenVINO IR (parity 1.4e-6) → NNCF INT8 (2.7×
  smaller, top-6 overlap 0.950 mean / 0.667 min).
- Hybrid distillation (N5): learn relevance, keep the selection policy symbolic.
  The raw student argmax board was measured useless; the fix was re-running the
  production MMR over the student's logits as a candidate pool.
- The failing gate, reported as failing: same-function-pair 0.1865 vs teacher
  0.1684.

### Paper B — AXIOM (4 pages, Paris)
**The negative-result paper.** Working title: *the closed-vocabulary ceiling in
distilling an open-vocabulary teacher.*

This is the better paper scientifically, and the lesson generalizes well beyond
AAC — it applies to any closed-label distillation of an open-vocabulary teacher.

- The observation: 5×-ing the distillation training data made teacher agreement
  **worse** (0.232 → 0.165), falsifying the data-volume hypothesis.
- The root cause, measured: a frequency-built closed vocabulary structurally
  cannot represent long-tail concrete answers. The 94-word vocab can express
  only **32.5%** of teacher word choices, and **49.1%** of what it can express is
  filler. An 8× larger vocabulary still only reaches **57%**.
- Why more data makes it worse rather than better: each new row adds more
  one-off concrete words, which pushes the frequency-built vocabulary *further*
  toward filler.
- The architectural fix rather than the scaling fix: read concrete answers out
  of the context instead of a fixed label space (`evaluation/phase4/answers.py`).
- **The honest caveat that must stay in:** the open path fires on only 2 of 84
  held-out rows. The claim is "the mechanism works and no gate regressed," not
  "this improved our numbers." It didn't measurably.

### The limitation both papers must state plainly

All evaluation is on **synthetic scenarios**, judged by an LLM, against a
teacher that is itself an LLM. `PHASE4_REPORT.md` already says this — "a student
that passes here may be imitating the teacher imitating GPT." Reviewers will hit
this hard. Foreground it in the limitations section rather than letting a
reviewer find it. There are also **zero human subjects**; testing is team-only
by self-imposed policy until proxy consent exists. At a systems/efficiency venue
that is acceptable if stated. At an accessibility venue it would not be — which
is another reason ODI and AXIOM are the right two.

---

## 6. Day by day

Human-blocked items are marked **[YOU]**. Everything else can go to Claude Code.

### Thu Aug 20 (today) — ~2 hours, all human, all unblocking
1. **[YOU]** Sign up for **Intel Tiber AI Cloud**. Account approval latency is
   unknown and it gates 30 of ~93 rubric points. This is first *because* it
   might take days, not because it takes long.
2. **[YOU]** Send the **IDR email** to festival organizers — does an Intel course
   or Intel AI for Youth count for an independent team? Three free points, one
   email, still unsent.
3. **[YOU]** **Read the Intel festival terms** for IP-assignment or publication
   clauses. This gates the patent decision.
4. **[YOU]** Email **ODI and AXIOM chairs** re: submitting different papers to
   both.
5. **[YOU]** Flip phases 0 and 2 in `phase_status.json` — both re-verified
   passing on Aug 17, left unflipped only because your rule says a human flips
   the dashboard.

### Fri Aug 21
- **[YOU]** Decide the patent question. If filing, start the provisional draft
  from `PATENT_NOVELTY_BRIEF.md` §3/§4/§8/§9.
- **Claude Code:** build `docs/VERIFIED_NUMBERS.md` — one file, every number
  re-run today, with the file and command that produced it. See §7; you have
  three conflicting speedup figures in circulation right now and this must be
  fixed before a single one goes into a paper or the entry.
- **Claude Code:** LaTeX scaffolds for both papers on the NeurIPS 2026 template,
  5pp and 4pp, with the existing figures from `evaluation/phase1/figures/` and
  `evaluation/phase3/figures/` wired in.
- Video script + shot list (I can draft this — ask).

### Sat–Sun Aug 22–23
- **[YOU]** **Shoot the video.** This is the only item with a hard scheduling
  constraint — it needs people on camera, and the weekend is when you have them.
  It cannot be a screen recording, cannot use a generated voice, and must be
  ≤2 minutes. Get the "carbon dioxide" moment on screen in the first 40 seconds.
- **Claude Code:** draft the ODI paper body.

### Mon Aug 24
- **[YOU]** **File the provisional.**
- **[YOU]** Run `mac/intel/bench.py` on a Xeon / Core Ultra Tiber instance and
  add the Intel-hardware column to `mac/intel/BENCHMARKS.md`. ~10 minutes once
  the account exists. Feeds *both* the Intel entry and the ODI paper.
- **Claude Code:** draft the AXIOM paper.

### Tue Aug 25
- **[YOU]** Video edit.
- Paper revisions against the verified numbers file.

### Wed Aug 26
- **[YOU]** 10-word title, 150-word description, consent forms, student status
  proof.
- Figures finalized, related-work sections written.

### Thu Aug 27
- Full read-through of both papers against the honesty contract in
  `DESKTOP_BRIEF.md` §5. Every claim traced to `VERIFIED_NUMBERS.md`.
- **[YOU]** Video final cut.
- Decision point: is there room for RTCA? If not, say no and stop thinking
  about it.

### Fri Aug 28
- Buffer. LIGHT deadline passes today under the safe assumption.
- **[YOU]** Anonymize both papers, check the appendix complies with double-blind
  (ODI explicitly requires this), OpenReview accounts set up and tested.

### Sat Aug 29
- **19:59 EDT — AXIOM. Hard stop.**
- **[YOU]** Intel festival entry submitted (treating Aug 30 as EOD Aug 29).

### Sun Aug 30
- **07:59 EDT — RTCA/ODI. Hard stop.** Submit ODI Saturday night, not Sunday
  morning.

---

## 7. The numbers hazard — fix this before anything is written

There are **three conflicting speedup figures** in your own documents right now:

| Source | Claim |
|---|---|
| `docs/COMPETITION_EVIDENCE.md` | ~7 ms, **~900×** |
| `docs/PATENT_NOVELTY_BRIEF.md` §9 | 43.6 ms, **156×** |
| `mac/evaluation/phase4/phase4_results.json` (current, re-run Aug 17) | **47.081 ms, 142.6×** |

The current measured end-to-end figure is **142.6×**. The 7 ms figure is the
bare MLP forward pass excluding embedding and MMR selection — it is not the
board. The patent brief already caught the first discrepancy and told the
drafter to use the end-to-end number; it just did so against a run that has
since been superseded.

The same drift affects several other numbers — the brief cites distinct-fn
4.048/ratio 0.934 and coverage 0.859/0.775, where the current file says
4.083/0.942 and 0.845/0.761.

**None of this is dishonest — it's ordinary drift from re-running evals.** But
four documents quoting four numbers into a competition entry and two
double-blind papers in the same week is exactly how an honest project acquires a
dishonest-looking sentence. One file, re-run today, everything cites it.

### The sentence to fix first

`docs/COMPETITION_EVIDENCE.md` line 71 currently reads:

> "a distilled 769-d MLP student **reproduces that policy** at ~7 ms (~900×
> faster than the teacher pipeline)"

That is the one claim in the dossier that your own measurements contradict
outright. `phase_status.json` and `PHASE4_REPORT.md` both say the student "is
fast and diverse but is **NOT** a faithful policy match" — dynamic_overlap
0.056, full top-7 overlap 0.59. Three numbers in that sentence are wrong (7 ms,
900×, "reproduces") and the dossier is roughly 90% of the submission text.

This is a single-sentence fix and it is the highest-priority edit in the repo.
Suggested replacement, using only supported numbers:

> a distilled 769-d MLP student supplies relevance candidates that the same
> symbolic MMR policy then selects over, producing a board in 47 ms against the
> teacher pipeline's 6.7 s (142.6×) while retaining 94% of the teacher's
> communicative-function diversity — though not the teacher's specific word
> choices.

**One more to source:** "making the list eight times bigger still only reaches
57%" appears in `DESKTOP_BRIEF.md` §5 and nowhere else — no measurement file
backs it. Either find the run that produced it or drop it. It is a load-bearing
sentence in the AXIOM paper's argument, so it needs a source before it is
written down again.

Other standing hazards, carried forward from `DESKTOP_BRIEF.md` §5 and §9:

- **Zero Intel-hardware measurements exist.** Everything is Apple ARM. Do not
  let either paper or the entry imply otherwise until Tiber runs.
- **The 5.6% teacher-agreement number is widely misread.** The ceiling is low by
  construction (§5). Explain it; don't bury it and don't lead with it.
- **The failing gate stays reported as failing.** 0.1865 vs 0.1684. Three review
  rounds have kept it. Keep it.
- **Do not quote the festival's 2025 participation figures** — sources conflict.

---

## 8. Grants — needs your input

You mentioned grants and I don't have enough to plan against them. What I can
say now:

- **A filed provisional materially strengthens most applications**, and some
  grants carry their own IP terms — another reason the patent decision comes
  first.
- **Non-archival workshop acceptances (Sep 29) are usable credentials** for
  applications after that date, which is most of them.
- Grant timelines are almost always long. Nothing here should displace anything
  in §6 unless a specific deadline says otherwise.

Tell me which grants and their deadlines and I'll fold them in.

---

## 9. What I'd hand Claude Code first

One task, before any paper writing:

> Re-run every evaluation gate and produce `docs/VERIFIED_NUMBERS.md`: a single
> table of every number this project might quote, each with the file, command,
> and run date that produced it. Flag every place `COMPETITION_EVIDENCE.md`,
> `PATENT_NOVELTY_BRIEF.md`, `HANDOFF.md`, or `DESKTOP_BRIEF.md` disagrees with
> the freshly measured value. Do not edit those documents — just produce the
> discrepancy list.

Everything downstream — both papers, the 150-word description, the video script
— cites that file and nothing else.
