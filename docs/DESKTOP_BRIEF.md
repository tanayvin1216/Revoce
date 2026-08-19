# VoiceLink — brief for talking to Claude in the desktop app

**Written 2026-08-19. Self-contained on purpose.**

Paste or upload this whole file at the start of a desktop conversation. The
desktop app cannot see the VoiceLink repository, so everything it needs is
written out here rather than pointed at. Nothing below requires opening a file.

---

## 0. What this app is and isn't good for here

**Good for:** the submission itself — the 10-word title, the 150-word
description, the 2-minute video script and shot list, the email to the
organisers, deciding what to claim and what not to claim, talking through
whether a number is honest, rehearsing how to explain the project to a
non-technical judge, and the patent/prior-art thinking.

**Not good for:** anything that needs the code run, tested, measured or
changed. All the numbers in this document are already measured — treat them as
given. If a question needs a fresh measurement, that is a job for Claude Code in
the terminal, not here.

**The single most useful thing this app can do right now** is help produce the
submission video script and the written entry, because those are the critical
path and they are pure writing work.

---

## 1. What VoiceLink is, in plain language

A wearable communication aid for people who cannot speak — autism-first,
extending to aphasia, ALS, Parkinson's.

People who can't speak often use an "AAC board": a screen of words and symbols
they tap to talk. The problem is that somebody has to program those words in
advance and guess what the person will need. The guess is usually wrong, and
changing it takes a caregiver and time.

VoiceLink removes the guessing. A small clip-on device watches and listens to
what is actually happening. A nearby computer works out what is going on and
puts the words that fit *this moment* on the board, live. Nobody programs
anything ahead of time.

Three pieces:

| Piece | What it does |
|---|---|
| A clip-on device (~$15 of parts) with a camera and microphone | Captures and streams. Does **no** thinking. |
| A computer nearby (a Mac today; an Intel mini-PC is the real target) | Does all the thinking. |
| Any tablet or phone | Shows the board as a web page. The person taps it. |

Everything happens on the local machine. Nothing is sent to the internet.

**One design rule the whole system is built on:** every single AI call in the
project goes through one function. The wearable never calls a model. The tablet
never calls a model. They can only reach intelligence through the computer, and
the computer only through that one function. This is why the AI engine can be
swapped from Mac to Intel by changing a setting instead of rewriting the system.
If a conversation ever drifts toward "just call the model from the tablet," that
breaks the thing that makes the project portable.

---

## 2. Status right now

- Working branch `phase4-distillation`. Everything is committed and pushed, and
  the working tree is clean.
- **144 automated tests pass, 0 failures, 0 skipped-as-known-broken.**
- The system works end to end. The classroom demo case — someone asks "what do
  plants take in from the air?" — produces this board:
  *yes · no · I need help · wait · oxygen · sun · **carbon dioxide** · water ·
  air · sunlight*. The correct answer is right there, and no pre-programmed
  board would have had it.

Build phases 0 through 5 are done. Phase 6 (integration, demo, video) is the one
that is not.

---

## 3. The thing that blocks everything else

**The submission deadline is probably 30 August 2026 — about 11 days from now —
and it is NOT confirmed.**

Two sources disagree, and Intel's official page blocks automated reading, so
this has to be checked by a person:

- CodeWeek says, word for word: global nominations opened 1 July 2026 and the
  last date is **30 August 2026**.
- A Polish university page says nominations for a *national* selection stage
  closed 1 July 2026.

These are probably two different stages, but under either reading the window is
closing or already shut. **Confirm this on Intel's own festival page before
spending effort on anything else.**

### What the entry has to contain

These requirements were found alongside the dates and they shape the work:

- Project title, **10 words maximum**
- Project description, **150 words maximum**
- **A video, 2 minutes maximum, with the team speaking directly to camera**
- **AI-generated voiceover is banned**
- English subtitles if the video isn't in English
- Signed consent forms, and proof of active student status
- Teams of up to 3 students; age bands are 13–17 and 18+

The video being mandatory, on-camera, and human-voiced is the important part. It
cannot be a screen recording with a generated narration. That makes it the
critical path, ahead of any further engineering.

---

## 4. What needs doing, in order

1. **Confirm the deadline.** Everything below is contingent on it.
2. **The video.** Script, shoot, edit, 2 minutes, on camera. It should contain at
   least one moment where the word that appears is obviously right in a way no
   pre-programmed board could have managed. The "carbon dioxide" classroom case
   or a seating question ("who do you want to sit next to?" → *aunt*) are the
   natural candidates.
3. **The written entry.** 10-word title, 150-word description.
4. **Intel hardware numbers.** Every benchmark so far was measured on Apple
   silicon. There are currently **zero** measurements on Intel hardware, and
   "use of Intel technology" is worth 30 of the ~93 points. Fixing this means
   signing up for Intel's Tiber AI Cloud (needs a human with a browser and an
   account) and running one benchmark script there. This is the biggest single
   scoring gap and it is roughly a 10-minute job once the account exists.
5. **Three free points.** There are 3 rubric points for "independent digital
   readiness" participation. Claiming them needs one email to the organisers
   asking whether an Intel course or Intel AI for Youth counts for an
   independent team. Nobody has sent it.
6. **Two dashboard entries.** Phases 0 and 2 were re-verified on 2026-08-17 and
   both pass, but were deliberately left un-flipped: the project has a rule that
   a human, not an agent, marks a phase green. Small task, needs you.

---

## 5. The numbers — and exactly how far each one can be pushed

This project has a deliberate culture of publishing its failures. Several
numbers below are unflattering and are meant to stay that way. **The fastest way
to damage this entry is to let a confident-sounding sentence overstate one of
them.** Each entry says what may and may not be claimed.

### Safe to claim

- **Choosing varied words beats choosing the top-scoring words.** Across 120
  test situations: distinct kinds of thing you could say, 4.68 vs 3.88;
  near-duplicate words on the board, 0.0004 vs 0.0274 (68x fewer); how often the
  board let you express your actual intent, 1.00 vs 0.81.
- **The small trained model is ~143x faster than the full pipeline** — 47
  milliseconds versus 6.7 seconds — while keeping 94% of the variety.
- **The Intel software path genuinely works**: the model converts to Intel's
  OpenVINO format with near-perfect fidelity, compresses to about a third of its
  size, and a speech model in that format beat the previous one (0.35s vs
  0.42s).
- **Nothing leaves the device.** This is architectural, not a promise.
- **Cost.** Roughly $15 of parts on the body, against dedicated AAC devices that
  run into the thousands with 3–18 month funding cycles.

### Claim only with the caveat attached

- **Concrete answers now work** — the board can offer "aunt" when someone is
  asked who they want to sit next to. True, and demonstrated. **But** the feature
  only triggers on 2 of the 84 test situations, so its effect on the overall
  scores is statistical noise. Say "this works and broke nothing." Do **not** say
  "this improved our scores." It didn't measurably.
- **The trained model matches the teacher's word choices 5.6% of the time.**
  This sounds terrible and is widely misread. The model can only pick from a
  fixed 94-word list, and that list can only express about a third of what the
  teacher says — so the ceiling is low by construction. The project deliberately
  stopped grading on word-matching and grades on "does the board let the person
  say what they mean" instead, where it scores 0.761 against the teacher's 0.845.

### Must be reported as failing

- **One quality measure still fails.** The board stacks slightly more
  same-purpose words than the teacher does (0.1865 vs 0.1684, where lower is
  better and the bar is "no worse than the teacher"). It has been reported as a
  failure through three review rounds. It stays a failure. Do not round it away
  or bury it.
- **Zero Intel hardware measurements exist.** See item 4 above.

### The honest negative result, which is an asset

Early on, the training data was increased fivefold and the model got **worse**,
not better. That falsified the obvious assumption and was published rather than
hidden.

The reason was found on 2026-08-17 and it's a genuinely good story: the model
picks from a fixed list of words built by keeping words that come up often.
Specific answers never come up often — "right arm" is the correct answer for
exactly one situation in the world — so every specific answer gets filtered out
and only vague filler survives. Adding more data adds more one-off words, which
pushes the list *further* toward filler. Measured: the fixed list can express
only 32.5% of what the teacher says, and 49% of what it can express is filler.
Making the list eight times bigger still only reaches 57%.

The fix was to stop trying. Specific answers are now read out of the actual
situation instead of a word list — which is why "aunt" can appear without ever
being in the list. **This is one of the strongest things to talk about in the
entry**: a wrong assumption, measured, disproven, root-caused, and fixed
architecturally.

---

## 6. Evidence for the "why this matters" section

Already gathered and cited in the project's evidence file:

- 1 in 31 U.S. 8-year-olds is identified with autism (CDC ADDM).
- 26.7% of children with autism are profoundly autistic — nonverbal, minimally
  verbal, or requiring extensive support.
- People using AAC communicate at roughly **10–18 words per minute**, against
  **125–185** for speech. That gap is the problem the project exists to close.
- 30–35% of funded AAC devices are abandoned within the first year.
- UN Sustainable Development Goals: target 10.2 (social inclusion regardless of
  disability) and 4.5 (equal access to education for people with disabilities)
  are the ones to name specifically. Naming the numbered target rather than just
  "SDG 10" is worth doing.

One caution: the festival's own 2025 participation figures conflict between
sources. Don't quote them.

---

## 7. Rubric, and where the entry stands

| Criterion | Points | Standing |
|---|---|---|
| Enriching Lives | 30 | Strong. Cited, specific, SDG targets named. |
| AI Innovation | 30 | Strong on substance; weak on demonstration until the video exists. |
| Use of Intel Technologies | 30 | Software work is real; **no Intel hardware numbers**. Biggest gap. |
| Independent digital readiness | 3 | Unclaimed. One email away. |

---

## 8. Ethics position, if a judge or a conversation goes there

Worth being able to state plainly, because it's a real strength:

- All processing is local; nothing is transmitted.
- Audio and video buffers are temporary — there is no persistent recording.
- There is a hardware kill switch.
- The personal profile never leaves the machine.
- The device captures bystanders who did not consent to being recorded. Because
  of that, testing is restricted to the team only until written proxy consent and
  pseudonym handling exist. This constraint is self-imposed and documented.
- Every tap the user makes is logged with the situation that produced it — this
  is the data that makes the system improve, and it cannot be recovered
  retroactively, which is why it was built in from the first version.

---

## 9. Traps

- **Don't let the entry overclaim.** The measured honesty is a differentiator;
  three review rounds were spent making the numbers harder rather than kinder. A
  polished sentence that inflates one of them would undercut the whole thing.
- **"Just make the word list bigger" is ruled out by measurement.** So is "add
  more training data." Both were tried. If a conversation drifts there, the
  answer is section 5.
- **Don't propose putting a model on the tablet or the wearable.** See section 1.
- **The video cannot use a generated voice**, and it needs faces on camera.
  Plan for that rather than discovering it late.
- **Don't quote the festival's 2025 participation numbers.**

---

## 10. Prompts worth opening with

- "Here's the brief. I have ~11 days and no video yet. Give me a 2-minute shot
  list and script that gets the 'carbon dioxide' moment on screen in the first 40
  seconds."
- "Write the 150-word description. It must not overclaim — read section 5 first
  and flag anything I've written that overstates a number."
- "Give me three options for a 10-word project title."
- "Draft the email to the festival organisers asking whether an Intel course
  counts for the independent-participation points for a team that isn't going
  through a school."
- "Play a judge who knows nothing about AAC. Ask me the five hardest questions
  about this project."
- "The word-match score is 5.6%. Help me explain why that's not the failure it
  looks like, in two sentences a non-technical judge would accept."
