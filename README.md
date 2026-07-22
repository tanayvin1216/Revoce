# VoiceLink

A wearable AAC (augmentative and alternative communication) system that puts the
right words on the board *at the moment the conversation needs them*.

A body-worn ESP32-S3 senses the wearer's surroundings and the conversation around
them. A nearby computer runs all the intelligence locally. A tablet shows the AAC
board, which re-populates in real time. Nothing leaves the machine.

Built for nonverbal and minimally-verbal people — the ones for whom a static,
hand-programmed board is the difference between joining a conversation and
sitting it out.

## Why

- ~4–5 million people in the U.S. can't reliably meet daily communication needs
  with speech.
- Dedicated speech-generating devices run $4,000–$12,000, with 3–18 month
  reimbursement cycles, and 30–35% are abandoned within a year — boards that
  don't match the user's real context are a named cause.
- AAC users communicate at roughly 10% of conversational speed.

VoiceLink's hardware is a ~$15 ESP32-S3, a tablet you already own, and one local
computer. Citations for all of the above are in `docs/COMPETITION_EVIDENCE.md`.

## Quickstart

Requires Python 3.9+. Apple Silicon for the local LLM/VLM; on Intel, serve the
student through OpenVINO instead (see `mac/intel/`).

```bash
git clone https://github.com/tanayvin1216/VoiceLink && cd VoiceLink
python3 -m venv .venv && source .venv/bin/activate
pip install -r mac/requirements-dev.txt      # runtime + test toolchain

cd mac
VOICELINK_PERCEPTION=0 pytest tests/ -q      # 105 passed, 10 skipped
python server.py                             # board at http://localhost:8080
```

No wearable on hand? `python fake_wearable.py` simulates the device stream.

The board is a plain HTTP page on the same origin as its WebSocket, so any
tablet or phone on the network can open it — deliberately no HTTPS
mixed-content wall, no app install.

## How it works

Every model call that produces a board passes through **one function**:

```python
rank_vocabulary(scene, transcript, context) -> ranked_words
```

The signature is fixed. The firmware never calls a model; the web app never
calls a model. That single seam is what lets the inference backend swap
(Mac/MLX today, Intel/OpenVINO next) without touching the firmware, the routes,
or the UI.

A board is built in three steps:

1. **Perception** — energy-based VAD finds complete utterances and transcribes
   them with faster-whisper `base.en`; a cheap grayscale frame-diff detector
   *gates* an expensive VLM call, so Qwen2.5-VL only fires when the scene
   actually changed.
2. **Ranking** — instead of asking for "the 7 most likely words" (which returns
   seven paraphrases of one thing), VoiceLink over-generates candidates along
   two axes — 9 AAC communication functions × concrete topics in play — then
   MMR-selects for *coverage*. The board offers seven genuinely different things
   to say. Four safety words (`yes / no / I need help / wait`, four distinct
   communicative functions) are pinned deterministically into every board.
3. **Logging** — every tap is written with a snapshot of the context that
   produced the board.

## Measured results

Two students, both trained on the same 600-board teacher corpus, both scored by
the same gate code on the same held-out split (n=84).

**v2** — 94-column classifier over a closed, frequency-floored vocabulary. This
is what ships today.
**v3** — open-vocabulary bi-encoder scoring (context, phrase) pairs. Opt-in via
`VOICELINK_STUDENT_VERSION=v3`.

| gate | v2 | v3 | teacher | bar |
|---|---|---|---|---|
| distinct-functions | 4.071 | **4.690** | 4.298 | ≥90% of teacher |
| near-duplicate rate | 0.0 | 0.0006 | 0.0006 | ≤0.08 |
| same-function-pair | 0.1899 | **0.1378** | 0.1735 | ≤ teacher |
| board-variety | 0.952 | 0.988 | — | ≥0.5 |
| coverage-of-intent (dynamic) | **0.803** | 0.789 | 0.873 | ≥90% of teacher |
| speedup (end-to-end MMR path) | 43.6 ms vs 6.8 s | — | — | ≥10× (156×) |

**Point estimates are not verdicts, and this project now says so out loud.**
Every gate is a relative claim, so each one carries a bootstrap CI on the
paired per-scenario margin. Under 5-fold cross-validation (n=471 scored
scenarios rather than 71):

| metric | v3 margin vs bar | 95% CI | verdict |
|---|---|---|---|
| distinct-functions | +0.6875 | [+0.6116, +0.7631] | **PASS** |
| same-function-pair | +0.0207 | [+0.0129, +0.0285] | **PASS** |
| coverage-of-intent | −0.0397 | [−0.0849, +0.0057] | INCONCLUSIVE |

What that bought us, honestly: v3's diversity advantage is the best-evidenced
result in the whole evaluation. Its coverage number is *below* the bar once
properly powered — the single-split figure that looked like a pass was noise
(fold-to-fold coverage swings 10 points). And **v2's long-standing
same-function-pair "FAIL" turned out to be statistically indistinguishable from
passing** — the project reasoned carefully for weeks about a number that was
inside its own error bars. That is why v2, not v3, is still the default.

The other honest results stay on the record: scaling teacher data 5× made
teacher-agreement *worse* (0.232 → 0.165), falsifying our data-volume
hypothesis. Write-ups in `mac/evaluation/phase4/`.

The benchmark corpus is synthetic and the baseline is itself an LLM prompt, so a
high score partly measures imitation of the teacher rather than real
communication value. No result here has been validated against a real AAC user.
`mac/evaluation/README.md` opens with that warning.

## Repo map

```
firmware/voicelink_capture/   ESP32-S3 sketch (camera + I2S mic + WS streaming)
mac/rank_vocabulary.py        THE SEAM — every board-producing model call
mac/server.py                 aiohttp: HTTP board + WebSocket + tap logging
mac/perception/               VAD, Whisper, change detector, VLM scene describer
mac/evaluation/               eval harness + phase1 / phase3 / phase4 experiments
mac/intel/                    ONNX → OpenVINO IR → NNCF INT8 lifecycle + benchmarks
mac/tests/                    seam, console, eval gates, Intel path, edge cases
mac/web/index.html            the AAC board
docs/COMPETITION_EVIDENCE.md  cited impact / innovation / Intel dossier
```

## Privacy

Architectural, not promissory: all inference is local, audio and video buffers
are ephemeral with no persistent recording, the wearable has a hardware kill
switch and a visible recording LED, and the personal-context store never leaves
the device. Testing is restricted to the team until written proxy consent and
pseudonym handling are in place — the wearable captures bystanders who did not
consent, and that is an open problem we are not pretending to have solved.

## Development tools

Developed with the help of generative AI coding assistants (Anthropic Claude,
OpenAI Codex) for code review, test generation, and research summarization.

## Attribution

Pictograms are from [ARASAAC](https://arasaac.org/), licensed CC BY-NC-SA —
non-commercial. Any commercial deployment would need a different symbol set.
