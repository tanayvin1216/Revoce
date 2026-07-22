# VoiceLink — Project Description

Written 2026-07-21 from the repo as it stands on branch `phase4-distillation`
(HEAD `eb1331c`). Intended as reusable context for writing about the project —
briefs, submissions, README intros, or handing to another agent.

## One-paragraph version

VoiceLink is a wearable AAC (augmentative and alternative communication) system for
nonverbal and minimally-verbal people. A small body-worn ESP32-S3 device continuously
senses the wearer's surroundings and the conversation around them; a nearby Mac
(deployment target: an Intel NUC-class mini-PC) runs all the intelligence locally; an
iPad or phone displays the AAC board as a plain web page. The board re-populates in
real time with the vocabulary the moment actually needs — so the right words are there
without anyone hand-programming the device in advance. Nothing leaves the machine in
normal operation.

## The three devices, one seam

- **Wearable (ESP32-S3, `firmware/voicelink_capture/`)** — Freenove ESP32-S3-WROOM CAM
  + SPH0645 I2S mic. Runs as its own Wi-Fi SoftAP so it works with no router. Captures
  VGA JPEG frames and continuous 16 kHz mono audio, streams both over a single
  WebSocket with a 1-byte type tag (`0x01` = JPEG frame, `0x02` = PCM chunk). Audio
  read/send lives in a high-priority FreeRTOS task pinned to core 0 so a slow JPEG grab
  can never stutter the mic. **It does zero processing.**
- **Mac server (`mac/server.py`, aiohttp)** — the only thing that talks to both other
  devices. Serves the board over plain HTTP (same origin as the WebSocket, deliberately
  avoiding the HTTPS mixed-content wall), pushes `vocab_update` messages to the page,
  and receives `tap` messages back. The wire schema is a documented stable contract.
- **AAC board (`mac/web/index.html`, plus a developer console `console.html`)** — 10
  large tap targets, no model access of its own.

**The architectural rule the whole project is built around:** every model call in the
system passes through one function,

```python
rank_vocabulary(scene, transcript, context) -> ranked_words
```

Its signature is fixed. The firmware never calls a model; the web app never calls a
model. That single boundary is what lets the inference backend be swapped (Mac/MLX
today, Intel/OpenVINO next) without touching the firmware, the routes, or the UI.

## How a board gets built

1. **Perception (`mac/perception/`)** — an audio worker does energy-based VAD to find
   complete utterances and transcribes with faster-whisper `base.en` int8; a vision
   worker uses a cheap grayscale frame-diff change detector to *gate* an expensive VLM
   call (Qwen2.5-VL-3B-Instruct-4bit via mlx-vlm), so the model only fires when the
   scene actually changed. Both backends are one-line swaps in `perception/config.py`.
2. **Ranking (`mac/rank_vocabulary.py`, 776 lines)** — instead of asking for "the 7 most
   likely words," the system over-generates a candidate pool along two axes (9 AAC
   communication functions × concrete topics in play) and then MMR-selects for
   *coverage*, so the board offers seven genuinely different things to say rather than
   seven paraphrases of one. Four safety words — `yes / no / I need help / wait`,
   deliberately four distinct communicative functions — are pinned deterministically
   into every board, which makes "never mode-collapsed" structurally true even when a
   generation goes bad.
3. **Logging** — every tap is written to `mac/taps.jsonl` with a snapshot of the context
   that produced the board. That interaction data is treated as irreplaceable and cannot
   be recovered retroactively.

## The ML work

- **Phase 3 — a self-test loop and a metric reframe.** 120 scenarios, one command, JSONL
  + plots. It showed the diverse strategy beating plain top-k on distinct-functions
  (4.68 vs 3.88), near-duplicate rate (68× lower), and coverage-of-intent (1.00 vs
  0.81) — and caught a real taxonomy bug in the process.
- **Phase 4 — knowledge distillation.** The LLM teacher (Qwen) is distilled into a tiny
  student ranker (`evaluation/phase4/student.py`): LayerNorm → Linear(769→192) → ReLU →
  Dropout → Linear(→94 dynamic vocab), trained as multi-label BCE over "which phrases
  belong on the board right now." Export-first design — only opset-17-clean layers, so
  ONNX/OpenVINO conversion is guaranteed. Measured 156× faster than the teacher on the
  honest end-to-end path (43.6 ms vs 6.8 s).
- **The honest negative result.** The project documents a falsified hypothesis rather
  than hiding it: scaling teacher data 5× made teacher-agreement *worse*
  (dynamic_overlap 0.232 → 0.165). The diagnosis — a pooled MiniLM `scene||transcript`
  embedding cannot represent the teacher's specific dynamic policy — is written up
  rather than papered over.
- **The "useless boards" investigation.** Metrics were passing while boards were
  unusable: the student amplified hedge-stacking 60× over the teacher ("I don't know"
  and its paraphrases eating the board). Fixed by semantically deduping the vocabulary
  (116 → 94 items, 7 hedge paraphrases collapsed to 1) and replacing raw top-k with MMR
  + answer-quota + hedge-cap selection. Hedge tokens in the top-7 went 2.107 → 0.012.
- **Phase 5 — Intel readiness (`mac/intel/`).** Full OpenVINO 2026.2.1 lifecycle:
  student ONNX → FP32 IR (parity 1.4e-06) → NNCF INT8 (2.7× smaller, top-6 overlap
  0.950 mean), MiniLM exported, Whisper `base.en` INT8 exported and beating
  faster-whisper on the smoke clip (0.35 s vs 0.42 s). `VOICELINK_STUDENT_BACKEND=openvino`
  swaps the runtime *behind the seam* — only the logits producer changes, which is the
  whole point.

## The culture of the codebase

The repo is unusually disciplined about honesty, and that shows up in artifacts, not
just prose:

- Gates are designed to be able to fail, and are recalibrated when they turn out to
  certify broken behavior. The old coverage gate scored the broken student 0.958; the
  rewritten gate correctly failed it at 0.563.
- Known gaps are published, not buried. `same-function-pair 0.188 vs teacher 0.168 —
  FAIL` sits in the final review table next to the passes, and two tests are marked
  `xfail` as deliberate open-problem markers.
- Benchmarks carry provenance labels: the OpenVINO numbers say plainly they were
  measured on Apple Silicon via the ARM CPU plugin and are *not* Intel-hardware numbers.
- The evaluation README opens with a warning that the synthetic benchmark dataset is
  GPT-generated and the baseline is itself an LLM prompt — so a high score "partly
  measures imitation of GPT, not real communication value."
- Privacy is architectural rather than promissory: local-only inference, ephemeral
  buffers with no persistent recording, a hardware kill switch, a personal-context store
  that never leaves the device, and a standing rule to build and test on the team only
  until written proxy consent and pseudonym handling exist.

## Scale and state

~7,600 lines of Python across `mac/` plus ~400 lines of ESP32 firmware, with seven test
files (~1,700 lines) covering the seam, the console, eval gates, the Intel path, and 36
adversarial edge cases. Phases 1, 3, 4, and 5 are green on the dashboard
(`phase_status.json`); phases 0 (setup/guardrails), 2 (console + device sim), and 6
(integration/demo) are pending. All completed work currently lives on
`phase4-distillation` and has not been merged to `main`.

## Why now

The project is built as an entry for the Intel AI Global Impact Festival, and
`docs/COMPETITION_EVIDENCE.md` grounds the impact claim in cited data — 4–5 million
Americans can't reliably meet daily communication needs with speech, dedicated
speech-generating devices run $4,000–$12,000 with 3–18-month reimbursement cycles,
30–35% are abandoned within a year with context-mismatched boards a named cause, and AAC
users communicate at roughly 10% of conversational speed. VoiceLink's hardware bill is an
~$15 ESP32-S3 plus a tablet you already own plus one local computer.

## Map of the repo

```
firmware/voicelink_capture/   ESP32-S3 Arduino sketch (camera + I2S mic + WS streaming)
mac/rank_vocabulary.py        THE SEAM — every model call in the system
mac/server.py                 aiohttp server: HTTP board + WebSocket + tap logging
mac/perception/               VAD, Whisper transcription, change detector, VLM scene describer
mac/evaluation/               eval harness + phase1 / phase3 / phase4 experiment dirs
mac/intel/                    ONNX → OpenVINO IR → NNCF INT8 lifecycle + benchmarks
mac/tests/                    seam, console, eval gates, Intel path, adversarial edge cases
mac/web/                      index.html (the AAC board) + console.html (dev console)
docs/COMPETITION_EVIDENCE.md  cited impact/innovation/Intel-technology dossier
PLAN.md / phase_status.json   canonical plan and the live phase dashboard
```
