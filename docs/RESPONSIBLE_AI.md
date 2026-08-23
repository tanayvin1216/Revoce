# VoiceLink — Responsible AI and Data Management

Factual, code-referenced account of how VoiceLink obtains, processes, stores and
deletes data. Every claim cites a file and line in this repository (line numbers
as of 2026-08-23, branch `phase4-distillation`). Verify live with:

```
cd mac && python3 privacy.py status          # CLI
curl http://<mac>:8080/api/privacy/status    # same data, from the running server
```

## 1. Data flow (what is obtained, by whom, from whom)

| Stage | Data | Where it lives | Code |
|---|---|---|---|
| Wearable (ESP32-S3) | VGA JPEG frames + 16 kHz mono PCM | RAM only on the device; streamed over a private Wi-Fi access point the device itself hosts | `firmware/voicelink_capture/voicelink_capture.ino:4-16`, SoftAP at `:307` |
| Mac, fast path | Demuxes binary frames/audio into an in-memory queue and a single-slot latest-frame holder | RAM | `mac/perception/wearable_client.py:35-51` |
| Mac, audio worker | Whisper (base.en, INT8, CPU) transcribes each complete utterance; keeps only the last 500 characters of text | RAM | `mac/perception/audio_worker.py:114-140` (`[-500:]` at `:134`); model config `mac/perception/config.py:18-21` |
| Mac, vision worker | A cheap change detector gates a small VLM (Qwen2.5-VL-3B, 4-bit) that emits ONE sentence about the scene | RAM | `mac/perception/vision_worker.py:53-81`; prompt `mac/perception/config.py:35-39` |
| Mac, context state | The structured text summary (scene sentence, last utterance, speaker heuristic, flags, timestamps) | RAM, overwritten on every update | `mac/context_state.py:23-94` |
| Mac, seam | `rank_vocabulary(scene, transcript, context)` — the only model call site | RAM | `mac/rank_vocabulary.py`, called from `mac/server.py:_make_board` |
| iPad/phone | Board words + symbols over a LAN WebSocket; taps go back | browser memory; only a TTS voice preference in `localStorage` | `mac/web/index.html:233,263` |
| Mac, tap log | Every tap + the text context snapshot that produced its board | `mac/taps.jsonl` (gitignored) | `mac/server.py:297-318`; `.gitignore:14` |
| Mac, personal context | Names/relationships/routines/today's events | `mac/personal_context.json` (gitignored) | `mac/server.py:585`; `.gitignore:17` |

Raw audio and raw frames never touch disk at any stage. The only persistent
artifacts are text: the tap log and the personal-context profile.

## 2. Retention and deletion

* **Audio**: an utterance buffer is a `bytearray` that is emptied the moment the
  utterance is finalized (`mac/perception/audio_worker.py:117`). The queue in front
  of it is bounded (200 chunks) and drops the oldest chunk when full
  (`mac/perception/runner.py:33`, `mac/perception/wearable_client.py:60-70`).
* **Frames**: a single-slot holder; each new frame overwrites the previous one
  (`mac/perception/vision_worker.py:22-36`). No history is kept.
* **Board history in memory**: the server remembers the last 200 boards (text only)
  so a tap can be joined to its board; older ones are evicted
  (`mac/server.py:68`, `:469-470`).
* **Tap log** (`mac/taps.jsonl`): append-only JSONL, one record per tap, containing
  the word, timestamps, board id, board items, the client IP, and the text context
  snapshot (`mac/server.py:306-317`). This is the one irreplaceable dataset
  (CLAUDE.md guardrail), so the **default retention is "keep"**. Deletion is
  explicit and tooled:
  * `VOICELINK_TAP_RETENTION_DAYS=N` — at server start, taps older than N days are
    removed via an atomic temp-file rewrite (`mac/server.py:587-590`,
    `mac/privacy.py:apply_retention`, `purge_taps`). `0`/unset keeps everything.
  * `python3 privacy.py purge --taps [--older-than DAYS] [--yes]` — same logic,
    on demand, with confirmation.
  * `python3 privacy.py pseudonymize --in taps.jsonl --out taps.pseudo.jsonl
    [--drop-transcripts]` — produces a shareable copy in which every name and
    alias from the personal-context profile is replaced by a stable `PERSON_n`
    token, the client IP is redacted, the embedded `personal_context` is stripped,
    and (optionally) all transcript text is replaced by `[REDACTED]`
    (`mac/privacy.py:Pseudonymizer`).
* **Personal context**: a local JSON file that is read at startup
  (`mac/server.py:585`) and passed into the seam in memory. It is gitignored and is
  never written into an export by the pseudonymizer. To delete: remove the file.
* **Verification**: `privacy.status()` scans the whole repository for raw-media
  extensions (`.jpg .jpeg .png .wav .pcm .mp4 .webm .m4a .flac .raw`) outside the
  two legitimate image directories (`symbols/` pictograms, `figures/` evaluation
  plots) and reports `no_media_artifacts` (`mac/privacy.py:scan_media_artifacts`).
  `mac/tests/test_privacy.py::test_real_repo_has_no_media_artifacts` asserts this
  on every test run.

## 3. Facial data

**No facial detection, recognition or identification exists anywhere in the
system.** There is no face model, no embedding store, and no face-related code
(`grep -ri face mac/ firmware/` finds nothing functional). The only vision model is
a scene describer prompted to write one sentence about "what they might want to talk
about or ask for" (`mac/perception/config.py:35-39`); frames are downscaled to at
most 768 px before the VLM (`:41`) and discarded after the sentence is produced.
The sentence may mention that "a person" is present (e.g. the mock describer's
"an open front door with a person arriving", `mac/perception/scene_describer.py:84`),
but never who. `privacy.status()` reports `"facial_recognition": false` as a
declared invariant; any future change that adds a face model must update that
field and this document.

## 4. Kill switches (how a bystander or the user stops capture)

* **Hardware**: a physical button on GPIO 14 (internal pull-up, active-low,
  50 ms debounce) toggles `capturing` (`voicelink_capture.ino:78-80, :113,
  :264-284`). While paused, the I2S DMA is still drained but nothing is converted
  or sent (`:220-224`), and no frames are grabbed. A red on-board LED is lit
  whenever capture is active, as a visible signal to people nearby
  (`:68-76, :278, :323`).
* **Software**: `POST /api/privacy/pause` toggles `CAPTURE_PAUSED`
  (`mac/server.py:451-461`); while set, the wearable client drops every incoming
  binary message before it reaches any buffer (`mac/perception/wearable_client.py:26,
  :40-41`). `GET /api/privacy/status` reports `capture_paused`.
* **Reconnect-safe**: the Mac client reconnects with backoff, so pausing/resuming on
  the device never requires a server restart (`wearable_client.py:22-48`).

## 5. Consent policy

* **Current status: team-only testing.** The system processes audio and video that
  may include other people, so until written proxy consent and pseudonym handling are
  in place, it is built and tested on the project team only (CLAUDE.md guardrail).
* **Proxy consent**: `python3 privacy.py export-consent-template` writes
  `docs/consent/PROXY_CONSENT_TEMPLATE.txt`, a plain-text consent form for a
  guardian/proxy of a participant who cannot consent for themselves. It describes
  what is captured, what is kept, the LED + button, and the right to pseudonymize
  or delete. It must be reviewed against the local ethics/IRB process before use.
* **Bystanders**: the LED and the button are the in-person disclosure mechanism;
  the consent text commits to telling bystanders the device is active.
* **AAC datasets**: the evaluation harness supports a gated real-conversation
  dataset (AACConversations) whose access is license-gated; the committed testbeds
  are synthetic (see §6).

## 6. Bias and fairness considerations

* **Synthetic evaluation scenarios**: the 120 Phase-3 scenarios were generated by
  the local Qwen model from templated seeds (`mac/evaluation/phase3/scengen.py:10,
  :236`) and the Phase-4 teacher corpus (600 rows) by the same local LLM. They
  reflect the authors' and the model's assumptions about AAC situations, not real
  user distributions. This is stated as a limitation, not hidden.
* **LLM judge is a proxy**: the Phase-3 judge is a 3x temperature-varied local Qwen
  ensemble, labelled `PROXY — LLM judge, NOT ground truth` and explicitly not a
  gate; it flags a too-lenient or zero-variance judge
  (`mac/evaluation/phase3/judge.py:1-15, :33, :104`). Gates run on deterministic
  metrics (`metrics3.py`).
* **Closed-vocabulary ceiling**: the distilled student can only output words from a
  fixed vocabulary that covers ~32.5% of what the teacher produces; this is documented
  as the root cause of the remaining coverage gap rather than papered over
  (`mac/evaluation/phase4/INVESTIGATION_useless_boards.md`, `docs/VERIFIED_NUMBERS.md`).
* **Speaker/addressed heuristics**: "who is speaking to the user" is a loudness +
  question-word heuristic (`mac/perception/audio_worker.py:31-62`), which will
  misfire for soft-spoken partners or non-English phrasing. English-only Whisper
  (`base.en`) is a known language limitation.
* **Pictograms**: ARASAAC symbols are served locally (`mac/server.py:673`); no
  user-specific imagery is generated.
* **Tap logging for later measurement**: every tap is logged with its board so that
  real preference data, not synthetic proxies, can eventually drive the ranker.

## 7. Security

* **LAN-only by design.** The wearable hosts its own Wi-Fi access point and talks
  only to the Mac (`voicelink_capture.ino:307`; `mac/perception/config.py:12`). The
  Mac server binds `0.0.0.0:8080` so the iPad on the same Wi-Fi can reach it
  (`mac/server.py:62-63`); nothing is exposed beyond the local network and no
  outbound cloud calls exist in the runtime path (the only remote fetch in the repo
  is the one-time pictogram download, `mac/fetch_symbols.py`).
* **Plain HTTP, deliberately.** The board page and its WebSocket must be same-origin
  on a LAN; a self-signed HTTPS page would block a `ws://` socket as mixed content
  and a public CA cannot issue for a private IP. Plain HTTP on a private network is
  therefore the honest choice (`mac/server.py:6-7`). The threat model is a hostile
  device on the same Wi-Fi; mitigation is the AP password and physical control of
  the network, not TLS.
* **No secrets in the repo**: `.env`, keys and the personal profile are gitignored
  (`.gitignore:1-5, :17`).
* **Models are local and pinned** by repo id (`mac/ranker_config.py:31-33`,
  `mac/perception/config.py:19, :34`) and must already be cached; the server never
  downloads at runtime.
* **Upload bound**: the console image route caps uploads at 25 MB
  (`mac/server.py:79`).

## 8. Environmental considerations

* The runtime uses small models chosen to fit on one laptop: Whisper `base.en` in
  INT8 on CPU (`mac/perception/config.py:19-21`), a 3B 4-bit VLM (`:34`) and a 3B
  4-bit LLM (`mac/ranker_config.py:31-33`).
* The VLM is gated by a 64x64 grayscale change detector and a 2 s minimum interval
  so it does not run on every frame (`mac/perception/change_detector.py`,
  `mac/perception/config.py:43-47`).
* The distilled Phase-4/5 student replaces the LLM on the hot path with a MiniLM
  encoder + small MLP, quantized to INT8 with NNCF for the Intel target
  (`mac/intel/quantize.py:1-7`), reported as a ~156x speedup on the real MMR path
  (`docs/claude/working-notes/phase5-competition-ml/pipeline-log.md:10`).

## 9. Generative-AI tool transparency

Tools used in building VoiceLink, as evidenced by the repository:

| Tool | Role | Evidence |
|---|---|---|
| Claude Code (Anthropic) — Fable 5, Opus 4.8, Sonnet, Haiku | Primary development assistant: research, planning, implementation in worktrees, code review, edge-case tests. Multi-agent process documented per phase. | `CLAUDE.md` "Delivery process"; `.claude/skills/fable-pipeline/SKILL.md`; `docs/claude/working-notes/*/pipeline-log.md`, `FINAL_REVIEW.md` |
| Codex (OpenAI) | Cross-review of diffs and rescue passes during the review loop. | `docs/claude/working-notes/phase5-competition-ml/pipeline-log.md:8-10`; `.claude/skills/codex*` |
| Qwen2.5-3B-Instruct (4-bit, MLX) | **System component**: the vocabulary ranker behind the seam; also the teacher for distillation and the generator of synthetic evaluation scenarios. | `mac/ranker_config.py:31-33`; `mac/evaluation/phase3/scengen.py:10`; `mac/evaluation/phase4/data.py:1-5` |
| Qwen2.5-VL-3B-Instruct (4-bit, MLX) | **System component**: scene describer. | `mac/perception/config.py:34` |
| Whisper base.en (faster-whisper, INT8) | **System component**: speech-to-text. | `mac/perception/config.py:18-21` |
| Local Qwen ensemble as LLM judge | Proxy evaluation signal only, never a gate. | `mac/evaluation/phase3/judge.py` |
| GPT-family models | Not used as a runtime component or as the committed eval judge. OpenAI models appear in this project only via Codex as a code reviewer (above); the judge interface allows a Claude or human backend to be dropped in (`judge.py:10-11`) but the committed default is local Qwen. | `grep -ri openai mac/` finds no runtime usage |

All figures, written deliverables and code produced with these tools were reviewed
by the human author; the competition video voice-over is recorded by the team, not
synthesized.

## 10. Quick reference

```
python3 mac/privacy.py status                                   # what is kept, where; media scan
python3 mac/privacy.py purge --taps --older-than 30 --yes       # rotate the tap log
python3 mac/privacy.py purge --taps --yes                       # delete it entirely
python3 mac/privacy.py pseudonymize --in mac/taps.jsonl --out taps.pseudo.jsonl --drop-transcripts
python3 mac/privacy.py export-consent-template                  # docs/consent/PROXY_CONSENT_TEMPLATE.txt
VOICELINK_TAP_RETENTION_DAYS=30 python3 mac/server.py           # enforce retention at startup
curl -X POST http://<mac>:8080/api/privacy/pause                # software kill switch (toggle)
```

Tests: `mac/tests/test_privacy.py` (15 tests) cover purge, retention, pseudonymization,
the media scan on this checkout, the consent template, and both routes.
