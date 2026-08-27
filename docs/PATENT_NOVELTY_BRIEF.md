# VoiceLink — Patent Novelty Research Brief

**Prepared:** 2026-08-08
**Purpose:** hand this to a research agent to perform a prior-art / novelty search in
support of a **US non-provisional utility patent application**.
**Source of truth:** the VoiceLink repository (private as of this date), branch
`phase4-distillation`, HEAD `b57cbaf`. Every technical assertion below is traceable to a
named file. Where a number is disputed inside our own repo, that is said explicitly.

---

## 0. Instructions to the researching agent

You are being asked to determine whether the concepts in §4 are **novel (35 U.S.C.
§102)** and **non-obvious (§103)**, and whether they are **patent-eligible (§101,
Alice/Mayo)**. You are not being asked to be encouraging. A confident, well-cited
"this is anticipated by X" is worth more to us than an optimistic maybe.

For **each** numbered concept in §4, return:

1. **Closest prior art** — patents (US, EP, WO, CN, JP), published applications,
   academic papers, commercial products, open-source projects, product manuals,
   conference demos, YouTube demos, and thesis work. Cite publication numbers and
   dates. Patent applications published *before* our priority date count as prior art
   even if never granted.
2. **An anticipation verdict** — does any single reference disclose every element?
   Quote the disclosing passage.
3. **An obviousness verdict** — could a PHOSITA (person having ordinary skill: an
   ML engineer with assistive-technology domain exposure) combine two or three
   references to reach this? Name the combination and the motivation to combine. Be
   adversarial; this is where most of these will die.
4. **A survivability rewrite** — if the concept as stated is anticipated, what
   narrower limitation *would* distinguish over the art? This is the most valuable
   thing you can give us.
5. **§101 risk rating** — is this claim, as phrased, an abstract idea (organizing
   information / mental process) dressed in generic computing? What hardware or
   technical-improvement hook rescues it?

Return a single ranked table at the end: concept → novelty confidence (high /
medium / low / dead) → recommended claim placement (independent / dependent / drop
into the specification only as support).

**Do not** treat our own repository, our own competition dossier, or this brief as
prior art authority. Our dossier's novelty claims (`docs/COMPETITION_EVIDENCE.md`
§"Novelty") were written for a competition, not a patent examiner, and were based on a
limited survey. Re-derive everything.

---

## 1. Field of the invention

Augmentative and Alternative Communication (AAC). Specifically: a wearable,
multimodal, locally-inferenced system that dynamically composes the vocabulary
displayed on an AAC selection board in real time, from ambient scene and speech
context, for a nonverbal or minimally-verbal user.

Adjacent fields the search must also cover, because that is where the killer art
will live:
- predictive text / next-utterance prediction for disabled users
- recommender-system **slate diversification** (this is the biggest §103 threat —
  see §5)
- context-aware / sensor-driven user interface adaptation
- egocentric (first-person) wearable camera scene understanding
- knowledge distillation of a large model's *ranking policy* into a small edge model

## 2. The problem being solved (for the background section)

An AAC board is a grid of tappable words/phrases. A caregiver or SLP programs it in
advance. It is therefore always a guess made hours or years before the moment it is
used. When the moment arrives — someone asks the user a question they have no button
for — the user cannot answer. The failure mode is not slowness; it is **absence**.

Two consequences drive the technical design:

- **The user cannot be given "the most likely word."** A conventional predictive
  system maximizes P(next token | context) and returns the mode. If the mode is wrong,
  the user has nothing. What the user needs is a *slate that covers the space of
  plausible replies*, so that whichever one they meant is reachable in one tap. This
  inversion — optimizing coverage/spread rather than likelihood — is the conceptual
  core of the invention.
- **The user is slow.** Composing a reply by tapping takes a nonverbal user many
  seconds to minutes. In that window, the conversation moves on: new speech arrives,
  the scene changes, and a naive "latest context" system throws away the very question
  the user is still trying to answer. The system therefore needs explicit *state* about
  which conversational obligation is still open.

## 3. System architecture (needed for the claim's physical anchoring)

Three physical nodes, one software boundary.

**(a) Body-worn capture device.** ESP32-S3 (Freenove ESP32-S3-WROOM CAM,
CAMERA_MODEL_ESP32S3_EYE pin map, OPI PSRAM) with an OV-series camera and an SPH0645
I2S MEMS microphone. It performs **zero** inference. It hosts its **own Wi-Fi SoftAP**
(so the system works with no router and no infrastructure network) and acts as the
WebSocket *server*; the compute node connects to it as a client. It multiplexes both
modalities onto **one** WebSocket connection using a 1-byte type tag:
`0x01` + JPEG frame, `0x02` + 16-bit LE mono PCM. VGA JPEG at one frame / 2000 ms;
16 kHz audio in 512-sample chunks. The I2S audio read+send runs as a high-priority
FreeRTOS task **pinned to core 0**; camera capture and WebSocket servicing run in
`loop()` on core 1, so a slow JPEG grab cannot stutter audio at the source.
Privacy hardware: a physical capture-toggle button (debounced, internal pull-up) and
an always-on visible recording-indicator LED (on-board WS2812 driven red) whose state
is bound to capture state — the bystander-facing signal.
*File:* `firmware/voicelink_capture/voicelink_capture.ino`.

**(b) Local compute node.** A single machine (Mac today; an Intel mini-PC is the
stated deployment target) running the entire perception + ranking pipeline. Nothing
leaves this machine in normal operation.

**(c) Display node.** Any tablet or phone. It opens a **plain-HTTP** single-origin page
served by the compute node, with a WebSocket for live board updates. Plain HTTP is a
deliberate choice: it avoids the browser mixed-content wall that would block a `ws://`
connection from an `https://` page on a LAN device with no valid certificate.
*Files:* `mac/server.py`, `mac/web/index.html`.

**The seam.** Every model call in the entire system passes through exactly one
function:

```
rank_vocabulary(scene, transcript, context) -> ranked_words
```

The firmware never calls a model. The web UI never calls a model. They reach
intelligence only through the compute node, and the compute node only through this
function. This is what permits the entire inference backend to be swapped (Apple
silicon → Intel/OpenVINO) by a config flag with no change to firmware, routes, or UI.
*File:* `mac/rank_vocabulary.py`.

---

## 4. Candidate novel concepts

Ranked roughly by how strongly I believe they can carry weight. Each has: what it is,
how it is actually implemented (with the specific values a dependent claim would
recite), and why it might be non-obvious.

---

### N1. Diversity-first slate composition for AAC — over-generate, then select for coverage rather than likelihood

**What.** Instead of asking a model for the *k* most likely phrases, the system makes
**one** generation call that deliberately over-generates a wide candidate pool spanning
a fixed taxonomy of communicative acts, then runs a **second, model-free selection
stage** that picks the *k* displayed items to maximize *coverage of the reply space*.

**How.**
- The generation prompt is **slot-structured**: it demands exactly one phrase for each
  of 9 named communicative functions — request, reject, comment, question,
  share-information, express-feeling, social, direct-attention, clarify — plus 4 more
  phrases each on a *different* topic in play. 13 candidates, each returned tagged
  with `{word, function, topic}`. The prompt states explicitly that two phrases meaning
  the same thing is a failure. The slot structure exists specifically to defeat the
  model's default failure mode of **mode collapse** onto paraphrases of the single
  likeliest reply.
- Selection is greedy **Maximal Marginal Relevance** over 384-d MiniLM sentence
  embeddings: `score(i) = λ·rel(i) − (1−λ)·max_{j∈selected} cos(c_i, c_j)`,
  λ = 0.6 default, relevance = cosine to a composed query vector.
- MMR is augmented with a **communication-function tie-break**: among candidates whose
  MMR scores fall within TIE_EPS = 0.02, one whose function tag has not yet appeared on
  the board wins. Diversity is thus enforced in *two* spaces at once — continuous
  embedding space and a discrete pragmatic-function space.
- The selection stage is pure numpy, sub-millisecond on a ~20-item pool, so the board
  can be re-selected without a second model call.

*Files:* `mac/rank_vocabulary.py` (`generate_pool`, `select_board`),
`mac/selection.py` (`mmr_select`).

**Why it may be non-obvious.** The published AAC prediction literature we surveyed
optimizes **keystroke savings** — an explicitly mode-seeking objective. Diversity is a
*cost* under that metric. Applying a diversity-maximizing slate objective to AAC is an
inversion of the field's standard objective function, motivated by a domain fact
(a wrong mode leaves the user mute) that does not apply to able-bodied predictive text.

**Measured.** 120-scenario self-test, diverse vs. naive top-k arm: distinct
communicative functions 4.68 vs 3.88; near-duplicate pair rate 0.0004 vs 0.0274 (68×
lower); coverage-of-intent 1.00 vs 0.81.

**Known risk.** MMR is Carbonell & Goldstein 1998. Slate diversification is
extensively patented in recommender systems. See §5 — the researcher must attack this
concept hardest.

---

### N2. Held "active question" state with TTL and consumption-on-tap

**What.** A dedicated piece of conversational state that models the *open obligation*
the user has not yet answered, decoupled from the rolling transcript.

**How.**
- The audio worker classifies each finalized utterance. If it is (i) from the
  communication partner and (ii) a genuine question (ends in `?`, or opens with a
  wh-word from {what, where, when, why, who, how, which, whose}), it is **latched** as
  `active_question` with a timestamp.
- `active_question` is **not overwritten** by subsequent non-question chatter, even
  though that chatter *does* overwrite `last_utterance` and the rolling transcript.
  This is the point: the board keeps answering the question the user is still working
  on while the room keeps talking.
- It expires on a **TTL of 90 seconds** — "long enough for a nonverbal user to compose
  a reply, short enough that a stale question doesn't drive the board forever."
- It is **cleared by the user's tap**: `_record_tap()` calls
  `ctx.clear_active_question()`. The user answering *consumes* the obligation.
- Downstream, the presence of the `active_question` key changes the semantics of the
  entire pipeline: `_current_question()` returns *only* the held question on the live
  path (so a cleared/expired question cannot resurrect from `last_utterance`), while a
  direct/console call with no such key falls back to the transcript.

*Files:* `mac/context_state.py` (`set_active_question`, `clear_active_question`,
`snapshot`, `ACTIVE_QUESTION_TTL_S`), `mac/perception/audio_worker.py`
(`is_question_to_user`, `is_addressed`), `mac/server.py` (`_record_tap`).

**Why it may be non-obvious.** This is a mechanism whose entire justification is the
**asymmetric latency between the two conversational parties**. Ordinary dialogue-state
tracking assumes both parties respond in seconds and therefore tracks "current turn."
A latch with a minutes-scale TTL, immune to interleaved speech, and consumed by a UI
selection event, is a design that only makes sense when one speaker takes 30× longer
than the other. **I consider this the single most distinctive mechanism in the system.**

---

### N3. Deterministically pinned safety strip, excluded from the learned vocabulary and re-attached at inference

**What.** A hybrid deterministic/learned slate: four safety words are guaranteed on
every board, are outside model control entirely, and — critically — are **removed from
the learned model's output vocabulary** so that model capacity is spent only on
situation-specific words.

**How.**
- `GUARANTEED_CORE = ["yes", "no", "I need help", "wait"]`, pinned first, in fixed
  order, on every board. They are chosen to be **four distinct communicative
  functions** (comment / reject / request / direct-attention), which makes "the board is
  never mode-collapsed" *structurally* true even when the model collapses completely.
- Board size is 10, chosen explicitly so 4 pinned + 6 dynamic leaves real room for
  situation-specific content.
- In the distilled model (N5), `build_vocab()` **excludes** core words from the
  vocabulary, and `multihot()` skips them as learning targets. They are re-attached at
  inference by `assemble_board(core, dynamic, k)`. Our v1 measured a **0.571 agreement
  floor** created purely by core-forcing; excluding core removed that artifact and freed
  capacity.
- Board length is guaranteed exactly *k* by `_pad_to_k()`, which truncates if long and
  tops up from a static stub list if generation degraded.
- Multi-layer degradation: LLM failure → core pool + heuristic board; pool < 4 items →
  static stubs; student artifacts missing → fall back to the LLM diverse path. The user
  is never stranded without yes/no/help.

*Files:* `mac/rank_vocabulary.py` (`GUARANTEED_CORE`, `_split_core`, `_pad_to_k`,
`_heuristic_or_stub`), `mac/evaluation/phase4/data.py` (`build_vocab`, `multihot`,
`assemble_board`).

**Why it may be non-obvious.** Two separable ideas. (a) A safety floor on a
*generative* assistive UI implemented as pinned slots outside the model's reach — an
architectural rather than a prompting/RLHF safety mechanism, for a user who cannot
verbally correct a bad output. (b) Deliberately **deleting the guaranteed items from
the learned label space** because their determinism is otherwise learned as a
degenerate constant and inflates apparent accuracy. (b) is a specific, measured,
counterintuitive training decision.

---

### N4. Answer mode — reserved answer quota, spread-selected, with a hedge cap

**What.** When the system detects that the user has been asked a question, the slate's
composition rules change: slots are reserved for *direct answers*, those answers are
selected to be maximally *different from each other*, and the "I don't know" family is
capped.

**How.**
- Question detected → an answer block is appended to the generation prompt requesting
  6 additional candidates tagged `{"function": "answer"}`, each a *different possible
  answer*, explicitly instructed to span the real response range (an affirmative, a
  negative, two or three specific on-topic answers, an "I am not sure"), and told that
  the scene is only background — **follow the question**. Token budget widens from 520
  to 720 so the JSON array is not truncated mid-answer.
- Selection reserves up to `n−1` slots for answer-tagged candidates — never all of
  them, so at least one diversity slot always survives.
- Those reserved answers are picked by MMR at a **much lower λ (ANSWER_LAM = 0.35)**
  than the general board (λ = 0.6). Rationale, measured: at high λ the six most
  *relevant* answers are six paraphrases of one answer; a low λ forces "in the back" /
  "not seated" / "no table" to win distinct slots.
- **Hedge cap.** A shared canonical hedge cluster (`mac/evaluation/hedges.py`) defines
  the "I don't know / I'm not sure / no idea / maybe" family, and at most **one** hedge
  phrase may occupy a board (`HEDGE_CAP = 1`). The same cluster is used to *deny
  coverage credit* in evaluation — a board does not get scored as having covered the
  intent by offering "I don't know" when concrete answers were available. One
  definition, so the runtime cap and the metric can never disagree.

*Files:* `mac/rank_vocabulary.py` (`_answer_block`, `_select_dynamic`, `ANSWER_LAM`),
`mac/evaluation/phase4/infer.py` (`_reserve_answers`, `HEDGE_CAP`, `ANSWER_DUP_GUARD`),
`mac/evaluation/hedges.py`.

**Why it may be non-obvious.** Slate-diversification art diversifies the whole slate
uniformly. Here the slate is **partitioned by conversational role** — reserved
answer slots at one diversity temperature, general-coverage slots at another — with the
partition triggered by a detected speech act. Plus an explicit anti-degeneracy cap on a
semantically-defined escape-hatch cluster, which I have not seen elsewhere.

---

### N5. Distilling a *slate policy* by learning relevance only and re-applying the symbolic selection policy at inference

**What.** A two-tier teacher/student where the student replaces only the expensive
*candidate generation*, and the exact same MMR + answer-quota + hedge-cap selection
policy runs on top of the student's output. The learned component supplies relevance;
the symbolic component supplies diversity.

**How.**
- **Teacher:** the LLM pool generation + MMR selection of N1/N4.
- **Student:** a small MLP — `LayerNorm → Linear → ReLU → Dropout → Linear` — over a
  fixed dynamic vocabulary (94 words at last training), trained multi-label with
  sigmoid + BCE on multi-hot teacher boards. Deliberately ONNX-clean (LayerNorm is
  opset-17 and in the OpenVINO operator set) so the export path is guaranteed.
- **The key inference-time decision:** the student's raw `torch.topk` board was
  measured to be *useless* — it produced walls of hedges and same-function filler
  (documented in `mac/evaluation/phase4/INVESTIGATION_useless_boards.md`). The fix was
  **not** more training. It was to treat the student's logits as a *candidate pool*
  (top `pool_mult × k_dyn`, pool_mult = 3) and re-run the production MMR, the answer
  quota, and the hedge cap over it. Stated in the code: "the trained model provides
  relevance candidates, the selection provides the diversity the raw argmax threw away."
- The dynamic-selection stage additionally classifies each candidate into a
  communicative function *at inference* (embedding nearest-exemplar, N7) and feeds
  those labels to MMR's tie-break — because omitting them measurably degraded
  same-function-pair rate from 0.168 (teacher) to 0.197.
- Both paths sit behind the same seam; `strategy="student"` falls back to the LLM
  diverse path if student artifacts are missing.

*Files:* `mac/evaluation/phase4/student.py`, `mac/evaluation/phase4/infer.py`,
`mac/rank_vocabulary.py` (`_student_board`, `_load_student`).

**Why it may be non-obvious.** Standard distillation tries to make the student
reproduce the teacher's *output*. Here the teacher's output is a *set* whose value is
its internal structure, and the finding is that a small student cannot learn that
structure but *can* learn the pointwise relevance the structure is built from — so you
distill half the pipeline and keep the other half symbolic. Related to but distinct
from "learning to rank then re-ranking": the re-ranker here is not learned, it is the
teacher's own deterministic policy, reused verbatim.

**Honest caveat for the specification.** Teacher-agreement is *poor*
(`dynamic_overlap` 0.056; full top-7 overlap 0.59). The student is fast and diverse but
is **not** a faithful policy match. And a documented negative result: 5×-ing the
training data made agreement **worse** (0.232 → 0.165), falsifying the data-volume
hypothesis. Do not let a drafter write an unsupported "the student faithfully
reproduces the teacher" into the spec.

---

### N6. Multimodal query weighting that subordinates vision to the open question

**What.** A specific fix for a named multimodal failure mode: a vivid visual scene
hijacking the answer to a spoken question.

**How.**
- The MMR relevance query is a weighted sum of separately-embedded parts:
  **transcript/question 0.7, scene 0.2, personal-context blurb 0.1**, then L2
  normalized. On the live path the *held question alone* fills the transcript slot, so
  later chatter in the rolling transcript cannot dilute it.
- The same subordination is enforced *lexically* in topic extraction: content terms
  are taken from the question and speech **first**, and the scene may fill at most
  `_SCENE_TOPIC_CAP = 3` of 6 topic slots.
- And *again* in the prompt: the answer block states the scene is "only minor
  background context."
- Worked example from the code comments: asked "where do I sit?" in a restaurant, the
  board must produce seating answers, not reservation/menu vocabulary.
- Topic extraction is deliberately **model-free and pure** (regex content terms ≥4
  chars minus a stopword list) so it can never block the audio path.

*Files:* `mac/selection.py` (`build_query_vector`, `_Q_WEIGHTS`),
`mac/rank_vocabulary.py` (`topics_from_context`, `_SCENE_TOPIC_CAP`, `_answer_block`).

**Why it may be non-obvious.** Most multimodal fusion work tries to weight modalities
by confidence or learn the weights. This asserts a fixed *pragmatic* priority —
a direct conversational obligation outranks ambient perception — and enforces it
redundantly at three independent layers (query vector, topic list, prompt).

---

### N7. Communicative-function coverage measured by nearest-exemplar embedding classification, used at both train/eval and runtime

**What.** A tag-free way to count "how many different communicative moves does this
board offer," used both as an evaluation metric and as a live input to selection.

**How.**
- Nine functions, each anchored by **multiple short first-person exemplars** (e.g.
  `specific-answer`: "it is the blue one", "in the back", "at three o'clock", "carbon
  dioxide", "my head hurts", "a window seat", "home").
- A board item is classified by its **single nearest exemplar** (max cosine), **not**
  by a per-function centroid. Documented reason: a centroid is a blunt instrument — the
  "I don't know" centroid became an attractor that swallowed both "yes" and a real
  answer like "carbon dioxide"; "wait" snapped onto "refuse".
- It is generation-tag-free, so it scores the LLM arm, the student arm, and any
  baseline **identically** — it cannot be gamed by an arm that emits self-flattering
  tags.
- The same classifier is called *at inference* inside the student path to supply MMR's
  function tie-break.

*Files:* `mac/evaluation/phase3/taxonomy.py`, `mac/evaluation/phase3/metrics3.py`.

**Note.** Evaluation methods are hard to claim. The likely patentable framing is the
**runtime** use: classifying candidate phrases into pragmatic functions by
nearest-exemplar embedding and using that classification to enforce spread in the
displayed slate. Ask the researcher to search that framing, not the metric framing.

---

### N8. Event-driven board regeneration gated by conversational events, with a cheap-perception gate in front of expensive perception

**What.** The board is not recomputed on a timer. It is recomputed when the world
gives a reason, and the expensive vision model is fired far less often than frames
arrive.

**How.**
- **Vision gate:** each frame is downscaled to 64×64 grayscale and compared to a
  reference keyframe by mean absolute difference; the VLM fires only when the diff
  clears `CHANGE_THRESHOLD = 12.0` (of 0–255) **and** at least `VLM_MIN_INTERVAL_S =
  2.0` has elapsed. The reference re-baselines on every reported change so slow drift
  does not re-fire. Described in-code as "thermal/latency discipline."
- **Latest-frame-only:** frames go into a single-slot thread-safe holder, not a queue.
  A backlog is never processed; the newest frame always wins.
- **Board trigger:** a poll loop fires on `scene_changed OR addressed`, deduplicated by
  a signature built from the rounded relevant timestamps, and debounced, so the ranker
  is not hammered.
- **Non-blocking guarantee, enforced structurally:** audio (VAD + Whisper) and vision
  (change detect + VLM) are separate OS threads; model backends are constructed *inside*
  the worker threads so heavy loads never block asyncio startup; the ranker call is
  dispatched via `asyncio.to_thread` so the ~2.5 s inference cannot stall WebSocket
  heartbeats or incoming taps. Audio can never stall on vision, and neither can stall
  the UI.
- Utterance segmentation is a cheap energy VAD (30 ms frames, int16 RMS ≥ 350 = speech,
  600 ms trailing silence ends an utterance, <300 ms blips discarded) so Whisper runs
  on **complete utterances**, not fixed windows.
- Speaker role is a near-field heuristic: RMS ≥ 800 ⇒ "partner", else "other";
  `no_speech_prob > 0.6` or `avg_logprob < −1.2` ⇒ "ambient".

*Files:* `mac/perception/change_detector.py`, `mac/perception/vision_worker.py`,
`mac/perception/audio_worker.py`, `mac/perception/runner.py`, `mac/perception/config.py`,
`mac/server.py` (`trigger_loop`, `heartbeat_loop`, `_make_board`).

**Why it may be non-obvious.** Each piece individually (frame differencing, VAD,
thread isolation) is old. The candidate claim is the *composite scheduling discipline*
for a body-worn always-on multimodal assistive device: a cheap detector gating an
expensive one under a hard real-time constraint on a different modality, where the
displayed output is regenerated on detected conversational events rather than
periodically. Search this as a **combination**, and expect the researcher to find
close art in wearable-camera and smart-glasses power management.

---

### N9. Interaction logging that captures the full generating context of every selection

**What.** Every tap is logged together with the *complete context snapshot that
produced the board it was on*, correlated by board ID.

**How.** Each board is stamped with a UUID `board_id` and retained in a bounded
in-memory map alongside its `produced_from` = `{scene, transcript, context}`. A tap
message from the display carries `{word, source, client_ts, board_id}`; the server
joins them and appends a JSONL record containing server receive time, client
timestamp, the word, the source, the board ID, **the full context snapshot**, and the
full board the word was chosen from — i.e., both the positive and the implicit
negatives. The same tap event clears the active question (N2).
*Files:* `mac/server.py` (`_record_tap`, `_produce_and_broadcast`),
`mac/taps.jsonl` (121 rows at time of writing).

**Assessment.** Logging is ancient art. This is almost certainly **specification
support, not a claim** — but it matters as the closed loop (deployed taps become the
training corpus that replaces synthetic teacher data), and a dependent claim about
retraining the ranker on tap-with-context records may survive. Ask the researcher to
check "implicit feedback from slate selection" recommender art, which is dense.

---

### N10. Vocabulary-integrity guards for an index-mapped output vocabulary on a quantized edge model

**What.** Safety machinery ensuring a quantized model artifact can never serve *wrong
words* to a user who cannot verbally correct them.

**How.**
- The vocabulary is fingerprinted order-sensitively (`vocab_sha`, SHA-256 of the
  newline-joined vocab, truncated). The hash is stamped into the torch checkpoint and
  written as a **sidecar file next to the OpenVINO IR**.
- At load: output dimension is compared against vocabulary length, *and* the stamped
  hash is compared against the current vocabulary's hash — because a same-sized
  **reorder** passes a length check but silently remaps every index. Mismatch raises at
  load time rather than serving.
- Training-data splits are fingerprinted the same way, so a regenerated corpus refuses
  to score against a stale partition (silent train/test leakage prevention).
- Rows are embedding-deduplicated at cosine > 0.95 to stop template twins straddling
  the split; scenario categories are capped at 15% of rows.
- Vocabulary is semantically deduplicated by **single-link connected components** of
  the cosine ≥ τ graph — not greedy-against-kept — because the hedge paraphrases only
  unite *transitively* ("i don't know" ~ "i am not sure" = 0.784 < 0.80, but both link
  through "i'm not sure" at 0.841/0.943). Each component keeps its highest-frequency
  member and a **canonical-redirect map** sends every dropped member to its
  representative, so multi-hot label mass is *redirected rather than lost*. Non-hedge
  merges are logged for audit and any non-hedge component larger than 4 **fails loudly**
  rather than silently eating vocabulary.
- The OpenVINO CPU plugin is pinned to `INFERENCE_PRECISION_HINT: f32` because the ARM
  plugin silently defaults to f16 (measured 1.1e-2 logit drift) — so "FP32" means FP32
  and INT8 quantization is the only intended precision loss.

*Files:* `mac/evaluation/phase4/data.py` (`vocab_sha`, `dedup_vocab`, `dedup_rows`,
`cap_categories`, `rows_fingerprint`), `mac/rank_vocabulary.py`
(`_assert_vocab_matches`, `_load_student_openvino`), `mac/intel/__init__.py`.

**Assessment.** Individually mundane (hashing, checksums). The candidate claim is
narrow and safety-framed: *for a communication-assistance device whose model output is
an index into a word list, binding the artifact to a content-addressed fingerprint of
that word list and refusing inference on mismatch.* Worth one dependent claim. The
transitive-closure vocabulary dedup with canonical label-mass redirection is the more
technically interesting piece; search that separately.

---

### N11. Backend-swappable inference through a single fixed-signature boundary

**What.** One function signature, config-flag backend selection across radically
different runtimes (mlx-lm LLM / heuristic / distilled torch MLP / OpenVINO INT8 IR),
with firmware and UI untouched.

**Assessment.** This is excellent engineering and terrible patent material —
it is an abstraction layer, and abstraction layers are the definition of well-understood
routine activity. Include in the specification to support "the claimed method is
runtime-agnostic," and to support enablement across hardware. **Do not claim it.** If
the researcher finds it claimed by someone else, that is interesting but not a threat.

---

## 5. Prior art the researcher MUST clear (leads, not conclusions)

Treat everything here as an unverified lead. Confirm dates, numbers, and content.

**Commercial AAC.** Tobii Dynavox (incl. TD Snap), PRC-Saltillo / Prentke Romich
(LAMP, Words for Life), AssistiveWare (Proloquo2Go, Proloquo), CoughDrop, Smartbox
(Grid 3), TouchChat. Search these as **patent assignees**, not just as products —
Tobii Dynavox and PRC in particular have real portfolios around vocabulary
organization and prediction. Check their patents for context-driven or
sensor-driven page/vocabulary switching.

**Context-aware AAC academic work — highest-risk area.** There is a body of work from
the University of Dundee (Annalu Waller, Rolf Black and colleagues) on sensor-driven,
context-aware narrative/vocabulary support for AAC users — the "How was School Today"
line of work is the one I would look for first; it used sensed context (location,
proximity, schedule) to generate conversational content on an AAC device. If a
reference like that discloses "sense context → change what the AAC device offers," it
does not anticipate N1/N2/N4 but it badly hurts any broad claim to
"context-aware vocabulary for AAC." **Frame the independent claim assuming this art
exists.**

**LLM/ML AAC prediction.** KWickChat (IUI 2022). PrAACT (2024). Broader work on LLM
sentence completion for AAC, and on "how AAC users feel about LLM assistance"
(e.g. arXiv 2501.10582). Search 2023–2026 heavily; this area is moving fast and the
most dangerous art is recent and possibly unindexed by patent databases — check arXiv,
ASSETS, CHI, IUI, SLPAT, and the ACL Anthology directly.

**Wearable / egocentric assistive.** Cognixion (ONE headset — AR + BCI AAC; they hold
IP, check it carefully), Envision Glasses, OrCam MyEye, Meta Ray-Ban + AI, Google Look
to Speak, Apple Live Speech / Personal Voice / Assistive Access, Microsoft Seeing AI.
OrCam and Cognixion are the most likely to hold blocking hardware-context claims.

**Slate diversification — the biggest §103 threat to N1.** MMR (Carbonell &
Goldstein, SIGIR 1998). Determinantal point processes for recommendation diversity
(Google/DeepMind, ~2018). Netflix, Spotify, Amazon, Pinterest, and LinkedIn all hold
patents on diversified slate/feed generation. Expect an examiner to argue: "applying
known slate diversification (Ref A) to known AAC word prediction (Ref B) is obvious."
**Your job is to find the strongest available rebuttal**: is there a documented teaching
*away* — i.e., does the AAC literature explicitly optimize keystroke savings in a way
that treats diversity as harmful? A documented teaching-away is worth more than any
argument we can invent.

**Dialogue state / speech acts.** Speech-act classification, dialogue-state tracking,
open-question tracking in task-oriented dialogue, "unanswered question" detection in
meeting-assistant patents (Microsoft, Google, Cisco/Webex, Zoom). For N2, search
specifically for **timeout/TTL-bounded pending-question state** and **state cleared by a
user interface selection**. I expect meeting-assistant art to have "detect open
questions"; I do **not** expect it to have the TTL-latch-consumed-by-tap combination
motivated by asymmetric response latency.

**Distillation & re-ranking.** Hinton et al. 2015. "Distilling a ranker."
Retrieve-then-rerank architectures. Two-tower recommenders with a candidate-generation
and a ranking stage — Google's YouTube recommender papers/patents are the canonical
form and are structurally similar to N5's pool-then-select. Confirm whether any of them
teach *keeping the second stage symbolic/unlearned while distilling only the first*.

**Wearable perception scheduling.** Frame-difference-gated expensive vision on
battery-limited wearables; smart-glasses and body-cam power management; Snap/Meta/Apple
patents on activity-gated capture. This is where N8 is most likely to die.

---

## 6. Search apparatus

**CPC / IPC classes to sweep** (verify current definitions):
- `G09B 21/00`, esp. `G09B 21/009` — teaching/communication aids for disabled persons
- `G06F 3/023`, `G06F 3/0482`, `G06F 3/04842` — input arrangements, menu/list selection
- `G06F 40/274`, `G06F 40/279`, `G06F 40/56` — predictive text, NLP, NLG
- `G10L 15/22`, `G10L 15/26`, `G10L 15/18`, `G10L 13/00` — speech interaction, STT, TTS
- `G10L 25/78` — voice activity detection
- `G06V 20/20`, `G06V 20/52`, `G06V 10/764` — scene understanding, surveillance, classification
- `G06N 3/096` — transfer learning / distillation; `G06N 3/0495` — network compression/quantization
- `G16H 20/70`, `G16H 40/63`, `G16H 80/00` — health informatics
- `A61F 4/00` — devices for aiding persons with disabilities
- `H04L 67/12`, `H04W 4/80` — device-to-device, local networking
- `G06F 16/9535`, `G06Q 30/0631` — filtered/diversified recommendation

**Databases:** Google Patents, USPTO Patent Public Search (PatFT/AppFT), Espacenet,
WIPO PATENTSCOPE, Lens.org, plus non-patent literature (Google Scholar, arXiv, ACM DL,
ACL Anthology, IEEE Xplore). Also search **product manuals and YouTube demos** — a
public demo is §102 prior art and will never appear in a patent database.

**Query seeds** (expand and combine):
- `("augmentative and alternative communication" OR AAC) AND (diversity OR diverse OR "maximal marginal relevance" OR redundancy) AND (vocabulary OR phrase OR board)`
- `AAC AND ("wearable camera" OR "body-worn" OR egocentric OR "first-person camera") AND (vocabulary OR "communication board")`
- `("communication board" OR "symbol board" OR "speech generating device") AND ("dynamically" OR "context") AND (populate OR compose OR select)`
- `("pending question" OR "unanswered question" OR "open question") AND (timeout OR "time to live" OR expire) AND (dialog OR conversation) AND (display OR suggest)`
- `("slate" OR "candidate set" OR "result set") AND (diversif* ) AND ("communicative function" OR "speech act" OR pragmatic)`
- `distillation AND ("candidate generation" OR "first stage") AND ("re-rank" OR "selection policy") AND diversity`
- `("frame difference" OR "change detection") AND (gate OR trigger OR throttle) AND ("vision language model" OR "scene description") AND wearable`
- `"speech generating device" AND (assignee:"Tobii" OR assignee:"PRC-Saltillo" OR assignee:"AssistiveWare" OR assignee:"Cognixion")`

---

## 7. Statutory and procedural flags (please address these too)

**7.1 §101 / Alice — the biggest structural risk.** A claim reciting "receive context,
rank words, display words" reads as an abstract idea implemented on a generic computer
and will draw a §101 rejection. The rescue arguments available to us, which the
researcher should pressure-test against recent Federal Circuit and PTAB decisions:
- The claim is anchored in a **specific hardware topology** (a body-worn capture device
  performing no inference, hosting its own access point, multiplexing tagged modalities
  over one connection to a local inference node that also serves the display).
- The claim recites a **concrete technical improvement**: a cheap detector gating an
  expensive model to meet a thermal/latency budget; thread isolation guaranteeing a
  non-blocking audio path; an artifact-integrity check preventing incorrect output.
- The improvement is to the **operation of the device**, not to the abstract quality of
  the words (Enfish / McRO framing).

**7.2 §102(b) grace period and foreign rights — TIME-SENSITIVE.** Under AIA, an
inventor's own public disclosure starts a **12-month** clock to file in the US. Most of
the rest of the world (EPO, China, and others) requires **absolute novelty** — any
public disclosure before filing destroys those rights permanently.

Things to audit before anything else:
- The GitHub repository at `github.com/tanayvin1216/VoiceLink` returns 404 to an
  unauthenticated API request, which indicates it is **private**. Confirm this
  directly, and confirm it has been private for its entire history. A repo that was
  public even briefly is a publication.
- The Intel AI Global Impact Festival submission — is any part of it public? Does
  entering grant the organizer publication rights? **Read the competition terms for an
  IP-assignment or publication clause before submitting anything.** A competition entry
  that publishes your architecture is a §102 event and may also transfer rights.
- `docs/WEBSITE_CONTENT.md` describes a public product site aimed at clinics and
  faculty. **Do not publish that site before filing.**
- The planned demo video (Phase 6). Public posting = disclosure.
- Any conversations with autism centers, clinics, or faculty without an NDA.

**Ask the researcher to state plainly:** given the earliest commit of 2026-06-13 and
today being 2026-08-08, and assuming no public disclosure has occurred, are foreign
rights still fully available?

**7.3 Inventorship and AI assistance — read this carefully.** This codebase was
developed through a heavily AI-assisted multi-agent pipeline (documented in
`CLAUDE.md` and `docs/claude/working-notes/`). Under USPTO's February 2024 guidance on
AI-assisted inventions: AI-assisted inventions **are** patentable, an AI system **cannot**
be a named inventor, and a natural person must have made a **significant contribution to
each claim**. Merely prompting a system to produce the invention may not suffice; the
significant contribution can lie in constructing the problem, the architecture, and the
specific design constraints. Ask the researcher to summarize the current standard and
what contemporaneous evidence of human conception should be preserved. Given the
working notes document who decided what, this is a solvable problem — but it should be
handled deliberately, before filing, not discovered later.

**7.4 Provisional vs. non-provisional.** You've said non-provisional, and that is a
legitimate choice — it starts examination immediately and gets you to a granted patent
sooner. The tradeoff worth knowing once: a provisional is far cheaper, secures the
priority date within days, permits "patent pending," and buys 12 months to refine
claims while you gather pilot data and Intel-hardware measurements — after which the
non-provisional claims priority back. Given that several of the concepts above are
still moving (the student's teacher-agreement is poor, an eval gate is failing, no
Intel-hardware numbers exist yet), the 12-month window would be genuinely useful here.
That is your call to make; proceed as you've decided either way.

**7.5 Entity status.** If filing as an individual or a small operation, check micro
entity (up to ~75% fee reduction) and small entity status. Ask the researcher for the
current USPTO fee schedule for a non-provisional utility filing (filing + search +
examination), and the current cost bands for a registered patent attorney or agent in
this art unit.

---

## 8. Draft claim scaffolds (for the researcher to attack, not to file)

These are deliberately written to be searched against and torn apart. A registered
practitioner drafts the real ones.

**Scaffold A — system, anchored in the hardware topology (N1 + N3 + hardware).**
A communication assistance system comprising: a body-worn capture device including an
image sensor and a microphone, the capture device performing no inference and
transmitting tagged image and audio payloads over a single persistent connection on a
wireless network hosted by the capture device; a local computing node that transcribes
the audio and generates a scene description; the node generating a candidate pool of
user-speakable phrases in which each candidate is associated with one of a plurality of
predefined communicative functions; the node selecting a display set by an objective
that rewards relevance to the context and **penalizes similarity to already-selected
candidates**, subject to a plurality of predetermined safety phrases being included
unconditionally and independently of the generation; and a display device on the same
local network rendering the display set as selectable elements; wherein no context data
is transmitted beyond the local network.

**Scaffold B — method, on the open-question latch (N2 + N4). My pick for the
strongest independent claim.**
A method comprising: detecting from ambient audio an utterance directed at a user of a
communication device that constitutes a question; storing the question as a pending
conversational obligation with an associated timestamp; **maintaining that pending
obligation unchanged across subsequently received utterances that are not questions**;
composing a set of candidate responses selected to be **mutually dissimilar** and
responsive to the pending obligation, reserving a proper subset of display positions
for such responses; displaying the set; and **clearing the pending obligation in
response to either a user selection or the expiry of a predetermined interval** from the
timestamp — wherein the interval is selected to exceed the expected composition time of
the user.

**Scaffold C — method, on the hybrid distillation (N5).**
A method of operating a resource-constrained ranking model, comprising: training a
model to predict, from a multimodal context representation, per-item relevance scores
over a fixed vocabulary, using as targets the display sets produced by a larger
generative model; **excluding from that vocabulary a set of items that are pinned
deterministically at inference**; at inference, computing relevance scores, selecting a
candidate pool larger than the display set therefrom, and applying to that pool the
**same diversity-based selection policy applied to the larger model's output** — such
that the smaller model supplies relevance and the selection policy supplies diversity.

**Scaffold D — dependent limitations worth reciting.** λ = 0.6 general / 0.35 answer;
tie-break epsilon 0.02; 4 pinned safety phrases spanning 4 distinct communicative
functions out of a 10-position board; 90-second obligation TTL; 0.7/0.2/0.1 query
weighting; scene topic cap of 3 of 6; at most one phrase from a semantically-defined
epistemic-hedge cluster; 64×64 grayscale change gate at threshold 12/255 with a 2-second
minimum inter-inference interval; single-slot latest-frame retention; 30 ms VAD frames,
RMS 350, 600 ms end-of-utterance silence, 300 ms minimum utterance; artifact bound to a
content-addressed vocabulary fingerprint with refusal on mismatch.

---

## 9. Measured results (enablement support — with our own disputes marked)

Per `docs/HANDOFF.md` §10, these are **as recorded by the runs that produced them and
were not re-verified** when that document was written. Re-run before any of them goes
into a filed specification.

| Result | Value | Source |
|---|---|---|
| Diverse vs top-k, distinct communicative functions | 4.68 vs 3.88 | 120-scenario self-test |
| Diverse vs top-k, near-duplicate pair rate | 0.0004 vs 0.0274 (68× lower) | 120-scenario self-test |
| Diverse vs top-k, coverage-of-intent | 1.00 vs 0.81 | 120-scenario self-test |
| Student vs teacher latency | 43.6 ms (IQR 39.4–45.8) vs 6.82 s (IQR 6.77–7.02) → **156×** | `phase4_results.json` |
| Student diversity retention | distinct-fn 4.048 vs teacher 4.333 (ratio 0.934); near-dup 0.0006 | `phase4_results.json` |
| Student coverage-of-intent | 0.859 vs teacher 0.944; dynamic 0.775 vs 0.845 | `phase4_results.json` |
| Teacher agreement (**poor**) | full top-7 overlap 0.59; dynamic overlap 0.056 | `phase4_results.json` |
| **Failing gate** | same-function-pair 0.1882 vs teacher 0.1684 — **FAIL** | `phase4_results.json` |
| Negative result | 5× training data made agreement worse: 0.232 → 0.165 | Phase-4 report |
| ONNX → OpenVINO IR parity | 1.4e-6 | Phase-5 |
| NNCF INT8 | 2.7× smaller, top-6 overlap 0.950 mean | Phase-5 |
| whisper-base.en INT8 (OpenVINO) vs faster-whisper | 0.35 s vs 0.42 s | Phase-5 |
| Vocabulary / training rows | 94 dynamic words, 559 rows | `phase4_results.json` |

**Two disputes the drafter must resolve before filing:**
1. `docs/COMPETITION_EVIDENCE.md` states "~7 ms (~900× faster)". `phase4_results.json`
   measures **43.6 ms / 156×**. The 7 ms figure appears to be the bare MLP forward pass
   excluding embedding and MMR selection; the 43.6 ms figure is the end-to-end board.
   **Use 156× and the end-to-end number.** An unsupported speed claim in a specification
   is worse than a modest supported one.
2. **All measurements were taken on Apple ARM silicon. There are currently zero
   measurements on Intel hardware**, despite the OpenVINO path being a stated
   deployment target. Do not let the specification imply otherwise.

---

## 10. What I believe is honestly *not* novel

State this up front to the attorney; it saves money and credibility.

MMR itself. Knowledge distillation itself. Whisper / ASR. VLM scene captioning.
MiniLM sentence embeddings. INT8 post-training quantization / NNCF / OpenVINO. WebSocket
push UIs. ESP32 camera streaming and SoftAP hosting. Energy VAD. Frame differencing.
Pinning frequently-used "core words" on an AAC board — this is **standard AAC practice**
(core vocabulary is a well-established concept in the field and must be disclosed as
such; what may be novel is *which* four, chosen for functional orthogonality, and their
exclusion from the learned label space). Local-only inference for privacy. Config-driven
backend abstraction (N11).

The case, if there is one, rests on: **the inversion of the ranking objective for a user
who cannot recover from a wrong mode (N1)**; **the open-obligation latch built around
asymmetric response latency (N2)**; **the role-partitioned slate with differential
diversity temperature and an anti-hedge cap (N4)**; **the hybrid distillation that
learns relevance and keeps the selection policy symbolic (N5)**; and the specific
composition of all of these on the three-node local-only topology of §3.

---

## 11. Deliverable requested

One document containing: the per-concept analysis of §0 for N1–N10; the ranked
survivability table; a consolidated list of the 10–15 most dangerous references with
publication numbers and dates and the specific elements each discloses; explicit answers
to the §7 statutory questions; and a recommendation on which one or two concepts should
anchor the independent claims. Cite everything. Where you cannot find art, say "no art
found" and state what you searched — do not infer novelty from an empty search.
