# VoiceLink — Competition Evidence Dossier (Intel AI Global Impact Festival)

Every claim below is cited or measured in-repo. Rubric: Metric 01 Enriching Lives
(30), Metric 02 AI Innovation (30), Metric 03 Intel Technologies (30), IDR (3).
Full source list: `docs/claude/working-notes/phase5-competition-ml/research-rubric.md`.

## Metric 01 — Enriching Lives: Impact & Inclusion

### The problem, with evidence
- 1 in 31 U.S. 8-year-olds is identified with autism spectrum disorder (CDC ADDM,
  2022 surveillance year, published April 2025) — up from 1 in 36 two years earlier.
  [CDC MMWR](https://www.cdc.gov/mmwr/volumes/74/ss/ss7402a1.htm)
- 26.7% of children with ASD have profound autism — nonverbal, minimally verbal, or
  IQ < 50 (CDC's first profound-autism prevalence estimate).
  [Autism Science Foundation](https://autismsciencefoundation.org/press_releases/cdc-profound-autism-statistics/)
- 4–5 million people in the U.S. alone (1.3–1.5% of the population) cannot reliably
  meet daily communication needs with natural speech. AAC need spans dementia
  (23.2%), Parkinson's (22.7%), ASD (18.9%), learning disabilities (13.3%),
  stroke/aphasia (9.9%).
  [CommunicationFIRST, AAC Counts II](https://communicationfirst.org/wp-content/uploads/2023/03/AAC-Counts-Report-Part-2-What-Existing-Data-Can-and-Cannot-Tell-Us.pdf)
- Access barriers: dedicated speech-generating devices cost $4,000–$12,000 with
  3–18-month reimbursement cycles; 30–35% of devices are abandoned within the first
  year, with boards that don't match the user's real context cited as a driving
  cause. [AAC device abandonment](https://www.tandfonline.com/doi/full/10.1080/07434618.2023.2199859)

### Why it is time-critical
- AAC users communicate at ~10–18 words/minute versus 125–185 wpm for natural
  speech — about 10% of conversational speed. Conversation partners disengage while
  the user navigates menus to find words someone should have pre-programmed.
  [ASHA Leader](https://leader.pubs.asha.org/do/10.1044/2020-1016-aac-awareness/full/)
- Children who remain minimally verbal past age ~5 face persistently worse
  social/educational outcomes; 2024 meta-analytic evidence shows aided AAC embedded
  in natural context produces the strongest language gains — the *context* the
  vocabulary appears in matters, not just the device.
  [Springer 2024 meta-analysis](https://link.springer.com/article/10.1007/s10803-024-06382-7)
- VoiceLink attacks exactly this measured deficit: the right words appear on the
  board at the moment the conversation needs them, with no manual pre-programming.

### Target audience, accessibility, scalability
- Target: nonverbal and minimally-verbal AAC users (autism-first, generalizing to
  aphasia/ALS/Parkinson's) and their families/SLPs, who currently hand-program boards.
- Equivalent UX: the board is a plain single-origin web page (any tablet/phone), 10
  large tap targets, 4 always-present safety words (yes / no / I need help / wait)
  pinned deterministically — the model can never remove the user's safety vocabulary.
- Scalability/GTM: commodity hardware (ESP32-S3 ≈ $15 wearable + any existing tablet
  + one local computer), no cloud subscription, no per-seat license; the deployment
  target is an Intel NUC-class mini-PC. Costs orders of magnitude less than the
  $4,000–$12,000 devices in the market it augments.

### SDG alignment (cite targets, not just goals)
- SDG 10 (Reduced Inequalities), Target 10.2: "empower and promote the social,
  economic and political inclusion of all, irrespective of age, sex, disability…" —
  the most literal fit. [UN DESA](https://social.desa.un.org/issues/disability/sustainable-development-goals-sdgs-and-disability)
- SDG 4 (Quality Education), Targets 4.5/4.a: inclusive, disability-sensitive
  learning environments — VoiceLink surfaces classroom vocabulary in real time.
- SDG 3 (Good Health & Well-Being), tertiary framing.

## Metric 02 — AI Innovation: Application & Implementation

### Why AI is necessary
Static AAC boards cannot know the user is on a bus being asked "where would you like
to sit?" Deciding *which 10 of thousands of possible words* belong on the board right
now requires multimodal understanding (vision + speech + personal context). No
rule-based system enumerates real life.

### The AI, honestly described
- Speech: Whisper (base.en) transcribes the conversation around the user, locally.
- Vision: Qwen2.5-VL-3B describes the scene from the wearable camera, locally.
- Ranking: a two-stage architecture — an LLM teacher over-generates candidates and
  MMR-selects a communicatively diverse board (answer-quota reservation for
  questions); a distilled 769-d MLP student supplies relevance candidates that the
  same symbolic MMR policy then selects over, producing a board in 47 ms against
  the teacher pipeline's 6.7 s (142.6×, `mac/evaluation/phase4/phase4_results.json`)
  while retaining 94% of the teacher's communicative-function diversity — though
  NOT the teacher's specific word choices (dynamic overlap 0.056; see
  `docs/VERIFIED_NUMBERS.md` before quoting any figure here).
- Sub-domains demonstrably in play: ASR (Whisper), CV/VLM (Qwen2.5-VL), NLP
  embeddings (MiniLM), knowledge distillation, diversity-aware ranking (MMR),
  post-training quantization (NNCF INT8).

### Novelty (grounded, not hyped)
- Published AAC prediction research (KWickChat IUI 2022, PrAACT 2024) optimizes
  keystroke savings; no surveyed commercial AAC product (TD Snap, Proloquo2Go,
  CoughDrop) publishes a candidate-set diversity metric or context-driven board
  composition of this kind. VoiceLink's measurable contribution: a
  wearable-context-to-board pipeline evaluated with anti-gaming slate metrics
  (coverage-of-intent on dynamic words, distinct-communicative-functions@7,
  same-function-pair rate) — more rigorous than the published baselines we found.
  Sources in `research-aac-ml.md`.

### Data: obtained and analyzed
- Training data is synthetic-but-audited: authored scenario corpora + teacher
  (LLM+MMR) boards, 600 rows, category-capped, embedding-deduplicated (cosine 0.95)
  to prevent template twins straddling the train/test split, split fingerprinted so
  a drifted corpus refuses to score against a stale partition.
- Every vocabulary tap in the live UI is logged with a context snapshot (taps.jsonl)
  — the consented-pilot dataset that eventually replaces synthetic evaluation.
- We publish our failures: v2.1 showed more data made teacher-agreement WORSE
  (falsifying our data-volume hypothesis); the Phase-4 investigation documents how
  our own metrics were gamed and how we fixed the gates
  (`mac/evaluation/phase4/INVESTIGATION_useless_boards.md`).

### Ethics (engineered responses to named, documented risks)
- Bystander privacy (wearable camera/mic captures people who did not consent — an
  open research problem [arXiv "Mind the Gap"](https://arxiv.org/pdf/2603.04930)):
  ALL inference is local (nothing leaves the Mac), buffers are ephemeral (no
  persistent audio/video recording), the wearable has a hardware kill switch, and
  testing is restricted to the team until written proxy consent + pseudonym handling
  exist.
- Vulnerable-user autonomy (AAC users are "enthusiastic but cautious" about LLM
  assistance, [arXiv 2501.10582](https://arxiv.org/pdf/2501.10582)): the model
  *suggests*; it never speaks for the user. Four safety words are pinned
  deterministically outside model control. The user's tap is always the utterance.
- Regulatory posture: designed to the EU AI Act high-risk bar (transparency,
  human oversight, logging) even though assistive AAC is not explicitly Annex III
  ([EU AI Act Art. 6](https://artificialintelligenceact.eu/article/6/)).
- Accessibility standard: WCAG 2.2 informs the board UI, while acknowledging W3C's
  own caveat that conformance alone does not guarantee cognitive accessibility
  ([W3C](https://www.w3.org/TR/WCAG22/)).
- Bias: vocabulary evaluation includes per-scenario-category breakdowns; the
  personal-context store (names, routines) never leaves the device.

### Readiness (honest tier)
- Working end-to-end prototype: wearable (ESP32-S3 camera+mic) → Mac pipeline →
  live web AAC board over WebSocket. Demonstrated in a controlled, team-only
  environment (consistent with the privacy guardrails above).
- Controlled testing: 120-scenario self-test harness with published metrics; eval
  gates run in CI fashion on every training run.
- We do not claim full-scale deployment. Next step is a consented pilot. (Prior
  global winners claimed the same tier — see research-rubric.md.)

## Metric 03 — Use of Intel Technologies

Narrative (mirrors the measured-artifact pattern of the 2025 global winner):
1. **Development**: models exported to OpenVINO IR (student ranker from ONNX opset
   17; MiniLM via optimum-intel; Whisper base.en via optimum-cli) and validated on
   the OpenVINO ARM CPU plugin during development — the same IR artifacts run
   unchanged on Intel silicon.
2. **Optimization**: NNCF post-training INT8 quantization of the student ranker and
   embedder with real calibration data from our eval scenarios; before/after
   accuracy AND latency tables generated by `benchmark_app` and committed at
   `mac/intel/BENCHMARKS.md` (our own measured numbers, not vendor claims).
3. **Deployment target**: Intel Core Ultra-class mini-PC as the household hub —
   chosen because OpenVINO gives CPU-only real-time inference (no discrete GPU in a
   living room), and the identical IR runs on it. Intel-hardware benchmark numbers
   via Intel Tiber AI Cloud are the flagged next step (account signup pending).
4. **Why Intel specifically**: the seam (`rank_vocabulary` as the single model entry
   point) was designed so the inference backend swaps by config flag
   (`VOICELINK_STUDENT_BACKEND=openvino`) — OpenVINO is not a bolt-on; it is the
   deployment backend of the architecture as designed.

## IDR Participation (3 pts) — OPEN ITEM FOR THE TEAM
Confirm with festival organizers whether Intel AI for Youth / a public Intel course
counts; independent-team enrollment mechanics could not be verified online
(intel.com blocks automated access). Do this by email early — it is 3 free points.

## Open items requiring a human
1. Intel Tiber AI Cloud signup (browser + account) → run `mac/intel/bench.py` on a
   Xeon/Core Ultra instance and add the Intel-hardware column to BENCHMARKS.md.
2. IDR participation confirmation (above).
3. Reconcile the festival's 2025 scale numbers before quoting any publicly
   (sources conflict: "32 countries / six global winners" vs "109 projects / 33
   countries / 47 awards").
