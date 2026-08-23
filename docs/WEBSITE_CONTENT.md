# VoiceLink — Website Content Pack

Everything an agent needs to build the VoiceLink product site. Copy is written to be
used as-is; anything in `[brackets]` is a note to the builder, not text for the page.

**Audience:** autism centers and clinics, university faculty, and students —
people we would want to collaborate, pilot, or research with.
**Not the audience:** competition judges, hardware vendors, investors. Do not frame
the site around any competition, chip vendor, or funding narrative.

---

## 1. Positioning

**Product name:** VoiceLink

**One-line description (use in the hero, meta description, and social card):**
> VoiceLink puts the right words on an AAC board at the moment they're needed — by
> paying attention to the room the way a good communication partner does.

**Alternate one-liners** (pick one, stay consistent across the site):
- A wearable that keeps a nonverbal person's communication board in step with the
  conversation happening around them.
- Context-aware vocabulary for AAC users — no one has to program the board in advance.

**Elevator paragraph (about section, ~70 words):**
> VoiceLink is a communication system for people who are nonverbal or minimally
> verbal. A small device worn on the body notices what's happening around the
> wearer — the room, and the conversation in it. A nearby computer turns that into
> the vocabulary most likely to be useful right now, and a tablet shows it as a
> board of large, tappable phrases. The words change as the moment changes. Nothing
> ever leaves the room.

---

## 2. The story (the "why" section)

Use this as the narrative spine of the page. It is the strongest thing we have —
lead the site with the problem, not the technology.

**Section headline options:**
- "The words are always in the wrong place."
- "Communication shouldn't require predicting your day in advance."

**Body copy:**

> Around 4–5 million people in the United States can't reliably meet their daily
> communication needs with speech. Many of them use AAC — augmentative and
> alternative communication — a board of words and symbols they tap to speak.
>
> The boards work. The problem is that someone has to decide what's on them, ahead
> of time. A parent, a teacher, or a speech-language pathologist sits down and
> programs in the vocabulary they think the day will need. Then the day happens.
> The doctor asks a question nobody anticipated. A classmate brings up something
> new. The words that were needed weren't there, and the person navigates through
> menus looking for them while the conversation moves on without them.
>
> AAC users communicate at roughly a tenth of the speed of speech. A large part of
> that gap isn't the tapping — it's the searching.
>
> Around a third of dedicated communication devices are abandoned within a year of
> being provided, and boards that don't match the user's real life are a documented
> reason why.
>
> VoiceLink starts from a different assumption: that the device should be doing the
> anticipating, continuously, instead of asking a human to do it once a week.

**The insight, stated plainly** (works well as a pull quote):
> A good human communication partner doesn't guess the one thing you're about to
> say. They keep several plausible things within your reach. That's the job we gave
> the system.

---

## 3. What it does — the user's experience

[This is the section that should carry the demo visuals. See §7.]

**Headline:** "Same person. Same board. Three different moments."

**Framing sentence:**
> The board has two parts: a row of core words that never move, and a set of
> suggestions underneath that change with the situation.

**Walk-through copy:**

> **In a classroom.** The teacher asks the class what plants take in from the air.
> Underneath the core row, the board fills with ways into that specific moment —
> an attempt at the answer, a way to say *I'm not sure*, a way to ask for the
> question again, a way to say *I want to answer* so the teacher waits.
>
> **At a doctor's appointment.** The doctor asks where it hurts. The suggestions
> become locations on the body, an *I don't know*, a *can my mom answer*, a
> *this is too much right now*.
>
> **At dinner.** Someone offers more pasta. The board offers yes, no, a *just a
> little*, a *can I have something else*, a *I'm finished* — and a way to change
> the subject, because the person may not want to talk about the pasta at all.
>
> Nobody programmed any of those boards. The system built each one from what it
> could see and hear, in the moment.

**The core row — explain this, it's a genuine design decision:**

> Five words never move, never change, and are never chosen by a model:
> **yes · no · help · stop · I need**
>
> They are rendered once and stay in the same place forever, so that reaching for
> them becomes muscle memory. No matter how badly a suggestion goes, the person
> always has yes, no, and a way to stop what's happening. The model has no ability
> to remove them.

**Other things that are true and worth a line each:**
- Tapping a phrase speaks it aloud, in a voice the user picks.
- A **lock** button freezes the board. If the user is mid-thought, suggestions stop
  updating until they unlock it — the screen never reshuffles under their finger.
- A quiet indicator shows when new suggestions are waiting, rather than swapping
  them in unannounced.
- The board is a web page. It runs on whatever tablet or phone the family already
  owns — no proprietary hardware to buy or wait for.

---

## 4. How it works — technical, at a high level

Keep this section confident and skimmable. Faculty and students should be able to
tell in thirty seconds that the engineering is real, without a wall of jargon.

**Headline:** "Three pieces, one boundary."

**The three pieces:**

| Piece | What it is | What it does |
|---|---|---|
| The wearable | A small clip-on or pendant device with a camera and a microphone (an ESP32-S3, roughly $15 in parts) | Senses only. Streams what it sees and hears to the computer. It runs no models and makes no decisions. |
| The hub | An ordinary computer in the room — a laptop or a small mini-PC | Runs everything: speech recognition, scene understanding, and vocabulary ranking. |
| The board | A web page on any tablet or phone | Displays the vocabulary and speaks the taps. Holds no intelligence of its own. |

**The pipeline, in four steps:**

1. **Listening.** Audio is monitored continuously for complete utterances — not
   fixed-length chunks, but the natural boundaries of someone finishing a sentence.
   Those utterances are transcribed locally with a speech recognition model.
2. **Seeing.** The camera feed is checked cheaply for actual change. A full scene
   understanding model only runs when the scene really is different, which is what
   makes continuous vision affordable on a machine sitting in a living room.
3. **Ranking.** A vision-language model and the transcript, plus the user's own
   personal context, go into the vocabulary ranker. It doesn't ask for the most
   likely words. It deliberately over-generates candidates across nine distinct
   communicative functions — requesting, rejecting, commenting, asking, sharing,
   expressing feeling, socializing, directing attention, clarifying — and across
   the concrete topics actually in play, then selects a spread rather than a
   cluster. The result is a board where the options are genuinely different things
   to say, instead of seven rewordings of one guess.
4. **Learning.** Every tap is recorded with a snapshot of the situation that
   produced the board. That record is what will eventually let the system learn a
   specific person's voice rather than a generic one.

**The architectural detail worth naming** (this is the thing engineers will
appreciate — keep it to one short block):

> Every model call in the entire system passes through a single function:
>
> ```
> rank_vocabulary(scene, transcript, context) → ranked_words
> ```
>
> The wearable never calls a model. The web page never calls a model. They reach
> intelligence only through that one boundary. It means the models underneath can
> be replaced — a faster one, a smaller one, one tuned to a specific person —
> without touching the device, the display, or anything a user sees.

**What runs when the board updates** [Builder: keep this block. It is the section
that distinguishes VoiceLink from a prompt sitting behind a web page, and the
technical readers on this list will look for it.]

> Choosing the words is not a language-model call.
>
> We trained a small ranking model of our own — a compact network over sentence
> embeddings that scores a fixed working vocabulary — and paired it with a
> selection algorithm we wrote: it spreads the chosen phrases apart so they aren't
> near-duplicates, reserves slots for an actual answer whenever the partner asked a
> question, and limits how many ways to say *I don't know* can occupy the board at
> once. Large models handle speech and scene description. The decision about which
> words a person gets is made by our model and our selection rules, in tens of
> milliseconds, on an ordinary CPU with no dedicated graphics card.
>
> That matters for a device meant to sit in someone's kitchen and answer within a
> conversation's natural pause.

---

## 5. Privacy — give this its own section

This will be the first question every clinic and IRB asks. Answer it before it's
asked, and answer it structurally.

**Headline:** "The recordings don't exist."

> A device that watches and listens all day, worn by someone who may not be able
> to object to how it's used, is a serious thing to build. We designed around that
> rather than adding a policy on top of it.
>
> - **Everything runs locally.** Speech recognition, scene understanding, and
>   ranking all happen on the computer in the room. No audio, video, or transcript
>   is sent to any external service.
> - **Nothing is stored.** Audio and video pass through memory and are discarded.
>   There is no recording to leak, subpoena, or lose.
> - **There is a physical off switch.** A button on the wearable stops capture. Not
>   a setting in an app — a button on the device.
> - **The personal context stays on the device.** Names, routines, and preferences
>   that make the suggestions good never leave the machine they're stored on.
> - **The model suggests; it never speaks.** The utterance is always the user's tap.
>   The system has no ability to say anything on its own.
>
> The system is currently tested only within our own team. Any work with real users
> begins with written consent — including proxy consent where appropriate — and
> pseudonymized handling, established before a single session.

---

## 6. Where the project is, and the collaboration ask

[Do not oversell readiness. The people receiving this link can tell, and honesty
here is what makes a collaboration email get answered.]

**Headline:** "Working prototype. Looking for the people who know what we don't."

> VoiceLink runs end to end today: the wearable streams, the computer builds the
> board, the tablet updates live. It has been evaluated across 120 constructed
> scenarios spanning classrooms, medical appointments, meals, travel, and direct
> questions, measured on whether the board offers genuinely distinct things to say
> rather than variations on one guess.
>
> What it hasn't had yet is contact with the people it's for. That's the next step,
> and it's the one we can't take alone.
>
> **We're looking for:**
> - **AAC clinicians and speech-language pathologists** willing to look at real
>   boards and tell us where the vocabulary is wrong, condescending, or missing.
> - **Autism centers and schools** interested in a small, consented pilot.
> - **Researchers** in AAC, HCI, assistive technology, or on-device machine
>   learning — for collaboration, co-authorship, or simply a hard critique.
> - **Students** who want to work on this. There is real, scoped work available in
>   embedded systems, on-device ML, evaluation design, and interface design.
>
> If any of that is you, we'd like to hear from you — including if your reaction is
> that we've got something wrong.

**Call to action:** a single email link. [Builder: insert the contact address. One
CTA on the page, repeated at top and bottom. No contact form, no newsletter signup,
no "book a demo" — the audience is professional and will just email.]

---

## 7. Visuals — what to produce, and exactly what goes in each

[These are specs. Where a real screenshot is required, do not mock it up in
Figma — capture it from a live run so it's truthful. Illustrative boards below are
written for layout purposes and must be replaced with real output before launch.]

### V1 — Hero image: the board in a real room
**The single most important image on the site.**
A tablet propped on a kitchen or classroom table, in a real room with real light and
a person's hands near it. The board is legible on screen. The wearable device is
visible in the frame — clipped to a shirt or sitting on the table — so the two
halves of the system are understood in one glance without a diagram.
Shoot slightly from above at an angle, not straight-on. Avoid stock-photo staging;
a slightly imperfect real room is far more convincing to this audience.
*Alt text:* "A tablet showing the VoiceLink communication board on a kitchen table,
with the wearable sensor clipped nearby."

### V2 — The triptych: one person, three moments
**The visual that explains the entire product.** Three board screenshots side by
side, each captioned with the situation that produced it. Identical layout in all
three, so the eye immediately catches that the top row is unchanged and the
suggestions underneath are completely different.

| | Situation | Core row (identical in all three) | Suggestions (illustrative) |
|---|---|---|---|
| A | Classroom — *"What do plants take in from the air?"* | yes · no · help · stop · I need | carbon dioxide · I'm not sure · can you repeat that · I want to answer · can I skip this one |
| B | Doctor's office — *"Where does it hurt?"* | yes · no · help · stop · I need | my stomach · it's hard to explain · can my mom answer · not right now · it started this morning |
| C | Dinner table — *"Do you want more pasta?"* | yes · no · help · stop · I need | just a little · I'm finished · something else please · it's too hot · can we talk about something else |

*Caption for the group:* "Nobody programmed these. The core row never moves; the
suggestions come from what the system saw and heard."

### V3 — Annotated board screenshot
One full board screenshot with two callouts:
- Top region → "Core. Always here, always in the same place. The model can't touch it."
- Lower region → "Suggestions. Rebuilt as the situation changes."
Plus small pointers to the lock button ("freeze the board mid-thought") and the
freshness indicator ("new suggestions waiting — nothing swaps under your finger").

### V4 — System diagram
Deliberately plain. Three labelled boxes, two arrows, one highlighted boundary.

```
   [ Wearable ]  ──sees & hears──▶  [ Computer in the room ]  ──words──▶  [ Tablet ]
    senses only                      all intelligence lives here            taps speak
                                                │
                                     everything stays here
```

Do not draw a cloud. Do not draw a server. The absence of a network in the diagram
is the point — consider labelling the empty space around the boxes "no internet
required."

### V5 — The demo video (30–45 seconds, no narration, captions only)
This does more work than any amount of copy. Storyboard:

| Time | Shot | Caption |
|---|---|---|
| 0:00–0:04 | Board sitting idle on a table, core row visible | "The board is waiting." |
| 0:04–0:10 | Someone walks up and asks a question out loud | *(the question, as a caption)* |
| 0:10–0:14 | Close on the board — suggestions repopulate | "Two seconds later." |
| 0:14–0:18 | A hand taps one; it speaks aloud | *(the phrase, as a caption)* |
| 0:18–0:30 | Cut to a completely different room, different question, board changes again | "Same board. Nobody reprogrammed anything." |
| 0:30–0:40 | The wearable in frame; a thumb presses the capture button; the light goes out | "And an off switch that's an actual switch." |

Shoot the screen directly rather than screen-recording — seeing a real hand on a
real tablet in a real room is the whole persuasive effect.

### V6 — The wearable, photographed honestly
A clean close-up of the actual device, on a plain surface, next to a coin or a hand
for scale. If it currently looks like a development board, photograph it as one and
caption it "current prototype." This audience respects a visible prototype far more
than a rendered product that doesn't exist yet.

### V7 — Before / after comparison
Two panels: a conventional AAC grid of general-purpose vocabulary labelled "programmed
last week," and a VoiceLink board labelled "built for this minute." Keep it fair and
non-disparaging — VoiceLink augments AAC, it does not replace it, and many people in
the audience have spent careers on the tools in the left panel.

### V8 — Optional: the freshness moment, as a short loop
A 3-second silent GIF of the suggestion region rebuilding. Useful on the "what it
does" section for readers who won't play a video.

---

## 8. Facts and figures the site can use

All of these are cited or measured; keep the numbers exactly as written.

**The problem**
- 4–5 million people in the U.S. (1.3–1.5% of the population) can't meet daily
  communication needs with natural speech. *(CommunicationFIRST, AAC Counts II)*
- AAC users communicate at roughly 10–18 words per minute, against 125–185 for
  natural speech — about a tenth of conversational speed. *(ASHA)*
- Dedicated speech-generating devices cost $4,000–$12,000, with 3–18 month
  reimbursement cycles.
- 30–35% of devices are abandoned within the first year; boards that don't match the
  user's real context are a documented cause. *(AAC device abandonment, 2023)*
- 1 in 31 U.S. 8-year-olds is identified with autism; about 27% of children with ASD
  have profound autism — nonverbal, minimally verbal, or IQ under 50. *(CDC ADDM,
  2022 surveillance year, published 2025)*
- 2024 meta-analytic evidence: aided AAC embedded in natural context produces the
  strongest language gains. The context the vocabulary appears in matters, not just
  the device. *(Springer, 2024)*

**The system**
- Wearable hardware cost: roughly $15 in parts, plus a tablet the family already
  owns, plus one computer.
- Board size: 10 tappable phrases at a time — 5 fixed core words and 5 suggestions.
- Nine communicative functions the ranker deliberately spans.
- Evaluated across 120 constructed scenarios in five categories: classroom, medical,
  dinner table, travel, and direct questions.
- Vocabulary ranking runs in tens of milliseconds on an ordinary CPU.
- The ranking model is trained in-house; word selection involves no language-model
  call at inference time.
- All processing local. No cloud service, no subscription, no per-seat license.

---

## 9. Suggested page structure

Single scrolling page. This audience reads top to bottom and then emails; multi-page
navigation adds friction with no benefit.

1. **Hero** — one line, V1 image, one CTA ("Get in touch").
2. **The problem** — §2 story. Two or three of the statistics inline, not in a stat bar.
3. **What it does** — §3 copy with the V2 triptych. The emotional and explanatory core
   of the page.
4. **The demo** — V5 video, full width, no autoplay with sound.
5. **How it works** — §4, with V4 diagram. Skimmable.
6. **Privacy** — §5. Give it real space; do not bury it in a footer.
7. **Where we are / who we're looking for** — §6. Ends in the CTA.
8. **Footer** — contact email, link to the code repository if it's public, a line
   acknowledging that AAC users and clinicians know things we don't.

---

## 10. Voice and design notes

**Voice**
- Plain, specific, unhurried. Short sentences. No exclamation marks.
- Describe what the system does, not what it "empowers" or "revolutionizes."
- Never speak for AAC users or narrate their inner experience. Describe situations,
  not feelings.
- Say "nonverbal and minimally verbal people," or "AAC users." Not "the nonverbal,"
  not "sufferers," not "special needs."
- Identity-first vs person-first language is genuinely contested within the autistic
  community. Prefer sentences that need neither. Where unavoidable, "autistic people"
  is the more widely preferred form among autistic adults.
- Credit AAC and the clinicians who work in it. VoiceLink adds to that practice; it
  does not correct it.

**Words to avoid entirely:** revolutionary, game-changing, unlock, empower,
seamless, AI-powered, cutting-edge, disrupt, giving a voice to the voiceless.
That last one especially — these users have a voice; the tooling is what's slow.

**Design**
- Calm, high-legibility, generous type. The product itself is a Swiss-grid,
  high-contrast, low-decoration interface; the site should look like it belongs to
  the same system.
- Accessibility is not optional on this site of all sites: real contrast ratios,
  keyboard navigation, alt text on every image, captions on the video, no
  motion-triggered layout shifts, respect `prefers-reduced-motion`.
- Let the product screenshots be the visual interest. No gradient meshes, no glow,
  no floating 3D devices.

**Claims to keep out of the copy**
- Do not describe VoiceLink as a medical device, or imply clinical validation.
- Do not claim deployment, users, or outcomes. It is a working prototype tested
  within the team.
- Do not promise availability dates, pricing, or a product launch.
