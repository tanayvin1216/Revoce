<div align="center">

<img width="220" alt="Revoce logo: a ten-slot AAC board" src="docs/figures/logo.png" />

# Revoce

Revoce is a wearable that listens to the conversation and looks at the room, so a
nonverbal person's AAC board has the words they need by the time they need them.
No one has to program it in advance.

**[revoce.ai](https://revoce.ai/)**

**Intel Global AI Impact Festival 2026 Global Award Winner** · **AI for Education Global Award Finalist** · **NeurIPS 2026 Workshops: SLM, ODI and GenAI4Health** · **International Society for Augmentative and Alternative Communication (ISAAC)**

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-0071C5?style=for-the-badge&logo=intel&logoColor=white)](https://docs.openvino.ai/)
[![ONNX](https://img.shields.io/badge/ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white)](https://onnx.ai/)
[![Espressif](https://img.shields.io/badge/ESP32--S3-E7352C?style=for-the-badge&logo=espressif&logoColor=white)](https://www.espressif.com/)

<br />

<a href="https://www.youtube.com/watch?v=2Iy_1oze-cM">
  <img width="560" alt="Watch the Revoce video for the Intel Global AI Impact Festival" src="https://img.youtube.com/vi/2Iy_1oze-cM/maxresdefault.jpg" />
</a>

<sub>▶ Our Intel Global AI Impact Festival 2026 video. Click to watch on YouTube.</sub>

</div>

---

## Why we built it

An AAC board is a grid of words and phrases that someone taps to communicate.
Somebody has to program that grid ahead of time, and nobody can predict every
situation a person will be in during the day. If the word you need isn't on the
board, you can't say it.

Around 4–5 million people in the U.S. can't reliably meet their daily
communication needs through speech. Speech-generating devices
cost $4,000 to $12,000 and can take 3 to 18 months to get reimbursed, and 30–35%
of them are abandoned within the first year, often because the vocabulary doesn't
match what the person needs to say. AAC users also communicate at about
a tenth of normal conversational speed.

## What it does

The board has ten slots. Four of them never change: *yes*, *no*, *I need help*
and *wait*. We picked those because each one does a different job, and we didn't
want a bad prediction to take them away. The other six update based on
what's going on around the user.

Say a student is in class and the teacher asks *"What do plants take in from the
air?"* The board might show:

> **carbon dioxide** · oxygen and water · sun · I am not sure · can you repeat that? · I know this one

Not everything on the board is an answer. The student can also say they're not
sure, ask to hear the question again, or show they want to join in.

Revoce can also suggest words that aren't in its built-in vocabulary. At a family
dinner, *"Who do you want to sit next to?"* can bring up **aunt**, even though that
word was never on the list. [The vocabulary problem](#the-vocabulary-problem)
explains how.

## How it works

There are three devices. The wearable captures audio and images, a computer runs
all the models, and a tablet shows the board. Neither the wearable nor the
tablet runs a model.

```mermaid
flowchart TB
    subgraph W["WEARABLE · ESP32-S3"]
        direction LR
        CAM["OV2640 camera<br/>VGA JPEG · 2000 ms"]
        MIC["SPH0645 I²S mic<br/>16 kHz PCM · 512-sample chunks"]
        BTN["capture button · GPIO 14<br/>recording LED"]
    end

    subgraph M["COMPUTER · all inference, all local"]
        direction LR
        PERC["perception<br/>Whisper + VLM + change gate"]
        SEAM["rank_vocabulary(scene, transcript, context)<br/><b>every model call passes here</b>"]
        BACK["backend: PyTorch · OpenVINO<br/>selected by config flag"]
        PERC --> SEAM --> BACK
    end

    subgraph T["TABLET · AAC board"]
        direction LR
        UI["10-slot board<br/>plain HTTP + WebSocket"]
        TAP["tap log<br/>+ context snapshot"]
    end

    CAM -- "frames" --> PERC
    MIC -- "audio" --> PERC
    BACK -- "ranked words" --> UI
    TAP -- "logged" --> M

    classDef wear fill:#fdece4,stroke:#eb6834,color:#3a1d10
    classDef mac fill:#e3edfa,stroke:#2a78d6,color:#0e2440
    classDef tab fill:#e2f4ee,stroke:#1baf7a,color:#0d3325
    class CAM,MIC,BTN wear
    class PERC,SEAM,BACK mac
    class UI,TAP tab
```

Every model call in the project goes through one function:

```python
rank_vocabulary(scene, transcript, context) -> ranked_words
```

Because of this, we could try different models and hardware without touching
anything else. Moving from PyTorch on a Mac to OpenVINO on Intel was a config
flag, with no changes to the firmware, the server or the UI.

### Picking the six words

We started by showing the six highest-scoring words, but that often gave six
versions of the same answer. Revoce now generates candidates across nine
communicative functions and picks six that are relevant and different from each
other.

The full path from the camera and mic to the board:

```mermaid
flowchart TD
    A["JPEG frame"] --> B["change gate<br/>skip the VLM if the scene hasn't moved"]
    B -->|"scene changed"| C["Qwen2.5-VL-3B<br/>one-sentence scene description"]
    B -->|"unchanged"| C2["reuse last description"]
    D["16 kHz audio"] --> E["Whisper base.en<br/>local transcription"]

    C --> F["context vector · 769-d"]
    C2 --> F
    E --> F
    G["personal context store<br/>people · events · sensory needs"] --> F

    F --> H{"board strategy"}

    H -->|"teacher path"| I["Qwen2.5-3B-Instruct<br/>over-generate candidates<br/>9 communicative functions"]
    H -->|"student path"| J["MiniLM embeddings<br/>384-d, L2-normalised"]
    J --> K["distilled MLP<br/>769 → 192 → 95"]

    I --> L["candidate pool"]
    K --> L
    F --> M["open-vocabulary extractor<br/>reads answers out of the context"]
    M --> L

    L --> N["answer quota<br/>reserved when a question was asked"]
    N --> O["MMR selection<br/>relevant AND mutually different"]
    O --> P["pin 4 core words"]
    P --> Q["resolve ARASAAC pictograms"]
    Q --> R["10-slot board → WebSocket → tablet"]
    R --> S["tap logged with the context that produced it"]

    classDef sense fill:#fdece4,stroke:#eb6834,color:#3a1d10
    classDef model fill:#e3edfa,stroke:#2a78d6,color:#0e2440
    classDef policy fill:#f2eefc,stroke:#4a3aa7,color:#1e1740
    classDef out fill:#e2f4ee,stroke:#1baf7a,color:#0d3325
    class A,D,B,C2,G sense
    class C,E,I,J,K,M model
    class F,H,L,N,O,P policy
    class Q,R,S out
```


## How well it works

### Choosing the words

We compared our selection method with plain top-k ranking on 120 scenarios. Both
used the same candidates, so the only difference is how the six words are picked.

<p align="center">
  <img alt="Top-k vs diverse selection: distinct functions 3.88 to 4.68, near-duplicate rate 0.0274 to 0.0004, coverage of intent 0.78 to 1.00, exact match 0.33 to 0.53" src="docs/figures/diversity.png">
</p>

Choosing for variety covered more of what the user might want to say and nearly
removed repeated suggestions.

### Speed

Our first version used a 3-billion-parameter model. The boards were good, but each
one took about seven seconds, which is too slow for a conversation.

We trained a much smaller model on the big one's boards. We generated 600
examples, kept 559 after removing duplicates, and set aside 84 that it never
trained on.

| | Small model | Original model |
|---|---:|---:|
| Time to build a board | **10.135 ms** | 7.05 s |
| Variety kept | 95.3% | — |
| Speedup | **695×** | — |

The 695× figure is based on eight timing runs of the original model. Its speed
depends on how busy the machine is (we've also measured 10.6 s and 11.7 s), so we
report the speedup as **over 400×**.

The small model has three parts: MiniLM encodes the input, a small neural network
scores the words, and the same selection step as before builds the board. The
scoring network is about 190 KB and runs in roughly 0.1 ms. Most of the remaining
time is spent in MiniLM, so moving MiniLM to OpenVINO cut the latency the most.

<p align="center">
  <img alt="Board latency on Apple M4: 9.70 ms with PyTorch, 9.65 ms with only the head on OpenVINO, 5.51 ms with both on OpenVINO" src="docs/figures/latency.png">
</p>

The small model's architecture:

```mermaid
flowchart TD
    A["scene sentence"] --> C
    B["transcript"] --> C
    C["MiniLM encoder<br/>all-MiniLM-L6-v2 · frozen · embeddings memoised"]
    C --> D["769-d input vector<br/>384 scene ‖ 384 transcript ‖ 1 is_question"]
    Q["is_question flag"] --> D
    D --> E["LayerNorm → Linear 769→192 → ReLU → Dropout 0.3"]
    E --> F["Linear 192→95<br/>logits over the 95-word dynamic vocabulary"]
    F --> G["MMR + answer quota + hedge cap<br/>the unchanged symbolic selection policy"]
    G --> H["10-slot board · 4 pinned core + 6 dynamic"]

    classDef sense fill:#fdece4,stroke:#eb6834,color:#3a1d10
    classDef model fill:#e3edfa,stroke:#2a78d6,color:#0e2440
    classDef policy fill:#f2eefc,stroke:#4a3aa7,color:#1e1740
    classDef out fill:#e2f4ee,stroke:#1baf7a,color:#0d3325
    class A,B,Q sense
    class C,D,E,F model
    class G policy
    class H out
```


### Evaluation

These checks test whether the small model responds to the situation instead of
showing the same general-purpose board every time. A model that did that would
fail them.

| Check | Small model | Original model | Result |
|---|---:|---:|---|
| Distinct functions @7 | 4.095 | 4.298 | ✅ Pass |
| Near-duplicate rate | 0.000 | 0.0006 | ✅ Pass |
| Board variety | 0.952 | — | ✅ Pass (80 of 84 boards different) |
| Coverage of intent | 0.789 | 0.873 | ⚠️ Inconclusive |
| Same-function-pair rate | 0.1893 | 0.1735 | ❌ Fail |
| Word agreement with original | 0.067 | — | Tracked only |

Two of these still need work. The same-function-pair rate has failed three rounds
of testing, and every fix we've tried pushes coverage of intent below its
threshold. Coverage of intent clears its threshold by only 0.003 on 71 examples,
so we count it as inconclusive. We're working on improving both.

### The vocabulary problem

As we added training data, the small model agreed *less* often with the big
model's exact words: 23%, then 16%, then 6.7%. It kept passing every speed and
variety check.

The cause is the vocabulary. The big model can use any word, but the small one
picks from a fixed list of 95. That list can only cover
about a third of what the big model says, and making it 8.5× longer only gets you
to 57%.

<p align="center">
  <img alt="Coverage against vocabulary size: 95 words reaches 33% of the teacher's words, and an 8.5x larger vocabulary still only reaches 57%" src="docs/figures/ceiling.png">
</p>

Word-for-word agreement wasn't a fair test, so we switched our main metric to
whether the board covers what the user might mean. We still report word
agreement separately.

We also added a step that takes answers directly from what the device sees and
hears. That's how *aunt* appears at dinner. We tested it on 60 questions where the answer is somewhere in the
scene or conversation. It got every *who* and *this-or-that* question right.
The fixed list alone got at most 8%.

<p align="center">
  <img alt="Correct answer placed on the board by question type: alternative 100%, who 100%, where 88%, colour 83%, which 83%, count 0%. The closed vocabulary alone reaches 8% on who questions and 0% elsewhere" src="docs/figures/deictic.png">
</p>

### Running on Intel

We exported the small model to ONNX and converted it to OpenVINO, with outputs
matching PyTorch to within `1.43e-6`. We also quantized it to INT8 with NNCF,
calibrated on 360 real inputs instead of random data. Every exported model includes
a hash of its word list, so loading it with the wrong list raises an error instead
of showing the wrong words.

We ran the same model file on an Apple M4 and an Intel Core Ultra 9 185H.

<p align="center">
  <img alt="Student head speedup relative to PyTorch on the same machine: 0.60x on Apple M4 through the ARM CPU plugin, 2.66x on an Intel Core Ultra 9 185H" src="docs/figures/inversion.png">
</p>

Each number compares OpenVINO with PyTorch on the same machine, not the two
machines with each other. Moving MiniLM to OpenVINO as well made the whole board
2.02× faster on the Intel CPU, 1.85× on the Arc iGPU and 1.76× on the Mac.

INT8 made the model smaller (236 KB instead of 662 KB) but not faster. The model
doesn't run on the Intel NPU yet, because the NPU needs a fixed batch size and
our export uses a dynamic one. A fixed-size export is the next step.

## Getting started

You'll need Python 3.9 or newer. The big language and vision models run on Apple
silicon. On Intel, you can run the small model through OpenVINO instead (see
[`mac/intel/`](mac/intel/)).

```bash
git clone https://github.com/tanayvin1216/VoiceLink && cd VoiceLink
python3 -m venv .venv && source .venv/bin/activate
pip install -r mac/requirements-dev.txt
```

Run the tests (213 should pass):

```bash
cd mac
VOICELINK_PERCEPTION=0 pytest tests/ -q
```

Start the system:

```bash
bash mac/demo.sh              # real wearable, real models
bash mac/demo.sh --fake       # no ESP32 needed
bash mac/demo.sh --student    # the small model instead of the big one
bash mac/demo.sh --intel      # small model, MiniLM and Whisper on OpenVINO
bash mac/demo.sh --mock       # no models at all
```

The flags can be combined. Open the board at `http://<your-mac>:8080/` on a
tablet on the same Wi-Fi. We use plain HTTP so the tablet can connect without
HTTPS mixed-content errors.

## What's in the repo

| Path | What's there |
|---|---|
| [`mac/rank_vocabulary.py`](mac/rank_vocabulary.py) | Picks the words; every model call goes through here |
| [`mac/server.py`](mac/server.py) | Web server, live updates and tap logging |
| [`mac/evaluation/phase4/`](mac/evaluation/phase4/) | The small model, training and test results |
| [`mac/intel/`](mac/intel/) | OpenVINO conversion, quantization and benchmarks |
| [`firmware/voicelink_capture/`](firmware/voicelink_capture/) | Code for the ESP32-S3 wearable |
| [`docs/figures/make_figures.py`](docs/figures/make_figures.py) | Rebuilds every chart here from the saved results |
| [`paper/`](paper/) | Our NeurIPS 2026 workshop papers |

The charts are generated from results saved in the repo. If a chart and its JSON
disagree, the JSON is correct. To rebuild them, run
`python3 docs/figures/make_figures.py`.

## Privacy

Revoce has a camera and a microphone and is worn around other people, so privacy
for the user and the people near them is built into the hardware and software.

- **Local processing.** All models run on the computer. Audio, video and
  transcripts are never sent to a server.
- **No recordings.** Audio and images are held briefly in memory and never
  written to disk.
- **Off button.** A physical button on the wearable pauses capture. While it's
  paused, no audio is processed or sent and no pictures are taken.
- **Recording light.** An LED stays on whenever the device is capturing.
- **Consent.** The repo includes tools to pseudonymize the tap logs and a consent
  form for guardians of people who can't consent for themselves.

Until written consent is in place, we only test on members of our team.

## Credits

The pictograms come from [ARASAAC](https://arasaac.org/) under the CC BY-NC-SA
license, which doesn't allow commercial use. A commercial version would need a
different symbol set.
