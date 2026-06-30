"""Phase-3 SCENARIO GENERATION — build the >=100-scenario testbed, cache it.

Each scenario is one moment where the AAC user must reply: a scene, the recent
transcript, the question (if any) on the table, the SET of acceptable
communicative moves, and one plausible logged `gold_utterance`. The board is
later scored on whether it COVERS the acceptable moves, not whether it guesses
the gold.

Two sources, in priority order:
  1. LLM (local Qwen via MlxRanker, temp ~0.8) — generated in batches PER
     CATEGORY. NOTE: we call MlxRanker.raw() and parse the object array ourselves
     (mirroring oracle.py). MlxRanker.rank() routes through the vocabulary-shaped
     _parse_json_array, which keeps only {word,...} objects and would DESTROY
     scenario rows.
  2. A deterministic, seeded TEMPLATE FALLBACK (this file, stdlib only) that
     guarantees >=100 scenarios even with no model — so the self-test always has a
     corpus, and the unit tests can exercise it offline.

    python3 evaluation/phase3/scengen.py            # load cache, or build fallback
    python3 evaluation/phase3/scengen.py --regen    # rebuild from the LLM

Row schema (one JSON object per line in scenarios.jsonl):
    {id, category, scene, transcript, profile, is_question,
     active_question, acceptable_moves: [...], gold_utterance}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAC = HERE.parent.parent
if str(MAC) not in sys.path:
    sys.path.insert(0, str(MAC))

SCENARIOS_PATH = HERE / "scenarios.jsonl"
CURATED_PATH = MAC / "evaluation" / "phase1" / "scenarios.json"

CATEGORIES = ["classroom", "doctor", "airport", "dinner", "answer-a-question"]
MIN_SCENARIOS = 100

# A minimal but real personal-context profile shape so the topk seam path's
# slice_for_prompt() has something to read (it only ever uses .get()).
_PROFILE = {
    "version": 1,
    "identity": {"name": "Sam", "pronouns": "they/them", "aac_user": True,
                 "primary_language": "en"},
    "relationships": [],
    "routines": [],
    "sensory_notes": {"prefers_quiet": False, "noise_sensitive": True,
                      "dietary": [], "comfort_items": ["headphones"]},
    "todays_events": [],
}


# ---------------------------------------------------------------------------
# Deterministic template fallback — guarantees >=100 scenarios with no model.
# ---------------------------------------------------------------------------
# Each category supplies prompt templates (a moment + its acceptable moves) and a
# single {slot} filler list. rows-per-category = len(prompts) * len(fillers).
_TEMPLATES: dict[str, dict] = {
    "classroom": {
        "scene": "A classroom. {who} stands at a whiteboard with a diagram; "
                 "other students sit at desks.",
        "slot": "who",
        "fillers": ["Ms. Lee", "Mr. Diaz", "the teacher", "Ms. Patel"],
        "prompts": [
            {"q": "{who}: What do plants take in from the air?", "is_q": True,
             "moves": ["carbon dioxide", "I don't know", "can you repeat that",
                       "I want to answer", "the sun", "water"]},
            {"q": "{who}: Who can tell me the answer to question three?", "is_q": True,
             "moves": ["I can", "I'm not sure", "can you help me", "ask someone else",
                       "the answer is six", "I need more time"]},
            {"q": "{who}: Today we are learning about volcanoes.", "is_q": False,
             "moves": ["that's interesting", "I have a question", "I already knew that",
                       "I feel excited", "can you say more", "I don't like this topic"]},
            {"q": "{who}: Did everyone finish the reading?", "is_q": True,
             "moves": ["yes", "no", "almost done", "I need help", "which page",
                       "I forgot"]},
            {"q": "{who}: How are you doing with the worksheet?", "is_q": True,
             "moves": ["it's hard", "I'm okay", "I need help", "I'm done",
                       "can I have more time", "I don't understand"]},
            {"q": "{who}: Great work everyone, let's take a break.", "is_q": False,
             "moves": ["thank you", "can I get water", "I want to keep going",
                       "I'm tired", "that was fun", "what's next"]},
        ],
    },
    "doctor": {
        "scene": "A doctor's exam room. {who} sits nearby holding a clipboard.",
        "slot": "who",
        "fillers": ["the doctor", "the nurse", "Dr. Okafor", "the assistant"],
        "prompts": [
            {"q": "{who}: Where does it hurt?", "is_q": True,
             "moves": ["my head", "my stomach", "my back", "it doesn't hurt",
                       "I'm not sure", "everywhere"]},
            {"q": "{who}: How are you feeling today?", "is_q": True,
             "moves": ["not good", "I'm okay", "I feel sick", "I'm scared",
                       "tired", "better than yesterday"]},
            {"q": "{who}: I'm going to check your blood pressure now.", "is_q": False,
             "moves": ["okay", "wait", "will it hurt", "I'm nervous", "go ahead",
                       "can you explain"]},
            {"q": "{who}: Are you taking any medications?", "is_q": True,
             "moves": ["yes", "no", "I don't know", "my mom knows", "just vitamins",
                       "can you check my chart"]},
            {"q": "{who}: Do you have any questions for me?", "is_q": True,
             "moves": ["when can I go home", "is it serious", "no questions",
                       "what is wrong", "can my family come in", "I'm worried"]},
            {"q": "{who}: Take a deep breath for me.", "is_q": False,
             "moves": ["okay", "I can't", "like this", "wait a second", "I'm trying",
                       "it's hard to breathe"]},
        ],
    },
    "airport": {
        "scene": "A busy airport terminal. {who} is at the desk; announcements play "
                 "overhead and crowds move past.",
        "slot": "who",
        "fillers": ["a gate agent", "the staff member", "an attendant", "the officer"],
        "prompts": [
            {"q": "{who}: Can I see your boarding pass?", "is_q": True,
             "moves": ["here it is", "one moment", "it's on my phone", "I can't find it",
                       "can you help me", "I need help"]},
            {"q": "{who}: Are you checking any bags today?", "is_q": True,
             "moves": ["yes", "no", "just this one", "I'm not sure", "how much is it",
                       "can you help me"]},
            {"q": "{who}: Your flight is delayed by two hours.", "is_q": False,
             "moves": ["okay", "why", "I'm frustrated", "where can I wait",
                       "will I miss my connection", "can I get a refund"]},
            {"q": "{who}: Window or aisle seat?", "is_q": True,
             "moves": ["window", "aisle", "I don't mind", "near the front",
                       "can I sit with my family", "the quietest one"]},
            {"q": "{who}: Do you need any assistance to the gate?", "is_q": True,
             "moves": ["yes please", "no thank you", "which way is it", "I'm okay",
                       "I need help", "how far is it"]},
            {"q": "{who}: It's very loud in here today.", "is_q": False,
             "moves": ["it's too loud", "I have headphones", "I'm overwhelmed",
                       "can we move", "I'm okay", "where is somewhere quiet"]},
        ],
    },
    "dinner": {
        "scene": "A dinner table at home. {who} is passing dishes around.",
        "slot": "who",
        "fillers": ["mom", "dad", "grandma", "a friend"],
        "prompts": [
            {"q": "{who}: Do you want more pasta?", "is_q": True,
             "moves": ["yes please", "no thank you", "just a little", "I'm full",
                       "I want something else", "can I have water"]},
            {"q": "{who}: How was your day?", "is_q": True,
             "moves": ["it was good", "it was hard", "I'm tired", "I have a story",
                       "okay I guess", "I don't want to talk about it"]},
            {"q": "{who}: We're having vegetables tonight.", "is_q": False,
             "moves": ["yum", "I don't like those", "can I have something else",
                       "thank you", "I'm not hungry", "just a small piece"]},
            {"q": "{who}: Can you pass the salt?", "is_q": True,
             "moves": ["here you go", "I can't reach", "which one", "sure",
                       "I need help", "wait"]},
            {"q": "{who}: What do you want to do after dinner?", "is_q": True,
             "moves": ["watch a movie", "play a game", "go to bed", "I don't know",
                       "read a book", "go outside"]},
            {"q": "{who}: You barely touched your food.", "is_q": False,
             "moves": ["I'm not hungry", "I don't feel well", "it's too much",
                       "I'll eat more", "I don't like it", "can I be done"]},
        ],
    },
    # A cross-setting bucket of pure ANSWER-MODE moments so the answer path is
    # always exercised; every prompt here is a direct question to the user.
    "answer-a-question": {
        "scene": "Someone is talking directly to the user and waiting for a reply.",
        "slot": "who",
        "fillers": ["a friend", "a stranger", "a coworker", "a neighbor"],
        "prompts": [
            {"q": "{who}: What's your name?", "is_q": True,
             "moves": ["I'm Sam", "you can call me Sam", "I'd rather not say",
                       "ask my helper", "can you repeat that", "why"]},
            {"q": "{who}: Where are you headed?", "is_q": True,
             "moves": ["home", "to the store", "I'm not sure", "with my family",
                       "nowhere special", "that way"]},
            {"q": "{who}: What's your favorite color?", "is_q": True,
             "moves": ["blue", "green", "I like red", "I can't pick", "all of them",
                       "why do you ask"]},
            {"q": "{who}: Do you want to come with us?", "is_q": True,
             "moves": ["yes", "no thanks", "maybe later", "where to", "I need to ask",
                       "I'm not sure"]},
            {"q": "{who}: How old are you?", "is_q": True,
             "moves": ["I'm twelve", "guess", "I'd rather not say", "why",
                       "older than that", "ask my mom"]},
            {"q": "{who}: Did you have a good weekend?", "is_q": True,
             "moves": ["yes it was great", "not really", "it was okay", "I was busy",
                       "I have a story", "I stayed home"]},
        ],
    },
}


def _template_scenarios(min_count: int = MIN_SCENARIOS) -> list[dict]:
    """Deterministic, model-free scenarios. Order is fixed, so reruns are stable.

    Produces len(prompts) * len(fillers) rows per category (>=100 total); if the
    banks above ever shrink below min_count, the last rows are duplicated with a
    distinct id suffix so the guarantee holds.
    """
    rows: list[dict] = []
    for category in CATEGORIES:
        spec = _TEMPLATES[category]
        slot = spec["slot"]
        for p_i, prompt in enumerate(spec["prompts"]):
            for f_i, filler in enumerate(spec["fillers"]):
                who = filler
                scene = spec["scene"].replace("{" + slot + "}", who)
                last = prompt["q"].replace("{" + slot + "}", who)
                moves = list(prompt["moves"])
                rows.append({
                    "id": f"{category}-{p_i}-{f_i}",
                    "category": category,
                    "scene": scene,
                    "transcript": last,
                    "profile": _PROFILE,
                    "is_question": bool(prompt["is_q"]),
                    "active_question": last if prompt["is_q"] else "",
                    "acceptable_moves": moves,
                    "gold_utterance": moves[0] if moves else "",
                })
    # Guarantee >=min_count even if the banks are edited down later.
    i = 0
    while len(rows) < min_count and rows:
        base = dict(rows[i % len(rows)])
        base["id"] = base["id"] + f"-dup{i}"
        rows.append(base)
        i += 1
    return rows


# ---------------------------------------------------------------------------
# LLM generation (Mac / Qwen only) — uses MlxRanker.raw, parses objects directly.
# ---------------------------------------------------------------------------
_GEN_SYSTEM = """\
You write evaluation scenarios for an AAC communication aid. Each scenario is one
moment where a nonverbal user must reply.

Output ONLY a JSON array of objects. No prose, no code fences. Each object:
  {"scene": "<one sentence>", "transcript": "<what was just said to the user>",
   "is_question": true|false, "active_question": "<the question, or empty>",
   "acceptable_moves": ["<5-7 GENUINELY DIFFERENT short first-person replies>"],
   "gold_utterance": "<one plausible reply from acceptable_moves>"}

acceptable_moves must span DIFFERENT communicative directions (an answer, a
question back, "I don't know", a refusal, a feeling, asking for help) — never
paraphrases of one another."""

_GEN_USER = """\
Category: {category}
Write {n} varied {category} scenarios as a JSON array now."""

_PER_CATEGORY = 24  # LLM target per category -> ~120 total before validation


def _parse_objects(text: str) -> list[dict]:
    """Extract a JSON array of objects from raw LLM text. Mirrors the salvage
    strategy in oracle._parse_strings, but for {...} objects."""
    import re

    t = (text or "").strip()
    if "```" in t:
        t = t[t.find("```") + 3:]
        t = t[t.find("\n") + 1:] if "\n" in t else t
        t = t[: t.find("```")] if "```" in t else t
    start, end = t.find("["), t.rfind("]")
    if start != -1 and end > start:
        try:
            arr = json.loads(t[start:end + 1])
            return [x for x in arr if isinstance(x, dict)]
        except json.JSONDecodeError:
            pass
    # Salvage every complete {...} block individually on truncation/junk.
    out: list[dict] = []
    for m in re.findall(r"\{[^{}]*\}", t):
        try:
            obj = json.loads(m)
            if isinstance(obj, dict):
                out.append(obj)
        except json.JSONDecodeError:
            continue
    return out


def _coerce(obj: dict, category: str, idx: int) -> dict | None:
    """Validate/normalize one generated object into a scenario row, or drop it."""
    scene = str(obj.get("scene") or "").strip()
    moves = [str(m).strip() for m in (obj.get("acceptable_moves") or []) if str(m).strip()]
    if not scene or len(moves) < 3:
        return None
    transcript = str(obj.get("transcript") or "").strip()
    is_q = bool(obj.get("is_question"))
    active_q = str(obj.get("active_question") or "").strip()
    if is_q and not active_q:
        active_q = transcript
    gold = str(obj.get("gold_utterance") or "").strip() or moves[0]
    return {
        "id": f"{category}-llm-{idx}",
        "category": category,
        "scene": scene,
        "transcript": transcript or active_q,
        "profile": _PROFILE,
        "is_question": is_q,
        "active_question": active_q if is_q else "",
        "acceptable_moves": moves,
        "gold_utterance": gold,
    }


def generate_llm(per_category: int = _PER_CATEGORY, temp: float = 0.8) -> list[dict]:
    """Generate scenarios with the local Qwen model (Apple-Silicon only).

    Falls back to (and tops up from) the template set so the result is always
    >=MIN_SCENARIOS, regardless of how the model behaves."""
    from evaluation.phase1.gen_phase1 import MlxRanker  # lazy: imports mlx

    ranker = MlxRanker()
    ranker._temp = temp
    rows: list[dict] = []
    for category in CATEGORIES:
        user = _GEN_USER.format(category=category, n=per_category)
        try:
            raw = ranker.raw(_GEN_SYSTEM, user, temp, max_tokens=2048)
            objs = _parse_objects(raw)
        except Exception as e:  # noqa: BLE001 - never let one category abort the run
            print(f"  {category} gen err: {e}", flush=True)
            objs = []
        kept = [r for r in (_coerce(o, category, i) for i, o in enumerate(objs)) if r]
        print(f"[{category}] parsed {len(objs)} -> kept {len(kept)}", flush=True)
        rows.extend(kept)

    if len(rows) < MIN_SCENARIOS:
        have = {r["id"] for r in rows}
        for fb in _template_scenarios(MIN_SCENARIOS):
            if len(rows) >= MIN_SCENARIOS:
                break
            if fb["id"] not in have:
                rows.append(fb)
    return rows


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------
def _write(rows: list[dict], path: Path = SCENARIOS_PATH) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_scenarios(regen: bool = False, path: Path = SCENARIOS_PATH) -> list[dict]:
    """Return the scenario corpus. Loads the JSONL cache if present; otherwise
    builds the deterministic template fallback and caches it. `regen=True` runs
    the LLM (Mac only) and overwrites the cache."""
    if regen:
        rows = generate_llm()
        _write(rows, path)
        return rows
    if path.exists():
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if rows:
            return rows
    rows = _template_scenarios()
    _write(rows, path)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Phase-3 scenario generation")
    p.add_argument("--regen", action="store_true",
                   help="rebuild from the local LLM (Apple-Silicon Mac only)")
    args = p.parse_args()
    rows = load_scenarios(regen=args.regen)
    by_cat: dict[str, int] = {}
    for r in rows:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    print(f"{len(rows)} scenarios -> {SCENARIOS_PATH}")
    for c in CATEGORIES:
        print(f"  {c:18s} {by_cat.get(c, 0)}")


if __name__ == "__main__":
    main()
