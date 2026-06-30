"""Teacher data-gen (Phase-4 build order, step 2) — run in the BACKGROUND.

The teacher is the slow path we are distilling: rank_vocabulary(strategy="diverse")
= one LLM pool-gen + MMR selection, driven exactly as the server drives it
(reusing selftest.run_seam, so topics_from_context and the seam are identical).

Scenario sources, deduped by (scene, transcript):
  1. the Phase-3 curated corpus (scengen)            — 120, 5 categories
  2. the Phase-1 real AAC turns (examples.json)      — 120, varied conversations
  3. filler-expanded Phase-3 variants                — to top up to --target

Resumable: rows already in teacher_data.jsonl (by id) are skipped, so a killed
overnight run picks up where it stopped.

    VOICELINK_PERCEPTION=0 python3 evaluation/phase4/gen_teacher.py --target 500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAC = HERE.parent.parent
if str(MAC) not in sys.path:
    sys.path.insert(0, str(MAC))

from evaluation.phase3 import scengen, selftest  # noqa: E402

OUT = HERE / "teacher_data.jsonl"
EXTRA_FILLERS = ["Alex", "Jordan", "Mr. Kim", "Ms. Rivera", "Aunt May",
                 "coach", "the helper", "a classmate", "the cashier", "Dr. Singh"]


def _phase1_scenarios() -> list[dict]:
    path = MAC / "evaluation" / "phase1" / "examples.json"
    if not path.exists():
        return []
    out = []
    for ex in json.loads(path.read_text()):
        last = (ex.get("last_utterance") or "").strip()
        transcript = ex.get("transcript") or last
        is_q = "?" in last or "?" in (transcript or "")
        out.append({
            "id": f"p1-{ex.get('conversation_id')}-{ex.get('turn_number')}",
            "category": "phase1-real",
            "scene": ex.get("scene") or "", "transcript": transcript,
            "profile": {}, "is_question": is_q,
            "active_question": last if is_q else "",
        })
    return out


def _expanded_phase3(target_extra: int) -> list[dict]:
    """Extra Phase-3 variants by cycling more filler names through the templates."""
    rows: list[dict] = []
    for category in scengen.CATEGORIES:
        spec = scengen._TEMPLATES[category]
        slot = spec["slot"]
        for p_i, prompt in enumerate(spec["prompts"]):
            for f_i, who in enumerate(EXTRA_FILLERS):
                scene = spec["scene"].replace("{" + slot + "}", who)
                last = prompt["q"].replace("{" + slot + "}", who)
                rows.append({
                    "id": f"{category}-x{p_i}-{f_i}", "category": category,
                    "scene": scene, "transcript": last, "profile": scengen._PROFILE,
                    "is_question": bool(prompt["is_q"]),
                    "active_question": last if prompt["is_q"] else "",
                })
                if len(rows) >= target_extra:
                    return rows
    return rows


def build_scenarios(target: int) -> list[dict]:
    base = scengen.load_scenarios() + _phase1_scenarios()
    seen = {(r["scene"], r["transcript"]) for r in base}
    out = list(base)
    for r in _expanded_phase3(max(0, target - len(base)) + 50):
        key = (r["scene"], r["transcript"])
        if key not in seen:
            seen.add(key)
            out.append(r)
        if len(out) >= target:
            break
    return out[:target]


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return ids


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=int, default=500)
    args = p.parse_args()

    scenarios = build_scenarios(args.target)
    done = _done_ids(OUT)
    todo = [s for s in scenarios if s["id"] not in done]
    print(f"{len(scenarios)} scenarios, {len(done)} already done, {len(todo)} to gen",
          flush=True)

    from evaluation.phase1.gen_phase1 import MlxRanker  # lazy: imports mlx
    ranker = MlxRanker()
    ranker._temp = 0.0
    print("teacher (Qwen) loaded", flush=True)

    with open(OUT, "a") as f:
        for i, sc in enumerate(todo, 1):
            t0 = time.time()
            try:
                board = selftest.run_seam(sc, "diverse", ranker)
            except Exception as e:  # noqa: BLE001
                print(f"  err {sc['id']}: {e}", flush=True)
                continue
            f.write(json.dumps({
                "id": sc["id"], "category": sc["category"],
                "scene": sc.get("scene", ""), "transcript": sc.get("transcript", ""),
                "is_question": bool(sc.get("is_question")), "board": board,
            }) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(todo):
                print(f"[{i}/{len(todo)}] {sc['id']} {time.time()-t0:.1f}s", flush=True)
    print("TEACHER DATA-GEN COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
