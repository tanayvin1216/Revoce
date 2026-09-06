"""Phase-3 SELF-TEST — the ONE COMMAND.

    VOICELINK_PERCEPTION=0 python3 evaluation/phase3/selftest.py
    python3 evaluation/phase3/selftest.py --regen        # rebuild scenarios via LLM
    python3 evaluation/phase3/selftest.py --from-boards   # re-score cache, NO LLM

What it does, end to end:
  1. load (or --regen) the >=100-scenario corpus (scengen.py);
  2. for each scenario, run the SEAM exactly as the server does (server.py:222-227)
     for BOTH strategy="diverse" and strategy="topk", caching every board to
     boards.jsonl so re-scoring is free;
  3. score every board on the four Phase-3 metrics (metrics3.py), aggregated
     overall and per category, into selftest_results.json;
  4. render the figures (plots3.py);
  5. print an explicit PASS/FAIL line for each of the four gates.

The model is loaded ONCE for the whole run and shared across both strategies and
all scenarios. Generation is the only slow part and is thermal-paced; --from-boards
skips it entirely (the path the unit test drives).
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

from evaluation.phase3 import metrics3, scengen  # noqa: E402

BOARDS_PATH = HERE / "boards.jsonl"
RESULTS_PATH = HERE / "selftest_results.json"
STRATEGIES = ("diverse", "topk")
TOPK = metrics3.TOPK

# --- PASS gates (all four must hold) ---------------------------------------
GATE_MIN_SCENARIOS = scengen.MIN_SCENARIOS
GATE_DF_DIVERSE_MIN = 4.5     # distinct-functions@7, diverse arm
GATE_DF_TOPK_MAX = 4.0        # distinct-functions@7, topk arm
GATE_DF_GAP_MIN = 0.75        # diverse - topk
GATE_DUP_DIVERSE_MAX = 0.05   # near-dup rate, diverse arm
GATE_DUP_RATIO = 3.0          # diverse near-dup must be >=3x lower than topk


# ---------------------------------------------------------------------------
# Seam driver — mirrors server.py:222-227 exactly.
# ---------------------------------------------------------------------------
def build_context(scenario: dict) -> tuple[str, str, dict]:
    """Reconstruct the (scene, transcript, context) the server would hand the seam.

    The live device passes an `active_question` in its ContextState snapshot, so we
    carry it here too: that is what holds answer-mode on a question and clears it
    otherwise (rank_vocabulary._current_question)."""
    scene = scenario.get("scene") or None
    transcript = scenario.get("transcript") or None
    context = {
        "speaker": "partner" if transcript else None,
        "addressed": bool(scenario.get("is_question") or transcript),
        "last_utterance": transcript,
        "recent_transcript": transcript or "",
        "active_question": scenario.get("active_question") or "",
        "scene_changed": bool(scene),
        "personal_context": scenario.get("profile") or {},
    }
    return scene, transcript, context


def run_seam(scenario: dict, strategy: str, ranker) -> list[str]:
    """One board from the seam, the server way: topics_from_context then
    rank_vocabulary(..., strategy=..., topics=topics)."""
    from rank_vocabulary import rank_vocabulary, topics_from_context

    scene, transcript, context = build_context(scenario)
    topics = topics_from_context(scene, transcript, context)
    board = rank_vocabulary(scene, transcript, context, ranker,
                            strategy=strategy, topics=topics)
    return [b["word"] for b in board]


def generate_boards(scenarios: list[dict], ranker, sleep: float = 0.0,
                    path: Path = BOARDS_PATH) -> list[dict]:
    """Run both strategies over every scenario, caching each board to boards.jsonl."""
    rows: list[dict] = []
    with open(path, "w") as f:
        for i, sc in enumerate(scenarios, 1):
            for strategy in STRATEGIES:
                t0 = time.time()
                try:
                    board = run_seam(sc, strategy, ranker)
                except Exception as e:  # noqa: BLE001 - never abort the whole run
                    board = []
                    print(f"  seam err {sc['id']}/{strategy}: {e}", flush=True)
                row = {
                    "id": sc["id"], "category": sc["category"], "strategy": strategy,
                    "scene": sc.get("scene", ""), "transcript": sc.get("transcript", ""),
                    "active_question": sc.get("active_question", ""),
                    "is_question": bool(sc.get("is_question")),
                    "acceptable_moves": sc.get("acceptable_moves", []),
                    "gold_utterance": sc.get("gold_utterance", ""),
                    "board": board,
                }
                f.write(json.dumps(row) + "\n")
                f.flush()
                rows.append(row)
                if sleep:  # thermal pacing — the LLM is the only hot part
                    time.sleep(sleep)
            print(f"[{i}/{len(scenarios)}] {sc['id']} {time.time()-t0:.1f}s", flush=True)
    return rows


def load_boards(path: Path = BOARDS_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"no cached boards at {path}; run without --from-boards first")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Scoring — pure metrics over cached boards (the only ML dep is embed).
# ---------------------------------------------------------------------------
def _score_one(row: dict, metric_k: int | None = None,
               dynamic_only: bool = False) -> dict:
    from evaluation.embed import embed  # lazy so --from-boards stays the only ML use

    metric_k = TOPK if metric_k is None else metric_k
    if metric_k < 1:
        raise ValueError("metric_k must be positive")
    original_board = row.get("board", [])
    board = original_board
    if dynamic_only:
        from rank_vocabulary import GUARANTEED_CORE
        core = {w.lower() for w in GUARANTEED_CORE}
        board = [w for w in board if w.lower() not in core]
    top = board[:metric_k]
    bvecs = embed(top) if top else embed([])
    return {
        "distinct_functions": metrics3.distinct_functions_at(board, metric_k, board_vecs=bvecs),
        "near_dup": metrics3.near_dup_rate(bvecs),
        "coverage_of_intent": metrics3.coverage_of_intent(
            board, row.get("acceptable_moves", []), metric_k, board_vecs=bvecs),
        "exact_match": metrics3.exact_match(original_board, row.get("gold_utterance", "")),
    }


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _aggregate(scored: list[dict]) -> dict:
    """Mean each metric for a list of per-board score dicts (one strategy)."""
    return {
        "n": len(scored),
        "distinct_functions": _mean([s["distinct_functions"] for s in scored]),
        "near_dup": _mean([s["near_dup"] for s in scored]),
        "coverage_of_intent": _mean([1.0 if s["coverage_of_intent"] else 0.0 for s in scored]),
        "exact_match": _mean([1.0 if s["exact_match"] else 0.0 for s in scored]),
    }


def score_boards(board_rows: list[dict], metric_k: int | None = None,
                 dynamic_only: bool = False) -> dict:
    """Aggregate the four metrics overall and per category, per strategy.

    Importable and model-free apart from embed(), so the unit test drives it on a
    tiny fixture with a monkeypatched embedder."""
    metric_k = TOPK if metric_k is None else metric_k
    if metric_k < 1:
        raise ValueError("metric_k must be positive")
    per_strategy: dict[str, list[dict]] = {s: [] for s in STRATEGIES}
    per_cat: dict[str, dict[str, list[dict]]] = {}
    scenario_ids: set[str] = set()

    for row in board_rows:
        strat = row.get("strategy")
        if strat not in per_strategy:
            continue
        scenario_ids.add(row.get("id", ""))
        s = _score_one(row, metric_k=metric_k, dynamic_only=dynamic_only)
        per_strategy[strat].append(s)
        cat = row.get("category", "unknown")
        per_cat.setdefault(cat, {st: [] for st in STRATEGIES})[strat].append(s)

    overall = {st: _aggregate(per_strategy[st]) for st in STRATEGIES}
    per_category = {
        cat: {st: _aggregate(scs[st]) for st in STRATEGIES} for cat, scs in per_cat.items()
    }
    return {
        "metric_window": {"k": metric_k, "dynamic_only": dynamic_only},
        "n_scenarios": len(scenario_ids),
        "overall": overall,
        "per_category": per_category,
    }


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------
def evaluate_gates(results: dict) -> dict:
    """Compute PASS/FAIL for each of the four gates and the overall verdict."""
    d = results["overall"]["diverse"]
    t = results["overall"]["topk"]
    n = results["n_scenarios"]

    df_gap = d["distinct_functions"] - t["distinct_functions"]
    gate_df = (d["distinct_functions"] >= GATE_DF_DIVERSE_MIN
               and t["distinct_functions"] <= GATE_DF_TOPK_MAX
               and df_gap >= GATE_DF_GAP_MIN)
    gate_dup = (d["near_dup"] <= GATE_DUP_DIVERSE_MAX
                and t["near_dup"] >= GATE_DUP_RATIO * d["near_dup"])
    gate_n = n >= GATE_MIN_SCENARIOS and _categories_present(results)

    # Gate 4 is structural: the judge output is explicitly labeled PROXY and the
    # human-comparison hook exists. Verified by import, not by a numeric run.
    try:
        from evaluation.phase3 import judge
        gate_judge = bool(getattr(judge, "PROXY_LABEL", "")) and callable(
            getattr(judge, "compare_to_human", None))
    except Exception:  # noqa: BLE001
        gate_judge = False

    gates = {
        "scenarios": {"pass": bool(gate_n), "n": n, "need": GATE_MIN_SCENARIOS},
        "distinct_functions": {
            "pass": bool(gate_df), "diverse": round(d["distinct_functions"], 3),
            "topk": round(t["distinct_functions"], 3), "gap": round(df_gap, 3),
            "need": f"diverse>={GATE_DF_DIVERSE_MIN}, topk<={GATE_DF_TOPK_MAX}, "
                    f"gap>={GATE_DF_GAP_MIN}",
        },
        "near_dup": {
            "pass": bool(gate_dup), "diverse": round(d["near_dup"], 4),
            "topk": round(t["near_dup"], 4),
            "need": f"diverse<={GATE_DUP_DIVERSE_MAX} and >={GATE_DUP_RATIO}x lower than topk",
        },
        "judge_proxy_labeled": {"pass": bool(gate_judge),
                                "need": "judge.PROXY_LABEL set + compare_to_human hook"},
    }
    results["gates"] = gates
    results["pass"] = all(g["pass"] for g in gates.values())
    return results


def _categories_present(results: dict) -> bool:
    return all(c in results["per_category"] for c in scengen.CATEGORIES)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_report(results: dict) -> None:
    bar = "=" * 74
    d = results["overall"]["diverse"]
    t = results["overall"]["topk"]
    print("\n" + bar)
    print("VoiceLink Phase-3 self-test — coverage-shaped metrics")
    print(bar)
    window = results.get("metric_window", {"k": 7, "dynamic_only": False})
    print(f"window: {window}; gates retain historical @7 thresholds")
    print(f"scenarios: {results['n_scenarios']}")
    print(f"{'metric':<26}{'diverse':>12}{'topk':>12}")
    print(f"{('distinct-functions@' + str(window['k'])):<26}{d['distinct_functions']:>12.3f}{t['distinct_functions']:>12.3f}")
    print(f"{'near-duplicate rate':<26}{d['near_dup']:>12.4f}{t['near_dup']:>12.4f}")
    print(f"{('coverage-of-intent@' + str(window['k'])):<26}{d['coverage_of_intent']:>12.3f}{t['coverage_of_intent']:>12.3f}")
    print(f"{'exact-match (diagnostic)':<26}{d['exact_match']:>12.3f}{t['exact_match']:>12.3f}")
    print(bar)
    for name, g in results["gates"].items():
        mark = "PASS" if g["pass"] else "FAIL"
        print(f"  [{mark}] {name:<22} ({g['need']})")
    print(bar)
    print(f">>> PHASE-3 GATE: {'PASS' if results['pass'] else 'FAIL'} <<<")
    print(bar)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run(args) -> int:
    if args.from_boards:
        board_rows = load_boards()
    else:
        scenarios = scengen.load_scenarios(regen=args.regen)
        if args.limit:
            scenarios = scenarios[: args.limit]
        print(f"loaded {len(scenarios)} scenarios; loading ranker once...", flush=True)
        from evaluation.phase1.gen_phase1 import MlxRanker  # lazy: imports mlx
        ranker = MlxRanker()
        ranker._temp = 0.0  # the seam picks the sampler; topk wants greedy
        board_rows = generate_boards(scenarios, ranker, sleep=args.sleep)

    results = evaluate_gates(score_boards(
        board_rows, metric_k=args.metric_k, dynamic_only=args.dynamic_only))
    output = args.output or RESULTS_PATH
    # Explicit analysis outputs are exclusive: preserve every prior result.
    with output.open("x" if args.output else "w") as f:
        f.write(json.dumps(results, indent=2))
    print(f"\nwrote {output}")

    try:
        from evaluation.phase3 import plots3
        if not args.output:
            plots3.render(results)
    except Exception as e:  # noqa: BLE001 - plotting must never fail the gate read
        print(f"(plots skipped: {e})")

    print_report(results)
    return 0 if results["pass"] else 1


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="VoiceLink Phase-3 self-test")
    p.add_argument("--regen", action="store_true", help="rebuild scenarios via the LLM")
    p.add_argument("--from-boards", action="store_true",
                   help="re-score cached boards.jsonl with no LLM")
    p.add_argument("--limit", type=int, default=0, help="cap scenarios (debug)")
    p.add_argument("--sleep", type=float, default=0.0,
                   help="seconds between LLM calls (thermal pacing)")
    p.add_argument("--metric-k", type=int, default=None,
                   help="score first K entries (default: historical 7)")
    p.add_argument("--dynamic-only", action="store_true",
                   help="remove core entries BEFORE applying --metric-k")
    p.add_argument("--output", type=Path, help="new JSON path; refuses overwrite, skips plots")
    args = p.parse_args(argv)
    if args.metric_k is not None and args.metric_k < 1:
        p.error("--metric-k must be positive")
    if (args.metric_k is not None or args.dynamic_only) and not args.output:
        p.error("explicit scoring windows require --output to preserve historical artifacts")
    if args.output and args.output.exists():
        p.error("--output already exists")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
