"""
Vocabulary-prediction evaluation harness.

ONE COMMAND prints the baseline top-k vocabulary recall:

    python3 mac/evaluation/harness.py                       # gated dataset test split
    python3 mac/evaluation/harness.py --source fixture       # bundled synthetic demo
    python3 mac/evaluation/harness.py --source real          # real taps from Task 3

Matching is selectable and transparent:
    --match lexical      # shared content word / substring (strict; a floor)
    --match semantic     # embedding cosine >= --sim-threshold (credits paraphrase)

Generation is expensive (~3-4s/example of local LLM); scoring is cheap. So save
the model's boards once and re-score under any matcher for free:
    ... --save-predictions preds.jsonl       # run the LLM once, cache boards
    ... --from-predictions preds.jsonl --match semantic   # instant re-score

The baseline is the Task 5 prompt (rank_vocabulary). Read the caveat it prints:
the synthetic data is GPT-generated and the baseline is itself an LLM prompt, so
beating it partly measures imitation of GPT, not real communication value.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

MAC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MAC_DIR))

from evaluation import dataset as ds         # noqa: E402
from evaluation import metrics               # noqa: E402

CAVEAT = (
    "CAVEAT — SYNTHETIC BENCHMARK. The willwade/AACConversations data is\n"
    "  GPT-generated and text-only, and this baseline is itself an LLM prompt.\n"
    "  A high score here partly measures IMITATION OF GPT, not real-world\n"
    "  communication value. Treat this as a cold-start sanity signal only.\n"
    "  The number that matters comes from REAL consented interaction logs:\n"
    "  run with `--source real` once a pilot has produced taps.jsonl (Task 3)."
)


def _load_examples(args):
    if args.source == "fixture":
        return ds.from_fixture(Path(__file__).resolve().parent / "fixture.json", args.limit)
    if args.source == "real":
        return ds.from_real_logs(Path(args.logs), args.limit)
    return ds.from_dataset(args.split, args.limit, args.seed)


def _build_ranker(name: str):
    import ranker_config
    from ranker import make_ranker
    if name:
        ranker_config.RANKER = name
    return make_ranker(ranker_config)


def _load_predictions(path: Path) -> list[dict]:
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "ground_truth" in rec and "predicted" in rec:
            out.append(rec)
    return out


def _generate(args, examples):
    """Run the baseline over examples, returning {ground_truth, predicted} records.
    Optionally streams them to --save-predictions so a future run can re-score."""
    from rank_vocabulary import rank_vocabulary

    ranker = _build_ranker(args.ranker)
    ranker_name = type(ranker).__name__
    save_f = open(args.save_predictions, "w") if args.save_predictions else None

    records, latencies = [], []
    for i, ex in enumerate(examples, 1):
        t0 = time.time()
        items = rank_vocabulary(ex.scene, ex.transcript, ex.context, ranker=ranker)
        latencies.append(time.time() - t0)
        rec = {"ground_truth": ex.ground_truth, "predicted": [it["word"] for it in items]}
        records.append(rec)
        if save_f:
            save_f.write(json.dumps(rec) + "\n")
            save_f.flush()
        if args.verbose:
            print(f"  [{i}/{len(examples)}] gt={ex.ground_truth!r} board={rec['predicted'][:5]}")
    if save_f:
        save_f.close()
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    return records, ranker_name, mean_latency


def run(args) -> int:
    matcher = metrics.make_matcher(args.match, args.sim_threshold)

    if args.from_predictions:
        records = _load_predictions(Path(args.from_predictions))
        if not records:
            print(f"No cached predictions at {args.from_predictions!r}.")
            return 1
        ranker_name, mean_latency = "cached-predictions", 0.0
    else:
        examples = _load_examples(args)
        if not examples:
            print(f"No examples from source={args.source!r}.")
            if args.source == "real":
                print("Real interaction logs don't exist yet — expected pre-pilot.")
            return 1
        records, ranker_name, mean_latency = _generate(args, examples)

    scores = [metrics.score_example(r["predicted"], r["ground_truth"], args.ks, matcher)
              for r in records]
    agg = metrics.aggregate(scores, args.ks, matcher.secondary_label)
    _print_report(args, ranker_name, agg, mean_latency)
    return 0


def _print_report(args, ranker_name, agg, mean_latency) -> None:
    bar = "=" * 74
    label = {"dataset": "SYNTHETIC dataset (willwade/AACConversations)",
             "fixture": "SYNTHETIC fixture (bundled demo)",
             "real": "REAL interaction logs (Task 3 taps)"}[args.source]
    headline_k = max(args.ks)
    match_desc = (f"semantic (cos>={args.sim_threshold})" if args.match == "semantic"
                  else "lexical (content-word / substring)")

    print("\n" + bar)
    print("VoiceLink — vocabulary-prediction baseline")
    print(bar)
    print(f"source : {label}")
    print(f"match  : {match_desc}")
    print(f"split  : {args.split if args.source == 'dataset' else args.source}   "
          f"N={agg.n}   ranker={ranker_name}")
    print()
    for k in args.ks:
        sec = agg.secondary[k]
        sec_str = f"{sec * 100:4.1f}%" if "coverage" in agg.secondary_label else f"{sec:.2f}"
        marker = "   <<< headline baseline" if k == headline_k else ""
        print(f"  recall@{k:<3} = {agg.recall[k] * 100:5.1f}%        "
              f"({agg.secondary_label} {sec_str}){marker}")
    print()
    mfr = f"{agg.mean_first_rank:.1f}" if agg.mean_first_rank is not None else "n/a"
    print(f"  time-to-relevant-word: MRR={agg.mrr:.3f}   mean rank of first usable word={mfr}")
    if mean_latency:
        print(f"  mean generation latency = {mean_latency:.2f}s / example")
    print()
    tag = "REAL-DATA RESULT" if args.source == "real" else "SYNTHETIC-BENCHMARK RESULT"
    print(f">>> {tag} ({args.match} match): "
          f"recall@{headline_k} = {agg.recall[headline_k] * 100:.1f}%  (N={agg.n}) <<<")
    print(bar)
    if args.source != "real":
        print(CAVEAT)
        print(bar)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="VoiceLink vocabulary-prediction baseline eval")
    p.add_argument("--source", choices=["dataset", "fixture", "real"], default="dataset")
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=200,
                   help="max examples (full test split ~17k -> ~14h of LLM calls)")
    p.add_argument("--ks", type=lambda s: [int(x) for x in s.split(",")], default=[1, 5, 10])
    p.add_argument("--match", choices=["lexical", "semantic"], default="lexical")
    p.add_argument("--sim-threshold", type=float, default=0.5)
    p.add_argument("--ranker", default="", help="override RANKER: local-llm | heuristic")
    p.add_argument("--logs", default=str(MAC_DIR / "taps.jsonl"))
    p.add_argument("--save-predictions", default="", help="cache boards to this JSONL")
    p.add_argument("--from-predictions", default="", help="re-score cached boards (no LLM)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
