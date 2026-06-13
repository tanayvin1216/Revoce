"""
Vocabulary-prediction evaluation harness.

ONE COMMAND prints the baseline top-k vocabulary recall:

    python3 mac/evaluation/harness.py                 # gated dataset test split
    python3 mac/evaluation/harness.py --source fixture # bundled synthetic demo
    python3 mac/evaluation/harness.py --source real    # real taps from Task 3

The baseline is the Task 5 prompt (rank_vocabulary). Read the caveat it prints:
the synthetic dataset is GPT-generated and the baseline is itself an LLM prompt,
so beating it partly measures imitation of GPT, not real communication value.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make mac/ importable so we can reach the real baseline (rank_vocabulary etc.).
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


def run(args) -> int:
    from rank_vocabulary import rank_vocabulary

    examples = _load_examples(args)
    if not examples:
        print(f"No examples from source={args.source!r} "
              f"({'taps.jsonl not found / empty' if args.source == 'real' else 'empty'}).")
        if args.source == "real":
            print("Real interaction logs don't exist yet — that's expected pre-pilot.")
        return 1

    ranker = _build_ranker(args.ranker)
    ranker_name = type(ranker).__name__

    scores, latencies = [], []
    for i, ex in enumerate(examples, 1):
        t0 = time.time()
        items = rank_vocabulary(ex.scene, ex.transcript, ex.context, ranker=ranker)
        latencies.append(time.time() - t0)
        words = [it["word"] for it in items]
        scores.append(metrics.score_example(words, ex.ground_truth, args.ks))
        if args.verbose:
            hit = scores[-1].first_rank
            print(f"  [{i}/{len(examples)}] gt={ex.ground_truth!r} "
                  f"first_relevant_rank={hit} board={words[:5]}")

    agg = metrics.aggregate(scores, args.ks)
    mean_latency = sum(latencies) / len(latencies)
    _print_report(args, ranker_name, agg, mean_latency)
    return 0


def _print_report(args, ranker_name, agg, mean_latency) -> None:
    bar = "=" * 74
    label = {"dataset": "SYNTHETIC dataset (willwade/AACConversations)",
             "fixture": "SYNTHETIC fixture (bundled demo)",
             "real": "REAL interaction logs (Task 3 taps)"}[args.source]
    headline_k = max(args.ks)

    print("\n" + bar)
    print("VoiceLink — vocabulary-prediction baseline")
    print(bar)
    print(f"source : {label}")
    print(f"split  : {args.split if args.source == 'dataset' else args.source}   "
          f"N={agg.n}   ranker={ranker_name}")
    print()
    for k in args.ks:
        marker = "   <<< headline baseline" if k == headline_k else ""
        print(f"  recall@{k:<3} = {agg.recall[k] * 100:5.1f}%        "
              f"(token coverage {agg.token_recall[k] * 100:4.1f}%){marker}")
    print()
    mfr = f"{agg.mean_first_rank:.1f}" if agg.mean_first_rank is not None else "n/a"
    print(f"  time-to-relevant-word: MRR={agg.mrr:.3f}   "
          f"mean rank of first usable word={mfr}")
    print(f"  mean generation latency = {mean_latency:.2f}s / example")
    print()
    tag = "REAL-DATA RESULT" if args.source == "real" else "SYNTHETIC-BENCHMARK RESULT"
    print(f">>> {tag}: recall@{headline_k} = {agg.recall[headline_k] * 100:.1f}%  (N={agg.n}) <<<")
    print(bar)
    if args.source != "real":
        print(CAVEAT)
        print(bar)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="VoiceLink vocabulary-prediction baseline eval")
    p.add_argument("--source", choices=["dataset", "fixture", "real"], default="dataset")
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=200,
                   help="max examples (the full test split is ~17k -> ~14h of LLM calls)")
    p.add_argument("--ks", type=lambda s: [int(x) for x in s.split(",")], default=[1, 5, 10])
    p.add_argument("--ranker", default="", help="override RANKER: local-llm | heuristic")
    p.add_argument("--logs", default=str(MAC_DIR / "taps.jsonl"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
