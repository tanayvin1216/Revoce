#!/usr/bin/env python3
"""I-06 step 1 — regenerate ONE frozen teacher candidate pool per Phase-3 scenario.

Why: boards.jsonl stores only the final selected boards, never the tagged
candidate pool that rank_vocabulary.generate_pool() produced, so a same-pool
selector ablation (relevance-only vs MMR vs the shipped policy) cannot be
derived from cache (audit feasibility item F1, G-03). This script makes
exactly one teacher call per scenario through the SAME path the Phase-3
harness uses (selftest.build_context -> topics_from_context -> generate_pool
with MlxRanker at temperature 0) and stores the pool with everything the
checker needs: the raw model text, parse count, exception (if any), per-call
wall-clock, answer-mode flag, and a SHA-256 hash of the pool content.

Run from mac/ with the interpreter the revision uses:

    VOICELINK_PERCEPTION=0 /usr/local/bin/python3 \
        ../paper/odi/revision/analyses/I-06_gen_pools.py \
        --output ../paper/odi/revision/analyses/I-06_pools.jsonl [--limit N]

The output is append-only and resumable (scenario ids already present are
skipped), so a wall-clock cap can never lose finished work. It never touches
boards.jsonl, selftest_results.json or any other historical artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

# ---- environment: the teacher is MLX, embeddings are not needed here --------
os.environ.setdefault("VOICELINK_PERCEPTION", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = Path(__file__).resolve().parent
MAC = HERE.parents[3] / "mac"
if str(MAC) not in sys.path:
    sys.path.insert(0, str(MAC))
os.chdir(MAC)  # the repo's relative artifact paths assume mac/ as cwd

from evaluation.phase3 import scengen, selftest  # noqa: E402
from evaluation.phase1.gen_phase1 import MlxRanker, MODEL_ID  # noqa: E402
import rank_vocabulary as rv  # noqa: E402


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def pool_hash(pool: list[dict]) -> str:
    """Content hash over (word, function, topic) in pool order — the identity
    the checker matches against when it re-scores the stored pools."""
    canon = [[p.get("word", ""), p.get("function", ""), p.get("topic", "")] for p in pool]
    return hashlib.sha256(json.dumps(canon, ensure_ascii=False).encode()).hexdigest()


class RecordingRanker:
    """Wraps MlxRanker so the LAST call's raw text, parsed count, exception and
    wall-clock are observable. generate_pool() swallows ranker exceptions and
    silently returns the core pool; without this wrapper a failed teacher call
    would look like a (tiny) valid pool. Same prompts, same greedy sampler."""

    def __init__(self, inner: MlxRanker) -> None:
        self.inner = inner
        self.last: dict = {}

    def rank(self, system: str, user: str, max_tokens: int | None = None):
        t0 = time.perf_counter()
        rec = {"max_tokens": max_tokens, "exception": None, "n_parsed": 0,
               "raw_text": None, "seconds": None, "system_sha": None, "user_sha": None}
        rec["system_sha"] = hashlib.sha256(system.encode()).hexdigest()[:16]
        rec["user_sha"] = hashlib.sha256(user.encode()).hexdigest()[:16]
        self.last = rec
        try:
            text = self.inner.raw(system, user, temp=0.0, max_tokens=max_tokens or 300)
            rec["raw_text"] = text
            from ranker import _parse_json_array
            items = _parse_json_array(text)
            rec["n_parsed"] = len(items)
            rec["raw_items"] = items
            return items
        except Exception as exc:  # generate_pool would swallow this; we record it
            rec["exception"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            rec["seconds"] = time.perf_counter() - t0


def done_ids(path: Path) -> set[str]:
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


def one_pool(sc: dict, ranker: RecordingRanker) -> dict:
    scene, transcript, context = selftest.build_context(sc)
    topics = rv.topics_from_context(scene, transcript, context)
    answer_mode = rv._is_question(transcript, context)
    t0 = time.perf_counter()
    pool = rv.generate_pool(scene, transcript, context, ranker=ranker, topics=topics)
    wall = time.perf_counter() - t0
    core_idx, rest_idx = rv._split_core(pool)
    core_words = {w.lower() for w in rv.GUARANTEED_CORE}
    stub_words = {s["word"].lower() for s in rv._STUB_WORDS}
    llm_words = {str(it.get("word", "")).strip().lower() for it in
                 (ranker.last.get("raw_items") or [])}
    n_from_llm = sum(1 for p in pool if p["word"].lower() not in
                     {c["word"].lower() for c in rv._CORE_POOL})
    rec = ranker.last
    status = "ok"
    if rec.get("exception"):
        status = "teacher_failed"
    elif rec.get("n_parsed", 0) == 0:
        status = "empty_parse"
    elif n_from_llm < 6:
        status = "thin_pool"  # fewer than six LLM items survived -> stubs would pad
    return {
        "id": sc["id"], "category": sc["category"],
        "scene": sc.get("scene", ""), "transcript": sc.get("transcript", ""),
        "active_question": sc.get("active_question", ""),
        "is_question": bool(sc.get("is_question")),
        "acceptable_moves": sc.get("acceptable_moves", []),
        "gold_utterance": sc.get("gold_utterance", ""),
        "context": context, "topics": topics, "answer_mode": answer_mode,
        "pool": pool, "pool_hash": pool_hash(pool),
        "n_pool": len(pool), "n_core_pinned": len(core_idx), "n_rest": len(rest_idx),
        "n_from_llm": n_from_llm,
        "n_answer_tagged": sum(1 for p in pool if p.get("function") == "answer"),
        "teacher": {"n_parsed": rec.get("n_parsed"), "exception": rec.get("exception"),
                    "seconds": rec.get("seconds"), "max_tokens": rec.get("max_tokens"),
                    "system_sha": rec.get("system_sha"), "user_sha": rec.get("user_sha"),
                    "raw_text": rec.get("raw_text")},
        "generate_pool_seconds": wall,
        "status": status,
        "_unused_llm_words": sorted(llm_words - {p["word"].lower() for p in pool}),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="cap scenarios (smoke test)")
    ap.add_argument("--max-seconds", type=float, default=2400.0,
                    help="stop starting new teacher calls after this wall-clock")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="where to write the run manifest (default: <output>.manifest.json)")
    args = ap.parse_args(argv)
    out: Path = args.output
    manifest_path = args.manifest or out.with_suffix(out.suffix + ".manifest.json")

    t_start = time.perf_counter()
    scenarios = scengen.load_scenarios(regen=False)
    assert len(scenarios) == 120, f"expected 120 Phase-3 scenarios, got {len(scenarios)}"
    if args.limit:
        scenarios = scenarios[: args.limit]

    already = done_ids(out)
    todo = [sc for sc in scenarios if sc["id"] not in already]
    print(f"[I-06] {len(scenarios)} scenarios, {len(already)} already pooled, {len(todo)} to run",
          flush=True)

    t_load = time.perf_counter()
    ranker = RecordingRanker(MlxRanker())
    load_s = time.perf_counter() - t_load
    print(f"[I-06] teacher loaded: {MODEL_ID} in {load_s:.1f}s", flush=True)

    import mlx.core as mx
    import mlx_lm
    manifest = {
        "issue": "I-06", "step": "generate_pools",
        "model_id": MODEL_ID,
        "hf_snapshot": sorted(p.name for p in (Path.home() / ".cache/huggingface/hub" /
                              ("models--" + MODEL_ID.replace("/", "--")) / "snapshots").glob("*")),
        "mlx_version": mx.__version__, "mlx_lm_version": getattr(mlx_lm, "__version__", "?"),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "temperature": 0.0, "sampler": "greedy (temp=0.0), same as selftest.run",
        "max_tokens_rule": "generate_pool: 720 in answer mode else 520",
        "inputs": {
            "scenarios.jsonl": sha256_file(MAC / "evaluation/phase3/scenarios.jsonl"),
            "rank_vocabulary.py": sha256_file(MAC / "rank_vocabulary.py"),
            "selftest.py": sha256_file(MAC / "evaluation/phase3/selftest.py"),
            "gen_phase1.py": sha256_file(MAC / "evaluation/phase1/gen_phase1.py"),
            "ranker.py": sha256_file(MAC / "ranker.py"),
        },
        "load_seconds": load_s, "limit": args.limit, "max_seconds": args.max_seconds,
        "resumed_from": len(already),
    }

    n_ok = n_bad = 0
    call_secs: list[float] = []
    with out.open("a") as f:
        for i, sc in enumerate(todo, 1):
            elapsed = time.perf_counter() - t_start
            if elapsed > args.max_seconds:
                print(f"[I-06] wall-clock cap {args.max_seconds}s reached after {i-1} calls; "
                      f"stopping (resumable)", flush=True)
                manifest["stopped_by_cap"] = True
                break
            row = one_pool(sc, ranker)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            call_secs.append(row["teacher"]["seconds"] or 0.0)
            if row["status"] == "ok":
                n_ok += 1
            else:
                n_bad += 1
            print(f"[{len(already)+i}/{len(scenarios)}] {sc['id']:<26} {row['status']:<14} "
                  f"pool={row['n_pool']:>2} rest={row['n_rest']:>2} llm={row['n_from_llm']:>2} "
                  f"ans={row['n_answer_tagged']:>2} {row['teacher']['seconds']:.1f}s", flush=True)
            if i == 1 and row["status"] != "ok":
                # The plan requires one successful teacher call before the batch.
                print("[I-06] FIRST teacher call did not produce a usable pool; aborting "
                      "before the batch. Inspect the row written above.", flush=True)
                manifest["aborted_after_first_call"] = row["status"]
                break

    total = time.perf_counter() - t_start
    manifest.update({
        "calls_made": len(call_secs), "ok": n_ok, "not_ok": n_bad,
        "teacher_seconds_mean": (sum(call_secs) / len(call_secs)) if call_secs else None,
        "teacher_seconds_min": min(call_secs) if call_secs else None,
        "teacher_seconds_max": max(call_secs) if call_secs else None,
        "wall_seconds_total": total,
        "output": str(out), "output_sha256": sha256_file(out) if out.exists() else None,
        "rows_in_output": len(done_ids(out)),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[I-06] done: {len(call_secs)} calls, ok={n_ok} not_ok={n_bad}, "
          f"{total:.1f}s wall; manifest -> {manifest_path}", flush=True)
    return 0 if n_bad == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
