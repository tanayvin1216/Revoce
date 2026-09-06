"""I-16 CHECKER: independent recount of vocabulary representability, merge counts.

Deliberately does NOT import the implementer's script or the repo's phase4
helpers: JSONL is parsed directly, core entries are hard-coded from
rank_vocabulary.GUARANTEED_CORE, and the canonical lookup is re-implemented.
Run from mac/ with VOICELINK_PERCEPTION=0:
  /usr/local/bin/python3 ../paper/odi/revision/analyses/I-16_recount_check.py
Writes I-16_recount_check.json next to this file (exclusive create).
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OLD = ROOT / "mac/evaluation/phase4"
NEW = HERE / "I-05_split_first_v2"
CORE = {w.lower() for w in ["yes", "no", "I need help", "wait"]}  # rank_vocabulary.py:38


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def norm(w: str) -> str:
    return w.strip().lower()


def load_raw(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            assert not r.get("strategy")  # teacher_data has no arm column
            rows.append(r)
    return rows


def recount(rows: list[dict], vocab: list[str], canon: dict[str, str]) -> dict:
    vs = set(vocab)
    slots = exact = canon_hits = zero_exact = zero_canon = 0
    slot_hist: dict[int, int] = {}
    for r in rows:
        dyn = [norm(w) for w in r["board"] if norm(w) not in CORE]
        slot_hist[len(dyn)] = slot_hist.get(len(dyn), 0) + 1
        e = sum(w in vs for w in dyn)
        c = sum((canon.get(w, w) in vs) for w in dyn)
        # a canonical hit is a superset of an exact hit because map keys are not in vocab
        assert c >= e
        slots += len(dyn)
        exact += e
        canon_hits += c
        zero_exact += e == 0
        zero_canon += c == 0
    n = len(rows)
    return dict(rows=n, slots=slots, slot_histogram=slot_hist,
                exact_hits=exact, exact_pct=round(100 * exact / slots, 3),
                canonical_hits=canon_hits, canonical_pct=round(100 * canon_hits / slots, 3),
                zero_exact=zero_exact, zero_exact_pct=round(100 * zero_exact / n, 3),
                zero_canonical=zero_canon, zero_canonical_pct=round(100 * zero_canon / n, 3))


def main() -> None:
    out_path = HERE / "I-16_recount_check.json"
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in [
        OLD / "vocab.json", OLD / "canonical.json", OLD / "teacher_data.jsonl", OLD / "splits.json",
        NEW / "vocab.json", NEW / "canonical.json", NEW / "rows.json", NEW / "splits.json"]}

    raw = load_raw(OLD / "teacher_data.jsonl")
    gathered = json.loads((NEW / "rows.json").read_text())
    splits = json.loads((NEW / "splits.json").read_text())
    assert splits == json.loads((OLD / "splits.json").read_text())
    idx = splits["train"] + splits["val"] + splits["test"]
    assert sorted(idx) == list(range(len(gathered))) and len(gathered) == 559
    assert (len(splits["train"]), len(splits["val"]), len(splits["test"])) == (391, 84, 84)

    # Independent fingerprint: sha256 over "\n".join(f"{scene} || {transcript}")-style
    # text is the repo's context_text; re-derive its exact form from data.py source.
    src = (OLD / "data.py").read_text()
    ctx_line = [l for l in src.splitlines() if l.strip().startswith("return") and "scene" in l and "transcript" in l]
    # Provenance: which gathered rows come from teacher_data.jsonl (by scene+transcript)?
    raw_keys = {(r["scene"], r["transcript"]) for r in raw}
    from_teacher = sum((r["scene"], r["transcript"]) in raw_keys for r in gathered)
    # Do the gathered boards equal the raw boards for the same context?
    raw_board = {(r["scene"], r["transcript"]): r["board"] for r in raw}
    same_board = sum(raw_board.get((r["scene"], r["transcript"])) == r["board"] for r in gathered)

    corpora = {"raw600": raw, "gathered559": gathered}
    corpora.update({s: [gathered[i] for i in splits[s]] for s in ("train", "val", "test")})

    results = {}
    for label, bundle in (("historical95", OLD), ("I05_train_only45", NEW)):
        vocab = json.loads((bundle / "vocab.json").read_text())
        canon = json.loads((bundle / "canonical.json").read_text())
        assert len(set(vocab)) == len(vocab)
        assert all(v == norm(v) for v in vocab), "vocab not normalised"
        assert not (set(canon) & set(vocab)) and set(canon.values()) <= set(vocab)
        clusters = {}
        for k, v in canon.items():
            clusters.setdefault(v, []).append(k)
        results[label] = dict(
            vocab_size=len(vocab), canonical_mappings=len(canon),
            canonical_clusters=len(clusters),
            cluster_sizes={v: len(k) + 1 for v, k in clusters.items()},
            corpora={k: recount(v, vocab, canon) for k, v in corpora.items()})

    # Also: what fraction of the I-05 vocabulary/map is a subset of the historical one?
    v_old = set(json.loads((OLD / "vocab.json").read_text()))
    v_new = set(json.loads((NEW / "vocab.json").read_text()))
    c_old = json.loads((OLD / "canonical.json").read_text())
    c_new = json.loads((NEW / "canonical.json").read_text())
    subset = dict(I05_vocab_subset_of_historical=v_new <= v_old,
                  I05_vocab_not_in_historical=sorted(v_new - v_old),
                  I05_map_subset_of_historical=all(c_old.get(k) == v for k, v in c_new.items()),
                  I05_map_keys_not_in_historical=sorted(set(c_new) - set(c_old)))

    payload = dict(issue="I-16", checker="second-pass", input_sha256=hashes,
                   provenance=dict(gathered_rows_with_context_in_teacher_data=from_teacher,
                                   gathered_rows_with_identical_board=same_board,
                                   context_text_source_line=ctx_line),
                   subset=subset, results=results)
    with out_path.open("x") as f:
        json.dump(payload, f, indent=2)

    # Compare against the implementer's recount.json and the paper's replacement table.
    impl = json.loads((HERE / "I-16_recount_v1/recount.json").read_text())
    mismatch = []
    for label in results:
        for corp, mine in results[label]["corpora"].items():
            theirs = impl["results"][label]["corpora"][corp]
            pairs = [("slots", mine["slots"], theirs["slots"]),
                     ("exact", mine["exact_hits"], theirs["exact"]["hits"]),
                     ("canonical", mine["canonical_hits"], theirs["canonical"]["hits"]),
                     ("zero_exact", mine["zero_exact"], theirs["exact"]["zero_hit_rows"]),
                     ("zero_canon", mine["zero_canonical"], theirs["canonical"]["zero_hit_rows"])]
            for name, a, b in pairs:
                if a != b:
                    mismatch.append((label, corp, name, a, b))
        if results[label]["vocab_size"] != impl["results"][label]["vocab_size"]:
            mismatch.append((label, "vocab_size"))
        if results[label]["canonical_mappings"] != impl["results"][label]["canonical_mappings"]:
            mismatch.append((label, "canonical_mappings"))
    print("hash check vs implementer:",
          all(impl["input_sha256"].get(k) == v for k, v in hashes.items()
              if k in impl["input_sha256"]))
    print("mismatches vs implementer:", mismatch or "NONE")
    print(json.dumps(payload["provenance"], indent=1))
    print(json.dumps(subset, indent=1))
    for label, r in results.items():
        print(label, "vocab", r["vocab_size"], "mappings", r["canonical_mappings"],
              "clusters", r["canonical_clusters"], "sizes", sorted(r["cluster_sizes"].values(), reverse=True))
        for corp, m in r["corpora"].items():
            print(f"  {corp:12s} slots {m['slots']:5d} hist {m['slot_histogram']} "
                  f"exact {m['exact_hits']:4d} ({m['exact_pct']:6.2f}%) "
                  f"canon {m['canonical_hits']:4d} ({m['canonical_pct']:6.2f}%) "
                  f"zeroE {m['zero_exact']:3d} ({m['zero_exact_pct']:5.2f}%) "
                  f"zeroC {m['zero_canonical']:3d} ({m['zero_canonical_pct']:5.2f}%)")


if __name__ == "__main__":
    sys.exit(main())
