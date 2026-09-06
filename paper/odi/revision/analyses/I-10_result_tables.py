"""I-10 — the extra tables for I-10_result.md, computed directly from the raw session files
of the corrected runs (I-10_bench_v3 + I-10_bench_v4; bench_e2e.py sha 70dbd66a…).

Everything here is derived from `samples_ms[round][scenario]` in those files; nothing is
timed. Run from mac/ with /usr/local/bin/python3 (only json/numpy needed):

    /usr/local/bin/python3 ../paper/odi/revision/analyses/I-10_result_tables.py

Writes I-10_result_tables.json next to this file (exclusive create — never overwrites).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = [HERE / "I-10_bench_v3", HERE / "I-10_bench_v4"]
OUT = HERE / "I-10_result_tables.json"
BACKENDS = ("torch", "ov-head", "ov-full")
FAMS = ("i05b", "hist", "i05")


def pct(v, q):
    return float(np.percentile(np.asarray(v, dtype=float), q))


def load_sessions():
    sessions = []
    for rd in RUNS:
        for p in sorted(rd.glob("*.json")):
            if p.name == "run_manifest.json":
                continue
            x = json.loads(p.read_text())
            for s in x["sessions"]:
                s["_run"] = rd.name
                s["_scenario_ids"] = x["scenario_ids"]
                s["_minilm"] = x["minilm_bundle_sha256"]
                s["_code"] = x["code_sha256"]
                sessions.append(s)
    return sessions


def flat(s):
    return [v for r in s["samples_ms"] for v in r]


def by_scenario(sessions, n=84):
    cols = [[] for _ in range(n)]
    for s in sessions:
        for rnd in s["samples_ms"]:
            for i, v in enumerate(rnd):
                cols[i].append(v)
    return cols


def main() -> None:
    if OUT.exists():
        sys.exit(f"{OUT} exists — refusing to overwrite")
    S = load_sessions()
    out: dict = {"runs": [r.name for r in RUNS], "n_sessions_loaded": len(S), "families": {}}
    # every corrected session must come from the fixed bench_e2e (no scenario-0 warm defect)
    codes = {s["_code"]["mac/intel/bench_e2e.py"] for s in S}
    assert codes == {"70dbd66aa525b359f90bdabeaac61e799480d948aef796e6acba008056f5e944"}, codes
    for s in S:
        if s["protocol"]["evict"] == "board":
            assert min(flat(s)) > 1.0, (s["session_id"], "sub-millisecond board => warm scenario")
    out["bench_e2e_sha256"] = sorted(codes)[0]

    for fam in FAMS:
        F = [s for s in S if s["family"] == fam]
        fo: dict = {}
        # provenance recorded in the sessions (must be one artifact per backend)
        prov = {}
        for b in BACKENDS:
            heads = {s["label_checks"]["head_artifact"] for s in F if s["backend"] == b}
            shas = {json.dumps(s["provenance"]["inputs_sha256"], sort_keys=True) for s in F if s["backend"] == b}
            assert len(heads) == 1 and len(shas) == 1, (fam, b, heads)
            prov[b] = {"head_artifact": heads.pop(), "inputs_sha256": json.loads(shas.pop()),
                       "vocab_size": F[0]["provenance"]["vocab_size"], "config_hash": F[0]["provenance"]["config_hash"],
                       "vocab_sha": F[0]["provenance"]["vocab_sha"]}
        fo["provenance"] = prov
        fo["minilm_bundle_sha256"] = F[0]["_minilm"]
        for b in BACKENDS:
            cb = [s for s in F if s["backend"] == b and s["tag"].startswith("order")]
            iso = [s for s in F if s["backend"] == b and s["tag"] == "isolated"]
            hp = [s for s in F if s["backend"] == b and s["tag"] == "histproto"]
            assert len(cb) == 6 and len(iso) == 1 and len(hp) == 1, (fam, b, len(cb), len(iso), len(hp))
            row: dict = {"n_counterbalanced_sessions": len(cb),
                         "timed_calls_counterbalanced": sum(len(flat(s)) for s in cb),
                         "timed_calls_isolated": len(flat(iso[0])), "timed_calls_histproto": len(flat(hp[0]))}
            # (1) position effect, both runs pooled per position (2 sessions each)
            pos = {}
            for p in (0, 1, 2):
                ss = [s for s in cb if s["position"] == p]
                v = [x for s in ss for x in flat(s)]
                pos[p] = {"sessions": [s["_run"] + "/" + s["tag"] for s in ss], "n": len(v),
                          "p50": pct(v, 50), "p90": pct(v, 90), "session_p50s": [pct(flat(s), 50) for s in ss]}
            row["by_position_pooled"] = pos
            allv = [x for s in cb for x in flat(s)]
            row["pooled_p50"], row["pooled_p90"] = pct(allv, 50), pct(allv, 90)
            row["position_p50_range"] = [min(p["p50"] for p in pos.values()), max(p["p50"] for p in pos.values())]
            row["session_p50_sd"] = float(np.std([pct(flat(s), 50) for s in cb], ddof=1))
            # (2) isolated vs in-process
            iv = flat(iso[0])
            row["isolated"] = {"p50": pct(iv, 50), "p90": pct(iv, 90), "delta_p50_vs_pooled": pct(iv, 50) - row["pooled_p50"],
                               "delta_p90_vs_pooled": pct(iv, 90) - row["pooled_p90"], "run": iso[0]["_run"]}
            # (3) per-round p90 of the first-position counterbalanced sessions vs the historical-protocol control
            row["first_position_per_round_p90"] = {s["_run"] + "/" + s["tag"]: [pct(r, 90) for r in s["samples_ms"]]
                                                   for s in cb if s["position"] == 0}
            row["first_position_per_round_p50"] = {s["_run"] + "/" + s["tag"]: [pct(r, 50) for r in s["samples_ms"]]
                                                   for s in cb if s["position"] == 0}
            hv = flat(hp[0])
            row["histproto"] = {"position": hp[0]["position"], "p50": pct(hv, 50), "p90": pct(hv, 90),
                                "per_round_p50": [pct(r, 50) for r in hp[0]["samples_ms"]],
                                "per_round_p90": [pct(r, 90) for r in hp[0]["samples_ms"]],
                                "rounds_2_3_p90": pct([v for r in hp[0]["samples_ms"][1:] for v in r], 90)}
            # (5) tail composition: scenario medians, question flags, text lengths
            cols = by_scenario(cb)
            med = [float(np.median(c)) for c in cols]
            order = sorted(range(84), key=lambda i: -med[i])
            ids = cb[0]["_scenario_ids"]
            row["slowest_10"] = [{"index": i, "median_ms": med[i], "is_question": ids[i]["is_question"],
                                  "category": ids[i]["category"]} for i in order[:10]]
            q = [i for i in range(84) if ids[i]["is_question"]]
            nq = [i for i in range(84) if not ids[i]["is_question"]]
            row["scenario_median_by_question_flag"] = {
                "question": {"n": len(q), "median_of_medians": float(np.median([med[i] for i in q])),
                             "p90_of_samples": pct([v for i in q for v in cols[i]], 90)},
                "non_question": {"n": len(nq), "median_of_medians": float(np.median([med[i] for i in nq])),
                                 "p90_of_samples": pct([v for i in nq for v in cols[i]], 90)}}
            thr = row["pooled_p90"]
            above = [(i, sum(v > thr for v in cols[i])) for i in range(84)]
            n_above = sum(c for _, c in above)
            scen_with_any = sum(1 for _, c in above if c > 0)
            top5 = sum(c for _, c in sorted(above, key=lambda t: -t[1])[:5])
            row["above_p90"] = {"n_samples_above": n_above, "scenarios_contributing": scen_with_any,
                                "share_from_top5_scenarios": top5 / max(n_above, 1)}
            # scenario-level median IQR: how much of the spread is between-scenario
            row["between_scenario_median_p25_p75"] = [pct(med, 25), pct(med, 75)]
            fo[b] = row
        # (4) paired per-scenario ratio torch / ov-full and torch / ov-head (medians over the 6 sessions)
        med = {b: [float(np.median(c)) for c in by_scenario([s for s in F if s["backend"] == b and s["tag"].startswith("order")])]
               for b in BACKENDS}
        fo["paired_ratio_torch_over_ov_full"] = {"median": float(np.median([a / b for a, b in zip(med["torch"], med["ov-full"])])),
                                                 "p25_p75": [pct([a / b for a, b in zip(med["torch"], med["ov-full"])], 25),
                                                             pct([a / b for a, b in zip(med["torch"], med["ov-full"])], 75)]}
        fo["paired_ratio_torch_over_ov_head"] = {"median": float(np.median([a / b for a, b in zip(med["torch"], med["ov-head"])]))}
        # board determinism and Jaccard, reference = torch boards of the same family
        ref = [s for s in F if s["backend"] == "torch"][0]["boards"]
        for b in BACKENDS:
            bs = [s["boards"] for s in F if s["backend"] == b]
            J = [len({w.lower() for w in x} & {w.lower() for w in y}) / len({w.lower() for w in x} | {w.lower() for w in y})
                 for x, y in zip(bs[0], ref)]
            fo[b]["boards"] = {"sessions": len(bs), "identical_across_sessions": all(x == bs[0] for x in bs),
                               "jaccard_vs_torch": float(np.mean(J)), "identical_vs_torch": int(sum(x == y for x, y in zip(bs[0], ref))),
                               "min_jaccard": float(min(J))}
        out["families"][fam] = fo
    OUT.write_text(json.dumps(out, indent=1))
    # short print
    for fam in FAMS:
        fo = out["families"][fam]
        print(f"\n== {fam}  ratio torch/ov-full median {fo['paired_ratio_torch_over_ov_full']['median']:.2f} "
              f"IQR {fo['paired_ratio_torch_over_ov_full']['p25_p75']}")
        for b in BACKENDS:
            r = fo[b]
            print(f"{b:8s} pooled p50 {r['pooled_p50']:.2f} p90 {r['pooled_p90']:.2f} | pos p50 "
                  f"{[round(r['by_position_pooled'][p]['p50'],2) for p in (0,1,2)]} p90 {[round(r['by_position_pooled'][p]['p90'],2) for p in (0,1,2)]} "
                  f"| iso {r['isolated']['p50']:.2f}/{r['isolated']['p90']:.2f} (d {r['isolated']['delta_p50_vs_pooled']:+.2f}/{r['isolated']['delta_p90_vs_pooled']:+.2f}) "
                  f"| sess p50 sd {r['session_p50_sd']:.2f}")
            print(f"         histproto pos{r['histproto']['position']} p50 {r['histproto']['p50']:.2f} p90 {r['histproto']['p90']:.2f} "
                  f"rounds p90 {[round(x,1) for x in r['histproto']['per_round_p90']]} rounds2-3 p90 {r['histproto']['rounds_2_3_p90']:.2f}")
            print(f"         pos0 per-round p90: { {k: [round(x,1) for x in v] for k, v in r['first_position_per_round_p90'].items()} }")
            print(f"         q/nq median-of-medians {r['scenario_median_by_question_flag']['question']['median_of_medians']:.2f} "
                  f"(n={r['scenario_median_by_question_flag']['question']['n']}) / "
                  f"{r['scenario_median_by_question_flag']['non_question']['median_of_medians']:.2f}; "
                  f"above-p90: {r['above_p90']}; slowest {[(d['index'], round(d['median_ms'],1), d['is_question']) for d in r['slowest_10'][:6]]}")
            print(f"         boards {r['boards']}")
        print("  prov:", {b: fo["provenance"][b]["head_artifact"].split("/")[-2:] for b in BACKENDS},
              {k: v[:12] for k, v in fo["provenance"]["ov-full"]["inputs_sha256"].items()})
    print("minilm", {k: v[:12] for k, v in out["families"]["i05b"]["minilm_bundle_sha256"].items()})
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
