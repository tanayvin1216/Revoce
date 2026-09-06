"""Render the I-04b score JSON (i05b family) as markdown tables, side by side with the I-04 historical family
(read-only from I-04_scores_v1.json, labelled `hist`) — used verbatim in I-04b_result.md."""
import json, sys
d = json.load(open(sys.argv[1]))                       # I-04b_scores_v1.json (i05b family)
h = json.load(open(sys.argv[2])) if len(sys.argv) > 3 else None   # I-04_scores_v1.json (hist + i05 families)
out_path = sys.argv[3] if len(sys.argv) > 3 else sys.argv[2]
cfgs = dict(d["configurations"])
cross = dict(d["cross_backend"])
if h is not None:
    for k, v in h["configurations"].items():
        cfgs[k] = v
    for k, v in h["cross_backend"].items():
        cross[k] = v
order = [f"{f}/{hd}/{e}" for f in ("i05b", "hist", "i05") for hd in ("torch", "ov-fp32", "ov-int8") for e in ("torch", "openvino")]
order = [k for k in order if k in cfgs]
out = []
def fmt_m(m): return f"{m['margin']:+.4f} [{m['ci'][0]:+.4f}, {m['ci'][1]:+.4f}]"
for src in ("direct", "seam"):
    for w in ("full10", "dyn6", "full7"):
        out.append(f"\n### {src} boards, window {w}\n")
        out.append("| config | embed used | head runtime | DF S/T | DF ret | DF margin | SFP S/T | SFP margin (T-S) | variety | dyn cov S/T (n=71) | ret | pairs 11/01/10/00 | cov margin S-0.9T | point | raw S-T |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for key in order:
            s = cfgs[key]; m = s["windows"][src][w]["summary"]; a = s["asserted"]
            p = m["paired_counts"]; c = m["dynamic_coverage_counts"]
            out.append(f"| {key} | {a['embed_backend_used']} | {a['head_runtime']} | {m['df'][0]:.3f}/{m['df'][1]:.3f} | {m['df_retention']:.3f} | {fmt_m(m['df_margin'])} | {m['sfp'][0]:.4f}/{m['sfp'][1]:.4f} | {fmt_m(m['sfp_margin'])} {m['sfp_margin']['verdict']} | {m['board_variety']:.3f} | {c[0]}/{c[1]} ({m['dynamic_coverage'][0]:.3f}/{m['dynamic_coverage'][1]:.3f}) | {m['dynamic_coverage_retention']:.4f} | {p['11']}/{p['01']}/{p['10']}/{p['00']} | {fmt_m(m['coverage_margin'])} {m['coverage_margin']['verdict']} | {'pass' if m['coverage_point_pass'] else 'FAIL'} | {m['raw_dynamic_coverage_difference']['mean']:+.4f} [{m['raw_dynamic_coverage_difference']['ci'][0]:+.4f}, {m['raw_dynamic_coverage_difference']['ci'][1]:+.4f}] |")
out.append("\n### Cross-backend vs the family torch/torch reference (direct boards)\n")
out.append("| config | board Jaccard mean/min | identical/84 | dyn-slot Jaccard | head top-6 overlap mean/min | max abs logit delta | full10 cov config-ref [CI] | flips ref-only/config-only | dyn6 cov config-ref [CI] | flips | full7 cov config-ref [CI] | flips ref-only/config-only |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
def cr(c, w):
    f = c[w]; m = f["config_minus_reference_dynamic_coverage"]
    return f"{m['mean']:+.4f} [{m['ci'][0]:+.4f}, {m['ci'][1]:+.4f}] | {f['flips']['reference_covered_config_not']}/{f['flips']['config_covered_reference_not']}"
for key in order:
    if key not in cross: continue
    c = cross[key]; j = c["direct"]["board_jaccard_vs_reference"]; hh = c["head_vs_reference_logits"]
    out.append(f"| {key} | {j['mean']:.4f}/{j['min']:.3f} | {j['identical']} | {c['direct']['dynamic_slot_jaccard_mean']:.4f} | {hh['top6_overlap_mean']:.4f}/{hh['top6_overlap_min']:.3f} | {hh['max_abs_logit_delta']:.3g} | {cr(c['direct'],'full10')} | {cr(c['direct'],'dyn6')} | {cr(c['direct'],'full7')} |")
out.append("\n### Seam boards vs the family torch/torch reference\n")
out.append("| config | board Jaccard mean/min | identical/84 | full10 cov config-ref [CI] | flips | dyn6 cov config-ref [CI] | flips | full7 cov config-ref [CI] | flips |")
out.append("|---|---|---|---|---|---|---|---|---|")
for key in order:
    if key not in cross: continue
    c = cross[key]; j = c["seam"]["board_jaccard_vs_reference"]
    out.append(f"| {key} | {j['mean']:.4f}/{j['min']:.3f} | {j['identical']} | {cr(c['seam'],'full10')} | {cr(c['seam'],'dyn6')} | {cr(c['seam'],'full7')} |")
out.append("\n### Flip rows (i05b, direct boards; held-out index in the 84-row order)\n")
for key in order:
    if not key.startswith("i05b") or key not in cross or key == "i05b/torch/torch": continue
    for w in ("full10", "dyn6", "full7"):
        fr = cross[key]["direct"][w].get("flip_rows", {})
        out.append(f"- {key} {w}: ref-only {fr.get('reference_covered_config_not')} config-only {fr.get('config_covered_reference_not')}")
out.append("\n### Latency join\n")
for k, v in d["latency_join"].items():
    out.append(f"- {k}: " + (f"{v['bench_row']} p50 {v['p50_ms']} p90 {v['p90_ms']} mean {v['mean_ms']} n={v['n']}x{v['repeats']} embed={v['embed_backend_used']} head={v['student_runtime']} archived Jaccard {v['board_jaccard_vs_torch_archived']} ({v['hardware']})" if v.get("source") else v["note"]))
out.append("\n### Continuity checks (I-04b)\n")
out.append("```json\n" + json.dumps(d["continuity_checks"]) + "\n```")
out.append(f"\nscoring embedder asserted: {d['scoring_embedder']}; versions {d['versions']}; wall {d['wall_seconds']} s")
open(out_path, "x").write("\n".join(out) + "\n")
print("\n".join(out))
