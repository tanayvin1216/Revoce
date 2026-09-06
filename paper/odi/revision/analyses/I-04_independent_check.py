"""I-04 checker: saved-artifact verification and torch selector replay; no OV JIT."""
import os, sys, json, hashlib
from pathlib import Path
os.environ.update(VOICELINK_PERCEPTION="0", VOICELINK_EMBED_BACKEND="torch", VOICELINK_STUDENT_BACKEND="torch", VOICELINK_EMBED_STRICT="1", HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
A=Path(__file__).resolve().parent
ROOT=A.parents[3]
sys.path.insert(0,str(ROOT/"mac"))
import numpy as np
import torch
torch.set_num_threads(1)
from evaluation.phase4 import infer, data, evaluate
from evaluation import embed
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report={}
scores=read(A/"I-04_scores_v1.json")
fresh=read(A/"I-04_scores_check_v1.json")
report["score_differences_excluding_wall_seconds"]=[k for k in scores if k!="wall_seconds" and scores[k]!=fresh[k]]
assert not report["score_differences_excluding_wall_seconds"]
gens={p.stem:read(p) for p in sorted((A/"I-04_boards_v1").glob("*.json"))}
assert len(gens)==12
hashes={}
for g in gens.values():
    assert len(g["rows"])==84 and g["asserted"]["embed_backend_used"]==g["embed"]
    assert not g["asserted"]["seam_fallback_reported"]
    for name,h in g["input_sha256"].items():
        path=A/name.removeprefix("analyses/") if name.startswith("analyses/") else ROOT/name
        hashes[name]=(h,sha(path))
report["input_hashes_checked"]=len(hashes)
report["input_hash_mismatches"]={n:v for n,v in hashes.items() if v[0]!=v[1]}
assert not report["input_hash_mismatches"]
report["fresh_torch"]={}
for fam in ("hist","i05"):
    g=gens[f"{fam}_torch_torch"]; f=read(A/f"I-04_{fam}_torch_check_v1.json")
    assert all({k:v for k,v in x.items() if k!="logits"}=={k:v for k,v in y.items() if k!="logits"} for x,y in zip(f["rows"],g["rows"]))
    err=float(np.max(np.abs(np.array([r["logits"] for r in f["rows"]])-np.array([r["logits"] for r in g["rows"]]))))
    assert err<1e-5
    report["fresh_torch"][fam]={"rows_except_logits_exact":84,"max_logit_delta":err,"metadata_differences":[k for k in g if k!="rows" and g[k]!=f[k]]}
report["recorded_reproduction"]={}
for p in (A/"I-04_boards_v1_repro").glob("*.json"):
    r=read(p); g=gens[p.stem]
    diff=[k for k in g if g[k]!=r[k]]
    assert diff==["wall_seconds"]
    report["recorded_reproduction"][p.name]=diff
report["comparisons"]={}
cal=[json.loads(l) for l in (ROOT/"mac/evaluation/phase4/v2_scenarios.jsonl").read_text().splitlines() if l.strip()]
calctx={(r["scene"].strip(),r["transcript"].strip()) for r in cal}
for fam in ("hist","i05"):
    ref=gens[f"{fam}_torch_torch"]["rows"]
    for head in ("torch","ov-fp32","ov-int8"):
        a=gens[f"{fam}_{head}_torch"]; b=gens[f"{fam}_{head}_openvino"]
        assert all(x[s]==y[s] for x,y in zip(a["rows"],b["rows"]) for s in ("direct_board","seam_board"))
        result={"embedder_ordered_boards_equal":84}
        for source in ("direct_board","seam_board"):
            js=[len(set(x[source])&set(y[source]))/len(set(x[source])|set(y[source])) for x,y in zip(a["rows"],ref)]
            result[source]={"ordered_equal":sum(x[source]==y[source] for x,y in zip(a["rows"],ref)),"set_equal":sum(j==1 for j in js),"jaccard_mean":float(np.mean(js)),"jaccard_min":min(js)}
        L=np.array([r["logits"] for r in a["rows"]]); R=np.array([r["logits"] for r in ref])
        overlaps=np.array([len(set(sorted(range(len(x)),key=lambda i:-x[i])[:6])&set(sorted(range(len(y)),key=lambda i:-y[i])[:6]))/6 for x,y in zip(L,R)])
        result["top6"]=[float(overlaps.mean()),float(overlaps.min())]
        result["max_logit_delta"]=float(abs(L-R).max())
        if fam=="hist" and head=="ov-int8":
            exposed=np.array([(r["scene"].strip(),r["transcript"].strip()) in calctx for r in ref])
            result["calibration_split"]={"n":[int(exposed.sum()),int((~exposed).sum())],"top6":[float(overlaps[exposed].mean()),float(overlaps[~exposed].mean())]}
        report["comparisons"][f"{fam}/{head}"]=result
# Rebuild direct boards using raw saved OV logits and current torch embeddings.
class SavedHead:
    def __init__(self,row): self.row=row
    def __call__(self,x):
        assert x.shape==(1,769)
        return np.asarray([self.row["logits"]],dtype=np.float32)
report["raw_logit_selector_replay"]={}
for fam in ("hist","i05"):
    vocab=read((ROOT/"mac/evaluation/phase4" if fam=="hist" else A/"I-05_split_first_v2")/"vocab.json")
    for head in ("ov-fp32","ov-int8"):
        g=gens[f"{fam}_{head}_torch"]
        rebuilt=[infer.student_board(SavedHead(r),vocab,r["scene"],r["transcript"],r["is_question_row"],10,6,g["core"]) for r in g["rows"]]
        n=sum(b==r["direct_board"] for b,r in zip(rebuilt,g["rows"]))
        assert n==84
        report["raw_logit_selector_replay"][f"{fam}/{head}"]=n
# Independent arithmetic on paired raw outcomes, including separate seed sensitivity.
report["coverage"]={}
for key in ("hist/ov-int8/openvino","i05/ov-int8/openvino"):
    report["coverage"][key]={}
    for window in ("full10","dyn6","full7"):
        w=scores["configurations"][key]["windows"]["seam"][window]
        pairs=np.array([(r["student_dynamic_coverage"],r["teacher_dynamic_coverage"]) for r in w["per_row_dynamic_coverage"] if r["student_dynamic_coverage"] is not None])
        delta=pairs[:,0]-0.9*pairs[:,1]
        def ci(seed):
            rng=np.random.default_rng(seed)
            return np.quantile(delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(axis=1),[.025,.975]).tolist()
        counts={str(s)+str(t):int(((pairs[:,0]==s)&(pairs[:,1]==t)).sum()) for s,t in ((1,1),(0,1),(1,0),(0,0))}
        assert counts==w["summary"]["paired_counts"]
        assert [round(v,4) for v in ci(0)]==w["summary"]["coverage_margin"]["ci"]
        report["coverage"][key][window]={"counts":counts,"margin":float(delta.mean()),"ci_seed0":ci(0),"ci_seed104":ci(104),"summary":w["summary"]}
e2e=read(ROOT/"mac/intel/artifacts/e2e_mac.json")
report["archived_latency"]={r["backend"]:{k:r.get(k) for k in ("p50_ms","p90_ms","n","repeats","embed_backend_used","samples_ms","boards")} for r in e2e["results"]}
report["scoring_embedder"]=embed.assert_backend("torch")
with (A/"I-04_independent_check_v1.json").open("x") as f: json.dump(report,f,indent=2)
print(json.dumps({k:v for k,v in report.items() if k!="coverage"},indent=2))
