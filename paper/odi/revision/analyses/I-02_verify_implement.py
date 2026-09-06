"""Runnable I-02 verification: repeat identity and harness-free board re-score."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
MAC=Path(__file__).resolve().parents[4]/'mac'
sys.path.insert(0,str(MAC))
os.environ.update(VOICELINK_PERCEPTION='0',HF_HUB_OFFLINE='1',VOICELINK_EMBED_BACKEND='torch')
import numpy as np
from evaluation.embed import embed
from evaluation.phase3 import metrics3 as m
from evaluation.coverage_metrics import bootstrap_ci

def main():
 p=argparse.ArgumentParser();p.add_argument('first',type=Path);p.add_argument('repeat',type=Path);p.add_argument('output',type=Path);a=p.parse_args();start=time.perf_counter()
 data=json.loads((a.first/'numbers.json').read_text())
 assert data==json.loads((a.repeat/'numbers.json').read_text())
 manifest=json.loads((a.first/'manifest.json').read_text())
 assert hashlib.sha256((a.first/'numbers.json').read_bytes()).hexdigest()==manifest['artifact_sha256']['numbers.json']
 checked=0
 for label,r in data.items():
  for arm,v in r['arms'].items():
   for win,info in v['windows'].items():
    k=6 if win=='dyn6' else int(win[4:]);per=[]
    for b,moves in zip(v['boards'],r['acceptable_moves']):
     core={x.lower() for x in b[:4]}
     strip=lambda b:[w for w in b if w.lower() not in core]
     board=(strip(b) if win=='dyn6' else b)[:k];vec=embed(board)
     targets=[x for x in moves if not m.is_hedge(x)]
     sc=dict(df=m.distinct_functions_at(board,k,vec),near_dup=m.near_dup_rate(vec),sfp=m.same_function_pair_rate(board,k,vec),hedges=sum(m.is_hedge(x) for x in board),dynamic_coverage=int(m.coverage_of_intent(strip(board),targets,k)) if targets else None)
     per.append(sc)
    assert per==info['per_row'];checked+=len(per)
    for key in per[0]:
     arr=np.array([x[key] for x in per if x[key] is not None],dtype=float)
     ref=np.array([x[key] for x in r['arms']['learned']['windows'][win]['per_row'] if x[key] is not None],dtype=float)
     assert float(arr.mean())==info['means'][key]
     ci=bootstrap_ci(arr-ref,n=10000,seed=0)
     got=info['baseline_minus_learned'][key]
     assert got['margin']==round(float((arr-ref).mean()),4)
     assert got['ci']==[round(float(x),4) for x in ci]
    # Full@10 and dyn@6 are the identical dynamic coverage observations.
    assert [x['dynamic_coverage'] for x in v['windows']['full10']['per_row']]==[x['dynamic_coverage'] for x in v['windows']['dyn6']['per_row']]
 # Audit rounded @7 DF / SFP / coverage cells, including both frequency definitions.
 audit={'learned':(4.095,.1893,.789),'cosine':(4.667,.1497,.803),'frequency':(3.762,.2166,.620),'vocab_order':(3.857,.2069,.676),'full':(4.690,.1485,.803),'random':(4.476,.1621,.577)}
 for arm,expected in audit.items():
  v=data['historical']['arms'][arm]['windows']['full7']['means']
  assert (round(v['df'],3),round(v['sfp'],4),round(v['dynamic_coverage'],3))==expected
 with a.output.open('x') as f:json.dump(dict(repeat_identical=True,harness_free_row_windows_checked=checked,all_means_and_paired_intervals_match=True,historical_audit_cells_match=True,wall_seconds=time.perf_counter()-start),f,indent=2)
 print(a.output.read_text())
if __name__=='__main__':main()
