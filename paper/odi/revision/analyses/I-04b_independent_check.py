import os,sys,json,hashlib,re
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent; R=A.parents[3]; sys.path.insert(0,str(R/'mac'))
os.environ.update(HF_HUB_OFFLINE='1',VOICELINK_PERCEPTION='0',VOICELINK_EMBED_BACKEND='torch',VOICELINK_DEVICE='cpu',VOICELINK_EMBED_STRICT='1')
from evaluation.embed import embed,assert_backend
from evaluation.metrics import content_tokens
from evaluation.hedges import is_hedge
from evaluation.phase4 import evaluate,infer
J=lambda p:json.loads(p.read_text()); sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
gs={p.stem:J(p) for p in (A/'I-04b_boards_v1').glob('*.json')}; ref=gs['i05b_torch_torch']; scores=J(A/'I-04b_scores_v1.json'); fresh=J(A/'I-04b_scores_check.json')
assert {k:v for k,v in scores.items() if k!='wall_seconds'}=={k:v for k,v in fresh.items() if k!='wall_seconds'}
new=J(A/'I-04b_torch_check.json')['rows']
assert all({k:v for k,v in a.items() if k!='logits'}=={k:v for k,v in b.items() if k!='logits'} for a,b in zip(new,ref['rows']))
assert np.max(np.abs(np.array([r['logits'] for r in new])-np.array([r['logits'] for r in ref['rows']])))<1e-5
held=evaluate._held_out_rows(); lookup=evaluate._acceptable_moves_lookup(); core={x.lower() for x in ref['core']}; out={'configurations':{},'hash_drift':{},'latency':[]}
def hit(board,moves):
 for m in moves:
  mt=content_tokens(m); ml=m.lower().strip()
  for b in board:
   bl=b.lower().strip()
   if mt and (re.search(r'(?<!\w)'+re.escape(bl)+r'(?!\w)',ml) or re.search(r'(?<!\w)'+re.escape(ml)+r'(?!\w)',bl) or content_tokens(bl)&mt):return 1
 return int(bool(board and moves) and (embed(board)@embed(moves).T>=.50).any())
for key,g in gs.items():
 for n,h in g['input_sha256'].items():
  p=A/n.removeprefix('analyses/') if n.startswith('analyses/') else R/n
  if sha(p)!=h:out['hash_drift'][n]=[h,sha(p)]
 for i,(r,h) in enumerate(zip(g['rows'],held)):
  assert r['index']==i and (r['scene'],r['transcript'],r['teacher_board'])==(h['scene'],h['transcript'],h['board'])
  assert r['moves']==lookup.get((h['scene'].strip(),h['transcript'].strip()),[])
  assert all(r[f]==ref['rows'][i][f] for f in ('scene','transcript','teacher_board','moves','is_question_row','is_question_seam'))
 c={}; score=scores['configurations'][f"i05b/{g['head']}/{g['embed']}"]
 for src in ('seam','direct'):
  js=[len(set(r[src+'_board'])&set(t[src+'_board']))/len(set(r[src+'_board'])|set(t[src+'_board'])) for r,t in zip(g['rows'],ref['rows'])]; c[src]={'jaccard':float(np.mean(js)),'identical_sets':sum(x==1 for x in js)}
  for win,k,dyn in [('full10',10,False),('dyn6',6,True),('full7',7,False)]:
   pairs=[]; flags=[]
   for r in g['rows']:
    moves=[m for m in r['moves'] if not is_hedge(m)]
    if not moves:flags.append({'index':r['index'],'student_dynamic_coverage':None,'teacher_dynamic_coverage':None});continue
    bds=[]
    for field in (src+'_board','teacher_board'):
     b=r[field]; b=[w for w in b if w.lower() not in core][:k] if dyn else [w for w in b[:k] if w.lower() not in core]; bds.append(hit(b,moves))
    s,t=bds;pairs.append((s,t));flags.append({'index':r['index'],'student_dynamic_coverage':s,'teacher_dynamic_coverage':t})
   stored=score['windows'][src][win];assert flags==stored['per_row_dynamic_coverage'];ar=np.array(pairs);delta=ar[:,0]-.9*ar[:,1];rng=np.random.default_rng(0)
   means=np.array([delta[rng.choice(len(delta),size=len(delta),replace=True)].mean() for _ in range(10000)]);ci=np.quantile(means,[.025,.975]);counts={str(s)+str(t):pairs.count((s,t)) for s,t in [(1,1),(0,1),(1,0),(0,0)]}
   assert counts==stored['summary']['paired_counts'];assert np.round(ci,4).tolist()==stored['summary']['coverage_margin']['ci']
   c[src][win]={'counts':ar.sum(0).tolist()+[len(pairs)],'pairs':counts,'margin':float(delta.mean()),'ci':ci.tolist(),'eligible_indices':[p['index'] for p in flags if p['student_dynamic_coverage'] is not None]}
 vocab=J(A/'I-05b_split_first/vocab.json')
 for r in g['rows']:
  class Saved:
   def __call__(self,x):return np.array([r['logits']],dtype=np.float32)
  assert infer.student_board(Saved(),vocab,r['scene'],r['transcript'],r['is_question_row'],10,6,g['core'])==r['direct_board']
 c['direct_replays']=84;out['configurations'][key]=c
for p in (A/'I-04b_boards_v1_repro').glob('*.json'):
 assert {k:v for k,v in J(p).items() if k!='wall_seconds'}=={k:v for k,v in gs[p.stem].items() if k!='wall_seconds'}
g=gs['i05b_ov-int8_openvino'];pooled=[]
for p in sorted((A/'I-10_bench_v1').glob('i05b_*.json')):
 d=J(p)
 for s in d['sessions']:
  if s['backend']!='ov-full':continue
  assert all(g['input_sha256'][n]==h for n,h in s['provenance']['inputs_sha256'].items())
  assert all(g['input_sha256']['mac/intel/artifacts/minilm-ov/'+n]==h for n,h in d['minilm_bundle_sha256'].items())
  common=set(g['input_sha256'])&set(d['code_sha256']);assert all(g['input_sha256'][n]==d['code_sha256'][n] for n in common)
  assert s['label_checks']['embed_backend_used']=='openvino' and s['label_checks']['student_runtime']=='OpenVinoStudent' and not s['label_checks']['seam_fallback_reported']
  assert s['boards']==[r['seam_board'] for r in g['rows']]
  assert [x['ctx_sha8'] for x in d['scenario_ids']]==[hashlib.sha256((r['scene']+'\x1f'+r['transcript']).encode()).hexdigest()[:8] for r in g['rows']]
  arr=np.array(s['samples_ms']);assert arr.shape==(s['protocol']['repeats'],84);quant=np.percentile(arr,[25,50,75,90]);assert all(abs(v-s[k])<1e-10 for v,k in zip(quant,['p25_ms','p50_ms','p75_ms','p90_ms']))
  out['latency'].append({'file':p.name,'sha256':sha(p),'tag':s['tag'],'samples':int(arr.size),'quantiles':quant.tolist(),'boards_equal':84,'common_source_hashes':len(common)})
  if s['tag']!='histproto':pooled.extend(arr.ravel().tolist())
out['pooled_latency']={'n':len(pooled),'quantiles':np.percentile(pooled,[25,50,75,90]).tolist()};assert_backend('torch')
with (A/'I-04b_independent_check.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps({k:v for k,v in out.items() if k!='configurations'},indent=2))
