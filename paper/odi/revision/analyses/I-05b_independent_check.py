"""Checker-only raw-artifact audit. Run from mac; outputs exclusively created."""
import os, sys, json, hashlib
from pathlib import Path
from collections import Counter
os.environ.update(VOICELINK_PERCEPTION='0',HF_HUB_OFFLINE='1',VOICELINK_EMBED_BACKEND='torch',VOICELINK_STUDENT_BACKEND='torch')
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import torch
from evaluation.phase4 import data, infer
from evaluation.phase4.student import StudentRanker
from evaluation.phase3 import metrics3
A=Path('../paper/odi/revision/analyses'); B=A/'I-05b_split_first'; R=A/'I-05b_rerun_check'
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=read(B/'manifest.json')
assert all(sha(B/n)==h for n,h in manifest['artifact_sha256'].items())
drift=[n for n,h in manifest['input_sha256'].items() if sha(Path('..')/n)!=h]
rows=read(B/'rows.json'); sp=read(B/'splits.json'); prep=read(B/'preprocessing.json'); vocab=read(B/'vocab.json'); canon=read(B/'canonical.json')
assert sp==read(Path('evaluation/phase4/splits.json'))
assert sorted(sp['train']+sp['val']+sp['test'])==list(range(559))
freq={s:Counter(w.strip().lower() for i in sp[s] for w in rows[i]['board'] if w.strip().lower() not in data.CORE_SET) for s in ('train','val','test')}
ranked=sorted((w for w,n in freq['train'].items() if n>=2),key=lambda w:(-freq['train'][w],w))
assert ranked[:116]==prep['raw_vocab']
v,c=data.dedup_vocab(ranked[:116],.8); assert v==vocab and c==canon
assert sorted(freq['train'],key=lambda w:(-freq['train'][w],w))[:116]==ranked[:116]
Y=np.asarray([[float(w in {canon.get(x.strip().lower(),x.strip().lower()) for x in rows[i]['board']}) for w in vocab] for i in sp['train']],dtype=np.float32)
np.testing.assert_array_equal(Y.sum(0),prep['positive_counts'])
np.testing.assert_array_equal(np.clip((391-Y.sum(0))/Y.sum(0),1,20),prep['pos_weight'])
np.testing.assert_array_equal(np.log(Y.mean(0)),prep['log_priors'])
boards=read(B/'boards.json'); old=read(A/'I-05_split_first_v2/boards.json'); hist=[x for x in read(A/'I-03_numbers_implement_v2.json')['per_row_table2'] if x['window']=='full10']
assert [b['row_index'] for b in boards]==sp['test']==[b['row_index'] for b in old]
assert [h['index'] for h in hist]==list(range(84))
for b,o,h in zip(boards,old,hist):
 assert b['context']==rows[b['row_index']]==o['context']
 assert b['acceptable_moves']==o['acceptable_moves']==h['moves']
 assert [w.lower() for w in b['teacher_board']]==[w.lower() for w in h['teacher_board']]
ck=torch.load(B/'student.pt',map_location='cpu',weights_only=False)
model=StudentRanker(ck['context_dim'],ck['hidden_dim'],ck['vocab_size'],ck['dropout']);model.load_state_dict(ck['state_dict']);model.eval()
for b in boards:
 r=b['context']; assert infer.student_board(model,vocab,r['scene'],r['transcript'],r['is_question'],10,6,ck['guaranteed_core'])==b['student_board']
# Own bootstrap, no project bootstrap helper.
def ci(x):
 x=np.asarray(x,dtype=float); rng=np.random.default_rng(0)
 return np.quantile(x[rng.integers(len(x),size=(10000,len(x)))].mean(1),[.025,.975]).tolist()
hv=set(read(Path('evaluation/phase4/vocab.json'))); ho=hv-set(vocab)
trace=read(A/'I-05b_leak_entries_v1.json'); comparison=read(A/'I-05b_compare_v1.json')
assert len(ho)==28
assert trace['historical_only_entries']=={w:{s:freq[s][w] for s in freq} for w in sorted(ho)}
assert sum(freq['train'][w]+freq['val'][w]<3 for w in ho)==19
out={}
for label,k,dyn in [('full10',10,False),('dyn6',6,True),('full7',7,False)]:
 flags={n:[] for n in ('i05b_95','i05_45','historical_95','teacher')}; dfs={n:[] for n in flags}; traces=[]; withonly=0
 for b,o,h in zip(boards,old,hist):
  moves=[m for m in b['acceptable_moves'] if not metrics3.is_hedge(m)]
  windows={}
  for n,board in [('i05b_95',b['student_board']),('i05_45',o['student_board']),('historical_95',h['student_board']),('teacher',b['teacher_board'])]:
   window=([w for w in board if w.lower() not in data.CORE_SET] if dyn else board)[:k];windows[n]=window
   dfs[n].append(metrics3.distinct_functions_at(window,k))
   if moves: flags[n].append(int(metrics3.coverage_of_intent([w for w in window if w.lower() not in data.CORE_SET],moves,k)))
  hw=windows['historical_95']; withonly+=any(w.lower() in ho for w in hw)
  if moves and flags['historical_95'][-1] and not flags['i05b_95'][-1]:
   carriers=[w for w in hw if w.lower() not in data.CORE_SET and metrics3.coverage_of_intent([w],moves,k)]
   shared=[w for w in hw if w.lower() not in data.CORE_SET and w.lower() not in ho]
   solely=not metrics3.coverage_of_intent(shared,moves,k)
   traces.append(dict(row_index=b['row_index'],carriers=carriers,solely_hist_only=solely))
 assert withonly==trace['windows'][label]['historical_boards_with_hist_only_entry_in_window']
 assert [(t['row_index'],t['carriers']) for t in traces]==[(t['row_index'],t['carriers']) for t in trace['windows'][label]['flip_rows']]
 assert sum(t['solely_hist_only'] for t in traces)==trace['windows'][label]['flip_rows_where_only_hist_only_entries_carry']
 diffs={}
 for a,z in [('i05b_95','i05_45'),('i05b_95','historical_95'),('i05_45','historical_95')]:
  delta=np.array(flags[a])-np.array(flags[z]);dd=np.array(dfs[a])-np.array(dfs[z]);key=a+'_minus_'+z
  got=dict(coverage_diff=float(delta.mean()),coverage_ci=ci(delta),df_diff=float(dd.mean()),df_ci=ci(dd))
  assert got==comparison['student_vs_student'][label][key];diffs[key]=got
 out[label]=dict(hits={n:sum(f) for n,f in flags.items()},differences=diffs,trace=traces)
rep={}
for name,p in [('historical_95',Path('evaluation/phase4')),('i05_45',A/'I-05_split_first_v2'),('i05b',B)]:
 vs=set(read(p/'vocab.json')); cm=read(p/'canonical.json'); rep[name]={}
 for s in freq:
  hits=sum(cm.get(w,w) in vs for i in sp[s] for w in [x.strip().lower() for x in rows[i]['board']] if w not in data.CORE_SET)
  assert hits==read(B/'metrics.json')['representability'][name][s]['canonical_hits']; rep[name][s]=hits
rerun_equal={n:sha(B/n)==sha(R/n) for n in ['boards.json','canonical.json','config.json','preprocessing.json','rows.json','splits.json','vocab.json']}
assert all(rerun_equal.values())
m=read(B/'metrics.json'); rm=read(R/'metrics.json'); assert m['metrics']==rm['metrics']
rc=torch.load(R/'student.pt',map_location='cpu',weights_only=False)
result=dict(input_hash_drift=drift,artifact_hashes_verified=True,rerun_equal=rerun_equal,metric_windows_equal=True,best_epoch=rm['best_epoch'],best_val_loss=rm['best_val_loss'],max_weight_difference=max(float((ck['state_dict'][k]-rc['state_dict'][k]).abs().max()) for k in ck['state_dict']),regenerated_boards=84,freq_ge3=sum(n>=3 for n in freq['train'].values()),freq_eq2=sum(n==2 for n in freq['train'].values()),last_in=ranked[115],first_out=ranked[116],representability=rep,windows=out)
with (A/'I-05b_independent_check.json').open('x') as f: json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
