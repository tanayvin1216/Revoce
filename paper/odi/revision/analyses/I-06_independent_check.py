"""I-06 raw-text replay and independent MMR/metric verification; no LLM calls."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path.cwd()))
import rank_vocabulary as rv
from ranker import _parse_json_array
from selection import build_query_vector
from evaluation.embed import embed,assert_backend
from evaluation.phase3 import selftest,taxonomy
A=Path('../paper/odi/revision/analyses')
def read(n): return json.loads((A/n).read_text())
def lines(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def compare(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: compare(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b): compare(x,y)
    elif isinstance(a,float): assert abs(a-b)<1e-6,(a,b)
    else: assert a==b,(a,b)
old=read('I-06_arms_scores.json'); new=read('I-06_reproduction_check.json')
for key in ['checks','aggregates','paired','diagnostics','historical_cached_reference','historical_board_lengths','per_row']: compare(old[key],new[key])
assert (A/'I-06_arms_scores_boards.jsonl').read_bytes()==(A/'I-06_reproduction_check_boards.jsonl').read_bytes()
paths={n:Path(p) for n,p in {'rank_vocabulary.py':'rank_vocabulary.py','selection.py':'selection.py','selftest.py':'evaluation/phase3/selftest.py','metrics3.py':'evaluation/phase3/metrics3.py','taxonomy.py':'evaluation/phase3/taxonomy.py','boards.jsonl':'evaluation/phase3/boards.jsonl','scenarios.jsonl':'evaluation/phase3/scenarios.jsonl','gen_phase1.py':'evaluation/phase1/gen_phase1.py','ranker.py':'ranker.py'}.items()}
for n,p in paths.items():
    if n+'_sha256' in old['inputs']: assert sha(p)==old['inputs'][n+'_sha256']
manifest=read('I-06_pools.jsonl.manifest.json')
for n,h in manifest['inputs'].items(): assert sha(paths[n])==h,n
assert sha(A/'I-06_pools.jsonl')==manifest['output_sha256']==old['inputs']['pools_sha256']
assert hashlib.sha256((A/'I-06_pools.jsonl').read_bytes().splitlines(keepends=True)[0]).hexdigest()==read('I-06_pools_smoke.manifest.json')['output_sha256']
pools=lines(A/'I-06_pools.jsonl'); scenarios={s['id']:s for s in lines(paths['scenarios.jsonl'])}
boards={(b['id'],b['strategy']):b for b in lines(A/'I-06_arms_scores_boards.jsonl')}
assert len(pools)==len(scenarios)==120 and len(boards)==480
assert_backend('torch')
flat=[(label,text) for label,texts in taxonomy.FUNCTION_EXEMPLARS_V3.items() for text in texts]
proto=embed([t for _,t in flat]); manual={w:{a:[] for a in old['arms']} for w in old['windows']}
class Replay:
    def __init__(self,row): self.row=row; self.calls=[]
    def rank(self,system,user,max_tokens=None):
        self.calls.append((hashlib.sha256(system.encode()).hexdigest()[:16],hashlib.sha256(user.encode()).hexdigest()[:16],max_tokens))
        return _parse_json_array(self.row['teacher']['raw_text'])
for p in pools:
    sc=scenarios[p['id']]; scene,transcript,ctx=selftest.build_context(sc)
    assert (ctx,scene,transcript)==(p['context'],p['scene'],p['transcript'])
    for k in ['acceptable_moves','gold_utterance','active_question','category','is_question']: assert p[k]==sc[k]
    topics=rv.topics_from_context(scene,transcript,ctx); assert topics==p['topics']
    replay=Replay(p); pool=rv.generate_pool(scene,transcript,ctx,ranker=replay,topics=topics)
    assert pool==p['pool']
    assert replay.calls==[(p['teacher']['system_sha'],p['teacher']['user_sha'],p['teacher']['max_tokens'])]
    assert p['teacher']['exception'] is None
    assert len(_parse_json_array(p['teacher']['raw_text']))==p['teacher']['n_parsed']>0
    canon=[[q.get('word',''),q.get('function',''),q.get('topic','')] for q in pool]
    assert hashlib.sha256(json.dumps(canon,ensure_ascii=False).encode()).hexdigest()==p['pool_hash']
    words=[q['word'] for q in pool if q['word'].lower() not in {c.lower() for c in rv.GUARANTEED_CORE}]
    vec=embed(words); query=build_query_vector(embed,scene,transcript,ctx); rel=np.clip(vec@query,0,1)
    rank=sorted(range(len(words)),key=lambda i:-float(rel[i]))[:6]
    assert [words[i] for i in rank]==boards[p['id'],'relevance']['board'][4:]
    sims=vec@vec.T; picked=[int(np.argmax(rel))]
    while len(picked)<6:
        candidates=[i for i in range(len(words)) if i not in picked]
        picked.append(max(candidates,key=lambda i:float(np.float32(0.6)*rel[i]-np.float32(0.4)*max(sims[i,j] for j in picked))))
    assert [words[i] for i in picked]==boards[p['id'],'mmr_plain']['board'][4:]
    for arm in old['arms']:
        b=boards[p['id'],arm]
        assert b['pool_hash']==p['pool_hash'] and len(b['board'])==len(set(b['board']))==10
        assert b['board'][:4]==rv.GUARANTEED_CORE and set(b['board'][4:])<=set(words)
        for win,c in old['windows'].items():
            top=b['board'][4:10] if c['dynamic_only'] else b['board'][:c['k']]
            v=embed(top); labels=[flat[i][0] for i in (v@proto.T).argmax(axis=1)]
            pairs=[(i,j) for i in range(len(top)) for j in range(i+1,len(top))]
            manual[win][arm].append({'df':len(set(labels)),'sfp':sum(labels[i]==labels[j] for i,j in pairs)/len(pairs),'nd':sum(float(v[i]@v[j])>=0.85 for i,j in pairs)/len(pairs)})
for win,arms in manual.items():
    for arm,rows in arms.items():
        for m,target in [('df','distinct_functions'),('sfp','same_function_pair'),('nd','near_dup')]: assert abs(float(np.mean([r[m] for r in rows]))-old['aggregates'][win][arm][target])<1e-12
thin=[{'id':r['id'],'boards':r['arms'],'full7_scores':{a:r['scores'][a]['full7'] for a in old['arms']}} for r in old['per_row'] if r['diag']['n_rest']==6]
sens={w:{a:float(np.mean([r['scores'][a][w]['distinct_functions'] for r in old['per_row'] if r['diag']['n_rest']>6])) for a in old['arms']} for w in old['windows']}
d=np.array([m['df']-r['df'] for m,r in zip(manual['full7']['mmr_plain'],manual['full7']['relevance'])],float)
rng=np.random.default_rng(0); ci=np.percentile(d[rng.integers(120,size=(10000,120))].mean(axis=1),[2.5,97.5])
assert np.allclose(ci,old['paired']['full7']['mmr_plain_minus_relevance']['distinct_functions']['ci95'],atol=1e-12)
result={'all_checks_passed':True,'pool_replays':120,'boards_identical':480,'manual_metric_windows':1440,'headline_df_delta':float(d.mean()),'headline_ci':ci.tolist(),'thin_pools':thin,'sensitivity_df_excluding_thin':sens,'teacher_mean_all_120':float(np.mean([p['teacher']['seconds'] for p in pools])),'hashes':{n:sha(A/n) for n in ['I-06_pools.jsonl','I-06_arms_scores.json','I-06_arms_scores_boards.jsonl']}}
with (A/'I-06_independent_check.json').open('x') as f: json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
