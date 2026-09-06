"""I-05b: I-05 train-only preprocessing with the frequency floor / cap chosen so the
training-only vocabulary has (as close as possible to) the historical 95 entries.

Derived from I-05_split_first_implement.py; the split, fit(), poisoning check,
training loop, checkpoint and evaluation windows are unchanged. Only the vocab
floor/cap are parameters, and representability is reported for three vocabularies
(historical, I-05, this run) on the same held-out rows.
Run from mac/. The output directory must not exist. No historical outputs written.
"""
import argparse, contextlib, hashlib, importlib.metadata, io, json, os, sys, time
from pathlib import Path
MAC=Path(__file__).resolve().parents[4]/'mac'
sys.path.insert(0,str(MAC))
os.environ.update(VOICELINK_PERCEPTION='0',HF_HUB_OFFLINE='1',VOICELINK_EMBED_BACKEND='torch',VOICELINK_STUDENT_BACKEND='torch',VOICELINK_DEVICE='cpu')
import numpy as np
import torch
from evaluation.phase4 import train as tr, data as d, evaluate as e, infer
from evaluation.phase4.student import StudentRanker
from evaluation.embed import assert_backend
from rank_vocabulary import GUARANTEED_CORE
ANALYSES=Path(__file__).resolve().parent
I05_BUNDLE=ANALYSES/'I-05_split_first_v2'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    with p.open('x') as f: json.dump(x,f,indent=2)
def fit(training_rows,min_freq,max_size,dedup_tau):
    # This function has no access to validation/test rows or their labels.
    raw,_=d.build_vocab(training_rows,min_freq,max_size)
    vocab,canonical=d.dedup_vocab(raw,dedup_tau)
    idx={w:i for i,w in enumerate(vocab)}
    Y=np.stack([d.multihot(r['board'],idx,canonical) for r in training_rows])
    return raw,vocab,canonical,Y,tr._pos_weight(Y),tr._log_priors(Y)
def representability(rows,indices,vocab,canonical,k_dyn=6):
    """Teacher non-core entries (lowercase/strip) representable exactly or via the
    canonical map. Reported over all non-core slots, over the first k_dyn non-core
    entries (the dyn@6 teacher window), and per row (>=1 representable)."""
    vs=set(vocab);total=exact=canon=zero=0;dyn_total=dyn_hit=0;rows_any=0
    for i in indices:
        words=[w.strip().lower() for w in rows[i]['board'] if w.strip().lower() not in d.CORE_SET]
        total+=len(words);exact+=sum(w in vs for w in words)
        hits=[canonical.get(w,w) in vs for w in words];count=sum(hits);canon+=count;zero+=count==0;rows_any+=count>0
        dyn_total+=len(words[:k_dyn]);dyn_hit+=sum(hits[:k_dyn])
    return dict(rows=len(indices),slots=total,exact_hits=exact,canonical_hits=canon,canonical_fraction=canon/total,
                dyn6_slots=dyn_total,dyn6_canonical_hits=dyn_hit,dyn6_canonical_fraction=dyn_hit/dyn_total,
                rows_with_any_representable=rows_any,rows_with_any_fraction=rows_any/len(indices),zero_hit_rows=zero)
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--min-freq',type=int,required=True);p.add_argument('--max-size',type=int,required=True)
    p.add_argument('--target-size',type=int,default=95)
    a=p.parse_args();out=a.out.resolve();out.mkdir(exist_ok=False);start=time.perf_counter()
    sources=[tr.HERE/x for x in ('config.yaml','student.pt','vocab.json','canonical.json','splits.json','teacher_data.jsonl','v2_scenarios.jsonl','train.py','data.py','student.py','infer.py','answers.py','evaluate.py')]
    sources += [MAC/x for x in ('selection.py','rank_vocabulary.py','evaluation/embed.py','evaluation/hedges.py','evaluation/coverage_metrics.py','evaluation/phase3/metrics3.py','evaluation/phase3/taxonomy.py','evaluation/phase3/boards.jsonl')]
    sources += [e.scengen.SCENARIOS_PATH,Path(__file__).resolve(),I05_BUNDLE/'vocab.json',I05_BUNDLE/'canonical.json',I05_BUNDLE/'manifest.json']
    inputs={str(x.relative_to(MAC.parent)):sha(x) for x in sources}
    cfg,base_hash=tr.load_config();tr.seed_everything(cfg['seed']);torch.set_num_threads(1)
    rows=tr.gather_rows();splits=d.split_indices(len(rows),cfg['train']['val_frac'],cfg['train']['test_frac'],cfg['seed'])
    splits['fingerprint']=d.rows_fingerprint(rows)
    assert splits==json.loads(d.SPLITS_PATH.read_text()),'partition drift'
    assert splits==json.loads((I05_BUNDLE/'splits.json').read_text()),'partition differs from I-05 bundle'
    assert [len(splits[k]) for k in ('train','val','test')]==[391,84,84]
    trainrows=[rows[i] for i in splits['train']];tau=cfg['vocab']['dedup_tau']
    # Recorded vocabulary-size sweep (training rows only) that motivates the chosen setting.
    sweep=[]
    for mf,ms in [(3,cfg['vocab']['max_size']),(2,cfg['vocab']['max_size']),(1,cfg['vocab']['max_size']),(1,95),(1,110),(1,115),(1,116),(1,117),(1,118),(2,116),(2,118)]:
        r0,_=d.build_vocab(trainrows,mf,ms)
        with contextlib.redirect_stdout(io.StringIO()): v0,c0=d.dedup_vocab(r0,tau)
        sweep.append(dict(min_freq=mf,max_size=ms,raw=len(r0),dedup=len(v0),merges=len(c0)))
    raw,vocab,canonical,ytrain,pw,logp=fit(trainrows,a.min_freq,a.max_size,tau)
    print(f'chosen min_freq={a.min_freq} max_size={a.max_size}: raw={len(raw)} dedup={len(vocab)} merges={len(canonical)} (target {a.target_size})',flush=True)
    # Regression test: poison every held-out label; fitted quantities must be identical.
    trainset=set(splits['train'])
    poisoned=[r if i in trainset else dict(r,board=['heldout sentinel']*10) for i,r in enumerate(rows)]
    with contextlib.redirect_stdout(io.StringIO()): again=fit([poisoned[i] for i in splits['train']],a.min_freq,a.max_size,tau)
    assert (raw,vocab,canonical)==again[:3]
    for x,y in zip((ytrain,pw.numpy(),logp.numpy()),(again[3],again[4].numpy(),again[5].numpy())): np.testing.assert_array_equal(x,y)
    assert set(canonical)<=set(raw) and set(canonical.values())<=set(vocab)
    X,Y=d.make_xy(rows,{w:i for i,w in enumerate(vocab)},canonical)
    np.testing.assert_array_equal(Y[splits['train']],ytrain)
    np.testing.assert_array_equal(pw.numpy(),np.clip((len(ytrain)-ytrain.sum(0))/ytrain.sum(0),1,20).astype('float32'))
    np.testing.assert_array_equal(logp.numpy(),np.log(ytrain.mean(0)).astype('float32'))
    # Explicit assumption: CPU/no IPEX; every hyperparameter except the vocab floor/cap unchanged from I-05.
    effective=dict(base_config=cfg,vocab_override=dict(min_freq=a.min_freq,max_size=a.max_size,dedup_tau=tau),preprocessing='train-only-v1-matched-vocab',device='cpu',torch_threads=1,ipex=False,loss=dict(tau=tr.TAU,listwise_coefficient=0.3))
    config_hash=hashlib.sha256(json.dumps(effective,sort_keys=True).encode()).hexdigest()[:12]
    for name,value in [('config',effective),('splits',splits),('vocab',vocab),('canonical',canonical),('rows',rows)]: write(out/(name+'.json'),value)
    write(out/'preprocessing.json',dict(fit_indices=splits['train'],vocab_sweep=sweep,raw_vocab=raw,positive_counts=ytrain.sum(0).tolist(),pos_weight=pw.tolist(),log_priors=logp.tolist(),heldout_label_poisoning_invariant=True))
    Xt,Yt=torch.from_numpy(X),torch.from_numpy(Y)
    tr.seed_everything(cfg['seed'])  # same RNG state at model init as I-05 (which seeded once before gather_rows; the sweep above consumes no torch RNG, but re-seed to be explicit)
    model=StudentRanker(cfg['model']['context_dim'],cfg['model']['hidden_dim'],len(vocab),cfg['model']['dropout'])
    opt=torch.optim.Adam(model.parameters(),lr=cfg['train']['lr'],weight_decay=cfg['train']['weight_decay'])
    loss=torch.nn.BCEWithLogitsLoss(pos_weight=pw);rng=np.random.default_rng(cfg['seed'])
    best=float('inf');state=None;stale=0;history=[];ts=time.perf_counter()
    for epoch in range(1,cfg['train']['epochs']+1):
        model.train()
        for batch in tr._batches(len(trainrows),cfg['train']['batch_size'],rng):
            ids=[splits['train'][i] for i in batch];opt.zero_grad()
            tr._distil_loss(model(Xt[ids]),Yt[ids],pw,logp,loss).backward();opt.step()
        model.eval()
        with torch.no_grad(): val=tr._distil_loss(model(Xt[splits['val']]),Yt[splits['val']],pw,logp,loss).item()
        history.append(dict(epoch=epoch,val_loss=val))
        if val<best:
            best=val;best_epoch=epoch;stale=0;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: stale+=1
        if stale>=cfg['train']['patience']: break
    train_wall=time.perf_counter()-ts;model.load_state_dict(state);model.eval()
    ck=dict(state_dict=state,context_dim=cfg['model']['context_dim'],hidden_dim=cfg['model']['hidden_dim'],vocab_size=len(vocab),vocab_sha=d.vocab_sha(vocab),dropout=cfg['model']['dropout'],config_hash=config_hash,n_rows=len(rows),guaranteed_core=GUARANTEED_CORE,k_dyn=6,split_fingerprint=splits['fingerprint'],preprocessing='train-only-v1-matched-vocab',vocab_min_freq=a.min_freq,vocab_max_size=a.max_size)
    with (out/'student.pt').open('xb') as f: torch.save(ck,f)
    saved=torch.load(out/'student.pt',map_location='cpu',weights_only=False)
    assert saved['vocab_sha']==d.vocab_sha(json.loads((out/'vocab.json').read_text()))
    model.load_state_dict(saved['state_dict']);lookup=e._acceptable_moves_lookup();boards=[]
    for i in splits['test']:
        r=rows[i];sb=infer.student_board(model,vocab,r['scene'],r['transcript'],r['is_question'],10,6,GUARANTEED_CORE)
        assert len(sb)==10 and sb[:4]==GUARANTEED_CORE
        boards.append(dict(row_index=i,context=r,student_board=sb,teacher_board=r['board'],acceptable_moves=lookup.get((r['scene'].strip(),r['transcript'].strip()),[])))
    boardmap={(b['context']['scene'],b['context']['transcript']):b['student_board'] for b in boards}
    metrics={name:e.held_out_eval(model,vocab,10,6,GUARANTEED_CORE,board_fn=lambda s,t,q:boardmap[s,t],splits_path=out/'splits.json',metric_k=k,dynamic_only=dyn) for name,k,dyn in [('full10',10,False),('dyn6',6,True),('full7',7,False)]}
    # Representability of the teacher's dynamic choices under three vocabularies, same rows.
    hist_vocab=json.loads((tr.HERE/'vocab.json').read_text());hist_canon=d.load_canonical(tr.HERE/'canonical.json')
    i05_vocab=json.loads((I05_BUNDLE/'vocab.json').read_text());i05_canon=d.load_canonical(I05_BUNDLE/'canonical.json')
    rep={}
    for label,v,c in [('historical_95',hist_vocab,hist_canon),('i05_45',i05_vocab,i05_canon),('i05b',vocab,canonical)]:
        rep[label]=dict(vocab_size=len(v),canonical_merges=len(c),**{split:representability(rows,splits[split],v,c) for split in ('train','val','test')})
    overlap=dict(i05b_and_historical=len(set(vocab)&set(hist_vocab)),i05b_and_i05=len(set(vocab)&set(i05_vocab)),i05_subset_of_i05b=set(i05_vocab)<=set(vocab),
                 i05b_only=sorted(set(vocab)-set(hist_vocab)),historical_only=sorted(set(hist_vocab)-set(vocab)))
    assert assert_backend('torch')=='torch'
    assert all(sha(MAC.parent/p)==h for p,h in inputs.items()),'inputs changed during run'
    write(out/'boards.json',boards)
    result=dict(issue='I-05b',vocab_setting=dict(min_freq=a.min_freq,max_size=a.max_size,dedup_tau=tau,target_size=a.target_size),config_hash=config_hash,base_config_hash=base_hash,split_fingerprint=splits['fingerprint'],vocab_sha=d.vocab_sha(vocab),raw_vocab_size=len(raw),vocab_size=len(vocab),canonical_merges=len(canonical),vocab_sweep=sweep,vocab_overlap=overlap,best_epoch=best_epoch,best_val_loss=best,epochs_run=epoch,training_wall_seconds=train_wall,history=history,metrics=metrics,representability=rep,wall_seconds=time.perf_counter()-start)
    write(out/'metrics.json',result)
    write(out/'manifest.json',dict(issue='I-05b',config_hash=config_hash,split_fingerprint=splits['fingerprint'],vocab_sha=d.vocab_sha(vocab),vocab_setting=result['vocab_setting'],seed=cfg['seed'],bootstrap=dict(seed=0,resamples=10000),head_backend='torch',embedding_backend='torch',training_device='cpu',versions={x:importlib.metadata.version(x) for x in ('torch','numpy','sentence-transformers','transformers')},input_sha256=inputs,artifact_sha256={x.name:sha(x) for x in sorted(out.iterdir()) if x.is_file()},exported_artifacts=[],references=dict(i05_bundle=str(I05_BUNDLE.relative_to(MAC.parent))),downstream='Torch-only matched-vocabulary control for I-05; no IR/quantisation claimed. Do not combine with historical IR.'))
    print(json.dumps({k:v for k,v in result.items() if k!='history'},indent=2))
if __name__=='__main__': main()
