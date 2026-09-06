"""I-05b independent verification and three-model comparison.

Run from mac/: script I05B_BUNDLE I05_BUNDLE I03_JSON OUTPUT.json
- verifies every I-05b artifact hash in its manifest;
- rebuilds per-row distinct-functions and non-hedge dynamic coverage for the
  historical (I-03 per-row boards), I-05 (45-entry) and I-05b (95-entry) students
  with metrics3 directly, never via evaluate.held_out_eval, and asserts the
  rebuilt paired counts / DF means / margins equal each source's reported values;
- paired row bootstrap (10,000, seed 0) of STUDENT-vs-STUDENT differences in
  dynamic coverage and DF at full10 / dyn6 / full7, plus per-row flip counts.
Never rewrites any bundle artifact; the output path is opened exclusively.
"""
import hashlib,json,os,sys
from pathlib import Path
import numpy as np
MAC=Path(__file__).resolve().parents[4]/'mac';sys.path.insert(0,str(MAC))
os.environ.update(VOICELINK_PERCEPTION='0',HF_HUB_OFFLINE='1',VOICELINK_EMBED_BACKEND='torch')
from evaluation.phase3 import metrics3
from evaluation.coverage_metrics import bootstrap_ci
b05b,b05,i03,output=(Path(x) for x in sys.argv[1:5])
CORE={'yes','no','wait','i need help'}
WINDOWS=[('full10',10,False),('dyn6',6,True),('full7',7,False)]

def verify_bundle(bundle):
    m=json.loads((bundle/'manifest.json').read_text())
    for name,h in m['artifact_sha256'].items():
        assert hashlib.sha256((bundle/name).read_bytes()).hexdigest()==h,(bundle,name)
    return m
m05b=verify_bundle(b05b);m05=verify_bundle(b05)
assert m05b['split_fingerprint']==m05['split_fingerprint']=='ab51f7fba4c9c39e'

def score(board,moves,k,dyn):
    window=([w for w in board if w.lower() not in CORE] if dyn else board)[:k]
    df=metrics3.distinct_functions_at(window,k)
    dynamic=[w for w in window if w.lower() not in CORE]
    honest=[m for m in moves if not metrics3.is_hedge(m)]
    cov=int(metrics3.coverage_of_intent(dynamic,honest,k)) if honest else None
    return df,cov

# --- rows: I-05b and I-05 boards.json share row order (splits test order); I-03 per-row is in the same order.
rows05b=json.loads((b05b/'boards.json').read_text());rows05=json.loads((b05/'boards.json').read_text())
assert [r['row_index'] for r in rows05b]==[r['row_index'] for r in rows05]
assert all(a['teacher_board']==b['teacher_board'] and a['acceptable_moves']==b['acceptable_moves'] for a,b in zip(rows05b,rows05))
i03d=json.loads(i03.read_text());per03={w:[r for r in i03d['per_row_table2'] if r['window']==w] for w,_,_ in WINDOWS}
for w,_,_ in WINDOWS:
    assert len(per03[w])==84
    for r03,r in zip(per03[w],rows05b):
        assert [x.lower() for x in r03['teacher_board']]==[x.lower() for x in r['teacher_board']],('row order mismatch',w,r['row_index'])
        assert r03['moves']==r['acceptable_moves']
hist_boards={w:[r['student_board'] for r in per03[w]] for w,_,_ in WINDOWS}
assert all(hist_boards['full10'][i]==hist_boards[w][i] for w in ('dyn6','full7') for i in range(84)),'historical boards differ by window'
models={'historical_95':hist_boards['full10'],'i05_45':[r['student_board'] for r in rows05],'i05b_95':[r['student_board'] for r in rows05b]}
teacher=[r['teacher_board'] for r in rows05b];moves=[r['acceptable_moves'] for r in rows05b]
reported={'historical_95':{w:i03d['table2'][w]['harness'] for w,_,_ in WINDOWS},
          'i05_45':json.loads((b05/'metrics.json').read_text())['metrics'],
          'i05b_95':json.loads((b05b/'metrics.json').read_text())['metrics']}

per={};summary={};flips={};diffs={}
for w,k,dyn in WINDOWS:
    tdf=[];tcov=[]
    for tb,mv in zip(teacher,moves):
        df,cov=score(tb,mv,k,dyn);tdf.append(df);tcov.append(cov)
    per[w]={'teacher':dict(df=tdf,cov=tcov)};summary[w]={}
    for name,boards in models.items():
        sdf=[];scov=[]
        for sb,mv in zip(boards,moves):
            df,cov=score(sb,mv,k,dyn);sdf.append(df);scov.append(cov)
        per[w][name]=dict(df=sdf,cov=scov)
        idx=[i for i,c in enumerate(tcov) if c is not None]
        ss=np.array([scov[i] for i in idx],float);tt=np.array([tcov[i] for i in idx],float)
        pairs={f'{int(s)}{int(t)}':0 for s in (1,0) for t in (1,0)}
        for s,t in zip(ss,tt): pairs[f'{int(s)}{int(t)}']+=1
        margin=float(np.mean(ss-.9*tt));lo,hi=bootstrap_ci((ss-.9*tt).tolist())
        dfm=float(np.mean(np.array(sdf)-.9*np.array(tdf)));dlo,dhi=bootstrap_ci((np.array(sdf)-.9*np.array(tdf)).tolist())
        rep=reported[name][w]
        assert pairs==rep['coverage_of_intent']['paired_counts_student_teacher'],(name,w,pairs)
        assert round(float(np.mean(sdf)),3)==rep['diversity_retention']['student_distinct_functions'],(name,w)
        assert round(float(np.mean(tdf)),3)==rep['diversity_retention']['teacher_distinct_functions'],(name,w)
        assert round(margin,4)==rep['coverage_of_intent']['margin']['margin'],(name,w,margin)
        assert [round(lo,4),round(hi,4)]==rep['coverage_of_intent']['margin']['ci'],(name,w,lo,hi)
        assert round(dfm,4)==rep['diversity_retention']['distinct_functions_margin']['margin'],(name,w)
        summary[w][name]=dict(student_hits=int(ss.sum()),teacher_hits=int(tt.sum()),n=len(idx),ratio=float(ss.sum()/tt.sum()),pairs=pairs,
                              coverage_margin=margin,coverage_margin_ci=[lo,hi],student_df=float(np.mean(sdf)),teacher_df=float(np.mean(tdf)),
                              df_retention=float(np.mean(sdf)/np.mean(tdf)),df_margin=dfm,df_margin_ci=[dlo,dhi])
    # student-vs-student paired differences
    flips[w]={};diffs[w]={}
    for a,b in [('i05b_95','i05_45'),('i05b_95','historical_95'),('i05_45','historical_95')]:
        ca=np.array([per[w][a]['cov'][i] for i in idx],float);cb=np.array([per[w][b]['cov'][i] for i in idx],float)
        da=np.array(per[w][a]['df'],float);db=np.array(per[w][b]['df'],float)
        key=f'{a}_minus_{b}'
        lo,hi=bootstrap_ci((ca-cb).tolist());dlo,dhi=bootstrap_ci((da-db).tolist())
        diffs[w][key]=dict(coverage_diff=float(np.mean(ca-cb)),coverage_ci=[lo,hi],df_diff=float(np.mean(da-db)),df_ci=[dlo,dhi])
        flips[w][key]=dict(a_only=int(((ca==1)&(cb==0)).sum()),b_only=int(((ca==0)&(cb==1)).sum()),both=int(((ca==1)&(cb==1)).sum()),neither=int(((ca==0)&(cb==0)).sum()),
                           identical_boards=int(sum(1 for x,y in zip(models[a],models[b]) if [t.lower() for t in x]==[t.lower() for t in y])))
    # rows where the I-05b student and the historical student cover identically
    flips[w]['i05b_vs_historical_same_coverage_rows']=int(sum(1 for i in idx if per[w]['i05b_95']['cov'][i]==per[w]['historical_95']['cov'][i]))
out=dict(artifact_hashes_verified=[str(b05b),str(b05)],reported_values_reproduced=True,bootstrap=dict(resamples=10000,seed=0,method='paired row percentile'),
         summary=summary,student_vs_student=diffs,flips=flips,
         per_row=[dict(window=w,row_index=rows05b[i]['row_index'],teacher_df=per[w]['teacher']['df'][i],teacher_cov=per[w]['teacher']['cov'][i],
                       **{f'{n}_df':per[w][n]['df'][i] for n in models},**{f'{n}_cov':per[w][n]['cov'][i] for n in models}) for w,_,_ in WINDOWS for i in range(84)])
with output.open('x') as f: json.dump(out,f,indent=2)
print(json.dumps(dict(summary=summary,student_vs_student=diffs,flips=flips),indent=2))
