"""Independent saved-bundle identities and per-row coverage reconstruction.
Run from mac with script BUNDLE NEW_OUTPUT.json. Never rewrites bundle artifacts.
"""
import hashlib,json,os,sys
from pathlib import Path
import numpy as np
MAC=Path(__file__).resolve().parents[4]/'mac';sys.path.insert(0,str(MAC))
os.environ.update(VOICELINK_PERCEPTION='0',HF_HUB_OFFLINE='1',VOICELINK_EMBED_BACKEND='torch')
from evaluation.phase3 import metrics3
from evaluation.coverage_metrics import bootstrap_ci
bundle=Path(sys.argv[1]);output=Path(sys.argv[2])
manifest=json.loads((bundle/'manifest.json').read_text())
for name,h in manifest['artifact_sha256'].items():
    assert hashlib.sha256((bundle/name).read_bytes()).hexdigest()==h
boards=json.loads((bundle/'boards.json').read_text());metrics=json.loads((bundle/'metrics.json').read_text())['metrics']
core={'yes','no','wait','i need help'};checks={};per=[]
for name,k,dyn in [('full10',10,False),('dyn6',6,True),('full7',7,False)]:
    pairs={'11':0,'01':0,'10':0,'00':0};ss=[];tt=[];sdf=[];tdf=[]
    for row in boards:
        moves=[m for m in row['acceptable_moves'] if not metrics3.is_hedge(m)]
        sc=[]
        for arm,dfs in [('student',sdf),('teacher',tdf)]:
            b=row[arm+'_board'];window=([w for w in b if w.lower() not in core] if dyn else b)[:k]
            dfs.append(metrics3.distinct_functions_at(window,k))
            dynamic=[w for w in window if w.lower() not in core]
            sc.append(int(metrics3.coverage_of_intent(dynamic,moves,k)) if moves else None)
        per.append(dict(window=name,row_index=row['row_index'],student_df=sdf[-1],teacher_df=tdf[-1],student_coverage=sc[0],teacher_coverage=sc[1]))
        if moves:
            ss.append(sc[0]);tt.append(sc[1]);pairs[f'{sc[0]}{sc[1]}']+=1
    expected=metrics[name]
    assert pairs==expected['coverage_of_intent']['paired_counts_student_teacher']
    assert round(float(np.mean(sdf)),3)==expected['diversity_retention']['student_distinct_functions']
    assert round(float(np.mean(tdf)),3)==expected['diversity_retention']['teacher_distinct_functions']
    margin=float(np.mean(np.array(ss)-.9*np.array(tt)))
    assert round(margin,4)==expected['coverage_of_intent']['margin']['margin']
    checks[name]=dict(student_hits=sum(ss),teacher_hits=sum(tt),n=len(ss),pairs=pairs,ratio=sum(ss)/sum(tt),margin=margin)
with output.open('x') as f: json.dump(dict(artifact_hashes_verified=True,checks=checks,per_row=per),f,indent=2)
print(json.dumps(checks,indent=2))
