"""I-05b diagnostic: do vocabulary entries that exist ONLY because of all-row (val/test-inclusive)
frequency counting reach the historical student's held-out boards, and do they carry the @7 rows
where the historical student covers and the I-05b student does not? Read-only over existing files."""
import json,os,sys
from pathlib import Path
MAC=Path(__file__).resolve().parents[4]/'mac';sys.path.insert(0,str(MAC))
os.environ.update(VOICELINK_PERCEPTION='0',HF_HUB_OFFLINE='1',VOICELINK_EMBED_BACKEND='torch')
from evaluation.phase3 import metrics3
A=Path(__file__).resolve().parent;out=A/'I-05b_leak_entries_v1.json'
hist=set(json.loads((MAC/'evaluation/phase4/vocab.json').read_text()));new=set(json.loads((A/'I-05b_split_first/vocab.json').read_text()))
rows=json.loads((A/'I-05b_split_first/rows.json').read_text());sp=json.loads((A/'I-05b_split_first/splits.json').read_text())
core={'yes','no','wait','i need help'}
def freq(idx):
    f={}
    for i in idx:
        for w in rows[i]['board']:
            k=w.strip().lower()
            if k and k not in core: f[k]=f.get(k,0)+1
    return f
ftr,fva,fte=freq(sp['train']),freq(sp['val']),freq(sp['test'])
hist_only=sorted(hist-new);new_only=sorted(new-hist)
tiers={w:dict(train=ftr.get(w,0),val=fva.get(w,0),test=fte.get(w,0)) for w in hist_only}
below3_train=[w for w in hist_only if ftr.get(w,0)<3]
needs_test=[w for w in hist_only if ftr.get(w,0)+fva.get(w,0)<3]
cmp=json.loads((A/'I-05b_compare_v1.json').read_text());per=cmp['per_row']
i03=json.loads((A/'I-03_numbers_implement_v2.json').read_text());hb={r['index']:r['student_board'] for r in i03['per_row_table2'] if r['window']=='full10'}
b05b=json.loads((A/'I-05b_split_first/boards.json').read_text())
res={}
for w,k in (('full10',10),('dyn6',6),('full7',7)):
    pr=[p for p in per if p['window']==w];boards_with=0;flip_rows=[]
    for pos,p in enumerate(pr):
        board=hb[pos];window=[x for x in board if x.lower() not in core][:6] if w=='dyn6' else board[:k]
        hits=[x for x in window if x.lower() in set(hist_only)]
        if hits: boards_with+=1
        if p['historical_95_cov']==1 and p['i05b_95_cov']==0:
            moves=[m for m in b05b[pos]['acceptable_moves'] if not metrics3.is_hedge(m)]
            dyn=[x for x in window if x.lower() not in core]
            # which entries in the historical window individually satisfy coverage?
            carriers=[x for x in dyn if metrics3.coverage_of_intent([x],moves,k)]
            flip_rows.append(dict(row_index=p['row_index'],historical_window=window,carriers=carriers,carriers_hist_only=[c for c in carriers if c.lower() in set(hist_only)],i05b_window=([x for x in b05b[pos]['student_board'] if x.lower() not in core][:6] if w=='dyn6' else b05b[pos]['student_board'][:k])))
    res[w]=dict(historical_boards_with_hist_only_entry_in_window=boards_with,hist_covers_i05b_not=len(flip_rows),flip_rows_where_only_hist_only_entries_carry=sum(1 for f in flip_rows if f['carriers'] and all(c.lower() in set(hist_only) for c in f['carriers'])),flip_rows=flip_rows)
summary=dict(historical_only=len(hist_only),historical_only_below_min_freq3_on_train=len(below3_train),historical_only_needing_test_rows_to_reach_3=len(needs_test),new_only=len(new_only),historical_only_entries=tiers,new_only_train_freq={w:ftr.get(w,0) for w in new_only},windows=res)
with out.open('x') as f: json.dump(summary,f,indent=2)
print(json.dumps({k:v for k,v in summary.items() if k not in ('historical_only_entries','new_only_train_freq')},indent=1,default=str)[:6000])
print('hist-only train/val/test counts:',{w:(t['train'],t['val'],t['test']) for w,t in tiers.items()})
