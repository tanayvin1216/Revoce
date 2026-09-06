"""Independent artifact/board checks; run from mac with perception disabled.
Assumption: archived OV logits are execution evidence, not a fresh OV run.
"""
import os, sys, json, hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
os.environ.update(VOICELINK_PERCEPTION='0', HF_HUB_OFFLINE='1', VOICELINK_EMBED_BACKEND='torch', VOICELINK_STUDENT_BACKEND='torch')
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'mac'))
import numpy as np
import torch
from evaluation.phase4 import infer, data
from evaluation.phase4.student import StudentRanker
from rank_vocabulary import GUARANTEED_CORE
from evaluation.embed import assert_backend
A = ROOT / 'paper/odi/revision/analyses'
E = A / 'I-07b_artifacts'
R = A / 'I-07b_artifacts_repro'
B = A / 'I-05b_split_first'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
report = {}
for folder in (E, R, B):
    m = read(folder / 'manifest.json')
    bad = [n for n,h in m['artifact_sha256'].items() if sha(folder/n) != h]
    assert not bad, bad
    report[folder.name + '_verified_artifacts'] = len(m['artifact_sha256'])
files = read(E/'manifest.json')['artifact_sha256']
assert all((E/n).read_bytes() == (R/n).read_bytes() for n in files)
report['repro_byte_identical_files'] = len(files)
rows, split, vocab = [read(B / n) for n in ('rows.json','splits.json','vocab.json')]
assert len(set(split['train'] + split['val'] + split['test'])) == 559
assert split == read(ROOT/'mac/evaluation/phase4/splits.json')
ck = torch.load(B/'student.pt', map_location='cpu', weights_only=False)
torch.set_num_threads(1)
model = StudentRanker(ck['context_dim'], ck['hidden_dim'], ck['vocab_size'], ck['dropout'])
model.load_state_dict(ck['state_dict']); model.eval()
X = np.load(E/'features_559x769.npy')
logs = {name: np.load(E / ('logits_' + name + '.npy')) for name in ('torch','ov_fp32','ov_int8')}
boards = read(E/'boards_test_by_backend.json')
assert [b['row_index'] for b in boards] == split['test']
class DumpHead:
    # Replay one row's archived output while requiring that the real selector supplies its feature.
    def __init__(self, i, name): self.i, self.name = i, name
    def __call__(self, x):
        assert np.max(np.abs(x[0]-X[self.i])) < 1e-6
        return logs[self.name][self.i:self.i+1]
counts = {n:0 for n in logs}
jacs, dyn, diffs = [], [], []
for b in boards:
    i = b['row_index']; r=rows[i]
    for name in logs:
        head = model if name == 'torch' else DumpHead(i, name)
        got = infer.student_board(head, vocab, r['scene'], r['transcript'], r['is_question'], 10, 6, GUARANTEED_CORE)
        assert got == b[name + '_board'], (i, name, got, b[name + '_board'])
        assert len(got) == len(set(got)) == 10 and got[:4] == GUARANTEED_CORE[:4]
        counts[name] += 1
    t,q = b['torch_board'], b['ov_int8_board']
    jac = lambda x,y: len(set(x)&set(y))/len(set(x)|set(y))
    jacs.append(jac(t,q)); dyn.append(jac(t[4:],q[4:]))
    if t != q: diffs.append(i)
report['fresh_torch_and_dump_replayed_boards_match'] = counts
report['board_agreement'] = dict(mean=float(np.mean(jacs)), minimum=min(jacs), dynamic_mean=float(np.mean(dyn)), identical=84-len(diffs), differing_ids=diffs)
# Independent top-six calculation uses torch.topk rather than the supplied verifier's argsort.
t = torch.from_numpy(logs['ov_fp32'][split['test']]).topk(6).indices.tolist()
q = torch.from_numpy(logs['ov_int8'][split['test']]).topk(6).indices.tolist()
hits = [len(set(a)&set(b)) for a,b in zip(t,q)]
report['test_top6'] = dict(shared_entries=sum(hits), denominator=84*6, mean=sum(hits)/(84*6), min=min(hits)/6, histogram={str(k):hits.count(k) for k in set(hits)})
report['ir_structure'] = {}
for xml in sorted(E.rglob('*.xml')):
    tree=ET.parse(xml); blob=xml.with_suffix('.bin').read_bytes()
    consts=[]
    for layer in tree.findall('./layers/layer'):
        if layer.get('type') == 'Const':
            d=layer.find('data').attrib
            assert int(d['offset'])+int(d['size']) <= len(blob)
            consts.append(d)
    out=tree.find("./layers/layer[@type='Result']/input/port")
    assert int(out.findall('dim')[-1].text) == 95
    report['ir_structure'][str(xml.relative_to(E))] = dict(constants=len(consts), i8_shapes=[d['shape'] for d in consts if d['element_type']=='i8'], subset=tree.find('./rt_info/nncf/quantization/subset_size').get('value') if tree.find('./rt_info/nncf') is not None else None)
    if xml.name=='student_int8.xml':
        layer=tree.find("./layers/layer[@name='net.4.weight']/data")
        d=layer.attrib
        weight=np.frombuffer(blob,dtype=np.float32,count=int(d['size'])//4,offset=int(d['offset'])).reshape(95,192)
        assert np.array_equal(weight,ck['state_dict']['net.4.weight'].numpy())
        assert d['element_type']=='f32'
for n in ('fp32','int8'):
    assert (E/f'student_{n}.vocab.sha').read_text()==ck['vocab_sha']==data.vocab_sha(vocab)
report['size_bytes']={n:sum((E/f'student_{n}.{ext}').stat().st_size for ext in ('xml','bin')) for n in ('fp32','int8')}

# Independently reconstruct context identity from raw text, without data.context_text.
contexts = [(r.get('scene', '') + ' || ' + r.get('transcript', '')).strip() for r in rows]
context_ids = [hashlib.sha256(c.encode()).hexdigest()[:16] for c in contexts]
assert hashlib.sha256('\n'.join(contexts).encode()).hexdigest()[:16] == split['fingerprint']
cal = read(E/'calibration_ids.json')
assert cal['row_indices'] == split['train']
assert cal['context_sha256'] == [context_ids[i] for i in split['train']]
assert len(set(contexts)) == 559
assert all(not (set(contexts[i] for i in split['train']) & set(contexts[i] for i in split[s])) for s in ('val','test'))
assert (E/'calibration_ids.json').read_bytes() == (A/'I-07_export_v1/calibration_ids.json').read_bytes()
assert (E/'features_559x769.npy').read_bytes() == (A/'I-07_export_v1/features_559x769.npy').read_bytes()
v2 = [json.loads(s) for s in (ROOT/'mac/evaluation/phase4/v2_scenarios.jsonl').read_text().splitlines() if s.strip()]
lookup = {contexts[i]: (s,i) for s in ('train','val','test') for i in split[s]}
joined = {s: [] for s in ('train','val','test','unmatched')}
for r in v2:
    c = (r.get('scene','') + ' || ' + r.get('transcript','')).strip()
    s,i = lookup.get(c, ('unmatched',None)); joined[s].append(i)
assert {s:len(v) for s,v in joined.items()} == dict(train=255,val=54,test=51,unmatched=0)
report['context_identity'] = dict(fingerprint=split['fingerprint'], calibration_row_indices=split['train'], calibration_context_ids=[context_ids[i] for i in split['train']], test_row_indices=split['test'], test_context_ids=[context_ids[i] for i in split['test']], historical_join=joined, calibration_val_overlap=0, calibration_test_overlap=0)
from collections import Counter
from evaluation.hedges import is_hedge
swaps=Counter(); inserted=Counter(); removed=Counter(); same_top6_changed=0; diff_top6_changed=0
for b,h in zip(boards,hits):
    t,q=set(b['torch_board']),set(b['ov_int8_board'])
    if t!=q:
        swaps[len(t-q)]+=1; inserted.update(q-t); removed.update(t-q)
        assert not any(is_hedge(w) for w in (t^q))
        same_top6_changed += h==6; diff_top6_changed += h!=6
report['board_changes'] = dict(swaps=dict(swaps),inserted=dict(inserted),removed=dict(removed),same_top6_changed=same_top6_changed,diff_top6_changed=diff_top6_changed)

assert_backend('torch')
report['all_pass']=True
with (A/'I-07b_independent_check.json').open('x') as f: json.dump(report,f,indent=2)
print(json.dumps(report,indent=2))
