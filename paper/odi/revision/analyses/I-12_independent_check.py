"""Read-only I-12 checks; no OpenVINO compilation. Run from mac/."""
import os, sys, json, hashlib, tempfile
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET
import numpy as np
sys.path.insert(0, str(Path.cwd()))
os.environ['VOICELINK_EMBED_BACKEND'] = 'torch'
os.environ['HF_HUB_OFFLINE'] = '1'
A = Path('../paper/odi/revision/analyses')
v = json.loads((A/'I-12_validation.json').read_text())
base = Path('intel/artifacts/minilm-ov')
fresh = Path('intel/artifacts/minilm-ov-20260905')
r = {'files': {}}
for name, expected in v['bundle'].items():
    b = (base/name).read_bytes()
    sha = hashlib.sha256(b).hexdigest()
    assert sha == expected['sha256'] and len(b) == expected['bytes']
    assert b == (fresh/name).read_bytes()
    r['files'][name] = {'bytes':len(b),'sha256':sha,'fresh_identical':True}
root = ET.parse(base/'openvino_model.xml').getroot()
r['xml_layers'] = len(root.findall('./layers/layer'))
ranges = [int(d.attrib['offset'])+int(d.attrib['size']) for d in root.findall('./layers/layer/data') if 'offset' in d.attrib and 'size' in d.attrib]
assert max(ranges) <= (base/'openvino_model.bin').stat().st_size
r['max_constant_end'] = max(ranges)
r['timings_rederived'] = {}
for backend, vals in v['single_text_latency'].items():
    x = np.array(vals['samples_ms'])
    stats = {'p50_ms':float(np.percentile(x,50)), 'p90_ms':float(np.percentile(x,90)), 'mean_ms':float(x.mean())}
    assert len(x) == vals['repeats'] == 50
    assert all(abs(stats[k]-vals[k]) < 0.0001 for k in stats)
    r['timings_rederived'][backend] = stats
from evaluation.phase4.evaluate import _held_out_rows
from evaluation import embed as em
rows = _held_out_rows()
contexts = sorted({row[k] for row in rows for k in ('scene','transcript') if row.get(k)})
vocab = json.loads(Path('evaluation/phase4/vocab.json').read_text())
assert (len(rows),len(contexts),len(vocab)) == (84,153,95)
r['held_out_rows']=len(rows)
r['torch_probes']={}
for name, texts in [('vocab_json',vocab),('held_out_contexts',contexts),('unit_test_probes',['what do plants take in from the air','I need help','carbon dioxide'])]:
    x=em.embed(texts)
    assert x.shape == (len(texts),384) and np.isfinite(x).all()
    assert np.allclose(np.linalg.norm(x,axis=1),1,atol=1e-5)
    r['torch_probes'][name]={'n':len(texts),'dim':384,'norm_min':float(np.linalg.norm(x,axis=1).min()),'norm_max':float(np.linalg.norm(x,axis=1).max()),'texts_sha256':hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest()}
r['torch_backend']=em.assert_backend('torch')
from intel import bench_e2e
import intel, rank_vocabulary as rv
with tempfile.TemporaryDirectory(prefix='I-12-check-') as temp:
    with patch.object(intel,'MINILM_OV_DIR',Path(temp)), patch.object(rv,'rank_vocabulary',side_effect=AssertionError('seam must not be reached')) as seam:
        try:
            bench_e2e.bench_backend('ov-full', rows[:2])
        except FileNotFoundError as e:
            r['negative_control']={'error':str(e),'seam_calls':seam.call_count}
        else:
            raise AssertionError('missing bundle did not raise')
        assert seam.call_count == 0
r['status']='pass'
with (A/'I-12_independent_check.json').open('x') as f:
    json.dump(r,f,indent=2)
print(json.dumps(r,indent=2))
