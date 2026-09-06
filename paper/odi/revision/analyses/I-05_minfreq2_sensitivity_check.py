"""CHECKER sensitivity variant (NOT a claimable configuration): same pipeline as the
implementer's script but vocab.min_freq=2 on the 391 training rows, to see how much of
the historical->new change is the smaller fitting set tightening the frequency floor.
Run from mac/. Writes to --out (must not exist)."""
import sys, importlib.util
from pathlib import Path
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('impl', HERE / 'I-05_split_first_implement.py')
impl = importlib.util.module_from_spec(spec); spec.loader.exec_module(impl)
_orig = impl.tr.load_config
def patched():
    cfg, h = _orig(); cfg['vocab']['min_freq'] = 2; return cfg, h + '-minfreq2'
impl.tr.load_config = patched
impl.main()
