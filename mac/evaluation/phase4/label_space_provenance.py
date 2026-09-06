"""How much of the v2 label space and the v3 phrase bank depends on held-out rows?
Rebuilds both from the TRAINING split only and diffs them against the shipped
(all-rows) versions. No model call beyond the MiniLM embedder used for dedup."""
import sys, json
from pathlib import Path
sys.path.insert(0, ".")
import yaml
from evaluation.phase4 import data as d, bank as bank_mod, train as train_mod
HERE = Path("evaluation/phase4")
cfg = yaml.safe_load((HERE / "config.yaml").read_text())
rows = train_mod.gather_rows()
splits = json.loads((HERE / "splits.json").read_text())
assert splits["fingerprint"] == d.rows_fingerprint(rows)
tr = [rows[i] for i in splits["train"]]; te = [rows[i] for i in splits["test"]]; va = [rows[i] for i in splits["val"]]
core = {w.lower() for w in d.CORE_SET} if hasattr(d, "CORE_SET") else set()
def dyn(r): return [w.strip().lower() for w in r["board"] if w.strip().lower() not in core]
out = {"n_rows": len(rows), "n_train": len(tr), "n_test": len(te), "n_val": len(va)}
# v2 vocab
v_all, _ = d.build_vocab(rows, cfg["vocab"]["min_freq"], cfg["vocab"]["max_size"])
v_all, canon_all = d.dedup_vocab(v_all, cfg["vocab"]["dedup_tau"])
v_tr, _ = d.build_vocab(tr, cfg["vocab"]["min_freq"], cfg["vocab"]["max_size"])
v_tr, canon_tr = d.dedup_vocab(v_tr, cfg["vocab"]["dedup_tau"])
shipped = json.loads(d.VOCAB_PATH.read_text())
shipped = shipped if isinstance(shipped, list) else shipped.get("vocab", shipped)
out["vocab"] = {"all_rows": len(v_all), "shipped": len(shipped), "matches_shipped": sorted(v_all) == sorted(w.lower() for w in shipped),
                "train_only": len(v_tr), "in_all_not_train": sorted(set(v_all) - set(v_tr)), "in_train_not_all": sorted(set(v_tr) - set(v_all))}
leak_labels = set(v_all) - set(v_tr)
te_dyn = [w for r in te for w in dyn(r)[:3]]  # top-7 window = 4 pins + 3 dynamic
out["vocab"]["test_top7_dynamic_items"] = len(te_dyn)
out["vocab"]["test_top7_items_in_leaked_labels"] = sum(1 for w in te_dyn if canon_all.get(w, w) in leak_labels or w in leak_labels)
# v3 bank
tau = float(cfg["vocab"].get("bank_dedup_tau", cfg["vocab"]["dedup_tau"]))
b_all, cb_all = bank_mod.build_phrase_bank(rows, dedup_tau=tau, min_freq=1, linkage="complete")
b_tr, cb_tr = bank_mod.build_phrase_bank(tr, dedup_tau=tau, min_freq=1, linkage="complete")
shipped_bank, _ = bank_mod.load_bank()
out["bank"] = {"all_rows": len(b_all), "shipped": len(shipped_bank), "matches_shipped": sorted(b_all) == sorted(w.lower() for w in shipped_bank),
               "train_only": len(b_tr), "phrases_only_from_heldout": len(set(b_all) - set(b_tr))}
leak_bank = set(b_all) - set(b_tr)
out["bank"]["test_top7_items_only_in_heldout_bank"] = sum(1 for w in te_dyn if cb_all.get(w, w) in leak_bank or w in leak_bank)
out["bank"]["test_top7_items_representable_all_bank"] = sum(1 for w in te_dyn if (cb_all.get(w, w) in set(b_all)) or w in set(b_all))
out["bank"]["test_top7_items_representable_train_bank"] = sum(1 for w in te_dyn if (cb_tr.get(w, w) in set(b_tr)) or w in set(b_tr))
print(json.dumps(out, indent=1))
(d.ART / "label_space_provenance.json").write_text(json.dumps(out, indent=1))
