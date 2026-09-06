"""Pooled test+val transfer table for each arm (168 paired boards)."""
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from evaluation.phase4 import function_transfer as ft
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ft.ART; OUT.mkdir(parents=True, exist_ok=True)
label = ft._embedding_labeller()
for st in ["v2", "v3", "prior", "random"]:
    pairs, meta = [], None
    for split in ["test", "val"]:
        p, meta = ft.collect_pairs(split, st)
        pairs += p
    res = ft.transfer_table(pairs, label, meta["core_words"])
    res["meta"] = {**meta, "split": "test+val", "n_rows": len(pairs)}
    (OUT / f"function_transfer_pooled_{st}.json").write_text(json.dumps(res, indent=1))
    o = res["overall"]
    print(st, "n", res["n_heldout"], "JF", o["function_jaccard"], o["function_jaccard_ci95"], "JW", o["word_jaccard"], o["word_jaccard_ci95"], "ret", o["retention_ratio"])
    for f in ["I-dont-know", "emotion"]:
        r = res["per_function"][f]
        print("   ", f, "teacher", r["teacher_presence"], "student", r["student_presence"], "ret", r["retention"], "CI", r["presence_margin_ci95"])
