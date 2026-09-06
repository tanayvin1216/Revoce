"""I-12: validate the MiniLM OpenVINO IR bundle end to end, torch-free path vs torch.

    cd mac && VOICELINK_PERCEPTION=0 /usr/local/bin/python3 \
        ../paper/odi/revision/analyses/I-12_validate_minilm_bundle.py [--out PATH]

Checks, in order (all recorded in the JSON written to --out):
  1. bundle completeness: every file the loader needs exists, with size + sha256;
  2. the OpenVINO embedder compiles and encodes a probe set (return code, actual
     backend reported by evaluation.embed.backend_in_use(), embedding dim, unit norm);
  3. tolerance vs the torch sentence-transformers path on the same texts: cosine
     min/mean, max |delta|, on (a) the 3 texts from tests/test_embed_backend.py,
     (b) the 95-entry Phase-4 vocabulary, (c) every scene + transcript of the 84
     held-out Phase-4 rows (the contexts bench_e2e / evaluate.py actually encode);
  4. single-text encode latency per backend (warm, repeated) — a sanity figure only.
Nothing here touches the seam or any result artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MAC = HERE.parents[3] / "mac"
if str(MAC) not in sys.path:
    sys.path.insert(0, str(MAC))
os.environ.setdefault("VOICELINK_PERCEPTION", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REQUIRED = ["openvino_model.xml", "openvino_model.bin", "openvino_tokenizer.xml",
            "openvino_tokenizer.bin", "config.json", "tokenizer.json", "tokenizer_config.json"]
OPTIONAL = ["openvino_detokenizer.xml", "openvino_detokenizer.bin"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fresh_embed(backend: str):
    os.environ["VOICELINK_EMBED_BACKEND"] = backend
    import evaluation.embed as em
    em = importlib.reload(em)
    em.cache_clear()
    return em


def probe_texts() -> dict[str, list[str]]:
    out = {"unit_test_probes": ["what do plants take in from the air", "I need help", "carbon dioxide"]}
    vocab = MAC / "evaluation" / "phase4" / "vocab.json"
    if vocab.exists():
        out["vocab_json"] = json.loads(vocab.read_text())
    try:
        from evaluation.phase4.evaluate import _held_out_rows
        rows = _held_out_rows()
        ctx = []
        for r in rows:
            for k in ("scene", "transcript"):
                t = r.get(k) or ""
                if t:
                    ctx.append(t)
        out["held_out_contexts"] = sorted(set(ctx))
        out["_held_out_rows"] = len(rows)
    except Exception as e:  # pragma: no cover
        out["_held_out_error"] = f"{type(e).__name__}: {e}"
    return out


def compare(ref: np.ndarray, got: np.ndarray) -> dict:
    cos = (ref * got).sum(1)
    return {"n": int(ref.shape[0]), "cos_min": float(cos.min()), "cos_mean": float(cos.mean()),
            "max_abs_delta": float(np.abs(ref - got).max()),
            "norm_min": float(np.linalg.norm(got, axis=1).min()),
            "norm_max": float(np.linalg.norm(got, axis=1).max())}


def time_single(em, texts: list[str], repeats: int) -> dict:
    # warm
    em.embed([texts[0]])
    samples = []
    for i in range(repeats):
        t = texts[i % len(texts)]
        em.cache_forget([t])
        t0 = time.perf_counter()
        em.embed([t])
        samples.append((time.perf_counter() - t0) * 1000.0)
    return {"repeats": repeats, "p50_ms": float(np.percentile(samples, 50)),
            "p90_ms": float(np.percentile(samples, 90)), "mean_ms": float(np.mean(samples)),
            "samples_ms": [round(s, 4) for s in samples]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "I-12_validation.json"))
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--timing-repeats", type=int, default=50)
    args = ap.parse_args()

    from intel import MINILM_OV_DIR, hardware_fingerprint
    model_dir = Path(args.model_dir) if args.model_dir else MINILM_OV_DIR
    report: dict = {"issue": "I-12", "model_dir": str(model_dir),
                    "hardware": hardware_fingerprint(),
                    "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                               text=True, cwd=MAC).stdout.strip()}
    import importlib.metadata as md
    report["versions"] = {p: md.version(p) for p in
                          ("optimum", "optimum-intel", "transformers", "openvino", "openvino-tokenizers",
                           "sentence-transformers", "torch")}

    # 1. bundle completeness
    files = {}
    for f in REQUIRED + OPTIONAL:
        p = model_dir / f
        files[f] = {"present": p.exists(), "required": f in REQUIRED,
                    "bytes": p.stat().st_size if p.exists() else None,
                    "sha256": sha256(p) if p.exists() else None}
    report["bundle"] = files
    missing = [f for f in REQUIRED if not files[f]["present"]]
    report["bundle_complete"] = not missing
    if missing:
        report["status"] = "fail"; report["missing"] = missing
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(json.dumps({k: report[k] for k in ("status", "missing")}))
        return 1

    # 2. compile + encode over OpenVINO, actual backend
    t0 = time.perf_counter()
    ov_em = fresh_embed("openvino")
    probes = probe_texts()
    v = ov_em.embed(probes["unit_test_probes"])
    report["ov_load_and_first_encode_s"] = round(time.perf_counter() - t0, 3)
    report["backend_in_use"] = ov_em.backend_in_use()
    report["embedding_dim"] = int(v.shape[1])
    if report["backend_in_use"] != "openvino":
        report["status"] = "fail"; report["reason"] = "embed fell back to torch"
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(json.dumps({k: report[k] for k in ("status", "reason")}))
        return 1
    ov_model = ov_em._get_model()
    report["ov_embedder"] = {"class": type(ov_model).__name__,
                             "pooled_output": getattr(ov_model, "_pooled_out", None),
                             "enc_inputs": sorted(getattr(ov_model, "_enc_inputs", []))}

    # 3. tolerance vs torch on the three probe sets
    ov_vecs = {k: ov_em.embed(t) for k, t in probes.items() if not k.startswith("_")}
    torch_em = fresh_embed("torch")
    assert torch_em.backend_in_use() in (None, "torch")
    tol = {}
    for k, texts in probes.items():
        if k.startswith("_"):
            continue
        ref = torch_em.embed(texts)
        assert torch_em.backend_in_use() == "torch"
        tol[k] = compare(ref, ov_vecs[k])
    report["tolerance_vs_torch"] = tol
    report["probe_counts"] = {k: len(t) for k, t in probes.items() if not k.startswith("_")}
    report["held_out_rows"] = probes.get("_held_out_rows")

    # 4. single-text latency, both backends (warm, the vocab is NOT what is timed)
    timing_texts = probes.get("held_out_contexts") or probes["unit_test_probes"]
    report["single_text_latency"] = {"torch": time_single(torch_em, timing_texts, args.timing_repeats)}
    ov_em = fresh_embed("openvino")
    ov_em.embed([timing_texts[0]])
    assert ov_em.backend_in_use() == "openvino"
    report["single_text_latency"]["openvino"] = time_single(ov_em, timing_texts, args.timing_repeats)

    worst = min(x["cos_min"] for x in tol.values())
    report["cos_min_overall"] = worst
    report["status"] = "pass" if (worst > 0.999 and report["embedding_dim"] == 384) else "fail"
    report["python"] = platform.python_version()
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({"status": report["status"], "backend_in_use": report["backend_in_use"],
                      "dim": report["embedding_dim"], "tolerance_vs_torch": tol,
                      "single_text_p50_ms": {k: v["p50_ms"] for k, v in report["single_text_latency"].items()}},
                     indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
