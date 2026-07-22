"""Quantization Pareto for the v3 bi-encoder student (OpenVINO + NNCF).

    VOICELINK_PERCEPTION=0 python3 intel/biencoder_pareto.py

Why this is not just "quantize the model"
-----------------------------------------
A bi-encoder has two halves with DIFFERENT cost profiles, so quantizing them as
one blob throws away the interesting decision:

  * The CONTEXT tower is the hot path. It runs once per board, on the device,
    inside the latency budget. Both its size and its speed matter.
  * The PHRASE tower never runs at serve time. It is applied offline to the
    whole bank and the projections are cached, so its latency is irrelevant.
    Only the quality of the cached projections matters — and its weights are
    ~40% of the checkpoint.

That asymmetry means the honest question is not "INT8 or not" but "which
precision for which tower", which is a 2-D sweep. This script runs it and
reports the Pareto front: on-disk size, agreement with the FP32 board, and
hot-path latency, for every (context, phrase) precision pair.

Precisions per tower: FP32, INT8 (NNCF post-training quantization with real
calibration data), INT4 (NNCF weight compression).

Agreement metric: top-k_dyn overlap against the FP32 model's board on held-out
contexts. Board overlap is what the user actually experiences; logit MSE is not.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MAC = HERE.parent
if str(MAC) not in sys.path:
    sys.path.insert(0, str(MAC))

from intel import ARTIFACTS_DIR  # noqa: E402

OUT_DIR = ARTIFACTS_DIR / "biencoder"
PHASE4 = MAC / "evaluation" / "phase4"
CKPT = PHASE4 / "student_biencoder.pt"

PRECISIONS = ("fp32", "int8", "int4")


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------

def _load_model():
    import torch
    from evaluation.phase4.student_biencoder import BiEncoderRanker
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    m = BiEncoderRanker(
        context_dim=ck["context_dim"], phrase_dim=ck["phrase_dim"],
        hidden_dim=ck["hidden_dim"], proj_dim=ck["proj_dim"],
        dropout=ck["dropout"],
    )
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


def export_towers() -> dict[str, Path]:
    """Each tower to its own ONNX graph -> its own FP32 IR."""
    import torch
    import openvino as ov
    from evaluation.phase4.student_biencoder import ContextTowerOnly, PhraseTowerOnly

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, ck = _load_model()
    paths: dict[str, Path] = {}

    for name, wrapper, dim in (
        ("context", ContextTowerOnly(model), ck["context_dim"]),
        ("phrase", PhraseTowerOnly(model), ck["phrase_dim"]),
    ):
        onnx_path = OUT_DIR / f"{name}_tower.onnx"
        dummy = torch.randn(1, dim)
        torch.onnx.export(
            wrapper.eval(), dummy, str(onnx_path),
            input_names=["x"], output_names=["proj"],
            dynamic_axes={"x": {0: "batch"}, "proj": {0: "batch"}},
            opset_version=17,
        )
        xml = OUT_DIR / f"{name}_fp32.xml"
        ov.save_model(ov.convert_model(str(onnx_path)), str(xml),
                      compress_to_fp16=False)
        paths[name] = xml
        print(f"[export] {name} tower -> {xml.name}", flush=True)

    return paths


# --------------------------------------------------------------------------
# quantization
# --------------------------------------------------------------------------

def _quantize(src_xml: Path, dst_xml: Path, calib: np.ndarray, mode: str) -> Path:
    import nncf
    import openvino as ov

    model = ov.Core().read_model(str(src_xml))
    if mode == "int8":
        ds = nncf.Dataset(calib, transform_func=lambda row: row[None, :])
        out = nncf.quantize(model, ds, subset_size=len(calib))
    elif mode == "int4":
        # Weight-only compression. group_size=-1 (per-channel) because these
        # layers are far too small for the default 128-wide groups.
        out = nncf.compress_weights(
            model, mode=nncf.CompressWeightsMode.INT4_ASYM,
            group_size=-1, ratio=1.0)
    else:
        raise ValueError(mode)
    ov.save_model(out, str(dst_xml), compress_to_fp16=False)
    return dst_xml


def build_variants(fp32: dict[str, Path], calib: dict[str, np.ndarray]) -> dict:
    """{tower: {precision: xml}} for every precision we can build."""
    variants: dict[str, dict[str, Path]] = {}
    for tower, xml in fp32.items():
        variants[tower] = {"fp32": xml}
        for mode in ("int8", "int4"):
            dst = OUT_DIR / f"{tower}_{mode}.xml"
            try:
                variants[tower][mode] = _quantize(xml, dst, calib[tower], mode)
                print(f"[nncf] {tower} {mode} -> {dst.name}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[nncf] {tower} {mode} FAILED: "
                      f"{type(exc).__name__}: {exc}", flush=True)
    return variants


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------

def _ir_bytes(xml: Path) -> int:
    return xml.stat().st_size + xml.with_suffix(".bin").stat().st_size


def _compiled(xml: Path):
    import openvino as ov
    # Pin f32 inference: the ARM CPU plugin otherwise defaults to f16, which
    # would silently blur the FP32 baseline we are measuring everything against.
    return ov.Core().compile_model(str(xml), "CPU",
                                   {"INFERENCE_PRECISION_HINT": "f32"})


def _project(xml: Path, X: np.ndarray) -> np.ndarray:
    c = _compiled(xml)
    out = c.output(0)
    return np.concatenate([c(X[i:i + 32])[out] for i in range(0, len(X), 32)])


def _topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Mean |top-k(a) ∩ top-k(b)| / k over rows."""
    ta = np.argsort(-a, axis=1)[:, :k]
    tb = np.argsort(-b, axis=1)[:, :k]
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(ta, tb)]))


def _hot_latency(xml: Path, x: np.ndarray, iters: int = 300) -> float:
    """Median ms for ONE context forward pass — the serving hot path."""
    c = _compiled(xml)
    out = c.output(0)
    one = x[:1]
    for _ in range(30):
        c(one)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        c(one)
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def sweep(variants: dict, Xc: np.ndarray, Xp: np.ndarray, k: int) -> list[dict]:
    """Every (context precision, phrase precision) pair, scored on board overlap."""
    ctx_proj = {p: _project(x, Xc) for p, x in variants["context"].items()}
    ph_proj = {p: _project(x, Xp) for p, x in variants["phrase"].items()}

    ref = ctx_proj["fp32"] @ ph_proj["fp32"].T
    lat = {p: _hot_latency(x, Xc) for p, x in variants["context"].items()}

    rows = []
    for cp, cproj in ctx_proj.items():
        for pp, pproj in ph_proj.items():
            scores = cproj @ pproj.T
            rows.append({
                "context": cp,
                "phrase": pp,
                "bytes": _ir_bytes(variants["context"][cp])
                         + _ir_bytes(variants["phrase"][pp]),
                "board_overlap": round(_topk_overlap(ref, scores, k), 4),
                "hot_ms": round(lat[cp], 4),
            })
    return rows


def _pareto(rows: list[dict]) -> list[dict]:
    """Keep variants not dominated on (smaller bytes, higher overlap)."""
    front = []
    for r in rows:
        if not any(o["bytes"] <= r["bytes"]
                   and o["board_overlap"] >= r["board_overlap"]
                   and (o["bytes"], -o["board_overlap"]) != (r["bytes"], -r["board_overlap"])
                   for o in rows):
            front.append(r)
    return front


def main() -> int:
    if not CKPT.exists():
        print(f"no checkpoint at {CKPT} — run train_biencoder.py first")
        return 1
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    from evaluation.phase4 import bank as bank_mod
    from evaluation.phase4 import data as d
    from evaluation.phase4.train import gather_rows

    rows_data = gather_rows()
    phrases, _ = bank_mod.load_bank()
    Xc = d.context_features(rows_data).astype(np.float32)
    Xp = bank_mod.phrase_features(phrases).astype(np.float32)

    import yaml
    cfg = yaml.safe_load((PHASE4 / "config.yaml").read_text())
    k = int(cfg["board"]["k_dyn"])

    fp32 = export_towers()
    variants = build_variants(fp32, {"context": Xc, "phrase": Xp})
    results = sweep(variants, Xc, Xp, k)

    front = {(r["context"], r["phrase"]) for r in _pareto(results)}
    results.sort(key=lambda r: (-r["board_overlap"], r["bytes"]))

    lines = [
        "| context | phrase | IR size | board overlap vs FP32 | hot-path ms | Pareto |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['context']} | {r['phrase']} | {r['bytes'] / 1024:.0f} KB | "
            f"{r['board_overlap']:.3f} | {r['hot_ms']:.3f} | "
            f"{'yes' if (r['context'], r['phrase']) in front else ''} |"
        )
    table = "\n".join(lines)
    print("\n" + table, flush=True)

    (OUT_DIR / "pareto.json").write_text(json.dumps(
        {"k_dyn": k, "n_contexts": len(Xc), "n_phrases": len(Xp),
         "results": results}, indent=2))
    (OUT_DIR / "PARETO.md").write_text(
        "# Bi-encoder quantization Pareto\n\n"
        f"Measured on {len(Xc)} contexts x {len(Xp)} bank phrases, "
        f"top-{k} board overlap against the FP32 model.\n"
        "Hot-path latency is ONE context forward pass; the phrase tower runs "
        "offline over the bank and is cached, so its precision costs no "
        "serving latency.\n\n"
        "Measured on Apple Silicon via the OpenVINO ARM CPU plugin — these are "
        "NOT Intel-hardware numbers.\n\n" + table + "\n")
    print(f"\nwrote {OUT_DIR / 'PARETO.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
