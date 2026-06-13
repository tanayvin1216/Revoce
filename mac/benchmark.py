"""
Perception benchmarks for VoiceLink (Mac).

Measures the two real backends and the change-detector gate, then writes figures
to mac/figures/ plus a results.json. Run on the target machine:

    python3 mac/benchmark.py

Backends are the real ones (faster-whisper base.en + mlx-vlm Qwen2.5-VL-3B-4bit);
make sure they're installed and the models are cached first.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import perception.config as cfg  # noqa: E402
from perception.change_detector import FrameChangeDetector  # noqa: E402
from perception.scene_describer import MlxVlmSceneDescriber  # noqa: E402
from perception.transcriber import make_transcriber  # noqa: E402

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)
SR = 16000


# ----------------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------------
def make_speech(seconds: float) -> np.ndarray:
    """Synthesize ~`seconds` of real speech with macOS `say` (16 kHz mono)."""
    sentence = "The quick brown fox jumps over the lazy dog. "
    text = sentence * max(1, round(seconds / 2.5))
    path = tempfile.mktemp(suffix=".wav")
    subprocess.run(["say", "-o", path, "--data-format=LEI16@16000", text],
                   check=True, capture_output=True)
    with wave.open(path, "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def make_image(dim: int) -> bytes:
    """A synthetic scene with some content (so the VLM's token count is realistic)."""
    img = np.full((dim, dim, 3), 205, np.uint8)
    cv2.rectangle(img, (dim // 8, dim // 8), (dim // 2, dim // 2), (160, 90, 40), -1)
    cv2.circle(img, (int(dim * 0.7), int(dim * 0.62)), dim // 8, (80, 160, 60), -1)
    cv2.putText(img, "scene", (dim // 10, int(dim * 0.92)),
                cv2.FONT_HERSHEY_SIMPLEX, dim / 400.0, (20, 20, 20), max(1, dim // 320))
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


# ----------------------------------------------------------------------------
# benchmarks
# ----------------------------------------------------------------------------
def bench_whisper() -> dict:
    print("\n== Whisper (faster-whisper base.en) ==")
    tx = make_transcriber(cfg)
    tx.transcribe(make_speech(2).astype(np.float32) / 32768.0, SR)  # warmup
    durations, latencies, rtfs = [], [], []
    for target in (2, 5, 10, 20, 30):
        pcm = make_speech(target)
        dur = len(pcm) / SR
        f32 = pcm.astype(np.float32) / 32768.0
        t0 = time.time()
        tx.transcribe(f32, SR)
        dt = time.time() - t0
        durations.append(dur)
        latencies.append(dt)
        rtfs.append(dt / dur)
        print(f"  audio={dur:5.1f}s  latency={dt:5.2f}s  rtf={dt / dur:.3f}")
    return {"durations_s": durations, "latency_s": latencies, "rtf": rtfs}


def bench_vlm() -> dict:
    print("\n== VLM (Qwen2.5-VL-3B-4bit via mlx-vlm), uncapped, by resolution ==")
    # no cap here — we want true cost vs resolution to justify VLM_MAX_IMAGE_DIM
    vlm = MlxVlmSceneDescriber(cfg.VLM_MODEL, cfg.VLM_PROMPT, cfg.VLM_MAX_TOKENS,
                               max_image_dim=10 ** 6)
    vlm.describe(make_image(256))  # warmup
    dims = [256, 512, 768, 1024, 1536, 2160]
    latencies = []
    for d in dims:
        img = make_image(d)
        t0 = time.time()
        vlm.describe(img)
        dt = time.time() - t0
        latencies.append(dt)
        print(f"  image={d:4d}px  latency={dt:6.2f}s")
    return {"dims_px": dims, "latency_s": latencies}


def bench_gate() -> dict:
    print("\n== Change detector (gate) cost per VGA frame ==")
    det = FrameChangeDetector(cfg.CHANGE_THRESHOLD)
    jpeg = make_image(640)
    times_ms = []
    for _ in range(300):
        t0 = time.perf_counter()
        arr = np.frombuffer(jpeg, np.uint8)
        gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        gray = cv2.resize(gray, (cfg.CHANGE_GRAY_SIZE, cfg.CHANGE_GRAY_SIZE))
        det.is_changed(gray)
        times_ms.append((time.perf_counter() - t0) * 1000)
    mean_ms = statistics.mean(times_ms)
    print(f"  gate cost: {mean_ms:.3f} ms/frame")
    return {"gate_ms_per_frame": mean_ms}


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------
def fig_vlm(vlm: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(vlm["dims_px"], vlm["latency_s"], "o-", color="#2f6f5e", lw=2)
    ax.axvline(cfg.VLM_MAX_IMAGE_DIM, ls="--", color="#b4452f")
    ax.annotate(f"frame cap = {cfg.VLM_MAX_IMAGE_DIM}px",
                xy=(cfg.VLM_MAX_IMAGE_DIM, max(vlm["latency_s"]) * 0.6),
                xytext=(cfg.VLM_MAX_IMAGE_DIM + 60, max(vlm["latency_s"]) * 0.6),
                color="#b4452f", fontsize=9)
    ax.set_title("VLM (Qwen2.5-VL-3B-4bit) latency vs image resolution")
    ax.set_xlabel("longest image dimension (px)")
    ax.set_ylabel("inference latency (s)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "vlm_latency_vs_resolution.png", dpi=130)
    plt.close(fig)


def fig_whisper(w: dict) -> None:
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(w["durations_s"], w["latency_s"], "o-", color="#2f6f5e", lw=2, label="latency")
    ax1.set_xlabel("audio clip duration (s)")
    ax1.set_ylabel("transcription latency (s)", color="#2f6f5e")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(w["durations_s"], w["rtf"], "s--", color="#d98324", lw=1.6, label="RTF")
    ax2.axhline(1.0, ls=":", color="#888")
    ax2.set_ylabel("real-time factor (proc/audio)", color="#d98324")
    ax2.set_ylim(0, max(1.1, max(w["rtf"]) * 1.3))
    ax1.set_title("Whisper (base.en) latency & real-time factor")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "whisper_latency_vs_duration.png", dpi=130)
    plt.close(fig)


def fig_gate(vlm: dict, gate: dict) -> None:
    vga_vlm_ms = vlm["latency_s"][vlm["dims_px"].index(768)] * 1000
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(["change detector\n(per frame)", "VLM call\n(per scene)"],
                  [gate["gate_ms_per_frame"], vga_vlm_ms],
                  color=["#2f6f5e", "#b4452f"])
    ax.set_yscale("log")
    ax.set_ylabel("cost per operation (ms, log scale)")
    ax.set_title("Why gate the VLM: cheap detector vs expensive VLM")
    for b, v in zip(bars, [gate["gate_ms_per_frame"], vga_vlm_ms]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f} ms",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gate_vs_vlm_cost.png", dpi=130)
    plt.close(fig)


def main() -> None:
    results = {}
    results["whisper"] = bench_whisper()
    results["vlm"] = bench_vlm()
    results["gate"] = bench_gate()

    fig_whisper(results["whisper"])
    fig_vlm(results["vlm"])
    fig_gate(results["vlm"], results["gate"])

    (FIG_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote figures + results.json to {FIG_DIR}")


if __name__ == "__main__":
    main()
