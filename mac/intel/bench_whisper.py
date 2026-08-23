"""Benchmark faster-whisper (int8, CPU) vs openvino_genai.WhisperPipeline (INT8 IR).

    VOICELINK_PERCEPTION=0 python3 intel/bench_whisper.py [--n 10] [--clip a.wav ...]

Same clip(s), both backends, N timed runs after a warm-up; prints p50/p90 and the
transcripts, writes artifacts/whisper_bench_<host>.json (with
intel.hardware_fingerprint()) and APPENDS a '## Whisper' section to BENCHMARKS.md.
Default clip: a macOS `say` sentence synthesized into the scratchpad (real speech,
reproducible). Honesty: on Apple Silicon this is the OpenVINO ARM CPU plugin.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MAC = HERE.parent
if str(MAC) not in sys.path:
    sys.path.insert(0, str(MAC))

from intel import WHISPER_OV_DIR, hardware_fingerprint  # noqa: E402
from perception.transcriber import (  # noqa: E402
    FasterWhisperTranscriber, OpenVinoWhisperTranscriber,
)

BENCHMARKS_MD = HERE / "BENCHMARKS.md"
SCRATCH = Path(os.environ.get(
    "VOICELINK_SCRATCH",
    "/private/tmp/claude-501/-Users-tanayvinaykya-VoiceLink/"
    "f03806ca-81fa-49f4-aa5a-2ef28f0f6222/scratchpad"))
DEFAULT_SENTENCES = [
    "Do you want some water or a cup of coffee?",
    "We are going to the park after lunch, bring your jacket because it is cold.",
]


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1, path
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def default_clips() -> list[Path]:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    out = []
    for i, s in enumerate(DEFAULT_SENTENCES):
        wav = SCRATCH / f"bench_clip{i}.wav"
        if not wav.exists():
            aiff = wav.with_suffix(".aiff")
            subprocess.run(["say", "-o", str(aiff), s], check=True)
            subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                            str(aiff), str(wav)], check=True)
        out.append(wav)
    return out


def time_backend(tr, pcm: np.ndarray, n: int) -> dict:
    tr.transcribe(pcm, 16000)  # warm-up
    samples, text = [], ""
    for _ in range(n):
        t0 = time.perf_counter()
        text = tr.transcribe(pcm, 16000).text
        samples.append((time.perf_counter() - t0) * 1000)
    return {"p50_ms": float(np.percentile(samples, 50)),
            "p90_ms": float(np.percentile(samples, 90)),
            "min_ms": float(min(samples)), "n": n, "text": text}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--clip", type=Path, nargs="*", help="16 kHz mono wav(s)")
    a = ap.parse_args()
    clips = a.clip or default_clips()

    fw = FasterWhisperTranscriber("base.en", "cpu", "int8")
    ov = OpenVinoWhisperTranscriber(WHISPER_OV_DIR)
    fp = hardware_fingerprint()
    rows = []
    for c in clips:
        pcm = load_wav(c)
        dur = len(pcm) / 16000
        r_fw = time_backend(fw, pcm, a.n)
        r_ov = time_backend(ov, pcm, a.n)
        rows.append({"clip": c.name, "audio_s": dur, "faster_whisper": r_fw,
                     "openvino_genai": r_ov,
                     "same_text": r_fw["text"].strip().lower() == r_ov["text"].strip().lower()})
        print(f"{c.name} ({dur:.1f}s): faster-whisper p50 {r_fw['p50_ms']:.0f} ms | "
              f"openvino p50 {r_ov['p50_ms']:.0f} ms | same text: {rows[-1]['same_text']}")
        print(f"   fw: {r_fw['text']!r}\n   ov: {r_ov['text']!r}")

    host = platform.node().split(".")[0] or "host"
    out = HERE / "artifacts" / f"whisper_bench_{host}.json"
    out.write_text(json.dumps({"hardware": fp, "ov_device": ov.device, "n": a.n,
                               "model": "whisper-base.en (both int8)", "rows": rows}, indent=2))
    print(f"wrote {out}")

    md = [f"\n## Whisper — {fp['cpu']}\n",
          f"`intel/bench_whisper.py --n {a.n}` | {fp['os']} | OpenVINO {fp['openvino']} | "
          f"devices {fp['ov_devices']} | OV device {ov.device} | "
          f"Intel hardware: {'yes' if fp['intel_hardware'] else 'no'}\n",
          "faster-whisper base.en int8 (CTranslate2, CPU) vs openvino_genai.WhisperPipeline "
          "base.en INT8 IR (INFERENCE_PRECISION_HINT=f32 on CPU). Real speech, macOS `say`.\n",
          "| clip | audio (s) | faster-whisper p50 (ms) | p90 | openvino p50 (ms) | p90 | same text |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['clip']} | {r['audio_s']:.1f} | {r['faster_whisper']['p50_ms']:.0f} | "
                  f"{r['faster_whisper']['p90_ms']:.0f} | {r['openvino_genai']['p50_ms']:.0f} | "
                  f"{r['openvino_genai']['p90_ms']:.0f} | {'yes' if r['same_text'] else 'no'} |")
    with BENCHMARKS_MD.open("a") as f:   # append only — never rewrite earlier sections
        f.write("\n".join(md) + "\n")
    print(f"appended '## Whisper' to {BENCHMARKS_MD}")


if __name__ == "__main__":
    main()
