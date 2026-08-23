"""Export openai/whisper-base.en to an INT8 OpenVINO IR for openvino_genai.WhisperPipeline.

    VOICELINK_PERCEPTION=0 python3 intel/export_whisper.py [--force]

Wraps optimum-cli. Writes to intel.WHISPER_OV_DIR (artifacts/whisper-base.en-ov-int8):
encoder/decoder IRs + openvino_tokenizer/detokenizer, which is exactly what
WhisperPipeline expects. The model is read from the HF cache when present.

Invocation order (first one that succeeds wins — optimum-intel's task names have
shifted across releases, and transformers 5.x changed Whisper's export path):
  1. --task automatic-speech-recognition-with-past  (the documented form)
  2. no --task                                       (let optimum infer it)
  3. --library transformers + --task automatic-speech-recognition
The winning invocation is recorded in <dir>/export_invocation.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAC = HERE.parent
if str(MAC) not in sys.path:
    sys.path.insert(0, str(MAC))

from intel import WHISPER_OV_DIR  # noqa: E402

MODEL_ID = "openai/whisper-base.en"
BASE = [sys.executable, "-m", "optimum.commands.optimum_cli", "export", "openvino",
        "--model", MODEL_ID, "--weight-format", "int8"]
CANDIDATES = [
    BASE + ["--task", "automatic-speech-recognition-with-past"],
    BASE,
    BASE + ["--library", "transformers", "--task", "automatic-speech-recognition"],
]
REQUIRED = ["openvino_encoder_model.xml", "openvino_decoder_model.xml",
            "openvino_tokenizer.xml", "openvino_detokenizer.xml"]


def exported_ok(out: Path) -> bool:
    return all((out / f).exists() for f in REQUIRED)


def export(out: Path = WHISPER_OV_DIR, force: bool = False) -> Path:
    if exported_ok(out) and not force:
        print(f"[export_whisper] already present: {out}")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    last_err = ""
    for cmd in CANDIDATES:
        if out.exists():
            shutil.rmtree(out)
        full = cmd + [str(out)]
        print("[export_whisper] $", " ".join(full))
        proc = subprocess.run(full, capture_output=True, text=True)
        if proc.returncode == 0 and exported_ok(out):
            (out / "export_invocation.json").write_text(json.dumps(
                {"cmd": full, "model": MODEL_ID}, indent=2))
            print(f"[export_whisper] OK -> {out}")
            return out
        last_err = (proc.stderr or proc.stdout)[-3000:]
        print(f"[export_whisper] failed (rc={proc.returncode}); trying next form\n{last_err}")
    raise RuntimeError(f"all optimum-cli invocations failed; last error:\n{last_err}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-export even if present")
    ap.add_argument("--out", type=Path, default=WHISPER_OV_DIR)
    a = ap.parse_args()
    export(a.out, a.force)
