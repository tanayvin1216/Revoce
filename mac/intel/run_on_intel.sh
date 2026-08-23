#!/usr/bin/env bash
# run_on_intel.sh — everything needed to produce Intel-hardware numbers on a
# fresh Intel box (Tiber / Intel AI Cloud Xeon, a Core Ultra laptop, an Arc box).
#
#   git clone <repo> && cd VoiceLink && bash mac/intel/run_on_intel.sh
#
# Produces, self-labelled with the CPU brand and the OpenVINO devices seen:
#   mac/intel/artifacts/e2e_<host>.json           end-to-end board latency, 3 backends
#   mac/intel/artifacts/whisper_bench_<host>.json  whisper faster-whisper vs OpenVINO GenAI
#   mac/intel/BENCHMARKS.md                        appended sections for the above
# Optional: VOICELINK_OV_DEVICE=NPU (Core Ultra) or GPU (Arc) re-runs the head there.
# Nothing here needs the wearable, a model API, or network beyond pip/HF downloads.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAC="$(dirname "$HERE")"
PY="${VOICELINK_PYTHON:-python3}"
export VOICELINK_PERCEPTION=0 TOKENIZERS_PARALLELISM=false

cd "$MAC"
echo "== 0. environment"
"$PY" -m pip install -q -r requirements.txt -r requirements-intel.txt
"$PY" - <<'PYEOF'
import json, sys
sys.path.insert(0, ".")
from intel import hardware_fingerprint
hw = hardware_fingerprint()
print(json.dumps(hw, indent=2))
if not hw["intel_hardware"]:
    print("WARNING: this is not Intel silicon; numbers will be labelled accordingly")
PYEOF

echo "== 1. artifacts (student IR is committed; MiniLM + Whisper IRs are regenerated)"
[ -f intel/artifacts/minilm-ov/openvino_model.bin ] || "$PY" intel/convert.py
[ -f intel/artifacts/whisper-base.en-ov-int8/openvino_encoder_model.xml ] || "$PY" intel/export_whisper.py

echo "== 2. student head micro-benchmark (torch vs OV FP32 vs OV INT8)"
"$PY" intel/bench.py --skip-benchmark-app >/dev/null

echo "== 3. end-to-end board latency through the seam"
"$PY" intel/bench_e2e.py --n 84
if [ -n "${VOICELINK_OV_DEVICE:-}" ] && [ "${VOICELINK_OV_DEVICE}" != "CPU" ]; then
  echo "== 3b. head on ${VOICELINK_OV_DEVICE}"
  "$PY" intel/bench_e2e.py --n 84 --device "${VOICELINK_OV_DEVICE}" --backends ov-head,ov-full
fi

echo "== 4. whisper"
"$PY" intel/bench_whisper.py || echo "whisper bench failed (see above) — not fatal"

echo "== 5. the seam still serves on this backend"
VOICELINK_STUDENT_BACKEND=openvino VOICELINK_EMBED_BACKEND=openvino \
  "$PY" -m pytest -q tests/test_intel.py tests/test_embed_backend.py tests/test_e2e_smoke.py

echo
echo "Done. Commit mac/intel/artifacts/e2e_*.json, whisper_bench_*.json and BENCHMARKS.md."
