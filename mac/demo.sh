#!/usr/bin/env bash
# demo.sh — one command to bring VoiceLink up for a demo or a video shoot.
#
#   bash mac/demo.sh              # real wearable (ws://192.168.4.1:81/), real models
#   bash mac/demo.sh --fake       # fake_wearable.py stands in for the ESP32-S3
#   bash mac/demo.sh --student    # serve the distilled student (torch) instead of the LLM
#   bash mac/demo.sh --intel      # student + MiniLM + Whisper all on OpenVINO
#   bash mac/demo.sh --mock       # no models at all (mock perception, heuristic ranker)
#
# Flags combine (e.g. --fake --intel). Ctrl-C stops everything it started.
# The interpreter is pinned because `python3` on this machine can resolve to
# the PlatformIO venv, which has none of the dependencies.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${VOICELINK_PYTHON:-/usr/local/bin/python3}"
command -v "$PY" >/dev/null || PY=python3

export TOKENIZERS_PARALLELISM=false
FAKE=0
for arg in "$@"; do
  case "$arg" in
    --fake)    FAKE=1; export VOICELINK_WEARABLE_URL="ws://127.0.0.1:8090/" ;;
    --student) export VOICELINK_BOARD_STRATEGY=student ;;
    --intel)   export VOICELINK_BOARD_STRATEGY=student VOICELINK_STUDENT_BACKEND=openvino \
                      VOICELINK_EMBED_BACKEND=openvino VOICELINK_TRANSCRIBER=openvino ;;
    --mock)    export VOICELINK_RANKER=heuristic VOICELINK_TRANSCRIBER=mock \
                      VOICELINK_SCENE_DESCRIBER=mock ;;
    -h|--help) sed -n 2,12p "$0"; exit 0 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

cd "$HERE"
if [ "$FAKE" = 1 ]; then
  "$PY" fake_wearable.py & PIDS+=($!)
  sleep 1
fi
echo "[demo] strategy=${VOICELINK_BOARD_STRATEGY:-diverse} student=${VOICELINK_STUDENT_BACKEND:-torch}" \
     "embed=${VOICELINK_EMBED_BACKEND:-torch} transcriber=${VOICELINK_TRANSCRIBER:-faster-whisper}" \
     "wearable=${VOICELINK_WEARABLE_URL:-ws://192.168.4.1:81/}"
exec "$PY" server.py
