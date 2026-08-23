#!/usr/bin/env bash
# run_on_gaudi.sh — the VoiceLink TRAINING-side lifecycle on an Intel Gaudi box.
#
# Gaudi is a PyTorch accelerator (OpenVINO does not run on it), so what belongs
# here is the teacher data generation (Qwen2.5-3B-Instruct through transformers)
# and the student distillation — the 'data' and 'modelling' stages. Deployment
# numbers come from intel/run_on_intel.sh on a Xeon / Core Ultra box instead.
#
# Versions below are the ones Intel's own quick start lists for its cloud
# instances (docs.habana.ai Intel_DevCloud_Quick_Start, driver 1.24.1, PyTorch
# 2.11). On a cloud instance the driver is preinstalled; verify with `hl-smi`.
#
#   git clone <repo> && cd VoiceLink && bash mac/intel/gaudi/run_on_gaudi.sh
#   ... or inside the container: bash mac/intel/gaudi/run_on_gaudi.sh --inside
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
IMAGE="${GAUDI_IMAGE:-vault.habana.ai/gaudi-docker/1.24.1/ubuntu22.04/habanalabs/pytorch-installer-2.11.0:latest}"

if [ "${1:-}" != "--inside" ]; then
  command -v hl-smi >/dev/null && hl-smi | head -20 || echo "hl-smi not found — is the Gaudi driver installed? (see docs.habana.ai Installation Guide)"
  echo "== launching $IMAGE"
  exec docker run -it --rm --runtime=habana -e HABANA_VISIBLE_DEVICES=all \
      -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host \
      -v "$REPO":/work -v "${HF_HOME:-$HOME/.cache/huggingface}":/root/.cache/huggingface \
      -w /work "$IMAGE" bash mac/intel/gaudi/run_on_gaudi.sh --inside
fi

# ---- inside the container -------------------------------------------------
cd "$REPO/mac"
export VOICELINK_PERCEPTION=0 TOKENIZERS_PARALLELISM=false
export VOICELINK_DEVICE=hpu VOICELINK_RANKER=hf PT_HPU_LAZY_MODE=1
echo "== 0. environment"
python3 -c "import habana_frameworks.torch.core, torch; print('torch', torch.__version__, 'hpu available:', torch.hpu.is_available(), torch.hpu.get_device_name())"
pip install -q -r requirements-intel.txt sentence-transformers pyyaml aiohttp pytest 2>/dev/null || true
pip install -q "optimum-habana" 2>/dev/null || echo "optimum-habana not installed; plain transformers generate on hpu is used"

echo "== 1. teacher smoke: one board through the seam on hpu"
python3 - <<'PYEOF'
import sys, time; sys.path.insert(0, ".")
from ranker import HfRanker
import rank_vocabulary as rv
r = HfRanker(temp=0.0)
t0 = time.perf_counter()
board = rv.rank_vocabulary("A classroom. Ms. Lee stands at a whiteboard with a diagram.",
                           "Ms. Lee: What do plants take in from the air?",
                           {"last_utterance": "What do plants take in from the air?", "addressed": True},
                           ranker=r, strategy="diverse")
print(f"device={r.device} model={r.model_id} {time.perf_counter()-t0:.1f}s")
print([w["word"] for w in board])
PYEOF

echo "== 2. teacher data for the deictic set (the documented next gap) — appends to teacher_data.jsonl only if --gen is passed"
if [ "${2:-}" = "--gen" ]; then python3 evaluation/phase4/gen_teacher.py; fi

echo "== 3. distil the student on hpu, export ONNX (CPU), run the gates"
python3 evaluation/phase4/train.py
python3 evaluation/phase4/evaluate.py --teacher-samples 0
python3 -m pytest -q tests/test_phase4.py tests/test_seam.py

echo
echo "Done. Artifacts: evaluation/phase4/student.pt, student.onnx, phase4_results.json (training device recorded in the log above)."
echo "Next: copy student.onnx to a Xeon/Core Ultra box and run intel/run_on_intel.sh for deployment numbers."
