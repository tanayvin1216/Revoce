#!/usr/bin/env bash
# setup-and-run.sh — one shot for the Task 6 baseline.
#
# Verifies you can reach the GATED willwade/AACConversations dataset, then runs
# the vocabulary-prediction baseline on the held-out TEST split and prints
# top-k recall (with the synthetic-data caveat).
#
# Usage:
#   bash mac/evaluation/setup-and-run.sh           # default 500 examples (~25 min)
#   bash mac/evaluation/setup-and-run.sh 200        # fewer examples (~10 min)
#
# Your hands-on part is ~5 min (accept gate + token); the rest is the local LLM
# churning ~3s/example, unattended. Nothing leaves the Mac except the dataset
# download.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

LIMIT="${1:-500}"
DATASET="willwade/AACConversations"
GATE_URL="https://huggingface.co/datasets/${DATASET}"
TOKENS_URL="https://huggingface.co/settings/tokens"

echo "VoiceLink — Task 6 baseline"
echo "  repo:  $REPO_ROOT"
echo "  limit: $LIMIT examples (~$(( LIMIT * 3 / 60 )) min of local LLM)"
echo

# 1) dependencies present?
python3 - <<'PY' || { echo "Missing deps. Run: pip install -r mac/requirements.txt datasets"; exit 1; }
import importlib
for m in ("datasets", "huggingface_hub", "mlx_lm"):
    importlib.import_module(m)
print("  deps ok")
PY

# 2) can we actually reach the gated data? (auth + gate acceptance)
echo "  checking dataset access..."
if ! python3 - "$DATASET" <<'PY'
import sys
from datasets import load_dataset
try:
    ds = load_dataset(sys.argv[1], split="test", streaming=True)
    next(iter(ds))           # pull one row -> forces auth + gate check
    print("  access ok")
except Exception as e:
    print("ACCESS_FAIL:", str(e)[:300], file=sys.stderr)
    sys.exit(3)
PY
then
    cat <<EOF

  ✗ Can't reach $DATASET yet. Two one-time steps:

    1) ACCEPT THE GATE — in a browser, signed into your Hugging Face account:
         open $GATE_URL
       click "Agree and access repository" (instant; it's auto-approved).

    2) AUTHENTICATE this Mac — pick ONE:
         hf auth login                  # paste a token when prompted
       or
         export HF_TOKEN=hf_xxxxxxxx     # then re-run this script

       Need a token? Create one at:  $TOKENS_URL
       Use a classic "Read" token, OR a fine-grained token with
       "Read access to contents of all public gated repos" enabled —
       a plain fine-grained token returns 403 on gated datasets.

  Then re-run:  bash mac/evaluation/setup-and-run.sh $LIMIT
EOF
    exit 1
fi

# 3) run the baseline (model is local + already cached; this is the slow part)
echo
echo "  access ok — running baseline on $LIMIT held-out examples..."
echo
exec python3 mac/evaluation/harness.py --source dataset --limit "$LIMIT" --ranker local-llm
