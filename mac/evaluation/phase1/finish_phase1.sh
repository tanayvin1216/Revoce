#!/usr/bin/env bash
# Run after gen_phase1.py finishes: score the cached generations, run the
# production simulation (needs the model), render figures, write the report.
set -e
cd "$(dirname "$0")/../.."   # -> mac/
echo "=== scoring ==="
python3 evaluation/phase1/score_phase1.py
echo "=== production simulation ==="
python3 evaluation/phase1/sim.py
echo "=== figures ==="
python3 evaluation/phase1/plots.py
echo "=== report ==="
python3 evaluation/phase1/report.py
echo "PHASE 1 DONE"
