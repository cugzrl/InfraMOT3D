#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
cd "${ROOT}"
for name in ab3dmot simpletrack immortal; do
  conda run -n track --no-capture-output python scripts/run_tracker.py --config "configs/${name}_centerpoint.yaml" --split val
  conda run -n track --no-capture-output python scripts/evaluate.py --config "configs/${name}_centerpoint.yaml" --split val
done
conda run -n track --no-capture-output python scripts/compare_trackers.py --preset centerpoint_classic --output outputs/centerpoint_classic_comparison.csv
