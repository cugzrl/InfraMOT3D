#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT/src"

conda run -n track python scripts/data/prepare_data.py --config configs/trackers/ab3dmot/gt.yaml
conda run -n track python scripts/tracking/run_tracker.py --config configs/trackers/ab3dmot/gt.yaml
conda run -n track python scripts/evaluation/evaluate_mot.py --config configs/trackers/ab3dmot/gt.yaml
