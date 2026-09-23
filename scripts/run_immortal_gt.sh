#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT/src"

conda run -n track python scripts/prepare_data.py --config configs/immortal_gt.yaml
conda run -n track python scripts/run_tracker.py --config configs/immortal_gt.yaml
conda run -n track python scripts/evaluate.py --config configs/immortal_gt.yaml
