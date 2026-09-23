#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src:${ROOT}/third_party/GRAE-3DMOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
cd "${ROOT}"
conda run -n track --no-capture-output python -u scripts/train_grae_v2xseq.py --config configs/grae_centerpoint.yaml
