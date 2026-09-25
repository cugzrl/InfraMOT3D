#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "${ROOT}"
conda run -n track --no-capture-output python -u scripts/tracking/train_3dmotformer_v2xseq.py \
  --config configs/trackers/3dmotformer/centerpoint.yaml
