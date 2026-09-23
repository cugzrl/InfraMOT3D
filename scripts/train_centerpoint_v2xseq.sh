#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BATCH_SIZE="${1:-2}"
EPOCHS="${2:-30}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
conda run -n track python -c 'import sys; sys.path.insert(0, "'"${ROOT}"'/src"); from inframot3d.detection.config_check import check_centerpoint_configs; check_centerpoint_configs("'"${ROOT}"'")'
cd "${ROOT}/third_party/OpenPCDet/tools"
export CUDA_VISIBLE_DEVICES=0
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export PYTHONPATH="${ROOT}/third_party/OpenPCDet:${PYTHONPATH:-}"
conda run -n track --no-capture-output python train.py \
  --cfg_file cfgs/v2x_seq_models/centerpoint.yaml \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --workers 4 \
  --launcher none
