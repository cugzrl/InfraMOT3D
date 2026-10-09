#!/usr/bin/env bash
# RESOLVE 单分辨率 CenterPoint，不占用已有 V2X-Seq 输出目录
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RES="${1:?128|64|16}"
GPU="${2:-1}"
source /home/kemove/anaconda3/etc/profile.d/conda.sh
conda activate track
cd "${ROOT}/third_party/OpenPCDet/tools"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export PYTHONPATH="${ROOT}/third_party/OpenPCDet:${PYTHONPATH:-}"
python train.py \
  --cfg_file "cfgs/resolve_models/centerpoint_res${RES}.yaml" \
  --batch_size 2 \
  --epochs 20 \
  --workers 4 \
  --fix_random_seed \
  --max_ckpt_save_num 3 \
  --launcher none
