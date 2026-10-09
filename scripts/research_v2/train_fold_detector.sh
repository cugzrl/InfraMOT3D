#!/usr/bin/env bash
# 训练两折交叉拟合 CenterPoint，参数 fold 与 GPU 编号，原始 centerpoint 输出目录不受影响
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FOLD="$1"
GPU="${2:-0}"
source /home/kemove/anaconda3/etc/profile.d/conda.sh
conda activate track
cd "${ROOT}/third_party/OpenPCDet/tools"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export PYTHONPATH="${ROOT}/third_party/OpenPCDet:${PYTHONPATH:-}"
python train.py \
  --cfg_file "cfgs/v2x_seq_models/centerpoint_fold${FOLD}.yaml" \
  --batch_size 2 \
  --epochs 30 \
  --workers 4 \
  --fix_random_seed \
  --max_ckpt_save_num 3 \
  --launcher none
