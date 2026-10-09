#!/usr/bin/env bash
# 等待交叉拟合 CenterPoint 写完 epoch 30，再导出 out-of-fold 检测与样本
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
A="${ROOT}/third_party/OpenPCDet/output/v2x_seq_models/centerpoint_foldA/default/ckpt/checkpoint_epoch_30.pth"
B="${ROOT}/third_party/OpenPCDet/output/v2x_seq_models/centerpoint_foldB/default/ckpt/checkpoint_epoch_30.pth"
LOCK="${ROOT}/outputs/research/roadside_joint_perception_v2/logs/fold_pipeline.lock"
LOG="${ROOT}/outputs/research/roadside_joint_perception_v2/logs/fold_pipeline.log"
source /home/kemove/anaconda3/etc/profile.d/conda.sh
conda activate track
cd "${ROOT}"
while [[ ! -f "${A}" || ! -f "${B}" ]]; do
  sleep 60
done
if [[ -f "${LOCK}" ]]; then
  echo "fold pipeline already started" | tee -a "${LOG}"
  exit 0
fi
date > "${LOCK}"
echo "fold ckpts ready $(date)" | tee -a "${LOG}"
while pgrep -f run_insample_diag.py >/dev/null; do
  echo "wait insample diag $(date)" | tee -a "${LOG}"
  sleep 30
done
# 两折检测器可并行，A 用 GPU 0，B 用 GPU 1
bash scripts/research_v2/run_fold_pipeline.sh A 0 >> "${LOG}" 2>&1 &
PIDA=$!
bash scripts/research_v2/run_fold_pipeline.sh B 1 >> "${LOG}" 2>&1 &
PIDB=$!
wait "${PIDA}"
wait "${PIDB}"
echo "fold pipelines done $(date)" | tee -a "${LOG}"
