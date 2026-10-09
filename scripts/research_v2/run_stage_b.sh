#!/usr/bin/env bash
# 阶段 B 批量实验：PIPE GPU PHASE [QUERY_ARGS]
# b1：只在管线 A 上比较训练 Query 来源
# b2：基线 + 各变体 3 个种子，QUERY_ARGS 为 B1 选出的训练方式，例如 "--query grae --perturb"
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
P="$1"
GPU="$2"
PHASE="$3"
QARGS="${4:---query grae}"
SPLIT="${SPLIT:-holdout}"
JOBS="${JOBS:-3}"
source /home/kemove/anaconda3/etc/profile.d/conda.sh
conda activate track
cd "${ROOT}"
export OMP_NUM_THREADS=4
LIST=$(mktemp)
if [[ "${PHASE}" == "b1" ]]; then
  for s in 0 1; do
    echo "--variant hist_bev --query gt --seed $s" >> "${LIST}"
    echo "--variant hist_bev --query grae --seed $s" >> "${LIST}"
    echo "--variant hist_bev --query grae --perturb --seed $s" >> "${LIST}"
  done
elif [[ "${PHASE}" == "b2" ]]; then
  for v in raw nms iso_cls; do echo "--variant $v" >> "${LIST}"; done
  for s in 0 1 2; do
    for v in cand geom hist bev hist_bev hist_bev_nosite; do echo "--variant $v ${QARGS} --seed $s" >> "${LIST}"; done
  done
fi
cat "${LIST}" | xargs -P "${JOBS}" -I{} bash -c "python scripts/research_v2/run_condition.py --pipeline ${P} --gpu ${GPU} --split ${SPLIT} {} || echo FAILED {}"
rm -f "${LIST}"
echo "stage ${PHASE} pipeline ${P} split ${SPLIT} done"
