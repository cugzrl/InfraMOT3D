#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src:${ROOT}/third_party/GRAE-3DMOT:${PYTHONPATH:-}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
cd "${ROOT}"
THRESHOLD="${ROOT}/configs/centerpoint_score_thresholds.yaml"
if [[ ! -f "${THRESHOLD}" ]]; then
  echo "缺少 ${THRESHOLD}，先运行 scripts/sweep_tracking_threshold.py"
  exit 1
fi
if [[ ! -f "${ROOT}/outputs/grae_centerpoint/ckpt/checkpoint-best.pth" ]]; then
  echo "缺少GRAE best checkpoint，先运行 scripts/train_grae_v2xseq.sh"
  exit 1
fi
echo "读取阈值 ${THRESHOLD}"
for name in ab3dmot simpletrack immortal; do
  conda run -n track --no-capture-output python -u scripts/run_tracker.py --config "configs/${name}_centerpoint.yaml" --split val
  conda run -n track --no-capture-output python -u scripts/evaluate.py --config "configs/${name}_centerpoint.yaml" --split val
  conda run -n track --no-capture-output python -u scripts/export_v2xseq_official_tracking.py \
    --config "configs/${name}_centerpoint.yaml" \
    --split val \
    --prediction-root "outputs/${name}_centerpoint/predictions" \
    --output "outputs/${name}_centerpoint/official_kitti"
  conda run -n track --no-capture-output python -u scripts/evaluate_v2xseq_official.py \
    --exported "outputs/${name}_centerpoint/official_kitti" \
    --output "outputs/${name}_centerpoint/official_metrics.json" \
    --name "${name}_centerpoint"
done
conda run -n track --no-capture-output python -u scripts/infer_grae_v2xseq.py --config configs/grae_centerpoint.yaml --split val
conda run -n track --no-capture-output python -u scripts/evaluate.py --config configs/grae_centerpoint.yaml --split val
conda run -n track --no-capture-output python -u scripts/export_v2xseq_official_tracking.py \
  --config configs/grae_centerpoint.yaml \
  --split val \
  --prediction-root outputs/grae_centerpoint/predictions \
  --output outputs/grae_centerpoint/official_kitti
conda run -n track --no-capture-output python -u scripts/evaluate_v2xseq_official.py \
  --exported outputs/grae_centerpoint/official_kitti \
  --output outputs/grae_centerpoint/official_metrics.json \
  --name grae_centerpoint
conda run -n track --no-capture-output python -u scripts/compare_trackers.py --preset centerpoint --output outputs/centerpoint_comparison.csv
