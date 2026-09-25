#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT}/src:${ROOT}/third_party/GRAE-3DMOT:${PYTHONPATH:-}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "${ROOT}"
START_SECONDS="$(date +%s)"
echo "先运行官方parity"
mkdir -p "${ROOT}/outputs/official_parity"
conda run -n track --no-capture-output python -u tests/evaluation/test_v2xseq_official_parity.py | tee "${ROOT}/outputs/official_parity/parity_run.log"
grep -q '^parity_ok$' "${ROOT}/outputs/official_parity/parity_run.log"
for name in ab3dmot simpletrack immortal; do
  conda run -n track --no-capture-output python -u scripts/tracking/run_tracker.py --config "configs/trackers/${name}/centerpoint.yaml" --split val
  conda run -n track --no-capture-output python -u scripts/evaluation/evaluate_mot.py --config "configs/trackers/${name}/centerpoint.yaml" --split val
done
conda run -n track --no-capture-output python -u scripts/tracking/infer_grae_v2xseq.py --config configs/trackers/grae/centerpoint.yaml --split val
conda run -n track --no-capture-output python -u scripts/evaluation/evaluate_mot.py --config configs/trackers/grae/centerpoint.yaml --split val
conda run -n track --no-capture-output python -u scripts/evaluation/collect_benchmark.py --experiment configs/experiments/v2xseq_centerpoint.yaml
ELAPSED="$(( $(date +%s) - START_SECONDS ))"
conda run -n track --no-capture-output python -u scripts/experiments/write_experiment_manifest.py --elapsed-seconds "${ELAPSED}"
