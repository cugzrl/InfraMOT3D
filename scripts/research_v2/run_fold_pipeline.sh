#!/usr/bin/env bash
# 交叉拟合数据管线：检测器 F 只在 fold F 上训练，在另一折与 val 上导出检测、GRAE 回放、BEV 缓存与样本
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
F="$1"
GPU="${2:-0}"
if [[ "$F" == "A" ]]; then O="B"; else O="A"; fi
source /home/kemove/anaconda3/etc/profile.d/conda.sh
conda activate track
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${ROOT}/src:${ROOT}/third_party/GRAE-3DMOT:${ROOT}/third_party/OpenPCDet"
R=outputs/research/roadside_joint_perception_v2
CKPT=third_party/OpenPCDet/output/v2x_seq_models/centerpoint_fold${F}/default/ckpt/checkpoint_epoch_30.pth
OOF=$(python -c "import json;print(' '.join(json.load(open('configs/research/v2xseq_train_folds.json'))['folds']['${O}']))")
VAL=$(python -c "import json;print(' '.join(json.load(open('configs/datasets/v2xseq_sequence_split.json'))['val']))")
DET=${R}/detections/det${F}
if [[ ! -f ${DET}/detection_manifest.json ]]; then
  python scripts/research_v2/infer_fold_detector.py --cfg third_party/OpenPCDet/tools/cfgs/v2x_seq_models/centerpoint_fold${F}.yaml --ckpt ${CKPT} --infos infos/v2x_seq_infos_fold${O}.pkl infos/v2x_seq_infos_val.pkl --output ${DET}
fi
python scripts/research_v2/diagnose_detections.py --detections ${DET} --sequences ${OOF} --name det${F}_oof_fold${O} --output ${R}/diagnostics/fold_detectors_${F}.json > /dev/null
python scripts/research_v2/diagnose_detections.py --detections ${DET} --sequences ${VAL} --name det${F}_val --output ${R}/diagnostics/fold_detectors_${F}.json > /dev/null
python scripts/research_v2/replay_grae.py --detections ${DET} --sequences ${OOF} --output ${R}/replays/det${F}_oof
python scripts/research_v2/replay_grae.py --detections ${DET} --sequences ${VAL} --output ${R}/replays/det${F}_val
python scripts/research_v2/cache_bev.py --cfg third_party/OpenPCDet/tools/cfgs/v2x_seq_models/centerpoint_fold${F}.yaml --ckpt ${CKPT} --output ${R}/bev_cache/det${F} --fit-sequences ${OOF} --sequences ${OOF} ${VAL}
python scripts/research_v2/build_samples.py --detections ${DET} --replay ${R}/replays/det${F}_oof --sequences ${OOF} --output ${R}/samples/det${F}
python scripts/research_v2/build_samples.py --detections ${DET} --replay ${R}/replays/det${F}_val --sequences ${VAL} --output ${R}/samples/det${F}
python scripts/research_v2/query_distribution.py --detections ${DET} --replay ${R}/replays/det${F}_oof --sequences ${OOF} --name det${F}_oof_fold${O} --output ${R}/diagnostics/query_distribution_${F}.json > /dev/null
python scripts/research_v2/query_distribution.py --detections ${DET} --replay ${R}/replays/det${F}_val --sequences ${VAL} --name det${F}_val --output ${R}/diagnostics/query_distribution_${F}.json > /dev/null
echo pipeline ${F} done
