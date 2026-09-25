# InfraMOT3D

路侧单激光雷达3D多目标跟踪工程。当前正式benchmark使用同一份CenterPoint检测，在V2X-Seq验证集上比较AB3DMOT、SimpleTrack、ImmortalTracker和GRAE-3DMOT。

V2X-Seq正式评估遵循DAIR-V2X官方tracking protocol。MOTA、MOTP、AMOTA、AMOTP、IDSW与官方定义一致。IDF1、FM、FP、FN额外在相同best score threshold和相同数据范围下报告。

## 目录结构

```text
InfraMOT3D
├── configs
│   ├── datasets
│   ├── detectors/centerpoint
│   ├── trackers
│   └── experiments
├── data
│   ├── v2x-seq-infrastructure
│   ├── converted
│   └── centerpoint_v2xseq
├── src/inframot3d
│   ├── detection
│   ├── tracking
│   └── evaluation
│       ├── evaluator.py
│       ├── metrics.py
│       └── protocols
├── scripts
│   ├── setup
│   ├── data
│   ├── detection
│   ├── tracking
│   ├── evaluation
│   ├── experiments
│   └── visualization
├── tests/evaluation
├── third_party
│   ├── OpenPCDet
│   ├── GRAE-3DMOT
│   └── DAIR-V2X
└── outputs
```

统一输入和预测都是现有JSONL。新增数据集时只增加Protocol、数据集配置和实验配置。

## 环境

使用已有conda环境`track`。进入仓库后：

```bash
export PYTHONPATH=$PWD/src
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

OpenPCDet安装见`docs/centerpoint_setup.md`。

```bash
bash scripts/setup/setup_openpcdet.sh
bash scripts/setup/setup_grae.sh
bash scripts/setup/setup_dair_v2x.sh
```

`third_party/DAIR-V2X`只作为V2X-Seq协议的对照实现，正式benchmark不再另出一张官方结果表。

## 数据准备

`data/v2x-seq-infrastructure`指向原始路侧数据。序列划分保持train 46、val 21、test 0，见`configs/datasets/v2xseq_sequence_split.json`。

```bash
conda run -n track python scripts/data/prepare_data.py --config configs/trackers/ab3dmot/gt.yaml
conda run -n track python scripts/data/prepare_centerpoint_data.py --config configs/detectors/centerpoint/v2xseq.yaml
```

## CenterPoint训练

检测配置是`configs/detectors/centerpoint/v2xseq.yaml`。当前权重是epoch 30，路径写在`outputs/centerpoint/detection_manifest.json`。不要在已完成的one-cycle训练上继续训练。

```bash
bash scripts/detection/train_centerpoint_v2xseq.sh 2 30
conda run -n track python scripts/detection/infer_centerpoint_v2xseq.py \
  --config configs/detectors/centerpoint/v2xseq.yaml \
  --split all \
  --score-threshold 0.01 \
  --skip-eval
```

共享检测池是`outputs/centerpoint/detections`，导出阈值保持`score>=0.01`。跟踪器自己决定如何使用这些分数。

## Tracker运行

AB3DMOT、SimpleTrack、ImmortalTracker读取同一份检测。类别分数只用于新建轨迹，配置在`configs/experiments/centerpoint_score_thresholds.yaml`。

```bash
conda run -n track python scripts/tracking/run_tracker.py --config configs/trackers/ab3dmot/centerpoint.yaml --split val
conda run -n track python scripts/tracking/run_tracker.py --config configs/trackers/simpletrack/centerpoint.yaml --split val
conda run -n track python scripts/tracking/run_tracker.py --config configs/trackers/immortal/centerpoint.yaml --split val
```

使用标注做输入时，把配置换成对应的`gt.yaml`，或直接运行`scripts/tracking/run_ab3dmot_gt.sh`。

## GRAE训练

GRAE配置是`configs/trackers/grae/centerpoint.yaml`。训练检测阈值`train.score_threshold=0.1`，推理`tracker.score_floor=0.1`，低于0.1的CenterPoint检测在GRAE内部丢弃。`association_alpha=0.24`只划分官方high/low两阶段关联。类别birth阈值只负责新建轨迹。

```bash
conda run -n track python scripts/data/prepare_grae_v2xseq.py --config configs/trackers/grae/centerpoint.yaml --split train
bash scripts/tracking/train_grae_v2xseq.sh
conda run -n track python scripts/tracking/select_grae_checkpoint.py --config configs/trackers/grae/centerpoint.yaml
conda run -n track python scripts/tracking/infer_grae_v2xseq.py --config configs/trackers/grae/centerpoint.yaml --split val
```

当前使用的权重是`outputs/grae_centerpoint/ckpt/checkpoint-best.pth`。推理不会重新训练。

## 统一evaluation

正式评估只运行`UnifiedMOTEvaluator`。`V2XSeqProtocol`把Car、Van、Bus、Truck合并为Car，范围是`[0,-39.68,-3,100,39.68,1]`，3D IoU阈值是0.25，recall工作点是41个。MOTA和MOTP使用与DAIR官方相同的best single score threshold。AMOTA和AMOTP按官方recall曲线平均。IDSW、IDF1、FM、FP、FN在同一个threshold上计算。过滤使用轨迹平均分，GRAE低分coasted track会被该threshold去掉。

```bash
conda run -n track python scripts/evaluation/evaluate_mot.py --config configs/trackers/ab3dmot/centerpoint.yaml --split val
```

每个Tracker写出`outputs/<method>/evaluation/metrics.json`。回归测试：

```bash
conda run -n track python -u tests/evaluation/test_v2xseq_official_parity.py
```

同一份预测上，MOTA、MOTP、AMOTA、AMOTP、IDSW、FP、FN、FM与DAIR官方Evaluator的误差小于1e-6。把GT当作预测时，MOTA和MOTP接近1，IDSW为0。

## 完整benchmark

```bash
bash scripts/experiments/run_v2xseq_centerpoint.sh
```

脚本先检查parity，再依次运行四个Tracker和统一评估，写出`outputs/benchmark/v2xseq_centerpoint.csv`与`outputs/benchmark/experiment_manifest.json`。

BEV和Open3D可视化入口在`scripts/visualization`，Open3D说明见`docs/open3d_visualization.md`。
