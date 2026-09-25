# InfraMOT3D

路侧3D多目标跟踪：在V2X-Seq验证集上进行实验

## 目录结构

```text
InfraMOT3D
├── configs
│   ├── datasets
│   ├── detectors/centerpoint
│   ├── trackers
│   ├── analysis
│   └── experiments
├── data
│   ├── v2x-seq-infrastructure
│   ├── converted
│   └── centerpoint_v2xseq
├── src/inframot3d
│   ├── detection
│   ├── tracking
│   ├── analysis
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
│   ├── analysis
│   └── visualization
├── tests/evaluation
├── third_party
│   ├── OpenPCDet
│   ├── GRAE-3DMOT
│   ├── DAIR-V2X
│   ├── FastPoly
│   └── 3DMOTFormer
└── outputs
```

## 环境

```bash
export PYTHONPATH=$PWD/src
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

OpenPCDet安装见`docs/centerpoint_setup.md`。

```bash
bash scripts/setup/setup_openpcdet.sh
bash scripts/setup/setup_grae.sh
bash scripts/setup/setup_dair_v2x.sh
bash scripts/setup/setup_fastpoly.sh
bash scripts/setup/setup_3dmotformer.sh
```

## 数据准备

路侧数据序列划分保持train 46、val 21、test 0，见`configs/datasets/v2xseq_sequence_split.json`

```bash
conda run -n track python scripts/data/prepare_data.py --config configs/trackers/ab3dmot/gt.yaml
conda run -n track python scripts/data/prepare_centerpoint_data.py --config configs/detectors/centerpoint/v2xseq.yaml
```

## CenterPoint训练

检测配置`configs/detectors/centerpoint/v2xseq.yaml`

```bash
bash scripts/detection/train_centerpoint_v2xseq.sh 2 30
conda run -n track python scripts/detection/infer_centerpoint_v2xseq.py \
  --config configs/detectors/centerpoint/v2xseq.yaml \
  --split all \
  --score-threshold 0.01 \
  --skip-eval
```

共享检测结果`outputs/centerpoint/detections`，导出阈值保持`score>=0.01`
## Tracker运行


```bash
conda run -n track python scripts/tracking/run_tracker.py --config configs/trackers/ab3dmot/centerpoint.yaml --split val
conda run -n track python scripts/tracking/run_tracker.py --config configs/trackers/simpletrack/centerpoint.yaml --split val
conda run -n track python scripts/tracking/run_tracker.py --config configs/trackers/immortal/centerpoint.yaml --split val
```

## GRAE训练

GRAE配置`configs/trackers/grae/centerpoint.yaml`

```bash
conda run -n track python scripts/data/prepare_grae_v2xseq.py --config configs/trackers/grae/centerpoint.yaml --split train
bash scripts/tracking/train_grae_v2xseq.sh
conda run -n track python scripts/tracking/select_grae_checkpoint.py --config configs/trackers/grae/centerpoint.yaml
conda run -n track python scripts/tracking/infer_grae_v2xseq.py --config configs/trackers/grae/centerpoint.yaml --split val
```

最佳权重`outputs/grae_centerpoint/ckpt/checkpoint-best.pth`

## Fast-Poly

Fast-Poly是velocity-free V2X-Seq adaptation，不是原论文结果直接复现。CenterPoint没有velocity head，`has_velo`为false，推理速度只来自Fast-Poly自己的运动模型。官方Kalman只接受固定`LiDAR_interval`。帧间隔检查见`outputs/analysis/scene_difficulty/frame_interval.json`，中位数与0.1秒的误差不超过0.02秒，因此继续使用0.1秒。

`configs/trackers/fastpoly/centerpoint.yaml`保存基础V2X适配。0078、0087、0093、0094上的坐标下降写到`outputs/fastpoly_centerpoint/calibration/best_config.yaml`和`calibration_results.json`。正式benchmark读取`best_config`。

```bash
conda run -n track python scripts/tracking/tune_fastpoly_calibration.py
conda run -n track python scripts/tracking/run_tracker.py --config outputs/fastpoly_centerpoint/calibration/best_config.yaml --split val
```

## 3DMOTFormer

3DMOTFormer是velocity-free detector-input V2X-Seq adaptation，不是原论文结果直接复现。检测速度输入在训练和推理都是`[0,0]`，模型自己的velocity prediction正常训练和使用。检测池导出阈值是0.01，跟踪前使用`score_threshold=0.1`，训练和推理相同。GT track_id只用于训练监督。

训练clip里含空检测帧的比例低于1%，统计在`outputs/3dmotformer_centerpoint/data_stats.json`，因此继续删除这些clip。空帧比例若明显高于1%，则不能静默删除。

```bash
conda run -n track python scripts/data/prepare_3dmotformer_v2xseq.py --config configs/trackers/3dmotformer/centerpoint.yaml
bash scripts/tracking/train_3dmotformer_v2xseq.sh
conda run -n track python scripts/tracking/select_3dmotformer_checkpoint.py --config configs/trackers/3dmotformer/centerpoint.yaml
conda run -n track python scripts/tracking/infer_3dmotformer_v2xseq.py --config configs/trackers/3dmotformer/centerpoint.yaml --split val
```

## 统一evaluation

正式评估只运行`UnifiedMOTEvaluator`。`V2XSeqProtocol`把Car、Van、Bus、Truck合并为Car，范围是`[0,-39.68,-3,100,39.68,1]`，3D IoU阈值0.25，recall工作点41个

```bash
conda run -n track python scripts/evaluation/evaluate_mot.py --config configs/trackers/ab3dmot/centerpoint.yaml --split val
```

每个Tracker写出`outputs/<method>/evaluation/metrics.json`。回归测试：

```bash
conda run -n track python -u tests/evaluation/test_v2xseq_official_parity.py
```

## 完整benchmark

```bash
bash scripts/experiments/run_v2xseq_centerpoint.sh
```

六个方法使用同一个CenterPoint checkpoint、同一份`outputs/centerpoint/detections`、同一个val split、同一个`UnifiedMOTEvaluator`和`V2XSeqProtocol`

## Scene Tracking Difficulty

```bash
conda run -n track python scripts/analysis/analyze_scene_difficulty.py --config configs/analysis/v2xseq_scene_difficulty.yaml
```

输出在`outputs/analysis/scene_difficulty/`。地图角色是`offline_analysis_map`。

BEV和Open3D可视化入口在`scripts/visualization`，Open3D说明见`docs/open3d_visualization.md`
