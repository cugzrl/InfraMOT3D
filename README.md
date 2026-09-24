# InfraMOT3D

面向路侧点云与图像的多模态3D目标跟踪和轨迹预测工程

包含V2X-Seq路侧标注转换、AB3DMOT等基线算法、3D MOT评估和BEV可视化

## 环境

conda`track`环境，Python为3.10，核心依赖：
NumPy、SciPy、FilterPy、Shapely、PyYAML、Matplotlib和OpenCV

```bash
conda run -n track python -m pytest tests -q
```

## 数据

`data/v2x-seq-infrastructure`软链接到原始路侧数据

统一帧格式见`docs/data_format.md`

## CenterPoint检测

OpenPCDet放在`third_party/OpenPCDet`，InfraMOT3D只做数据转换、训练调用和检测结果适配。安装和版本说明见`docs/centerpoint_setup.md`。

```bash
bash scripts/setup_openpcdet.sh
export PYTHONPATH=$PWD/src
conda run -n track python scripts/prepare_centerpoint_data.py --config configs/centerpoint_v2xseq.yaml --force
bash scripts/train_centerpoint_v2xseq.sh 2 30
conda run -n track python scripts/infer_centerpoint_v2xseq.py --config configs/centerpoint_v2xseq.yaml
```

`train_centerpoint_v2xseq.sh`的两个参数是单卡batch size和epoch，默认`2 30`。脚本固定`CUDA_VISIBLE_DEVICES=0`。

推理写出`outputs/centerpoint/detections/{sequence_id}.jsonl`和`outputs/centerpoint/detection_manifest.json`。检测指标写在`outputs/centerpoint/detection_metrics.json`，与MOT指标分开。

跟踪读取同一份检测结果时，在配置里增加：

```yaml
input:
  type: detection
  detection_root: outputs/centerpoint/detections
```

`configs/ab3dmot_centerpoint.yaml`是AB3DMOT示例。不写`input`时仍使用标注，使用GT的测试实验保持不变。

推理可把分数阈值降到`0.01`，只影响导出，不改已训练权重。四个Tracker读同一份检测。

```bash
export PYTHONPATH=$PWD/src
conda run -n track python scripts/infer_centerpoint_v2xseq.py --config configs/centerpoint_v2xseq.yaml --split all --score-threshold 0.01 --skip-eval
bash scripts/run_centerpoint_baselines.sh
bash scripts/setup_grae.sh
conda run -n track python scripts/prepare_grae_v2xseq.py --config configs/grae_centerpoint.yaml --split train
bash scripts/train_grae_v2xseq.sh
conda run -n track python scripts/infer_grae_v2xseq.py --config configs/grae_centerpoint.yaml --split val
conda run -n track python scripts/evaluate.py --config configs/grae_centerpoint.yaml --split val
bash scripts/run_centerpoint_tracking_benchmark.sh
```

全类别结果在`outputs/all_class_full_range.csv`，使用路侧全范围。官方Car结果在`outputs/official_v2xseq_car.csv`，只含合并后的Car，并使用DAIR-V2X的`extended_range`。两套protocol不能直接横向比较数值。

GRAE的`velocity`为`[0,0]`。空检测帧只推进轨迹寿命，不改最后一次真实观测时间。类别`high_threshold`控制新建轨迹，低分检测只参与二阶段关联。

## 运行

```bash
bash scripts/run_ab3dmot_gt.sh
```

也可以分步运行

```bash
export PYTHONPATH=$PWD/src
conda run -n track python scripts/prepare_data.py --config configs/ab3dmot_gt.yaml
conda run -n track python scripts/run_tracker.py --config configs/ab3dmot_gt.yaml
conda run -n track python scripts/evaluate.py --config configs/ab3dmot_gt.yaml
conda run -n track python scripts/visualize.py --config configs/ab3dmot_gt.yaml --sequence 0000 --max-frames 100 --video
```

## 结果

- `outputs/ab3dmot_gt/predictions`保存逐序列跟踪结果
- `outputs/ab3dmot_gt/metrics_summary.json`保存总体和分类指标
- `outputs/ab3dmot_gt/metrics_by_sequence.json`保存逐序列指标
- `outputs/ab3dmot_gt/runtime.json`保存运行速度
- `outputs/ab3dmot_gt/visualizations`保存可视化


## 目录

```text
InfraMOT3D
├── configs                  算法与评估配置
├── data
│   ├── v2x-seq-infrastructure  原始路侧数据软链接
│   ├── converted            统一格式数据
│   └── centerpoint_v2xseq  CenterPoint训练数据
├── third_party/OpenPCDet    检测框架
├── third_party/GRAE-3DMOT   GRAE跟踪官方代码
├── third_party/DAIR-V2X     V2X-Seq官方跟踪评估
├── docs                     数据格式和实验说明
├── outputs                  预测结果、指标和可视化
├── scripts                  转换、检测、跟踪、评估和可视化入口
├── src/inframot3d
│   ├── data                 数据适配器
│   ├── detection            CenterPoint适配
│   ├── tracking             3D跟踪算法
│   ├── prediction           轨迹预测算法
│   ├── fusion               点云图像融合算法
│   ├── evaluation           跟踪评估
│   └── visualization        可视化
```
