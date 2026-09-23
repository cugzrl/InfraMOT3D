# InfraMOT3D

面向路侧点云与图像的多模态3D目标跟踪和轨迹预测工程

包含V2X-Seq路侧标注转换、AB3DMOT基线、3D MOT评估和BEV可视化

## 环境

`track`环境，Python为3.10，核心依赖为NumPy、SciPy、FilterPy、Shapely、PyYAML、Matplotlib和OpenCV

```bash
conda run -n track python -m pytest tests -q
```

## 数据

`data/v2x-seq-infrastructure`软链接到原始路侧数据

统一帧格式见`docs/data_format.md`

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
│   └── converted            统一格式数据
├── docs                     数据格式和实验说明
├── outputs                  预测结果、指标和可视化
├── scripts                  转换、跟踪、评估和可视化入口
├── src/inframot3d
│   ├── data                 数据适配器
│   ├── tracking             3D跟踪算法
│   ├── prediction           轨迹预测算法
│   ├── fusion               点云图像融合算法
│   ├── evaluation           跟踪评估
│   └── visualization        可视化
└── tests                    单元测试
```
