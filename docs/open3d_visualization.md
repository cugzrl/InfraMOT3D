# Open3D 离屏可视化

脚本使用 Open3D 的 EGL 离屏渲染，不需要桌面或可见窗口。

跟踪结果和画面参数是分开的。`--tracker` 读取 `configs/trackers/<tracker>/centerpoint.yaml` 里的 `project.output_root/predictions`。画面参数统一来自 `configs/visualization/open3d_v2xseq.yaml`。`--tracker` 和 `--config` 只能使用其中一个。不写这两个参数时，仍默认读取 `configs/trackers/ab3dmot/gt.yaml`。

`--tracker` 可选 `ab3dmot`、`simpletrack`、`immortal`、`grae`、`fastpoly`、`3dmotformer`。Fast-Poly 的正式 benchmark 使用 `outputs/fastpoly_centerpoint/calibration/best_config.yaml`，它和基础配置的 `output_root` 相同，因此可视化读取 `outputs/fastpoly_centerpoint/predictions`，不会重新运行 tracker。

`--num-frames` 控制渲染帧数，原有 `--max-frames` 仍然可用。两者同时出现时数值必须相同。

## 视角

- `classic` 保留原有深色斜视角
- `roadside` 使用纯白背景、灰色场景点云，跟踪框内的点与 3D 框使用同一套马卡龙色

`roadside` 范围是前向 `0～165 m`、横向 `-40～40 m`、高度 `-4～4 m`。左上角图像使用每帧相机内参、畸变和虚拟激光雷达到相机外参投影 3D 框。右上角显示当前 tracker 名称。

## SimpleTrack

```bash
PYTHONPATH=src EGL_PLATFORM=surfaceless conda run -n track \
python scripts/visualization/visualize_open3d.py \
  --tracker simpletrack \
  --sequence 0000 \
  --view roadside \
  --mode track \
  --start-frame 0 \
  --num-frames 100 \
  --video
```

## AB3DMOT

```bash
PYTHONPATH=src EGL_PLATFORM=surfaceless conda run -n track \
python scripts/visualization/visualize_open3d.py \
  --tracker ab3dmot \
  --sequence 0000 \
  --view roadside \
  --mode track \
  --num-frames 100 \
  --video
```

## 3DMOTFormer

```bash
PYTHONPATH=src EGL_PLATFORM=surfaceless conda run -n track \
python scripts/visualization/visualize_open3d.py \
  --tracker 3dmotformer \
  --sequence 0003 \
  --view roadside \
  --mode track \
  --num-frames 100 \
  --video
```

3DMOTFormer、Fast-Poly 和 ImmortalTracker 目前只有 val 预测。序列 `0000` 不在其中，可改用 `0003`。Fast-Poly 把上面的 `--tracker 3dmotformer` 换成 `--tracker fastpoly` 即可。

## 保留原视角

```bash
EGL_PLATFORM=surfaceless conda run -n track python scripts/visualization/visualize_open3d.py \
  --config configs/trackers/ab3dmot/gt.yaml \
  --sequence 0000 \
  --view classic \
  --mode both \
  --video
```

可使用 `--gif` 生成 GIF，使用 `--no-image-inset` 关闭左上角图像。默认图片写到对应 tracker 的 `outputs/<tracker>/open3d_visualizations/<tracker名>/<序列>/roadside/<模式>/`。
