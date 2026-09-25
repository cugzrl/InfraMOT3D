# Open3D离屏可视化

脚本使用Open3D的EGL离屏渲染，不需要桌面或可见窗口

## 视角

- `classic`保留原有深色斜视角
- `roadside`提供前向路侧视角、浅色点云和左上角相机图像

`roadside`根据当前序列统计采用前向`0～165 m`、横向`-40～40 m`和高度`-4～4 m`的范围

左上角图像使用每帧相机内参、畸变参数和虚拟激光雷达到相机外参投影3D框

## 生成新视角视频

```bash
export PYTHONPATH=$PWD/src
EGL_PLATFORM=surfaceless conda run -n track python scripts/visualization/visualize_open3d.py \
  --config configs/trackers/ab3dmot/gt.yaml \
  --sequence 0000 \
  --view roadside \
  --mode both \
  --video
```

## 保留原视角

```bash
EGL_PLATFORM=surfaceless conda run -n track python scripts/visualization/visualize_open3d.py \
  --config configs/trackers/ab3dmot/gt.yaml \
  --sequence 0000 \
  --view classic \
  --mode both \
  --video
```

可使用`--gif`生成GIF，使用`--no-image-inset`关闭左上角图像
