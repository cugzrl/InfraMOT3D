# Open3D离屏可视化

脚本使用Open3D的EGL离屏渲染，不需要桌面或可见窗口

每帧读取原始路侧PCD，并叠加GT框、AB3DMOT轨迹框或二者

- GT框使用绿色
- 跟踪框根据轨迹ID稳定着色
- 未匹配到当前检测的预测框降低亮度
- 点云按照高度着色

生成GT与跟踪对照视频：

```bash
export PYTHONPATH=$PWD/src
EGL_PLATFORM=surfaceless conda run -n track python scripts/visualize_open3d.py \
  --config configs/ab3dmot_gt.yaml \
  --sequence 0000 \
  --mode both \
  --video
```

生成GIF：

```bash
EGL_PLATFORM=surfaceless conda run -n track python scripts/visualize_open3d.py \
  --config configs/ab3dmot_gt.yaml \
  --sequence 0000 \
  --mode track \
  --max-frames 80 \
  --gif
```

默认保留逐帧PNG，输出目录为`outputs/ab3dmot_gt/open3d_visualizations`
