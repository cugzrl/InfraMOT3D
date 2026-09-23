# CenterPoint环境

继续使用已有conda环境`track`，不新建环境，也不降级驱动。

当前检测相关版本：

- Python 3.10.18
- PyTorch 2.7.1+cu118
- torchvision 0.22.1+cu118
- CUDA toolkit 11.8，路径`/usr/local/cuda-11.8`
- 系统默认`nvcc`是12.6，编译OpenPCDet时改用11.8 toolkit，与PyTorch的CUDA版本一致
- spconv-cu118 2.3.8
- cumm-cu118 0.7.11
- SharedArray 3.2.4
- OpenPCDet commit `233f849`，仓库https://github.com/open-mmlab/OpenPCDet

`environment.yml`的pip段记录了新增的`spconv-cu118`、`cumm-cu118`和`SharedArray`。`numba`、`easydict`、`tensorboardX`、`pyquaternion`、`scikit-image`在`track`中已经存在。

OpenPCDet对NumPy 1.26做了小范围兼容修改：`np.int`、`np.float`、`np.bool`改为当前NumPy仍支持的类型。当前`protobuf`较新，训练脚本设置`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`，避免旧版`tensorboardX`生成代码报错。CenterPoint网络没有改。

安装：

```bash
bash scripts/setup_openpcdet.sh
```

脚本会在`third_party/OpenPCDet`不存在时克隆官方仓库，复制V2X-Seq数据集和配置，并使用与PyTorch匹配的CUDA toolkit编译CUDA扩展。
