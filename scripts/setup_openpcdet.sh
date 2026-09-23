#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPCDET="${ROOT}/third_party/OpenPCDet"
OVERLAY="${ROOT}/scripts/openpcdet_overlay"

if [[ ! -d "${OPENPCDET}/.git" ]]; then
  mkdir -p "${ROOT}/third_party"
  git clone --depth 1 https://github.com/open-mmlab/OpenPCDet.git "${OPENPCDET}"
fi

conda run -n track python -c 'import sys,torch,torchvision; print("python", sys.version.split()[0]); print("torch", torch.__version__, "cuda", torch.version.cuda); print("torchvision", torchvision.__version__); print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'
if ! conda run -n track python -c 'import spconv' >/dev/null 2>&1; then
  conda run -n track pip install 'spconv-cu118==2.3.8'
fi
conda run -n track python -c 'import spconv; print("spconv", spconv.__version__)'
if ! conda run -n track python -c 'import SharedArray' >/dev/null 2>&1; then
  conda run -n track pip install SharedArray
fi

mkdir -p "${OPENPCDET}/pcdet/datasets/v2x_seq" "${OPENPCDET}/tools/cfgs/v2x_seq_models" "${OPENPCDET}/tools/cfgs/dataset_configs"
cp -a "${OVERLAY}/pcdet/datasets/v2x_seq/." "${OPENPCDET}/pcdet/datasets/v2x_seq/"
cp -a "${OVERLAY}/tools/cfgs/dataset_configs/v2x_seq_dataset.yaml" "${OPENPCDET}/tools/cfgs/dataset_configs/"
cp -a "${OVERLAY}/tools/cfgs/v2x_seq_models/centerpoint.yaml" "${OPENPCDET}/tools/cfgs/v2x_seq_models/"

python3 - "${OPENPCDET}" <<'PY'
import re
import sys
from pathlib import Path
root = Path(sys.argv[1])
init_path = root / "pcdet/datasets/__init__.py"
text = init_path.read_text(encoding="utf-8")
if "def _optional_dataset" not in text:
    old = """from .dataset import DatasetTemplate
from .kitti.kitti_dataset import KittiDataset
from .nuscenes.nuscenes_dataset import NuScenesDataset
from .waymo.waymo_dataset import WaymoDataset
from .pandaset.pandaset_dataset import PandasetDataset
from .lyft.lyft_dataset import LyftDataset
from .once.once_dataset import ONCEDataset
from .argo2.argo2_dataset import Argo2Dataset
from .custom.custom_dataset import CustomDataset
"""
    new = """import importlib

from .dataset import DatasetTemplate
from .custom.custom_dataset import CustomDataset
from .v2x_seq.v2x_seq_dataset import V2XSeqDataset


def _optional_dataset(module_name, class_name):
    try:
        module = importlib.import_module(module_name, __package__)
    except ModuleNotFoundError:
        return None
    return getattr(module, class_name)
"""
    if old not in text:
        raise SystemExit("无法识别OpenPCDet数据集注册代码")
    text = text.replace(old, new)
    text = text.replace(
        """__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'KittiDataset': KittiDataset,
    'NuScenesDataset': NuScenesDataset,
    'WaymoDataset': WaymoDataset,
    'PandasetDataset': PandasetDataset,
    'LyftDataset': LyftDataset,
    'ONCEDataset': ONCEDataset,
    'CustomDataset': CustomDataset,
    'Argo2Dataset': Argo2Dataset
}""",
        """__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'CustomDataset': CustomDataset,
    'V2XSeqDataset': V2XSeqDataset,
}
for _module_name, _class_name in (
    ('.kitti.kitti_dataset', 'KittiDataset'),
    ('.nuscenes.nuscenes_dataset', 'NuScenesDataset'),
    ('.waymo.waymo_dataset', 'WaymoDataset'),
    ('.pandaset.pandaset_dataset', 'PandasetDataset'),
    ('.lyft.lyft_dataset', 'LyftDataset'),
    ('.once.once_dataset', 'ONCEDataset'),
    ('.argo2.argo2_dataset', 'Argo2Dataset'),
):
    _dataset_cls = _optional_dataset(_module_name, _class_name)
    if _dataset_cls is not None:
        __all__[_class_name] = _dataset_cls""",
    )
    init_path.write_text(text, encoding="utf-8")
replacements = {
    root / "pcdet/models/backbones_2d/base_bev_backbone.py": [(".astype(np.int)", ".astype(np.int64)")],
    root / "pcdet/utils/box_utils.py": [("dtype=np.bool", "dtype=bool")],
    root / "pcdet/datasets/augmentor/augmentor_utils.py": [("dtype=np.bool", "dtype=bool")],
    root / "pcdet/datasets/augmentor/database_sampler.py": [
        ("dtype=np.int)", "dtype=np.int64)"),
        (".astype(np.int)", ".astype(np.int64)"),
        ("dtype=np.float)", "dtype=np.float64)"),
    ],
}
template = root / "pcdet/models/detectors/detector3d_template.py"
template_text = template.read_text(encoding="utf-8")
template_text = re.sub(
    r"torch\.load\((filename|pre_trained_path|optimizer_filename), map_location=loc_type\)",
    r"torch.load(\1, map_location=loc_type, weights_only=False)",
    template_text,
)
template.write_text(template_text, encoding="utf-8")
for path, pairs in replacements.items():
    content = path.read_text(encoding="utf-8")
    for old, new in pairs:
        content = content.replace(old, new)
    path.write_text(content, encoding="utf-8")
PY

if ! compgen -G "${OPENPCDET}/pcdet/ops/iou3d_nms/"'*cuda*.so' > /dev/null; then
  TORCH_CUDA="$(conda run -n track python -c 'import torch; print(torch.version.cuda)')"
  if [[ -d "/usr/local/cuda-${TORCH_CUDA}" ]]; then
    export CUDA_HOME="/usr/local/cuda-${TORCH_CUDA}"
  fi
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
  export MAX_JOBS="${MAX_JOBS:-8}"
  (
    cd "${OPENPCDET}"
    conda run -n track --no-capture-output python setup.py build_ext --inplace
  )
fi
conda run -n track python -c 'import sys; sys.path.insert(0, "'"${OPENPCDET}"'"); from pcdet.ops.iou3d_nms import iou3d_nms_cuda; print("openpcdet_cuda_ok")'
