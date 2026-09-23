from pathlib import Path

import numpy as np
import yaml

CHECKED_KEYS = ("POINT_CLOUD_RANGE", "VOXEL_SIZE", "MAX_POINTS_PER_VOXEL", "MAX_NUMBER_OF_VOXELS")


def _load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _voxel_processor(dataset_cfg):
    for item in dataset_cfg["DATA_PROCESSOR"]:
        if item["NAME"] == "transform_points_to_voxels":
            return item
    raise KeyError("OpenPCDet配置缺少transform_points_to_voxels")


def _same(left, right):
    return np.allclose(np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64))


def check_centerpoint_configs(root):
    root = Path(root)
    project = _load_yaml(root / "configs/centerpoint_v2xseq.yaml")
    dataset_paths = [
        root / "scripts/openpcdet_overlay/tools/cfgs/dataset_configs/v2x_seq_dataset.yaml",
        root / "third_party/OpenPCDet/tools/cfgs/dataset_configs/v2x_seq_dataset.yaml",
    ]
    expected = {
        "POINT_CLOUD_RANGE": project["point_cloud_range"],
        "VOXEL_SIZE": project["voxel_size"],
        "MAX_POINTS_PER_VOXEL": project["max_points_per_voxel"],
        "MAX_NUMBER_OF_VOXELS": project["max_number_of_voxels"],
    }
    for path in dataset_paths:
        if not path.exists():
            raise FileNotFoundError("缺少OpenPCDet数据配置 %s" % path)
        processor = _voxel_processor(_load_yaml(path))
        actual = {
            "POINT_CLOUD_RANGE": _load_yaml(path)["POINT_CLOUD_RANGE"],
            "VOXEL_SIZE": processor["VOXEL_SIZE"],
            "MAX_POINTS_PER_VOXEL": processor["MAX_POINTS_PER_VOXEL"],
            "MAX_NUMBER_OF_VOXELS": processor["MAX_NUMBER_OF_VOXELS"],
        }
        for key in ("POINT_CLOUD_RANGE", "VOXEL_SIZE"):
            if not _same(expected[key], actual[key]):
                raise ValueError("%s不一致 %s %s %s" % (key, path, expected[key], actual[key]))
        if int(actual["MAX_POINTS_PER_VOXEL"]) != int(expected["MAX_POINTS_PER_VOXEL"]):
            raise ValueError("MAX_POINTS_PER_VOXEL不一致 %s" % path)
        voxels = actual["MAX_NUMBER_OF_VOXELS"]
        limit = int(expected["MAX_NUMBER_OF_VOXELS"])
        values = voxels.values() if isinstance(voxels, dict) else [voxels]
        if any(int(value) != limit for value in values):
            raise ValueError("MAX_NUMBER_OF_VOXELS不一致 %s" % path)
    return expected
