import os
import sys
from pathlib import Path

from inframot3d.detection.v2x_seq_converter import openpcdet_to_box


def ensure_openpcdet(root):
    root = str(Path(root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def load_openpcdet_cfg(cfg_file):
    cfg_file = Path(cfg_file).resolve()
    tools = cfg_file.parents[2]
    ensure_openpcdet(tools.parent)
    cwd = Path.cwd()
    os.chdir(tools)
    try:
        from pcdet.config import cfg, cfg_from_yaml_file

        cfg_from_yaml_file(str(cfg_file), cfg)
    finally:
        os.chdir(cwd)
    data_path = Path(cfg.DATA_CONFIG.DATA_PATH)
    if not data_path.is_absolute():
        cfg.DATA_CONFIG.DATA_PATH = str((tools / data_path).resolve())
    return cfg


def prediction_to_object(name, score, box):
    item = {"class_name": str(name), "score": float(score), "box": openpcdet_to_box(box)}
    if len(box) >= 9:
        item["velocity"] = [float(box[7]), float(box[8])]
    return item
