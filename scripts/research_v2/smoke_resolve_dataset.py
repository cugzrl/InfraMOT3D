"""检查 RESOLVE 转换结果能被 OpenPCDet 读成一个 batch"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT / "third_party/OpenPCDet/tools")
sys.path.insert(0, str(ROOT / "third_party/OpenPCDet"))
sys.path.insert(0, str(ROOT / "src"))

from inframot3d.detection.openpcdet_adapter import load_openpcdet_cfg
from pcdet.datasets import build_dataloader


class _Logger:
    def info(self, message):
        print(message)


def main():
    cfg = load_openpcdet_cfg(ROOT / "third_party/OpenPCDet/tools/cfgs/resolve_models/centerpoint_res16.yaml")
    dataset, _, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=1, dist=False, workers=0, logger=_Logger(), training=False)
    sample = dataset[0]
    points = sample["points"]
    boxes = sample.get("gt_boxes", None)
    print("frames", len(dataset), "points", tuple(points.shape), "gt", None if boxes is None else tuple(boxes.shape), "frame", sample.get("frame_id"))


if __name__ == "__main__":
    main()
