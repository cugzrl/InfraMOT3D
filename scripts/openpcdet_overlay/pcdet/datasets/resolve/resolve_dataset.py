import copy
import pickle
from pathlib import Path

import numpy as np

from ..dataset import DatasetTemplate
from ..v2x_seq.v2x_seq_dataset import evaluate_annos

# 与官方 create_sunlakes_data_v3.py 一致的点云旋转角
ROT_DEG = -142.5
# 不同厂商强度量纲差异很大，统一做对数归一化
INTENSITY_SCALE = float(np.log1p(65535.0))
RES_DIRS = {"128": "128_128b_sync", "64": "64_64b_sync", "16": "16_16b_sync"}


def rotation_matrix():
    theta = np.deg2rad(ROT_DEG)
    return np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def load_resolve_points(path):
    points = np.fromfile(str(path), dtype=np.float32).reshape(-1, 4).copy()
    points = points[np.isfinite(points).all(axis=1)]
    points[:, :3] = points[:, :3] @ rotation_matrix().T
    points[:, 3] = np.log1p(np.maximum(points[:, 3], 0.0)) / INTENSITY_SCALE
    return points


class ResolveDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger)
        self.resolution = str(dataset_cfg.RESOLUTION)
        self.raw_root = Path(dataset_cfg.RAW_ROOT)
        self.infos = []
        for info_path in self.dataset_cfg.INFO_PATH[self.mode]:
            path = self.root_path / info_path
            if path.exists():
                with open(path, "rb") as stream:
                    self.infos.extend(pickle.load(stream))
        if self.logger is not None:
            self.logger.info("RESOLVE %s 线样本数 %d" % (self.resolution, len(self.infos)))

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs
        return len(self.infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)
        info = copy.deepcopy(self.infos[index])
        path = self.raw_root / info["session"] / RES_DIRS[self.resolution] / ("%s.bin" % info["token"])
        input_dict = {"frame_id": info["frame_id"], "points": load_resolve_points(path)}
        boxes, names = info["gt_boxes"], info["gt_names"]
        if self.training:
            # 训练时按官方做法只保留当前分辨率下至少一个点的目标
            keep = info["num_points"][self.resolution] > 0
            boxes, names = boxes[keep], names[keep]
        input_dict.update({"gt_names": names, "gt_boxes": boxes.astype(np.float32)})
        return self.prepare_data(data_dict=input_dict)

    def evaluation(self, det_annos, class_names, **kwargs):
        gt_annos = []
        for info in self.infos:
            keep = info["num_points"]["128"] > 0
            gt_annos.append({"name": info["gt_names"][keep], "gt_boxes_lidar": info["gt_boxes"][keep]})
        thresholds = {name: 0.5 if name in ("car", "truck", "bus", "trailer", "construction_vehicle", "barrier") else 0.25 for name in class_names}
        metrics = evaluate_annos(det_annos, gt_annos, class_names, thresholds, 0.1)
        lines = ["class precision recall ap num_gt num_pred"]
        flat = {}
        for name in class_names:
            item = metrics[name]
            lines.append("%s %.4f %.4f %.4f %d %d" % (name, item["precision"], item["recall"], item["ap"], item["num_gt"], item["num_pred"]))
            flat["%s/ap" % name] = float(item["ap"])
        return "\n".join(lines), flat
