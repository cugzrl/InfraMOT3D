import math
import sys
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion


def ensure_fastpoly(root):
    root = str(Path(root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _int_key(key):
    try:
        return int(key)
    except (TypeError, ValueError):
        return key


def _normalize(value):
    if isinstance(value, dict):
        return {_int_key(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def inframot_to_fastpoly(box):
    # InfraMOT3D框是x y z yaw length width height
    x, y, z, yaw, length, width, height = [float(value) for value in box]
    assert length > 0.0 and width > 0.0 and height > 0.0
    assert math.isfinite(yaw)
    # Fast-Poly沿用nuScenes的width length height和wxyz四元数
    quaternion = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw)
    return {
        "center": [x, y, z],
        "size": [width, length, height],
        "rotation": [float(value) for value in quaternion.elements],
    }


def fastpoly_to_inframot(box):
    width, length, height = [float(value) for value in box.wlh]
    yaw = float(box.yaw)
    assert width > 0.0 and length > 0.0 and height > 0.0
    assert math.isfinite(yaw)
    center = [float(value) for value in box.center]
    assert all(math.isfinite(value) for value in center)
    return [center[0], center[1], center[2], yaw, length, width, height]


def _register_classes(class_names):
    from data.script import NUSC_CONSTANT as constants

    constants.CLASS_SEG_TO_STR_CLASS.clear()
    constants.CLASS_STR_TO_SEG_CLASS.clear()
    constants.CLASS_SEG_TO_STR_CLASS.update({name: index for index, name in enumerate(class_names)})
    constants.CLASS_STR_TO_SEG_CLASS.update({index: name for index, name in enumerate(class_names)})
    # 过程噪声按相近官方类别填写，键必须覆盖全部类别编号
    process_noise = {
        "Car": 0.49,
        "Van": 0.49,
        "Bus": 0.49,
        "Truck": 0.49,
        "Pedestrian": 0.36,
        "Cyclist": 0.81,
        "Motorcyclist": 0.25,
        "Barrowlist": 0.36,
    }
    constants.FINETUNE_Q.clear()
    constants.FINETUNE_R.clear()
    for index, name in enumerate(class_names):
        constants.FINETUNE_Q[index] = float(process_noise.get(name, 0.49))
        constants.FINETUNE_R[index] = 1.0


class FastPolyTracker:
    def __init__(self, fastpoly_root, config, class_names):
        ensure_fastpoly(fastpoly_root)
        self.class_names = list(class_names)
        self.config = _normalize(config)
        self.config["basic"]["has_velo"] = False
        self.config["basic"]["CLASS_NUM"] = len(self.class_names)
        self.config["debug"]["is_debug"] = False
        _register_classes(self.class_names)
        from pre_processing import arraydet2box, scale_nms
        from tracking.nusc_tracker import Tracker

        self._arraydet2box = arraydet2box
        self._scale_nms = scale_nms
        self.tracker = Tracker(self.config)
        self.frame_id = 0
        self.sequence_id = 0

    def reset(self):
        self.tracker.reset()
        self.frame_id = 0
        self.sequence_id += 1

    def _detections(self, objects):
        from data.script.NUSC_CONSTANT import CLASS_SEG_TO_STR_CLASS

        rows = []
        for item in objects:
            name = item["class_name"]
            if name not in CLASS_SEG_TO_STR_CLASS:
                continue
            # 推理不读取source_track_id，速度由运动模型自己估计
            converted = inframot_to_fastpoly(item["box"])
            score = float(item["score"])
            assert math.isfinite(score)
            label = int(CLASS_SEG_TO_STR_CLASS[name])
            limit = self.config["preprocessing"]["SF_thre"][label]
            if score <= float(limit):
                continue
            rows.append(
                converted["center"]
                + converted["size"]
                + [0.0, 0.0]
                + converted["rotation"]
                + [score, label]
            )
        if not rows:
            return np.zeros((0, 14), dtype=np.float64), np.zeros(0), np.zeros(0), np.zeros(0), 0
        np_dets = np.asarray(rows, dtype=np.float64)
        assert np_dets.shape[1] == 14
        boxes, bottom, normal = self._arraydet2box(np_dets)
        infos = {
            "np_dets": np_dets,
            "np_dets_bottom_corners": bottom,
            "np_dets_norm_corners": normal,
            "box_dets": boxes,
        }
        keep = self._scale_nms(
            box_infos=infos,
            metrics=self.config["preprocessing"]["NMS_metric"],
            thres=self.config["preprocessing"]["NMS_thre"],
            factors=self.config["preprocessing"]["SCALE"],
            voxel_mask_size=self.config["preprocessing"]["voxel_mask_size"],
            use_voxel_mask=self.config["preprocessing"]["voxel_mask"],
        )
        if len(keep) == 0:
            return np.zeros((0, 14), dtype=np.float64), np.zeros(0), np.zeros(0), np.zeros(0), 0
        return np_dets[keep], bottom[keep], normal[keep], boxes[keep], len(keep)

    def update(self, objects, timestamp=None):
        if timestamp is not None:
            assert math.isfinite(float(timestamp))
        self.frame_id += 1
        np_dets, bottom, normal, boxes, count = self._detections(objects)
        data_info = {
            "is_first_frame": self.frame_id == 1,
            "timestamp": self.frame_id,
            "sample_token": str(self.frame_id),
            "seq_id": self.sequence_id,
            "frame_id": self.frame_id,
            "has_velo": False,
            "np_dets": np_dets,
            "np_dets_bottom_corners": bottom,
            "np_dets_norm_corners": normal,
            "box_dets": boxes,
            "no_dets": count == 0,
            "det_num": count,
        }
        self.tracker.tracking(data_info)
        if data_info.get("no_val_track_result"):
            return []
        outputs = []
        for box in data_info["box_track_res"]:
            score = float(box.score)
            track_id = int(box.tracking_id)
            assert track_id >= 0
            assert math.isfinite(score)
            outputs.append(
                {
                    "class_name": str(box.name),
                    "track_id": track_id,
                    "score": score,
                    "box": fastpoly_to_inframot(box),
                }
            )
        return outputs
