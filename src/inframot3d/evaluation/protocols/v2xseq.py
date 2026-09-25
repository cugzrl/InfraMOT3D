import json
import math
import sys
from pathlib import Path

import numpy as np

from inframot3d.evaluation.protocols.base import BaseProtocol
from inframot3d.io import read_json, read_jsonl


OFFICIAL_RANGE = [0.0, -39.68, -3.0, 100.0, 39.68, 1.0]
MERGED = {"Car": "Car", "Van": "Car", "Bus": "Car", "Truck": "Car"}
RECALL_POINTS = 41


class V2XSeqProtocol(BaseProtocol):
    name = "v2xseq"

    def __init__(self, root):
        self.root = Path(root)
        self.range = list(OFFICIAL_RANGE)
        self._bbox_filter = None
        self._corner_fn = None

    def normalize_class(self, class_name):
        return MERGED.get(class_name)

    def iou_threshold(self):
        return 0.25

    def get_eval_classes(self):
        return ["Car"]

    def score_sampling(self):
        # 阈值由官方getThresholds按41个recall点选取
        return {"num_sample_pts": RECALL_POINTS, "method": "DAIR-V2X getThresholds"}

    def _tools(self):
        if self._bbox_filter is not None:
            return self._bbox_filter, self._corner_fn
        path = self.root / "third_party" / "DAIR-V2X" / "v2x" / "v2x_utils" / "gen_eval_tracking_data"
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
        from filter import RectFilter, get_lidar_3d_8points, range2box

        self._bbox_filter = RectFilter(range2box(np.array(self.range))[0])
        self._corner_fn = get_lidar_3d_8points
        return self._bbox_filter, self._corner_fn

    def inside_range(self, box):
        # 与官方RectFilter一致，任一角点落入extended_range即保留
        bbox_filter, corner_fn = self._tools()
        x, y, z, yaw, length, width, height = [float(value) for value in box]
        corners = corner_fn([length, width, height], [x, y, z], yaw)
        return bool(bbox_filter(corners))

    def filter_gt(self, objects):
        kept = []
        for item in objects:
            if self.normalize_class(item["class_name"]) is None:
                continue
            if not self.inside_range(item["box"]):
                continue
            kept.append(item)
        return kept

    def filter_prediction(self, objects):
        kept = []
        for item in objects:
            if self.normalize_class(item["class_name"]) is None:
                continue
            if not self.inside_range(item["box"]):
                continue
            kept.append(item)
        return kept

    def aggregate_metrics(self, raw):
        payload = {
            "MOTA": float(raw["MOTA"]),
            "MOTP": float(raw["MOTP"]),
            "AMOTA": float(raw["AMOTA"]),
            "AMOTP": float(raw["AMOTP"]),
            "IDSW": int(raw["IDSW"]),
            "IDF1": float(raw["IDF1"]),
            "FM": int(raw["FM"]),
            "FP": int(raw["FP"]),
            "FN": int(raw["FN"]),
            "best_score_threshold": float(raw["best_score_threshold"]),
            "TP": int(raw["TP"]),
            "GT": int(raw["GT"]),
            "protocol": self.name,
            "evaluation_range": list(self.range),
            "iou_threshold": self.iou_threshold(),
        }
        return payload

    def export_split(self, converted_root, prediction_root, data_root, split_ids, output_dir):
        converted_root = Path(converted_root)
        prediction_root = Path(prediction_root)
        data_root = Path(data_root)
        output_dir = Path(output_dir)
        gt_dir = output_dir / "label"
        pred_dir = output_dir / "pred"
        gt_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)
        manifest = read_json(converted_root / "manifest.json")
        allowed = set(split_ids)
        seqmap = []
        for entry in manifest["sequences"]:
            sequence_id = entry["sequence_id"]
            if sequence_id not in allowed:
                continue
            gt_rows = list(read_jsonl(converted_root / entry["path"]))
            pred_rows = list(read_jsonl(prediction_root / ("%s.jsonl" % sequence_id)))
            last_index = _write_sequence(self, gt_rows, pred_rows, data_root, gt_dir / ("%s.txt" % sequence_id), pred_dir / ("%s.txt" % sequence_id))
            seqmap.append("%s empty 0 %d" % (sequence_id, last_index))
        (output_dir / "evaluate_tracking.seqmap.val").write_text("\n".join(seqmap) + "\n", encoding="utf-8")
        return len(seqmap)


def _trans(point, rotation, translation):
    point = np.asarray(point, dtype=np.float64).reshape(3, 1)
    return (rotation @ point + translation).reshape(3)


def _lidar_corners(length, width, height, center, yaw):
    rotation = np.array(
        [[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    corners = np.array(
        [
            [length / 2, length / 2, -length / 2, -length / 2, length / 2, length / 2, -length / 2, -length / 2],
            [width / 2, -width / 2, -width / 2, width / 2, width / 2, -width / 2, -width / 2, width / 2],
            [-height / 2, -height / 2, -height / 2, -height / 2, height / 2, height / 2, height / 2, height / 2],
        ],
        dtype=np.float64,
    )
    return (rotation @ corners + np.asarray(center, dtype=np.float64).reshape(3, 1)).T


def _camera_yaw(corners, location):
    dx = corners[0][0] - corners[3][0]
    dz = corners[0][2] - corners[3][2]
    rotation_y = -math.atan2(dz, dx)
    alpha = rotation_y - (-math.atan2(-location[2], -location[0])) + math.pi / 2
    if alpha > math.pi:
        alpha -= 2.0 * math.pi
    if alpha <= -math.pi:
        alpha += 2.0 * math.pi
    return alpha, rotation_y


def _load_calib(data_root, frame_id):
    path = Path(data_root) / "calib" / "virtuallidar_to_camera" / ("%s.json" % frame_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rotation = np.asarray(payload["rotation"], dtype=np.float64).reshape(3, 3)
    translation = np.asarray(payload["translation"], dtype=np.float64).reshape(3, 1)
    return rotation, translation


def _kitti_line(frame_index, track_id, box, rotation, translation, score):
    x, y, z, yaw, length, width, height = [float(value) for value in box]
    corners = _lidar_corners(length, width, height, [x, y, z], yaw)
    camera_corners = [_trans(point, rotation, translation) for point in corners]
    bottom = _trans([x, y, z - height / 2.0], rotation, translation)
    alpha, rotation_y = _camera_yaw(camera_corners, bottom)
    fields = [
        str(int(frame_index)),
        str(int(track_id)),
        "Car",
        "0",
        "0",
        "%.6f" % alpha,
        "0",
        "0",
        "100",
        "100",
        "%.6f" % height,
        "%.6f" % width,
        "%.6f" % length,
        "%.6f" % bottom[0],
        "%.6f" % bottom[1],
        "%.6f" % bottom[2],
        "%.6f" % rotation_y,
    ]
    if score is not None:
        fields.append("%.6f" % float(score))
    return " ".join(fields)


def _write_sequence(protocol, gt_rows, pred_rows, data_root, gt_path, pred_path):
    pred_by_frame = {int(row["frame_index"]): row for row in pred_rows}
    for expected, row in enumerate(gt_rows):
        if int(row["frame_index"]) != expected:
            raise ValueError("帧序号不连续 %s" % row["frame_id"])
    kept_frames = []
    for row in gt_rows:
        frame_index = int(row["frame_index"])
        kept = protocol.filter_gt(row["objects"])
        if not kept:
            continue
        kept_frames.append((frame_index, row, kept))
    remap = {frame_index: new_index for new_index, (frame_index, _, _) in enumerate(kept_frames)}
    gt_lines = []
    pred_lines = []
    calib_cache = {}
    last_index = max(remap.values()) if remap else 0
    for frame_index, row, kept in kept_frames:
        frame_id = row["frame_id"]
        if frame_id not in calib_cache:
            calib_cache[frame_id] = _load_calib(data_root, frame_id)
        rotation, translation = calib_cache[frame_id]
        mapped = remap[frame_index]
        for item in kept:
            gt_lines.append(_kitti_line(mapped, item["source_track_id"], item["box"], rotation, translation, None))
        pred_row = pred_by_frame.get(frame_index)
        if pred_row is None:
            raise ValueError("预测缺少帧 %s" % frame_id)
        if pred_row["frame_id"] != frame_id or pred_row["sequence_id"] != row["sequence_id"]:
            raise ValueError("预测帧未对齐 %s" % frame_id)
        for item in protocol.filter_prediction(pred_row["objects"]):
            pred_lines.append(_kitti_line(mapped, item["track_id"], item["box"], rotation, translation, item.get("score", 1.0)))
    gt_path.write_text("\n".join(gt_lines) + ("\n" if gt_lines else ""), encoding="utf-8")
    pred_path.write_text("\n".join(pred_lines) + ("\n" if pred_lines else ""), encoding="utf-8")
    return last_index
