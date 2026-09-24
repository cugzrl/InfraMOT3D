import argparse
import json
import math
from pathlib import Path

import numpy as np

from inframot3d.config import load_config
from inframot3d.io import read_json, read_jsonl, write_json


MERGED = {"Car": "Car", "Van": "Car", "Bus": "Car", "Truck": "Car"}


def _resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


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
    path = data_root / "calib" / "virtuallidar_to_camera" / ("%s.json" % frame_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rotation = np.asarray(payload["rotation"], dtype=np.float64).reshape(3, 3)
    translation = np.asarray(payload["translation"], dtype=np.float64).reshape(3, 1)
    return rotation, translation


def _kitti_line(frame_index, track_id, box, rotation, translation, score):
    x, y, z, yaw, length, width, height = [float(value) for value in box]
    center = [x, y, z]
    corners = _lidar_corners(length, width, height, center, yaw)
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


def _write_sequence(gt_rows, pred_rows, data_root, gt_path, pred_path):
    pred_by_frame = {int(row["frame_index"]): row for row in pred_rows}
    for expected, row in enumerate(gt_rows):
        if int(row["frame_index"]) != expected:
            raise ValueError("帧序号不连续 %s" % row["frame_id"])
    gt_lines = []
    pred_lines = []
    calib_cache = {}
    last_index = 0
    for row in gt_rows:
        frame_index = int(row["frame_index"])
        last_index = max(last_index, frame_index)
        frame_id = row["frame_id"]
        if frame_id not in calib_cache:
            calib_cache[frame_id] = _load_calib(data_root, frame_id)
        rotation, translation = calib_cache[frame_id]
        for item in row["objects"]:
            if item["class_name"] not in MERGED:
                continue
            gt_lines.append(_kitti_line(frame_index, item["source_track_id"], item["box"], rotation, translation, None))
        pred_row = pred_by_frame.get(frame_index)
        if pred_row is None:
            raise ValueError("预测缺少帧 %s" % frame_id)
        if pred_row["frame_id"] != frame_id or pred_row["sequence_id"] != row["sequence_id"]:
            raise ValueError("预测帧未对齐 %s" % frame_id)
        for item in pred_row["objects"]:
            if item["class_name"] not in MERGED:
                continue
            pred_lines.append(
                _kitti_line(frame_index, item["track_id"], item["box"], rotation, translation, item["score"])
            )
    gt_path.write_text("\n".join(gt_lines) + ("\n" if gt_lines else ""), encoding="utf-8")
    pred_path.write_text("\n".join(pred_lines) + ("\n" if pred_lines else ""), encoding="utf-8")
    return last_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    root = config["_root"]
    split_ids = read_json(root / config["split_file"])[args.split]
    data_root = Path(config["project"]["data_root"])
    converted_root = Path(config["project"]["converted_root"])
    manifest = read_json(converted_root / "manifest.json")
    prediction_root = _resolve(root, args.prediction_root)
    output = _resolve(root, args.output)
    gt_dir = output / "label"
    pred_dir = output / "pred"
    gt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    seqmap = []
    for entry in manifest["sequences"]:
        sequence_id = entry["sequence_id"]
        if sequence_id not in set(split_ids):
            continue
        gt_rows = list(read_jsonl(converted_root / entry["path"]))
        pred_rows = list(read_jsonl(prediction_root / ("%s.jsonl" % sequence_id)))
        last_index = _write_sequence(
            gt_rows,
            pred_rows,
            data_root,
            gt_dir / ("%s.txt" % sequence_id),
            pred_dir / ("%s.txt" % sequence_id),
        )
        seqmap.append("%s empty 0 %d" % (sequence_id, last_index))
        print("导出序列%s 帧%d" % (sequence_id, last_index + 1))
    (output / "evaluate_tracking.seqmap.val").write_text("\n".join(seqmap) + "\n", encoding="utf-8")
    write_json(
        output / "export_meta.json",
        {
            "supported_classes": ["Car"],
            "merged_into_car": ["Van", "Bus", "Truck"],
            "unsupported_classes": ["Pedestrian", "Cyclist", "Motorcyclist", "Barrowlist"],
            "iou": "3D",
            "iou_threshold": 0.25,
            "sequences": len(seqmap),
        },
    )


if __name__ == "__main__":
    main()
