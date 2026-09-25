import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from inframot3d.config import load_config
from inframot3d.io import read_json, read_jsonl, write_json
from inframot3d.tracking.motformer_adapter import to_motformer_box


def _resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def _column(rows, width, dtype):
    if not rows:
        return np.zeros((0, width), dtype=dtype)
    return np.asarray(rows, dtype=dtype)


def _vector(rows, dtype):
    if not rows:
        return np.zeros((0,), dtype=dtype)
    return np.asarray(rows, dtype=dtype)


def _pack_boxes(objects, class_to_index, with_score, track_key):
    translation, size, yaw, category = [], [], [], []
    extra = []
    for item in objects:
        name = item["class_name"]
        if name not in class_to_index:
            continue
        converted = to_motformer_box(item["box"])
        translation.append(converted[:3])
        size.append(converted[3:6])
        yaw.append([converted[6]])
        category.append(class_to_index[name])
        if with_score:
            extra.append(float(item["score"]))
        else:
            extra.append(item[track_key])
    return translation, size, yaw, category, extra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/trackers/3dmotformer/centerpoint.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    root = config["_root"]
    split = read_json(_resolve(root, config["split_file"]))
    calibration = set(str(value) for value in config["train"]["calibration_sequences"])
    if calibration & set(split["val"]):
        raise SystemExit("calibration序列不能来自val")
    train_ids = [str(value) for value in split["train"] if str(value) not in calibration]
    class_to_index = {name: index for index, name in enumerate(config["classes"])}
    converted_root = Path(config["project"]["converted_root"])
    detection_root = _resolve(root, config["input"]["detection_root"])
    output_root = _resolve(root, config["project"]["motformer_data_root"]) / "training"
    if output_root.exists():
        for child in output_root.glob("*"):
            if child.is_dir():
                for frame_file in child.glob("*.pkl"):
                    frame_file.unlink()
                child.rmdir()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = read_json(converted_root / "manifest.json")
    by_id = {entry["sequence_id"]: entry for entry in manifest["sequences"]}
    for sequence_id in train_ids:
        detections = list(read_jsonl(detection_root / ("%s.jsonl" % sequence_id)))
        ground_truth = list(read_jsonl(converted_root / by_id[sequence_id]["path"]))
        if len(detections) != len(ground_truth):
            raise SystemExit("检测和标注帧数不一致 %s" % sequence_id)
        by_frame = {row["frame_id"]: row for row in detections}
        ordered = []
        for frame in ground_truth:
            row = by_frame.get(frame["frame_id"])
            if row is None or int(row["timestamp"]) != int(frame["timestamp"]):
                raise SystemExit("帧未对齐 %s %s" % (sequence_id, frame["frame_id"]))
            ordered.append((frame, row))
        identity = {}
        next_lookup = []
        for index, (frame, _) in enumerate(ordered):
            current = {}
            for item in frame["objects"]:
                if item["class_name"] not in class_to_index:
                    continue
                key = (item["class_name"], str(item["source_track_id"]))
                if key not in identity:
                    identity[key] = len(identity)
                current[key] = item["box"]
            next_lookup.append(current)
        scene_dir = output_root / sequence_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        for index, (frame, detection) in enumerate(ordered):
            later = next_lookup[index + 1] if index + 1 < len(next_lookup) else {}
            det_trans, det_size, det_yaw, det_class, det_score = _pack_boxes(
                detection["objects"], class_to_index, True, None
            )
            gt_objects = []
            for item in frame["objects"]:
                if item["class_name"] not in class_to_index:
                    continue
                key = (item["class_name"], str(item["source_track_id"]))
                copied = dict(item)
                copied["numeric_track_id"] = identity[key]
                copied["next_box"] = later.get(key)
                gt_objects.append(copied)
            gt_trans, gt_size, gt_yaw, gt_class, gt_track = _pack_boxes(
                gt_objects, class_to_index, False, "numeric_track_id"
            )
            next_exist, next_trans, next_size, next_yaw = [], [], [], []
            for item in gt_objects:
                next_box = item["next_box"]
                if next_box is None:
                    next_exist.append(False)
                    next_trans.append([0.0, 0.0, 0.0])
                    next_size.append([0.0, 0.0, 0.0])
                    next_yaw.append([0.0])
                else:
                    converted = to_motformer_box(next_box)
                    next_exist.append(True)
                    next_trans.append(converted[:3])
                    next_size.append(converted[3:6])
                    next_yaw.append([converted[6]])
            count = len(det_class)
            content = {
                "dets": {
                    "translation": _column(det_trans, 3, np.float32),
                    "size": _column(det_size, 3, np.float32),
                    "yaw": _column(det_yaw, 1, np.float32),
                    "velocity": np.zeros((count, 2), dtype=np.float32),
                    "class": _vector(det_class, np.int32),
                    "score": _vector(det_score, np.float32),
                },
                "gts": {
                    "translation": _column(gt_trans, 3, np.float32),
                    "size": _column(gt_size, 3, np.float32),
                    "yaw": _column(gt_yaw, 1, np.float32),
                    "class": _vector(gt_class, np.int32),
                    "tracking_id": _vector(gt_track, np.int32),
                    "next_exist": _vector(next_exist, np.bool_),
                    "next_translation": _column(next_trans, 3, np.float32),
                    "next_size": _column(next_size, 3, np.float32),
                    "next_yaw": _column(next_yaw, 1, np.float32),
                },
                "num_dets": int(count),
                "num_gts": int(len(gt_class)),
                "ego_translation": np.zeros(3, dtype=np.float32),
                "timestamp": np.int64(frame["timestamp"]),
                "token": str(frame["frame_id"]),
            }
            with (scene_dir / ("%04d.pkl" % index)).open("wb") as stream:
                pickle.dump(content, stream)
        print("完成序列%s 帧%d" % (sequence_id, len(ordered)))
    gaps = []
    previous = None
    for sequence_id in train_ids + [str(value) for value in split["val"]]:
        frames = list(read_jsonl(converted_root / by_id[sequence_id]["path"]))
        stats_frames = frames
        for frame in stats_frames:
            current = int(frame["timestamp"]) / 1e6
            if previous is not None and frame is stats_frames[0]:
                previous = None
            if previous is not None:
                gaps.append(current - previous)
            previous = current
    values = np.asarray(gaps, dtype=np.float64)
    stats = {
        "count": int(values.size),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "min": float(values.min()),
        "max": float(values.max()),
        "lidar_interval": float(config["train"]["lidar_interval"]),
        "train_sequences": train_ids,
        "held_out_calibration": sorted(calibration),
    }
    if abs(stats["median"] - stats["lidar_interval"]) > 0.02:
        raise SystemExit("帧间隔中位数%.4f与配置%.4f相差过大" % (stats["median"], stats["lidar_interval"]))
    write_json(_resolve(root, config["project"]["motformer_data_root"]) / "interval.json", stats)
    print("训练序列%d 帧间隔中位数%.4f" % (len(train_ids), stats["median"]))


if __name__ == "__main__":
    main()
