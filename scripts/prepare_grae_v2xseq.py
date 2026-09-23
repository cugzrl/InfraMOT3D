import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inframot3d.config import load_config
from inframot3d.io import read_json, read_jsonl, write_json
from inframot3d.tracking.grae_adapter import detection_records, match_detection_frame


def _resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def _track_index(mapping, source_id):
    key = str(source_id)
    if key not in mapping:
        mapping[key] = len(mapping) + 1
    return mapping[key]


def _convert_sequence(detection_path, gt_path, class_to_index):
    detections = list(read_jsonl(detection_path))
    ground_truth = list(read_jsonl(gt_path))
    if len(detections) != len(ground_truth):
        raise ValueError("帧数不一致 %s" % detection_path)
    mapping = {}
    frames = []
    matched = 0
    total = 0
    for det_row, gt_row in zip(detections, ground_truth):
        if det_row["frame_id"] != gt_row["frame_id"] or int(det_row["timestamp"]) != int(gt_row["timestamp"]):
            raise ValueError("帧未对齐 %s" % detection_path)
        gt_objects = []
        for item in gt_row["objects"]:
            if item["class_name"] not in class_to_index:
                continue
            gt_objects.append(
                {
                    "class_name": item["class_name"],
                    "box": item["box"],
                    "track_index": _track_index(mapping, item["source_track_id"]),
                }
            )
        objects = [
            {"class_name": item["class_name"], "score": float(item["score"]), "box": [float(value) for value in item["box"]]}
            for item in det_row["objects"]
        ]
        frame = detection_records(objects, class_to_index, int(det_row["timestamp"]) / 1e6)
        frame["sample_token"] = str(det_row["frame_id"])
        frame["sequence_id"] = str(det_row["sequence_id"])
        tracking_ids, hit = match_detection_frame(objects, gt_objects, class_to_index)
        frame["tracking_id"] = tracking_ids
        matched += hit
        total += len(objects)
        frames.append(frame)
    return frames, matched, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grae_centerpoint.yaml")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    args = parser.parse_args()
    config = load_config(args.config)
    root = config["_root"]
    classes = list(config["classes"])
    class_to_index = {name: index for index, name in enumerate(classes)}
    split_ids = set(read_json(_resolve(root, config["split_file"]))[args.split])
    detection_root = _resolve(root, config["input"]["detection_root"])
    converted_root = Path(config["project"]["converted_root"])
    manifest = read_json(converted_root / "manifest.json")
    sequences = []
    matched = 0
    total = 0
    for entry in manifest["sequences"]:
        sequence_id = entry["sequence_id"]
        if sequence_id not in split_ids:
            continue
        frames, hit, count = _convert_sequence(
            detection_root / ("%s.jsonl" % sequence_id),
            converted_root / entry["path"],
            class_to_index,
        )
        sequences.append(frames)
        matched += hit
        total += count
        print("序列%s 检测%d 匹配%d" % (sequence_id, count, hit))
    output_root = _resolve(root, config["project"]["grae_data_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / ("v2x_seq_%s.pkl" % args.split)
    with target.open("wb") as stream:
        pickle.dump(sequences, stream)
    ratio = matched / total if total else 0.0
    write_json(
        output_root / ("match_%s.json" % args.split),
        {"split": args.split, "sequences": len(sequences), "detections": total, "matched": matched, "ratio": ratio},
    )
    print("匹配 %d / %d %.2f%%" % (matched, total, ratio * 100.0))


if __name__ == "__main__":
    main()
