import argparse
import time
from pathlib import Path

import yaml

from inframot3d.config import load_config
from inframot3d.io import read_json, read_jsonl, write_json, write_jsonl
from inframot3d.tracking import create_tracker


def _allowed_sequences(config, sequences, split):
    allowed = set(sequences) if sequences else None
    if not split:
        return allowed
    split_file = Path(config.get("split_file", "configs/datasets/v2xseq_sequence_split.json"))
    if not split_file.is_absolute():
        split_file = config["_root"] / split_file
    split_ids = set(read_json(split_file)[split])
    return split_ids if allowed is None else allowed & split_ids


def _resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_inputs(config, converted_root, entry):
    input_type = (config.get("input") or {}).get("type", "ground_truth")
    frames = list(read_jsonl(converted_root / entry["path"]))
    if input_type == "ground_truth":
        return input_type, None, frames
    if input_type != "detection":
        raise ValueError("未知输入类型%s" % input_type)
    detection_root = _resolve(config["_root"], config["input"]["detection_root"])
    detections = list(read_jsonl(detection_root / ("%s.jsonl" % entry["sequence_id"])))
    if len(detections) != len(frames):
        raise ValueError("检测帧数与转换数据不一致 %s" % entry["sequence_id"])
    by_id = {row["frame_id"]: row for row in detections}
    aligned = []
    for frame in frames:
        row = by_id.get(frame["frame_id"])
        if row is None or row["sequence_id"] != frame["sequence_id"] or int(row["timestamp"]) != int(frame["timestamp"]):
            raise ValueError("检测帧未对齐 %s %s" % (entry["sequence_id"], frame["frame_id"]))
        if int(row["frame_index"]) != int(frame["frame_index"]):
            raise ValueError("检测帧序号未对齐 %s %s" % (entry["sequence_id"], frame["frame_id"]))
        aligned.append(row)
    return input_type, detection_root, aligned


def _apply_score_thresholds(config):
    tracker = config.get("tracker") or {}
    relative = tracker.get("score_thresholds_file")
    if not relative:
        return
    path = Path(relative)
    if not path.is_absolute():
        path = config["_root"] / path
    if not path.is_file():
        raise FileNotFoundError("缺少类别分数阈值 %s" % path)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    tracker["score_thresholds"] = {str(name): float(value) for name, value in loaded["score_thresholds"].items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/trackers/ab3dmot/gt.yaml")
    parser.add_argument("--sequences", nargs="*")
    parser.add_argument("--split", choices=["train", "val", "test"])
    args = parser.parse_args()
    config = load_config(args.config)
    _apply_score_thresholds(config)
    converted_root = Path(config["project"]["converted_root"])
    output_root = Path(config["project"]["output_root"])
    prediction_root = output_root / "predictions"
    manifest = read_json(converted_root / "manifest.json")
    allowed = _allowed_sequences(config, args.sequences, args.split)
    input_type = (config.get("input") or {}).get("type", "ground_truth")
    detection_root = None
    total_frames = 0
    total_objects = 0
    start = time.perf_counter()
    for entry in manifest["sequences"]:
        sequence_id = entry["sequence_id"]
        if allowed is not None and sequence_id not in allowed:
            continue
        input_type, detection_root, frames = _load_inputs(config, converted_root, entry)
        tracker = create_tracker(config["tracker"])
        rows = []
        for frame in frames:
            # source_track_id只留给评估，不进入跟踪器
            objects = [
                {"class_name": item["class_name"], "score": float(item.get("score", 1.0)), "box": item["box"]}
                for item in frame["objects"]
            ]
            objects = tracker.update(objects, timestamp=frame["timestamp"])
            rows.append(
                {
                    "sequence_id": frame["sequence_id"],
                    "frame_index": frame["frame_index"],
                    "frame_id": frame["frame_id"],
                    "timestamp": frame["timestamp"],
                    "objects": objects,
                }
            )
            total_frames += 1
            total_objects += len(objects)
        write_jsonl(prediction_root / f"{sequence_id}.jsonl", rows)
        print(f"完成序列{sequence_id} 帧{len(rows)}")
    elapsed = time.perf_counter() - start
    runtime = {
        "tracker": config["tracker"]["name"],
        "input": input_type,
        "num_frames": total_frames,
        "num_output_objects": total_objects,
        "elapsed_seconds": elapsed,
        "fps": total_frames / elapsed if elapsed else 0.0,
    }
    if detection_root is not None:
        runtime["detection_root"] = str(detection_root)
    write_json(output_root / "runtime.json", runtime)
    print(f"跟踪完成 帧{total_frames} 用时{elapsed:.2f}秒 FPS{runtime['fps']:.2f}")


if __name__ == "__main__":
    main()
