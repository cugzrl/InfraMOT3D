import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from inframot3d.config import load_config
from inframot3d.io import read_json, read_jsonl, write_jsonl
from inframot3d.tracking.motformer_adapter import (
    MotformerTracker,
    build_model,
    configure_runtime,
    load_checkpoint,
)


def _resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def _best_checkpoint(output_root):
    best = output_root / "ckpt" / "checkpoint-best.pth"
    if best.is_file():
        return best
    checkpoints = list((output_root / "ckpt").glob("checkpoint-epoch*.pth"))
    if not checkpoints:
        raise FileNotFoundError("未找到3DMOTFormer checkpoint")
    return max(checkpoints, key=lambda path: int(path.stem.replace("checkpoint-epoch", "")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/trackers/3dmotformer/centerpoint.yaml")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--sequences", nargs="*")
    parser.add_argument("--prediction-root", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    root = config["_root"]
    motformer_root = _resolve(root, config["project"]["motformer_root"])
    configure_runtime(
        motformer_root,
        config["classes"],
        config["train"]["lidar_interval"],
        config["train"]["max_velo"],
    )
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model = build_model(motformer_root, config["model"], config["num_classes"], device)
    output_root = Path(config["project"]["output_root"])
    checkpoint = Path(args.ckpt) if args.ckpt else _best_checkpoint(output_root)
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    print("使用checkpoint %s" % checkpoint.name)
    load_checkpoint(model, checkpoint, device)
    tracker = MotformerTracker(
        model,
        config["classes"],
        config["tracker"]["max_age"],
        config["tracker"]["active_track_thresh"],
        config["model"]["graph_truncation_dist"],
        score_threshold=config["tracker"]["score_threshold"],
    )
    split_ids = set(read_json(_resolve(root, config["split_file"]))[args.split])
    if args.sequences:
        split_ids &= set(args.sequences)
    detection_root = _resolve(root, config["input"]["detection_root"])
    converted_root = Path(config["project"]["converted_root"])
    manifest = read_json(converted_root / "manifest.json")
    prediction_root = Path(args.prediction_root) if args.prediction_root else output_root / "predictions"
    if not prediction_root.is_absolute():
        prediction_root = root / prediction_root
    start = time.perf_counter()
    total_frames = 0
    total_objects = 0
    for entry in manifest["sequences"]:
        sequence_id = entry["sequence_id"]
        if sequence_id not in split_ids:
            continue
        tracker.reset()
        detections = list(read_jsonl(detection_root / ("%s.jsonl" % sequence_id)))
        ground_truth = list(read_jsonl(converted_root / entry["path"]))
        if len(detections) != len(ground_truth):
            raise SystemExit("检测帧数与转换数据不一致 %s" % sequence_id)
        by_id = {row["frame_id"]: row for row in detections}
        rows = []
        for frame in ground_truth:
            row = by_id.get(frame["frame_id"])
            if row is None or int(row["timestamp"]) != int(frame["timestamp"]):
                raise SystemExit("检测帧未对齐 %s %s" % (sequence_id, frame["frame_id"]))
            objects = [
                {"class_name": item["class_name"], "score": float(item["score"]), "box": item["box"]}
                for item in row["objects"]
            ]
            outputs = tracker.update(objects, timestamp=int(row["timestamp"]) / 1e6)
            for item in outputs:
                if item["track_id"] < 0 or not math.isfinite(item["score"]):
                    raise SystemExit("预测不合法 %s" % sequence_id)
                if not all(math.isfinite(value) for value in item["box"]):
                    raise SystemExit("预测框含非有限值 %s" % sequence_id)
            rows.append(
                {
                    "sequence_id": row["sequence_id"],
                    "frame_index": row["frame_index"],
                    "frame_id": row["frame_id"],
                    "timestamp": row["timestamp"],
                    "objects": outputs,
                }
            )
            total_frames += 1
            total_objects += len(outputs)
        write_jsonl(prediction_root / ("%s.jsonl" % sequence_id), rows)
        print("完成序列%s 帧%d" % (sequence_id, len(rows)))
    print("推理完成 帧%d 目标%d 用时%.1f秒" % (total_frames, total_objects, time.perf_counter() - start))


if __name__ == "__main__":
    main()
