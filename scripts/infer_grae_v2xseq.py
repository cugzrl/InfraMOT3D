import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inframot3d.config import load_config
from inframot3d.io import read_json, read_jsonl, write_json, write_jsonl
from inframot3d.tracking.grae_adapter import GraeTracker, build_model, load_checkpoint


def _resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def _best_checkpoint(output_root):
    best = output_root / "ckpt" / "checkpoint-best.pth"
    if best.is_file():
        return best
    checkpoints = list((output_root / "ckpt").glob("checkpoint-epoch*.pth"))
    if not checkpoints:
        raise FileNotFoundError("未找到GRAE checkpoint")
    return max(checkpoints, key=lambda path: int(path.stem.replace("checkpoint-epoch", "")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grae_centerpoint.yaml")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--sequences", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config)
    root = config["_root"]
    device = "cuda"
    model = build_model(
        _resolve(root, config["project"]["grae_root"]),
        config["model"]["in_channels"],
        config["model"]["layers"],
        config["num_classes"],
        device,
    )
    checkpoint = Path(args.ckpt) if args.ckpt else _best_checkpoint(Path(config["project"]["output_root"]))
    print("使用checkpoint %s" % checkpoint.name)
    load_checkpoint(model, checkpoint, device)
    tracker = GraeTracker(
        model,
        config["classes"],
        config["tracker"]["alpha"],
        config["tracker"]["conf_threshold"],
        config["tracker"]["age"],
    )
    split_file = _resolve(root, config["split_file"])
    allowed = set(read_json(split_file)[args.split])
    if args.sequences:
        allowed &= set(args.sequences)
    detection_root = _resolve(root, config["input"]["detection_root"])
    converted_root = Path(config["project"]["converted_root"])
    manifest = read_json(converted_root / "manifest.json")
    output_root = Path(config["project"]["output_root"])
    prediction_root = output_root / "predictions"
    start = time.perf_counter()
    total_frames = 0
    total_objects = 0
    for entry in manifest["sequences"]:
        sequence_id = entry["sequence_id"]
        if sequence_id not in allowed:
            continue
        tracker.reset()
        rows = []
        detections = list(read_jsonl(detection_root / ("%s.jsonl" % sequence_id)))
        ground_truth = list(read_jsonl(converted_root / entry["path"]))
        if len(detections) != len(ground_truth):
            raise ValueError("检测帧数不一致 %s" % sequence_id)
        for det_row, gt_row in zip(detections, ground_truth):
            if det_row["frame_id"] != gt_row["frame_id"] or int(det_row["timestamp"]) != int(gt_row["timestamp"]):
                raise ValueError("检测帧未对齐 %s" % sequence_id)
            objects = tracker.update(det_row["objects"], int(det_row["timestamp"]) / 1e6, det_row["frame_id"])
            rows.append(
                {
                    "sequence_id": det_row["sequence_id"],
                    "frame_index": det_row["frame_index"],
                    "frame_id": det_row["frame_id"],
                    "timestamp": det_row["timestamp"],
                    "objects": objects,
                }
            )
            total_frames += 1
            total_objects += len(objects)
        write_jsonl(prediction_root / ("%s.jsonl" % sequence_id), rows)
        print("完成序列%s 帧%d" % (sequence_id, len(rows)))
    elapsed = time.perf_counter() - start
    write_json(
        output_root / "runtime.json",
        {
            "tracker": "GRAE-3DMOT",
            "input": "detection",
            "checkpoint": str(checkpoint),
            "detection_root": str(detection_root),
            "num_frames": total_frames,
            "num_output_objects": total_objects,
            "elapsed_seconds": elapsed,
            "fps": total_frames / elapsed if elapsed else 0.0,
        },
    )
    print("跟踪完成 帧%d 用时%.2f秒" % (total_frames, elapsed))


if __name__ == "__main__":
    main()
