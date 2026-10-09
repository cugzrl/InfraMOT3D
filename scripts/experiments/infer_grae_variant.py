import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from inframot3d.config import load_config
from inframot3d.io import read_json, read_jsonl, write_json, write_jsonl
from inframot3d.tracking.grae_adapter import GraeTracker, build_model, load_checkpoint
from inframot3d.tracking.grae_recovery import TrackConditionedRecovery
from inframot3d.tracking.grae_semantic import VehicleSemanticGate


def _resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_recovery(path, channels, device):
    if not path:
        return None
    module = TrackConditionedRecovery(int(channels)).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint["recovery"] if isinstance(checkpoint, dict) and "recovery" in checkpoint else checkpoint
    module.load_state_dict(state)
    module.eval()
    return module


def load_semantic(path, channels, num_classes, device):
    if not path:
        return None
    module = VehicleSemanticGate(int(channels), int(num_classes)).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint["semantic"] if isinstance(checkpoint, dict) and "semantic" in checkpoint else checkpoint
    module.load_state_dict(state)
    module.eval()
    return module


def track_sequences(
    config,
    checkpoint,
    prediction_root,
    split="val",
    sequences=None,
    score_floor=0.1,
    class_compat="exact",
    motion_mode="zero",
    recovery_path=None,
    device="cuda",
    weak_association="fusion",
    semantic_path=None,
):
    root = config["_root"]
    model = build_model(
        _resolve(root, config["project"]["grae_root"]),
        config["model"]["in_channels"],
        config["model"]["layers"],
        config["num_classes"],
        device,
    )
    load_checkpoint(model, checkpoint, device)
    recovery = load_recovery(recovery_path, config["model"]["in_channels"], device)
    semantic = load_semantic(semantic_path, config["model"]["in_channels"], config["num_classes"], device)
    birth_thresholds = yaml.safe_load(_resolve(root, config["tracker"]["birth_thresholds_file"]).read_text(encoding="utf-8"))[
        "score_thresholds"
    ]
    tracker = GraeTracker(
        model,
        config["classes"],
        birth_thresholds,
        association_alpha=config["tracker"]["association_alpha"],
        age=config["tracker"]["age"],
        score_floor=float(score_floor),
        class_compat=class_compat,
        motion_mode=motion_mode,
        recovery=recovery,
        weak_association=weak_association,
        semantic=semantic,
    )
    allowed = set(read_json(_resolve(root, config["split_file"]))[split])
    if sequences:
        allowed &= set(sequences)
    detection_root = _resolve(root, config["input"]["detection_root"])
    converted_root = Path(config["project"]["converted_root"])
    manifest = read_json(converted_root / "manifest.json")
    prediction_root = Path(prediction_root)
    prediction_root.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    total_frames = 0
    total_objects = 0
    used = []
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
            # 推理不读取 gt
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
        used.append(sequence_id)
        print("完成序列%s 帧%d" % (sequence_id, len(rows)), flush=True)
    elapsed = time.perf_counter() - start
    runtime = {
        "checkpoint": str(checkpoint),
        "recovery": None if recovery_path is None else str(recovery_path),
        "score_floor": float(score_floor),
        "class_compat": class_compat,
        "motion_mode": motion_mode,
        "weak_association": weak_association,
        "semantic": None if semantic_path is None else str(semantic_path),
        "split": split,
        "sequences": used,
        "num_frames": total_frames,
        "num_output_objects": total_objects,
        "elapsed_seconds": elapsed,
        "fps": total_frames / elapsed if elapsed else 0.0,
    }
    return runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/trackers/grae/centerpoint.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--sequences", nargs="*")
    parser.add_argument("--score-floor", type=float, default=0.1)
    parser.add_argument("--class-compat", default="exact", choices=["exact", "superclass"])
    parser.add_argument("--motion-mode", default="zero", choices=["zero", "constant_velocity"])
    parser.add_argument("--recovery", default=None)
    parser.add_argument("--weak-association", default="fusion", choices=["fusion", "learned"])
    parser.add_argument("--semantic", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_config(args.config)
    runtime = track_sequences(
        config,
        args.ckpt,
        Path(args.output),
        split=args.split,
        sequences=args.sequences,
        score_floor=args.score_floor,
        class_compat=args.class_compat,
        motion_mode=args.motion_mode,
        recovery_path=args.recovery,
        device=args.device,
        weak_association=args.weak_association,
        semantic_path=args.semantic,
    )
    write_json(Path(args.output) / "runtime.json", runtime)
    print("跟踪完成 帧%d 用时%.2f秒" % (runtime["num_frames"], runtime["elapsed_seconds"]), flush=True)


if __name__ == "__main__":
    main()
