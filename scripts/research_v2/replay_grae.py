"""在线回放原始 GRAE，保存每帧关联前的轨迹状态与关联结果

推理只读取检测框和分数，不读取 GT
输出同时是 GRAE 的跟踪预测，可直接送入统一评估
"""

import argparse
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from inframot3d.config import load_config
from inframot3d.io import read_json, read_jsonl, write_json, write_jsonl
from inframot3d.tracking.grae_adapter import GraeTracker, build_model, load_checkpoint


def build_tracker(config, checkpoint, device="cuda"):
    model = build_model(
        ROOT / config["project"]["grae_root"],
        config["model"]["in_channels"],
        config["model"]["layers"],
        config["num_classes"],
        device,
    )
    load_checkpoint(model, checkpoint, device)
    birth = yaml.safe_load((ROOT / config["tracker"]["birth_thresholds_file"]).read_text(encoding="utf-8"))["score_thresholds"]
    tracker = GraeTracker(
        model,
        config["classes"],
        birth,
        association_alpha=config["tracker"]["association_alpha"],
        age=config["tracker"]["age"],
        score_floor=config["tracker"].get("score_floor", 0.1),
    )
    tracker.enable_debug(True)
    return tracker


def replay_sequence(tracker, det_rows):
    tracker.reset()
    rows = []
    states = []
    for det_row in det_rows:
        objects = tracker.update(det_row["objects"], int(det_row["timestamp"]) / 1e6, det_row["frame_id"])
        debug = tracker.last_debug or {}
        group = (debug.get("groups") or [{}])[0]
        assign = {}
        for item in group.get("assignments") or []:
            assign[int(item["input_index"])] = [int(item["track_id"]), item["stage"]]
        for item in group.get("created") or []:
            assign[int(item["input_index"])] = [int(item["track_id"]), "birth"]
        tracks = [
            {
                "track_id": int(item["track_id"]),
                "class_index": int(item["class_index"]),
                "box": item["box"],
                "score": item["score"],
                "age": int(item.get("age", 0)),
            }
            for item in group.get("pre_tracks") or []
        ]
        states.append(
            {
                "frame_id": det_row["frame_id"],
                "frame_index": det_row["frame_index"],
                "timestamp": det_row["timestamp"],
                "tracks": tracks,
                "assign": {str(k): v for k, v in assign.items()},
            }
        )
        rows.append(
            {
                "sequence_id": det_row["sequence_id"],
                "frame_index": det_row["frame_index"],
                "frame_id": det_row["frame_id"],
                "timestamp": det_row["timestamp"],
                "objects": objects,
            }
        )
    return rows, states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/trackers/grae/centerpoint.yaml")
    parser.add_argument("--ckpt", default="outputs/grae_centerpoint/ckpt/checkpoint-best.pth")
    parser.add_argument("--detections", required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    tracker = build_tracker(config, ROOT / args.ckpt)
    output = ROOT / args.output
    det_root = ROOT / args.detections
    start = time.perf_counter()
    frames = 0
    for sequence_id in args.sequences:
        det_rows = list(read_jsonl(det_root / ("%s.jsonl" % sequence_id)))
        rows, states = replay_sequence(tracker, det_rows)
        write_jsonl(output / "predictions" / ("%s.jsonl" % sequence_id), rows)
        write_jsonl(output / "states" / ("%s.jsonl" % sequence_id), states)
        frames += len(rows)
    write_json(
        output / "replay_meta.json",
        {
            "tracker": "GRAE-3DMOT",
            "checkpoint": args.ckpt,
            "detections": args.detections,
            "sequences": args.sequences,
            "frames": frames,
            "seconds": time.perf_counter() - start,
            "note": "GRAE 权重由全部 train 序列（除 4 个标定序列）的样本内检测训练，best epoch 1",
        },
    )
    print("frames", frames, "seconds", round(time.perf_counter() - start, 1))


if __name__ == "__main__":
    main()
