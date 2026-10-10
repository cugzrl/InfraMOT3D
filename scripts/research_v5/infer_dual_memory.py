"""Run causal CenterPoint detection with a trained dual BEV memory."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/research_v3"))

from inframot3d.detection.openpcdet_adapter import prediction_to_object
from inframot3d.io import read_json, read_jsonl, write_json, write_jsonl
from inframot3d.perception.centerpoint_bev import batch_from_points, load_frozen_centerpoint
from inframot3d.perception.dual_bev_memory import DualBEVMemory
from infer_bev_memory import lidar_hit_map

CONVERTED = ROOT / "data/converted/v2x_seq_infrastructure"
POINTS = ROOT / "data/centerpoint_v2xseq/points"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--disable-memory", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    saved = torch.load(ROOT / args.checkpoint, map_location=device, weights_only=False)
    train_args = saved["args"]
    model, dataset, load_gpu = load_frozen_centerpoint(
        ROOT / "third_party/OpenPCDet", ROOT / f"third_party/OpenPCDet/tools/cfgs/v2x_seq_models/centerpoint_fold{train_args['fold']}.yaml",
        ROOT / saved["base_checkpoint"], device,
    )
    model.backbone_2d.load_state_dict(saved["backbone_2d"])
    model.dense_head.load_state_dict(saved["dense_head"])
    model.dense_head.model_cfg.POST_PROCESSING.SCORE_THRESH = 0.01
    memory = DualBEVMemory(
        input_channels=512, hidden_channels=train_args["hidden_channels"],
        memory_stride=train_args["memory_stride"],
    ).to(device).eval()
    memory.load_state_dict(saved["memory"])
    manifest = read_json(CONVERTED / "manifest.json")
    paths = {item["sequence_id"]: item["path"] for item in manifest["sequences"]}
    output = ROOT / args.output
    durations = []
    for seq in args.sequences:
        rows = list(read_jsonl(CONVERTED / paths[seq]))
        if args.max_frames:
            rows = rows[:args.max_frames]
        state = None
        detections = []
        for row in rows:
            start = time.perf_counter()
            points = np.load(POINTS / f"{seq}_{row['frame_id']}.npy")
            batch = batch_from_points(dataset, load_gpu, points, device)
            with torch.no_grad():
                for module in model.module_list:
                    if module is model.dense_head:
                        break
                    batch = module(batch)
                current = batch["spatial_features_2d"]
                hit = lidar_hit_map(points, current.shape[-2], current.shape[-1], device)
                enhanced, state, _ = memory(
                    current, state, seq, int(row["timestamp"]) / 1e6, hit,
                    enabled=not args.disable_memory,
                )
                batch["spatial_features_2d"] = enhanced
                batch = model.dense_head(batch)
                pred_dicts, _ = model.post_processing(batch)
                anno = dataset.generate_prediction_dicts(batch, pred_dicts, dataset.class_names)[0]
            objects = [
                prediction_to_object(name, score, box)
                for name, score, box in zip(anno["name"], anno["score"], anno["boxes_lidar"])
            ]
            detections.append({
                "sequence_id": seq, "frame_index": row["frame_index"],
                "frame_id": row["frame_id"], "timestamp": row["timestamp"],
                "objects": objects,
            })
            durations.append(time.perf_counter() - start)
        write_jsonl(output / f"{seq}.jsonl", detections)
        print(seq, "frames", len(rows), "detections", sum(len(item["objects"]) for item in detections), flush=True)
    write_json(output / "manifest.json", {
        "checkpoint": args.checkpoint, "epoch": saved["epoch"],
        "sequences": args.sequences, "disable_memory": args.disable_memory,
        "max_frames": args.max_frames, "mean_seconds_per_frame": float(np.mean(durations)),
        "note": "Online causal memory; no GT, future frames, or precomputed D0 tracks at inference.",
    })


if __name__ == "__main__":
    main()
