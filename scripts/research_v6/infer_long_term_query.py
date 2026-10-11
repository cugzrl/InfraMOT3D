"""Causal CenterPoint inference with long-term target-memory feedback."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from inframot3d.detection.openpcdet_adapter import prediction_to_object
from inframot3d.io import read_json, read_jsonl, write_json, write_jsonl
from inframot3d.perception.centerpoint_bev import batch_from_points, load_frozen_centerpoint
from inframot3d.perception.long_term_query_feedback import LongTermQueryFeedback
from inframot3d.perception.reliable_query_feedback import ReliableQueryFeedback

CONVERTED = ROOT / "data/converted/v2x_seq_infrastructure"
POINTS = ROOT / "data/centerpoint_v2xseq/points"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--disable-feedback", action="store_true")
    parser.add_argument("--reset-state-every-frame", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    saved = torch.load(ROOT / args.checkpoint, map_location=device, weights_only=False)
    fold = saved["args"]["fold"]
    model, dataset, load_gpu = load_frozen_centerpoint(
        ROOT / "third_party/OpenPCDet",
        ROOT / f"third_party/OpenPCDet/tools/cfgs/v2x_seq_models/centerpoint_fold{fold}.yaml",
        ROOT / saved["base_checkpoint"], device,
    )
    model.backbone_2d.load_state_dict(saved["backbone_2d"])
    model.dense_head.load_state_dict(saved["dense_head"])
    model.dense_head.model_cfg.POST_PROCESSING.SCORE_THRESH = 0.01
    reliable = saved["args"].get("architecture") == "reliable"
    query_type = ReliableQueryFeedback if reliable else LongTermQueryFeedback
    query = query_type(bev_channels=512, query_channels=64).to(device).eval()
    query.load_state_dict(saved["query"])
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
        queries = []
        for row in rows:
            start = time.perf_counter()
            points = np.load(POINTS / f"{seq}_{row['frame_id']}.npy")
            batch = batch_from_points(dataset, load_gpu, points, device)
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
            stamp = int(row["timestamp"]) / 1e6
            with torch.no_grad():
                for module in model.module_list:
                    if module is model.dense_head:
                        break
                    batch = module(batch)
                current = batch["spatial_features_2d"]
                if args.reset_state_every_frame:
                    state = None
                if reliable and not args.disable_feedback:
                    model.dense_head(batch)
                    current_predictions, _ = model.post_processing(batch)
                    current_prediction = current_predictions[0]
                    enhanced, state, info = query.read(
                        current, state, seq, stamp, enabled=True,
                        current_boxes=current_prediction["pred_boxes"],
                        current_scores=current_prediction["pred_scores"],
                        current_labels=current_prediction["pred_labels"],
                    )
                else:
                    enhanced, state, info = query.read(
                        current, state, seq, stamp,
                        enabled=not args.disable_feedback,
                    )
                batch["spatial_features_2d"] = enhanced
                batch = model.dense_head(batch)
                pred_dicts, _ = model.post_processing(batch)
                pred = pred_dicts[0]
                state = query.update(current, state if state is not None else query.empty(current, seq, stamp),
                                     pred["pred_boxes"], pred["pred_scores"], pred["pred_labels"], stamp)
                anno = dataset.generate_prediction_dicts(batch, pred_dicts, dataset.class_names)[0]
            objects = [prediction_to_object(name, score, box)
                       for name, score, box in zip(anno["name"], anno["score"], anno["boxes_lidar"])]
            detections.append({
                "sequence_id": seq, "frame_index": row["frame_index"],
                "frame_id": row["frame_id"], "timestamp": row["timestamp"],
                "objects": objects,
            })
            queries.append({
                "sequence_id": seq, "frame_id": row["frame_id"],
                "timestamp": row["timestamp"], "feedback_valid": bool(info["valid"]),
                "attention_entropy": (float(info["attention_entropy"])
                                      if "attention_entropy" in info else None),
                "attention_peak": (float(info["attention_peak"])
                                   if "attention_peak" in info else None),
                "attention_center": (float(info["attention_center"])
                                     if "attention_center" in info else None),
                "attention_offset_cells": (float(info["attention_offset_cells"])
                                           if "attention_offset_cells" in info else None),
                "feedback_gain": (float(info["feedback_gain"])
                                  if "feedback_gain" in info else None),
                "feedback_rms_ratio": (float(info["feedback_rms_ratio"])
                                       if "feedback_rms_ratio" in info else None),
                "route_mean": (float(info["route_mean"])
                               if "route_mean" in info else None),
                "agreement_mean": (float(info["agreement_mean"])
                                   if "agreement_mean" in info else None),
                "support_mean": (float(info["support_mean"])
                                 if "support_mean" in info else None),
                "objects": [
                    {"query_id": int(state.ids[i]), "box": state.boxes[i].tolist(),
                     "velocity": state.velocity[i].tolist(), "class_index": int(state.labels[i]),
                     "confidence": float(state.confidence[i]),
                     "uncertainty": float(state.uncertainty[i]),
                     "misses": int(state.misses[i]), "hits": int(state.hits[i])}
                    for i in range(len(state.ids))
                ],
            })
            durations.append(time.perf_counter() - start)
        write_jsonl(output / f"{seq}.jsonl", detections)
        write_jsonl(output / "queries" / f"{seq}.jsonl", queries)
        print(seq, "frames", len(rows), "detections", sum(len(item["objects"]) for item in detections),
              "mean_queries", round(np.mean([len(item["objects"]) for item in queries]), 2), flush=True)
    write_json(output / "manifest.json", {
        "checkpoint": args.checkpoint, "epoch": saved["epoch"],
        "sequences": args.sequences, "disable_feedback": args.disable_feedback,
        "reset_state_every_frame": args.reset_state_every_frame,
        "mean_seconds_per_frame": float(np.mean(durations)),
        "note": "Queries are born and updated only from online model detections; feedback enters BEV before CenterHead.",
    })


if __name__ == "__main__":
    main()
