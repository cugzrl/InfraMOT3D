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
from train_dual_memory import labels

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
    parser.add_argument("--reset-state-every-frame", action="store_true")
    parser.add_argument("--diagnostics-output")
    parser.add_argument("--diagnostics-only", action="store_true")
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
    diagnostic = {
        "frames_with_history": 0, "route_sum": [0.0, 0.0, 0.0],
        "route_dynamic_sum": [0.0, 0.0, 0.0],
        "route_background_sum": [0.0, 0.0, 0.0],
        "route_spatial_std_sum": [0.0, 0.0, 0.0],
        "pixels": 0, "dynamic_pixels": 0, "background_pixels": 0,
        "flow_epe_sum": 0.0, "flow_moving_epe_sum": 0.0,
        "flow_pred_magnitude_sum": 0.0, "flow_target_magnitude_sum": 0.0,
        "flow_pixels": 0, "moving_pixels": 0,
        "dynamic_intersection": 0, "dynamic_union": 0,
    }
    for seq in args.sequences:
        rows = list(read_jsonl(CONVERTED / paths[seq]))
        if args.max_frames:
            rows = rows[:args.max_frames]
        state = None
        detections = []
        for index, row in enumerate(rows):
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
                if args.reset_state_every_frame:
                    state = None
                enhanced, state, aux = memory(
                    current, state, seq, int(row["timestamp"]) / 1e6, hit,
                    enabled=not args.disable_memory,
                )
                if args.diagnostics_output and aux and aux["memory_valid"]:
                    occupancy, matched, target_flow = labels(
                        row, rows[index - 1] if index else None,
                        aux["flow"].shape, device, memory.max_dynamic_gap,
                    )
                    route = aux["route"]
                    area = occupancy.bool()
                    background = ~area
                    diagnostic["frames_with_history"] += 1
                    diagnostic["pixels"] += route.shape[-2] * route.shape[-1]
                    diagnostic["dynamic_pixels"] += int(area.sum())
                    diagnostic["background_pixels"] += int(background.sum())
                    means = route.mean(dim=(-2, -1))[0]
                    stds = route.std(dim=(-2, -1))[0]
                    for channel in range(3):
                        diagnostic["route_sum"][channel] += float(route[:, channel].sum())
                        diagnostic["route_dynamic_sum"][channel] += float(route[:, channel][area[:, 0]].sum())
                        diagnostic["route_background_sum"][channel] += float(route[:, channel][background[:, 0]].sum())
                        diagnostic["route_spatial_std_sum"][channel] += float(stds[channel])
                    predicted_mask = aux["dynamic_logits"].sigmoid() >= 0.5
                    diagnostic["dynamic_intersection"] += int((predicted_mask & area).sum())
                    diagnostic["dynamic_union"] += int((predicted_mask | area).sum())
                    if aux["dynamic_valid"] and matched.any():
                        epe = (aux["flow"] - target_flow).square().sum(dim=1, keepdim=True).sqrt()
                        pred_mag = aux["flow"].square().sum(dim=1, keepdim=True).sqrt()
                        target_mag = target_flow.square().sum(dim=1, keepdim=True).sqrt()
                        moving = matched & (target_mag >= 0.5)
                        diagnostic["flow_epe_sum"] += float(epe[matched].sum())
                        diagnostic["flow_pred_magnitude_sum"] += float(pred_mag[matched].sum())
                        diagnostic["flow_target_magnitude_sum"] += float(target_mag[matched].sum())
                        diagnostic["flow_pixels"] += int(matched.sum())
                        diagnostic["flow_moving_epe_sum"] += float(epe[moving].sum())
                        diagnostic["moving_pixels"] += int(moving.sum())
                batch["spatial_features_2d"] = enhanced
                batch = model.dense_head(batch)
                pred_dicts, _ = model.post_processing(batch)
                anno = dataset.generate_prediction_dicts(batch, pred_dicts, dataset.class_names)[0]
            if not args.diagnostics_only:
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
        if not args.diagnostics_only:
            write_jsonl(output / f"{seq}.jsonl", detections)
        print(seq, "frames", len(rows), "detections", sum(len(item["objects"]) for item in detections), flush=True)
    write_json(output / "manifest.json", {
        "checkpoint": args.checkpoint, "epoch": saved["epoch"],
        "sequences": args.sequences, "disable_memory": args.disable_memory,
        "reset_state_every_frame": args.reset_state_every_frame,
        "max_frames": args.max_frames, "mean_seconds_per_frame": float(np.mean(durations)),
        "note": "Online causal memory; no GT, future frames, or precomputed D0 tracks at inference.",
    })
    if args.diagnostics_output:
        pixels = max(diagnostic["pixels"], 1)
        dyn = max(diagnostic["dynamic_pixels"], 1)
        background = max(diagnostic["background_pixels"], 1)
        diagnostic["route_mean"] = [x / pixels for x in diagnostic["route_sum"]]
        diagnostic["route_dynamic_mean"] = [x / dyn for x in diagnostic["route_dynamic_sum"]]
        diagnostic["route_background_mean"] = [x / background for x in diagnostic["route_background_sum"]]
        diagnostic["route_spatial_std_mean"] = [
            x / max(diagnostic["frames_with_history"], 1) for x in diagnostic["route_spatial_std_sum"]
        ]
        diagnostic["dynamic_iou"] = diagnostic["dynamic_intersection"] / max(diagnostic["dynamic_union"], 1)
        diagnostic["flow_epe_cells"] = diagnostic["flow_epe_sum"] / max(diagnostic["flow_pixels"], 1)
        diagnostic["flow_moving_epe_cells"] = diagnostic["flow_moving_epe_sum"] / max(diagnostic["moving_pixels"], 1)
        diagnostic["flow_pred_magnitude_cells"] = diagnostic["flow_pred_magnitude_sum"] / max(diagnostic["flow_pixels"], 1)
        diagnostic["flow_target_magnitude_cells"] = diagnostic["flow_target_magnitude_sum"] / max(diagnostic["flow_pixels"], 1)
        diagnostic["note"] = "GT is used only for this post-forward offline diagnostic, never as an inference input."
        write_json(ROOT / args.diagnostics_output, diagnostic)


if __name__ == "__main__":
    main()
