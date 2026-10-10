"""Causal, frozen CenterPoint pre-head BEV fusion feasibility test.

This deliberately small D0/D1 experiment changes spatial_features_2d before
CenterHead. It does not train or alter the detector checkpoint. Sequence state is
reset at every sequence boundary and when timestamps jump beyond max_gap.
"""

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from inframot3d.detection.openpcdet_adapter import prediction_to_object
from inframot3d.io import read_json, read_jsonl, write_json, write_jsonl
from inframot3d.perception.centerpoint_bev import batch_from_points, load_frozen_centerpoint
from inframot3d.perception.scene_memory import LongTermSceneMemory, SceneMemoryState
from static_gate import StaticGate

CONVERTED = ROOT / "data/converted/v2x_seq_infrastructure"
POINTS = ROOT / "data/centerpoint_v2xseq/points"
CELL_METERS = 0.8
Y_ORIGIN = -56.0


def lidar_hit_map(points, height, width, device):
    """Actual-return evidence per BEV cell; no GT or learned visibility labels."""
    xy = np.asarray(points[:, :2], dtype=np.float32)
    col = np.floor(xy[:, 0] / CELL_METERS).astype(np.int32)
    row = np.floor((xy[:, 1] - Y_ORIGIN) / CELL_METERS).astype(np.int32)
    ok = (row >= 0) & (row < height) & (col >= 0) & (col < width)
    count = np.bincount(row[ok] * width + col[ok], minlength=height * width)
    hit = torch.from_numpy(count.reshape(1, 1, height, width)).to(device=device, dtype=torch.float32)
    return (hit / 3.0).clamp_(0.0, 1.0)


def dynamic_mask(tracks, height, width, device):
    yy, xx = torch.meshgrid(torch.arange(height, device=device),
                            torch.arange(width, device=device), indexing="ij")
    mask = torch.zeros((1, 1, height, width), dtype=torch.bool, device=device)
    for obj in tracks:
        if obj["score"] < 0.2:
            continue
        box = obj["box"]
        x = float(box[0]) / CELL_METERS
        y = (float(box[1]) - Y_ORIGIN) / CELL_METERS
        mask[0, 0] |= ((xx - x).square() + (yy - y).square() <= 25.0)
    return mask


def motion_velocity(track_rows, index):
    """Estimate velocity from observed online outputs no later than t-1."""
    if index < 2:
        return {}
    a = track_rows[index - 2]
    b = track_rows[index - 1]
    dt = (int(b["timestamp"]) - int(a["timestamp"])) / 1e6
    if dt <= 0 or dt > 0.3:
        return {}
    old = {int(x["track_id"]): x for x in a["objects"] if x["score"] >= 0.2}
    velocities = {}
    for obj in b["objects"]:
        tid = int(obj["track_id"])
        if obj["score"] < 0.2 or tid not in old:
            continue
        velocity = (np.asarray(obj["box"][:2]) - np.asarray(old[tid]["box"][:2])) / dt
        if np.linalg.norm(velocity) <= 30.0:
            velocities[tid] = velocity
    return velocities


def warp_dynamic_feature(source, current, source_tracks, velocities, dt):
    """Shift local BEV evidence around online tracks, leaving static cells fixed."""
    if not velocities or dt <= 0:
        return source
    _, _, height, width = source.shape
    yy, xx = torch.meshgrid(torch.arange(height, device=source.device),
                            torch.arange(width, device=source.device), indexing="ij")
    sx = xx.float().clone()
    sy = yy.float().clone()
    old_area = torch.zeros((height, width), dtype=torch.bool, device=source.device)
    new_area = torch.zeros_like(old_area)
    for obj in source_tracks:
        tid = int(obj["track_id"])
        if tid not in velocities or obj["score"] < 0.2:
            continue
        xy = np.asarray(obj["box"][:2], dtype=float)
        delta = velocities[tid] * dt / CELL_METERS
        old_x = xy[0] / CELL_METERS
        old_y = (xy[1] - Y_ORIGIN) / CELL_METERS
        new_x, new_y = old_x + delta[0], old_y + delta[1]
        radius = 5.0  # 4 m, deliberately fixed for the D2 feasibility test
        old_area |= (xx - old_x).square() + (yy - old_y).square() <= radius ** 2
        target = (xx - new_x).square() + (yy - new_y).square() <= radius ** 2
        new_area |= target
        sx[target] = xx[target] - float(delta[0])
        sy[target] = yy[target] - float(delta[1])
    grid = torch.stack([(sx + 0.5) * (2.0 / width) - 1.0,
                        (sy + 0.5) * (2.0 / height) - 1.0], -1).unsqueeze(0)
    warped = F.grid_sample(source, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    stale = old_area & ~new_area
    warped[:, :, stale] = current[:, :, stale]
    return warped


def infer_sequence(model, dataset, load_gpu, rows, seq, window, history_weight,
                   max_gap, max_frames, mode, track_rows, static_weight, static_shuffle,
                   gate_module):
    history = deque(maxlen=window - 1)
    memory_state = None
    scene_memory = None
    result = []
    timings = []
    if model.module_list[-1] is not model.dense_head:
        raise RuntimeError("Unexpected CenterPoint layout: dense head is not last")
    for index, row in enumerate(rows[:max_frames] if max_frames else rows):
        timestamp = int(row["timestamp"]) / 1e6
        if history and timestamp - history[-1][0] > max_gap:
            history.clear()
            memory_state = None
        start = time.perf_counter()
        points = np.load(POINTS / f"{seq}_{row['frame_id']}.npy")
        batch = batch_from_points(dataset, load_gpu, points, torch.device("cuda"))
        with torch.no_grad():
            for module in model.module_list:
                if module is model.dense_head:
                    break
                batch = module(batch)
            current = batch["spatial_features_2d"]
            _, _, height, width = current.shape
            hit = lidar_hit_map(points, height, width, current.device) if mode == "static" else None
            if mode == "static" and scene_memory is None:
                scene_memory = LongTermSceneMemory(current.shape[1], gate=gate_module,
                                                  read_weight=static_weight, max_gap=max_gap)
            if history and history_weight:
                velocities = motion_velocity(track_rows, index) if mode in ("motion", "static") else {}
                historical = []
                for past_time, past_feature, past_index in history:
                    if mode in ("motion", "static"):
                        historical.append(warp_dynamic_feature(
                            past_feature, current, track_rows[past_index]["objects"],
                            velocities, timestamp - past_time))
                    else:
                        historical.append(past_feature)
                past = torch.stack(historical).mean(0)
                batch["spatial_features_2d"] = (1.0 - history_weight) * current + history_weight * past
            else:
                batch["spatial_features_2d"] = current
            if mode == "static":
                read_state = memory_state
                if static_shuffle and read_state is not None:
                    read_state = SceneMemoryState(
                        torch.roll(read_state.feature, shifts=(20, 20), dims=(-2, -1)),
                        torch.roll(read_state.valid, shifts=(20, 20), dims=(-2, -1)),
                        read_state.sequence_id, read_state.timestamp, read_state.updates)
                batch["spatial_features_2d"], _, _ = scene_memory.read(
                    current, batch["spatial_features_2d"], read_state, seq, timestamp, hit)
            batch = model.dense_head(batch)
            pred_dicts, _ = model.post_processing(batch)
            anno = dataset.generate_prediction_dicts(batch, pred_dicts, dataset.class_names)[0]
        history.append((timestamp, current.detach(), index))
        if mode == "static":
            # Update for t+1 after current detection; the mask comes from D0 tracks.
            mask = dynamic_mask(track_rows[index]["objects"], height, width,
                                current.device)
            memory_state = scene_memory.write(current, memory_state, seq, timestamp,
                                              hit, mask)
        objects = [prediction_to_object(name, score, box) for name, score, box in
                   zip(anno["name"], anno["score"], anno["boxes_lidar"])]
        result.append({"sequence_id": seq, "frame_index": row["frame_index"],
                       "frame_id": row["frame_id"], "timestamp": row["timestamp"],
                       "objects": objects})
        timings.append(time.perf_counter() - start)
    return result, timings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sequences", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--history-weight", type=float, default=0.0)
    ap.add_argument("--max-gap", type=float, default=0.3)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--mode", choices=("plain", "motion", "static"), default="plain")
    ap.add_argument("--track-history-root")
    ap.add_argument("--static-weight", type=float, default=0.0)
    ap.add_argument("--static-shuffle", action="store_true")
    ap.add_argument("--static-gate-checkpoint")
    args = ap.parse_args()
    if args.window < 1 or not 0.0 <= args.history_weight <= 1.0:
        ap.error("window must be >=1 and history-weight must be in [0,1]")
    if args.mode in ("motion", "static") and not args.track_history_root:
        ap.error("motion/static modes require --track-history-root")
    if not 0.0 <= args.static_weight <= 1.0:
        ap.error("static-weight must be in [0,1]")
    model, dataset, load_gpu = load_frozen_centerpoint(
        ROOT / "third_party" / "OpenPCDet", ROOT / args.cfg, ROOT / args.ckpt,
        torch.device("cuda"))
    model.dense_head.model_cfg.POST_PROCESSING.SCORE_THRESH = 0.01
    gate_module = None
    if args.static_gate_checkpoint:
        gate_module = StaticGate().cuda().eval()
        checkpoint = torch.load(ROOT / args.static_gate_checkpoint, map_location="cuda")
        gate_module.load_state_dict(checkpoint["state_dict"])
    manifest = read_json(CONVERTED / "manifest.json")
    paths = {x["sequence_id"]: x["path"] for x in manifest["sequences"]}
    out = ROOT / args.output
    speed = []
    for seq in args.sequences:
        rows = list(read_jsonl(CONVERTED / paths[seq]))
        track_rows = list(read_jsonl(ROOT / args.track_history_root / f"{seq}.jsonl")) if args.track_history_root else None
        if track_rows is not None and [x["frame_id"] for x in track_rows] != [x["frame_id"] for x in rows]:
            raise ValueError(f"track history does not match sequence {seq}")
        predictions, timings = infer_sequence(model, dataset, load_gpu, rows, seq,
                                              args.window, args.history_weight,
                                              args.max_gap, args.max_frames, args.mode, track_rows,
                                              args.static_weight, args.static_shuffle, gate_module)
        write_jsonl(out / f"{seq}.jsonl", predictions)
        speed.extend(timings)
        print(seq, "frames", len(predictions), "detections", sum(len(x["objects"]) for x in predictions), flush=True)
    write_json(out / "manifest.json", {"cfg": args.cfg, "ckpt": args.ckpt,
               "sequences": args.sequences, "window": args.window,
               "history_weight": args.history_weight, "max_gap": args.max_gap,
               "mode": args.mode, "track_history_root": args.track_history_root,
               "static_weight": args.static_weight, "static_shuffle": args.static_shuffle,
               "static_gate_checkpoint": args.static_gate_checkpoint,
               "max_frames": args.max_frames, "note": "frozen backbone and dense head; causal pre-head BEV fusion",
               "mean_seconds_per_frame": float(np.mean(speed)) if speed else None,
               "p90_seconds_per_frame": float(np.percentile(speed, 90)) if speed else None})


if __name__ == "__main__":
    main()
