"""用历史轨迹查询当前 BEV，再把修正后的检测送入原始 GRAE"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from inframot3d.config import load_config
from inframot3d.io import read_json, read_jsonl, write_json, write_jsonl
from inframot3d.perception.centerpoint_bev import batch_from_points, bev_feature, load_frozen_centerpoint
from inframot3d.perception.lane_context import LaneContext
from inframot3d.perception.road_query import TrackBevQuery, motion_center, pack_state, wrap_angle
from inframot3d.tracking.grae_adapter import GraeTracker, build_model, load_checkpoint

VEHICLES = {"Car", "Van", "Bus", "Truck"}


def _resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_module(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    module = TrackBevQuery(bev_channels=int(checkpoint["bev_channels"]), **checkpoint["spec"]).to(device)
    module.load_state_dict(checkpoint["model"])
    module.eval()
    return module


def apply_queries(objects, tracks, exist, delta, threshold):
    """存在性高于阈值时提高附近低分框，否则补一个新框"""
    updated = [dict(item) for item in objects]
    for index, track in enumerate(tracks):
        probability = float(exist[index])
        if probability < threshold:
            continue
        center = motion_center(track["box"], track["velocity"], track["dt"])
        pred_x = center[0] + float(delta[index, 0])
        pred_y = center[1] + float(delta[index, 1])
        nearest = None
        best = 1e9
        for obj_index, item in enumerate(updated):
            if item["class_name"] not in VEHICLES:
                continue
            distance = ((item["box"][0] - pred_x) ** 2 + (item["box"][1] - pred_y) ** 2) ** 0.5
            if distance < best:
                best = distance
                nearest = obj_index
        if nearest is not None and best <= 2.0:
            updated[nearest]["score"] = max(float(updated[nearest]["score"]), probability)
            continue
        box = list(track["box"])
        box[0] = pred_x
        box[1] = pred_y
        box[3] = wrap_angle(float(track["box"][3]) + float(delta[index, 2]))
        updated.append({"class_name": track["class_name"], "score": probability, "box": box})
    return updated


def main():
    config = load_config("configs/research/road_query.yaml")
    root = config["_root"]
    device = torch.device(config["train"]["device"])
    checkpoint = Path(sys.argv[1])
    split_name = sys.argv[2]
    threshold = float(sys.argv[3])
    output_dir = Path(sys.argv[4])
    module = load_module(checkpoint, device)
    detector, dataset, load_gpu = load_frozen_centerpoint(
        _resolve(root, config["project"]["openpcdet_root"]),
        _resolve(root, config["project"]["openpcdet_cfg"]),
        _resolve(root, config["project"]["checkpoint"]),
        device,
    )
    grae_config = load_config(_resolve(root, config["project"]["grae_config"]))
    grae = build_model(
        _resolve(root, grae_config["project"]["grae_root"]),
        grae_config["model"]["in_channels"],
        grae_config["model"]["layers"],
        grae_config["num_classes"],
        device,
    )
    load_checkpoint(grae, _resolve(root, "outputs/grae_centerpoint/ckpt/checkpoint-best.pth"), device)
    birth = yaml.safe_load(
        _resolve(root, grae_config["tracker"]["birth_thresholds_file"]).read_text(encoding="utf-8")
    )["score_thresholds"]
    lanes = LaneContext(_resolve(root, config["project"]["map_dir"]))
    data_root = _resolve(root, config["project"]["data_root"])
    poses = {}
    for frame in read_json(data_root / "data_info.json"):
        poses[(str(frame["sequence_id"]), str(frame["frame_id"]))] = frame
    converted = _resolve(root, config["project"]["converted_root"])
    manifest = read_json(converted / "manifest.json")
    detection_root = _resolve(root, grae_config["input"]["detection_root"])
    points_root = _resolve(root, config["project"]["points_root"])
    allowed = set(read_json(_resolve(root, config["split_file"]))[split_name])
    if len(sys.argv) > 5:
        allowed &= set(sys.argv[5].split(","))
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    frames_done = 0
    for entry in manifest["sequences"]:
        sequence_id = entry["sequence_id"]
        if sequence_id not in allowed:
            continue
        tracker = GraeTracker(
            grae,
            grae_config["classes"],
            birth,
            association_alpha=grae_config["tracker"]["association_alpha"],
            age=grae_config["tracker"]["age"],
            score_floor=0.1,
        )
        detections = {
            row["frame_id"]: row
            for row in read_jsonl(detection_root / ("%s.jsonl" % sequence_id))
        }
        labels = list(read_jsonl(converted / entry["path"]))
        history = {}
        rows = []
        for frame in labels:
            det_row = detections[str(frame["frame_id"])]
            objects = [{"class_name": item["class_name"], "score": item["score"], "box": list(item["box"])} for item in det_row["objects"]]
            tracks = list(history.values())
            if tracks and module.use_bev:
                cloud = np.load(points_root / ("%s_%s.npy" % (sequence_id, frame["frame_id"])))
                feature = bev_feature(detector, batch_from_points(dataset, load_gpu, cloud, device))
                meta = poses[(sequence_id, str(frame["frame_id"]))]
                payload = read_json(data_root / meta["calib_virtuallidar_to_world_path"])
                rotation = np.asarray(payload["rotation"], dtype=np.float64)
                translation = np.asarray(payload["translation"], dtype=np.float64).reshape(3)
                state = torch.tensor([item["state"] for item in tracks], dtype=torch.float32, device=device)
                center = torch.tensor([motion_center(item["box"], item["velocity"], item["dt"]) for item in tracks], dtype=torch.float32, device=device)
                tangent_np, valid_np, _ = lanes.tangent_lidar(meta["intersection_loc"], rotation, translation, center.detach().cpu().numpy())
                tangent = torch.tensor(tangent_np, dtype=torch.float32, device=device)
                valid = torch.tensor(valid_np, dtype=torch.float32, device=device)
                with torch.no_grad():
                    logit, delta = module(state, feature, center, tangent, valid)
                    delta = torch.cat([delta[:, :2].clamp(-4, 4), delta[:, 2:3].clamp(-0.5, 0.5)], dim=1)
                objects = apply_queries(objects, tracks, torch.sigmoid(logit).cpu(), delta.cpu(), threshold)
            elif tracks:
                state = torch.tensor([item["state"] for item in tracks], dtype=torch.float32, device=device)
                with torch.no_grad():
                    logit, delta = module(state)
                    delta = torch.cat([delta[:, :2].clamp(-4, 4), delta[:, 2:3].clamp(-0.5, 0.5)], dim=1)
                objects = apply_queries(objects, tracks, torch.sigmoid(logit).cpu(), delta.cpu(), threshold)
            tracked = tracker.update(objects, int(frame["timestamp"]) / 1e6, "%s_%s" % (sequence_id, frame["frame_id"]))
            rows.append(
                {
                    "sequence_id": sequence_id,
                    "frame_index": frame["frame_index"],
                    "frame_id": frame["frame_id"],
                    "timestamp": frame["timestamp"],
                    "objects": tracked,
                }
            )
            new_history = {}
            for item in tracked:
                if item["class_name"] not in VEHICLES:
                    continue
                old = history.get(item["track_id"])
                dt = 0.1
                velocity = [0.0, 0.0]
                if old is not None:
                    dt = max((int(frame["timestamp"]) - int(old["timestamp"])) / 1e6, 1e-3)
                    velocity = [(item["box"][0] - old["box"][0]) / dt, (item["box"][1] - old["box"][1]) / dt]
                new_history[item["track_id"]] = {
                    "class_name": item["class_name"],
                    "box": item["box"],
                    "velocity": velocity,
                    "dt": dt,
                    "timestamp": frame["timestamp"],
                    "state": pack_state(item["box"], velocity, dt),
                }
            history = new_history
            frames_done += 1
        write_jsonl(output_dir / ("%s.jsonl" % sequence_id), rows)
    elapsed = time.perf_counter() - started
    write_json(output_dir / "runtime.json", {"frames": frames_done, "seconds": elapsed, "threshold": threshold, "checkpoint": str(checkpoint)})
    print("frames", frames_done, "seconds", round(elapsed, 1))


if __name__ == "__main__":
    main()
