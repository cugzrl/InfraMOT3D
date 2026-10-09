"""训练历史查询模块，CenterPoint 骨干保持冻结"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from inframot3d.config import load_config
from inframot3d.io import read_json, read_jsonl
from inframot3d.perception.centerpoint_bev import batch_from_points, bev_feature, load_frozen_centerpoint
from inframot3d.perception.lane_context import LaneContext
from inframot3d.perception.road_query import TrackBevQuery, motion_center, pack_state, query_loss, wrap_angle

VEHICLES = {"Car", "Van", "Bus", "Truck"}
VARIANTS = {
    "history": {"use_bev": False, "use_lane": False},
    "bev": {"use_bev": True, "use_lane": False},
    "lane": {"use_bev": True, "use_lane": True},
}


def _resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_frames(converted_root):
    manifest = read_json(converted_root / "manifest.json")
    grouped = {}
    for entry in manifest["sequences"]:
        grouped[entry["sequence_id"]] = list(read_jsonl(converted_root / entry["path"]))
    return grouped


def pose_table(data_root):
    table = {}
    for frame in read_json(data_root / "data_info.json"):
        table[(str(frame["sequence_id"]), str(frame["frame_id"]))] = frame
    return table


def read_pose(data_root, frame):
    payload = read_json(data_root / frame["calib_virtuallidar_to_world_path"])
    rotation = np.asarray(payload["rotation"], dtype=np.float64)
    translation = np.asarray(payload["translation"], dtype=np.float64).reshape(3)
    return rotation, translation


def vehicle_objects(frame):
    return [item for item in frame["objects"] if item["class_name"] in VEHICLES]


def make_queries(previous, current, speed, dt, max_queries):
    current_ids = {item["source_track_id"]: item for item in vehicle_objects(current)}
    rows = []
    for item in vehicle_objects(previous):
        track_speed = speed.get(item["source_track_id"], [0.0, 0.0])
        center = motion_center(item["box"], track_speed, dt)
        target = current_ids.get(item["source_track_id"])
        if target is None:
            box_target = [0.0, 0.0, 0.0]
            exist = 0.0
        else:
            exist = 1.0
            box_target = [
                max(-4.0, min(4.0, float(target["box"][0]) - center[0])),
                max(-4.0, min(4.0, float(target["box"][1]) - center[1])),
                max(-0.5, min(0.5, wrap_angle(float(target["box"][3]) - float(item["box"][3])))),
            ]
        rows.append(
            {
                "state": pack_state(item["box"], track_speed, dt),
                "center": center,
                "exist": exist,
                "box_target": box_target,
            }
        )
    rows.sort(key=lambda item: item["center"][0] ** 2 + item["center"][1] ** 2)
    return rows[:max_queries]


def update_speed(previous_xy, frame, dt):
    speeds = {}
    for item in vehicle_objects(frame):
        track_id = item["source_track_id"]
        if track_id not in previous_xy or dt <= 1e-6:
            speeds[track_id] = [0.0, 0.0]
        else:
            old = previous_xy[track_id]
            speeds[track_id] = [(float(item["box"][0]) - old[0]) / dt, (float(item["box"][1]) - old[1]) / dt]
    return speeds


def query_tensors(queries, lanes, intersection, rotation, translation, device):
    state = torch.tensor([item["state"] for item in queries], dtype=torch.float32, device=device)
    center = torch.tensor([item["center"] for item in queries], dtype=torch.float32, device=device)
    exist = torch.tensor([item["exist"] for item in queries], dtype=torch.float32, device=device)
    box_target = torch.tensor([item["box_target"] for item in queries], dtype=torch.float32, device=device)
    tangent_np, valid_np, _ = lanes.tangent_lidar(intersection, rotation, translation, center.detach().cpu().numpy())
    tangent = torch.tensor(tangent_np, dtype=torch.float32, device=device)
    valid = torch.tensor(valid_np, dtype=torch.float32, device=device)
    return state, center, tangent, valid, exist, box_target


def heatmap_alignment(model, batch, frame):
    with torch.no_grad():
        model(batch)
    heatmap = torch.sigmoid(model.dense_head.forward_ret_dict["pred_dicts"][0]["hm"][0, 0])
    height, width = heatmap.shape
    values = []
    for item in vehicle_objects(frame):
        x, y = float(item["box"][0]), float(item["box"][1])
        ix = int((x - 0.0) / 0.8)
        iy = int((y + 56.0) / 0.8)
        if 0 <= iy < height and 0 <= ix < width:
            values.append(float(heatmap[iy, ix]))
    return {
        "shape": [int(height), int(width)],
        "gt_mean": float(np.mean(values)) if values else None,
        "map_mean": float(heatmap.mean()),
        "n": len(values),
    }


def frame_feature(model, dataset, load_gpu, points, device):
    return bev_feature(model, batch_from_points(dataset, load_gpu, points, device))


def main():
    config = load_config("configs/research/road_query.yaml")
    root = config["_root"]
    data_root = _resolve(root, config["project"]["data_root"])
    device = torch.device(config["train"]["device"])
    output = _resolve(root, config["project"]["output_root"])
    ckpt_dir = output / "checkpoints"
    (output / "configs").mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    detector, dataset, load_gpu = load_frozen_centerpoint(
        _resolve(root, config["project"]["openpcdet_root"]),
        _resolve(root, config["project"]["openpcdet_cfg"]),
        _resolve(root, config["project"]["checkpoint"]),
        device,
    )
    grouped = load_frames(_resolve(root, config["project"]["converted_root"]))
    poses = pose_table(data_root)
    lanes = LaneContext(_resolve(root, config["project"]["map_dir"]))
    points_root = _resolve(root, config["project"]["points_root"])
    split = read_json(_resolve(root, config["split_file"]))
    calib = set(config["calib_sequences"])
    train_ids = [item for item in split["train"] if item not in calib]
    sample_id = train_ids[0]
    sample = grouped[sample_id][0]
    points = np.load(points_root / ("%s_%s.npy" % (sample_id, sample["frame_id"])))
    sample_batch = batch_from_points(dataset, load_gpu, points, device)
    feature = bev_feature(detector, sample_batch)
    alignment = heatmap_alignment(detector, sample_batch, sample)
    print("bev", tuple(feature.shape), "alignment", alignment)
    if alignment["gt_mean"] is None or alignment["gt_mean"] <= alignment["map_mean"] * 3:
        raise RuntimeError("BEV 坐标与车辆位置不一致 %s" % alignment)

    def new_modules():
        modules = {name: TrackBevQuery(bev_channels=feature.shape[1], **spec).to(device) for name, spec in VARIANTS.items()}
        optimizers = {
            name: torch.optim.AdamW(module.parameters(), lr=float(config["train"]["lr"]), weight_decay=0.0)
            for name, module in modules.items()
        }
        return modules, optimizers

    modules, optimizers = new_modules()
    print("params", {name: sum(p.numel() for p in module.parameters()) for name, module in modules.items()})

    def step_modules(frame_bev, queries, intersection, rotation, translation, names):
        state, center, tangent, valid, exist, box_target = query_tensors(
            queries, lanes, intersection, rotation, translation, device
        )
        losses = {}
        grad_norm = 0.0
        for name in names:
            module = modules[name]
            module.train()
            logit, delta = module(state, frame_bev, center, tangent, valid)
            delta = torch.cat([delta[:, :2].clamp(-4.0, 4.0), delta[:, 2:3].clamp(-0.5, 0.5)], dim=1)
            loss, _, _ = query_loss(logit, delta, exist, box_target)
            optimizers[name].zero_grad()
            loss.backward()
            if module.use_bev:
                grad_norm = float(module.proj.weight.grad.norm())
            optimizers[name].step()
            losses[name] = float(loss.detach())
        return losses, grad_norm

    overfit_frames = grouped[train_ids[0]][:12]
    start_loss = None
    end_loss = None
    for step in range(12):
        previous_xy = None
        previous_dt = None
        step_losses = []
        grad_norm = 0.0
        for index in range(1, len(overfit_frames)):
            previous = overfit_frames[index - 1]
            current = overfit_frames[index]
            dt = max((int(current["timestamp"]) - int(previous["timestamp"])) / 1e6, 1e-3)
            speed = update_speed(previous_xy, previous, previous_dt) if previous_xy is not None else {}
            previous_xy = {item["source_track_id"]: item["box"][:2] for item in vehicle_objects(previous)}
            previous_dt = dt
            queries = make_queries(previous, current, speed, dt, 16)
            if not queries:
                continue
            cloud = np.load(points_root / ("%s_%s.npy" % (train_ids[0], current["frame_id"])))
            current_bev = frame_feature(detector, dataset, load_gpu, cloud, device)
            meta = poses[(train_ids[0], str(current["frame_id"]))]
            rotation, translation = read_pose(data_root, meta)
            losses, grad_norm = step_modules(
                current_bev, queries, meta["intersection_loc"], rotation, translation, ["lane"]
            )
            step_losses.append(losses["lane"])
        if step_losses:
            start_loss = step_losses[0] if start_loss is None else start_loss
            end_loss = float(np.mean(step_losses))
            print("overfit", step, "loss", round(end_loss, 4), "grad", round(grad_norm, 5))
    if start_loss is None or not (end_loss < start_loss and grad_norm > 0):
        raise RuntimeError("过拟合没有下降 %s -> %s grad %s" % (start_loss, end_loss, grad_norm))
    modules, optimizers = new_modules()
    if "--smoke" in sys.argv:
        print("smoke ok")
        return

    history = []
    for epoch in range(int(config["train"]["epochs"])):
        totals = {name: [] for name in modules}
        for sequence_id in train_ids:
            frames = grouped[sequence_id]
            previous_xy = None
            previous_dt = None
            for index in range(1, len(frames)):
                previous = frames[index - 1]
                current = frames[index]
                dt = max((int(current["timestamp"]) - int(previous["timestamp"])) / 1e6, 1e-3)
                speed = update_speed(previous_xy, previous, previous_dt) if previous_xy is not None else {}
                previous_xy = {item["source_track_id"]: item["box"][:2] for item in vehicle_objects(previous)}
                previous_dt = dt
                queries = make_queries(previous, current, speed, dt, int(config["train"]["max_queries"]))
                cloud_path = points_root / ("%s_%s.npy" % (sequence_id, current["frame_id"]))
                if not queries or not cloud_path.exists():
                    continue
                current_bev = frame_feature(detector, dataset, load_gpu, np.load(cloud_path), device)
                meta = poses[(sequence_id, str(current["frame_id"]))]
                rotation, translation = read_pose(data_root, meta)
                losses, _ = step_modules(
                    current_bev, queries, meta["intersection_loc"], rotation, translation, list(modules)
                )
                for name, value in losses.items():
                    totals[name].append(value)
        row = {"epoch": epoch + 1}
        row.update({name: float(np.mean(values)) for name, values in totals.items()})
        history.append(row)
        print("epoch", row)
        for name, module in modules.items():
            torch.save(
                {"model": module.state_dict(), "variant": name, "epoch": epoch + 1, "spec": VARIANTS[name], "bev_channels": int(feature.shape[1])},
                ckpt_dir / ("%s-epoch%d.pth" % (name, epoch + 1)),
            )
    (output / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    yaml.safe_dump({"copied_from": "configs/research/road_query.yaml"}, (output / "configs" / "road_query.yaml").open("w"), allow_unicode=True)


if __name__ == "__main__":
    main()
