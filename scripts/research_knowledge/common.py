"""路侧长期知识检索的数据、几何和检索工具

只服务离线建库与验证
"""

import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree

from inframot3d.geometry import wrap_angle
from inframot3d.io import read_json, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[2]
VEHICLE_FALLBACK = ("Car", "Van", "Bus", "Truck")


def load_config(path=None):
    path = Path(path) if path else Path(__file__).resolve().parent / "config.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["_root"] = ROOT
    config["_config_path"] = str(path)
    return config


def project_path(config, value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def output_dir(config):
    path = project_path(config, config["project"]["output_root"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path, rows):
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def round_or_none(value, digits=4):
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None
        return round(float(value), digits)
    return value


def to_local(xy, origin, yaw):
    # 世界位移转到以 yaw 为前向的局部坐标
    delta = np.asarray(xy, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    forward = cosine * delta[..., 0] + sine * delta[..., 1]
    lateral = -sine * delta[..., 0] + cosine * delta[..., 1]
    return np.stack([forward, lateral], axis=-1)


def from_local(local_xy, origin, yaw):
    local_xy = np.asarray(local_xy, dtype=np.float64)
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    world_x = cosine * local_xy[..., 0] - sine * local_xy[..., 1]
    world_y = sine * local_xy[..., 0] + cosine * local_xy[..., 1]
    origin = np.asarray(origin, dtype=np.float64)
    return np.stack([world_x + origin[..., 0], world_y + origin[..., 1]], axis=-1)


def angle_diff(left, right):
    return (np.asarray(left) - np.asarray(right) + np.pi) % (2.0 * np.pi) - np.pi


def nearest_indices(times, queries, tolerance):
    times = np.asarray(times, dtype=np.float64)
    queries = np.asarray(queries, dtype=np.float64)
    if times.size == 0:
        return np.zeros(queries.shape, dtype=np.int64), np.zeros(queries.shape, dtype=bool)
    index = np.searchsorted(times, queries)
    index = np.clip(index, 0, times.size - 1)
    previous = np.clip(index - 1, 0, times.size - 1)
    use_previous = np.abs(times[previous] - queries) <= np.abs(times[index] - queries)
    chosen = np.where(use_previous, previous, index)
    ok = np.abs(times[chosen] - queries) <= float(tolerance)
    return chosen.astype(np.int64), ok


def load_pose(path):
    payload = read_json(path)
    rotation = np.asarray(payload["rotation"], dtype=np.float64)
    translation = np.asarray(payload["translation"], dtype=np.float64).reshape(3)
    return rotation, translation


def world_pose(rotation, translation, xyz, yaw):
    xyz = np.asarray(xyz, dtype=np.float64)
    world = rotation @ xyz + translation
    forward = rotation[:2, :2] @ np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
    world_yaw = float(np.arctan2(forward[1], forward[0]))
    return world, world_yaw


def build_catalog(config):
    # 序列元数据、折角色和起止位姿
    data_root = project_path(config, config["project"]["data_root"])
    split = read_json(project_path(config, config["split_file"]))
    folds = read_json(project_path(config, config["fold_file"]))
    membership = {}
    for name in ("train", "val", "test"):
        for sequence_id in split.get(name, []):
            membership[str(sequence_id)] = name
    fold_of = {}
    role_of = {}
    for fold_name, sequence_ids in folds["folds"].items():
        for sequence_id in sequence_ids:
            fold_of[str(sequence_id)] = fold_name
            role_of[str(sequence_id)] = "fold_only"
    for fold_name, sequence_ids in folds["inner_train"].items():
        for sequence_id in sequence_ids:
            role_of[str(sequence_id)] = "inner_train"
    for fold_name, sequence_ids in folds["inner_holdout"].items():
        for sequence_id in sequence_ids:
            role_of[str(sequence_id)] = "inner_holdout"
    grouped = defaultdict(list)
    for frame in read_json(data_root / "data_info.json"):
        grouped[str(frame["sequence_id"])].append(frame)
    catalog = {}
    for sequence_id, frames in grouped.items():
        frames = sorted(frames, key=lambda item: int(item["pointcloud_timestamp"]))
        sample_ids = [0, len(frames) // 2, len(frames) - 1]
        poses = [load_pose(data_root / frames[index]["calib_virtuallidar_to_world_path"]) for index in sample_ids]
        drift = 0.0
        rotation_gap = 0.0
        for rotation, translation in poses[1:]:
            drift = max(drift, float(np.linalg.norm(translation - poses[0][1])))
            rotation_gap = max(rotation_gap, float(np.linalg.norm(rotation - poses[0][0])))
        start = int(frames[0]["pointcloud_timestamp"])
        end = int(frames[-1]["pointcloud_timestamp"])
        catalog[sequence_id] = {
            "sequence_id": sequence_id,
            "split": membership.get(sequence_id, "other"),
            "fold": fold_of.get(sequence_id),
            "role": role_of.get(sequence_id, "outside_fold"),
            "intersection": str(frames[0]["intersection_loc"]),
            "lidar_id": str(frames[0]["lidar_id"]),
            "num_frames": len(frames),
            "t0": start / 1e6,
            "t1": end / 1e6,
            "timestamp_start_us": start,
            "timestamp_end_us": end,
            "rotation": poses[0][0],
            "translation": poses[0][1],
            "within_sequence_drift_m": drift,
            "within_sequence_rotation_l2": rotation_gap,
            "frame_ids": [str(item["frame_id"]) for item in frames],
        }
    return catalog, folds, split


def ranges_overlap(left, right):
    return left["t1"] >= right["t0"] and left["t0"] <= right["t1"]


def forbidden_sequences(catalog, query_sequence):
    # 查询序列自身，以及同路口采集时间重叠的序列
    query = catalog[query_sequence]
    banned = {query_sequence}
    for sequence_id, other in catalog.items():
        if other["intersection"] != query["intersection"]:
            continue
        if ranges_overlap(query, other):
            banned.add(sequence_id)
    return banned


def time_gap(catalog, left_id, right_id):
    left = catalog[left_id]
    right = catalog[right_id]
    if ranges_overlap(left, right):
        return 0.0
    if left["t1"] < right["t0"]:
        return float(right["t0"] - left["t1"])
    return float(left["t0"] - right["t1"])


def load_tracks(config, catalog):
    converted = project_path(config, config["project"]["converted_root"])
    gap = float(config["segment_gap_s"])
    speed_limit = float(config["speed_limit_m_s"])
    tracks = []
    anomalies = []
    class_counter = Counter()
    for sequence_id in sorted(catalog):
        meta = catalog[sequence_id]
        rows = list(read_jsonl(converted / "sequences" / ("%s.jsonl" % sequence_id)))
        if len(rows) != meta["num_frames"]:
            raise ValueError("序列 %s 帧数与 data_info 不一致" % sequence_id)
        duplicate_ids = set()
        frame_counts = []
        for row in rows:
            counts = Counter(str(item["source_track_id"]) for item in row["objects"])
            frame_counts.append(counts)
            for track_id, count in counts.items():
                if count > 1:
                    duplicate_ids.add(track_id)
        if duplicate_ids:
            anomalies.append(
                {
                    "sequence_id": sequence_id,
                    "intersection": meta["intersection"],
                    "type": "duplicate_id_same_frame",
                    "track_ids": sorted(duplicate_ids),
                    "frames": int(sum(1 for counts in frame_counts if any(counts[item] > 1 for item in duplicate_ids))),
                }
            )
        buckets = defaultdict(list)
        rotation = meta["rotation"]
        translation = meta["translation"]
        for row_index, row in enumerate(rows):
            stamp = int(row["timestamp"]) / 1e6
            for obj_index, obj in enumerate(row["objects"]):
                track_id = str(obj["source_track_id"])
                class_counter[obj["class_name"]] += 1
                if track_id in duplicate_ids:
                    continue
                if frame_counts[row_index][track_id] != 1:
                    continue
                box = [float(value) for value in obj["box"]]
                if not np.isfinite(box).all() or min(box[4], box[5], box[6]) <= 0.0:
                    continue
                world, world_yaw = world_pose(rotation, translation, box[:3], box[3])
                buckets[track_id].append(
                    {
                        "time": stamp,
                        "frame_index": int(row["frame_index"]),
                        "class_name": obj["class_name"],
                        "lidar_xy": np.array(box[:2], dtype=np.float64),
                        "lidar_yaw": float(box[3]),
                        "world_xy": world[:2].astype(np.float64),
                        "world_yaw": world_yaw,
                    }
                )
        for track_id, samples in buckets.items():
            samples.sort(key=lambda item: (item["time"], item["frame_index"]))
            segments = []
            current = [samples[0]]
            for item in samples[1:]:
                dt = item["time"] - current[-1]["time"]
                distance = float(np.linalg.norm(item["world_xy"] - current[-1]["world_xy"]))
                jump = dt > 1e-3 and distance / dt > speed_limit
                if dt > gap or dt <= 0.0 or jump:
                    if jump:
                        anomalies.append(
                            {
                                "sequence_id": sequence_id,
                                "intersection": meta["intersection"],
                                "type": "speed_jump",
                                "track_ids": [track_id],
                                "frames": 1,
                                "speed_m_s": round(distance / max(dt, 1e-6), 3),
                            }
                        )
                    segments.append(current)
                    current = [item]
                else:
                    current.append(item)
            segments.append(current)
            for segment_index, segment in enumerate(segments):
                classes = Counter(item["class_name"] for item in segment)
                tracks.append(
                    {
                        "sequence_id": sequence_id,
                        "intersection": meta["intersection"],
                        "fold": meta["fold"],
                        "role": meta["role"],
                        "split": meta["split"],
                        "track_id": track_id,
                        "segment": int(segment_index),
                        "uid": "%s:%s:%d" % (sequence_id, track_id, segment_index),
                        "class_name": classes.most_common(1)[0][0],
                        "time": np.array([item["time"] for item in segment], dtype=np.float64),
                        "frame_index": np.array([item["frame_index"] for item in segment], dtype=np.int64),
                        "lidar_xy": np.stack([item["lidar_xy"] for item in segment]),
                        "lidar_yaw": np.array([item["lidar_yaw"] for item in segment], dtype=np.float64),
                        "world_xy": np.stack([item["world_xy"] for item in segment]),
                        "world_yaw": np.unwrap([item["world_yaw"] for item in segment]),
                    }
                )
    return tracks, anomalies, class_counter


def save_tracks(path, tracks):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(tracks, stream, protocol=4)


def load_cached_tracks(path):
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def parse_centerline(items):
    points = []
    for item in items:
        if not isinstance(item, str):
            continue
        left = item.find("(")
        comma = item.find(",")
        right = item.find(")")
        if left < 0 or comma < 0 or right < 0:
            continue
        points.append((float(item[left + 1 : comma]), float(item[comma + 1 : right])))
    return np.asarray(points, dtype=np.float64)


class LaneIndex:
    def __init__(self, map_root, cache_dir):
        self.map_root = Path(map_root)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = {}

    def _load(self, intersection):
        if intersection in self._cache:
            return self._cache[intersection]
        cache_path = self.cache_dir / ("lanes_%s.npz" % intersection)
        if cache_path.is_file():
            payload = np.load(cache_path, allow_pickle=True)
            xy = payload["xy"]
            turn = payload["turn"]
        else:
            raw = json.loads((self.map_root / ("%s.json" % intersection)).read_text(encoding="utf-8"))
            centers = []
            turns = []
            for lane in raw["LANE"].values():
                pts = parse_centerline(lane.get("centerline", []))
                if len(pts) < 2:
                    continue
                step = max(1, int(round(2.0 / max(float(np.median(np.linalg.norm(np.diff(pts, axis=0), axis=1))), 1e-3))))
                pts = pts[::step]
                name = str(lane.get("turn_direction") or "NONE")
                if name == "NONE":
                    name = "STRAIGHT"
                centers.append(pts)
                turns.append(np.full(len(pts), name))
            xy = np.concatenate(centers, axis=0)
            turn = np.concatenate(turns, axis=0)
            np.savez_compressed(cache_path, xy=xy, turn=turn)
        tree = cKDTree(xy)
        self._cache[intersection] = (tree, turn)
        return self._cache[intersection]

    def region(self, intersection, world_xy, radius):
        tree, turn = self._load(intersection)
        points = np.asarray(world_xy, dtype=np.float64).reshape(-1, 2)
        dist, index = tree.query(points, k=1)
        names = np.where(dist <= float(radius), turn[index], "OFFROAD")
        return names, dist

    def centerlines_in_box(self, intersection, bounds, margin=30.0):
        tree, _ = self._load(intersection)
        xy = tree.data
        x0, y0, x1, y1 = bounds
        mask = (
            (xy[:, 0] >= x0 - margin)
            & (xy[:, 0] <= x1 + margin)
            & (xy[:, 1] >= y0 - margin)
            & (xy[:, 1] <= y1 + margin)
        )
        return xy[mask]


def track_pattern(track, static_speed, static_disp, turn_yaw, turn_lateral):
    times = track["time"]
    xy = track["world_xy"]
    if len(times) < 2:
        return "unknown"
    duration = float(times[-1] - times[0])
    displacement = float(np.linalg.norm(xy[-1] - xy[0]))
    speed = displacement / max(duration, 1e-3)
    local_end = to_local(xy[-1], xy[0], track["world_yaw"][0])
    yaw_change = abs(float(angle_diff(track["world_yaw"][-1], track["world_yaw"][0])))
    if speed < static_speed and displacement < static_disp:
        return "static"
    if yaw_change > turn_yaw or abs(float(local_end[1])) > turn_lateral:
        return "turn"
    return "straight"


def reference_heading(world_xy, world_yaw, times, min_speed):
    # 有速度时用最近 0.5 s 位移方向，避免车头标注翻转
    if len(times) < 2:
        return float(wrap_angle(world_yaw[-1]))
    target = times[-1] - 0.5
    index, ok = nearest_indices(times, np.array([target]), 0.2)
    origin = world_xy[int(index[0])] if ok[0] else world_xy[0]
    delta = world_xy[-1] - origin
    distance = float(np.linalg.norm(delta))
    dt = float(times[-1] - (times[int(index[0])] if ok[0] else times[0]))
    if dt > 1e-3 and distance / dt >= min_speed and distance > 0.2:
        return float(np.arctan2(delta[1], delta[0]))
    return float(wrap_angle(world_yaw[-1]))


def sample_window(track, anchor, history_s, future_s, dt, tolerance):
    times = track["time"]
    history_n = int(round(history_s / dt))
    future_n = int(round(future_s / dt))
    history_t = anchor - history_s + dt * np.arange(history_n + 1)
    future_t = anchor + dt * np.arange(1, future_n + 1)
    hist_index, hist_ok = nearest_indices(times, history_t, tolerance)
    fut_index, fut_ok = nearest_indices(times, future_t, tolerance)
    return hist_index, hist_ok, fut_index, fut_ok


def iter_motion_anchors(track, config, lanes, need_future):
    # 锚点必须正好落在当前帧，历史窗完整，建库时还要求未来窗完整
    times = track["time"]
    history_s = float(config["history_s"])
    tolerance = float(config["sample_tolerance_s"])
    for position in range(len(times)):
        anchor = float(times[position])
        if anchor - float(times[0]) < history_s - tolerance:
            continue
        hist_index, hist_ok, fut_index, fut_ok = sample_window(
            track, anchor, history_s, float(config["future_s"]), float(config["dt_s"]), tolerance
        )
        if not np.all(hist_ok) or int(hist_index[-1]) != position:
            continue
        if need_future and not np.all(fut_ok):
            continue
        sample = make_sample(track, position, hist_index, lanes, config)
        future = np.full((fut_index.shape[0], 2), np.nan, dtype=np.float64)
        if np.any(fut_ok):
            future[fut_ok] = to_local(track["world_xy"][fut_index[fut_ok]], sample["origin"], sample["heading"])
        yield position, sample, future, fut_ok


def make_sample(track, anchor_pos, hist_index, lanes, config):
    dt = float(config["dt_s"])
    origin = track["world_xy"][anchor_pos]
    heading = reference_heading(
        track["world_xy"][hist_index],
        track["world_yaw"][hist_index],
        track["time"][hist_index],
        float(config["motion"]["heading_speed_m_s"]),
    )
    hist_local = to_local(track["world_xy"][hist_index], origin, heading)
    past = hist_local[hist_index.shape[0] // 2]
    speed = float(np.linalg.norm(hist_local[-1] - past) / max(0.5, dt))
    yaw_rate = float(angle_diff(heading, reference_heading(
        track["world_xy"][hist_index[: hist_index.shape[0] // 2 + 1]],
        track["world_yaw"][hist_index[: hist_index.shape[0] // 2 + 1]],
        track["time"][hist_index[: hist_index.shape[0] // 2 + 1]],
        float(config["motion"]["heading_speed_m_s"]),
    )) / 0.5)
    names, dist = lanes.region(track["intersection"], origin.reshape(1, 2), config["retrieval"]["lane_radius_m"])
    return {
        "origin": origin.astype(np.float64),
        "heading": float(heading),
        "hist": hist_local.astype(np.float64),
        "speed": speed,
        "yaw_rate": yaw_rate,
        "region": str(names[0]),
        "lane_dist": float(dist[0]),
        "lidar_xy": track["lidar_xy"][anchor_pos].astype(np.float64),
    }


def future_pattern(future_local, config):
    motion = config["motion"]
    end = future_local[-1]
    distance = float(np.linalg.norm(end))
    if distance < float(motion["static_disp_m"]):
        return "static"
    heading = abs(float(np.arctan2(end[1], end[0])))
    if heading > np.deg2rad(float(motion["turn_yaw_deg"])) or abs(float(end[1])) > float(motion["turn_lateral_m"]):
        return "turn"
    return "straight"


def stack_bank(records):
    if not records:
        return None
    regions = sorted({item["region"] for item in records})
    region_index = {name: index for index, name in enumerate(regions)}
    classes = sorted({item["class_name"] for item in records})
    class_index = {name: index for index, name in enumerate(classes)}
    sequences = sorted({item["sequence_id"] for item in records})
    sequence_index = {name: index for index, name in enumerate(sequences)}
    uids = sorted({item["uid"] for item in records})
    uid_index = {name: index for index, name in enumerate(uids)}
    return {
        "origin": np.stack([item["origin"] for item in records]),
        "heading": np.array([item["heading"] for item in records], dtype=np.float64),
        "hist": np.stack([item["hist"] for item in records]),
        "future": np.stack([item["future"] for item in records]),
        "speed": np.array([item["speed"] for item in records], dtype=np.float64),
        "yaw_rate": np.array([item["yaw_rate"] for item in records], dtype=np.float64),
        "time": np.array([item["time"] for item in records], dtype=np.float64),
        "region": np.array([region_index[item["region"]] for item in records], dtype=np.int16),
        "class_id": np.array([class_index[item["class_name"]] for item in records], dtype=np.int16),
        "sequence_id": np.array([sequence_index[item["sequence_id"]] for item in records], dtype=np.int16),
        "uid": np.array([uid_index[item["uid"]] for item in records], dtype=np.int32),
        "region_names": np.array(regions),
        "class_names": np.array(classes),
        "sequence_names": np.array(sequences),
        "uid_names": np.array(uids),
    }


def save_bank(path, bank):
    np.savez_compressed(path, **bank)


def load_bank(path):
    payload = np.load(path, allow_pickle=False)
    return {key: payload[key] for key in payload.files}


def retrieval_cost(bank, query, mask, config):
    spec = config["retrieval"]
    hist = bank["hist"][mask]
    if hist.size == 0:
        return np.zeros(0, dtype=np.float64)
    spatial = np.linalg.norm(bank["origin"][mask] - query["origin"], axis=1) / float(spec["spatial_sigma_m"])
    motion = np.sqrt(np.mean((hist - query["hist"]) ** 2, axis=(1, 2))) / float(spec["motion_sigma_m"])
    speed = np.abs(bank["speed"][mask] - query["speed"]) / float(spec["speed_sigma_m_s"])
    heading = np.abs(angle_diff(bank["heading"][mask], query["heading"])) / np.deg2rad(float(spec["heading_sigma_deg"]))
    region_names = bank["region_names"]
    bank_region = region_names[bank["region"][mask]]
    penalty = np.zeros(int(mask.sum()), dtype=np.float64)
    differ = (bank_region != query["region"]) & (bank_region != "OFFROAD") & (query["region"] != "OFFROAD")
    penalty[differ] = float(spec["region_penalty"])
    return spatial + motion + speed + heading + penalty


def select_topk(bank, query, allowed_sequences, config):
    # 同类别、允许序列，同一条独立轨迹只保留最近样本
    class_names = [str(name) for name in bank["class_names"]]
    if query["class_name"] not in class_names:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    class_id = class_names.index(query["class_name"])
    sequence_names = [str(name) for name in bank["sequence_names"]]
    allowed_ids = [sequence_names.index(str(name)) for name in allowed_sequences if str(name) in sequence_names]
    if not allowed_ids:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    sequence_ok = np.isin(bank["sequence_id"], np.asarray(allowed_ids, dtype=np.int16))
    mask = (bank["class_id"] == class_id) & sequence_ok
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return indices, np.zeros(0, dtype=np.float64)
    spatial = np.linalg.norm(bank["origin"][indices] - query["origin"], axis=1)
    near = spatial <= float(config["retrieval"]["spatial_gate_m"])
    indices = indices[near]
    if indices.size == 0:
        return indices, np.zeros(0, dtype=np.float64)
    mask = np.zeros(bank["class_id"].shape[0], dtype=bool)
    mask[indices] = True
    cost = retrieval_cost(bank, query, mask, config)
    order = np.argsort(cost, kind="mergesort")
    chosen = []
    chosen_cost = []
    seen = set()
    k = int(config["retrieval"]["top_k"])
    for position in order:
        uid = int(bank["uid"][indices[position]])
        if uid in seen:
            continue
        seen.add(uid)
        chosen.append(int(indices[position]))
        chosen_cost.append(float(cost[position]))
        if len(chosen) >= k:
            break
    return np.asarray(chosen, dtype=np.int64), np.asarray(chosen_cost, dtype=np.float64)


def interpolate_future(future, tau, dt):
    # future[i] 是锚点之后第 i+1 个步长的局部位移
    future = np.asarray(future, dtype=np.float64)
    if tau < 0.0 or tau > dt * len(future) + 1e-6:
        return None
    alpha = float(tau) / float(dt)
    if alpha <= 1e-8:
        return np.zeros(2, dtype=np.float64)
    lower = int(np.floor(alpha))
    upper = min(lower + 1, len(future))
    if upper == lower:
        return future[-1].copy()
    fraction = alpha - lower
    previous = np.zeros(2, dtype=np.float64) if lower == 0 else future[lower - 1]
    return (1.0 - fraction) * previous + fraction * future[upper - 1]


def blend_futures(futures, costs):
    if len(futures) == 0:
        return None
    if len(futures) == 1:
        return futures[0]
    weight = 1.0 / (costs + 0.1)
    weight = weight / weight.sum()
    return np.sum(futures * weight[:, None, None], axis=0)


def constant_velocity(hist, dt):
    # hist 最后一点是当前原点，中点对应约 0.5 s 前
    mid = hist.shape[0] // 2
    velocity = (hist[-1] - hist[mid]) / 0.5
    steps = np.arange(1, int(round(2.0 / dt)) + 1, dtype=np.float64) * dt
    return velocity.reshape(1, 2) * steps.reshape(-1, 1)


def constant_acceleration(hist, dt):
    # 用 0.8 s 到 0.5 s 和最近 0.3 s 估计加速度
    n = hist.shape[0]
    v_now = (hist[-1] - hist[n - 4]) / 0.3
    v_prev = (hist[n // 2] - hist[n // 2 - 3]) / 0.3
    accel = (v_now - v_prev) / 0.5
    steps = np.arange(1, int(round(2.0 / dt)) + 1, dtype=np.float64) * dt
    return v_now.reshape(1, 2) * steps.reshape(-1, 1) + 0.5 * accel.reshape(1, 2) * (steps.reshape(-1, 1) ** 2)


def static_future(dt):
    steps = int(round(2.0 / dt))
    return np.zeros((steps, 2), dtype=np.float64)


def horizon_index(horizon, dt):
    return int(round(float(horizon) / float(dt))) - 1


def ade_fde(pred, gt, valid):
    err = np.linalg.norm(pred - gt, axis=-1)
    if not np.all(valid):
        return None, None
    return float(err.mean()), float(err[-1])


def assert_bank_isolation(bank_sequences, query_sequence, catalog):
    if query_sequence in set(bank_sequences):
        raise AssertionError("检索库包含待评估序列 %s" % query_sequence)
    banned = forbidden_sequences(catalog, query_sequence)
    leaked = sorted(set(bank_sequences) & banned)
    if leaked:
        raise AssertionError("检索库含重叠序列 %s -> %s" % (query_sequence, ",".join(leaked)))
    for sequence_id in bank_sequences:
        meta = catalog[sequence_id]
        if meta["split"] != "train" or meta["role"] != "inner_train":
            raise AssertionError("检索库序列 %s 不是 inner_train" % sequence_id)


def cell_index(x, y, bev, cell):
    from inframot3d.analysis.scene_difficulty import _bin_index, _grid_shape

    shape = _grid_shape(bev, cell)
    return _bin_index(float(x), float(y), bev, cell, shape), shape


def load_hard_regions(config):
    # 复用已发布场景困难图，取三时段平均后的前 20% 作为困难区域
    root = project_path(config, config["scene_difficulty_root"])
    fraction = float(config["reliability"]["hard_fraction"])
    regions = {}
    for scene_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        meta_path = scene_dir / "map_meta.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        stacked = np.stack(
            [
                np.load(scene_dir / "difficulty_early.npy"),
                np.load(scene_dir / "difficulty_middle.npy"),
                np.load(scene_dir / "difficulty_late.npy"),
            ]
        )
        finite_maps = [item for item in stacked]
        with np.errstate(invalid="ignore"):
            difficulty = np.nanmean(np.stack(finite_maps), axis=0)
        finite = np.isfinite(difficulty)
        hard = np.zeros(difficulty.shape, dtype=bool)
        if int(finite.sum()) > 0:
            threshold = float(np.quantile(difficulty[finite], 1.0 - fraction))
            hard = finite & (difficulty >= threshold)
        regions[meta["scene_id"]] = {
            "intersection": meta["scene_id"].split("_lidar")[0],
            "hard": hard,
            "difficulty": difficulty,
            "bev": meta["bev"],
            "grid_m": float(meta["grid_m"]),
        }
    return regions


def scene_of(regions, intersection):
    matches = [item for item in regions.values() if item["intersection"] == intersection]
    if len(matches) != 1:
        raise KeyError("路口 %s 的困难区域图数量为 %d" % (intersection, len(matches)))
    return matches[0]


def region_name(regions, intersection, x, y):
    scene = scene_of(regions, intersection)
    index, _ = cell_index(x, y, scene["bev"], scene["grid_m"])
    if index is None or not np.isfinite(scene["difficulty"][index]):
        return "unknown"
    return "hard" if bool(scene["hard"][index]) else "normal"
