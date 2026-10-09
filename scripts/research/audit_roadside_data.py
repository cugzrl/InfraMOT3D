"""审查 V2X-Seq 地图对齐、轨迹时长和遮挡与检测的关系"""

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path("/home/kemove/devdata1/zrl/InfraMOT3D")
DATA = ROOT / "data/v2x-seq-infrastructure"
MAPS = ROOT / "data/v2x_seq_maps"
DETS = ROOT / "outputs/centerpoint/detections"
OUT = ROOT / "outputs/research/roadside_joint_perception"
VEHICLES = {"Car", "Van", "Bus", "Truck"}
PAIR = re.compile(r"\(([-\d.eE]+),\s*([-\d.eE]+)\)")


def parse_xy(items):
    points = []
    for item in items:
        found = PAIR.findall(item if isinstance(item, str) else "")
        if found:
            points.append([float(found[0][0]), float(found[0][1])])
    return np.asarray(points, dtype=np.float64)


def load_lanes(path):
    payload = json.loads(path.read_text())
    centers = []
    tangents = []
    turns = []
    for lane in payload["LANE"].values():
        pts = parse_xy(lane["centerline"])
        if len(pts) < 2:
            continue
        delta = np.diff(pts, axis=0)
        delta = np.vstack([delta, delta[-1]])
        norm = np.linalg.norm(delta, axis=1, keepdims=True)
        delta = delta / np.clip(norm, 1e-6, None)
        centers.append(pts)
        tangents.append(delta)
        turns.append(np.full(len(pts), hash(str(lane.get("turn_direction"))) % 1))
    xy = np.concatenate(centers, axis=0)
    tangent = np.concatenate(tangents, axis=0)
    turn = np.array([str(lane.get("turn_direction")) for lane in payload["LANE"].values() for _ in range(max(len(parse_xy(lane["centerline"])), 0)) if len(parse_xy(lane["centerline"])) >= 2])
    # 上面 turn 会重复解析，改用同步收集
    return xy, tangent, payload


def load_lane_index(path):
    payload = json.loads(path.read_text())
    centers = []
    tangents = []
    turn_names = []
    for lane in payload["LANE"].values():
        pts = parse_xy(lane["centerline"])
        if len(pts) < 2:
            continue
        delta = np.diff(pts, axis=0)
        delta = np.vstack([delta, delta[-1]])
        norm = np.linalg.norm(delta, axis=1, keepdims=True)
        delta = delta / np.clip(norm, 1e-6, None)
        centers.append(pts)
        tangents.append(delta)
        turn_names.extend([str(lane.get("turn_direction"))] * len(pts))
    xy = np.concatenate(centers, axis=0)
    tangent = np.concatenate(tangents, axis=0)
    return cKDTree(xy), xy, tangent, np.asarray(turn_names), {key: len(payload[key]) for key in payload}


def pose_of(path):
    payload = json.loads(path.read_text())
    rotation = np.asarray(payload["rotation"], dtype=np.float64)
    translation = np.asarray(payload["translation"], dtype=np.float64).reshape(3)
    return rotation, translation


def angle_diff(a, b):
    return np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "n": int(arr.size),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "mean": float(arr.mean()),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = json.loads((DATA / "data_info.json").read_text())
    grouped = defaultdict(list)
    for frame in frames:
        grouped[str(frame["sequence_id"])].append(frame)
    for items in grouped.values():
        items.sort(key=lambda item: int(item["pointcloud_timestamp"]))

    intersections = defaultdict(list)
    for sequence_id, items in grouped.items():
        intersections[items[0]["intersection_loc"]].append(sequence_id)
    map_names = {path.stem for path in MAPS.glob("*.json")}
    coverage = []
    for name, sequences in sorted(intersections.items()):
        coverage.append(
            {
                "intersection": name,
                "sequences": len(sequences),
                "frames": sum(len(grouped[sid]) for sid in sequences),
                "has_map": name in map_names,
            }
        )

    trees = {}
    alignment = []
    heading_rows = []
    control_rows = []
    motion_rows = []
    for name, sequences in sorted(intersections.items()):
        if name not in map_names:
            continue
        tree, xy, tangent, turns, counts = load_lane_index(MAPS / ("%s.json" % name))
        trees[name] = (tree, tangent, turns)
        distances = []
        headings = []
        shifted = []
        by_turn = defaultdict(list)
        for sequence_id in sequences:
            items = grouped[sequence_id]
            for frame in items[::5]:
                labels = json.loads((DATA / frame["label_lidar_std_path"]).read_text())
                rotation, translation = pose_of(DATA / frame["calib_virtuallidar_to_world_path"])
                for obj in labels:
                    if obj["type"] not in VEHICLES:
                        continue
                    loc = obj["3d_location"]
                    local = np.array([loc["x"], loc["y"], loc["z"]], dtype=np.float64)
                    world = rotation @ local + translation
                    dist, index = tree.query(world[:2], k=1)
                    distances.append(float(dist))
                    direction = rotation[:2, :2] @ np.array([np.cos(obj["rotation"]), np.sin(obj["rotation"])])
                    vehicle_yaw = np.arctan2(direction[1], direction[0])
                    lane_yaw = np.arctan2(tangent[index, 1], tangent[index, 0])
                    err = float(min(angle_diff(vehicle_yaw, lane_yaw), angle_diff(vehicle_yaw, lane_yaw + np.pi)))
                    headings.append(err)
                    by_turn[str(turns[index])].append(err)
                    shifted_xy = world[:2] + np.array([15.0, 0.0])
                    shifted.append(float(tree.query(shifted_xy, k=1)[0]))
        alignment.append(
            {
                "intersection": name,
                "lane_counts": counts,
                "centerline_distance_m": summarize(distances),
                "within_2m": float(np.mean(np.asarray(distances) <= 2.0)),
                "within_4m": float(np.mean(np.asarray(distances) <= 4.0)),
                "shifted_15m_distance": summarize(shifted),
                "shifted_within_2m": float(np.mean(np.asarray(shifted) <= 2.0)),
                "heading_error_rad": summarize(headings),
                "heading_within_20deg": float(np.mean(np.asarray(headings) <= np.deg2rad(20))),
                "heading_by_turn": {key: summarize(value) for key, value in by_turn.items()},
            }
        )
        heading_rows.extend(headings)
        control_rows.extend(shifted)

    # 轨迹断点与未来监督，按完整序列
    horizons = [0.5, 1.0, 2.0]
    horizon_hits = {value: 0 for value in horizons}
    horizon_base = 0
    durations = []
    gaps = []
    dts = []
    for sequence_id, items in grouped.items():
        tracks = defaultdict(list)
        for frame in items:
            stamp = int(frame["pointcloud_timestamp"]) / 1e6
            labels = json.loads((DATA / frame["label_lidar_std_path"]).read_text())
            for obj in labels:
                if obj["type"] not in VEHICLES:
                    continue
                loc = obj["3d_location"]
                tracks[obj["track_id"]].append((stamp, float(loc["x"]), float(loc["y"])))
        for states in tracks.values():
            states.sort()
            if len(states) < 2:
                continue
            times = np.array([item[0] for item in states])
            step = np.diff(times)
            dts.extend(step[(step > 0) & (step < 1.0)].tolist())
            breaks = np.where(step > 0.25)[0]
            gaps.append(int(len(breaks)))
            start = 0
            cuts = list(breaks + 1) + [len(states)]
            for end in cuts:
                segment = states[start:end]
                if len(segment) >= 2:
                    durations.append(segment[-1][0] - segment[0][0])
                    stamps = np.array([item[0] for item in segment])
                    for origin in stamps:
                        horizon_base += 1
                        for horizon in horizons:
                            if np.any(np.abs(stamps - (origin + horizon)) <= 0.15):
                                horizon_hits[horizon] += 1
                start = end
    trajectory = {
        "dt_s": summarize(dts),
        "segment_duration_s": summarize(durations),
        "segments": len(durations),
        "segments_ge_2s": int(np.sum(np.asarray(durations) >= 2.0)),
        "mean_breaks_per_track": float(np.mean(gaps)) if gaps else 0.0,
        "future_coverage": {
            str(key): {"hits": value, "base": horizon_base, "rate": value / max(horizon_base, 1)}
            for key, value in horizon_hits.items()
        },
    }

    # 遮挡与当前检测，验证集抽样
    split = json.loads((ROOT / "configs/datasets/v2xseq_sequence_split.json").read_text())
    val_ids = [str(item) for item in split["val"]]
    occ_bins = defaultdict(lambda: {"n": 0, "hit": 0})
    for sequence_id in val_ids:
        det_path = DETS / ("%s.jsonl" % sequence_id)
        if not det_path.exists():
            continue
        det_rows = [json.loads(line) for line in det_path.read_text().splitlines() if line.strip()]
        det_by_frame = {row["frame_id"]: row for row in det_rows}
        for frame in grouped[sequence_id][::2]:
            labels = json.loads((DATA / frame["label_lidar_std_path"]).read_text())
            preds = [
                obj for obj in det_by_frame.get(str(frame["frame_id"]), {}).get("objects", [])
                if obj["class_name"] in VEHICLES and obj["score"] >= 0.1
            ]
            pred_xy = np.array([obj["box"][:2] for obj in preds], dtype=np.float64) if preds else np.zeros((0, 2))
            for obj in labels:
                if obj["type"] not in VEHICLES:
                    continue
                loc = obj["3d_location"]
                center = np.array([loc["x"], loc["y"]], dtype=np.float64)
                radius = float(np.hypot(center[0], center[1]))
                if radius > 100:
                    band = "100+"
                elif radius > 50:
                    band = "50-100"
                else:
                    band = "0-50"
                key = "%s|occ%s" % (band, obj.get("occluded_state"))
                occ_bins[key]["n"] += 1
                if len(pred_xy):
                    if np.min(np.linalg.norm(pred_xy - center, axis=1)) <= 2.0:
                        occ_bins[key]["hit"] += 1
    occlusion = {
        key: {**value, "recall": value["hit"] / max(value["n"], 1)}
        for key, value in sorted(occ_bins.items())
    }
    report = {
        "coverage": coverage,
        "alignment": alignment,
        "trajectory": trajectory,
        "occlusion_detection": occlusion,
    }
    (OUT / "audit_stats.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"coverage": coverage, "trajectory": trajectory}, indent=2)[:4000])
    for item in alignment:
        print(item["intersection"], item["centerline_distance_m"], "within2", round(item["within_2m"], 3), "shift2", round(item["shifted_within_2m"], 3), "head20", round(item["heading_within_20deg"], 3))


if __name__ == "__main__":
    main()
