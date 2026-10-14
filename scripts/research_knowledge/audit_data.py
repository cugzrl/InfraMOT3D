"""路口划分、世界坐标对齐和 GT 质量审查"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inframot3d.analysis.scene_difficulty import build_scenes

from common import (
    LaneIndex,
    build_catalog,
    load_config,
    load_tracks,
    output_dir,
    project_path,
    ranges_overlap,
    round_or_none,
    save_tracks,
    time_gap,
    track_pattern,
    write_csv,
    write_json,
)


def _summarize(values):
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"n": 0}
    return {
        "n": int(array.size),
        "mean": round(float(array.mean()), 4),
        "p50": round(float(np.median(array)), 4),
        "p90": round(float(np.percentile(array, 90)), 4),
        "min": round(float(array.min()), 4),
        "max": round(float(array.max()), 4),
    }


def _coverage(points, cell):
    if len(points) == 0:
        return {"cells": 0, "area_m2": 0.0}
    xy = np.asarray(points, dtype=np.float64)
    keys = np.unique(np.floor(xy / cell).astype(np.int64), axis=0)
    return {"cells": int(len(keys)), "area_m2": round(float(len(keys) * cell * cell), 1)}


def plot_intersection(path, lanes, intersection, tracks, lane_dist):
    chosen = [item for item in tracks if item["intersection"] == intersection and item["class_name"] in {"Car", "Van", "Bus", "Truck"}]
    if not chosen:
        return
    points = np.concatenate([item["world_xy"][::5] for item in chosen], axis=0)
    bounds = [float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 0].max()), float(points[:, 1].max())]
    road = lanes.centerlines_in_box(intersection, bounds)
    figure, axis = plt.subplots(figsize=(8.0, 8.0))
    if len(road):
        axis.scatter(road[:, 0], road[:, 1], s=1, c="#b0b0b0", linewidths=0)
    sequences = sorted({item["sequence_id"] for item in chosen})
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(sequences), 1)))
    for color, sequence_id in zip(colors, sequences):
        subset = [item for item in chosen if item["sequence_id"] == sequence_id]
        for track in subset[:: max(1, len(subset) // 12)]:
            axis.plot(track["world_xy"][:, 0], track["world_xy"][:, 1], color=color, linewidth=0.8, alpha=0.8)
    axis.set_aspect("equal")
    axis.set_xlabel("world x m")
    axis.set_ylabel("world y m")
    axis.set_title("%s median lane distance %.2f m" % (intersection, lane_dist))
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def main():
    config = load_config()
    out = output_dir(config) / "audit"
    out.mkdir(parents=True, exist_ok=True)
    catalog, folds, split = build_catalog(config)
    data_root = project_path(config, config["project"]["data_root"])
    scene_config = {
        "_root": ROOT,
        "project": {"data_root": data_root},
        "split_file": config["split_file"],
        "pose_translation_tolerance_m": config["pose_translation_tolerance_m"],
        "pose_rotation_l2_tolerance": config["pose_rotation_l2_tolerance"],
    }
    scenes, scene_rows, scene_warnings = build_scenes(scene_config)
    print("场景分组 %d 个，警告 %d 条" % (len(scenes), len(scene_warnings)), flush=True)
    tracks, anomalies, class_counter = load_tracks(config, catalog)
    save_tracks(out.parent / "cache" / "tracks.pkl", tracks)
    print("轨迹段 %d" % len(tracks), flush=True)

    motion = config["motion"]
    lanes = LaneIndex(project_path(config, config["project"]["map_root"]), out.parent / "cache")
    vehicles = set(config["vehicle_classes"])
    by_intersection = defaultdict(list)
    for track in tracks:
        track["pattern"] = track_pattern(
            track,
            float(motion["static_speed_m_s"]),
            float(motion["static_disp_m"]),
            np.deg2rad(float(motion["turn_yaw_deg"])),
            float(motion["turn_lateral_m"]),
        )
        by_intersection[track["intersection"]].append(track)

    sequence_rows = []
    lane_by_sequence = {}
    for sequence_id, meta in sorted(catalog.items()):
        subset = [item for item in tracks if item["sequence_id"] == sequence_id]
        vehicle_tracks = [item for item in subset if item["class_name"] in vehicles]
        if vehicle_tracks:
            samples = np.concatenate([item["world_xy"][::5] for item in vehicle_tracks], axis=0)
            _, dist = lanes.region(meta["intersection"], samples, 1e9)
            lane_by_sequence[sequence_id] = dist
        else:
            dist = np.zeros(0)
        classes = Counter(item["class_name"] for item in subset)
        patterns = Counter(item["pattern"] for item in vehicle_tracks)
        durations = [float(item["time"][-1] - item["time"][0]) for item in vehicle_tracks if len(item["time"]) > 1]
        sequence_rows.append(
            {
                "sequence_id": sequence_id,
                "split": meta["split"],
                "fold": meta["fold"] or "",
                "role": meta["role"],
                "intersection": meta["intersection"],
                "lidar_id": meta["lidar_id"],
                "frames": meta["num_frames"],
                "duration_s": round(meta["t1"] - meta["t0"], 3),
                "t0": meta["timestamp_start_us"],
                "t1": meta["timestamp_end_us"],
                "tracks": len(subset),
                "vehicle_tracks": len(vehicle_tracks),
                "classes": dict(classes),
                "patterns": dict(patterns),
                "duration_p50_s": round_or_none(float(np.median(durations)) if durations else None),
                "pose_drift_m": round(meta["within_sequence_drift_m"], 4),
                "pose_rotation_l2": round(meta["within_sequence_rotation_l2"], 6),
                "origin_x": round(float(meta["translation"][0]), 3),
                "origin_y": round(float(meta["translation"][1]), 3),
                "lane_dist_p50_m": round_or_none(float(np.median(dist)) if dist.size else None),
            }
        )
    write_csv(out / "sequence_table.csv", [{key: value for key, value in row.items() if key not in {"classes", "patterns"}} | {"classes": str(row["classes"]), "patterns": str(row["patterns"])} for row in sequence_rows])

    intersection_rows = []
    overlap_rows = []
    plot_dir = out / "maps"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for intersection in sorted({item["intersection"] for item in catalog.values()}):
        members = [item for item in catalog.values() if item["intersection"] == intersection]
        member_tracks = by_intersection[intersection]
        vehicle_tracks = [item for item in member_tracks if item["class_name"] in vehicles]
        train_tracks = [item for item in vehicle_tracks if item["role"] == "inner_train"]
        usable = [item for item in train_tracks if len(item["time"]) >= 2 and float(item["time"][-1] - item["time"][0]) >= 1.0]
        classes = Counter(item["class_name"] for item in member_tracks)
        patterns = Counter(item["pattern"] for item in vehicle_tracks)
        durations = [float(item["time"][-1] - item["time"][0]) for item in vehicle_tracks if len(item["time"]) > 1]
        lengths = [int(len(item["time"])) for item in vehicle_tracks]
        points = np.concatenate([item["world_xy"] for item in vehicle_tracks], axis=0) if vehicle_tracks else np.zeros((0, 2))
        usable_points = np.concatenate([item["world_xy"] for item in usable], axis=0) if usable else np.zeros((0, 2))
        if points.size:
            _, dist = lanes.region(intersection, points[::10], 1e9)
        else:
            dist = np.zeros(0)
        origins = np.stack([item["translation"][:2] for item in members])
        center = np.median(origins, axis=0)
        origin_gap = np.linalg.norm(origins - center, axis=1)
        frames = sum(item["num_frames"] for item in members)
        intersection_rows.append(
            {
                "intersection": intersection,
                "sequences": len(members),
                "train_sequences": sum(item["split"] == "train" for item in members),
                "val_sequences": sum(item["split"] == "val" for item in members),
                "frames": frames,
                "inner_train_sequences": sum(item["role"] == "inner_train" for item in members),
                "inner_holdout_sequences": sum(item["role"] == "inner_holdout" for item in members),
                "tracks": len(member_tracks),
                "vehicle_tracks": len(vehicle_tracks),
                "usable_history_tracks": len(usable),
                "class_counts": str(dict(classes)),
                "pattern_counts": str(dict(patterns)),
                "track_frames_p50": round_or_none(float(np.median(lengths)) if lengths else None),
                "duration_p50_s": round_or_none(float(np.median(durations)) if durations else None),
                "duration_p90_s": round_or_none(float(np.percentile(durations, 90)) if durations else None),
                "origin_gap_max_m": round(float(origin_gap.max()), 4),
                "pose_drift_max_m": round(max(item["within_sequence_drift_m"] for item in members), 4),
                "lane_dist_p50_m": round_or_none(float(np.median(dist)) if dist.size else None),
                "lane_within_4m": round_or_none(float(np.mean(dist <= 4.0)) if dist.size else None),
                "all_cells_10m": _coverage(points, 10.0)["cells"],
                "history_cells_10m": _coverage(usable_points, 10.0)["cells"],
                "history_area_m2": _coverage(usable_points, 10.0)["area_m2"],
            }
        )
        ordered = sorted(members, key=lambda item: item["t0"])
        for left, right in zip(ordered, ordered[1:]):
            overlap_rows.append(
                {
                    "intersection": intersection,
                    "left": left["sequence_id"],
                    "right": right["sequence_id"],
                    "left_role": left["role"],
                    "right_role": right["role"],
                    "overlap": int(ranges_overlap(left, right)),
                    "gap_s": round(time_gap(catalog, left["sequence_id"], right["sequence_id"]), 3),
                }
            )
        for fold_name in ("A", "B"):
            holdouts = [item["sequence_id"] for item in members if item["role"] == "inner_holdout" and item["fold"] == fold_name]
            trains = [item["sequence_id"] for item in members if item["role"] == "inner_train" and item["fold"] == fold_name]
            for holdout in holdouts:
                for train in trains:
                    overlap_rows.append(
                        {
                            "intersection": intersection,
                            "left": train,
                            "right": holdout,
                            "left_role": "inner_train",
                            "right_role": "inner_holdout",
                            "overlap": int(ranges_overlap(catalog[train], catalog[holdout])),
                            "gap_s": round(time_gap(catalog, train, holdout), 3),
                        }
                    )
        median = float(np.median(dist)) if dist.size else float("nan")
        plot_intersection(plot_dir / ("%s_tracks.png" % intersection), lanes, intersection, member_tracks, median)
        print("完成路口 %s" % intersection, flush=True)

    write_csv(out / "intersection_table.csv", intersection_rows)
    write_csv(out / "time_overlap.csv", overlap_rows)
    write_json(out / "id_anomalies.json", {"anomalies": anomalies, "class_object_counts": dict(class_counter)})
    duplicate = [item for item in anomalies if item["type"] == "duplicate_id_same_frame"]
    fold_check = {}
    for fold_name in ("A", "B"):
        train = set(folds["inner_train"][fold_name])
        holdout = set(folds["inner_holdout"][fold_name])
        fold_ids = set(folds["folds"][fold_name])
        fold_check[fold_name] = {
            "disjoint": len(train & holdout) == 0,
            "covers_fold": train | holdout == fold_ids,
            "val_in_bank": sorted(train & set(split["val"])),
        }
        if not fold_check[fold_name]["disjoint"] or not fold_check[fold_name]["covers_fold"] or fold_check[fold_name]["val_in_bank"]:
            raise AssertionError("折划分不满足隔离 %s" % fold_name)
    pose_ok = all(row["origin_gap_max_m"] <= float(config["pose_translation_tolerance_m"]) for row in intersection_rows)
    drift_ok = all(row["pose_drift_max_m"] <= float(config["pose_translation_tolerance_m"]) for row in intersection_rows)
    report = {
        "sequences": {name: len(split[name]) for name in ("train", "val", "test")},
        "fold_check": fold_check,
        "pose_consistent": pose_ok and drift_ok,
        "scene_warnings": scene_warnings,
        "pooled_scenes": [item["scene_id"] for item in scenes if item["pooled"]],
        "duplicate_id_sequences": duplicate,
        "lane_distance": _summarize(np.concatenate(list(lane_by_sequence.values())) if lane_by_sequence else []),
        "intersections": intersection_rows,
    }
    write_json(out / "quality_report.json", report)
    lines = [
        "# 路口划分与数据质量",
        "",
        "官方划分 train %d 条、val %d 条。折 A/B 的 inner_train 与 inner_holdout 互斥，且不包含 val。"
        % (len(split["train"]), len(split["val"])),
        "",
        "同路口虚拟雷达原点最大偏差 %.3f m，序列内最大平移漂移 %.3f m。轨迹相对车道中心线的中位距离 %.2f m。"
        % (
            max(row["origin_gap_max_m"] for row in intersection_rows),
            max(row["pose_drift_max_m"] for row in intersection_rows),
            report["lane_distance"].get("p50", float("nan")),
        ),
        "",
        "同帧重复 GT ID 出现在 %d 条序列。0066 的重复 ID 为 %s。"
        % (
            len(duplicate),
            ",".join(item for row in duplicate if row["sequence_id"] == "0066" for item in row["track_ids"]) or "无",
        ),
        "",
        "场景分组警告：%s" % ("；".join(scene_warnings) if scene_warnings else "无"),
        "",
        "可用于检索的历史轨迹只统计 inner_train 中时长不少于 1 s 的车辆轨迹，见 intersection_table.csv 的 usable_history_tracks。",
        "",
    ]
    (out / "quality_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("审查完成", flush=True)


if __name__ == "__main__":
    main()
