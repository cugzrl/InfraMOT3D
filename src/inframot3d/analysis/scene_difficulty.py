import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, spearmanr
from shapely.geometry import Polygon

from inframot3d.evaluation.protocols.v2xseq import V2XSeqProtocol
from inframot3d.geometry import bev_corners
from inframot3d.io import read_json, read_jsonl, write_json

# 离线分析图，不能在val或test推理时交给tracker
MAP_ROLE = "offline_analysis_map"
COUNT_KEYS = ("gt", "det_miss", "det_score", "det_match", "iou_sum", "trk_miss", "idsw", "frag")


def _path(root, value):
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path


def _percentiles(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def audit_frame_interval(converted_root, split_ids, lidar_interval, tolerance):
    # 只统计序列内部相邻帧，不把序列之间的空隙算进间隔
    gaps = []
    for sequence_id in sorted(split_ids):
        rows = list(read_jsonl(Path(converted_root) / "sequences" / ("%s.jsonl" % sequence_id)))
        stamps = [int(row["timestamp"]) / 1e6 for row in rows]
        gaps.extend(stamps[index] - stamps[index - 1] for index in range(1, len(stamps)))
    if not gaps:
        raise SystemExit("没有可统计的帧间隔")
    stats = _percentiles(gaps)
    stats["lidar_interval"] = float(lidar_interval)
    stats["tolerance"] = float(tolerance)
    stats["median_error"] = abs(stats["median"] - float(lidar_interval))
    stats["keep_fixed_interval"] = stats["median_error"] <= float(tolerance)
    if not stats["keep_fixed_interval"]:
        raise SystemExit(
            "帧间隔中位数%.6f与LiDAR_interval %.3f相差%.6f，超过%.3f"
            % (stats["median"], lidar_interval, stats["median_error"], tolerance)
        )
    return stats


def audit_empty_clips(detection_root, split_ids, calibration_ids, classes, score_threshold, sample_length):
    allowed = set(split_ids) - set(calibration_ids)
    names = set(classes)
    total = 0
    empty = 0
    frames = 0
    empty_frames = 0
    for sequence_id in sorted(allowed):
        rows = list(read_jsonl(Path(detection_root) / ("%s.jsonl" % sequence_id)))
        counts = []
        for row in rows:
            frames += 1
            kept = 0
            for item in row["objects"]:
                if item["class_name"] not in names or float(item["score"]) < float(score_threshold):
                    continue
                kept += 1
            if kept == 0:
                empty_frames += 1
            counts.append(kept)
        width = int(sample_length)
        for start in range(0, len(counts) - width + 1):
            total += 1
            if any(value == 0 for value in counts[start : start + width]):
                empty += 1
    ratio = float(empty) / float(total) if total else 0.0
    return {
        "train_sequences": len(allowed),
        "frames": frames,
        "empty_frames": empty_frames,
        "sample_length": int(sample_length),
        "score_threshold": float(score_threshold),
        "total_clips": total,
        "empty_detection_clips": empty,
        "empty_ratio": ratio,
        "policy": "删除含空检测帧的clip" if ratio < 0.01 else "保留空检测帧并跳过association loss",
        "below_one_percent": ratio < 0.01,
    }


def _pose(path):
    payload = read_json(path)
    rotation = np.asarray(payload["rotation"], dtype=np.float64)
    translation = np.asarray(payload["translation"], dtype=np.float64).reshape(-1)
    return rotation, translation


def build_scenes(config):
    root = config["_root"]
    data_root = config["project"]["data_root"]
    frames = read_json(data_root / "data_info.json")
    grouped = defaultdict(list)
    for frame in frames:
        grouped[str(frame["sequence_id"])].append(frame)
    split = read_json(_path(root, config["split_file"]))
    membership = {}
    for name in ("train", "val", "test"):
        for sequence_id in split.get(name, []):
            membership[str(sequence_id)] = name
    sequences = []
    for sequence_id, items in grouped.items():
        ordered = sorted(items, key=lambda item: int(item["pointcloud_timestamp"]))
        samples = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
        poses = [_pose(data_root / item["calib_virtuallidar_to_world_path"]) for item in samples]
        drift = 0.0
        for index in range(1, len(poses)):
            drift = max(drift, float(np.linalg.norm(poses[index][1] - poses[0][1])))
        start = int(ordered[0]["pointcloud_timestamp"])
        end = int(ordered[-1]["pointcloud_timestamp"])
        sequences.append(
            {
                "sequence_id": sequence_id,
                "split": membership.get(sequence_id, "other"),
                "intersection_loc": str(ordered[0]["intersection_loc"]),
                "lidar_id": str(ordered[0]["lidar_id"]),
                "num_frames": len(ordered),
                "timestamp_start": start,
                "timestamp_end": end,
                "duration_s": (end - start) / 1e6,
                "rotation": poses[0][0],
                "translation": poses[0][1],
                "within_sequence_drift_m": drift,
            }
        )
    translation_tol = float(config["pose_translation_tolerance_m"])
    rotation_tol = float(config["pose_rotation_l2_tolerance"])
    buckets = defaultdict(list)
    for sequence in sequences:
        buckets[(sequence["intersection_loc"], sequence["lidar_id"])].append(sequence)
    scenes = []
    rows = []
    warnings = []
    for key, members in sorted(buckets.items()):
        members = sorted(members, key=lambda item: (item["timestamp_start"], item["sequence_id"]))
        center = np.median(np.stack([item["translation"] for item in members]), axis=0)
        reference = min(members, key=lambda item: float(np.linalg.norm(item["translation"] - center)))
        grouped_members = []
        for sequence in members:
            distance = float(np.linalg.norm(sequence["translation"] - reference["translation"]))
            rotation_gap = float(np.linalg.norm(sequence["rotation"] - reference["rotation"]))
            current = dict(sequence)
            current["translation_distance_m"] = distance
            current["rotation_l2"] = rotation_gap
            current["same_pose"] = distance <= translation_tol and rotation_gap <= rotation_tol
            if current["same_pose"]:
                grouped_members.append(current)
            else:
                warnings.append("%s的序列%s虚拟雷达位姿偏离同站点参考，不能并入同一场景" % (key[0], sequence["sequence_id"]))
                alone = _scene("%s_lidar%s_seq%s" % (key[0], key[1], sequence["sequence_id"]), [current], False)
                scenes.append(alone)
                rows.append(_metadata_row(alone["scene_id"], current))
        if grouped_members:
            scene = _scene("%s_lidar%s" % (key[0], key[1]), grouped_members, True)
            scenes.append(scene)
            for sequence in grouped_members:
                rows.append(_metadata_row(scene["scene_id"], sequence))
    warnings.append("V2X-Seq只能进行短时空间稳定性验证，长期10min/20min实验需要后续使用真实R2长序列数据")
    return scenes, rows, warnings


def _scene(scene_id, sequences, pooled):
    ordered = sorted(sequences, key=lambda item: (item["timestamp_start"], item["sequence_id"]))
    labeled = float(sum(item["duration_s"] for item in ordered))
    wall = (ordered[-1]["timestamp_end"] - ordered[0]["timestamp_start"]) / 1e6
    return {
        "scene_id": scene_id,
        "sequences": ordered,
        "pooled": pooled and len(ordered) > 1,
        "labeled_duration_s": labeled,
        "wall_duration_s": float(wall),
        "num_frames": int(sum(item["num_frames"] for item in ordered)),
        "intersection_loc": ordered[0]["intersection_loc"],
        "lidar_id": ordered[0]["lidar_id"],
    }


def _metadata_row(scene_id, sequence):
    translation = sequence["translation"]
    return {
        "scene_id": scene_id,
        "sequence_id": sequence["sequence_id"],
        "split": sequence["split"],
        "intersection_loc": sequence["intersection_loc"],
        "lidar_id": sequence["lidar_id"],
        "num_frames": sequence["num_frames"],
        "timestamp_start_us": sequence["timestamp_start"],
        "timestamp_end_us": sequence["timestamp_end"],
        "duration_s": round(float(sequence["duration_s"]), 3),
        "translation_x": float(translation[0]),
        "translation_y": float(translation[1]),
        "translation_z": float(translation[2]),
        "translation_distance_m": round(float(sequence["translation_distance_m"]), 6),
        "rotation_l2": round(float(sequence["rotation_l2"]), 6),
        "within_sequence_drift_m": round(float(sequence["within_sequence_drift_m"]), 6),
        "coordinate_system": "virtual_lidar",
        "pooled": bool(sequence["same_pose"]),
    }


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _keep_vehicle(protocol, item):
    if protocol.normalize_class(item["class_name"]) is None:
        return False
    box = [float(value) for value in item["box"]]
    if not np.isfinite(box).all() or min(box[4], box[5], box[6]) <= 0.0:
        return False
    return protocol.inside_range(box)


def _prepare_box(box):
    values = [float(item) for item in box]
    corners = bev_corners(values)
    return {
        "xmin": float(corners[:, 0].min()),
        "xmax": float(corners[:, 0].max()),
        "ymin": float(corners[:, 1].min()),
        "ymax": float(corners[:, 1].max()),
        "zmin": values[2] - values[6] / 2.0,
        "zmax": values[2] + values[6] / 2.0,
        "volume": values[4] * values[5] * values[6],
        "poly": Polygon(corners),
        "xy": values[:2],
    }


def _prepared_iou(left, right):
    # 与geometry.iou_3d相同，先用外接框去掉明显不相交的对
    if left["xmax"] < right["xmin"] or right["xmax"] < left["xmin"] or left["ymax"] < right["ymin"] or right["ymax"] < left["ymin"]:
        return 0.0
    height = min(left["zmax"], right["zmax"]) - max(left["zmin"], right["zmin"])
    if height <= 0.0:
        return 0.0
    area = left["poly"].intersection(right["poly"]).area if left["poly"].intersects(right["poly"]) else 0.0
    intersection = area * height
    union = left["volume"] + right["volume"] - intersection
    return intersection / union if union > 0.0 else 0.0


def _match(gt_boxes, pred_boxes, iou_threshold, gate):
    if not gt_boxes or not pred_boxes:
        return []
    gt_ready = [_prepare_box(box) for box in gt_boxes]
    pred_ready = [_prepare_box(box) for box in pred_boxes]
    gt_xy = np.asarray([item["xy"] for item in gt_ready], dtype=np.float64)
    pred_xy = np.asarray([item["xy"] for item in pred_ready], dtype=np.float64)
    distance = np.hypot(gt_xy[:, None, 0] - pred_xy[None, :, 0], gt_xy[:, None, 1] - pred_xy[None, :, 1])
    pairs = np.argwhere(distance <= float(gate))
    cost = np.ones((len(gt_ready), len(pred_ready)), dtype=np.float64)
    for gt_index, pred_index in pairs:
        try:
            value = _prepared_iou(gt_ready[gt_index], pred_ready[pred_index])
        except Exception:
            value = 0.0
        if value >= float(iou_threshold):
            cost[gt_index, pred_index] = 1.0 - value
    rows, cols = linear_sum_assignment(cost)
    matches = []
    for row, col in zip(rows, cols):
        if cost[row, col] < 1.0:
            matches.append((int(row), int(col), float(1.0 - cost[row, col])))
    return matches


def _objects(row, protocol):
    kept = []
    for item in row["objects"]:
        if not _keep_vehicle(protocol, item):
            continue
        kept.append(item)
    return kept


def collect_events(config, scenes):
    root = config["_root"]
    protocol = V2XSeqProtocol(root)
    converted = config["project"]["converted_root"]
    detection_root = _path(root, config["detection_root"])
    trackers = []
    for tracker in config["trackers"]:
        trackers.append((tracker["name"], _path(root, tracker["prediction_root"])))
    events = []
    split_info = {}
    for scene in scenes:
        frame_keys = []
        for sequence in scene["sequences"]:
            gt_rows = list(read_jsonl(converted / "sequences" / ("%s.jsonl" % sequence["sequence_id"])))
            det_rows = {int(row["timestamp"]): row for row in read_jsonl(detection_root / ("%s.jsonl" % sequence["sequence_id"]))}
            predictions = {}
            for name, folder in trackers:
                path = folder / ("%s.jsonl" % sequence["sequence_id"])
                if not path.is_file():
                    raise SystemExit("缺少%s预测 %s" % (name, path))
                predictions[name] = {int(row["timestamp"]): row for row in read_jsonl(path)}
            history = {name: {} for name, _ in trackers}
            for row in gt_rows:
                stamp = int(row["timestamp"])
                frame_keys.append((stamp, sequence["sequence_id"], int(row["frame_index"])))
                gt_items = _objects(row, protocol)
                det_items = _objects(det_rows.get(stamp, {"objects": []}), protocol)
                det_matches = _match(
                    [item["box"] for item in gt_items],
                    [item["box"] for item in det_items],
                    config["iou_threshold"],
                    config["match_center_gate_m"],
                )
                det_by_gt = {gt_index: (pred_index, iou) for gt_index, pred_index, iou in det_matches}
                tracker_matches = {}
                for name, _ in trackers:
                    pred_items = _objects(predictions[name].get(stamp, {"objects": []}), protocol)
                    paired = _match(
                        [item["box"] for item in gt_items],
                        [item["box"] for item in pred_items],
                        config["iou_threshold"],
                        config["match_center_gate_m"],
                    )
                    tracker_matches[name] = (pred_items, {gt_index: pred_index for gt_index, pred_index, _ in paired})
                for gt_index, gt_item in enumerate(gt_items):
                    center = [float(value) for value in gt_item["box"][:2]]
                    gid = str(gt_item["source_track_id"])
                    record = {
                        "scene_id": scene["scene_id"],
                        "sequence_id": sequence["sequence_id"],
                        "frame_index": int(row["frame_index"]),
                        "timestamp": stamp,
                        "gt_id": gid,
                        "x": center[0],
                        "y": center[1],
                        "det_miss": 1,
                        "det_score": None,
                        "det_iou": None,
                        "trackers": {},
                    }
                    if gt_index in det_by_gt:
                        pred_index, iou = det_by_gt[gt_index]
                        record["det_miss"] = 0
                        record["det_score"] = float(det_items[pred_index]["score"])
                        record["det_iou"] = iou
                    for name, _ in trackers:
                        pred_items, mapping = tracker_matches[name]
                        state = history[name].get(gid)
                        miss = 1
                        idsw = 0
                        frag = 0
                        if gt_index in mapping:
                            track_id = int(pred_items[mapping[gt_index]]["track_id"])
                            if state is not None and state["matched"]:
                                if state["missed"]:
                                    frag = 1
                                elif track_id != state["track_id"]:
                                    idsw = 1
                            history[name][gid] = {"matched": True, "track_id": track_id, "missed": False}
                            miss = 0
                        elif state is not None and state["matched"]:
                            state["missed"] = True
                        record["trackers"][name] = {"miss": miss, "idsw": idsw, "frag": frag}
                    events.append((stamp, sequence["sequence_id"], int(row["frame_index"]), record))
            print("完成场景%s序列%s" % (scene["scene_id"], sequence["sequence_id"]), flush=True)
        frame_keys = sorted(set(frame_keys))
        cuts = _time_cuts(frame_keys)
        split_info[scene["scene_id"]] = cuts
    tagged = []
    for stamp, sequence_id, frame_index, record in events:
        record["split"] = split_info[record["scene_id"]]["lookup"][(stamp, sequence_id, frame_index)]
        tagged.append(record)
    return tagged, split_info


def _time_cuts(frame_keys):
    count = len(frame_keys)
    bounds = [0, count // 3, (2 * count) // 3, count]
    names = ("early", "middle", "late")
    lookup = {}
    ranges = {}
    for name, start, end in zip(names, bounds[:-1], bounds[1:]):
        chunk = frame_keys[start:end]
        for key in chunk:
            lookup[key] = name
        ranges[name] = {
            "frames": len(chunk),
            "timestamp_start_us": chunk[0][0] if chunk else None,
            "timestamp_end_us": chunk[-1][0] if chunk else None,
        }
    return {"lookup": lookup, "ranges": ranges, "frames": count}


def _grid_shape(bev, cell):
    x_span = float(bev["x_max"]) - float(bev["x_min"])
    y_span = float(bev["y_max"]) - float(bev["y_min"])
    nx = int(np.ceil(x_span / float(cell) - 1e-9))
    ny = int(np.ceil(y_span / float(cell) - 1e-9))
    return ny, nx


def _bin_index(x, y, bev, cell, shape):
    ny, nx = shape
    if x < float(bev["x_min"]) or x > float(bev["x_max"]) or y < float(bev["y_min"]) or y > float(bev["y_max"]):
        return None
    ix = int(np.floor((min(x, float(bev["x_max"]) - 1e-6) - float(bev["x_min"])) / float(cell)))
    iy = int(np.floor((min(y, float(bev["y_max"]) - 1e-6) - float(bev["y_min"])) / float(cell)))
    if ix < 0 or iy < 0 or ix >= nx or iy >= ny:
        return None
    return iy, ix


def _empty_counts(shape):
    return {key: np.zeros(shape, dtype=np.float64) for key in COUNT_KEYS}


def accumulate(events, scenes, trackers, bev, cell):
    shape = _grid_shape(bev, cell)
    names = [item["name"] for item in trackers]
    splits = ("early", "middle", "late", "all")
    cubes = {}
    for scene in scenes:
        for name in names:
            for split in splits:
                cubes[(scene["scene_id"], name, split)] = _empty_counts(shape)
    for event in events:
        index = _bin_index(event["x"], event["y"], bev, cell, shape)
        if index is None:
            continue
        for split in (event["split"], "all"):
            for name in names:
                cube = cubes[(event["scene_id"], name, split)]
                cube["gt"][index] += 1
                cube["det_miss"][index] += event["det_miss"]
                if event["det_score"] is not None:
                    cube["det_score"][index] += event["det_score"]
                    cube["det_match"][index] += 1
                    cube["iou_sum"][index] += event["det_iou"]
                tracker = event["trackers"][name]
                cube["trk_miss"][index] += tracker["miss"]
                cube["idsw"][index] += tracker["idsw"]
                cube["frag"][index] += tracker["frag"]
    return cubes, shape


def _normalize(values):
    output = np.full(values.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(values)
    if int(mask.sum()) == 0:
        return output
    low = float(values[mask].min())
    high = float(values[mask].max())
    if high <= low:
        output[mask] = 0.0
    else:
        output[mask] = (values[mask] - low) / (high - low)
    return output


def component_maps(counts, min_observations):
    gt = counts["gt"]
    valid = gt >= float(min_observations)
    det_miss = np.divide(counts["det_miss"], gt, out=np.full_like(gt, np.nan), where=valid)
    score = np.divide(counts["det_score"], counts["det_match"], out=np.full_like(gt, np.nan), where=counts["det_match"] > 0)
    iou = np.divide(counts["iou_sum"], counts["det_match"], out=np.full_like(gt, np.nan), where=counts["det_match"] > 0)
    one_minus_iou = np.full_like(gt, np.nan)
    one_minus_iou[valid] = 1.0
    matched = valid & (counts["det_match"] > 0)
    one_minus_iou[matched] = 1.0 - iou[matched]
    trk_miss = np.divide(counts["trk_miss"], gt, out=np.full_like(gt, np.nan), where=valid)
    idsw = np.divide(counts["idsw"], gt, out=np.full_like(gt, np.nan), where=valid)
    frag = np.divide(counts["frag"], gt, out=np.full_like(gt, np.nan), where=valid)
    difficulty = np.mean(
        np.stack([_normalize(det_miss), _normalize(trk_miss), _normalize(idsw), _normalize(frag), _normalize(one_minus_iou)]),
        axis=0,
    )
    return {
        "gt": gt,
        "valid": valid,
        "detection_miss": det_miss,
        "mean_score": score,
        "mean_iou": iou,
        "tracking_miss": trk_miss,
        "idsw": idsw,
        "fragment": frag,
        "difficulty": difficulty,
    }


def _correlation(left, right):
    mask = np.isfinite(left) & np.isfinite(right)
    count = int(mask.sum())
    if count < 3 or np.unique(left[mask]).size < 2 or np.unique(right[mask]).size < 2:
        return count, None, None
    return count, float(pearsonr(left[mask], right[mask]).statistic), float(spearmanr(left[mask], right[mask]).statistic)


def _top_on_mask(values, mask, fraction, hardest):
    indices = np.argwhere(mask)
    if indices.size == 0:
        return []
    scores = values[mask]
    width = max(1, int(round(float(fraction) * int(mask.sum()))))
    order = np.argsort(-scores if hardest else scores)
    chosen = indices[order[:width]]
    return [tuple(int(value) for value in item) for item in chosen]


def _jaccard(left, right):
    if not left and not right:
        return None
    union = set(left) | set(right)
    if not union:
        return None
    return float(len(set(left) & set(right)) / len(union))


def _pool(counts, indices, key):
    if not indices:
        return 0.0, None
    gt = 0.0
    value = 0.0
    for index in indices:
        gt += float(counts["gt"][index])
        value += float(counts[key][index])
    if gt <= 0.0:
        return 0.0, None
    return gt, value / gt


def stability_rows(scene, cubes, names, cell, min_observations):
    rows = []
    comparisons = (("early", "middle", "early_middle"), ("early", "late", "early_late"), ("middle", "late", "middle_late"))
    per_tracker = {}
    for name in names:
        per_tracker[name] = {split: component_maps(cubes[(scene["scene_id"], name, split)], min_observations) for split in ("early", "middle", "late", "all")}
    consensus = {}
    for split in ("early", "middle", "late", "all"):
        stacked = np.stack([_normalize(per_tracker[name][split]["difficulty"]) for name in names])
        consensus[split] = np.mean(stacked, axis=0)
    maps = {"consensus": consensus, "trackers": per_tracker}
    for name in list(names) + ["consensus"]:
        for left_name, right_name, label in comparisons:
            if name == "consensus":
                left = consensus[left_name]
                right = consensus[right_name]
                right_counts = _mean_counts(cubes, scene["scene_id"], names, right_name)
            else:
                left = per_tracker[name][left_name]["difficulty"]
                right = per_tracker[name][right_name]["difficulty"]
                right_counts = cubes[(scene["scene_id"], name, right_name)]
            count, pearson, spearman = _correlation(left, right)
            both = np.isfinite(left) & np.isfinite(right)
            overlap = _jaccard(_top_on_mask(left, both, 0.2, True), _top_on_mask(right, both, 0.2, True))
            hard_gt = easy_gt = hard_miss = easy_miss = hard_det = easy_det = hard_idsw = easy_idsw = hard_frag = easy_frag = None
            if label == "early_late":
                early_values = consensus["early"] if name == "consensus" else left
                early_mask = np.isfinite(early_values)
                hard = _top_on_mask(early_values, early_mask, 0.2, True)
                easy = _top_on_mask(early_values, early_mask, 0.2, False)
                if not (set(hard) & set(easy)):
                    hard_gt, hard_miss = _pool(right_counts, hard, "trk_miss")
                    easy_gt, easy_miss = _pool(right_counts, easy, "trk_miss")
                    _, hard_det = _pool(right_counts, hard, "det_miss")
                    _, easy_det = _pool(right_counts, easy, "det_miss")
                    _, hard_idsw = _pool(right_counts, hard, "idsw")
                    _, easy_idsw = _pool(right_counts, easy, "idsw")
                    _, hard_frag = _pool(right_counts, hard, "frag")
                    _, easy_frag = _pool(right_counts, easy, "frag")
            rows.append(
                {
                    "scene_id": scene["scene_id"],
                    "tracker": name,
                    "grid_m": cell,
                    "comparison": label,
                    "n_grids": count,
                    "pearson": _round(pearson),
                    "spearman": _round(spearman),
                    "top20_jaccard": _round(overlap),
                    "late_hard_gt": _round(hard_gt),
                    "late_easy_gt": _round(easy_gt),
                    "late_hard_tracking_miss": _round(hard_miss),
                    "late_easy_tracking_miss": _round(easy_miss),
                    "late_hard_detection_miss": _round(hard_det),
                    "late_easy_detection_miss": _round(easy_det),
                    "late_hard_idsw": _round(hard_idsw),
                    "late_easy_idsw": _round(easy_idsw),
                    "late_hard_fragment": _round(hard_frag),
                    "late_easy_fragment": _round(easy_frag),
                }
            )
    return rows, maps


def _mean_counts(cubes, scene_id, names, split):
    base = cubes[(scene_id, names[0], split)]
    merged = {key: np.zeros_like(base[key]) for key in COUNT_KEYS}
    for name in names:
        current = cubes[(scene_id, name, split)]
        for key in ("trk_miss", "idsw", "frag"):
            merged[key] += current[key]
        if name == names[0]:
            for key in ("gt", "det_miss", "det_score", "det_match", "iou_sum"):
                merged[key] += current[key]
    for key in ("trk_miss", "idsw", "frag"):
        merged[key] /= float(len(names))
    return merged


def _round(value):
    if value is None:
        return ""
    return round(float(value), 6)


def _font():
    from matplotlib import font_manager

    path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if path.is_file():
        return font_manager.FontProperties(fname=str(path))
    return None


def _save_heatmap(path, grid, bev, vmin, vmax, title):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams["axes.unicode_minus"] = False
    figure, axis = plt.subplots(figsize=(8.2, 4.6))
    image = np.ma.masked_invalid(grid)
    show = axis.imshow(
        image,
        origin="lower",
        extent=[float(bev["x_min"]), float(bev["x_max"]), float(bev["y_min"]), float(bev["y_max"])],
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        cmap="magma",
    )
    font = _font()
    axis.set_xlabel("x/m", fontproperties=font)
    axis.set_ylabel("y/m", fontproperties=font)
    axis.set_title(title, fontproperties=font)
    figure.colorbar(show, ax=axis, fraction=0.046)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _scale(grid, floor):
    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        return 0.0, floor
    return 0.0, max(floor, float(finite.max()))


def save_scene_maps(scene_dir, scene_id, maps, names, bev, cell):
    full = maps["trackers"][names[0]]["all"]
    consensus_tracking = np.mean(np.stack([maps["trackers"][name]["all"]["tracking_miss"] for name in names]), axis=0)
    consensus_idsw = np.mean(np.stack([maps["trackers"][name]["all"]["idsw"] for name in names]), axis=0)
    consensus_frag = np.mean(np.stack([maps["trackers"][name]["all"]["fragment"] for name in names]), axis=0)
    _, density_max = _scale(full["gt"], 1.0)
    panels = {
        "gt_density.png": (full["gt"], 0.0, density_max, "GT数量"),
        "detection_miss.png": (full["detection_miss"], 0.0, 1.0, "检测漏检率"),
        "tracking_miss.png": (consensus_tracking, 0.0, 1.0, "跟踪漏检率"),
        "idsw.png": (consensus_idsw, 0.0, 1.0, "IDSW率"),
        "fragment.png": (consensus_frag, 0.0, 1.0, "fragment率"),
    }
    for filename, (grid, vmin, vmax, title) in panels.items():
        _save_heatmap(scene_dir / filename, grid, bev, vmin, vmax, "%s %s" % (scene_id, title))
    difficulties = [maps["consensus"][name] for name in ("early", "middle", "late")]
    finite = np.concatenate([item[np.isfinite(item)] for item in difficulties if np.isfinite(item).any()])
    vmax = float(finite.max()) if finite.size else 1.0
    vmax = max(vmax, 1e-6)
    for name, grid in zip(("early", "middle", "late"), difficulties):
        _save_heatmap(scene_dir / ("difficulty_%s.png" % name), grid, bev, 0.0, vmax, "%s difficulty_%s" % (scene_id, name))
        np.save(scene_dir / ("difficulty_%s.npy" % name), grid.astype(np.float32))
    write_json(
        scene_dir / "map_meta.json",
        {
            "map_role": MAP_ROLE,
            "scene_id": scene_id,
            "grid_m": cell,
            "bev": bev,
            "shape": list(difficulties[0].shape),
            "difficulty_vmin": 0.0,
            "difficulty_vmax": vmax,
            "note": "offline_analysis_map与未来online_scene_memory分开，val推理不能读取",
        },
    )
    for name in names:
        folder = scene_dir / "trackers" / name
        folder.mkdir(parents=True, exist_ok=True)
        tracker_maps = [maps["trackers"][name][split]["difficulty"] for split in ("early", "middle", "late")]
        values = np.concatenate([item[np.isfinite(item)] for item in tracker_maps if np.isfinite(item).any()])
        tracker_vmax = max(float(values.max()) if values.size else 1.0, 1e-6)
        for split, grid in zip(("early", "middle", "late"), tracker_maps):
            _save_heatmap(folder / ("difficulty_%s.png" % split), grid, bev, 0.0, tracker_vmax, "%s %s %s" % (scene_id, name, split))
            np.save(folder / ("difficulty_%s.npy" % split), grid.astype(np.float32))


def _fmt(value):
    if value == "" or value is None:
        return "无"
    return "%.4f" % float(value)


def write_summary(path, scenes, split_info, rows, warnings, interval, empty_stats, primary, min_observations, judgment_min):
    lines = ["# Scene Tracking Difficulty", ""]
    lines.append("地图角色是`offline_analysis_map`。它只描述离线统计现象，val和test推理不能读取，也不能写进未来的`online_scene_memory`。")
    lines.append("")
    lines.append("## 数据检查")
    lines.append("")
    lines.append(
        "帧间隔count %d，mean %.6f，median %.6f，p10 %.6f，p90 %.6f，min %.6f，max %.6f。中位数与0.1秒相差%.6f，Fast-Poly继续使用固定0.1秒。"
        % (interval["count"], interval["mean"], interval["median"], interval["p10"], interval["p90"], interval["min"], interval["max"], interval["median_error"])
    )
    if empty_stats["below_one_percent"]:
        empty_note = "比例低于1%，训练继续删除这些clip，推理仍按原时间顺序推进"
    else:
        empty_note = "比例高于1%，不能静默删除，训练需要保留空帧并跳过association loss"
    lines.append(
        "3DMOTFormer训练clip共%d，含空检测帧%d，删除比例%.6f。%s。"
        % (empty_stats["total_clips"], empty_stats["empty_detection_clips"], empty_stats["empty_ratio"], empty_note)
    )
    lines.append("")
    lines.append("## 场景")
    lines.append("")
    for warning in warnings:
        lines.append("- %s" % warning)
    lines.append("")
    lines.append("| 场景 | 序列数 | 帧数 | 标注时长/s | 墙钟跨度/s | 是否合并 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for scene in scenes:
        lines.append(
            "| %s | %d | %d | %.1f | %.1f | %s |"
            % (scene["scene_id"], len(scene["sequences"]), scene["num_frames"], scene["labeled_duration_s"], scene["wall_duration_s"], "是" if scene["pooled"] else "否")
        )
    lines.append("")
    for scene in scenes:
        ids = ",".join(item["sequence_id"] for item in scene["sequences"])
        lines.append("%s包含序列%s。" % (scene["scene_id"], ids))
        info = split_info[scene["scene_id"]]["ranges"]
        for name in ("early", "middle", "late"):
            start = info[name]["timestamp_start_us"]
            end = info[name]["timestamp_end_us"]
            span = (end - start) / 1e6 if start is not None else 0.0
            lines.append("%s的%s有%d帧，时间跨度%.1f秒。" % (scene["scene_id"], name, info[name]["frames"], span))
    lines.append("")
    lines.append("10分钟和20分钟窗口没有启用。各场景标注时长都短于10分钟，墙钟跨度里大部分是没有标注的空隙。")
    lines.append("")
    lines.append("## 稳定性")
    lines.append("")
    lines.append("网格默认%.0fm，只有`gt_count>=%d`的格子参与相关性和Top20%%。2m和10m结果在同一张表里，主结论使用5m的consensus。" % (primary, min_observations))
    lines.append("")
    lines.append("| 场景 | 对比 | 有效格 | Pearson | Spearman | Top20重合 | late hard/easy miss | late hard/easy IDSW | late hard/easy fragment |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    focus = []
    for row in rows:
        if row["tracker"] != "consensus" or float(row["grid_m"]) != float(primary):
            continue
        miss = "%s / %s" % (_fmt(row["late_hard_tracking_miss"]), _fmt(row["late_easy_tracking_miss"]))
        idsw = "%s / %s" % (_fmt(row["late_hard_idsw"]), _fmt(row["late_easy_idsw"]))
        frag = "%s / %s" % (_fmt(row["late_hard_fragment"]), _fmt(row["late_easy_fragment"]))
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (row["scene_id"], row["comparison"], row["n_grids"], _fmt(row["pearson"]), _fmt(row["spearman"]), _fmt(row["top20_jaccard"]), miss, idsw, frag)
        )
        if row["comparison"] == "early_late":
            focus.append(row)
    lines.append("")
    usable = [row for row in focus if row["spearman"] != "" and int(row["n_grids"]) >= 3]
    if not usable:
        judgment = "有效格子不足，motivation不能判断。"
    else:
        spearman_values = [float(row["spearman"]) for row in usable]
        mean_spearman = float(np.mean(spearman_values))
        miss_higher = 0
        error_higher = 0
        for row in usable:
            if row["late_hard_tracking_miss"] != "" and row["late_easy_tracking_miss"] != "" and float(row["late_hard_tracking_miss"]) > float(row["late_easy_tracking_miss"]):
                miss_higher += 1
            idsw_higher = row["late_hard_idsw"] != "" and row["late_easy_idsw"] != "" and float(row["late_hard_idsw"]) > float(row["late_easy_idsw"])
            frag_higher = row["late_hard_fragment"] != "" and row["late_easy_fragment"] != "" and float(row["late_hard_fragment"]) > float(row["late_easy_fragment"])
            if idsw_higher or frag_higher:
                error_higher += 1
        lines.append("early-late Spearman均值%.4f。%d/%d个场景的hard区域未来tracking miss更高，%d/%d个场景的hard区域未来IDSW或fragment更高。" % (mean_spearman, miss_higher, len(usable), error_higher, len(usable)))
        supported = mean_spearman >= float(judgment_min) and miss_higher > len(usable) / 2.0 and error_higher > len(usable) / 2.0
        if supported:
            judgment = "短时空间稳定性得到支持：early难度和late难度相关，且early划出的hard区域在late仍有更高错误率。这个支持只覆盖V2X-Seq的短时拼接，不覆盖10分钟或20分钟。"
        elif mean_spearman < 0.2:
            judgment = "early-late相关性低，motivation没有得到数据支持。"
        else:
            judgment = "相关性或未来错误率只满足一部分，motivation只得到部分支持。"
    lines.append("")
    lines.append("## 判断")
    lines.append("")
    lines.append(judgment)
    lines.append("")
    lines.append("判断规则写在这里：5m consensus的early-late Spearman均值达到%.1f，并且多数场景的late hard tracking miss高于easy，同时多数场景的late hard IDSW或fragment高于easy，才记为支持。所有场景都列入上表。" % judgment_min)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return judgment


def run(config):
    root = config["_root"]
    output_root = config["project"]["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    split = read_json(_path(root, config["split_file"]))
    split_ids = [str(item) for name in ("train", "val") for item in split[name]]
    interval = audit_frame_interval(config["project"]["converted_root"], split_ids, config["lidar_interval"], config["interval_tolerance"])
    write_json(output_root / "frame_interval.json", interval)
    print(
        "帧间隔 median %.6f mean %.6f p10 %.6f p90 %.6f min %.6f max %.6f"
        % (interval["median"], interval["mean"], interval["p10"], interval["p90"], interval["min"], interval["max"]),
        flush=True,
    )
    motformer = config["motformer"]
    empty_stats = audit_empty_clips(
        _path(root, config["detection_root"]),
        split["train"],
        motformer["calibration_sequences"],
        motformer["classes"],
        motformer["score_threshold"],
        motformer["sample_length"],
    )
    write_json(_path(root, config["motformer_stats"]), empty_stats)
    print("空检测clip %d/%d 比例 %.6f" % (empty_stats["empty_detection_clips"], empty_stats["total_clips"], empty_stats["empty_ratio"]), flush=True)
    scenes, metadata, warnings = build_scenes(config)
    _write_csv(output_root / "scene_metadata.csv", metadata)
    events, split_info = collect_events(config, scenes)
    names = [item["name"] for item in config["trackers"]]
    rows = []
    primary = float(config["primary_grid"])
    for cell in config["grid_sizes"]:
        cubes, _ = accumulate(events, scenes, config["trackers"], config["bev"], float(cell))
        for scene in scenes:
            scene_rows, maps = stability_rows(scene, cubes, names, float(cell), int(config["min_observations"]))
            rows.extend(scene_rows)
            if float(cell) == primary:
                save_scene_maps(output_root / scene["scene_id"], scene["scene_id"], maps, names, config["bev"], float(cell))
        print("完成网格%.1fm" % float(cell), flush=True)
    _write_csv(output_root / "stability_summary.csv", rows)
    judgment = write_summary(
        output_root / "summary.md",
        scenes,
        split_info,
        rows,
        warnings,
        interval,
        empty_stats,
        primary,
        int(config["min_observations"]),
        config["judgment_min_spearman"],
    )
    print(judgment, flush=True)
