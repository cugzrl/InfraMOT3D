import csv
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from inframot3d.analysis.scene_difficulty import _keep_vehicle, _match
from inframot3d.analysis.weak_observation import _point_count
from inframot3d.config import load_config
from inframot3d.evaluation import UnifiedMOTEvaluator
from inframot3d.evaluation.hota import evaluate_hota
from inframot3d.evaluation.protocols.v2xseq import V2XSeqProtocol
from inframot3d.io import read_json, read_jsonl, write_json, write_jsonl
from inframot3d.tracking.grae_adapter import GraeTracker, _bev_iou, build_model, load_checkpoint


PAIR_FIELDS = (
    "sequence_id",
    "frame_id",
    "frame_index",
    "det_index",
    "track_id",
    "det_x",
    "det_y",
    "track_x",
    "track_y",
    "euclidean_m",
    "sqrt_norm_feature",
    "exp_neg_distance",
    "logit",
    "learned_score",
    "fusion_score",
    "match_score",
    "det_score",
    "range_m",
    "point_count",
    "point_density",
    "local_density",
    "track_age",
    "seconds_since_match",
    "dt_seconds",
    "class_ok",
    "gate_pass",
    "matched",
    "det_gt_id",
    "track_gt_id",
    "label",
    "strict_label",
    "identity_status",
)
DETECTION_FIELDS = (
    "sequence_id",
    "frame_id",
    "frame_index",
    "gt_id",
    "class_name",
    "gt_range_m",
    "matched",
    "det_index",
    "det_range_m",
    "error_xy_m",
    "error_x_m",
    "error_y_m",
    "abs_error_x_m",
    "abs_error_y_m",
    "bev_iou",
    "iou_3d",
    "score",
    "point_count",
    "point_density",
    "local_density",
    "local_gt_count",
)
FAILURE_FIELDS = (
    "sequence_id",
    "frame_id",
    "frame_index",
    "gt_id",
    "range_m",
    "has_detection",
    "in_candidate_pool",
    "loc_error_m",
    "has_track",
    "pred_error_m",
    "cv_error_m",
    "class_ok",
    "gate_pass",
    "assigned",
    "large_loc",
    "large_pred",
    "primary",
    "identity_state",
    "track_id",
    "det_score",
    "point_count",
    "seconds_since_match",
    "velocity_abs",
)
PRIMARY_REASONS = (
    "detection_missing",
    "track_management",
    "class_constraint",
    "detection_localization",
    "motion_prediction",
    "gate_rejection",
    "association_competition",
    "output_suppression",
)


def _path(root, value):
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path


def match_euclidean(dx, dy):
    # 匹配使用的平面标准欧氏距离，单位 m
    return float(math.hypot(float(dx), float(dy)))


def feature_sqrt_norm(dx, dy):
    # 与适配器 temporal_dist 一致，是欧氏距离再开方，不是第二套物理距离
    return float(math.sqrt(match_euclidean(dx, dy)))


def label_from_ids(det_gt, track_gt, conflict, shared):
    if conflict:
        return "unknown", "conflict"
    if shared:
        return "unknown", "shared"
    if det_gt is None and track_gt is None:
        return "unknown", "unknown_both"
    if det_gt is None:
        return "unknown", "unknown_det"
    if track_gt is None:
        return "unknown", "unknown_track"
    if str(det_gt) == str(track_gt):
        return "positive", "known"
    return "negative", "known"


def strict_label(label, status, det_iou, det_center, track_quality, strict_iou, strict_center):
    if label not in {"positive", "negative"} or status != "known":
        return "unknown"
    if det_iou is None or det_center is None or track_quality is None:
        return "unknown"
    if float(det_iou) < float(strict_iou) or float(det_center) > float(strict_center):
        return "unknown"
    if float(track_quality) < float(strict_iou):
        return "unknown"
    return label


def exclusive_primary(
    has_detection,
    has_track,
    class_ok,
    gate_pass,
    assigned,
    output_ok,
    loc_error,
    pred_error,
    loc_limit,
    pred_limit,
):
    # 成功不进入失败分母，其余按决策阶段互斥归因
    if assigned and output_ok:
        return "success"
    if not has_detection:
        return "detection_missing"
    if not has_track:
        return "track_management"
    if not class_ok:
        return "class_constraint"
    large_loc = loc_error is not None and float(loc_error) > float(loc_limit)
    large_pred = pred_error is not None and float(pred_error) > float(pred_limit)
    if not gate_pass:
        if large_pred and (not large_loc or float(pred_error) >= float(loc_error)):
            return "motion_prediction"
        if large_loc:
            return "detection_localization"
        return "gate_rejection"
    if not assigned:
        return "association_competition"
    return "output_suppression"


def bin_name(value, edges):
    number = float(value)
    bounds = [float(item) for item in edges]
    for left, right in zip(bounds[:-1], bounds[1:]):
        if number < right:
            return "%g-%g" % (left, right)
    return "%g+" % bounds[-1]


def _describe(values):
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "p90": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
    }


def _safe_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size == 0 or int(labels.min()) == int(labels.max()):
        return None
    return float(roc_auc_score(labels, scores))


def _safe_ap(labels, scores):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size == 0 or int(labels.sum()) == 0:
        return None
    return float(average_precision_score(labels, scores))


def recall_at_fpr(labels, scores, fpr_target):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positive = int(labels.sum())
    negative = int(labels.size - positive)
    if positive == 0 or negative == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    ordered = labels[order]
    false_positive = 0
    true_positive = 0
    best = 0.0
    limit = float(fpr_target) * negative
    for value in ordered:
        if int(value) == 1:
            true_positive += 1
        else:
            false_positive += 1
        if false_positive <= limit + 1.0e-9:
            best = true_positive / positive
        else:
            break
    return float(best)


def _fmt(value, digits=6):
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError("出现非有限数值")
        return "%.*f" % (digits, float(value))
    return str(value)


def _write_header(path, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=list(fields)).writeheader()


def _append_rows(path, fields, rows):
    if not rows:
        return
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields))
        for row in rows:
            writer.writerow({key: _fmt(row.get(key)) for key in fields})


class IdentityBook:
    def __init__(self):
        self.gt_of_track = {}
        self.conflict = set()
        self.quality = {}
        self.last_det_xy = {}
        self.prev_det_xy = {}
        self.last_time = {}
        self.prev_time = {}

    def state(self, track_id):
        track_id = int(track_id)
        if track_id in self.conflict:
            return None, True, None
        if track_id not in self.gt_of_track:
            return None, False, None
        return self.gt_of_track[track_id], False, self.quality.get(track_id)

    def commit(self, track_id, det_gt, det_xy, iou, timestamp_us):
        track_id = int(track_id)
        if det_gt is None:
            return
        previous = self.gt_of_track.get(track_id)
        if previous is not None and str(previous) != str(det_gt):
            self.conflict.add(track_id)
        self.gt_of_track[track_id] = str(det_gt)
        if track_id in self.last_det_xy:
            self.prev_det_xy[track_id] = self.last_det_xy[track_id]
            self.prev_time[track_id] = self.last_time[track_id]
        self.last_det_xy[track_id] = (float(det_xy[0]), float(det_xy[1]))
        self.last_time[track_id] = int(timestamp_us)
        if track_id not in self.conflict and iou is not None:
            self.quality[track_id] = float(iou)

    def motion_oracle(self, track_id, now_us, gt_xy):
        track_id = int(track_id)
        if track_id not in self.prev_det_xy or track_id not in self.last_det_xy:
            return None
        dt_history = (int(self.last_time[track_id]) - int(self.prev_time[track_id])) / 1.0e6
        dt_now = (int(now_us) - int(self.last_time[track_id])) / 1.0e6
        if dt_history <= 1.0e-6 or dt_now < 0.0:
            return None
        previous = np.asarray(self.prev_det_xy[track_id], dtype=np.float64)
        last = np.asarray(self.last_det_xy[track_id], dtype=np.float64)
        predicted = last + (last - previous) / dt_history * dt_now
        return match_euclidean(predicted[0] - float(gt_xy[0]), predicted[1] - float(gt_xy[1]))


def _pose(path):
    payload = read_json(path)
    translation = np.asarray(payload["translation"], dtype=np.float64).reshape(-1)
    return translation


def audit_sensor_origin(config, sequence_ids):
    data_root = Path(config["project"]["data_root"])
    grouped = defaultdict(list)
    allowed = set(sequence_ids)
    for frame in read_json(data_root / "data_info.json"):
        sequence_id = str(frame["sequence_id"])
        if sequence_id in allowed:
            grouped[sequence_id].append(frame)
    drifts = []
    for sequence_id in sequence_ids:
        items = sorted(grouped[sequence_id], key=lambda item: int(item["pointcloud_timestamp"]))
        if len(items) < 2:
            continue
        first = _pose(data_root / items[0]["calib_virtuallidar_to_world_path"])
        last = _pose(data_root / items[-1]["calib_virtuallidar_to_world_path"])
        drifts.append(float(np.linalg.norm(first - last)))
    sample_id = sequence_ids[0]
    sample = sorted(grouped[sample_id], key=lambda item: int(item["pointcloud_timestamp"]))[0]
    points = np.load(
        Path(config["project"]["centerpoint_root"]) / "points" / ("%s_%s.npy" % (sample_id, sample["frame_id"]))
    )
    radius = np.hypot(points[:, 0], points[:, 1])
    p10 = float(np.percentile(radius, 10))
    max_drift = float(max(drifts)) if drifts else None
    origin_ok = max_drift is not None and max_drift <= 0.2 and p10 <= 20.0
    return {
        "coordinate_system": "virtual_lidar",
        "range_definition": "检测或真值中心到虚拟激光雷达原点的平面距离 hypot(x, y)，单位 m",
        "origin_check": "序列内 virtuallidar_to_world 平移漂移，以及点云到原点的距离分布",
        "max_within_sequence_translation_drift_m": max_drift,
        "median_within_sequence_translation_drift_m": float(np.median(drifts)) if drifts else None,
        "sample_sequence": sample_id,
        "sample_point_radius_p10_m": p10,
        "sample_point_radius_median_m": float(np.median(radius)),
        "use_virtual_lidar_origin": bool(origin_ok),
    }


def _indexed_vehicles(row, protocol):
    output = []
    for index, item in enumerate(row.get("objects", [])):
        if _keep_vehicle(protocol, item):
            output.append((int(index), item))
    return output


def _local_counts(objects, radius):
    centers = [np.asarray(item["box"][:2], dtype=np.float64) for item in objects]
    counts = []
    limit = float(radius)
    for index, center in enumerate(centers):
        total = 0
        for other_index, other in enumerate(centers):
            if other_index == index:
                continue
            if float(np.linalg.norm(center - other)) <= limit:
                total += 1
        counts.append(total)
    return counts


def _density(point_count, box):
    volume = max(float(box[4]) * float(box[5]) * float(box[6]), 1.0e-3)
    return float(point_count) / volume


def _load_tracker(config):
    root = config["_root"]
    tracker_config = load_config(_path(root, config["tracker_config"]))
    device = str(config.get("device", "cuda"))
    model = build_model(
        _path(root, tracker_config["project"]["grae_root"]),
        tracker_config["model"]["in_channels"],
        tracker_config["model"]["layers"],
        tracker_config["num_classes"],
        device,
    )
    checkpoint = _path(root, config["checkpoint"])
    load_checkpoint(model, checkpoint, device)
    birth = yaml.safe_load(
        _path(root, tracker_config["tracker"]["birth_thresholds_file"]).read_text(encoding="utf-8")
    )["score_thresholds"]
    tracker = GraeTracker(
        model,
        tracker_config["classes"],
        birth,
        association_alpha=tracker_config["tracker"]["association_alpha"],
        age=tracker_config["tracker"]["age"],
        score_floor=tracker_config["tracker"].get("score_floor", 0.1),
    )
    return tracker, tracker_config, checkpoint


def _equivalent(left, right, sequence_id):
    if len(left) != len(right):
        raise AssertionError("序列%s调试输出帧数变化" % sequence_id)
    for left_row, right_row in zip(left, right):
        if int(left_row["timestamp"]) != int(right_row["timestamp"]):
            raise AssertionError("序列%s时间戳变化" % sequence_id)
        if len(left_row["objects"]) != len(right_row["objects"]):
            raise AssertionError("序列%s输出数量变化" % sequence_id)
        for left_item, right_item in zip(left_row["objects"], right_row["objects"]):
            if int(left_item["track_id"]) != int(right_item["track_id"]):
                raise AssertionError("序列%s track id变化" % sequence_id)
            if left_item["class_name"] != right_item["class_name"]:
                raise AssertionError("序列%s类别变化" % sequence_id)
            if abs(float(left_item["score"]) - float(right_item["score"])) > 1.0e-5:
                raise AssertionError("序列%s分数变化" % sequence_id)
            if not np.allclose(left_item["box"], right_item["box"], atol=1.0e-5, rtol=0.0):
                raise AssertionError("序列%s box变化" % sequence_id)


def _prediction_row(source, objects):
    return {
        "sequence_id": source["sequence_id"],
        "frame_index": source["frame_index"],
        "frame_id": source["frame_id"],
        "timestamp": source["timestamp"],
        "objects": objects,
    }


def _check_pair_geometry(detection, track, distance, feature_distance, geometry_score, learned, fusion):
    dx = float(detection["box"][0]) - float(track["predicted_xy"][0])
    dy = float(detection["box"][1]) - float(track["predicted_xy"][1])
    euclidean = match_euclidean(dx, dy)
    feature = feature_sqrt_norm(dx, dy)
    if abs(euclidean - float(distance)) > 1.0e-4:
        raise AssertionError("标准欧氏距离与调试矩阵不一致")
    if abs(feature - float(feature_distance)) > 1.0e-4:
        raise AssertionError("sqrt(norm) 特征距离与调试矩阵不一致")
    expected_geometry = math.exp(-euclidean)
    if abs(expected_geometry - float(geometry_score)) > 1.0e-4:
        raise AssertionError("几何分数不是 exp(-欧氏距离)")
    expected_fusion = 0.5 * float(learned) + 0.5 * float(geometry_score)
    if abs(expected_fusion - float(fusion)) > 1.0e-4:
        raise AssertionError("融合分数不是 0.5/0.5")
    for name, value in (("learned", learned), ("fusion", fusion), ("geometry", geometry_score)):
        if not math.isfinite(float(value)):
            raise AssertionError("%s 分数非法" % name)


def diagnose_sequence(config, tracker, protocol, sequence_id, detection_writer, pair_writer, failure_writer, store):
    root = config["_root"]
    converted = Path(config["project"]["converted_root"])
    detection_root = _path(root, config["detection_root"])
    point_root = Path(config["project"]["centerpoint_root"]) / "points"
    baseline_root = _path(root, config["baseline_prediction_root"])
    gt_rows = list(read_jsonl(converted / "sequences" / ("%s.jsonl" % sequence_id)))
    det_rows = list(read_jsonl(detection_root / ("%s.jsonl" % sequence_id)))
    baseline_rows = list(read_jsonl(baseline_root / ("%s.jsonl" % sequence_id)))
    if not (len(gt_rows) == len(det_rows) == len(baseline_rows)):
        raise ValueError("序列%s帧数不一致" % sequence_id)
    tracker.reset()
    tracker.association_mode = "fusion"
    tracker.high_cost_limit = 0.9
    tracker.low_cost_limit = 0.8
    tracker.enable_debug(True)
    book = IdentityBook()
    previously_live = set()
    predictions = []
    radius = float(config["local_radius_m"])
    loc_limit = float(config["localization_limit_m"])
    pred_limit = float(config["prediction_limit_m"])
    counts = Counter()
    velocity_max = 0.0
    for gt_row, det_row in zip(gt_rows, det_rows):
        if det_row["frame_id"] != gt_row["frame_id"] or int(det_row["timestamp"]) != int(gt_row["timestamp"]):
            raise ValueError("序列%s帧未对齐" % sequence_id)
        cleaned = [
            {"class_name": item["class_name"], "score": float(item["score"]), "box": [float(value) for value in item["box"]]}
            for item in det_row["objects"]
        ]
        outputs = tracker.update(cleaned, int(det_row["timestamp"]) / 1.0e6, det_row["frame_id"])
        predictions.append(_prediction_row(det_row, outputs))
        snapshot = tracker.last_debug
        if snapshot is None:
            raise AssertionError("诊断开关未产生 last_debug")
        points = np.load(point_root / ("%s_%s.npy" % (sequence_id, det_row["frame_id"])), mmap_mode="r")
        point_counts = [_point_count(points, item["box"]) for item in det_row["objects"]]
        gt_items = protocol.filter_gt(gt_row["objects"])
        indexed = _indexed_vehicles(det_row, protocol)
        matches = _match(
            [item["box"] for item in gt_items],
            [item["box"] for _, item in indexed],
            config["match_iou"],
            config["match_center_gate_m"],
        )
        det_info = {}
        gt_match = {}
        for gt_index, local_index, iou in matches:
            raw_index, detection = indexed[local_index]
            gt_item = gt_items[gt_index]
            error_x = float(detection["box"][0]) - float(gt_item["box"][0])
            error_y = float(detection["box"][1]) - float(gt_item["box"][1])
            det_info[int(raw_index)] = {
                "gt_id": str(gt_item["source_track_id"]),
                "iou": float(iou),
                "center": match_euclidean(error_x, error_y),
                "gt_xy": (float(gt_item["box"][0]), float(gt_item["box"][1])),
            }
            gt_match[gt_index] = int(raw_index)
        vehicle_items = [item for _, item in indexed]
        vehicle_local = _local_counts(vehicle_items, radius)
        local_by_index = {raw_index: vehicle_local[position] for position, (raw_index, _) in enumerate(indexed)}
        gt_local = _local_counts(gt_items, radius) if gt_items else []
        detection_rows = []
        for gt_index, gt_item in enumerate(gt_items):
            gt_range = match_euclidean(gt_item["box"][0], gt_item["box"][1])
            row = {
                "sequence_id": sequence_id,
                "frame_id": gt_row["frame_id"],
                "frame_index": int(gt_row["frame_index"]),
                "gt_id": str(gt_item["source_track_id"]),
                "class_name": gt_item["class_name"],
                "gt_range_m": gt_range,
                "matched": 0,
                "det_index": "",
                "det_range_m": "",
                "error_xy_m": "",
                "error_x_m": "",
                "error_y_m": "",
                "abs_error_x_m": "",
                "abs_error_y_m": "",
                "bev_iou": "",
                "iou_3d": "",
                "score": "",
                "point_count": "",
                "point_density": "",
                "local_density": "",
                "local_gt_count": gt_local[gt_index],
            }
            raw_index = gt_match.get(gt_index)
            if raw_index is not None:
                detection = det_row["objects"][raw_index]
                info = det_info[raw_index]
                error_x = float(detection["box"][0]) - float(gt_item["box"][0])
                error_y = float(detection["box"][1]) - float(gt_item["box"][1])
                count = int(point_counts[raw_index])
                row.update(
                    {
                        "matched": 1,
                        "det_index": int(raw_index),
                        "det_range_m": match_euclidean(detection["box"][0], detection["box"][1]),
                        "error_xy_m": info["center"],
                        "error_x_m": error_x,
                        "error_y_m": error_y,
                        "abs_error_x_m": abs(error_x),
                        "abs_error_y_m": abs(error_y),
                        "bev_iou": _bev_iou(detection["box"], gt_item["box"]),
                        "iou_3d": info["iou"],
                        "score": float(detection["score"]),
                        "point_count": count,
                        "point_density": _density(count, detection["box"]),
                        "local_density": int(local_by_index.get(raw_index, 0)),
                    }
                )
            detection_rows.append(row)
        _append_rows(detection_writer, DETECTION_FIELDS, detection_rows)
        group = snapshot["groups"][0]
        candidates = group.get("candidate_detections", [])
        pre_tracks = group.get("pre_tracks", [])
        association = group.get("association", {})
        assignments = group.get("assignments", [])
        created = group.get("created", [])
        assigned_pairs = {(int(item["input_index"]), int(item["track_id"])) for item in assignments}
        output_ids = {int(value) for value in snapshot.get("output_track_ids", [])}
        pre_ids = [int(item["track_id"]) for item in pre_tracks]
        owners = defaultdict(list)
        for track_id in pre_ids:
            gt_id, conflict, _ = book.state(track_id)
            if gt_id is not None and not conflict:
                owners[gt_id].append(track_id)
        shared_ids = {track_id for values in owners.values() if len(values) > 1 for track_id in values}
        candidate_row = {int(item["input_index"]): index for index, item in enumerate(candidates)}
        track_column = {int(item["track_id"]): index for index, item in enumerate(pre_tracks)}
        velocity_max = max(velocity_max, float(association.get("velocity_before_max", 0.0) or 0.0))
        pair_rows = []
        if candidates and pre_tracks and association.get("distance_matrix"):
            for det_pos, detection in enumerate(candidates):
                raw_index = int(detection["input_index"])
                source = det_row["objects"][raw_index]
                count = int(point_counts[raw_index])
                det_gt = det_info.get(raw_index)
                for track_pos, track in enumerate(pre_tracks):
                    track_id = int(track["track_id"])
                    if association.get("affinity_matrix"):
                        _check_pair_geometry(
                            detection,
                            track,
                            association["distance_matrix"][det_pos][track_pos],
                            association["feature_distance_matrix"][det_pos][track_pos],
                            association["geometry_score_matrix"][det_pos][track_pos],
                            association["affinity_matrix"][det_pos][track_pos],
                            association["fusion_score_matrix"][det_pos][track_pos],
                        )
                    track_gt, conflict, quality = book.state(track_id)
                    label, status = label_from_ids(
                        None if det_gt is None else det_gt["gt_id"],
                        track_gt,
                        conflict,
                        track_id in shared_ids,
                    )
                    strict = strict_label(
                        label,
                        status,
                        None if det_gt is None else det_gt["iou"],
                        None if det_gt is None else det_gt["center"],
                        quality,
                        config["strict_iou"],
                        config["strict_center_m"],
                    )
                    seconds = ""
                    if track_id in book.last_time:
                        seconds = (int(det_row["timestamp"]) - int(book.last_time[track_id])) / 1.0e6
                    velocity = track.get("velocity", [0.0, 0.0])
                    velocity_max = max(velocity_max, abs(float(velocity[0])), abs(float(velocity[1])))
                    box_gap = match_euclidean(
                        float(track["box"][0]) - float(track["predicted_xy"][0]),
                        float(track["box"][1]) - float(track["predicted_xy"][1]),
                    )
                    if box_gap > 1.0e-4:
                        raise AssertionError("零速度外推后预测位置偏离轨迹框")
                    row = {
                        "sequence_id": sequence_id,
                        "frame_id": det_row["frame_id"],
                        "frame_index": int(det_row["frame_index"]),
                        "det_index": raw_index,
                        "track_id": track_id,
                        "det_x": float(detection["box"][0]),
                        "det_y": float(detection["box"][1]),
                        "track_x": float(track["predicted_xy"][0]),
                        "track_y": float(track["predicted_xy"][1]),
                        "euclidean_m": float(association["distance_matrix"][det_pos][track_pos]),
                        "sqrt_norm_feature": float(association["feature_distance_matrix"][det_pos][track_pos]),
                        "exp_neg_distance": float(association["geometry_score_matrix"][det_pos][track_pos]),
                        "logit": float(association["logit_matrix"][det_pos][track_pos]),
                        "learned_score": float(association["affinity_matrix"][det_pos][track_pos]),
                        "fusion_score": float(association["fusion_score_matrix"][det_pos][track_pos]),
                        "match_score": float(association["blended_score_matrix"][det_pos][track_pos]),
                        "det_score": float(source["score"]),
                        "range_m": match_euclidean(detection["box"][0], detection["box"][1]),
                        "point_count": count,
                        "point_density": _density(count, detection["box"]),
                        "local_density": int(local_by_index.get(raw_index, 0)),
                        "track_age": int(track["age"]),
                        "seconds_since_match": seconds,
                        "dt_seconds": float(association.get("dt_seconds", 0.0)),
                        "class_ok": int(int(detection["class_index"]) == int(track["class_index"])),
                        "gate_pass": int(bool(association["gate_mask"][det_pos][track_pos])),
                        "matched": int((raw_index, track_id) in assigned_pairs),
                        "det_gt_id": "" if det_gt is None else det_gt["gt_id"],
                        "track_gt_id": "" if track_gt is None else track_gt,
                        "label": label,
                        "strict_label": strict,
                        "identity_status": status,
                    }
                    pair_rows.append(row)
                    counts[label] += 1
                    counts["strict_" + strict] += 1
                    counts["pairs"] += 1
                    if label in {"positive", "negative"}:
                        bucket = store[sequence_id]
                        bucket["distance"].append(row["euclidean_m"])
                        bucket["geometry"].append(row["exp_neg_distance"])
                        bucket["learned"].append(row["learned_score"])
                        bucket["fusion"].append(row["fusion_score"])
                        bucket["label"].append(1 if label == "positive" else 0)
                        bucket["strict"].append(1 if strict == label else 0)
                        bucket["range"].append(row["range_m"])
                        bucket["points"].append(row["point_count"])
                        bucket["local"].append(row["local_density"])
                        bucket["matched"].append(row["matched"])
                        if label == "negative" and int(row["matched"]) == 1:
                            _push_example(store["near_matched"], row, False)
                        elif label == "negative":
                            _push_example(store["near_any"], row, False)
                        if label == "positive" and int(row["matched"]) == 0:
                            _push_example(store["far_missed"], row, True)
                        elif label == "positive":
                            _push_example(store["far_any"], row, True)
        _append_rows(pair_writer, PAIR_FIELDS, pair_rows)
        failure_rows = []
        known_tracks = defaultdict(list)
        for track_id, gt_id in book.gt_of_track.items():
            if track_id not in book.conflict:
                known_tracks[gt_id].append(int(track_id))
        for gt_index, gt_item in enumerate(gt_items):
            gt_id = str(gt_item["source_track_id"])
            holders = known_tracks.get(gt_id, [])
            live = [track_id for track_id in holders if track_id in track_column]
            if not holders:
                identity_state = "unseen"
                primary = "new_object"
                expected = None
            elif len(live) > 1 or any(track_id in shared_ids for track_id in live):
                identity_state = "ambiguous"
                primary = "identity_ambiguous"
                expected = None
            elif len(live) == 1:
                identity_state = "unique"
                expected = live[0]
            elif gt_id in previously_live:
                identity_state = "expired"
                primary = "track_management"
                expected = None
            else:
                identity_state = "absent_continued"
                primary = "track_absent_continued"
                expected = None
            raw_index = gt_match.get(gt_index)
            has_detection = raw_index is not None
            in_pool = raw_index in candidate_row if has_detection else False
            loc_error = None if not has_detection else det_info[raw_index]["center"]
            has_track = expected is not None
            pred_error = None
            cv_error = None
            class_ok = False
            gate_pass = False
            assigned = False
            seconds = ""
            velocity_abs = ""
            det_score = ""
            point_count = ""
            if has_track:
                track = pre_tracks[track_column[expected]]
                pred_error = match_euclidean(
                    float(track["predicted_xy"][0]) - float(gt_item["box"][0]),
                    float(track["predicted_xy"][1]) - float(gt_item["box"][1]),
                )
                cv_error = book.motion_oracle(expected, det_row["timestamp"], gt_item["box"][:2])
                if expected in book.last_time:
                    seconds = (int(det_row["timestamp"]) - int(book.last_time[expected])) / 1.0e6
                velocity = track.get("velocity", [0.0, 0.0])
                velocity_abs = max(abs(float(velocity[0])), abs(float(velocity[1])))
            if has_detection and has_track and in_pool:
                det_pos = candidate_row[raw_index]
                track_pos = track_column[expected]
                class_ok = int(candidates[det_pos]["class_index"]) == int(pre_tracks[track_pos]["class_index"])
                gate_pass = bool(association["gate_mask"][det_pos][track_pos])
                assigned = (raw_index, expected) in assigned_pairs
                det_score = float(det_row["objects"][raw_index]["score"])
                point_count = int(point_counts[raw_index])
            elif has_detection:
                det_score = float(det_row["objects"][raw_index]["score"])
                point_count = int(point_counts[raw_index])
            output_ok = expected in output_ids if expected is not None else False
            if identity_state == "unique":
                primary = exclusive_primary(
                    has_detection and in_pool,
                    has_track,
                    bool(class_ok),
                    bool(gate_pass),
                    bool(assigned),
                    bool(output_ok),
                    loc_error,
                    pred_error,
                    loc_limit,
                    pred_limit,
                )
            failure_rows.append(
                {
                    "sequence_id": sequence_id,
                    "frame_id": gt_row["frame_id"],
                    "frame_index": int(gt_row["frame_index"]),
                    "gt_id": gt_id,
                    "range_m": match_euclidean(gt_item["box"][0], gt_item["box"][1]),
                    "has_detection": int(bool(has_detection)),
                    "in_candidate_pool": int(bool(in_pool)),
                    "loc_error_m": "" if loc_error is None else loc_error,
                    "has_track": int(bool(has_track)),
                    "pred_error_m": "" if pred_error is None else pred_error,
                    "cv_error_m": "" if cv_error is None else cv_error,
                    "class_ok": int(bool(class_ok)),
                    "gate_pass": int(bool(gate_pass)),
                    "assigned": int(bool(assigned)),
                    "large_loc": int(loc_error is not None and loc_error > loc_limit),
                    "large_pred": int(pred_error is not None and pred_error > pred_limit),
                    "primary": primary,
                    "identity_state": identity_state,
                    "track_id": "" if expected is None else expected,
                    "det_score": det_score,
                    "point_count": point_count,
                    "seconds_since_match": seconds,
                    "velocity_abs": velocity_abs,
                }
            )
            store["failures"].append(failure_rows[-1])
        _append_rows(failure_writer, FAILURE_FIELDS, failure_rows)
        created_gt = set()
        for item in list(assignments) + list(created):
            raw_index = int(item["input_index"])
            info = det_info.get(raw_index)
            source = det_row["objects"][raw_index]
            if info is not None and item in created:
                created_gt.add(info["gt_id"])
            book.commit(
                item["track_id"],
                None if info is None else info["gt_id"],
                source["box"][:2],
                None if info is None else info["iou"],
                det_row["timestamp"],
            )
        previously_live = set(owners.keys()) | created_gt
    _equivalent(predictions, baseline_rows, sequence_id)
    return counts, velocity_max


def _push_example(bucket, row, reverse):
    bucket.append(dict(row))
    bucket.sort(key=lambda item: float(item["euclidean_m"]), reverse=reverse)
    del bucket[3:]


def _empty_store():
    return {"failures": [], "near_matched": [], "far_missed": [], "near_any": [], "far_any": []}


def _sequence_store():
    return {
        "distance": [],
        "geometry": [],
        "learned": [],
        "fusion": [],
        "label": [],
        "strict": [],
        "range": [],
        "points": [],
        "local": [],
        "matched": [],
    }


def _labeled_sequences(store):
    return [key for key, value in store.items() if isinstance(value, dict) and "label" in value]


def _as_arrays(bucket):
    return {key: np.asarray(value) for key, value in bucket.items()}


def _concat(parts, key):
    if not parts:
        return np.asarray([])
    return np.concatenate([part[key] for part in parts])


def _score_row(group, subgroup, labels, geometry, learned, fusion, fpr_targets):
    row = {
        "record_type": "ranking",
        "group": group,
        "subgroup": subgroup,
        "count": int(len(labels)),
        "n_pos": int(np.sum(labels == 1)) if len(labels) else 0,
        "n_neg": int(np.sum(labels == 0)) if len(labels) else 0,
        "value_geometry": _safe_auc(labels, geometry),
        "value_learned": _safe_auc(labels, learned),
        "value_fusion": _safe_auc(labels, fusion),
        "ap_geometry": _safe_ap(labels, geometry),
        "ap_learned": _safe_ap(labels, learned),
        "ap_fusion": _safe_ap(labels, fusion),
        "note": "",
    }
    for target in fpr_targets:
        tag = "%02d" % int(round(float(target) * 100))
        row["recall_fpr%s_geometry" % tag] = recall_at_fpr(labels, geometry, target)
        row["recall_fpr%s_learned" % tag] = recall_at_fpr(labels, learned, target)
        row["recall_fpr%s_fusion" % tag] = recall_at_fpr(labels, fusion, target)
    if row["n_pos"] < 20 or row["n_neg"] < 20:
        row["note"] = "样本不足"
    return row


def _mask_metrics(arrays, mask, group, subgroup, fpr_targets):
    if mask is None:
        mask = np.ones(len(arrays["label"]), dtype=bool)
    if int(mask.sum()) == 0:
        return _score_row(group, subgroup, np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([]), fpr_targets)
    return _score_row(
        group,
        subgroup,
        arrays["label"][mask],
        arrays["geometry"][mask],
        arrays["learned"][mask],
        arrays["fusion"][mask],
        fpr_targets,
    )


def _rate_contrast(store, distance_name, left_mask_fn, right_mask_fn, config):
    sequence_ids = _labeled_sequences(store)
    left_positive = []
    left_count = []
    right_positive = []
    right_count = []
    for sequence_id in sequence_ids:
        arrays = _as_arrays(store[sequence_id])
        if len(arrays["label"]) == 0:
            continue
        distance_mask = np.asarray([bin_name(value, config["distance_edges_m"]) == distance_name for value in arrays["distance"]])
        left = distance_mask & left_mask_fn(arrays)
        right = distance_mask & right_mask_fn(arrays)
        if int(left.sum()) == 0 and int(right.sum()) == 0:
            continue
        left_positive.append(int(arrays["label"][left].sum()))
        left_count.append(int(left.sum()))
        right_positive.append(int(arrays["label"][right].sum()))
        right_count.append(int(right.sum()))
    if not left_count:
        return None
    left_positive = np.asarray(left_positive, dtype=np.float64)
    left_count = np.asarray(left_count, dtype=np.float64)
    right_positive = np.asarray(right_positive, dtype=np.float64)
    right_count = np.asarray(right_count, dtype=np.float64)
    if left_count.sum() < int(config["min_cell_count"]) or right_count.sum() < int(config["min_cell_count"]):
        return None
    observed = float(left_positive.sum() / left_count.sum() - right_positive.sum() / right_count.sum())
    rng = np.random.default_rng(int(config["seed"]))
    samples = []
    count = len(left_count)
    for _ in range(int(config["bootstrap_samples"])):
        chosen = rng.integers(0, count, size=count)
        left_total = left_count[chosen].sum()
        right_total = right_count[chosen].sum()
        if left_total <= 0.0 or right_total <= 0.0:
            continue
        samples.append(float(left_positive[chosen].sum() / left_total - right_positive[chosen].sum() / right_total))
    if len(samples) < 100:
        return None
    low, high = np.percentile(samples, [2.5, 97.5])
    return {
        "difference": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
        "left_count": int(left_count.sum()),
        "right_count": int(right_count.sum()),
        "left_rate": float(left_positive.sum() / left_count.sum()),
        "right_rate": float(right_positive.sum() / right_count.sum()),
    }


def _range_mask(name, edges):
    def apply(arrays):
        return np.asarray([bin_name(value, edges) == name for value in arrays["range"]])

    return apply


def _point_mask(name, edges):
    def apply(arrays):
        return np.asarray([bin_name(value, edges) == name for value in arrays["points"]])

    return apply


def _local_mask(name, edges):
    def apply(arrays):
        return np.asarray([bin_name(value, edges) == name for value in arrays["local"]])

    return apply


def collect_metrics(config, store, label_counts):
    sequence_ids = _labeled_sequences(store)
    parts = [_as_arrays(store[sequence_id]) for sequence_id in sequence_ids if store[sequence_id]["label"]]
    if not parts:
        return {"rows": [], "arrays": None, "contrasts": []}
    arrays = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    fpr_targets = config["fpr_targets"]
    rows = [_mask_metrics(arrays, None, "all", "known", fpr_targets)]
    strict = arrays["strict"] == 1
    rows.append(_mask_metrics(arrays, strict, "all", "strict", fpr_targets))
    distance_names = []
    edges = config["distance_edges_m"]
    for left, right in zip(edges[:-1], edges[1:]):
        distance_names.append("%g-%g" % (float(left), float(right)))
    distance_names.append("%g+" % float(edges[-1]))
    for name in distance_names:
        mask = np.asarray([bin_name(value, edges) == name for value in arrays["distance"]])
        rows.append(_mask_metrics(arrays, mask, "distance", name, fpr_targets))
    for name in [bin_name((float(left) + float(right)) / 2.0, config["range_edges_m"]) for left, right in zip(config["range_edges_m"][:-1], config["range_edges_m"][1:])] + ["%g+" % float(config["range_edges_m"][-1])]:
        mask = np.asarray([bin_name(value, config["range_edges_m"]) == name for value in arrays["range"]])
        rows.append(_mask_metrics(arrays, mask, "range", name, fpr_targets))
    point_names = ["%g-%g" % (float(a), float(b)) for a, b in zip(config["point_edges"][:-1], config["point_edges"][1:])]
    point_names.append("%g+" % float(config["point_edges"][-1]))
    for name in point_names:
        mask = np.asarray([bin_name(value, config["point_edges"]) == name for value in arrays["points"]])
        rows.append(_mask_metrics(arrays, mask, "points", name, fpr_targets))
    contrasts = []
    range_names = ["%g-%g" % (float(a), float(b)) for a, b in zip(config["range_edges_m"][:-1], config["range_edges_m"][1:])]
    if len(range_names) >= 3:
        for distance_name in distance_names:
            result = _rate_contrast(
                store,
                distance_name,
                _range_mask(range_names[0], config["range_edges_m"]),
                _range_mask(range_names[2], config["range_edges_m"]),
                config,
            )
            if result:
                result.update({"family": "range", "distance": distance_name, "left": range_names[0], "right": range_names[2]})
                contrasts.append(result)
    for distance_name in distance_names:
        result = _rate_contrast(
            store,
            distance_name,
            _point_mask(point_names[0], config["point_edges"]),
            _point_mask(point_names[-1], config["point_edges"]),
            config,
        )
        if result:
            result.update({"family": "points", "distance": distance_name, "left": point_names[0], "right": point_names[-1]})
            contrasts.append(result)
        result = _rate_contrast(
            store,
            distance_name,
            _local_mask("0-1", config["local_edges"]),
            _local_mask("%g+" % float(config["local_edges"][-1]), config["local_edges"]),
            config,
        )
        if result:
            result.update({"family": "local", "distance": distance_name, "left": "0-1", "right": "%g+" % float(config["local_edges"][-1])})
            contrasts.append(result)
    rows.append(
        {
            "record_type": "label_count",
            "group": "all",
            "subgroup": "pairs",
            "count": int(label_counts.get("pairs", 0)),
            "n_pos": int(label_counts.get("positive", 0)),
            "n_neg": int(label_counts.get("negative", 0)),
            "value_geometry": int(label_counts.get("unknown", 0)),
            "value_learned": int(label_counts.get("strict_positive", 0)),
            "value_fusion": int(label_counts.get("strict_negative", 0)),
            "note": "value_geometry 列为未知候选数，后两列为严格正负样本数",
        }
    )
    return {"rows": rows, "arrays": arrays, "contrasts": contrasts, "distance_names": distance_names}


def _positive_bins(distance, labels, edges):
    names = ["%g-%g" % (float(a), float(b)) for a, b in zip(edges[:-1], edges[1:])] + ["%g+" % float(edges[-1])]
    output = []
    for name in names:
        mask = np.asarray([bin_name(value, edges) == name for value in distance])
        count = int(mask.sum())
        output.append(
            {
                "bin": name,
                "count": count,
                "positive": int(labels[mask].sum()) if count else 0,
                "rate": float(labels[mask].mean()) if count else None,
            }
        )
    return output


def _grouped_error(rows):
    matched = [row for row in rows if int(row["matched"]) == 1]
    grouped = defaultdict(list)
    for row in matched:
        grouped[bin_name(row["det_range_m"], [0, 50, 100, 150])].append(row)
    summary = {}
    for name, items in grouped.items():
        summary[name] = {
            "error": _describe([item["error_xy_m"] for item in items]),
            "error_x": _describe([item["error_x_m"] for item in items]),
            "error_y": _describe([item["error_y_m"] for item in items]),
            "points": _describe([item["point_count"] for item in items]),
            "score": _describe([item["score"] for item in items]),
        }
    misses = defaultdict(lambda: {"gt": 0, "miss": 0})
    for row in rows:
        name = bin_name(row["gt_range_m"], [0, 50, 100, 150])
        misses[name]["gt"] += 1
        misses[name]["miss"] += int(row["matched"]) == 0
    return summary, {key: {"gt": value["gt"], "miss": value["miss"], "miss_rate": value["miss"] / value["gt"] if value["gt"] else None} for key, value in misses.items()}


def _spearman(rows):
    matched = [row for row in rows if int(row["matched"]) == 1]
    error = np.asarray([float(row["error_xy_m"]) for row in matched], dtype=np.float64)
    output = {}
    for name, key in (("range", "det_range_m"), ("points", "point_count"), ("score", "score")):
        covariate = np.asarray([float(row[key]) for row in matched], dtype=np.float64)
        if len(error) < 3:
            output[name] = None
            continue
        rho, pvalue = spearmanr(covariate, error)
        output[name] = {"rho": float(rho), "p_value": float(pvalue), "count": int(len(error))}
    return output


def _load_detection_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["matched"] = int(row["matched"])
        for key in ("gt_range_m", "det_range_m", "error_xy_m", "error_x_m", "error_y_m", "point_count", "score", "point_density"):
            row[key] = float(row[key]) if row[key] != "" else None
    return rows


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _save(path, figure):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    figure.clf()
    import matplotlib.pyplot as plt

    plt.close(figure)


def plot_figures(config, detection_rows, metrics, output_dir):
    plt = _style()
    figure_dir = output_dir / "figures"
    summary, misses = _grouped_error(detection_rows)
    order = ["0-50", "50-100", "100-150", "150+"]
    present = [name for name in order if name in summary]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    if present:
        means = [summary[name]["error"]["mean"] for name in present]
        medians = [summary[name]["error"]["median"] for name in present]
        tails = [summary[name]["error"]["p90"] for name in present]
        axes[0].plot(present, means, marker="o", label="均值")
        axes[0].plot(present, medians, marker="o", label="中位数")
        axes[0].plot(present, tails, marker="o", label="P90")
        axes[0].set_ylabel("水平定位误差 / m")
        axes[0].set_xlabel("观测距离 / m")
        axes[0].legend()
        miss_names = [name for name in order if name in misses]
        axes[1].bar(miss_names, [misses[name]["miss_rate"] for name in miss_names], color="#4C78A8")
        axes[1].set_ylabel("GT 漏检率")
        axes[1].set_xlabel("GT 距离 / m")
    axes[0].set_title("匹配检测的定位误差")
    axes[1].set_title("各距离段漏检率")
    _save(figure_dir / "figure_1_range_vs_localization.png", figure)

    matched = [row for row in detection_rows if row["matched"] == 1 and row["point_count"] is not None]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    if matched:
        points = np.asarray([row["point_count"] for row in matched])
        errors = np.asarray([row["error_xy_m"] for row in matched])
        scores = np.asarray([row["score"] for row in matched])
        axes[0].scatter(points, errors, s=6, alpha=0.25, c="#4C78A8")
        axes[0].set_xscale("log")
        axes[0].set_xlabel("检测框内点数")
        axes[0].set_ylabel("水平定位误差 / m")
        axes[1].scatter(scores, errors, s=6, alpha=0.25, c="#E45756")
        axes[1].set_xlabel("检测置信度")
        axes[1].set_ylabel("水平定位误差 / m")
    axes[0].set_title("点数与定位误差")
    axes[1].set_title("置信度与定位误差")
    _save(figure_dir / "figure_2_observation_quality.png", figure)

    arrays = metrics.get("arrays")
    if arrays is None or len(arrays["label"]) == 0:
        return ["figure_3_distance_vs_association.png", "figure_4_conditional_reliability.png", "figure_5_score_comparison.png", "figure_6_failure_cases.png"]
    bins = _positive_bins(arrays["distance"], arrays["label"], config["distance_edges_m"])
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    usable = [item for item in bins if item["count"] > 0]
    axis.bar([item["bin"] for item in usable], [item["rate"] for item in usable], color="#4C78A8")
    axis.set_xlabel("标准欧氏距离 / m")
    axis.set_ylabel("真实正关联比例")
    axis.set_title("几何距离与真实正关联比例")
    _save(figure_dir / "figure_3_distance_vs_association.png", figure)

    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    distance_names = metrics["distance_names"]
    range_names = ["0-50", "50-100", "100-150"]
    width = 0.25
    positions = np.arange(len(distance_names))
    for offset, range_name in enumerate(range_names):
        rates = []
        for distance_name in distance_names:
            mask = np.asarray(
                [
                    bin_name(distance, config["distance_edges_m"]) == distance_name and bin_name(obs, config["range_edges_m"]) == range_name
                    for distance, obs in zip(arrays["distance"], arrays["range"])
                ]
            )
            count = int(mask.sum())
            rates.append(float(arrays["label"][mask].mean()) if count >= int(config["min_cell_count"]) else np.nan)
        axis.bar(positions + (offset - 1) * width, rates, width=width, label=range_name + " m")
    axis.set_xticks(positions)
    axis.set_xticklabels(distance_names)
    axis.set_ylabel("真实正关联比例")
    axis.set_xlabel("几何距离区间 / m")
    axis.set_title("相同距离下不同观测距离的关联可靠性")
    axis.legend()
    _save(figure_dir / "figure_4_conditional_reliability.png", figure)

    figure, axis = plt.subplots(figsize=(6.5, 5))
    from sklearn.metrics import roc_curve

    for key, label, color in (
        ("geometry", "exp(-distance)", "#9AA0A6"),
        ("learned", "GRAE Sigmoid", "#E45756"),
        ("fusion", "0.5/0.5 融合", "#4C78A8"),
    ):
        fpr, tpr, _ = roc_curve(arrays["label"], arrays[key])
        axis.plot(fpr, tpr, color=color, label=label)
    axis.plot([0, 1], [0, 1], linestyle="--", color="#BBBBBB")
    axis.set_xlabel("假正例率")
    axis.set_ylabel("真正例率")
    axis.set_title("三种关联分数的区分能力")
    axis.legend()
    _save(figure_dir / "figure_5_score_comparison.png", figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    negatives = metrics["examples"]["near"]
    positives = metrics["examples"]["far"]
    _draw_cases(axes[0], negatives[:3], "近距离负关联")
    _draw_cases(axes[1], positives[:3], "远距离正关联")
    _save(figure_dir / "figure_6_failure_cases.png", figure)
    return []


def _draw_cases(axis, cases, title):
    axis.set_title(title)
    axis.set_aspect("equal")
    axis.set_xlabel("x / m")
    axis.set_ylabel("y / m")
    if not cases:
        axis.text(0.5, 0.5, "没有足够案例", transform=axis.transAxes, ha="center")
        return
    for index, item in enumerate(cases):
        axis.scatter(item["det_x"], item["det_y"], marker="o", s=30)
        axis.scatter(item["track_x"], item["track_y"], marker="x", s=30)
        axis.plot([item["det_x"], item["track_x"]], [item["det_y"], item["track_y"]], linewidth=1)
        axis.annotate(
            "%.1fm" % float(item["euclidean_m"]),
            ((float(item["det_x"]) + float(item["track_x"])) / 2.0, (float(item["det_y"]) + float(item["track_y"])) / 2.0),
            fontsize=8,
        )
    axis.scatter([], [], marker="o", label="检测")
    axis.scatter([], [], marker="x", label="轨迹预测")
    axis.legend()


def plot_failures(config, failures, output_dir):
    plt = _style()
    figure_dir = output_dir / "figures"
    opportunities = [
        row
        for row in failures
        if row["primary"] not in {"new_object", "identity_ambiguous", "track_absent_continued"}
    ]
    failed = [row for row in opportunities if row["primary"] != "success"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    counts = Counter(row["primary"] for row in failed)
    names = [name for name in PRIMARY_REASONS if counts[name]]
    axes[0].barh(names, [counts[name] for name in names], color="#4C78A8")
    axes[0].set_xlabel("失败次数")
    axes[0].set_title("互斥主要原因")
    range_names = ["0-50", "50-100", "100-150", "150+"]
    bottoms = np.zeros(len(range_names))
    for reason in names:
        values = []
        for name in range_names:
            values.append(sum(1 for row in failed if row["primary"] == reason and bin_name(row["range_m"], config["range_edges_m"]) == name))
        axes[1].bar(range_names, values, bottom=bottoms, label=reason)
        bottoms = bottoms + np.asarray(values, dtype=np.float64)
    axes[1].set_xlabel("GT 距离 / m")
    axes[1].set_ylabel("失败次数")
    axes[1].set_title("不同距离的失败构成")
    axes[1].legend(fontsize=7)
    _save(figure_dir / "figure_7_error_attribution.png", figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    live = [row for row in opportunities if row["pred_error_m"] != "" and row["primary"] in {"success", *PRIMARY_REASONS}]
    if live:
        edges = [0, 0.5, 1, 2, 4, 8]
        labels = ["0-0.5", "0.5-1", "1-2", "2-4", "4-8", "8+"]
        rates = []
        numbers = []
        for index, name in enumerate(labels):
            if index < len(edges) - 1:
                chosen = [row for row in live if edges[index] <= float(row["pred_error_m"]) < edges[index + 1]]
            else:
                chosen = [row for row in live if float(row["pred_error_m"]) >= edges[-1]]
            numbers.append(len(chosen))
            rates.append(sum(row["primary"] != "success" for row in chosen) / len(chosen) if chosen else np.nan)
        axes[0].bar(labels, rates, color="#E45756")
        axes[0].set_ylabel("关联失败率")
        axes[0].set_xlabel("零速度预测误差 / m")
    zero_values = [float(row["pred_error_m"]) for row in live]
    cv_values = [float(row["cv_error_m"]) for row in live if row["cv_error_m"] != ""]
    if zero_values:
        axes[1].hist(zero_values, bins=40, alpha=0.7, label="零速度", color="#4C78A8")
    if cv_values:
        axes[1].hist(cv_values, bins=40, alpha=0.55, label="检测差分外推", color="#F58518")
    axes[1].set_xlabel("到当前 GT 的平面误差 / m")
    axes[1].legend()
    axes[0].set_title("预测误差与关联失败")
    axes[1].set_title("零速度与差分外推误差")
    _save(figure_dir / "figure_8_motion_error.png", figure)


def _read_failures(path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["range_m"] = float(row["range_m"])
        for key in ("pred_error_m", "cv_error_m", "loc_error_m"):
            row[key] = row[key]
    return rows


def failure_table(failures):
    opportunities = [
        row
        for row in failures
        if row["primary"] not in {"new_object", "identity_ambiguous", "track_absent_continued"}
    ]
    failed = [row for row in opportunities if row["primary"] != "success"]
    counts = Counter(row["primary"] for row in failed)
    total = len(failed)
    rows = []
    for reason in PRIMARY_REASONS:
        rows.append(
            {
                "reason": reason,
                "count": int(counts[reason]),
                "share_of_failures": (counts[reason] / total) if total else None,
            }
        )
    raw = {
        "opportunities": len(opportunities),
        "failures": total,
        "success": sum(row["primary"] == "success" for row in opportunities),
        "new_object": sum(row["primary"] == "new_object" for row in failures),
        "identity_ambiguous": sum(row["primary"] == "identity_ambiguous" for row in failures),
        "track_absent_continued": sum(row["primary"] == "track_absent_continued" for row in failures),
        "large_loc_on_failure": sum(int(row["large_loc"]) for row in failed),
        "large_pred_on_failure": sum(int(row["large_pred"]) for row in failed),
        "large_loc_on_opportunity": sum(int(row["large_loc"]) for row in opportunities),
        "large_pred_on_opportunity": sum(int(row["large_pred"]) for row in opportunities),
    }
    return rows, raw


def decide(config, contrasts, ranking, raw_failures, failure_rows):
    reliable = [
        item
        for item in contrasts
        if item["ci_excludes_zero"] and abs(item["difference"]) >= float(config["conditional_gap_min"])
    ]
    learned = ranking.get("value_learned")
    geometry = ranking.get("value_geometry")
    gain = None if learned is None or geometry is None else float(learned) - float(geometry)
    shares = {row["reason"]: row["share_of_failures"] or 0.0 for row in failure_rows}
    miss = shares.get("detection_missing", 0.0)
    examined = len(contrasts)
    if len(reliable) == 0:
        decision = "NO-GO"
        text = "预注册对比里，没有观测条件对比同时满足样本量、绝对差和序列重采样置信区间。"
    elif gain is not None and gain >= float(config["learned_gain_min"]) and len(reliable) < 2:
        decision = "CONDITIONAL"
        text = "个别距离层能看到观测条件差异，同时 GRAE 学习分数的整体排序增益已经达到预注册幅度。"
    elif len(reliable) >= 2 and miss < float(config["missing_share_max"]) and (gain is None or gain < float(config["learned_gain_min"])):
        decision = "GO"
        text = "至少两个预注册对比显示相同距离下关联可靠性随观测条件变化，且学习分数没有把整体排序差距拉开到预注册幅度。"
    else:
        decision = "CONDITIONAL"
        text = "观测条件差异存在，但它只出现在部分对比中，或与检测缺失、学习分数增益同时出现。"
    return {
        "decision": decision,
        "text": text,
        "reliable_contrasts": reliable,
        "examined_contrasts": examined,
        "auroc_gain": gain,
        "missing_share": miss,
    }


def _markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def _num(value, digits=3):
    if value is None:
        return "—"
    return "%.*f" % (digits, float(value))


def write_summary(config, output_dir, sensor, detection_rows, metrics, label_counts, judgment, ablation_rows, skipped_figures):
    summary, misses = _grouped_error(detection_rows)
    spearman = _spearman(detection_rows)
    failure_rows, raw = failure_table(_read_failures(output_dir / "failure_attribution.csv") if False else config["_failures"])
    ranking = next(row for row in metrics["rows"] if row["record_type"] == "ranking" and row["subgroup"] == "known")
    strict = next(row for row in metrics["rows"] if row["record_type"] == "ranking" and row["subgroup"] == "strict")
    distance_rows = [row for row in metrics["rows"] if row["group"] == "distance"]
    error_lines = []
    for name in ["0-50", "50-100", "100-150", "150+"]:
        if name not in summary:
            continue
        item = summary[name]["error"]
        miss = misses.get(name, {})
        error_lines.append(
            [
                name,
                item["count"],
                _num(item["mean"]),
                _num(item["median"]),
                _num(item["p90"]),
                miss.get("gt", 0),
                _num(miss.get("miss_rate")),
            ]
        )
    contrast_lines = []
    for item in metrics["contrasts"]:
        contrast_lines.append(
            [
                item["family"],
                item["distance"],
                item["left"],
                item["right"],
                item["left_count"],
                item["right_count"],
                _num(item["left_rate"]),
                _num(item["right_rate"]),
                _num(item["difference"]),
                "%s, %s" % (_num(item["ci_low"]), _num(item["ci_high"])),
            ]
        )
    rank_lines = []
    for row in [ranking, strict] + distance_rows:
        rank_lines.append(
            [
                row["group"] + "/" + row["subgroup"],
                row["n_pos"],
                row["n_neg"],
                _num(row["value_geometry"], 4),
                _num(row["value_learned"], 4),
                _num(row["value_fusion"], 4),
                _num(row["ap_geometry"], 4),
                _num(row["ap_learned"], 4),
                _num(row["ap_fusion"], 4),
                row["note"],
            ]
        )
    failure_lines = [
        [row["reason"], row["count"], _num(row["share_of_failures"])]
        for row in failure_rows
        if row["count"]
    ]
    positive = int(label_counts.get("positive", 0))
    negative = int(label_counts.get("negative", 0))
    unknown = int(label_counts.get("unknown", 0))
    pairs = int(label_counts.get("pairs", 0))
    known = positive + negative
    text = []
    text.append("# GRAE 路侧几何关联可靠性")
    text.append("")
    text.append("## 1. 实验目的与设置")
    text.append("")
    text.append("本实验检验 GRAE 的几何距离能否描述路侧目标在不同观测距离、点云密度和局部密集程度下的真实关联可靠性。数据为 V2X-Seq-SPD 路侧验证集，检测为 CenterPoint，跟踪器为已训练的 GRAE-3DMOT。真值只用于离线标签和统计，不进入网络输入、匹配或轨迹更新。")
    text.append("")
    text.append("观测距离取虚拟激光雷达坐标系下目标中心到原点的平面距离。序列内 `virtuallidar_to_world` 最大平移漂移为 %s m，抽查点云半径 P10 为 %s m。%s"
        % (
            _num(sensor["max_within_sequence_translation_drift_m"]),
            _num(sensor["sample_point_radius_p10_m"]),
            "因此原点可作为该路侧雷达的传感器参考位置。" if sensor["use_virtual_lidar_origin"] else "原点核对未通过，距离解释需要保留这一限制。",
        )
    )
    text.append("")
    text.append("网络跨帧特征使用 `sqrt(norm)`，也就是标准欧氏距离再开方。最终匹配使用标准欧氏距离，几何分数为 `exp(-euclidean)`。原始融合分数是 Sigmoid 关联分数与该几何分数的 0.5/0.5 平均，类别不一致时匹配代价再减 1e6。高分检测阈值为 0.24，高分阶段接受融合分数不低于 0.1，低分阶段不低于 0.2。轨迹速度在每帧关联前被置零，预测位置等于上次关联到的检测中心。")
    text.append("")
    text.append("检测与真值沿用官方车辆类别过滤、3D IoU 不低于 0.25 且中心距离不超过 20 m 的一对一匹配。候选正负标签要求检测身份和历史轨迹身份都来自这一匹配的跨帧传递。身份冲突、同一真值对应多条存活轨迹，或任意一端无法确定时记为未知，不记入负样本。严格子集还要求当前匹配 IoU 不低于 0.5、中心误差不超过 2 m，且轨迹身份建立时的 IoU 也不低于 0.5。")
    text.append("")
    text.append("## 2. 核心结果")
    text.append("")
    text.append("可标注候选 %d 对，其中正样本 %d、负样本 %d、未知 %d，可标注比例 %s。严格正样本 %d，严格负样本 %d。"
        % (known, positive, negative, unknown, _num(known / pairs if pairs else None), int(label_counts.get("strict_positive", 0)), int(label_counts.get("strict_negative", 0))))
    text.append("")
    text.append("### 定位误差与漏检")
    text.append("")
    text.append(_markdown_table(["距离段 / m", "匹配样本", "误差均值", "中位数", "P90", "GT 数", "漏检率"], error_lines))
    text.append("")
    text.append("定位误差与距离、点数、置信度的 Spearman 相关分别为 %s、%s、%s。相关 p 值没有按序列聚类校正，只作为描述。"
        % (
            _num(None if spearman["range"] is None else spearman["range"]["rho"]),
            _num(None if spearman["points"] is None else spearman["points"]["rho"]),
            _num(None if spearman["score"] is None else spearman["score"]["rho"]),
        )
    )
    text.append("")
    text.append("这部分误差只来自成功匹配的检测。漏检率按真值距离单独统计，避免只用可见检测判断远距离几何误差。")
    text.append("")
    text.append("### 相同距离下的关联可靠性")
    text.append("")
    if contrast_lines:
        text.append(_markdown_table(["条件", "距离层", "左侧", "右侧", "左样本", "右样本", "左正关联比例", "右正关联比例", "差值", "95% 序列区间"], contrast_lines))
    else:
        text.append("预注册的条件对比没有同时达到两侧至少 %d 个可标注候选，未给出比例差。" % int(config["min_cell_count"]))
    text.append("")
    text.append("### 三种分数的排序能力")
    text.append("")
    text.append(_markdown_table(["子集", "正样本", "负样本", "几何 AUROC", "GRAE AUROC", "融合 AUROC", "几何 AP", "GRAE AP", "融合 AP", "备注"], rank_lines))
    text.append("")
    overlap = [row for row in distance_rows if row["n_pos"] and row["n_neg"] and row["note"] == "样本不足"]
    if overlap:
        text.append("正负样本同时出现、但正样本少于 20 的距离层有 %d 个。这些层是几何距离可能失效的重叠区，当前样本不足以做条件对比。" % len(overlap))
    text.append("")
    text.append("`exp(-distance)` 只作为排序分数，不解释为已经校准的关联概率。本轮没有在训练集上拟合额外校准器。")
    text.append("")
    text.append("## 3. GRAE 的真实问题")
    text.append("")
    text.append("速度在关联前的最大绝对值是 %s。历史位置没有使用运动外推。" % _num(config.get("_velocity_max"), 6))
    text.append("")
    text.append("下面的比例只在互斥主要原因内部归一，原始的大定位误差和大预测误差允许重叠。可分析机会 %d 次，成功 %d 次，失败 %d 次。新出现真值 %d 次，身份歧义 %d 次，轨迹删除后的后续帧 %d 次，这些都不进入失败分母。失败里大定位误差标记 %d 次，大预测误差标记 %d 次。"
        % (
            raw["opportunities"],
            raw["success"],
            raw["failures"],
            raw["new_object"],
            raw["identity_ambiguous"],
            raw["track_absent_continued"],
            raw["large_loc_on_failure"],
            raw["large_pred_on_failure"],
        )
    )
    text.append("")
    text.append(_markdown_table(["主要原因", "次数", "占失败比例"], failure_lines))
    text.append("")
    dominant = max(failure_rows, key=lambda item: item["count"]) if failure_rows else None
    if dominant and dominant["count"]:
        text.append("互斥归因里次数最多的是 `%s`。这不等于该因素单独造成全部误差，只说明按当前规则它是最常见的主因。" % dominant["reason"])
    text.append("")
    text.append("## 4. 对后续研究的启示")
    text.append("")
    text.append("判定规则在看验证集结果之前写定：每个条件对比两侧至少 %d 个候选，正关联比例绝对差至少 %.2f，并且序列重采样 95%% 区间不包含 0。学习分数相对几何分数的 AUROC 增益达到 %.2f 视为排序上已有明显补偿。检测缺失占失败比例达到 %.2f 视为系统主瓶颈。"
        % (int(config["min_cell_count"]), float(config["conditional_gap_min"]), float(config["learned_gain_min"]), float(config["missing_share_max"])))
    text.append("")
    text.append("符合规则的对比有 %d 个，共检查 %d 个。学习分数 AUROC 增益为 %s。检测缺失占失败比例为 %s。"
        % (len(judgment["reliable_contrasts"]), judgment["examined_contrasts"], _num(judgment["auroc_gain"]), _num(judgment["missing_share"])))
    text.append("")
    text.append(judgment["text"])
    text.append("")
    if judgment["missing_share"] >= float(config["missing_share_max"]):
        text.append("检测缺失占比已经达到预注册上限。仅更换关联距离不能恢复没有进入候选池的目标，后续若以召回为主要目标，需要单独研究检测与跟踪的联合优化。")
    else:
        text.append("检测缺失没有占到失败的一半以上，关联阶段本身仍有可分析的误差。")
    text.append("")
    motion_share = next((row["share_of_failures"] or 0.0 for row in failure_rows if row["reason"] == "motion_prediction"), 0.0)
    competition_share = next((row["share_of_failures"] or 0.0 for row in failure_rows if row["reason"] == "association_competition"), 0.0)
    text.append("运动预测主因占比 %s，关联竞争主因占比 %s。零速度使预测位置停留在上次检测，检测差分外推只作为离线误差对照，不能当成新跟踪器已经生效的证据。" % (_num(motion_share), _num(competition_share)))
    text.append("")
    if ablation_rows:
        text.append("固定原始门限和校准序列选门限的跟踪指标见 `ablation_metrics.csv`。门限只在训练划分的 0078、0087、0093、0094 上按 HOTA 选择，验证集没有参与选择。")
    else:
        text.append("本轮输出没有包含受控关联对比。")
    text.append("")
    text.append("## 5. 最终研究判断")
    text.append("")
    text.append("**%s**" % judgment["decision"])
    text.append("")
    text.append("该判断只针对是否把路侧观测不确定性模块作为下一步核心创新。它不证明道路运动约束或双向预测已经有效，也不否定这些方向值得单独做对照实验。")
    if skipped_figures:
        text.append("")
        text.append("以下图因样本不足没有生成：%s。" % "、".join(skipped_figures))
    text.append("")
    (output_dir / "summary.md").write_text("\n".join(text), encoding="utf-8")
    return {
        "known_fraction": known / pairs if pairs else None,
        "auroc_geometry": ranking["value_geometry"],
        "auroc_learned": ranking["value_learned"],
        "auroc_fusion": ranking["value_fusion"],
        "decision": judgment["decision"],
        "reliable_contrasts": len(judgment["reliable_contrasts"]),
        "dominant_failure": None if dominant is None else dominant["reason"],
        "motion_share": motion_share,
        "missing_share": judgment["missing_share"],
        "competition_share": competition_share,
    }


def _track_split(tracker, config, sequence_ids, prediction_root, mode, high_limit, low_limit):
    root = config["_root"]
    converted = Path(config["project"]["converted_root"])
    detection_root = _path(root, config["detection_root"])
    prediction_root.mkdir(parents=True, exist_ok=True)
    tracker.association_mode = mode
    tracker.high_cost_limit = float(high_limit)
    tracker.low_cost_limit = float(low_limit)
    tracker.enable_debug(False)
    for sequence_id in sequence_ids:
        tracker.reset()
        det_rows = list(read_jsonl(detection_root / ("%s.jsonl" % sequence_id)))
        gt_rows = list(read_jsonl(converted / "sequences" / ("%s.jsonl" % sequence_id)))
        if len(det_rows) != len(gt_rows):
            raise ValueError("序列%s帧数不一致" % sequence_id)
        rows = []
        for det_row, gt_row in zip(det_rows, gt_rows):
            if det_row["frame_id"] != gt_row["frame_id"]:
                raise ValueError("序列%s帧未对齐" % sequence_id)
            cleaned = [
                {"class_name": item["class_name"], "score": float(item["score"]), "box": item["box"]}
                for item in det_row["objects"]
            ]
            objects = tracker.update(cleaned, int(det_row["timestamp"]) / 1.0e6, det_row["frame_id"])
            rows.append(_prediction_row(det_row, objects))
        write_jsonl(prediction_root / ("%s.jsonl" % sequence_id), rows)
        print("完成关联方案 %s 序列 %s" % (mode, sequence_id), flush=True)


def _select_gate(tracker, config, mode, score_threshold):
    root = config["_root"]
    calibration_ids = [str(item) for item in config["calibration_sequences"]]
    protocol = V2XSeqProtocol(root)
    best = None
    records = []
    for gate in config["gate_grid"]:
        high_gate = float(gate)
        low_gate = high_gate + float(config["low_gate_offset"])
        if low_gate >= 1.0:
            continue
        prediction_root = Path(config["project"]["output_root"]) / "cache" / "calibration" / mode / ("%0.2f" % high_gate)
        _track_split(tracker, config, calibration_ids, prediction_root, mode, 1.0 - high_gate, 1.0 - low_gate)
        metrics = evaluate_hota(config, prediction_root, calibration_ids, protocol, score_threshold)
        records.append({"mode": mode, "high_gate": high_gate, "low_gate": low_gate, "HOTA": metrics["HOTA"]})
        closer = best is not None and abs(high_gate - 0.1) < abs(best["high_gate"] - 0.1)
        if best is None or metrics["HOTA"] > best["HOTA"] + 1.0e-4 or (abs(metrics["HOTA"] - best["HOTA"]) <= 1.0e-4 and closer):
            best = {"high_gate": high_gate, "low_gate": low_gate, "HOTA": metrics["HOTA"]}
    return best, records


def _official(config, prediction_root, output_dir, sequence_ids, score_threshold):
    evaluator = UnifiedMOTEvaluator(config["_root"], "v2xseq")
    frozen_dir = output_dir / "frozen"
    sweep_dir = output_dir / "sweep"
    frozen = evaluator.evaluate(config, prediction_root, frozen_dir, sequences=sequence_ids, score_threshold=score_threshold)
    hota = evaluate_hota(config, prediction_root, sequence_ids, evaluator.protocol, score_threshold)
    sweep = evaluator.evaluate(config, prediction_root, sweep_dir, sequences=sequence_ids, score_threshold=None)
    frozen.update(hota)
    frozen["AMOTA"] = sweep["AMOTA"]
    frozen["AMOTP"] = sweep["AMOTP"]
    frozen["sweep_best_score_threshold"] = sweep["best_score_threshold"]
    frozen["sweep_MOTA"] = sweep["MOTA"]
    return frozen


def run_ablation(config, tracker, sequence_ids, score_threshold):
    output_dir = Path(config["project"]["output_root"])
    rows = []
    calibration_notes = []
    plans = []
    for mode in ("learned", "geometry", "fusion"):
        plans.append({"mode": mode, "policy": "fixed_gate", "high_gate": 0.1, "low_gate": 0.2, "selection_hota": ""})
        selected, records = _select_gate(tracker, config, mode, score_threshold)
        calibration_notes.extend(records)
        plans.append(
            {
                "mode": mode,
                "policy": "calibrated_gate",
                "high_gate": selected["high_gate"],
                "low_gate": selected["low_gate"],
                "selection_hota": selected["HOTA"],
            }
        )
    write_json(output_dir / "cache" / "gate_selection.json", calibration_notes)
    evaluated = {}
    for plan in plans:
        gate_key = (plan["mode"], round(float(plan["high_gate"]), 5), round(float(plan["low_gate"]), 5))
        if gate_key in evaluated:
            copied = dict(evaluated[gate_key])
            copied["policy"] = plan["policy"]
            copied["selection_hota"] = plan.get("selection_hota", "")
            rows.append(copied)
            print("复用评估 %s %s" % (plan["mode"], plan["policy"]), flush=True)
            continue
        if plan["mode"] == "fusion" and abs(float(plan["high_gate"]) - 0.1) < 1.0e-9:
            prediction_root = _path(config["_root"], config["baseline_prediction_root"])
        else:
            prediction_root = output_dir / "predictions" / plan["mode"] / plan["policy"]
            _track_split(
                tracker,
                config,
                sequence_ids,
                prediction_root,
                plan["mode"],
                1.0 - plan["high_gate"],
                1.0 - plan["low_gate"],
            )
        metrics = _official(
            config,
            prediction_root,
            output_dir / "ablation_eval" / plan["mode"] / plan["policy"],
            sequence_ids,
            score_threshold,
        )
        rows.append(
            {
                "mode": plan["mode"],
                "policy": plan["policy"],
                "high_gate": plan["high_gate"],
                "low_gate": plan["low_gate"],
                "selection_hota": plan.get("selection_hota", ""),
                "MOTA": metrics["MOTA"],
                "AMOTA": metrics["AMOTA"],
                "IDF1": metrics["IDF1"],
                "IDSW": metrics["IDSW"],
                "FP": metrics["FP"],
                "FN": metrics["FN"],
                "HOTA": metrics["HOTA"],
                "DetA": metrics["DetA"],
                "AssA": metrics["AssA"],
                "published_score_threshold": score_threshold,
                "sweep_best_score_threshold": metrics["sweep_best_score_threshold"],
                "sweep_MOTA": metrics["sweep_MOTA"],
            }
        )
        evaluated[gate_key] = rows[-1]
        print("完成评估 %s %s" % (plan["mode"], plan["policy"]), flush=True)
    fields = list(rows[0].keys())
    _write_header(output_dir / "ablation_metrics.csv", fields)
    _append_rows(output_dir / "ablation_metrics.csv", fields, rows)
    return rows


def _git_revision(root):
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        return ""


def run(config):
    root = config["_root"]
    output_dir = Path(config["project"]["output_root"])
    output_dir.mkdir(parents=True, exist_ok=True)
    split = read_json(_path(root, config["split_file"]))
    sequence_ids = [str(item) for item in config.get("sequence_override") or split["val"]]
    allowed = set(split["train"]) | set(split["val"]) | set(split["test"])
    unknown = [item for item in sequence_ids if item not in allowed]
    if unknown:
        raise ValueError("序列不在划分文件中 %s" % ",".join(unknown))
    config["project"]["centerpoint_root"] = _path(root, config["project"]["centerpoint_root"])
    sensor = audit_sensor_origin(config, sequence_ids)
    if not sensor["use_virtual_lidar_origin"]:
        raise RuntimeError("虚拟激光雷达原点未能核对为传感器参考位置")
    tracker, tracker_config, checkpoint = _load_tracker(config)
    protocol = V2XSeqProtocol(root)
    detection_path = output_dir / "detection_error.csv"
    pair_path = output_dir / "association_pairs.csv"
    failure_path = output_dir / "failure_attribution.csv"
    _write_header(detection_path, DETECTION_FIELDS)
    _write_header(pair_path, PAIR_FIELDS)
    _write_header(failure_path, FAILURE_FIELDS)
    store = _empty_store()
    label_counts = Counter()
    velocity_max = 0.0
    for sequence_id in sequence_ids:
        store[sequence_id] = _sequence_store()
        counts, sequence_velocity = diagnose_sequence(
            config,
            tracker,
            protocol,
            sequence_id,
            detection_path,
            pair_path,
            failure_path,
            store,
        )
        label_counts.update(counts)
        velocity_max = max(velocity_max, float(sequence_velocity))
        print(
            "完成诊断 %s 候选 %d 正 %d 负 %d 未知 %d"
            % (sequence_id, counts["pairs"], counts["positive"], counts["negative"], counts["unknown"]),
            flush=True,
        )
    if velocity_max > 1.0e-6:
        raise AssertionError("GRAE 速度不是零")
    metrics = collect_metrics(config, store, label_counts)
    metrics["examples"] = {
        "near": store["near_matched"] or store["near_any"],
        "far": store["far_missed"] or store["far_any"],
    }
    metric_fields = sorted({key for row in metrics["rows"] for key in row})
    _write_header(output_dir / "association_metrics.csv", metric_fields)
    _append_rows(output_dir / "association_metrics.csv", metric_fields, metrics["rows"])
    detection_rows = _load_detection_rows(detection_path)
    failures = _read_failures(failure_path)
    config["_failures"] = failures
    config["_velocity_max"] = velocity_max
    skipped = plot_figures(config, detection_rows, metrics, output_dir)
    plot_failures(config, failures, output_dir)
    ranking = next(row for row in metrics["rows"] if row["subgroup"] == "known" and row["record_type"] == "ranking")
    failure_rows, _ = failure_table(failures)
    judgment = decide(config, metrics["contrasts"], ranking, None, failure_rows)
    published = read_json(_path(root, config["baseline_metrics"]))
    ablation_rows = []
    if config.get("run_ablation", True) and not config.get("sequence_override"):
        ablation_rows = run_ablation(config, tracker, sequence_ids, float(published["best_score_threshold"]))
    elif not (output_dir / "ablation_metrics.csv").exists():
        _write_header(output_dir / "ablation_metrics.csv", ("mode", "policy", "note"))
        _append_rows(output_dir / "ablation_metrics.csv", ("mode", "policy", "note"), [{"mode": "", "policy": "", "note": "未运行受控对比"}])
    brief = write_summary(
        config,
        output_dir,
        sensor,
        detection_rows,
        metrics,
        label_counts,
        judgment,
        ablation_rows,
        skipped,
    )
    payload = {
        "seed": int(config["seed"]),
        "checkpoint": str(checkpoint),
        "split_file": str(config["split_file"]),
        "sequences": sequence_ids,
        "calibration_sequences": [str(item) for item in config["calibration_sequences"]],
        "git_revision": _git_revision(root),
        "tracker": {
            "association_alpha": tracker_config["tracker"]["association_alpha"],
            "score_floor": tracker_config["tracker"].get("score_floor", 0.1),
            "age": tracker_config["tracker"]["age"],
            "high_cost_limit": 0.9,
            "low_cost_limit": 0.8,
            "association_mode_for_diagnosis": "fusion",
        },
        "distance_definition": {
            "matching": "planar euclidean hypot(dx, dy)",
            "network_feature": "sqrt(euclidean)",
            "geometry_score": "exp(-euclidean)",
            "fusion": "0.5 * sigmoid(logit) + 0.5 * exp(-euclidean)",
        },
        "sensor": sensor,
        "match": {
            "iou": config["match_iou"],
            "center_gate_m": config["match_center_gate_m"],
            "strict_iou": config["strict_iou"],
            "strict_center_m": config["strict_center_m"],
        },
        "decision_rule": {
            "min_cell_count": config["min_cell_count"],
            "conditional_gap_min": config["conditional_gap_min"],
            "learned_gain_min": config["learned_gain_min"],
            "missing_share_max": config["missing_share_max"],
            "bootstrap_samples": config["bootstrap_samples"],
        },
        "result_brief": brief,
        "prediction_matches_published_grae": True,
    }
    write_json(output_dir / "experiment_config.json", payload)
    print("几何 AUROC %s，GRAE AUROC %s，融合 AUROC %s" % (_num(brief["auroc_geometry"]), _num(brief["auroc_learned"]), _num(brief["auroc_fusion"])))
    print("可靠条件对比 %d 个，判断 %s" % (brief["reliable_contrasts"], brief["decision"]))
    print("主要失败 %s，检测缺失占比 %s，运动预测占比 %s" % (brief["dominant_failure"], _num(brief["missing_share"]), _num(brief["motion_share"])))
    return brief
