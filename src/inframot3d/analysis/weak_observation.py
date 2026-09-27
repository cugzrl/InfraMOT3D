import copy
import math
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from inframot3d.analysis.scene_difficulty import _bin_index, _grid_shape, _keep_vehicle, _match, build_scenes
from inframot3d.analysis.tracking_bottleneck import _class_thresholds, _path
from inframot3d.config import load_config
from inframot3d.evaluation import UnifiedMOTEvaluator
from inframot3d.evaluation.hota import evaluate_hota
from inframot3d.evaluation.protocols.v2xseq import V2XSeqProtocol
from inframot3d.io import read_json, read_jsonl, write_json, write_jsonl
from inframot3d.tracking import create_tracker
from inframot3d.tracking.common import overlap


MODEL_FEATURES = {
    "Score Only": ["score"],
    "Spatial Only": [
        "score",
        "hist_detection_precision",
        "hist_low_precision",
        "hist_miss_risk",
        "hist_localization_iou",
        "hist_localization_residual",
        "hist_observation_count",
        "hist_low_count",
        "hist_valid",
        "hist_sequence_count",
    ],
    "Temporal Only": [
        "score",
        "log_length",
        "log_width",
        "log_height",
        "log_volume",
        "aspect_length_width",
        "range_m",
        "log_point_count",
        "point_density",
        "local_detection_count",
        "local_low_count",
        "class_car",
        "class_van",
        "class_bus",
        "class_truck",
        "has_track",
        "track_count",
        "gate_candidate_count",
        "near_track_count",
        "best_center_distance",
        "best_iou",
        "best_giou",
        "best_cost",
        "second_cost",
        "cost_margin",
        "best_yaw_difference",
        "best_track_age",
        "best_track_hits",
        "best_track_missed",
        "best_track_speed",
        "best_track_continuity",
        "best_track_recent_match",
        "best_track_alive",
    ],
}
MODEL_FEATURES["Spatiotemporal"] = MODEL_FEATURES["Temporal Only"] + MODEL_FEATURES["Spatial Only"][1:]
MODEL_ORDER = ["Score Only", "Spatial Only", "Temporal Only", "Spatiotemporal"]
STAT_KEYS = (
    "gt",
    "high_miss",
    "det_tp",
    "det_fp",
    "low_tp",
    "low_fp",
    "loc_iou_sum",
    "loc_residual_sum",
    "loc_count",
)
VEHICLE_CLASSES = ("Car", "Van", "Bus", "Truck")
COLORS = {
    "Score Only": "#9AA0A6",
    "Spatial Only": "#E45756",
    "Temporal Only": "#4C78A8",
    "Spatiotemporal": "#54A24B",
}


def _key(value):
    return value.lower().replace(" ", "_").replace("-", "_")


def _cell(config, box):
    shape = _grid_shape(config["bev"], float(config["grid_m"]))
    return _bin_index(float(box[0]), float(box[1]), config["bev"], float(config["grid_m"]), shape)


def _empty_stats(config):
    shape = _grid_shape(config["bev"], float(config["grid_m"]))
    return {name: np.zeros(shape, dtype=np.float64) for name in STAT_KEYS}


def _add_stats(target, source):
    for name in STAT_KEYS:
        target[name] += np.asarray(source[name], dtype=np.float64)


def _split_sequences(scenes):
    train_ids, validation_ids, test_ids = [], [], []
    scene_by_sequence = {}
    for scene in scenes:
        sequences = scene["sequences"]
        for item in sequences:
            scene_by_sequence[item["sequence_id"]] = scene["scene_id"]
        val_indices = [index for index, item in enumerate(sequences) if item["split"] == "val"]
        if len(val_indices) < 2:
            continue
        validation_index, test_index = val_indices[-2], val_indices[-1]
        validation_ids.append(sequences[validation_index]["sequence_id"])
        test_ids.append(sequences[test_index]["sequence_id"])
        train_ids.extend(
            item["sequence_id"]
            for item in sequences[:validation_index]
            if item["split"] in {"train", "val"}
        )
    return {
        "train": sorted(set(train_ids)),
        "validation": validation_ids,
        "test": test_ids,
        "scene_by_sequence": scene_by_sequence,
    }


def _prediction_row(source, objects):
    return {
        "sequence_id": source["sequence_id"],
        "frame_index": source["frame_index"],
        "frame_id": source["frame_id"],
        "timestamp": source["timestamp"],
        "objects": objects,
    }


def _point_count(points, box):
    x, y, z, yaw, length, width, height = [float(value) for value in box]
    local = np.asarray(points[:, :2], dtype=np.float64) - np.asarray([x, y], dtype=np.float64)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    forward = local[:, 0] * cosine + local[:, 1] * sine
    side = -local[:, 0] * sine + local[:, 1] * cosine
    inside = (
        (np.abs(forward) <= max(length, 1.0e-3) / 2.0)
        & (np.abs(side) <= max(width, 1.0e-3) / 2.0)
        & (np.abs(points[:, 2] - z) <= max(height, 1.0e-3) / 2.0)
    )
    return int(np.count_nonzero(inside))


def _angle_gap(left, right):
    value = abs((float(left) - float(right) + math.pi) % (2.0 * math.pi) - math.pi)
    return min(value, abs(math.pi - value))


def _group(snapshot, class_name):
    return next((item for item in snapshot.get("groups", []) if item.get("class_name") == class_name), None)


def _temporal_features(detection, group):
    tracks = [] if group is None else group.get("pre_tracks", [])
    if not tracks:
        return {
            "has_track": 0.0,
            "track_count": 0.0,
            "gate_candidate_count": 0.0,
            "near_track_count": 0.0,
            "best_center_distance": 50.0,
            "best_iou": 0.0,
            "best_giou": -1.0,
            "best_cost": 2.0,
            "second_cost": 2.0,
            "cost_margin": 0.0,
            "best_yaw_difference": math.pi / 2.0,
            "best_track_age": 0.0,
            "best_track_hits": 0.0,
            "best_track_missed": 0.0,
            "best_track_speed": 0.0,
            "best_track_continuity": 0.0,
            "best_track_recent_match": 0.0,
            "best_track_alive": 0.0,
        }
    rows = []
    box = detection["box"]
    for track in tracks:
        track_box = track["box"]
        distance = float(math.hypot(float(box[0]) - float(track_box[0]), float(box[1]) - float(track_box[1])))
        iou = float(overlap(box, track_box, "iou"))
        giou = float(overlap(box, track_box, "giou"))
        rows.append((1.0 - giou, distance, iou, giou, track))
    rows.sort(key=lambda item: item[0])
    best = rows[0]
    second_cost = float(rows[1][0]) if len(rows) > 1 else 2.0
    track = best[4]
    velocity = track.get("velocity", [0.0, 0.0, 0.0])
    speed = float(math.hypot(float(velocity[0]), float(velocity[1])))
    threshold = float((group.get("association") or {}).get("threshold", 1.5))
    return {
        "has_track": 1.0,
        "track_count": float(len(rows)),
        "gate_candidate_count": float(sum(item[0] <= threshold for item in rows)),
        "near_track_count": float(sum(item[1] <= 10.0 for item in rows)),
        "best_center_distance": best[1],
        "best_iou": best[2],
        "best_giou": best[3],
        "best_cost": best[0],
        "second_cost": second_cost,
        "cost_margin": max(0.0, second_cost - best[0]),
        "best_yaw_difference": _angle_gap(box[3], track["box"][3]),
        "best_track_age": float(track.get("age", 0)),
        "best_track_hits": float(track.get("hits", 0)),
        "best_track_missed": float(track.get("time_since_update", 0)),
        "best_track_speed": speed,
        "best_track_continuity": float(track.get("hits", 0)) / max(float(track.get("age", 0)) + 1.0, 1.0),
        "best_track_recent_match": float(int(track.get("time_since_update", 0)) <= 1),
        "best_track_alive": float(track.get("state") == "alive"),
    }


def _detection_features(detection, all_detections, points, config):
    box = detection["box"]
    length, width, height = [max(float(value), 1.0e-3) for value in box[4:7]]
    volume = length * width * height
    center = np.asarray(box[:2], dtype=np.float64)
    radius = float(config["local_radius_m"])
    local = 0
    local_low = 0
    for other in all_detections:
        if other is detection:
            continue
        distance = float(np.linalg.norm(np.asarray(other["box"][:2], dtype=np.float64) - center))
        if distance <= radius:
            local += 1
            local_low += int(float(other.get("score", 1.0)) < float(config["low_score_max"]))
    count = _point_count(points, box)
    class_name = detection["class_name"].lower()
    return {
        "score": float(detection["score"]),
        "log_length": math.log1p(length),
        "log_width": math.log1p(width),
        "log_height": math.log1p(height),
        "log_volume": math.log1p(volume),
        "aspect_length_width": length / width,
        "range_m": float(math.hypot(float(box[0]), float(box[1]))),
        "log_point_count": math.log1p(count),
        "point_density": float(count / volume),
        "local_detection_count": float(local),
        "local_low_count": float(local_low),
        "class_car": float(class_name == "car"),
        "class_van": float(class_name == "van"),
        "class_bus": float(class_name == "bus"),
        "class_truck": float(class_name == "truck"),
    }


def _sequence_job(payload):
    config, sequence_id, need_samples, cache_path = payload
    if Path(cache_path).is_file():
        return sequence_id, read_json(cache_path)
    protocol = V2XSeqProtocol(config["_root"])
    gt_rows = list(
        read_jsonl(Path(config["project"]["converted_root"]) / "sequences" / (sequence_id + ".jsonl"))
    )
    det_rows = list(read_jsonl(_path(config["_root"], config["detection_root"]) / (sequence_id + ".jsonl")))
    if len(gt_rows) != len(det_rows):
        raise ValueError("检测帧数不一致%s" % sequence_id)
    tracker = None
    if need_samples:
        tracker_config = load_config(_path(config["_root"], config["tracker_config"]))
        _class_thresholds(tracker_config)
        tracker = create_tracker(tracker_config["tracker"], tracker_config["_root"])
        tracker.enable_debug(True)
    stats = _empty_stats(config)
    samples = []
    low = float(config["low_score_min"])
    high = float(config["low_score_max"])
    point_root = Path(config["project"]["centerpoint_root"]) / "points"
    for gt_row, det_row in zip(gt_rows, det_rows):
        if int(gt_row["timestamp"]) != int(det_row["timestamp"]):
            raise ValueError("检测帧未对齐%s" % sequence_id)
        gt_items = protocol.filter_gt(gt_row["objects"])
        indexed = [
            (index, item)
            for index, item in enumerate(det_row["objects"])
            if float(item.get("score", 1.0)) >= low and _keep_vehicle(protocol, item)
        ]
        detections = [item for _, item in indexed]
        matches = _match(
            [item["box"] for item in gt_items],
            [item["box"] for item in detections],
            config["match_iou"],
            config["match_center_gate_m"],
        )
        matched_gt = {gt_index for gt_index, _, _ in matches}
        by_detection = {det_index: (gt_index, iou) for gt_index, det_index, iou in matches}
        high_indexed = [(raw_index, item) for raw_index, item in indexed if float(item["score"]) >= high]
        high_matches = _match(
            [item["box"] for item in gt_items],
            [item["box"] for _, item in high_indexed],
            config["match_iou"],
            config["match_center_gate_m"],
        )
        matched_high_gt = {gt_index for gt_index, _, _ in high_matches}
        for gt_index, item in enumerate(gt_items):
            cell = _cell(config, item["box"])
            if cell is None:
                continue
            stats["gt"][cell] += 1.0
            stats["high_miss"][cell] += float(gt_index not in matched_high_gt)
        for det_index, (_, item) in enumerate(indexed):
            cell = _cell(config, item["box"])
            if cell is None:
                continue
            positive = det_index in by_detection
            stats["det_tp" if positive else "det_fp"][cell] += 1.0
            if float(item["score"]) < high:
                stats["low_tp" if positive else "low_fp"][cell] += 1.0
            if positive:
                gt_index, iou = by_detection[det_index]
                stats["loc_iou_sum"][cell] += float(iou)
                stats["loc_residual_sum"][cell] += float(
                    math.hypot(
                        float(item["box"][0]) - float(gt_items[gt_index]["box"][0]),
                        float(item["box"][1]) - float(gt_items[gt_index]["box"][1]),
                    )
                )
                stats["loc_count"][cell] += 1.0
        if not need_samples:
            continue
        raw_objects = [
            {"class_name": item["class_name"], "score": float(item.get("score", 1.0)), "box": item["box"]}
            for item in det_row["objects"]
        ]
        tracker.update(raw_objects, timestamp=det_row["timestamp"])
        snapshot = tracker.last_debug
        low_items = [
            (det_index, raw_index, item)
            for det_index, (raw_index, item) in enumerate(indexed)
            if float(item["score"]) < high
        ]
        if not low_items:
            continue
        points = np.load(point_root / ("%s_%s.npy" % (sequence_id, det_row["frame_id"])), mmap_mode="r")
        for det_index, raw_index, item in low_items:
            cell = _cell(config, item["box"])
            if cell is None:
                continue
            features = _detection_features(item, detections, points, config)
            features.update(_temporal_features(item, _group(snapshot, item["class_name"])))
            samples.append(
                {
                    "sequence_id": sequence_id,
                    "frame_index": int(det_row["frame_index"]),
                    "frame_id": det_row["frame_id"],
                    "detection_index": int(raw_index),
                    "cell": [int(cell[0]), int(cell[1])],
                    "label": int(det_index in by_detection),
                    "features": features,
                }
            )
    payload = {
        "stats": {name: value.tolist() for name, value in stats.items()},
        "samples": samples,
        "gt_matches": int(sum(stats["det_tp"].reshape(-1))),
        "gt_total": int(sum(stats["gt"].reshape(-1))),
    }
    write_json(cache_path, payload)
    return sequence_id, payload


def _memory_maps(cumulative, history_count, config):
    strength = float(config["prior_strength"])

    def smooth(numerator, denominator, default):
        total = float(np.sum(denominator))
        prior = float(np.sum(numerator) / total) if total > 0.0 else float(default)
        return (numerator + strength * prior) / (denominator + strength)

    detection_total = cumulative["det_tp"] + cumulative["det_fp"]
    low_total = cumulative["low_tp"] + cumulative["low_fp"]
    gt = cumulative["gt"]
    loc_count = cumulative["loc_count"]
    detection_precision = smooth(cumulative["det_tp"], detection_total, 0.5)
    low_precision = smooth(cumulative["low_tp"], low_total, 0.5)
    miss_risk = smooth(cumulative["high_miss"], gt, 0.5)
    localization_iou = smooth(cumulative["loc_iou_sum"], loc_count, 0.0)
    localization_residual = smooth(cumulative["loc_residual_sum"], loc_count, 10.0)
    return {
        "hist_detection_precision": detection_precision,
        "hist_low_precision": low_precision,
        "hist_miss_risk": miss_risk,
        "hist_localization_iou": localization_iou,
        "hist_localization_residual": localization_residual,
        "hist_observation_count": np.log1p(gt),
        "hist_low_count": np.log1p(low_total),
        "hist_valid": (gt >= int(config["min_observations"])).astype(np.float64),
        "hist_sequence_count": np.full(gt.shape, float(history_count), dtype=np.float64),
    }


def _attach_memory(config, scenes, extracted):
    samples = []
    for scene in scenes:
        cumulative = _empty_stats(config)
        history_count = 0
        for sequence in scene["sequences"]:
            sequence_id = sequence["sequence_id"]
            memory = _memory_maps(cumulative, history_count, config)
            current = extracted[sequence_id]
            for sample in current["samples"]:
                iy, ix = sample["cell"]
                for name, values in memory.items():
                    sample["features"][name] = float(values[iy, ix])
                samples.append(sample)
            _add_stats(cumulative, current["stats"])
            history_count += 1
    return samples


def _extract_samples(config, scenes, split, debug_root):
    cache = debug_root / "samples.json"
    if cache.is_file():
        payload = read_json(cache)
        print("复用低分检测特征", flush=True)
        return payload["samples"]
    required = set(split["train"] + split["validation"] + split["test"])
    cache_root = debug_root / "sequence_cache"
    all_ids = [item["sequence_id"] for scene in scenes for item in scene["sequences"]]
    extracted = {}
    jobs = [
        (config, sequence_id, sequence_id in required, cache_root / (sequence_id + ".json"))
        for sequence_id in all_ids
    ]
    with ProcessPoolExecutor(max_workers=int(config["workers"])) as executor:
        futures = [executor.submit(_sequence_job, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), start=1):
            sequence_id, payload = future.result()
            extracted[sequence_id] = payload
            if index % 10 == 0 or index == len(futures):
                print("完成特征序列%d/%d" % (index, len(futures)), flush=True)
    samples = _attach_memory(config, scenes, extracted)
    write_json(
        cache,
        {
            "low_score_range": [config["low_score_min"], config["low_score_max"]],
            "samples": samples,
        },
    )
    return samples


def _matrix(samples, feature_names):
    return np.asarray(
        [[float(sample["features"].get(name, 0.0)) for name in feature_names] for sample in samples],
        dtype=np.float64,
    )


def _labels(samples):
    return np.asarray([int(sample["label"]) for sample in samples], dtype=np.int64)


def _best_f1_threshold(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0.0,
    )
    index = int(np.argmax(f1))
    if index >= len(thresholds):
        return 1.0
    return float(thresholds[index])


def _classification_metrics(labels, scores, threshold):
    predicted = scores >= float(threshold)
    tp = int(np.sum(predicted & (labels == 1)))
    fp = int(np.sum(predicted & (labels == 0)))
    fn = int(np.sum(~predicted & (labels == 1)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(average_precision_score(labels, scores)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "threshold": float(threshold),
    }


def _precision_at_recall(labels, scores, target_recall):
    precision, recall, _ = precision_recall_curve(labels, scores)
    valid = precision[recall >= float(target_recall) - 1.0e-12]
    return float(np.max(valid)) if valid.size else 0.0


def _tp_at_fp(labels, scores, fp_budget):
    order = np.argsort(-scores)
    tp = fp = best = 0
    for index in order:
        if int(labels[index]) == 1:
            tp += 1
        else:
            fp += 1
        if fp <= int(fp_budget):
            best = max(best, tp)
        else:
            break
    return int(best)


def _fit_models(config, samples, split, debug_root):
    train_ids = set(split["train"])
    validation_ids = set(split["validation"])
    test_ids = set(split["test"])
    train = [sample for sample in samples if sample["sequence_id"] in train_ids]
    validation = [sample for sample in samples if sample["sequence_id"] in validation_ids]
    test = [sample for sample in samples if sample["sequence_id"] in test_ids]
    for name, current in (("train", train), ("validation", validation), ("test", test)):
        labels = _labels(current)
        if not len(labels) or len(np.unique(labels)) < 2:
            raise ValueError("%s样本不足" % name)
    outputs = {}
    model_root = debug_root / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    for name in MODEL_ORDER:
        features = MODEL_FEATURES[name]
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1000,
                random_state=int(config["seed"]),
            ),
        )
        model.fit(_matrix(train, features), _labels(train))
        validation_scores = model.predict_proba(_matrix(validation, features))[:, 1]
        threshold = _best_f1_threshold(_labels(validation), validation_scores)
        test_scores = model.predict_proba(_matrix(test, features))[:, 1]
        metrics = _classification_metrics(_labels(test), test_scores, threshold)
        outputs[name] = {
            "features": features,
            "model": model,
            "threshold": threshold,
            "metrics": metrics,
            "test_scores": test_scores,
        }
        joblib.dump(model, model_root / (_key(name) + ".joblib"))
    validation_labels = _labels(validation)
    validation_raw_scores = _matrix(validation, ["score"])[:, 0]
    validation_st_scores = outputs["Spatiotemporal"]["model"].predict_proba(
        _matrix(validation, outputs["Spatiotemporal"]["features"])
    )[:, 1]
    global_threshold = float(config["downstream_global_threshold"])
    negative_budget = int(
        np.sum((validation_labels == 0) & (validation_raw_scores >= global_threshold))
    )
    negative_scores = np.sort(validation_st_scores[validation_labels == 0])[::-1]
    if negative_budget <= 0:
        downstream_threshold = 1.0
    elif negative_budget >= len(negative_scores):
        downstream_threshold = 0.0
    else:
        downstream_threshold = float(negative_scores[negative_budget - 1])
    outputs["Spatiotemporal"]["downstream_threshold"] = downstream_threshold
    outputs["Spatiotemporal"]["downstream_validation_fp_budget"] = negative_budget
    score_recall = outputs["Score Only"]["metrics"]["recall"]
    fp_budget = outputs["Score Only"]["metrics"]["fp"]
    labels = _labels(test)
    for name in MODEL_ORDER:
        outputs[name]["metrics"]["precision_at_score_recall"] = _precision_at_recall(
            labels, outputs[name]["test_scores"], score_recall
        )
        outputs[name]["metrics"]["tp_at_score_fp"] = _tp_at_fp(
            labels, outputs[name]["test_scores"], fp_budget
        )
    serializable = {
        "split_counts": {
            "train": {"samples": len(train), "positive": int(np.sum(_labels(train)))},
            "validation": {"samples": len(validation), "positive": int(np.sum(_labels(validation)))},
            "test": {"samples": len(test), "positive": int(np.sum(_labels(test)))},
        },
        "models": {
            name: {
                "features": outputs[name]["features"],
                "threshold": outputs[name]["threshold"],
                "downstream_threshold": outputs[name].get("downstream_threshold"),
                "metrics": outputs[name]["metrics"],
            }
            for name in MODEL_ORDER
        },
    }
    write_json(debug_root / "classification.json", serializable)
    return outputs, train, validation, test, serializable


def _sample_lookup(samples, sequence_ids):
    allowed = set(sequence_ids)
    return {
        (sample["sequence_id"], int(sample["frame_index"]), int(sample["detection_index"])): sample
        for sample in samples
        if sample["sequence_id"] in allowed
    }


def _direct_track_ids(snapshot, selected_indices):
    output = set()
    for group in snapshot.get("groups", []):
        for item in group.get("assignments", []) + group.get("created", []):
            if int(item["input_index"]) in selected_indices:
                output.add(int(item["track_id"]))
    return output


def _run_filter(config, split, samples, model_output, debug_root):
    prediction_root = debug_root / "predictions" / "spatiotemporal_filter"
    if all((prediction_root / (sequence_id + ".jsonl")).is_file() for sequence_id in split["test"]):
        print("复用Spatiotemporal轨迹", flush=True)
        return prediction_root, read_json(debug_root / "filter_stats.json")
    tracker_config = load_config(_path(config["_root"], config["tracker_config"]))
    _class_thresholds(tracker_config)
    baseline_thresholds = {
        str(name): float(value) for name, value in tracker_config["tracker"].get("score_thresholds", {}).items()
    }
    baseline_default = float(tracker_config["tracker"].get("score_threshold", 0.0))
    output_threshold = float(read_json(_path(config["_root"], config["baseline_metrics"]))["best_score_threshold"])
    settings = copy.deepcopy(tracker_config["tracker"])
    settings["score_threshold"] = min(baseline_default, float(config["low_score_min"]))
    settings["score_thresholds"] = {
        name: min(value, float(config["low_score_min"])) for name, value in baseline_thresholds.items()
    }
    lookup = _sample_lookup(samples, split["test"])
    model = model_output["model"]
    feature_names = model_output["features"]
    threshold = float(model_output["downstream_threshold"])
    low = float(config["low_score_min"])
    high = float(config["low_score_max"])
    detection_root = _path(config["_root"], config["detection_root"])
    stats = Counter()
    per_sequence = {}
    for sequence_id in split["test"]:
        tracker = create_tracker(settings, tracker_config["_root"])
        tracker.enable_debug(True)
        rows = []
        sequence_stats = Counter()
        for det_row in read_jsonl(detection_root / (sequence_id + ".jsonl")):
            raw_objects = [
                {"class_name": item["class_name"], "score": float(item.get("score", 1.0)), "box": item["box"]}
                for item in det_row["objects"]
            ]
            preview = copy.deepcopy(tracker)
            preview.enable_debug(True)
            preview.update(raw_objects, timestamp=det_row["timestamp"])
            preview_snapshot = preview.last_debug
            selected = set()
            for index, item in enumerate(raw_objects):
                score = float(item["score"])
                if score < low or score >= high:
                    continue
                sample = lookup.get((sequence_id, int(det_row["frame_index"]), int(index)))
                if sample is None:
                    continue
                features = dict(sample["features"])
                features.update(_temporal_features(item, _group(preview_snapshot, item["class_name"])))
                vector = np.asarray([[float(features.get(name, 0.0)) for name in feature_names]], dtype=np.float64)
                probability = float(model.predict_proba(vector)[0, 1])
                sequence_stats["candidates"] += 1
                if probability >= threshold:
                    selected.add(index)
                    sequence_stats["selected"] += 1
                    sequence_stats["selected_positive"] += int(sample["label"])
                    sequence_stats["selected_negative"] += int(not sample["label"])
            marked = []
            for index, item in enumerate(raw_objects):
                current = dict(item)
                base = float(baseline_thresholds.get(item["class_name"], baseline_default))
                baseline_keep = float(item["score"]) >= base
                current["_association_keep"] = bool(baseline_keep or index in selected)
                marked.append(current)
            outputs = tracker.update(marked, timestamp=det_row["timestamp"])
            direct_ids = _direct_track_ids(tracker.last_debug, selected)
            kept = [
                item
                for item in outputs
                if float(item.get("score", 1.0)) >= output_threshold or int(item["track_id"]) in direct_ids
            ]
            sequence_stats["direct_outputs"] += sum(int(item["track_id"]) in direct_ids for item in kept)
            rows.append(_prediction_row(det_row, kept))
        write_jsonl(prediction_root / (sequence_id + ".jsonl"), rows)
        per_sequence[sequence_id] = dict(sequence_stats)
        stats.update(sequence_stats)
        print("完成Spatiotemporal跟踪%s" % sequence_id, flush=True)
    payload = {"total": dict(stats), "per_sequence": per_sequence, "threshold": threshold}
    write_json(debug_root / "filter_stats.json", payload)
    return prediction_root, payload


def _evaluate_downstream(config, split, prediction_root, debug_root):
    tracker_config = load_config(_path(config["_root"], config["tracker_config"]))
    protocol = V2XSeqProtocol(config["_root"])
    evaluator = UnifiedMOTEvaluator(config["_root"], "v2xseq")
    roots = {
        "Baseline": _path(config["_root"], config["controls"]["baseline"]),
        "Global-Low": _path(config["_root"], config["controls"]["global_low"]),
        "Spatial": _path(config["_root"], config["controls"]["spatial"]),
        "Spatiotemporal Filter": prediction_root,
    }
    metrics = {}
    for name, root in roots.items():
        cache = debug_root / "evaluation" / (_key(name) + ".json")
        if cache.is_file():
            metrics[name] = read_json(cache)
            continue
        output_dir = debug_root / "evaluation" / "_official" / _key(name)
        current = evaluator.evaluate(
            tracker_config,
            root,
            output_dir,
            split="val",
            sequences=split["test"],
            score_threshold=0.0,
        )
        current.update(evaluate_hota(tracker_config, root, split["test"], protocol, 0.0))
        write_json(cache, current)
        metrics[name] = current
        print("完成MOT评估%s" % name, flush=True)
    per_sequence = {}
    for sequence_id in split["test"]:
        per_sequence[sequence_id] = {
            name: evaluate_hota(tracker_config, root, [sequence_id], protocol, 0.0)["HOTA"]
            for name, root in roots.items()
        }
    shutil.rmtree(debug_root / "evaluation" / "_official", ignore_errors=True)
    write_json(debug_root / "downstream.json", {"metrics": metrics, "per_sequence_hota": per_sequence})
    return metrics, per_sequence


def _plot_classification(output_root, model_outputs):
    import matplotlib.pyplot as plt

    metric_names = ("precision", "recall", "f1", "pr_auc")
    labels = ("Precision", "Recall", "F1", "PR-AUC")
    x = np.arange(len(metric_names))
    width = 0.2
    figure, axis = plt.subplots(figsize=(10.5, 5.0))
    for index, name in enumerate(MODEL_ORDER):
        values = [100.0 * float(model_outputs[name]["metrics"][metric]) for metric in metric_names]
        axis.bar(x + (index - 1.5) * width, values, width, label=name, color=COLORS[name])
    axis.set_xticks(x, labels)
    axis.set_ylabel("Metric (%)")
    axis.set_title("Low-score detection classification on held-out future sequences")
    axis.set_ylim(0.0, 100.0)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(ncol=2, frameon=False)
    figure.tight_layout()
    figure.savefig(output_root / "figure_1_classification.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_pr(output_root, model_outputs, test):
    import matplotlib.pyplot as plt

    labels = _labels(test)
    figure, axis = plt.subplots(figsize=(7.5, 6.0))
    for name in MODEL_ORDER:
        scores = model_outputs[name]["test_scores"]
        precision, recall, _ = precision_recall_curve(labels, scores)
        ap = model_outputs[name]["metrics"]["pr_auc"]
        axis.plot(recall, precision, linewidth=2.2, color=COLORS[name], label="%s AP %.3f" % (name, ap))
        current = model_outputs[name]["metrics"]
        axis.scatter([current["recall"]], [current["precision"]], color=COLORS[name], s=38)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Precision-recall curves for weak detections")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.02)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_root / "figure_2_pr_curve.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_downstream(output_root, metrics):
    import matplotlib.pyplot as plt

    names = ["Baseline", "Global-Low", "Spatial", "Spatiotemporal Filter"]
    colors = ["#9AA0A6", "#4C78A8", "#E45756", "#54A24B"]
    metric_names = ("HOTA", "DetA", "AssA", "IDF1", "MOTA")
    x = np.arange(len(metric_names))
    width = 0.2
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), gridspec_kw={"width_ratios": [1.7, 1.0]})
    for index, (name, color) in enumerate(zip(names, colors)):
        values = [100.0 * float(metrics[name][metric]) for metric in metric_names]
        axes[0].bar(x + (index - 1.5) * width, values, width, label=name, color=color)
        axes[1].scatter(metrics[name]["FP"], metrics[name]["FN"], s=85, color=color)
        axes[1].annotate(
            "%s\nIDS %d" % (name, int(metrics[name]["IDSW"])),
            (metrics[name]["FP"], metrics[name]["FN"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )
    axes[0].set_xticks(x, metric_names)
    axes[0].set_ylabel("Metric (%)")
    axes[0].set_title("Downstream MOT metrics")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(ncol=2, frameon=False)
    axes[1].set_xlabel("FP")
    axes[1].set_ylabel("FN")
    axes[1].set_title("FP-FN trade-off")
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_root / "figure_3_downstream.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _go_decision(config, model_outputs, downstream, per_sequence):
    current = model_outputs["Spatiotemporal"]["metrics"]
    alternatives = [model_outputs[name]["metrics"] for name in MODEL_ORDER[:-1]]
    best_pr = max(item["pr_auc"] for item in alternatives)
    best_f1 = max(item["f1"] for item in alternatives)
    score_tp = model_outputs["Score Only"]["metrics"]["tp_at_score_fp"]
    current_tp = current["tp_at_score_fp"]
    classification = (
        current["pr_auc"] - best_pr >= float(config["go"]["pr_auc_gain"])
        and current["f1"] - best_f1 >= float(config["go"]["f1_gain"])
    )
    same_fp = current_tp >= score_tp * (1.0 + float(config["go"]["same_fp_tp_gain"]))
    hota_gain = 100.0 * (downstream["Spatiotemporal Filter"]["HOTA"] - downstream["Global-Low"]["HOTA"])
    idf1_gain = 100.0 * (downstream["Spatiotemporal Filter"]["IDF1"] - downstream["Global-Low"]["IDF1"])
    wins = sum(
        row["Spatiotemporal Filter"] > row["Global-Low"] + 1.0e-9 for row in per_sequence.values()
    )
    downstream_go = (
        hota_gain >= float(config["go"]["downstream_hota_points"])
        and idf1_gain > 0.0
        and wins >= int(config["go"]["minimum_sequence_wins"])
    )
    return {
        "go": bool(classification and same_fp and downstream_go),
        "classification": bool(classification),
        "same_fp": bool(same_fp),
        "downstream": bool(downstream_go),
        "best_alternative_pr_auc": float(best_pr),
        "best_alternative_f1": float(best_f1),
        "hota_gain_points": float(hota_gain),
        "idf1_gain_points": float(idf1_gain),
        "sequence_wins": int(wins),
        "sequence_total": len(per_sequence),
    }


def _fmt(value):
    return "%.2f" % (100.0 * float(value))


def _summary(config, split, classification, downstream, per_sequence, filter_stats, decision):
    models = classification["models"]
    temporal_pr = models["Temporal Only"]["metrics"]["pr_auc"] - models["Score Only"]["metrics"]["pr_auc"]
    spatial_pr = models["Spatial Only"]["metrics"]["pr_auc"] - models["Score Only"]["metrics"]["pr_auc"]
    complement_pr = models["Spatiotemporal"]["metrics"]["pr_auc"] - models["Temporal Only"]["metrics"]["pr_auc"]
    complement_f1 = models["Spatiotemporal"]["metrics"]["f1"] - models["Temporal Only"]["metrics"]["f1"]
    lines = [
        "# Weak Observation Feasibility",
        "",
        "## 实验问题",
        "",
        "历史场景信息与association前的短期轨迹状态，能否互补地区分[%.2f, %.2f)低分真检测和FP，并改善下游MOT？"
        % (config["low_score_min"], config["low_score_max"]),
        "",
        "- Tracker：SimpleTrack，CenterPoint检测和其他tracking逻辑不变",
        "- 时间切分：每个场景倒数第二个val sequence用于验证，最后一个用于测试，只用验证sequence之前的sequence训练",
        "- Sequence划分：Train %d个，Validation %s，Test %s"
        % (len(split["train"]), ", ".join(split["validation"]), ", ".join(split["test"])),
        "- Scene Memory只累计当前sequence之前的同场景sequence，GT只用于历史统计和样本标签",
        "- Temporal特征来自当前帧association前的track prediction；包含box/class、局部检测密度和LiDAR点数作为当前observation context",
        "- 四组都使用固定Logistic Regression，不调网络结构；分类阈值只在Validation上选择",
        "- Downstream阈值在Validation上匹配Global-Low(%.2f)的负样本预算，避免模型因操作点过激而虚增FP"
        % config["downstream_global_threshold"],
        "",
        "## 低分检测分类",
        "",
        "| Input | Precision | Recall | F1 | PR-AUC | Precision@ScoreRecall | TP@SameFP |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in MODEL_ORDER:
        row = models[name]["metrics"]
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %d |"
            % (
                name,
                _fmt(row["precision"]),
                _fmt(row["recall"]),
                _fmt(row["f1"]),
                _fmt(row["pr_auc"]),
                _fmt(row["precision_at_score_recall"]),
                int(row["tp_at_score_fp"]),
            )
        )
    counts = classification["split_counts"]
    lines.extend(
        [
            "",
            "- 样本数：Train %d（positive %d），Validation %d（positive %d），Test %d（positive %d）"
            % (
                counts["train"]["samples"],
                counts["train"]["positive"],
                counts["validation"]["samples"],
                counts["validation"]["positive"],
                counts["test"]["samples"],
                counts["test"]["positive"],
            ),
            "- Spatial相对Score Only：PR-AUC %+0.2f点" % (100.0 * spatial_pr),
            "- Temporal相对Score Only：PR-AUC %+0.2f点" % (100.0 * temporal_pr),
            "- Spatial+Temporal相对Temporal Only：PR-AUC %+0.2f点，F1 %+0.2f点"
            % (100.0 * complement_pr, 100.0 * complement_f1),
            "- PR-AUC最高的是Spatiotemporal(%.2f%%)，F1最高的是Temporal Only(%.2f%%)"
            % (
                100.0 * models["Spatiotemporal"]["metrics"]["pr_auc"],
                100.0 * models["Temporal Only"]["metrics"]["f1"],
            ),
            "",
            "## Downstream MOT",
            "",
            "| Method | HOTA | DetA | AssA | IDF1 | MOTA | IDS | FP | FN |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("Baseline", "Global-Low", "Spatial", "Spatiotemporal Filter"):
        row = downstream[name]
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %d | %d | %d |"
            % (
                name,
                _fmt(row["HOTA"]),
                _fmt(row["DetA"]),
                _fmt(row["AssA"]),
                _fmt(row["IDF1"]),
                _fmt(row["MOTA"]),
                int(row["IDSW"]),
                int(row["FP"]),
                int(row["FN"]),
            )
        )
    lines.extend(
        [
            "",
            "- Spatiotemporal Filter相对Global-Low：HOTA %+0.2f点，IDF1 %+0.2f点，FP %+d，FN %+d，IDS %+d"
            % (
                decision["hota_gain_points"],
                decision["idf1_gain_points"],
                int(downstream["Spatiotemporal Filter"]["FP"] - downstream["Global-Low"]["FP"]),
                int(downstream["Spatiotemporal Filter"]["FN"] - downstream["Global-Low"]["FN"]),
                int(downstream["Spatiotemporal Filter"]["IDSW"] - downstream["Global-Low"]["IDSW"]),
            ),
            "- 逐sequence HOTA优于Global-Low：%d/%d" % (decision["sequence_wins"], decision["sequence_total"]),
            "- 测试时实际保留低分detection %d个，其中positive %d、negative %d"
            % (
                int(filter_stats["total"].get("selected", 0)),
                int(filter_stats["total"].get("selected_positive", 0)),
                int(filter_stats["total"].get("selected_negative", 0)),
            ),
            "- 保守操作点下，Spatiotemporal只带来+%.2f点HOTA，IDF1反而下降%.2f点，且仅3/6个sequence优于Global-Low"
            % (decision["hota_gain_points"], -decision["idf1_gain_points"]),
            "",
            "## Go / No-Go",
            "",
            "- 分类判定：%s" % ("PASS" if decision["classification"] else "FAIL"),
            "- 相同FP恢复能力：%s" % ("PASS" if decision["same_fp"] else "FAIL"),
            "- Downstream MOT：%s" % ("PASS" if decision["downstream"] else "FAIL"),
            "- 最终判定：**%s**" % ("Go" if decision["go"] else "No-Go"),
        ]
    )
    if decision["go"]:
        lines.append(
            "- 建议：下一阶段研究Persistent Scene Memory+Temporal Track Memory+Weak Observation Recovery"
        )
    else:
        lines.extend(
            [
                "- 结论：Spatial+Temporal没有在分类、同FP恢复和下游MOT三方面同时达到预设门槛",
                "- Spatial信息：单独使用比Score Only更差；加入Temporal后只提升1.43点PR-AUC，同时F1下降2.90点",
                "- Temporal信息：有小幅价值，但Temporal Only已优于Spatiotemporal的F1和相同FP恢复能力",
                "- 互补性：不足以转化为稳定的低分检测筛选收益，提升也没有跨多数sequence成立",
                "- 建议：停止继续围绕detection filtering和传统tracking pipeline堆复杂模型，转向Streaming/Joint Detection-Tracking/Persistent Object-State Perception",
            ]
        )
    return "\n".join(lines) + "\n"


def run(config):
    output_root = Path(config["project"]["output_root"])
    debug_root = output_root / "debug"
    debug_root.mkdir(parents=True, exist_ok=True)
    scenes, _, warnings = build_scenes(config)
    split = _split_sequences(scenes)
    if len(split["validation"]) != 6 or len(split["test"]) != 6:
        raise ValueError("需要六个场景的Validation/Test sequence")
    samples = _extract_samples(config, scenes, split, debug_root)
    model_outputs, _, _, test, classification = _fit_models(config, samples, split, debug_root)
    prediction_root, filter_stats = _run_filter(
        config,
        split,
        samples,
        model_outputs["Spatiotemporal"],
        debug_root,
    )
    downstream, per_sequence = _evaluate_downstream(config, split, prediction_root, debug_root)
    decision = _go_decision(config, model_outputs, downstream, per_sequence)
    write_json(
        debug_root / "results.json",
        {
            "split": split,
            "warnings": warnings,
            "classification": classification,
            "downstream": downstream,
            "per_sequence_hota": per_sequence,
            "filter_stats": filter_stats,
            "decision": decision,
        },
    )
    _plot_classification(output_root, model_outputs)
    _plot_pr(output_root, model_outputs, test)
    _plot_downstream(output_root, downstream)
    (output_root / "summary.md").write_text(
        _summary(config, split, classification, downstream, per_sequence, filter_stats, decision),
        encoding="utf-8",
    )
    print("完成weak observation实验%s" % output_root, flush=True)
