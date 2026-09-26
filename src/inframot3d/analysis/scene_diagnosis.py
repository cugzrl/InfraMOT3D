import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

from inframot3d.analysis.scene_difficulty import (
    MAP_ROLE,
    _bin_index,
    _grid_shape,
    _match,
    _objects,
    build_scenes,
)
from inframot3d.evaluation.protocols.v2xseq import V2XSeqProtocol
from inframot3d.io import read_json, read_jsonl, write_json


COUNT_KEYS = ("gt", "miss", "idsw", "frag")
FAILURE_TYPES = (
    "detection_missing",
    "localization_weakness",
    "tracking_miss_good_detection",
    "direct_id_switch",
    "reinit_after_gap_proxy",
    "same_id_reconnect",
)


def _path(root, value):
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _round(value, digits=6):
    if value is None or not math.isfinite(float(value)):
        return ""
    return round(float(value), digits)


def _rate(events, total, scale=1.0):
    if float(total) <= 0:
        return None
    return float(events) / float(total) * float(scale)


def _risk_ratio(hard_events, hard_total, normal_events, normal_total):
    if float(hard_total) <= 0 or float(normal_total) <= 0:
        return None
    hard = (float(hard_events) + 0.5) / (float(hard_total) + 1.0)
    normal = (float(normal_events) + 0.5) / (float(normal_total) + 1.0)
    return hard / normal


def _minimum(config, cell):
    values = config["min_observations"]
    key = str(int(cell)) if float(cell).is_integer() else str(float(cell))
    return int(values[key])


def _empty_counts(shape):
    return {key: np.zeros(shape, dtype=np.float64) for key in COUNT_KEYS}


def _centers(items):
    if not items:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray([[float(item["box"][0]), float(item["box"][1])] for item in items], dtype=np.float64)


def _candidate_stats(center, items, matched_index, radius):
    points = _centers(items)
    if points.size == 0:
        return 0, None
    distances = np.hypot(points[:, 0] - float(center[0]), points[:, 1] - float(center[1]))
    count = int(np.sum(distances <= float(radius)))
    if matched_index is None:
        return count, None
    correct = float(distances[int(matched_index)])
    others = np.delete(distances, int(matched_index))
    margin = float(others.min() - correct) if others.size else None
    return count, margin


def _local_neighbors(center, gt_items, gt_index, radius):
    points = _centers(gt_items)
    if points.size == 0:
        return 0
    distances = np.hypot(points[:, 0] - float(center[0]), points[:, 1] - float(center[1]))
    mask = distances <= float(radius)
    mask[int(gt_index)] = False
    return int(mask.sum())


def _center_residual(left, right):
    return float(math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1])))


def _annotate_gaps(events, tracker_names):
    groups = defaultdict(list)
    for event in events:
        groups[(event["sequence_id"], event["gt_id"])].append(event)
        event["det_gap_bucket"] = ""
        for name in tracker_names:
            event["trackers"][name]["frag_start"] = 0
    for values in groups.values():
        ordered = sorted(values, key=lambda item: int(item["frame_index"]))
        start = None
        width = 0
        previous = None
        for event in ordered + [None]:
            contiguous = event is not None and (previous is None or int(event["frame_index"]) == previous + 1)
            missing = event is not None and int(event["det_miss"]) == 1
            if start is not None and (not missing or not contiguous):
                if width == 1:
                    bucket = "1"
                elif width <= 3:
                    bucket = "2-3"
                elif width <= 5:
                    bucket = "4-5"
                else:
                    bucket = ">5"
                start["det_gap_bucket"] = bucket
                start = None
                width = 0
            if missing:
                if start is None:
                    start = event
                width += 1
            previous = int(event["frame_index"]) if event is not None else None
        for name in tracker_names:
            matched_before = False
            gap_start = None
            previous = None
            for event in ordered:
                contiguous = previous is None or int(event["frame_index"]) == previous + 1
                if not contiguous:
                    matched_before = False
                    gap_start = None
                values_by_tracker = event["trackers"][name]
                if int(values_by_tracker["miss"]) == 0:
                    if matched_before and gap_start is not None:
                        gap_start["trackers"][name]["frag_start"] = 1
                    matched_before = True
                    gap_start = None
                elif matched_before and gap_start is None:
                    gap_start = event
                previous = int(event["frame_index"])


def collect_events(config, scenes):
    root = config["_root"]
    protocol = V2XSeqProtocol(root)
    converted = config["project"]["converted_root"]
    detection_root = _path(root, config["detection_root"])
    trackers = [(item["name"], _path(root, item["prediction_root"])) for item in config["trackers"]]
    events = []
    for scene in scenes:
        for sequence in scene["sequences"]:
            sequence_id = sequence["sequence_id"]
            gt_rows = list(read_jsonl(converted / "sequences" / (sequence_id + ".jsonl")))
            det_rows = {int(row["timestamp"]): row for row in read_jsonl(detection_root / (sequence_id + ".jsonl"))}
            predictions = {}
            for name, folder in trackers:
                path = folder / (sequence_id + ".jsonl")
                if not path.is_file():
                    raise SystemExit("缺少%s预测 %s" % (name, path))
                predictions[name] = {int(row["timestamp"]): row for row in read_jsonl(path)}
            states = {name: {} for name, _ in trackers}
            for row in gt_rows:
                stamp = int(row["timestamp"])
                frame_index = int(row["frame_index"])
                gt_items = _objects(row, protocol)
                det_items = _objects(det_rows.get(stamp, {"objects": []}), protocol)
                det_matches = _match(
                    [item["box"] for item in gt_items],
                    [item["box"] for item in det_items],
                    config["iou_threshold"],
                    config["match_center_gate_m"],
                )
                det_by_gt = {left: (right, iou) for left, right, iou in det_matches}
                tracker_matches = {}
                for name, _ in trackers:
                    items = _objects(predictions[name].get(stamp, {"objects": []}), protocol)
                    pairs = _match(
                        [item["box"] for item in gt_items],
                        [item["box"] for item in items],
                        config["iou_threshold"],
                        config["match_center_gate_m"],
                    )
                    tracker_matches[name] = (items, {left: (right, iou) for left, right, iou in pairs})
                for gt_index, gt_item in enumerate(gt_items):
                    center = [float(value) for value in gt_item["box"][:2]]
                    gt_id = str(gt_item["source_track_id"])
                    det_index = None
                    det_iou = None
                    det_score = None
                    det_center = None
                    if gt_index in det_by_gt:
                        det_index, det_iou = det_by_gt[gt_index]
                        det_score = float(det_items[det_index].get("score", 1.0))
                        det_center = [float(value) for value in det_items[det_index]["box"][:2]]
                    det_candidates, det_margin = _candidate_stats(
                        center,
                        det_items,
                        det_index,
                        config["candidate_radius_m"],
                    )
                    det_residual = _center_residual(center, det_center) if det_center is not None else None
                    det_weak = det_index is not None and (
                        float(det_iou) < float(config["localization_good_iou"])
                        or float(det_residual) > float(config["localization_good_center_m"])
                    )
                    event = {
                        "scene_id": scene["scene_id"],
                        "sequence_id": sequence_id,
                        "sequence_split": sequence["split"],
                        "frame_index": frame_index,
                        "timestamp": stamp,
                        "gt_id": gt_id,
                        "x": center[0],
                        "y": center[1],
                        "det_miss": int(det_index is None),
                        "det_score": det_score,
                        "det_iou": det_iou,
                        "det_residual_m": det_residual,
                        "det_weak": int(det_weak),
                        "det_candidate_count": det_candidates,
                        "det_center_margin_m": det_margin,
                        "local_neighbors": _local_neighbors(
                            center,
                            gt_items,
                            gt_index,
                            config["local_density_radius_m"],
                        ),
                        "trackers": {},
                    }
                    for name, _ in trackers:
                        pred_items, mapping = tracker_matches[name]
                        matched = gt_index in mapping
                        pred_index = None
                        track_iou = None
                        track_id = None
                        if matched:
                            pred_index, track_iou = mapping[gt_index]
                            track_id = str(pred_items[pred_index]["track_id"])
                        candidate_count, center_margin = _candidate_stats(
                            center,
                            pred_items,
                            pred_index,
                            config["candidate_radius_m"],
                        )
                        post_residual = None
                        if pred_index is not None and det_center is not None:
                            post_residual = _center_residual(pred_items[pred_index]["box"][:2], det_center)
                        state = states[name].get(gt_id)
                        if state is not None and frame_index != int(state["last_gt_frame"]) + 1:
                            state = None
                        idsw = 0
                        reconnect = 0
                        reinit = 0
                        gap_length = 0
                        if matched:
                            if state is not None and state["matched_before"]:
                                if int(state["gap_length"]) > 0:
                                    reconnect = 1
                                    gap_length = int(state["gap_length"])
                                    reinit = int(track_id != state["last_track_id"])
                                elif track_id != state["last_track_id"]:
                                    idsw = 1
                            states[name][gt_id] = {
                                "matched_before": True,
                                "last_track_id": track_id,
                                "gap_length": 0,
                                "last_gt_frame": frame_index,
                            }
                        else:
                            if state is None:
                                states[name][gt_id] = {
                                    "matched_before": False,
                                    "last_track_id": None,
                                    "gap_length": 0,
                                    "last_gt_frame": frame_index,
                                }
                            else:
                                state["gap_length"] = int(state["gap_length"]) + 1
                                state["last_gt_frame"] = frame_index
                        event["trackers"][name] = {
                            "miss": int(not matched),
                            "idsw": idsw,
                            "reconnect": reconnect,
                            "reinit_after_gap": reinit,
                            "gap_length": gap_length,
                            "track_iou": track_iou,
                            "post_residual_m": post_residual,
                            "candidate_count": candidate_count,
                            "center_margin_m": center_margin,
                        }
                    events.append(event)
            print("完成诊断事件%s %s" % (scene["scene_id"], sequence_id), flush=True)
    _annotate_gaps(events, [name for name, _ in trackers])
    return events


def _event_index(events):
    grouped = defaultdict(list)
    for event in events:
        grouped[(event["scene_id"], event["sequence_id"])].append(event)
    return grouped


def _accumulate(records, tracker, bev, cell):
    shape = _grid_shape(bev, cell)
    counts = _empty_counts(shape)
    for event in records:
        index = _bin_index(event["x"], event["y"], bev, cell, shape)
        if index is None:
            continue
        values = event["trackers"][tracker]
        counts["gt"][index] += 1
        counts["miss"][index] += int(values["miss"])
        counts["idsw"][index] += int(values["idsw"])
        counts["frag"][index] += int(values["frag_start"])
    return counts


def _risk_map(counts, prior_strength, minimum):
    gt = counts["gt"]
    total = float(gt.sum())
    global_rate = float(counts["miss"].sum() / total) if total > 0 else 0.0
    risk = (counts["miss"] + float(prior_strength) * global_rate) / (gt + float(prior_strength))
    risk = risk.astype(np.float64)
    risk[gt < int(minimum)] = np.nan
    return risk


def _cell_features(indices, counts, bev, cell):
    output = {}
    for iy, ix in indices:
        x = float(bev["x_min"]) + (ix + 0.5) * float(cell)
        y = float(bev["y_min"]) + (iy + 0.5) * float(cell)
        output[(iy, ix)] = (math.hypot(x, y), math.log1p(float(counts["gt"][iy, ix])))
    return output


def _matched_normal(hard, candidates, counts, bev, cell):
    if not hard or not candidates:
        return []
    features = _cell_features(list(hard) + list(candidates), counts, bev, cell)
    ranges = np.asarray([item[0] for item in features.values()], dtype=np.float64)
    logs = np.asarray([item[1] for item in features.values()], dtype=np.float64)
    range_scale = max(float(np.std(ranges)), 1.0)
    log_scale = max(float(np.std(logs)), 0.2)
    remaining = set(candidates)
    selected = []
    for source in hard:
        if not remaining:
            break
        source_range, source_log = features[source]
        target = min(
            remaining,
            key=lambda item: abs(features[item][0] - source_range) / range_scale
            + abs(features[item][1] - source_log) / log_scale,
        )
        selected.append(target)
        remaining.remove(target)
    return selected


def _select_regions(risk, counts, bev, cell, fraction):
    valid = [tuple(int(value) for value in item) for item in np.argwhere(np.isfinite(risk))]
    if len(valid) < 4:
        return [], []
    width = max(1, int(math.ceil(float(fraction) * len(valid))))
    hard = sorted(valid, key=lambda item: float(risk[item]), reverse=True)[:width]
    hard_set = set(hard)
    normal = _matched_normal(hard, [item for item in valid if item not in hard_set], counts, bev, cell)
    return hard, normal


def _rank_normalize(values):
    output = np.full(values.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(values)
    if int(mask.sum()) == 0:
        return output
    ranked = rankdata(values[mask], method="average")
    output[mask] = (ranked - 1.0) / max(float(ranked.size - 1), 1.0)
    return output


def _pool(counts, indices):
    values = {key: 0.0 for key in COUNT_KEYS}
    for index in indices:
        for key in COUNT_KEYS:
            values[key] += float(counts[key][index])
    return values


def _correlation(history_risk, future_counts, prior_strength, minimum):
    future_risk = _risk_map(future_counts, prior_strength, minimum)
    mask = np.isfinite(history_risk) & np.isfinite(future_risk)
    if int(mask.sum()) < 3:
        return None
    left = history_risk[mask]
    right = future_risk[mask]
    if np.unique(left).size < 2 or np.unique(right).size < 2:
        return None
    return float(spearmanr(left, right).statistic)


def _risk_row(prefix, scene_id, target_id, cell, history_ids, history_tracker, future_tracker, hard, normal, future_counts, history_risk, config):
    hard_values = _pool(future_counts, hard)
    normal_values = _pool(future_counts, normal)
    hard_identity = hard_values["idsw"] + hard_values["frag"]
    normal_identity = normal_values["idsw"] + normal_values["frag"]
    return {
        "protocol": prefix,
        "scene_id": scene_id,
        "target_sequence": target_id,
        "grid_m": float(cell),
        "history_sequences": ";".join(history_ids),
        "history_tracker": history_tracker,
        "future_tracker": future_tracker,
        "hard_cells": len(hard),
        "normal_cells": len(normal),
        "hard_gt": int(hard_values["gt"]),
        "normal_gt": int(normal_values["gt"]),
        "hard_miss": int(hard_values["miss"]),
        "normal_miss": int(normal_values["miss"]),
        "hard_idsw": int(hard_values["idsw"]),
        "normal_idsw": int(normal_values["idsw"]),
        "hard_fragment": int(hard_values["frag"]),
        "normal_fragment": int(normal_values["frag"]),
        "hard_miss_rate": _round(_rate(hard_values["miss"], hard_values["gt"])),
        "normal_miss_rate": _round(_rate(normal_values["miss"], normal_values["gt"])),
        "miss_risk_ratio": _round(
            _risk_ratio(hard_values["miss"], hard_values["gt"], normal_values["miss"], normal_values["gt"])
        ),
        "hard_identity_per_1k": _round(_rate(hard_identity, hard_values["gt"], 1000.0)),
        "normal_identity_per_1k": _round(_rate(normal_identity, normal_values["gt"], 1000.0)),
        "identity_risk_ratio": _round(
            _risk_ratio(hard_identity, hard_values["gt"], normal_identity, normal_values["gt"])
        ),
        "spearman": _round(
            _correlation(history_risk, future_counts, config["prior_strength"], _minimum(config, cell))
        ),
    }


def _jaccard(left, right):
    union = set(left) | set(right)
    return float(len(set(left) & set(right)) / len(union)) if union else None


def _overlap(left, right):
    denominator = min(len(left), len(right))
    return float(len(set(left) & set(right)) / denominator) if denominator else None


def _future_targets(scene):
    output = []
    sequences = scene["sequences"]
    for index, sequence in enumerate(sequences):
        if sequence["split"] != "val" or index == 0:
            continue
        output.append((index, sequence["sequence_id"], [item["sequence_id"] for item in sequences[:index]]))
    return output


def _failure_type(event, tracker):
    values = event["trackers"][tracker]
    if int(values["miss"]) == 1:
        if int(event["det_miss"]) == 1:
            return "detection_missing"
        if int(event["det_weak"]) == 1:
            return "localization_weakness"
        return "tracking_miss_good_detection"
    if int(values["idsw"]) == 1:
        return "direct_id_switch"
    if int(values["reconnect"]) == 1:
        return "reinit_after_gap_proxy" if int(values["reinit_after_gap"]) == 1 else "same_id_reconnect"
    return None


def _update_diagnostics(records, scene_id, tracker_names, hard, normal, config, cell, stores):
    hard_set = set(hard)
    normal_set = set(normal)
    shape = _grid_shape(config["bev"], cell)
    exposure, failure, condition_num, condition_den, gaps, metrics = stores
    for event in records:
        index = _bin_index(event["x"], event["y"], config["bev"], cell, shape)
        if index in hard_set:
            region = "hard"
        elif index in normal_set:
            region = "normal"
        else:
            continue
        if event["det_gap_bucket"]:
            gaps[(scene_id, region, event["det_gap_bucket"])] += 1
        for tracker in tracker_names:
            key = (scene_id, tracker, region)
            exposure[key] += 1
            metric_values = {
                "detection_confidence": event["det_score"],
                "detection_iou": event["det_iou"],
                "detection_center_residual_m": event["det_residual_m"],
                "local_neighbors": event["local_neighbors"],
                "detection_center_margin_m": event["det_center_margin_m"],
                "post_update_residual_m": event["trackers"][tracker]["post_residual_m"],
                "tracker_output_center_margin_m": event["trackers"][tracker]["center_margin_m"],
            }
            for metric, value in metric_values.items():
                if value is not None and math.isfinite(float(value)):
                    metrics[(scene_id, tracker, region, metric)].append(float(value))
            category = _failure_type(event, tracker)
            if category is not None:
                failure[(scene_id, tracker, region, category)] += 1
            values = event["trackers"][tracker]
            conditions = {
                "detection_missing": (int(event["det_miss"]) == 1, True),
                "weak_localization_given_detection": (int(event["det_weak"]) == 1, int(event["det_miss"]) == 0),
                "tracking_miss_given_good_detection": (
                    int(values["miss"]) == 1 and int(event["det_miss"]) == 0 and int(event["det_weak"]) == 0,
                    int(event["det_miss"]) == 0 and int(event["det_weak"]) == 0,
                ),
                "direct_id_switch": (int(values["idsw"]) == 1, True),
                "fragment_onset": (int(values["frag_start"]) == 1, True),
                "reinit_after_gap_proxy": (int(values["reinit_after_gap"]) == 1, True),
                "large_post_update_residual": (
                    values["post_residual_m"] is not None
                    and float(values["post_residual_m"]) > float(config["large_post_residual_m"]),
                    values["post_residual_m"] is not None,
                ),
                "high_local_density": (int(event["local_neighbors"]) >= 2, True),
                "detection_ambiguity_proxy": (
                    event["det_center_margin_m"] is not None
                    and float(event["det_center_margin_m"]) < float(config["ambiguity_margin_m"]),
                    event["det_center_margin_m"] is not None,
                ),
                "tracker_output_ambiguity_proxy": (
                    values["center_margin_m"] is not None
                    and float(values["center_margin_m"]) < float(config["ambiguity_margin_m"]),
                    values["center_margin_m"] is not None,
                ),
            }
            for condition, (positive, observed) in conditions.items():
                if observed:
                    condition_den[(scene_id, tracker, region, condition)] += 1
                    condition_num[(scene_id, tracker, region, condition)] += int(positive)


def _run_folds(scenes, grouped, config):
    tracker_names = [item["name"] for item in config["trackers"]]
    self_rows = []
    cross_rows = []
    overlap_rows = []
    consensus_rows = []
    vote_rows = []
    exposure = Counter()
    failure = Counter()
    condition_num = Counter()
    condition_den = Counter()
    gaps = Counter()
    metrics = defaultdict(list)
    figure_data = None
    for scene in scenes:
        for target_index, target_id, history_ids in _future_targets(scene):
            history_records = []
            for sequence_id in history_ids:
                history_records.extend(grouped[(scene["scene_id"], sequence_id)])
            target_records = grouped[(scene["scene_id"], target_id)]
            for cell_value in config["grid_sizes"]:
                cell = float(cell_value)
                minimum = _minimum(config, cell)
                history_counts = {
                    name: _accumulate(history_records, name, config["bev"], cell) for name in tracker_names
                }
                future_counts = {
                    name: _accumulate(target_records, name, config["bev"], cell) for name in tracker_names
                }
                risks = {
                    name: _risk_map(history_counts[name], config["prior_strength"], minimum) for name in tracker_names
                }
                regions = {
                    name: _select_regions(
                        risks[name],
                        history_counts[name],
                        config["bev"],
                        cell,
                        config["hard_fraction"],
                    )
                    for name in tracker_names
                }
                for history_tracker in tracker_names:
                    hard, normal = regions[history_tracker]
                    if not hard or not normal:
                        continue
                    for future_tracker in tracker_names:
                        row = _risk_row(
                            "cross_tracker",
                            scene["scene_id"],
                            target_id,
                            cell,
                            history_ids,
                            history_tracker,
                            future_tracker,
                            hard,
                            normal,
                            future_counts[future_tracker],
                            risks[history_tracker],
                            config,
                        )
                        cross_rows.append(row)
                        if history_tracker == future_tracker:
                            self_rows.append(dict(row, protocol="self_persistence"))
                for left_index, left in enumerate(tracker_names):
                    left_hard = regions[left][0]
                    if not left_hard:
                        continue
                    for right in tracker_names[left_index + 1 :]:
                        right_hard = regions[right][0]
                        if not right_hard:
                            continue
                        overlap_rows.append(
                            {
                                "scene_id": scene["scene_id"],
                                "target_sequence": target_id,
                                "grid_m": cell,
                                "left_tracker": left,
                                "right_tracker": right,
                                "left_hard_cells": len(left_hard),
                                "right_hard_cells": len(right_hard),
                                "intersection_cells": len(set(left_hard) & set(right_hard)),
                                "hard_iou": _round(_jaccard(left_hard, right_hard)),
                                "overlap_ratio": _round(_overlap(left_hard, right_hard)),
                            }
                        )
                normalized = np.stack([_rank_normalize(risks[name]) for name in tracker_names])
                valid_count = np.sum(np.isfinite(normalized), axis=0)
                consensus = np.divide(
                    np.nansum(normalized, axis=0),
                    valid_count,
                    out=np.full(valid_count.shape, np.nan, dtype=np.float64),
                    where=valid_count > 0,
                )
                reference_counts = history_counts[tracker_names[0]]
                consensus_hard, consensus_normal = _select_regions(
                    consensus,
                    reference_counts,
                    config["bev"],
                    cell,
                    config["hard_fraction"],
                )
                if not consensus_hard or not consensus_normal:
                    continue
                for future_tracker in tracker_names:
                    consensus_rows.append(
                        _risk_row(
                            "consensus_top20",
                            scene["scene_id"],
                            target_id,
                            cell,
                            history_ids,
                            "Consensus",
                            future_tracker,
                            consensus_hard,
                            consensus_normal,
                            future_counts[future_tracker],
                            consensus,
                            config,
                        )
                    )
                votes = np.zeros(consensus.shape, dtype=np.int64)
                for name in tracker_names:
                    for index in regions[name][0]:
                        votes[index] += 1
                valid = [tuple(int(value) for value in item) for item in np.argwhere(np.isfinite(consensus))]
                for vote in range(len(tracker_names) + 1):
                    vote_rows.append(
                        {
                            "scene_id": scene["scene_id"],
                            "target_sequence": target_id,
                            "grid_m": cell,
                            "tracker_votes": vote,
                            "cells": int(sum(int(votes[index]) == vote for index in valid)),
                            "valid_cells": len(valid),
                        }
                    )
                for threshold in config["consensus_thresholds"]:
                    threshold_hard = [index for index in valid if int(votes[index]) >= int(threshold)]
                    if not threshold_hard:
                        continue
                    hard_set = set(threshold_hard)
                    threshold_normal = _matched_normal(
                        threshold_hard,
                        [index for index in valid if index not in hard_set],
                        reference_counts,
                        config["bev"],
                        cell,
                    )
                    if not threshold_normal:
                        continue
                    for future_tracker in tracker_names:
                        consensus_rows.append(
                            _risk_row(
                                "consensus_%dof%d" % (int(threshold), len(tracker_names)),
                                scene["scene_id"],
                                target_id,
                                cell,
                                history_ids,
                                "Consensus",
                                future_tracker,
                                threshold_hard,
                                threshold_normal,
                                future_counts[future_tracker],
                                consensus,
                                config,
                            )
                        )
                if cell == float(config["primary_grid"]):
                    _update_diagnostics(
                        target_records,
                        scene["scene_id"],
                        tracker_names,
                        consensus_hard,
                        consensus_normal,
                        config,
                        cell,
                        (exposure, failure, condition_num, condition_den, gaps, metrics),
                    )
                    if (
                        scene["scene_id"] == config["figure_scene"]
                        and target_id == str(config["figure_target_sequence"])
                    ):
                        figure_data = {
                            "scene_id": scene["scene_id"],
                            "target_sequence": target_id,
                            "history_ids": history_ids,
                            "risks": {name: _rank_normalize(risks[name]) for name in tracker_names},
                            "consensus": consensus,
                            "regions": regions,
                            "consensus_hard": consensus_hard,
                            "consensus_normal": consensus_normal,
                        }
    stores = {
        "exposure": exposure,
        "failure": failure,
        "condition_num": condition_num,
        "condition_den": condition_den,
        "gaps": gaps,
        "metrics": metrics,
    }
    return self_rows, cross_rows, overlap_rows, consensus_rows, vote_rows, stores, figure_data


def _aggregate_risk(rows, keys):
    count_keys = (
        "hard_gt",
        "normal_gt",
        "hard_miss",
        "normal_miss",
        "hard_idsw",
        "normal_idsw",
        "hard_fragment",
        "normal_fragment",
    )
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for group_key, current in grouped.items():
        values = {key: sum(int(row[key]) for row in current) for key in count_keys}
        hard_identity = values["hard_idsw"] + values["hard_fragment"]
        normal_identity = values["normal_idsw"] + values["normal_fragment"]
        correlations = [float(row["spearman"]) for row in current if row["spearman"] != ""]
        result = {key: value for key, value in zip(keys, group_key)}
        result.update(values)
        result.update(
            {
                "folds": len(current),
                "scenes": len({row["scene_id"] for row in current}),
                "hard_miss_rate": _round(_rate(values["hard_miss"], values["hard_gt"])),
                "normal_miss_rate": _round(_rate(values["normal_miss"], values["normal_gt"])),
                "miss_risk_ratio": _round(
                    _risk_ratio(values["hard_miss"], values["hard_gt"], values["normal_miss"], values["normal_gt"])
                ),
                "hard_identity_per_1k": _round(_rate(hard_identity, values["hard_gt"], 1000.0)),
                "normal_identity_per_1k": _round(_rate(normal_identity, values["normal_gt"], 1000.0)),
                "identity_risk_ratio": _round(
                    _risk_ratio(hard_identity, values["hard_gt"], normal_identity, values["normal_gt"])
                ),
                "spearman_mean": _round(float(np.mean(correlations)) if correlations else None),
            }
        )
        output.append(result)
    return output


def _aggregate_overlap(rows, keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for group_key, current in grouped.items():
        ious = [float(row["hard_iou"]) for row in current if row["hard_iou"] != ""]
        overlaps = [float(row["overlap_ratio"]) for row in current if row["overlap_ratio"] != ""]
        result = {key: value for key, value in zip(keys, group_key)}
        result.update(
            {
                "folds": len(current),
                "hard_iou_mean": _round(float(np.mean(ious)) if ious else None),
                "hard_iou_median": _round(float(np.median(ious)) if ious else None),
                "overlap_ratio_mean": _round(float(np.mean(overlaps)) if overlaps else None),
            }
        )
        output.append(result)
    return output


def _failure_rows(stores, tracker_names, include_scene):
    scenes = sorted({key[0] for key in stores["exposure"]})
    scene_groups = scenes if include_scene else ["ALL"]
    output = []
    for scene_group in scene_groups:
        selected_scenes = [scene_group] if include_scene else scenes
        for tracker_group in list(tracker_names) + ["ALL"]:
            selected_trackers = tracker_names if tracker_group == "ALL" else [tracker_group]
            for region in ("hard", "normal"):
                exposure = sum(
                    stores["exposure"][(scene, tracker, region)]
                    for scene in selected_scenes
                    for tracker in selected_trackers
                )
                total_failures = sum(
                    stores["failure"][(scene, tracker, region, failure_type)]
                    for scene in selected_scenes
                    for tracker in selected_trackers
                    for failure_type in FAILURE_TYPES
                )
                for failure_type in FAILURE_TYPES:
                    count = sum(
                        stores["failure"][(scene, tracker, region, failure_type)]
                        for scene in selected_scenes
                        for tracker in selected_trackers
                    )
                    row = {
                        "tracker": tracker_group,
                        "region": region,
                        "failure_type": failure_type,
                        "count": count,
                        "gt_exposure": exposure,
                        "per_1k_gt": _round(_rate(count, exposure, 1000.0)),
                        "composition_percent": _round(_rate(count, total_failures, 100.0)),
                    }
                    if include_scene:
                        row = {"scene_id": scene_group, **row}
                    output.append(row)
    return output


def _condition_rows(stores, tracker_names, include_scene):
    scenes = sorted({key[0] for key in stores["exposure"]})
    conditions = sorted({key[3] for key in stores["condition_den"]})
    scene_groups = scenes if include_scene else ["ALL"]
    output = []
    for scene_group in scene_groups:
        selected_scenes = [scene_group] if include_scene else scenes
        for tracker_group in list(tracker_names) + ["ALL"]:
            selected_trackers = tracker_names if tracker_group == "ALL" else [tracker_group]
            for condition in conditions:
                values = {}
                for region in ("hard", "normal"):
                    numerator = sum(
                        stores["condition_num"][(scene, tracker, region, condition)]
                        for scene in selected_scenes
                        for tracker in selected_trackers
                    )
                    denominator = sum(
                        stores["condition_den"][(scene, tracker, region, condition)]
                        for scene in selected_scenes
                        for tracker in selected_trackers
                    )
                    values[region] = (numerator, denominator)
                hard_num, hard_den = values["hard"]
                normal_num, normal_den = values["normal"]
                row = {
                    "tracker": tracker_group,
                    "condition": condition,
                    "hard_count": hard_num,
                    "hard_denominator": hard_den,
                    "hard_rate": _round(_rate(hard_num, hard_den)),
                    "normal_count": normal_num,
                    "normal_denominator": normal_den,
                    "normal_rate": _round(_rate(normal_num, normal_den)),
                    "risk_ratio": _round(_risk_ratio(hard_num, hard_den, normal_num, normal_den)),
                }
                if include_scene:
                    row = {"scene_id": scene_group, **row}
                output.append(row)
    return output


def _gap_rows(stores, tracker_names, include_scene):
    scenes = sorted({key[0] for key in stores["exposure"]})
    scene_groups = scenes if include_scene else ["ALL"]
    output = []
    reference = tracker_names[0]
    for scene_group in scene_groups:
        selected_scenes = [scene_group] if include_scene else scenes
        for bucket in ("1", "2-3", "4-5", ">5"):
            row = {"gap_length": bucket}
            for region in ("hard", "normal"):
                count = sum(stores["gaps"][(scene, region, bucket)] for scene in selected_scenes)
                exposure = sum(stores["exposure"][(scene, reference, region)] for scene in selected_scenes)
                row[region + "_episodes"] = count
                row[region + "_per_1k_gt"] = _round(_rate(count, exposure, 1000.0))
            row["risk_ratio"] = _round(
                _risk_ratio(
                    row["hard_episodes"],
                    sum(stores["exposure"][(scene, reference, "hard")] for scene in selected_scenes),
                    row["normal_episodes"],
                    sum(stores["exposure"][(scene, reference, "normal")] for scene in selected_scenes),
                )
            )
            if include_scene:
                row = {"scene_id": scene_group, **row}
            output.append(row)
    return output


def _metric_rows(stores, tracker_names, include_scene):
    scenes = sorted({key[0] for key in stores["exposure"]})
    metrics = sorted({key[3] for key in stores["metrics"]})
    scene_groups = scenes if include_scene else ["ALL"]
    output = []
    for scene_group in scene_groups:
        selected_scenes = [scene_group] if include_scene else scenes
        for tracker_group in list(tracker_names) + ["ALL"]:
            selected_trackers = tracker_names if tracker_group == "ALL" else [tracker_group]
            for metric in metrics:
                row = {"tracker": tracker_group, "metric": metric}
                region_values = {}
                for region in ("hard", "normal"):
                    values = []
                    for scene in selected_scenes:
                        for tracker in selected_trackers:
                            values.extend(stores["metrics"][(scene, tracker, region, metric)])
                    array = np.asarray(values, dtype=np.float64)
                    region_values[region] = array
                    row[region + "_count"] = int(array.size)
                    row[region + "_mean"] = _round(float(array.mean()) if array.size else None)
                    row[region + "_median"] = _round(float(np.median(array)) if array.size else None)
                    row[region + "_p90"] = _round(float(np.percentile(array, 90)) if array.size else None)
                hard_mean = float(region_values["hard"].mean()) if region_values["hard"].size else None
                normal_mean = float(region_values["normal"].mean()) if region_values["normal"].size else None
                row["mean_ratio"] = _round(
                    hard_mean / normal_mean
                    if hard_mean is not None and normal_mean is not None and normal_mean != 0.0
                    else None
                )
                row["mean_difference"] = _round(
                    hard_mean - normal_mean if hard_mean is not None and normal_mean is not None else None
                )
                if include_scene:
                    row = {"scene_id": scene_group, **row}
                output.append(row)
    return output


def _fmt(value, digits=2):
    if value == "" or value is None:
        return "--"
    return ("%%.%df" % digits) % float(value)


def _write_main_tables(output_root, self_summary, cross_summary, consensus_summary, overlap_summary, condition_rows, metric_rows):
    main_self = [row for row in self_summary if float(row["grid_m"]) == 5.0]
    fine_self = [row for row in self_summary if float(row["grid_m"]) == 2.0]
    lines = [
        "# Main tables",
        "",
        "## Self-persistence",
        "",
        "| Tracker | Hard miss | Normal miss | Miss RR | Hard identity/1k | Normal identity/1k | Identity RR | Scenes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(main_self, key=lambda item: item["future_tracker"]):
        lines.append(
            "| %s | %.2f%% | %.2f%% | %s | %s | %s | %s | %d |"
            % (
                row["future_tracker"],
                float(row["hard_miss_rate"]) * 100.0,
                float(row["normal_miss_rate"]) * 100.0,
                _fmt(row["miss_risk_ratio"]),
                _fmt(row["hard_identity_per_1k"]),
                _fmt(row["normal_identity_per_1k"]),
                _fmt(row["identity_risk_ratio"]),
                int(row["scenes"]),
            )
        )
    lines.extend(
        [
            "",
            "## Grid sensitivity",
            "",
            "| Grid | Tracker | Miss RR | Identity RR | Spearman |",
            "| ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(self_summary, key=lambda item: (float(item["grid_m"]), item["future_tracker"])):
        lines.append(
            "| %.0fm | %s | %s | %s | %s |"
            % (
                float(row["grid_m"]),
                row["future_tracker"],
                _fmt(row["miss_risk_ratio"]),
                _fmt(row["identity_risk_ratio"]),
                _fmt(row["spearman_mean"], 3),
            )
        )
    lines.extend(
        [
            "",
            "## Consensus top-20% future risk",
            "",
            "| Future tracker | Hard miss | Normal miss | Miss RR | Identity RR |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    main_consensus = [
        row
        for row in consensus_summary
        if row["protocol"] == "consensus_top20" and float(row["grid_m"]) == 5.0
    ]
    for row in sorted(main_consensus, key=lambda item: item["future_tracker"]):
        lines.append(
            "| %s | %.2f%% | %.2f%% | %s | %s |"
            % (
                row["future_tracker"],
                float(row["hard_miss_rate"]) * 100.0,
                float(row["normal_miss_rate"]) * 100.0,
                _fmt(row["miss_risk_ratio"]),
                _fmt(row["identity_risk_ratio"]),
            )
        )
    lines.extend(
        [
            "",
            "## Hard-region overlap",
            "",
            "| Tracker A | Tracker B | Hard IoU | Overlap ratio |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in sorted(
        [item for item in overlap_summary if float(item["grid_m"]) == 5.0],
        key=lambda item: (item["left_tracker"], item["right_tracker"]),
    ):
        lines.append(
            "| %s | %s | %s | %s |"
            % (row["left_tracker"], row["right_tracker"], _fmt(row["hard_iou_mean"], 3), _fmt(row["overlap_ratio_mean"], 3))
        )
    lines.extend(
        [
            "",
            "## Failure conditions in consensus regions",
            "",
            "| Condition | Hard | Normal | Risk ratio |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in condition_rows:
        if row["tracker"] != "ALL":
            continue
        lines.append(
            "| %s | %.2f%% | %.2f%% | %s |"
            % (
                row["condition"],
                float(row["hard_rate"]) * 100.0 if row["hard_rate"] != "" else 0.0,
                float(row["normal_rate"]) * 100.0 if row["normal_rate"] != "" else 0.0,
                _fmt(row["risk_ratio"]),
            )
        )
    lines.extend(
        [
            "| native_motion_prediction_residual | N/A | N/A | N/A |",
            "| strict_gate_failure | N/A | N/A | N/A |",
            "| native_association_margin | N/A | N/A | N/A |",
            "| internal_early_track_death | N/A | N/A | N/A |",
        ]
    )
    lines.extend(
        [
            "",
            "## Continuous diagnostics in consensus regions",
            "",
            "| Metric | Hard mean | Normal mean | Mean ratio | Hard median | Normal median |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in metric_rows:
        if row["tracker"] != "ALL":
            continue
        lines.append(
            "| %s | %s | %s | %s | %s | %s |"
            % (
                row["metric"],
                _fmt(row["hard_mean"], 3),
                _fmt(row["normal_mean"], 3),
                _fmt(row["mean_ratio"], 3),
                _fmt(row["hard_median"], 3),
                _fmt(row["normal_median"], 3),
            )
        )
    Path(output_root / "main_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_figure(figure, output_root, stem):
    figure_dir = Path(output_root) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / (stem + ".png")
    pdf = figure_dir / (stem + ".pdf")
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    return str(png.relative_to(output_root)), str(pdf.relative_to(output_root))


def _figure_maps(output_root, figure_data, tracker_names, config):
    if figure_data is None:
        return []
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    names = list(tracker_names) + ["Consensus"]
    maps = [figure_data["risks"][name] for name in tracker_names] + [figure_data["consensus"]]
    hards = [figure_data["regions"][name][0] for name in tracker_names] + [figure_data["consensus_hard"]]
    bev = config["bev"]
    cell = float(config["primary_grid"])
    figure, axes = plt.subplots(2, 3, figsize=(13.0, 7.2), sharex=True, sharey=True)
    image = None
    for axis, name, risk, hard in zip(axes.flat, names, maps, hards):
        image = axis.imshow(
            np.ma.masked_invalid(risk),
            origin="lower",
            extent=[bev["x_min"], bev["x_max"], bev["y_min"], bev["y_max"]],
            aspect="equal",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
        )
        hard_grid = np.zeros(risk.shape, dtype=np.float64)
        for index in hard:
            hard_grid[index] = 1.0
        xs = float(bev["x_min"]) + (np.arange(risk.shape[1]) + 0.5) * cell
        ys = float(bev["y_min"]) + (np.arange(risk.shape[0]) + 0.5) * cell
        axis.contour(xs, ys, hard_grid, levels=[0.5], colors=["#00ffff"], linewidths=1.2)
        axis.scatter([0.0], [0.0], marker="^", s=45, c="white", edgecolors="black", linewidths=0.6)
        axis.set_title(name)
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
    figure.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="Historical risk percentile")
    figure.suptitle(
        "%s | history %s -> target %s"
        % (figure_data["scene_id"], ",".join(figure_data["history_ids"]), figure_data["target_sequence"]),
        fontsize=11,
    )
    paths = _save_figure(figure, output_root, "figure_a_cross_tracker_risk")
    plt.close(figure)
    write_json(
        Path(output_root) / "figures" / "figure_a_cross_tracker_risk.json",
        {
            "map_role": MAP_ROLE,
            "scene_id": figure_data["scene_id"],
            "target_sequence": figure_data["target_sequence"],
            "history_sequences": figure_data["history_ids"],
            "grid_m": cell,
            "hard_fraction": config["hard_fraction"],
        },
    )
    return list(paths)


def _figure_failure(output_root, failure_rows):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    labels = {
        "detection_missing": "Detection\nmissing",
        "localization_weakness": "Localization\nweakness",
        "tracking_miss_good_detection": "Tracking miss\nwith good det",
        "direct_id_switch": "Direct\nID switch",
        "reinit_after_gap_proxy": "Re-init after\ngap proxy",
        "same_id_reconnect": "Same-ID\nreconnect",
    }
    selected = [row for row in failure_rows if row["tracker"] == "ALL"]
    lookup = {(row["region"], row["failure_type"]): row for row in selected}
    colors = ["#4c78a8", "#f58518", "#e45756", "#72b7b2", "#54a24b", "#b279a2"]
    figure, axes = plt.subplots(1, 2, figsize=(13.4, 4.4))
    bottoms = np.zeros(2, dtype=np.float64)
    for failure_type, color in zip(FAILURE_TYPES, colors):
        values = [float(lookup[(region, failure_type)]["composition_percent"]) for region in ("hard", "normal")]
        axes[0].bar([0, 1], values, bottom=bottoms, color=color, label=labels[failure_type].replace("\n", " "))
        bottoms += np.asarray(values)
    axes[0].set_xticks([0, 1], ["Consensus hard", "Matched normal"])
    axes[0].set_ylabel("Failure composition (%)")
    axes[0].set_ylim(0.0, 100.0)
    axes[0].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    x = np.arange(len(FAILURE_TYPES))
    width = 0.38
    hard_values = [float(lookup[("hard", item)]["per_1k_gt"]) for item in FAILURE_TYPES]
    normal_values = [float(lookup[("normal", item)]["per_1k_gt"]) for item in FAILURE_TYPES]
    axes[1].bar(x - width / 2.0, hard_values, width=width, color="#d62728", label="Consensus hard")
    axes[1].bar(x + width / 2.0, normal_values, width=width, color="#8c8c8c", label="Matched normal")
    axes[1].set_xticks(x, [labels[item] for item in FAILURE_TYPES], fontsize=8)
    axes[1].set_ylabel("Events / 1k GT frames")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("Future failure decomposition in consensus regions")
    figure.tight_layout()
    paths = _save_figure(figure, output_root, "figure_b_failure_decomposition")
    plt.close(figure)
    return list(paths)


def _figure_consistency(output_root, scene_overlap, scene_consensus, scenes, tracker_names):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm

    scene_ids = [scene["scene_id"] for scene in scenes]
    overlap_lookup = {
        row["scene_id"]: float(row["hard_iou_mean"])
        for row in scene_overlap
        if float(row["grid_m"]) == 5.0 and row["hard_iou_mean"] != ""
    }
    consensus_lookup = {
        (row["scene_id"], row["future_tracker"]): float(row["miss_risk_ratio"])
        for row in scene_consensus
        if row["protocol"] == "consensus_top20"
        and float(row["grid_m"]) == 5.0
        and row["miss_risk_ratio"] != ""
    }
    matrix = np.full((len(scene_ids), len(tracker_names)), np.nan, dtype=np.float64)
    for row_index, scene_id in enumerate(scene_ids):
        for col_index, tracker in enumerate(tracker_names):
            if (scene_id, tracker) in consensus_lookup:
                matrix[row_index, col_index] = consensus_lookup[(scene_id, tracker)]
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), gridspec_kw={"width_ratios": [0.8, 2.2]})
    overlap_values = [overlap_lookup.get(scene_id, np.nan) for scene_id in scene_ids]
    axes[0].barh(np.arange(len(scene_ids)), overlap_values, color="#4c78a8")
    axes[0].axvline(0.111, color="#e45756", linestyle="--", linewidth=1.0, label="Random 20% expectation")
    axes[0].set_yticks(np.arange(len(scene_ids)), scene_ids, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Mean pairwise hard IoU")
    axes[0].legend(fontsize=7)
    image = axes[1].imshow(
        np.ma.masked_invalid(matrix),
        aspect="auto",
        cmap="RdYlBu_r",
        norm=LogNorm(vmin=0.5, vmax=max(2.0, float(np.nanmax(matrix)))),
    )
    axes[1].set_xticks(np.arange(len(tracker_names)), tracker_names, rotation=25, ha="right", fontsize=8)
    axes[1].set_yticks(np.arange(len(scene_ids)), scene_ids, fontsize=8)
    axes[1].set_title("Consensus-hard future miss risk ratio")
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            if math.isfinite(float(matrix[row_index, col_index])):
                axes[1].text(col_index, row_index, "%.2f" % matrix[row_index, col_index], ha="center", va="center", fontsize=7)
    figure.colorbar(image, ax=axes[1], fraction=0.04, pad=0.03, label="Hard / normal RR")
    figure.tight_layout()
    paths = _save_figure(figure, output_root, "figure_c_cross_tracker_consistency")
    plt.close(figure)
    return list(paths)


def _write_scene_table(path, self_scene, consensus_scene):
    self_lookup = {
        (row["scene_id"], row["future_tracker"]): row
        for row in self_scene
        if float(row["grid_m"]) == 5.0
    }
    consensus_lookup = {
        (row["scene_id"], row["future_tracker"]): row
        for row in consensus_scene
        if float(row["grid_m"]) == 5.0 and row["protocol"] == "consensus_top20"
    }
    keys = sorted(set(self_lookup) | set(consensus_lookup))
    lines = [
        "# Per-scene results",
        "",
        "| Scene | Tracker | Self hard/normal miss | Self miss RR | Self identity RR | Consensus hard/normal miss | Consensus miss RR | Consensus identity RR |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scene_id, tracker in keys:
        self_row = self_lookup.get((scene_id, tracker), {})
        consensus_row = consensus_lookup.get((scene_id, tracker), {})
        lines.append(
            "| %s | %s | %s/%s | %s | %s | %s/%s | %s | %s |"
            % (
                scene_id,
                tracker,
                _fmt(float(self_row["hard_miss_rate"]) * 100.0 if self_row.get("hard_miss_rate") != "" else None),
                _fmt(float(self_row["normal_miss_rate"]) * 100.0 if self_row.get("normal_miss_rate") != "" else None),
                _fmt(self_row.get("miss_risk_ratio")),
                _fmt(self_row.get("identity_risk_ratio")),
                _fmt(float(consensus_row["hard_miss_rate"]) * 100.0 if consensus_row.get("hard_miss_rate") != "" else None),
                _fmt(float(consensus_row["normal_miss_rate"]) * 100.0 if consensus_row.get("normal_miss_rate") != "" else None),
                _fmt(consensus_row.get("miss_risk_ratio")),
                _fmt(consensus_row.get("identity_risk_ratio")),
            )
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _lookup_condition(condition_rows, name):
    for row in condition_rows:
        if row["tracker"] == "ALL" and row["condition"] == name:
            return row
    return {}


def _write_protocol(path, config, tracker_names, targets):
    lines = [
        "# Protocol and observability",
        "",
        "## Causal split",
        "",
        "- 只把官方val中的sequence作为future target",
        "- 每个target只使用同一固定场景中时间戳更早的全部sequence构建历史风险图",
        "- target sequence的GT、检测错误和跟踪错误不参与hard region选择",
        "- 主设置为5m网格、historical miss risk、top20% hard cells",
        "- matched normal按距离传感器的范围和历史GT覆盖量一对一匹配",
        "",
        "## Trackers",
        "",
    ]
    lines.extend(["- %s" % name for name in tracker_names])
    lines.extend(
        [
            "",
            "## 可严格观测",
            "",
            "- GT到检测的miss、confidence、IoU和中心残差",
            "- GT到tracker输出的miss、ID切换和gap后重连",
            "- hard region的跨tracker重合和跨tracker未来预测",
            "- tracker输出与正确检测的post-update中心残差",
            "",
            "## 不能从统一输出严格观测",
            "",
            "- 五个tracker没有统一保存pre-association Kalman预测态",
            "- 五个tracker没有统一保存原生gate候选集、cost matrix和best-vs-second margin",
            "- GRAE-3DMOT、Fast-Poly和3DMOTFormer没有输出内部track存活或删除状态",
            "- 因此本实验不把几何代理量命名为严格gate failure，也不把gap后新ID直接断言为内部track死亡",
            "- tracking miss with good detection是motion、gate和association未分解的可观测上界",
            "- re-init after gap proxy只表示gap后输出ID改变",
            "",
            "## 诊断阈值",
            "",
            "- detection存在要求3DIoU不低于%.2f" % float(config["iou_threshold"]),
            "- localization良好要求3DIoU不低于%.2f且中心残差不超过%.1fm"
            % (float(config["localization_good_iou"]), float(config["localization_good_center_m"])),
            "- local density统计%.1fm内其他GT数" % float(config["local_density_radius_m"]),
            "- ambiguity proxy使用正确候选与最近竞争候选的中心距离margin",
            "",
            "## Target folds",
            "",
        ]
    )
    for scene_id, target_id, history_ids in targets:
        lines.append("- %s: %s -> %s" % (scene_id, ",".join(history_ids), target_id))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(
    path,
    scenes,
    tracker_names,
    self_rows,
    self_summary,
    cross_summary,
    overlap_summary,
    consensus_summary,
    condition_rows,
    failure_rows,
    vote_rows,
    figures,
):
    main_self = [row for row in self_summary if float(row["grid_m"]) == 5.0]
    fine_self = [row for row in self_summary if float(row["grid_m"]) == 2.0]
    main_overlap = [row for row in overlap_summary if float(row["grid_m"]) == 5.0]
    offdiag = [
        row
        for row in cross_summary
        if float(row["grid_m"]) == 5.0 and row["history_tracker"] != row["future_tracker"]
    ]
    consensus = [
        row
        for row in consensus_summary
        if float(row["grid_m"]) == 5.0 and row["protocol"] == "consensus_top20"
    ]
    mean_overlap = float(np.mean([float(row["hard_iou_mean"]) for row in main_overlap])) if main_overlap else float("nan")
    mean_offdiag_rr = float(np.mean([float(row["miss_risk_ratio"]) for row in offdiag])) if offdiag else float("nan")
    self_main_raw = [row for row in self_rows if float(row["grid_m"]) == 5.0]
    self_wins = sum(float(row["miss_risk_ratio"]) > 1.0 for row in self_main_raw if row["miss_risk_ratio"] != "")
    self_total = sum(row["miss_risk_ratio"] != "" for row in self_main_raw)
    self_identity_wins = sum(
        float(row["identity_risk_ratio"]) > 1.0 for row in main_self if row["identity_risk_ratio"] != ""
    )
    consensus_wins = sum(float(row["miss_risk_ratio"]) > 1.0 for row in consensus if row["miss_risk_ratio"] != "")
    detection = _lookup_condition(condition_rows, "detection_missing")
    localization = _lookup_condition(condition_rows, "weak_localization_given_detection")
    good_det_miss = _lookup_condition(condition_rows, "tracking_miss_given_good_detection")
    post_residual = _lookup_condition(condition_rows, "large_post_update_residual")
    direct_switch = _lookup_condition(condition_rows, "direct_id_switch")
    reinit = _lookup_condition(condition_rows, "reinit_after_gap_proxy")
    hard_failures = {
        row["failure_type"]: float(row["composition_percent"])
        for row in failure_rows
        if row["tracker"] == "ALL" and row["region"] == "hard"
    }
    largest = max(hard_failures, key=hard_failures.get) if hard_failures else ""
    shared_votes = [row for row in vote_rows if float(row["grid_m"]) == 5.0 and int(row["tracker_votes"]) >= 3]
    any_votes = [row for row in vote_rows if float(row["grid_m"]) == 5.0 and int(row["tracker_votes"]) >= 1]
    shared_cells = sum(int(row["cells"]) for row in shared_votes)
    any_cells = sum(int(row["cells"]) for row in any_votes)
    shared_fraction = float(shared_cells / any_cells) if any_cells else 0.0
    scene_persistence = "Go" if self_total and self_wins > self_total / 2 else "No-Go"
    cross_consistency = "Go（固定传感器+共同detector范围内）" if mean_overlap > 0.111 and mean_offdiag_rr > 1.0 else "No-Go"
    detection_judgment = "值得" if detection and float(detection.get("risk_ratio", 0.0) or 0.0) > 1.25 else "暂不值得"
    association_score = max(
        float(good_det_miss.get("risk_ratio", 0.0) or 0.0),
        float(direct_switch.get("risk_ratio", 0.0) or 0.0),
    )
    association_judgment = "值得" if association_score > 1.25 else "暂不值得"
    track_judgment = "值得" if reinit and float(reinit.get("risk_ratio", 0.0) or 0.0) > 1.25 and hard_failures.get("reinit_after_gap_proxy", 0.0) >= 10.0 else "暂不值得"
    motion_evidence = float(post_residual.get("risk_ratio", 0.0) or 0.0)
    motion_judgment = "值得补充原生预测态后验证" if motion_evidence > 1.25 else "当前代理证据不足"
    failure_names = {
        "detection_missing": "detection missing",
        "localization_weakness": "localization weakness",
        "tracking_miss_good_detection": "good detection下的tracking miss",
        "direct_id_switch": "direct ID switch",
        "reinit_after_gap_proxy": "gap后新ID代理",
        "same_id_reconnect": "same-ID reconnect",
    }
    lines = [
        "# Scene diagnosis summary",
        "",
        "## 一句话结论",
        "",
        "多个tracker的历史高风险区域在future sequence中仍然更容易出错；跨tracker空间共识是否足够强以及后续应优先处理哪类错误，以本页下列实测数字为准。",
        "",
        "## 核心结果",
        "",
        "- 5m self-persistence中，%d/%d个tracker-scene fold的future miss风险比大于1" % (self_wins, self_total),
        "- pooled self-persistence miss RR均值在5m为%.2f、2m为%.2f"
        % (
            float(np.mean([float(row["miss_risk_ratio"]) for row in main_self])),
            float(np.mean([float(row["miss_risk_ratio"]) for row in fine_self])) if fine_self else float("nan"),
        ),
        "- 五个tracker两两historical hard-region IoU均值为%.3f；独立随机top20%%的期望IoU约为0.111" % mean_overlap,
        "- off-diagonal History(A)->Future(B)的miss风险比均值为%.2f" % mean_offdiag_rr,
        "- consensus top20%%对%d/%d个tracker的pooled future miss风险比大于1" % (consensus_wins, len(consensus)),
        "- 至少3/5个tracker共同判hard的cell占所有被任一tracker判hard cell的%.1f%%" % (shared_fraction * 100.0),
        "- consensus hard区域failure composition最大项是%s，占%.1f%%" % (failure_names.get(largest, largest), hard_failures.get(largest, 0.0)),
        "- self-persistence的identity风险只在%d/%d个tracker上升，identity failure比miss更tracker-specific" % (self_identity_wins, len(main_self)),
        "- 五个tracker共享同一个CenterPoint detector，因此跨tracker共识证明风险不只属于某个tracker，但尚不能证明它与detector选择无关",
        "",
        "## 各tracker self-persistence",
        "",
        "| Tracker | Hard miss | Normal miss | Miss RR | Identity RR |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(main_self, key=lambda item: item["future_tracker"]):
        lines.append(
            "| %s | %.2f%% | %.2f%% | %s | %s |"
            % (
                row["future_tracker"],
                float(row["hard_miss_rate"]) * 100.0,
                float(row["normal_miss_rate"]) * 100.0,
                _fmt(row["miss_risk_ratio"]),
                _fmt(row["identity_risk_ratio"]),
            )
        )
    lines.extend(
        [
            "",
            "## 可观测failure condition",
            "",
            "| Condition | Hard | Normal | RR |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in condition_rows:
        if row["tracker"] != "ALL":
            continue
        hard_rate = float(row["hard_rate"]) * 100.0 if row["hard_rate"] != "" else 0.0
        normal_rate = float(row["normal_rate"]) * 100.0 if row["normal_rate"] != "" else 0.0
        lines.append("| %s | %.2f%% | %.2f%% | %s |" % (row["condition"], hard_rate, normal_rate, _fmt(row["risk_ratio"])))
    lines.extend(
        [
            "",
            "## Go / No-Go",
            "",
            "- Scene-level spatial persistence：%s" % scene_persistence,
            "- Cross-tracker scene consistency：%s" % cross_consistency,
            "- Detection reliability memory：%s" % detection_judgment,
            "- Motion-aware memory：%s" % motion_judgment,
            "- Association-aware memory：%s" % association_judgment,
            "- Track-management memory：%s" % track_judgment,
            "",
            "## 解释边界",
            "",
            "五个tracker的统一保存结果没有原生pre-association预测态、gate候选和cost matrix，因此本轮不能严格把good detection下的tracking miss继续拆成motion error、gate rejection和wrong match。post-update residual、candidate margin和gap后新ID都明确标为proxy。不要据此宣称已经测得原生gate failure。",
            "",
            "## Figures",
            "",
        ]
    )
    lines.extend(["- `%s`" % item for item in figures if item.endswith(".png")])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config):
    output_root = Path(config["project"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    scenes, metadata, warnings = build_scenes(config)
    tracker_names = [item["name"] for item in config["trackers"]]
    all_sequence_ids = [item["sequence_id"] for scene in scenes for item in scene["sequences"]]
    prediction_audit = []
    for tracker in config["trackers"]:
        folder = _path(config["_root"], tracker["prediction_root"])
        missing = [sequence_id for sequence_id in all_sequence_ids if not (folder / (sequence_id + ".jsonl")).is_file()]
        prediction_audit.append(
            {
                "tracker": tracker["name"],
                "prediction_root": str(folder),
                "required_sequences": len(all_sequence_ids),
                "missing_sequences": len(missing),
                "missing_ids": ";".join(missing),
            }
        )
        if missing:
            raise SystemExit("%s缺少%d个sequence预测" % (tracker["name"], len(missing)))
    _write_csv(output_root / "prediction_audit.csv", prediction_audit)
    _write_csv(output_root / "scene_metadata.csv", metadata)
    events = collect_events(config, scenes)
    grouped = _event_index(events)
    self_rows, cross_rows, overlap_rows, consensus_rows, vote_rows, stores, figure_data = _run_folds(
        scenes, grouped, config
    )
    self_summary = _aggregate_risk(self_rows, ["future_tracker", "grid_m"])
    self_scene = _aggregate_risk(self_rows, ["scene_id", "future_tracker", "grid_m"])
    cross_summary = _aggregate_risk(cross_rows, ["history_tracker", "future_tracker", "grid_m"])
    consensus_summary = _aggregate_risk(consensus_rows, ["protocol", "future_tracker", "grid_m"])
    consensus_scene = _aggregate_risk(consensus_rows, ["scene_id", "protocol", "future_tracker", "grid_m"])
    overlap_summary = _aggregate_overlap(overlap_rows, ["left_tracker", "right_tracker", "grid_m"])
    overlap_scene = _aggregate_overlap(overlap_rows, ["scene_id", "grid_m"])
    failure_rows = _failure_rows(stores, tracker_names, False)
    failure_scene = _failure_rows(stores, tracker_names, True)
    condition_rows = _condition_rows(stores, tracker_names, False)
    condition_scene = _condition_rows(stores, tracker_names, True)
    gap_rows = _gap_rows(stores, tracker_names, False)
    gap_scene = _gap_rows(stores, tracker_names, True)
    metric_rows = _metric_rows(stores, tracker_names, False)
    metric_scene = _metric_rows(stores, tracker_names, True)
    observability_rows = [
        {"requested_failure": "detection_missing", "status": "exact", "reported_as": "detection_missing"},
        {"requested_failure": "localization_error", "status": "exact_output_level", "reported_as": "detection_iou_and_center_residual"},
        {"requested_failure": "motion_prediction_error", "status": "unavailable", "reported_as": "post_update_residual_proxy_only"},
        {"requested_failure": "association_gate_failure", "status": "unavailable", "reported_as": "tracking_miss_with_good_detection_upper_bound"},
        {"requested_failure": "association_ambiguity", "status": "proxy", "reported_as": "center_margin_and_local_density"},
        {"requested_failure": "wrong_association", "status": "output_level", "reported_as": "direct_id_switch"},
        {"requested_failure": "early_track_termination", "status": "proxy", "reported_as": "reinit_after_gap_proxy"},
    ]
    outputs = {
        "self_persistence_folds.csv": self_rows,
        "self_persistence.csv": self_summary,
        "self_persistence_by_scene.csv": self_scene,
        "cross_prediction_folds.csv": cross_rows,
        "cross_prediction_matrix.csv": cross_summary,
        "hard_overlap_folds.csv": overlap_rows,
        "hard_overlap.csv": overlap_summary,
        "hard_overlap_by_scene.csv": overlap_scene,
        "consensus_future_folds.csv": consensus_rows,
        "consensus_future.csv": consensus_summary,
        "consensus_by_scene.csv": consensus_scene,
        "consensus_votes.csv": vote_rows,
        "failure_decomposition.csv": failure_rows,
        "failure_decomposition_by_scene.csv": failure_scene,
        "condition_rates.csv": condition_rows,
        "condition_rates_by_scene.csv": condition_scene,
        "detection_gap.csv": gap_rows,
        "detection_gap_by_scene.csv": gap_scene,
        "continuous_metrics.csv": metric_rows,
        "continuous_metrics_by_scene.csv": metric_scene,
        "failure_observability.csv": observability_rows,
    }
    for name, rows in outputs.items():
        _write_csv(output_root / name, rows)
    _write_main_tables(
        output_root,
        self_summary,
        cross_summary,
        consensus_summary,
        overlap_summary,
        condition_rows,
        metric_rows,
    )
    _write_scene_table(output_root / "scene_table.md", self_scene, consensus_scene)
    targets = [
        (scene["scene_id"], target_id, history_ids)
        for scene in scenes
        for _, target_id, history_ids in _future_targets(scene)
    ]
    _write_protocol(output_root / "protocol.md", config, tracker_names, targets)
    figures = []
    figures.extend(_figure_maps(output_root, figure_data, tracker_names, config))
    figures.extend(_figure_failure(output_root, failure_rows))
    figures.extend(_figure_consistency(output_root, overlap_scene, consensus_scene, scenes, tracker_names))
    _write_summary(
        output_root / "summary.md",
        scenes,
        tracker_names,
        self_rows,
        self_summary,
        cross_summary,
        overlap_summary,
        consensus_summary,
        condition_rows,
        failure_rows,
        vote_rows,
        figures,
    )
    write_json(
        output_root / "manifest.json",
        {
            "map_role": MAP_ROLE,
            "experiment": "scene_diagnosis",
            "scenes": [scene["scene_id"] for scene in scenes],
            "trackers": tracker_names,
            "target_folds": len(targets),
            "grid_sizes": config["grid_sizes"],
            "hard_fraction": config["hard_fraction"],
            "warnings": warnings,
            "observability_limit": "原生motion prediction、gate候选和内部track death没有统一日志",
            "figures": figures,
            "tables": sorted(outputs),
        },
    )
    print("scene_diagnosis完成 target fold %d event %d" % (len(targets), len(events)), flush=True)
