import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from inframot3d.analysis.scene_diagnosis import _rank_normalize, _risk_map, _select_regions
from inframot3d.analysis.scene_difficulty import _bin_index, _grid_shape, _keep_vehicle, _match, build_scenes
from inframot3d.config import load_config
from inframot3d.evaluation import UnifiedMOTEvaluator
from inframot3d.evaluation.hota import evaluate_hota
from inframot3d.evaluation.protocols.v2xseq import V2XSeqProtocol
from inframot3d.io import read_json, read_jsonl, write_json, write_jsonl
from inframot3d.tracking import create_tracker
from inframot3d.tracking.grae_adapter import GraeTracker, build_model, load_checkpoint


REASONS = (
    "score_filtering",
    "track_already_dead",
    "gate_rejection",
    "wrong_assignment",
    "output_suppression",
    "other",
)


def _path(root, value):
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path


def _class_thresholds(config):
    tracker = config["tracker"]
    relative = tracker.get("score_thresholds_file")
    if not relative:
        return
    payload = yaml.safe_load(_path(config["_root"], relative).read_text(encoding="utf-8"))
    tracker["score_thresholds"] = {str(key): float(value) for key, value in payload["score_thresholds"].items()}


def _indexed_objects(row, protocol):
    output = []
    for index, item in enumerate(row.get("objects", [])):
        if _keep_vehicle(protocol, item):
            output.append((int(index), item))
    return output


def _pair(gt_items, indexed_items, threshold, gate):
    items = [item for _, item in indexed_items]
    pairs = _match(
        [item["box"] for item in gt_items],
        [item["box"] for item in items],
        threshold,
        gate,
    )
    return {gt_index: (indexed_items[item_index][0], items[item_index], iou) for gt_index, item_index, iou in pairs}


def _center_residual(left, right):
    return float(math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1])))


def _empty_counts(config):
    shape = _grid_shape(config["bev"], float(config["grid_m"]))
    return {key: np.zeros(shape, dtype=np.float64) for key in ("gt", "miss", "idsw", "frag")}


def _add_counts(target, source):
    for key in target:
        target[key] += source[key]


def _sequence_counts(config, protocol, sequence_id, prediction_root):
    converted = config["project"]["converted_root"]
    gt_rows = list(read_jsonl(converted / "sequences" / (sequence_id + ".jsonl")))
    prediction_rows = {int(row["timestamp"]): row for row in read_jsonl(prediction_root / (sequence_id + ".jsonl"))}
    counts = _empty_counts(config)
    shape = counts["gt"].shape
    for gt_row in gt_rows:
        gt_items = protocol.filter_gt(gt_row["objects"])
        prediction = prediction_rows.get(int(gt_row["timestamp"]), {"objects": []})
        indexed = _indexed_objects(prediction, protocol)
        matches = _pair(gt_items, indexed, config["match_iou"], config["match_center_gate_m"])
        for gt_index, gt_item in enumerate(gt_items):
            index = _bin_index(
                float(gt_item["box"][0]),
                float(gt_item["box"][1]),
                config["bev"],
                float(config["grid_m"]),
                shape,
            )
            if index is None:
                continue
            counts["gt"][index] += 1
            counts["miss"][index] += int(gt_index not in matches)
    return counts


def _historical_regions(config, scenes, tracker_specs, protocol):
    root = config["_root"]
    cache = {}
    for scene in scenes:
        for sequence in scene["sequences"]:
            sequence_id = sequence["sequence_id"]
            for spec in tracker_specs:
                key = (sequence_id, spec["name"])
                cache[key] = _sequence_counts(
                    config,
                    protocol,
                    sequence_id,
                    _path(root, spec["prediction_root"]),
                )
    regions = {}
    sequence_scene = {}
    for scene in scenes:
        for sequence in scene["sequences"]:
            sequence_scene[sequence["sequence_id"]] = scene["scene_id"]
        for target_index, target in enumerate(scene["sequences"]):
            if target["split"] != "val" or target_index == 0:
                continue
            history_ids = [item["sequence_id"] for item in scene["sequences"][:target_index]]
            risks = []
            reference = None
            for spec in tracker_specs:
                current = _empty_counts(config)
                for sequence_id in history_ids:
                    _add_counts(current, cache[(sequence_id, spec["name"])])
                if reference is None:
                    reference = current
                risks.append(_rank_normalize(_risk_map(current, config["prior_strength"], config["min_observations"])))
            stacked = np.stack(risks)
            valid = np.sum(np.isfinite(stacked), axis=0)
            consensus = np.divide(
                np.nansum(stacked, axis=0),
                valid,
                out=np.full(valid.shape, np.nan, dtype=np.float64),
                where=valid > 0,
            )
            hard, normal = _select_regions(
                consensus,
                reference,
                config["bev"],
                float(config["grid_m"]),
                config["hard_fraction"],
            )
            if hard and normal:
                regions[(scene["scene_id"], target["sequence_id"])] = {
                    "hard": set(hard),
                    "normal": set(normal),
                }
    return regions, sequence_scene


def _event_region(config, regions, scene_id, sequence_id, box):
    current = regions.get((scene_id, sequence_id))
    if current is None:
        return "unknown"
    shape = _grid_shape(config["bev"], float(config["grid_m"]))
    index = _bin_index(float(box[0]), float(box[1]), config["bev"], float(config["grid_m"]), shape)
    if index in current["hard"]:
        return "hard"
    if index in current["normal"]:
        return "normal"
    return "other"


def _oracle_predictions(config, protocol, sequence_ids, output_root, score_threshold):
    detection_root = _path(config["_root"], config["detection_root"])
    converted = config["project"]["converted_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    for sequence_id in sequence_ids:
        gt_rows = list(read_jsonl(converted / "sequences" / (sequence_id + ".jsonl")))
        det_rows = list(read_jsonl(detection_root / (sequence_id + ".jsonl")))
        gt_id_map = {}
        next_gt_id = 1
        rows = []
        for gt_row, det_row in zip(gt_rows, det_rows):
            gt_items = protocol.filter_gt(gt_row["objects"])
            indexed = [
                item
                for item in _indexed_objects(det_row, protocol)
                if float(item[1].get("score", 1.0)) >= float(score_threshold)
            ]
            matches = _pair(gt_items, indexed, config["match_iou"], config["match_center_gate_m"])
            det_to_gt = {raw_index: gt_index for gt_index, (raw_index, _, _) in matches.items()}
            objects = []
            for raw_index, item in enumerate(det_row["objects"]):
                if raw_index in det_to_gt:
                    gt_item = gt_items[det_to_gt[raw_index]]
                    source_id = str(gt_item["source_track_id"])
                    if source_id not in gt_id_map:
                        gt_id_map[source_id] = next_gt_id
                        next_gt_id += 1
                    track_id = gt_id_map[source_id]
                else:
                    track_id = 10000000 + int(det_row["frame_index"]) * 1000 + int(raw_index)
                objects.append(
                    {
                        "class_name": item["class_name"],
                        "track_id": int(track_id),
                        "score": float(item.get("score", 1.0)),
                        "box": [float(value) for value in item["box"]],
                    }
                )
            rows.append(
                {
                    "sequence_id": det_row["sequence_id"],
                    "frame_index": det_row["frame_index"],
                    "frame_id": det_row["frame_id"],
                    "timestamp": det_row["timestamp"],
                    "objects": objects,
                }
            )
        write_jsonl(output_root / (sequence_id + ".jsonl"), rows)


def _best_checkpoint(output_root):
    best = Path(output_root) / "ckpt" / "checkpoint-best.pth"
    if best.is_file():
        return best
    checkpoints = list((Path(output_root) / "ckpt").glob("checkpoint-epoch*.pth"))
    if not checkpoints:
        raise FileNotFoundError("未找到GRAE checkpoint")
    return max(checkpoints, key=lambda path: int(path.stem.replace("checkpoint-epoch", "")))


def _prepare_trackers(config, tracker_specs):
    prepared = {}
    for spec in tracker_specs:
        tracker_config = load_config(_path(config["_root"], spec["config"]))
        if spec["kind"] == "classic":
            _class_thresholds(tracker_config)
            prepared[spec["name"]] = {"config": tracker_config, "tracker": None}
            continue
        device = "cuda"
        model = build_model(
            _path(config["_root"], tracker_config["project"]["grae_root"]),
            tracker_config["model"]["in_channels"],
            tracker_config["model"]["layers"],
            tracker_config["num_classes"],
            device,
        )
        load_checkpoint(model, _best_checkpoint(tracker_config["project"]["output_root"]), device)
        thresholds = yaml.safe_load(
            _path(config["_root"], tracker_config["tracker"]["birth_thresholds_file"]).read_text(encoding="utf-8")
        )["score_thresholds"]
        tracker = GraeTracker(
            model,
            tracker_config["classes"],
            thresholds,
            association_alpha=tracker_config["tracker"]["association_alpha"],
            age=tracker_config["tracker"]["age"],
            score_floor=tracker_config["tracker"].get("score_floor", 0.1),
        )
        prepared[spec["name"]] = {"config": tracker_config, "tracker": tracker}
    return prepared


def _sequence_tracker(spec, prepared, debug):
    current = prepared[spec["name"]]
    if spec["kind"] == "classic":
        tracker = create_tracker(current["config"]["tracker"], current["config"]["_root"])
    else:
        tracker = current["tracker"]
        tracker.reset()
    tracker.enable_debug(debug)
    return tracker


def _equivalent_rows(left, right, tracker_name, sequence_id):
    if len(left) != len(right):
        raise AssertionError("%s序列%s帧数变化" % (tracker_name, sequence_id))
    for left_row, right_row in zip(left, right):
        if int(left_row["timestamp"]) != int(right_row["timestamp"]):
            raise AssertionError("%s序列%s时间戳变化" % (tracker_name, sequence_id))
        left_objects = left_row["objects"]
        right_objects = right_row["objects"]
        if len(left_objects) != len(right_objects):
            raise AssertionError("%s序列%s输出数量变化" % (tracker_name, sequence_id))
        for left_item, right_item in zip(left_objects, right_objects):
            if int(left_item["track_id"]) != int(right_item["track_id"]):
                raise AssertionError("%s序列%s track id变化" % (tracker_name, sequence_id))
            if left_item["class_name"] != right_item["class_name"]:
                raise AssertionError("%s序列%s类别变化" % (tracker_name, sequence_id))
            if not np.allclose(left_item["box"], right_item["box"], atol=1.0e-6, rtol=0.0):
                raise AssertionError("%s序列%s box变化" % (tracker_name, sequence_id))


def _filtered_indices(snapshot):
    values = set(int(value) for value in snapshot.get("filtered_detection_indices", []))
    for group in snapshot.get("groups", []):
        values.update(int(value) for value in group.get("filtered_detection_indices", []))
    return values


def _group_for_detection(snapshot, detection_index):
    for group in snapshot.get("groups", []):
        if any(int(item["input_index"]) == int(detection_index) for item in group.get("candidate_detections", [])):
            return group
    return None


def _classify_case(snapshot, detection_index, expected_track_id):
    detection_index = int(detection_index)
    filtered = _filtered_indices(snapshot)
    if detection_index in filtered:
        return "score_filtering", {"stage": "input_filter"}
    group = _group_for_detection(snapshot, detection_index)
    groups = snapshot.get("groups", [])
    pre_tracks = [item for current in groups for item in current.get("pre_tracks", [])]
    pre_ids = {int(item["track_id"]) for item in pre_tracks}
    if expected_track_id is not None and int(expected_track_id) not in pre_ids:
        return "track_already_dead", {"expected_track_id": int(expected_track_id)}
    if group is None:
        return "other", {"stage": "missing_candidate"}
    candidates = group.get("candidate_detections", [])
    row = next(index for index, item in enumerate(candidates) if int(item["input_index"]) == detection_index)
    assignments = group.get("assignments", [])
    detection_assignments = [item for item in assignments if int(item["input_index"]) == detection_index]
    output_ids = {int(value) for value in snapshot.get("output_track_ids", [])}
    if expected_track_id is None:
        if detection_index in {int(value) for value in group.get("birth_suppressed_indices", [])}:
            return "score_filtering", {"stage": "birth_threshold"}
        created = [item for item in group.get("created", []) if int(item["input_index"]) == detection_index]
        if created:
            track_id = int(created[0]["track_id"])
            if track_id not in output_ids:
                return "output_suppression", {"created_track_id": track_id}
            return "other", {"stage": "new_track_output_geometry", "created_track_id": track_id}
        if detection_assignments:
            return "wrong_assignment", {"assigned_track_id": int(detection_assignments[0]["track_id"])}
        return "other", {"stage": "no_previous_track"}
    expected_track_id = int(expected_track_id)
    group_tracks = group.get("pre_tracks", [])
    column = next(
        (index for index, item in enumerate(group_tracks) if int(item["track_id"]) == expected_track_id),
        None,
    )
    if column is None:
        return "gate_rejection", {"stage": "class_partition", "expected_track_id": expected_track_id}
    association = group.get("association", {})
    gate_mask = association.get("gate_mask", [])
    gate_value = bool(gate_mask[row][column]) if row < len(gate_mask) and column < len(gate_mask[row]) else False
    if not gate_value:
        return "gate_rejection", {
            "stage": "numeric_gate",
            "expected_track_id": expected_track_id,
            "matrix_row": row,
            "matrix_column": column,
        }
    assigned = next((item for item in detection_assignments if int(item["track_id"]) == expected_track_id), None)
    if assigned is None:
        return "wrong_assignment", {
            "expected_track_id": expected_track_id,
            "assigned_track_ids": [int(item["track_id"]) for item in detection_assignments],
        }
    if expected_track_id not in output_ids:
        return "output_suppression", {"expected_track_id": expected_track_id}
    return "other", {"stage": "matched_output_geometry", "expected_track_id": expected_track_id}


def _tracker_timestamp(spec, timestamp):
    return float(timestamp) / 1.0e6 if spec["kind"] == "grae" else int(timestamp)


def _compact_internal(snapshot, detection_index):
    group = _group_for_detection(snapshot, detection_index)
    if group is None:
        return {
            "input_detection": next(
                (
                    item
                    for item in snapshot.get("input_detections", [])
                    if int(item["input_index"]) == int(detection_index)
                ),
                None,
            ),
            "filtered_detection_indices": sorted(_filtered_indices(snapshot)),
            "pre_tracks": [item for current in snapshot.get("groups", []) for item in current.get("pre_tracks", [])],
            "output_track_ids": snapshot.get("output_track_ids", []),
        }
    candidates = group.get("candidate_detections", [])
    row = next(
        (index for index, item in enumerate(candidates) if int(item["input_index"]) == int(detection_index)),
        None,
    )
    association = group.get("association", {})
    compact_association = {}
    for key in ("affinity_matrix", "distance_matrix", "cost_matrix", "gate_mask"):
        matrix = association.get(key, [])
        compact_association[key.replace("_matrix", "_row")] = matrix[row] if row is not None and row < len(matrix) else []
    return {
        "class_name": group.get("class_name"),
        "candidate_detection": candidates[row] if row is not None else None,
        "candidate_count": len(candidates),
        "pre_tracks": group.get("pre_tracks", []),
        "association": compact_association,
        "assignments": group.get("assignments", []),
        "created": group.get("created", []),
        "birth_suppressed_indices": group.get("birth_suppressed_indices", []),
        "track_states": group.get("track_states", []),
        "output_track_ids": snapshot.get("output_track_ids", []),
    }


def _prediction_row(source, objects):
    return {
        "sequence_id": source["sequence_id"],
        "frame_index": source["frame_index"],
        "frame_id": source["frame_id"],
        "timestamp": source["timestamp"],
        "objects": objects,
    }


def _debug_replay(config, tracker_specs, prepared, protocol, sequence_ids, regions, sequence_scene, temporary):
    converted = Path(config["project"]["converted_root"])
    detection_root = _path(config["_root"], config["detection_root"])
    cases = []
    opportunities = Counter()
    replay_root = temporary / "debug_replay"
    debug_root = temporary / "debug"
    for spec in tracker_specs:
        tracker_name = spec["name"]
        score_threshold = float(read_json(_path(config["_root"], spec["evaluation"]))["best_score_threshold"])
        output_root = replay_root / tracker_name.lower().replace("-", "_")
        prediction_root = _path(config["_root"], spec["prediction_root"])
        tracker_cases = []
        for sequence_id in sequence_ids:
            tracker = _sequence_tracker(spec, prepared, True)
            gt_rows = list(read_jsonl(converted / "sequences" / (sequence_id + ".jsonl")))
            det_rows = list(read_jsonl(detection_root / (sequence_id + ".jsonl")))
            baseline_rows = list(read_jsonl(prediction_root / (sequence_id + ".jsonl")))
            if not (len(gt_rows) == len(det_rows) == len(baseline_rows)):
                raise ValueError("调试回放帧数不一致%s %s" % (tracker_name, sequence_id))
            gt_state = {}
            replay_rows = []
            scene_id = sequence_scene[sequence_id]
            for gt_row, det_row in zip(gt_rows, det_rows):
                inputs = [
                    {
                        "class_name": item["class_name"],
                        "score": float(item.get("score", 1.0)),
                        "box": item["box"],
                    }
                    for item in det_row["objects"]
                ]
                outputs = tracker.update(
                    inputs,
                    _tracker_timestamp(spec, det_row["timestamp"]),
                    **({"sample_token": det_row["frame_id"]} if spec["kind"] == "grae" else {}),
                )
                replay_rows.append(_prediction_row(det_row, outputs))
                gt_items = protocol.filter_gt(gt_row["objects"])
                indexed_detections = _indexed_objects(det_row, protocol)
                detection_matches = _pair(
                    gt_items,
                    indexed_detections,
                    config["match_iou"],
                    config["match_center_gate_m"],
                )
                indexed_raw_outputs = _indexed_objects({"objects": outputs}, protocol)
                indexed_outputs = [
                    item for item in indexed_raw_outputs if float(item[1].get("score", 1.0)) >= score_threshold
                ]
                raw_output_matches = _pair(
                    gt_items,
                    indexed_raw_outputs,
                    config["match_iou"],
                    config["match_center_gate_m"],
                )
                output_matches = _pair(
                    gt_items,
                    indexed_outputs,
                    config["match_iou"],
                    config["match_center_gate_m"],
                )
                for gt_index, gt_item in enumerate(gt_items):
                    detection_match = detection_matches.get(gt_index)
                    if detection_match is None:
                        continue
                    detection_index, detection, iou = detection_match
                    center_residual = _center_residual(gt_item["box"], detection["box"])
                    if iou < float(config["good_detection_iou"]):
                        continue
                    if center_residual > float(config["good_detection_center_m"]):
                        continue
                    source_id = str(gt_item["source_track_id"])
                    region = _event_region(config, regions, scene_id, sequence_id, gt_item["box"])
                    opportunities[(tracker_name, "all")] += 1
                    if region in {"hard", "normal"}:
                        opportunities[(tracker_name, region)] += 1
                    if gt_index in output_matches:
                        continue
                    expected_track_id = gt_state.get(source_id)
                    if gt_index in raw_output_matches:
                        raw_output = raw_output_matches[gt_index][1]
                        reason = "score_filtering"
                        evidence = {
                            "stage": "output_score_threshold",
                            "output_score": float(raw_output.get("score", 1.0)),
                            "score_threshold": score_threshold,
                        }
                    else:
                        reason, evidence = _classify_case(tracker.last_debug, detection_index, expected_track_id)
                    event = {
                        "tracker": tracker_name,
                        "scene_id": scene_id,
                        "sequence_id": sequence_id,
                        "frame_index": int(gt_row["frame_index"]),
                        "frame_id": gt_row["frame_id"],
                        "gt_id": source_id,
                        "region": region,
                        "reason": reason,
                        "expected_track_id": expected_track_id,
                        "detection_index": int(detection_index),
                        "detection_score": float(detection.get("score", 1.0)),
                        "detection_iou": float(iou),
                        "center_residual_m": center_residual,
                        "evidence": evidence,
                        "internal_state": _compact_internal(tracker.last_debug, detection_index),
                    }
                    cases.append(event)
                    tracker_cases.append(event)
                for gt_index, (_, output, _) in output_matches.items():
                    source_id = str(gt_items[gt_index]["source_track_id"])
                    gt_state[source_id] = int(output["track_id"])
            _equivalent_rows(replay_rows, baseline_rows, tracker_name, sequence_id)
            write_jsonl(output_root / (sequence_id + ".jsonl"), replay_rows)
            print("完成debug回放%s %s" % (tracker_name, sequence_id), flush=True)
        write_jsonl(debug_root / (tracker_name.lower().replace("-", "_") + ".jsonl"), tracker_cases)
    return cases, opportunities


def _good_detection_opportunities(config, tracker_specs, protocol, sequence_ids, regions, sequence_scene):
    converted = Path(config["project"]["converted_root"])
    detection_root = _path(config["_root"], config["detection_root"])
    opportunities = Counter()
    for sequence_id in sequence_ids:
        gt_rows = list(read_jsonl(converted / "sequences" / (sequence_id + ".jsonl")))
        det_rows = list(read_jsonl(detection_root / (sequence_id + ".jsonl")))
        scene_id = sequence_scene[sequence_id]
        for gt_row, det_row in zip(gt_rows, det_rows):
            gt_items = protocol.filter_gt(gt_row["objects"])
            matches = _pair(
                gt_items,
                _indexed_objects(det_row, protocol),
                config["match_iou"],
                config["match_center_gate_m"],
            )
            for gt_index, gt_item in enumerate(gt_items):
                match = matches.get(gt_index)
                if match is None:
                    continue
                _, detection, iou = match
                if iou < float(config["good_detection_iou"]):
                    continue
                if _center_residual(gt_item["box"], detection["box"]) > float(config["good_detection_center_m"]):
                    continue
                region = _event_region(config, regions, scene_id, sequence_id, gt_item["box"])
                for spec in tracker_specs:
                    opportunities[(spec["name"], "all")] += 1
                    if region in {"hard", "normal"}:
                        opportunities[(spec["name"], region)] += 1
    return opportunities


def _prediction_complete(root, sequence_ids):
    return all((Path(root) / (sequence_id + ".jsonl")).is_file() for sequence_id in sequence_ids)


def _gt_predictions(config, tracker_specs, prepared, sequence_ids, temporary):
    converted = Path(config["project"]["converted_root"])
    prediction_roots = {}
    for spec in tracker_specs:
        output_root = temporary / "predictions" / (spec["name"].lower().replace("-", "_") + "_gt")
        prediction_roots[spec["name"]] = output_root
        if _prediction_complete(output_root, sequence_ids):
            print("复用GT Detection%s" % spec["name"], flush=True)
            continue
        for sequence_id in sequence_ids:
            tracker = _sequence_tracker(spec, prepared, False)
            gt_rows = list(read_jsonl(converted / "sequences" / (sequence_id + ".jsonl")))
            rows = []
            for gt_row in gt_rows:
                inputs = [
                    {"class_name": item["class_name"], "score": 1.0, "box": item["box"]}
                    for item in gt_row["objects"]
                ]
                outputs = tracker.update(
                    inputs,
                    _tracker_timestamp(spec, gt_row["timestamp"]),
                    **({"sample_token": gt_row["frame_id"]} if spec["kind"] == "grae" else {}),
                )
                rows.append(_prediction_row(gt_row, outputs))
            write_jsonl(output_root / (sequence_id + ".jsonl"), rows)
            print("完成GT Detection%s %s" % (spec["name"], sequence_id), flush=True)
    return prediction_roots


def _stabilized_predictions(prediction_root, output_root, sequence_ids):
    if _prediction_complete(output_root, sequence_ids):
        return output_root
    for sequence_id in sequence_ids:
        rows = []
        for row in read_jsonl(Path(prediction_root) / (sequence_id + ".jsonl")):
            current = dict(row)
            objects = []
            for item in row["objects"]:
                copied = dict(item)
                box = [float(value) for value in item["box"]]
                box[3] += 1.0e-6
                copied["box"] = box
                objects.append(copied)
            current["objects"] = objects
            rows.append(current)
        write_jsonl(Path(output_root) / (sequence_id + ".jsonl"), rows)
    return output_root


def _evaluate_one(config, evaluator, prediction_root, output_dir, sequence_ids, threshold, stabilize=False):
    official_root = prediction_root
    if stabilize:
        official_root = _stabilized_predictions(
            prediction_root,
            Path(output_dir).parents[1] / "predictions" / (Path(output_dir).name + "_eval"),
            sequence_ids,
        )
    metrics = evaluator.evaluate(
        config,
        official_root,
        output_dir,
        split="val",
        score_threshold=float(threshold),
    )
    metrics.update(evaluate_hota(config, prediction_root, sequence_ids, evaluator.protocol, float(threshold)))
    return metrics


def _evaluate_variants(config, tracker_specs, sequence_ids, oracle_roots, gt_roots, temporary):
    evaluator = UnifiedMOTEvaluator(config["_root"], "v2xseq")
    rows = []
    for spec in tracker_specs:
        tracker_config = load_config(_path(config["_root"], spec["config"]))
        baseline = read_json(_path(config["_root"], spec["evaluation"]))
        threshold = float(baseline["best_score_threshold"])
        baseline = dict(baseline)
        baseline.update(
            evaluate_hota(
                tracker_config,
                _path(config["_root"], spec["prediction_root"]),
                sequence_ids,
                evaluator.protocol,
                threshold,
            )
        )
        oracle = _evaluate_one(
            tracker_config,
            evaluator,
            oracle_roots[spec["name"]],
            temporary / "evaluation" / (spec["name"].lower().replace("-", "_") + "_oracle"),
            sequence_ids,
            threshold,
        )
        gt_detection = _evaluate_one(
            tracker_config,
            evaluator,
            gt_roots[spec["name"]],
            temporary / "evaluation" / (spec["name"].lower().replace("-", "_") + "_gt"),
            sequence_ids,
            threshold,
            stabilize=True,
        )
        for variant, metrics in (
            ("Baseline", baseline),
            ("Oracle Association", oracle),
            ("GT Detection", gt_detection),
        ):
            rows.append({"tracker": spec["name"], "variant": variant, **metrics})
        print("完成上限评估%s" % spec["name"], flush=True)
    return rows


def _case_counts(cases):
    counts = Counter()
    for item in cases:
        counts[(item["tracker"], item["region"], item["reason"])] += 1
        counts[(item["tracker"], "all", item["reason"])] += 1
        counts[("Pooled", item["region"], item["reason"])] += 1
        counts[("Pooled", "all", item["reason"])] += 1
    return counts


def _reason_label(reason):
    return {
        "score_filtering": "Score filtering",
        "track_already_dead": "Track absent",
        "gate_rejection": "Gate rejection",
        "wrong_assignment": "Wrong assignment",
        "output_suppression": "Output suppression",
        "other": "Other",
    }[reason]


def _plot_composition(output_root, tracker_specs, counts):
    import matplotlib.pyplot as plt

    names = [spec["name"] for spec in tracker_specs] + ["Pooled"]
    colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B", "#B279A2"]
    figure, axis = plt.subplots(figsize=(10.5, 4.8))
    left = np.zeros(len(names), dtype=np.float64)
    for reason, color in zip(REASONS, colors):
        values = []
        for name in names:
            total = sum(counts[(name, "all", current)] for current in REASONS)
            values.append(100.0 * counts[(name, "all", reason)] / max(total, 1))
        axis.barh(names, values, left=left, label=_reason_label(reason), color=color)
        left += np.asarray(values)
    for index, value in enumerate(left):
        axis.text(100.8, index, "n=%d" % sum(counts[(names[index], "all", reason)] for reason in REASONS), va="center")
    axis.set_xlim(0, 112)
    axis.set_xlabel("Share of good-detection tracker misses (%)")
    axis.set_title("Why does the tracker miss when a good detection exists?")
    axis.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_root / "figure_1_good_detection_miss.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_oracles(output_root, tracker_specs, metrics_rows):
    import matplotlib.pyplot as plt

    metric_names = ("HOTA", "AssA", "IDF1", "MOTA")
    variants = ("Baseline", "Oracle Association", "GT Detection")
    colors = ("#9AA0A6", "#4C78A8", "#F58518")
    figure, axes = plt.subplots(1, len(tracker_specs), figsize=(12.5, 4.8), sharey=True)
    lookup = {(row["tracker"], row["variant"]): row for row in metrics_rows}
    x = np.arange(len(metric_names))
    width = 0.25
    for axis, spec in zip(axes, tracker_specs):
        for variant_index, (variant, color) in enumerate(zip(variants, colors)):
            row = lookup[(spec["name"], variant)]
            values = [100.0 * float(row[name]) for name in metric_names]
            axis.bar(x + (variant_index - 1) * width, values, width, color=color, label=variant)
        baseline = lookup[(spec["name"], "Baseline")]
        oracle = lookup[(spec["name"], "Oracle Association")]
        gt_detection = lookup[(spec["name"], "GT Detection")]
        axis.set_title(
            "%s\nIDS %d→%d | GT-det %d\nFRAG %d→%d | GT-det %d"
            % (
                spec["name"],
                int(baseline["IDSW"]),
                int(oracle["IDSW"]),
                int(gt_detection["IDSW"]),
                int(baseline["FM"]),
                int(oracle["FM"]),
                int(gt_detection["FM"]),
            )
        )
        axis.set_xticks(x, metric_names)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Metric (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.suptitle("Detector ceiling versus association ceiling", y=1.03)
    figure.tight_layout(rect=[0.0, 0.12, 1.0, 1.0])
    figure.savefig(output_root / "figure_2_oracle_metrics.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_regions(output_root, counts, opportunities, tracker_specs):
    import matplotlib.pyplot as plt

    hard_opportunities = sum(opportunities[(spec["name"], "hard")] for spec in tracker_specs)
    normal_opportunities = sum(opportunities[(spec["name"], "normal")] for spec in tracker_specs)
    hard = [1000.0 * counts[("Pooled", "hard", reason)] / max(hard_opportunities, 1) for reason in REASONS]
    normal = [1000.0 * counts[("Pooled", "normal", reason)] / max(normal_opportunities, 1) for reason in REASONS]
    x = np.arange(len(REASONS))
    figure, axis = plt.subplots(figsize=(11.0, 4.8))
    axis.bar(x - 0.2, hard, 0.4, label="Historical hard region", color="#E45756")
    axis.bar(x + 0.2, normal, 0.4, label="Matched normal region", color="#72B7B2")
    axis.set_xticks(x, [_reason_label(reason).replace(" ", "\n") for reason in REASONS])
    axis.set_ylabel("Cases per 1,000 good detections")
    axis.set_title("Failure mechanism in hard and matched normal regions")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_root / "figure_3_hard_vs_normal.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _percent(value):
    return "%.2f" % (100.0 * float(value))


def _summary(config, tracker_specs, cases, opportunities, counts, metrics_rows):
    lookup = {(row["tracker"], row["variant"]): row for row in metrics_rows}
    total_cases = len(cases)
    pooled = {reason: counts[("Pooled", "all", reason)] for reason in REASONS}
    score_stages = Counter(
        item.get("evidence", {}).get("stage", "unknown")
        for item in cases
        if item["reason"] == "score_filtering"
    )
    oracle_hota = []
    detector_hota = []
    for spec in tracker_specs:
        baseline = lookup[(spec["name"], "Baseline")]
        oracle = lookup[(spec["name"], "Oracle Association")]
        gt_detection = lookup[(spec["name"], "GT Detection")]
        oracle_hota.append(float(oracle["HOTA"]) - float(baseline["HOTA"]))
        detector_hota.append(float(gt_detection["HOTA"]) - float(baseline["HOTA"]))
    mean_oracle = float(np.mean(oracle_hota))
    mean_detector = float(np.mean(detector_hota))
    if mean_detector > 1.5 * max(mean_oracle, 1.0e-9):
        bottleneck = "Detector上限提升明显更大，当前主要瓶颈在检测器"
    elif mean_oracle > 1.5 * max(mean_detector, 1.0e-9):
        bottleneck = "Oracle Association提升明显更大，当前主要瓶颈在跟踪与关联"
    else:
        bottleneck = "两个Oracle都带来明显收益，检测质量和跟踪关联共同限制性能"
    major_reason = max(REASONS, key=lambda reason: pooled[reason])
    tracker_action = {
        "score_filtering": "优先做感知可靠性与分数阈值调节",
        "track_already_dead": "优先检查轨迹生命周期与短时丢失恢复",
        "gate_rejection": "优先让场景经验调节motion uncertainty与association gate",
        "wrong_assignment": "优先改进association cost和候选竞争处理",
        "output_suppression": "优先检查track confirmation与output rule",
        "other": "先补充无法归类案例的内部证据",
    }[major_reason]
    lines = [
        "# Tracking Bottleneck Diagnosis",
        "",
        "## 实验设置",
        "",
        "- Tracker：AB3DMOT、SimpleTrack、GRAE-3DMOT",
        "- 检测输入：同一组CenterPoint检测",
        "- good detection：同类GT匹配且3D IoU≥%.2f、中心距离≤%.1fm" % (config["good_detection_iou"], config["good_detection_center_m"]),
        "- Hard Region：仅使用目标sequence之前的sequence构建5m三tracker consensus risk，取最高20%",
        "- Oracle Association：保留CenterPoint的box、class、score、漏检和FP，在各tracker固定baseline分数阈值上仅用GT匹配赋予身份；它是受检测结果约束的association上限，不补检测gap中的box",
        "- GT Detection：用GT box、GT class、score=1驱动原tracker，不将GT identity交给tracker",
        "- 官方评估器无法处理预测与GT完全重合的平行边，仅在GT Detection评估副本中将yaw加1e-6 rad规避NaN，tracker输入和保存输出不变",
        "- debug回放逐帧与已有baseline核对，track id、类别、数量和box不变",
        "",
        "## good detection仍被tracker miss的原因",
        "",
        "共记录%d个案例，合并组成如下：" % total_cases,
        "",
    ]
    for reason in REASONS:
        lines.append("- %s：%d（%.1f%%）" % (_reason_label(reason), pooled[reason], 100.0 * pooled[reason] / max(total_cases, 1)))
    lines.extend(
        [
            "",
            "Score filtering内部细分：输入/预处理阈值%d例，track birth阈值%d例，输出评估分数阈值%d例"
            % (
                score_stages["input_filter"],
                score_stages["birth_threshold"],
                score_stages["output_score_threshold"],
            )
        ]
    )
    lines.extend(["", "各tracker的主要原因：", ""])
    for spec in tracker_specs:
        current_total = sum(counts[(spec["name"], "all", reason)] for reason in REASONS)
        current_reason = max(REASONS, key=lambda reason: counts[(spec["name"], "all", reason)])
        lines.append(
            "- %s：%s %d/%d（%.1f%%）"
            % (
                spec["name"],
                _reason_label(current_reason),
                counts[(spec["name"], "all", current_reason)],
                current_total,
                100.0 * counts[(spec["name"], "all", current_reason)] / max(current_total, 1),
            )
        )
    hard_opportunities = sum(opportunities[(spec["name"], "hard")] for spec in tracker_specs)
    normal_opportunities = sum(opportunities[(spec["name"], "normal")] for spec in tracker_specs)
    hard_cases = sum(counts[("Pooled", "hard", reason)] for reason in REASONS)
    normal_cases = sum(counts[("Pooled", "normal", reason)] for reason in REASONS)
    lines.extend(
        [
            "",
            "Hard Region每1000个good detection出现%.2f次tracker miss，Matched Normal为%.2f次，风险比%.2fx"
            % (
                1000.0 * hard_cases / max(hard_opportunities, 1),
                1000.0 * normal_cases / max(normal_opportunities, 1),
                (hard_cases / max(hard_opportunities, 1)) / max(normal_cases / max(normal_opportunities, 1), 1.0e-12),
            ),
            "",
            "## Oracle上限",
            "",
            "| Tracker | Setting | HOTA | AssA | IDF1 | MOTA | IDS | FRAG |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for spec in tracker_specs:
        for variant in ("Baseline", "Oracle Association", "GT Detection"):
            row = lookup[(spec["name"], variant)]
            lines.append(
                "| %s | %s | %s | %s | %s | %s | %d | %d |"
                % (
                    spec["name"],
                    variant,
                    _percent(row["HOTA"]),
                    _percent(row["AssA"]),
                    _percent(row["IDF1"]),
                    _percent(row["MOTA"]),
                    int(row["IDSW"]),
                    int(row["FM"]),
                )
            )
    lines.extend(["", "相对baseline的关键变化：", ""])
    for spec, oracle_delta, detector_delta in zip(tracker_specs, oracle_hota, detector_hota):
        baseline = lookup[(spec["name"], "Baseline")]
        oracle = lookup[(spec["name"], "Oracle Association")]
        gt_detection = lookup[(spec["name"], "GT Detection")]
        lines.append(
            "- %s：Oracle Association HOTA %+0.2f、IDF1 %+0.2f、IDS %d→%d；GT Detection HOTA %+0.2f、IDF1 %+0.2f、IDS %d→%d"
            % (
                spec["name"],
                100.0 * oracle_delta,
                100.0 * (float(oracle["IDF1"]) - float(baseline["IDF1"])),
                int(baseline["IDSW"]),
                int(oracle["IDSW"]),
                100.0 * detector_delta,
                100.0 * (float(gt_detection["IDF1"]) - float(baseline["IDF1"])),
                int(baseline["IDSW"]),
                int(gt_detection["IDSW"]),
            )
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- %s" % bottleneck,
            "- 三个tracker平均HOTA上限：Oracle Association %+0.2f点，GT Detection %+0.2f点" % (100.0 * mean_oracle, 100.0 * mean_detector),
            "- good detection仍miss的最大单一原因是%s，因此%s" % (_reason_label(major_reason), tracker_action),
            "- 是否值得继续Scene-Memory tracking：%s"
            % ("值得，但应聚焦场景化的感知可靠性和关联不确定性，而不是再调max_age" if mean_oracle > 0.01 else "单纯跟踪模块的上限较小，应先聚焦检测可靠性"),
            "- 如果修改tracker：第一优先级是在历史高风险区利用低分检测和校准分数阈值；association是次要方向，对SimpleTrack的IDS改善明显；motion gate和track management在本轮直接原因中占比较小",
            "- Oracle Association不输出检测gap中的预测box，因此FRAG仍可能高于baseline；这部分不应解读为association变差",
            "- 下一步：优先围绕上述主因做最小规则实验，并用Oracle差距作为收益上限；暂不引入GRU、Attention或神经Scene Memory",
            "- 限制：三个tracker共用CenterPoint检测，本轮可比较跟踪器差异，但不能据此声称检测器间的普遍性",
        ]
    )
    return "\n".join(lines) + "\n"


def run(config):
    output_root = Path(config["project"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / "temporary"
    temporary.mkdir(parents=True, exist_ok=True)
    tracker_specs = config["trackers"]
    split_ids = read_json(_path(config["_root"], config["split_file"]))["val"]
    sequence_ids = [str(value) for value in split_ids]
    protocol = V2XSeqProtocol(config["_root"])
    scenes, _, warnings = build_scenes(config)
    if warnings:
        write_json(temporary / "scene_warnings.json", warnings)
    regions, sequence_scene = _historical_regions(config, scenes, tracker_specs, protocol)
    debug_paths = [temporary / "debug" / (spec["name"].lower().replace("-", "_") + ".jsonl") for spec in tracker_specs]
    replay_roots = [temporary / "debug_replay" / spec["name"].lower().replace("-", "_") for spec in tracker_specs]
    debug_complete = all(path.is_file() for path in debug_paths) and all(
        _prediction_complete(path, sequence_ids) for path in replay_roots
    )
    gt_roots_expected = {
        spec["name"]: temporary / "predictions" / (spec["name"].lower().replace("-", "_") + "_gt")
        for spec in tracker_specs
    }
    gt_complete = all(_prediction_complete(path, sequence_ids) for path in gt_roots_expected.values())
    prepared = _prepare_trackers(config, tracker_specs) if not debug_complete or not gt_complete else {}
    if debug_complete:
        cases = []
        for path in debug_paths:
            current = list(read_jsonl(path))
            changed = False
            for item in current:
                if item.get("evidence", {}).get("stage") == "evaluation_score_threshold":
                    item["reason"] = "score_filtering"
                    item["evidence"]["stage"] = "output_score_threshold"
                    changed = True
            if changed:
                write_jsonl(path, current)
            cases.extend(current)
        opportunities = _good_detection_opportunities(
            config,
            tracker_specs,
            protocol,
            sequence_ids,
            regions,
            sequence_scene,
        )
        print("复用debug回放", flush=True)
    else:
        cases, opportunities = _debug_replay(
            config,
            tracker_specs,
            prepared,
            protocol,
            sequence_ids,
            regions,
            sequence_scene,
            temporary,
        )
    oracle_roots = {}
    for spec in tracker_specs:
        oracle_root = temporary / "predictions" / (
            spec["name"].lower().replace("-", "_") + "_oracle_association"
        )
        oracle_roots[spec["name"]] = oracle_root
        if not _prediction_complete(oracle_root, sequence_ids):
            threshold = float(read_json(_path(config["_root"], spec["evaluation"]))["best_score_threshold"])
            _oracle_predictions(config, protocol, sequence_ids, oracle_root, threshold)
        else:
            print("复用Oracle Association%s" % spec["name"], flush=True)
    gt_roots = _gt_predictions(config, tracker_specs, prepared, sequence_ids, temporary)
    metrics_rows = _evaluate_variants(config, tracker_specs, sequence_ids, oracle_roots, gt_roots, temporary)
    counts = _case_counts(cases)
    _plot_composition(output_root, tracker_specs, counts)
    _plot_oracles(output_root, tracker_specs, metrics_rows)
    _plot_regions(output_root, counts, opportunities, tracker_specs)
    write_json(
        temporary / "aggregate.json",
        {
            "case_count": len(cases),
            "opportunities": {"|".join(key): value for key, value in opportunities.items()},
            "reason_counts": {"|".join(key): value for key, value in counts.items()},
            "metrics": metrics_rows,
        },
    )
    (output_root / "summary.md").write_text(
        _summary(config, tracker_specs, cases, opportunities, counts, metrics_rows),
        encoding="utf-8",
    )
    print("完成tracking bottleneck diagnosis %s" % output_root, flush=True)
