from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

from inframot3d.geometry import bev_corners
from inframot3d.io import read_json, read_jsonl


def _prepare_box(box):
    values = [float(value) for value in box]
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
    }


def _prepared_iou(left, right):
    if left["xmax"] < right["xmin"] or right["xmax"] < left["xmin"]:
        return 0.0
    if left["ymax"] < right["ymin"] or right["ymax"] < left["ymin"]:
        return 0.0
    height = min(left["zmax"], right["zmax"]) - max(left["zmin"], right["zmin"])
    if height <= 0.0:
        return 0.0
    if not left["poly"].intersects(right["poly"]):
        return 0.0
    intersection = left["poly"].intersection(right["poly"]).area * height
    union = left["volume"] + right["volume"] - intersection
    return intersection / union if union > 0.0 else 0.0


def _similarity(gt_objects, tracker_objects):
    matrix = np.zeros((len(gt_objects), len(tracker_objects)), dtype=np.float64)
    gt_boxes = [_prepare_box(item["box"]) for item in gt_objects]
    tracker_boxes = [_prepare_box(item["box"]) for item in tracker_objects]
    for row, gt_box in enumerate(gt_boxes):
        for column, tracker_box in enumerate(tracker_boxes):
            matrix[row, column] = float(_prepared_iou(gt_box, tracker_box))
    return matrix


def _sequence_frames(config, prediction_root, sequence_id, protocol, score_threshold):
    manifest = read_json(Path(config["project"]["converted_root"]) / "manifest.json")
    entry = next(item for item in manifest["sequences"] if item["sequence_id"] == sequence_id)
    gt_rows = list(read_jsonl(Path(config["project"]["converted_root"]) / entry["path"]))
    tracker_rows = list(read_jsonl(Path(prediction_root) / ("%s.jsonl" % sequence_id)))
    tracker_by_frame = {int(row["frame_index"]): row for row in tracker_rows}
    gt_ids = {}
    tracker_ids = {}
    frames = []
    for row in gt_rows:
        gt_objects = protocol.filter_gt(row["objects"])
        if not gt_objects:
            continue
        tracker_row = tracker_by_frame[int(row["frame_index"])]
        tracker_objects = [
            item
            for item in protocol.filter_prediction(tracker_row["objects"])
            if float(item.get("score", 1.0)) >= float(score_threshold)
        ]
        current_gt_ids = np.asarray(
            [gt_ids.setdefault(str(item["source_track_id"]), len(gt_ids)) for item in gt_objects],
            dtype=np.int64,
        )
        current_tracker_ids = np.asarray(
            [tracker_ids.setdefault(str(item["track_id"]), len(tracker_ids)) for item in tracker_objects],
            dtype=np.int64,
        )
        frames.append((current_gt_ids, current_tracker_ids, _similarity(gt_objects, tracker_objects)))
    return frames, len(gt_ids), len(tracker_ids)


def _sequence_counts(frames, num_gt_ids, num_tracker_ids, alphas):
    gt_count = np.zeros(num_gt_ids, dtype=np.float64)
    tracker_count = np.zeros(num_tracker_ids, dtype=np.float64)
    potential = np.zeros((num_gt_ids, num_tracker_ids), dtype=np.float64)
    for gt_ids, tracker_ids, similarity in frames:
        gt_count[gt_ids] += 1.0
        tracker_count[tracker_ids] += 1.0
        if not len(gt_ids) or not len(tracker_ids):
            continue
        denominator = similarity.sum(1)[:, None] + similarity.sum(0)[None, :] - similarity
        score = np.divide(similarity, denominator, out=np.zeros_like(similarity), where=denominator > 0.0)
        potential[np.ix_(gt_ids, tracker_ids)] += score
    denominator = gt_count[:, None] + tracker_count[None, :] - potential
    alignment = np.divide(potential, denominator, out=np.zeros_like(potential), where=denominator > 0.0)
    output = []
    for alpha in alphas:
        matches = np.zeros_like(potential)
        true_positive = 0
        false_positive = 0
        false_negative = 0
        localization = 0.0
        for gt_ids, tracker_ids, similarity in frames:
            if len(gt_ids) and len(tracker_ids):
                score = alignment[np.ix_(gt_ids, tracker_ids)] * similarity
                rows, columns = linear_sum_assignment(-score)
                keep = similarity[rows, columns] >= float(alpha) - np.finfo(float).eps
                rows = rows[keep]
                columns = columns[keep]
                matched_gt = gt_ids[rows]
                matched_tracker = tracker_ids[columns]
                matches[matched_gt, matched_tracker] += 1.0
                localization += float(similarity[rows, columns].sum())
                current_true_positive = len(rows)
            else:
                current_true_positive = 0
            true_positive += current_true_positive
            false_negative += len(gt_ids) - current_true_positive
            false_positive += len(tracker_ids) - current_true_positive
        association_denominator = gt_count[:, None] + tracker_count[None, :] - matches
        association = np.divide(
            matches,
            association_denominator,
            out=np.zeros_like(matches),
            where=association_denominator > 0.0,
        )
        output.append(
            {
                "TP": true_positive,
                "FP": false_positive,
                "FN": false_negative,
                "AssA_num": float((matches * association).sum()),
                "LocA_num": localization,
            }
        )
    return output


def evaluate_hota(config, prediction_root, sequence_ids, protocol, score_threshold):
    alphas = np.arange(0.05, 0.96, 0.05)
    totals = [dict(TP=0, FP=0, FN=0, AssA_num=0.0, LocA_num=0.0) for _ in alphas]
    for sequence_id in sequence_ids:
        frames, num_gt_ids, num_tracker_ids = _sequence_frames(
            config, prediction_root, sequence_id, protocol, score_threshold
        )
        current = _sequence_counts(frames, num_gt_ids, num_tracker_ids, alphas)
        for total, row in zip(totals, current):
            for key in total:
                total[key] += row[key]
    hota = []
    detection = []
    association = []
    localization = []
    for row in totals:
        detection_value = row["TP"] / max(1, row["TP"] + row["FP"] + row["FN"])
        association_value = row["AssA_num"] / max(1, row["TP"])
        localization_value = row["LocA_num"] / max(1, row["TP"])
        detection.append(detection_value)
        association.append(association_value)
        localization.append(localization_value)
        hota.append(np.sqrt(detection_value * association_value))
    return {
        "HOTA": float(np.mean(hota)),
        "DetA": float(np.mean(detection)),
        "AssA": float(np.mean(association)),
        "LocA": float(np.mean(localization)),
        "HOTA_score_threshold": float(score_threshold),
    }
