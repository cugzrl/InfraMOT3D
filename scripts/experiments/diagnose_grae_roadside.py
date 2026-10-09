import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from inframot3d.analysis.scene_difficulty import _match
from inframot3d.config import load_config
from inframot3d.evaluation.protocols import build_protocol
from inframot3d.io import read_json, read_jsonl


def _range_bin(box):
    distance = float(np.hypot(box[0], box[1]))
    if distance < 50.0:
        return "0-50"
    if distance < 100.0:
        return "50-100"
    return "100+"


def _speed_bin(speed):
    if speed < 1.0:
        return "0-1"
    if speed < 5.0:
        return "1-5"
    if speed < 15.0:
        return "5-15"
    return "15+"


def main():
    config = load_config("configs/trackers/grae/centerpoint.yaml")
    root = config["_root"]
    protocol = build_protocol("v2xseq", root)
    classes = set(config["classes"])
    vehicles = {"Car", "Van", "Bus", "Truck"}
    split_ids = set(read_json(root / config["split_file"])["val"])
    manifest = read_json(Path(config["project"]["converted_root"]) / "manifest.json")
    detection_root = root / config["input"]["detection_root"]
    bins = [(0.01, 0.1, "0.01-0.1"), (0.1, 0.4, "0.1-0.4"), (0.4, 1.01, "0.4+")]
    det_count = Counter()
    matched_count = Counter()
    range_count = Counter()
    speed_count = Counter()
    density_count = Counter()
    class_pairs = Counter()
    cross_vehicle = 0
    cross_other = 0
    same_class = 0
    flips = Counter()
    flip_tracks = set()
    vehicle_tracks = set()
    low_matched_tracks = set()
    gt_frames = 0
    for entry in manifest["sequences"]:
        sequence_id = entry["sequence_id"]
        if sequence_id not in split_ids:
            continue
        detections = list(read_jsonl(detection_root / ("%s.jsonl" % sequence_id)))
        ground_truth = list(read_jsonl(Path(config["project"]["converted_root"]) / entry["path"]))
        previous = {}
        previous_time = None
        for det_row, gt_row in zip(detections, ground_truth):
            gt_objects = protocol.filter_gt(gt_row["objects"])
            gt_frames += len(gt_objects)
            timestamp = int(gt_row["timestamp"]) / 1e6
            current = {}
            for item in gt_objects:
                current[str(item["source_track_id"])] = item
                vehicle_tracks.add((sequence_id, str(item["source_track_id"])))
            if previous_time is not None:
                gap = timestamp - previous_time
                if 0.0 < gap < 1.0:
                    for track_id, item in current.items():
                        if track_id not in previous:
                            continue
                        old = previous[track_id]["class_name"]
                        new = item["class_name"]
                        if old != new:
                            flips[(old, new)] += 1
                            flip_tracks.add((sequence_id, track_id))
            dets = []
            for item in det_row["objects"]:
                score = float(item.get("score", 0.0))
                if item["class_name"] not in classes or score < 0.01:
                    continue
                if protocol.normalize_class(item["class_name"]) is None:
                    continue
                if not protocol.inside_range(item["box"]):
                    continue
                dets.append(item)
            matches = _match(
                [item["box"] for item in gt_objects],
                [item["box"] for item in dets],
                0.25,
                20.0,
            )
            matched_det = {column for _, column, _ in matches}
            centers = np.asarray([[float(item["box"][0]), float(item["box"][1])] for item in gt_objects], dtype=np.float64)
            for row, column, _ in matches:
                det = dets[column]
                gt = gt_objects[row]
                score = float(det["score"])
                name = next(label for low, high, label in bins if low <= score < high)
                matched_count[name] += 1
                if det["class_name"] == gt["class_name"]:
                    same_class += 1
                else:
                    class_pairs[(gt["class_name"], det["class_name"])] += 1
                    if det["class_name"] in vehicles and gt["class_name"] in vehicles:
                        cross_vehicle += 1
                    else:
                        cross_other += 1
                if name == "0.01-0.1":
                    low_matched_tracks.add((sequence_id, str(gt["source_track_id"])))
                    range_count[_range_bin(gt["box"])] += 1
                    if len(centers):
                        gap = np.hypot(centers[:, 0] - centers[row, 0], centers[:, 1] - centers[row, 1])
                        neighbors = int(np.sum((gap > 0.0) & (gap <= 8.0)))
                        if neighbors == 0:
                            density_count["0"] += 1
                        elif neighbors <= 2:
                            density_count["1-2"] += 1
                        else:
                            density_count["3+"] += 1
                    track_id = str(gt["source_track_id"])
                    if previous_time is not None and track_id in previous:
                        gap_t = timestamp - previous_time
                        if 0.0 < gap_t < 1.0:
                            old = previous[track_id]["box"]
                            speed = float(np.hypot(gt["box"][0] - old[0], gt["box"][1] - old[1]) / gap_t)
                            speed_count[_speed_bin(speed)] += 1
            for index, det in enumerate(dets):
                score = float(det["score"])
                name = next(label for low, high, label in bins if low <= score < high)
                det_count[name] += 1
                if index not in matched_det:
                    det_count["fp:" + name] += 1
            previous = current
            previous_time = timestamp
        print("诊断序列%s" % sequence_id, flush=True)
    output = {
        "val_vehicle_gt_boxes": gt_frames,
        "vehicle_tracks": len(vehicle_tracks),
        "detection_bins": dict(det_count),
        "geometric_matches": dict(matched_count),
        "low_score_match_rate": matched_count["0.01-0.1"] / det_count["0.01-0.1"] if det_count["0.01-0.1"] else None,
        "low_score_range": dict(range_count),
        "low_score_speed": dict(speed_count),
        "low_score_neighbors_8m": dict(density_count),
        "low_score_tracks": len(low_matched_tracks),
        "same_class_matches": same_class,
        "cross_class_vehicle": cross_vehicle,
        "cross_class_other": cross_other,
        "class_pairs": [{"gt": key[0], "det": key[1], "count": value} for key, value in class_pairs.most_common()],
        "class_flip_events": [{"from": key[0], "to": key[1], "count": value} for key, value in flips.most_common()],
        "class_flip_tracks": len(flip_tracks),
    }
    target = Path("outputs/experiments/grae_roadside_discovery")
    target.mkdir(parents=True, exist_ok=True)
    (target / "diagnosis_stats.json").write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
