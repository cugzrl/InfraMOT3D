from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from inframot3d.geometry import iou_3d
from inframot3d.io import read_json, read_jsonl, write_json


def _match(gt_objects, pred_objects, threshold):
    if not gt_objects or not pred_objects:
        return [], list(range(len(gt_objects))), list(range(len(pred_objects)))
    matrix = np.empty((len(gt_objects), len(pred_objects)), dtype=float)
    for row, gt_value in enumerate(gt_objects):
        for column, pred_value in enumerate(pred_objects):
            matrix[row, column] = iou_3d(gt_value["box"], pred_value["box"])
    rows, columns = linear_sum_assignment(-matrix)
    matches = [
        (int(row), int(column), float(matrix[row, column]))
        for row, column in zip(rows, columns)
        if matrix[row, column] >= threshold
    ]
    matched_gt = {row for row, _, _ in matches}
    matched_pred = {column for _, column, _ in matches}
    return (
        matches,
        [index for index in range(len(gt_objects)) if index not in matched_gt],
        [index for index in range(len(pred_objects)) if index not in matched_pred],
    )


class Accumulator:
    def __init__(self):
        self.gt = 0
        self.pred = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.id_switches = 0
        self.fragmentations = 0
        self.iou_sum = 0.0
        self.pairs = Counter()
        self.track_total = Counter()
        self.track_matched = Counter()
        self.last_prediction = {}
        self.ever_matched = set()
        self.was_matched = {}

    def add_frame(self, sequence_id, class_name, gt_objects, pred_objects, threshold):
        matches, unmatched_gt, unmatched_pred = _match(gt_objects, pred_objects, threshold)
        self.gt += len(gt_objects)
        self.pred += len(pred_objects)
        self.tp += len(matches)
        self.fn += len(unmatched_gt)
        self.fp += len(unmatched_pred)
        self.iou_sum += sum(value for _, _, value in matches)
        for value in gt_objects:
            key = (sequence_id, class_name, str(value["source_track_id"]))
            self.track_total[key] += 1
        for gt_index, pred_index, _ in matches:
            gt_id = str(gt_objects[gt_index]["source_track_id"])
            pred_id = str(pred_objects[pred_index]["track_id"])
            gt_key = (sequence_id, class_name, gt_id)
            pred_key = (sequence_id, class_name, pred_id)
            self.track_matched[gt_key] += 1
            self.pairs[(gt_key, pred_key)] += 1
            if gt_key in self.last_prediction and self.last_prediction[gt_key] != pred_key:
                self.id_switches += 1
            if gt_key in self.ever_matched and self.was_matched.get(gt_key) is False:
                self.fragmentations += 1
            self.last_prediction[gt_key] = pred_key
            self.ever_matched.add(gt_key)
            self.was_matched[gt_key] = True
        for index in unmatched_gt:
            value = gt_objects[index]
            key = (sequence_id, class_name, str(value["source_track_id"]))
            self.was_matched[key] = False

    def _idtp(self):
        grouped = defaultdict(list)
        for (gt_key, pred_key), count in self.pairs.items():
            grouped[(gt_key[0], gt_key[1])].append((gt_key, pred_key, count))
        total = 0
        for values in grouped.values():
            gt_keys = sorted({value[0] for value in values})
            pred_keys = sorted({value[1] for value in values})
            gt_index = {key: index for index, key in enumerate(gt_keys)}
            pred_index = {key: index for index, key in enumerate(pred_keys)}
            matrix = np.zeros((len(gt_keys), len(pred_keys)), dtype=np.int64)
            for gt_key, pred_key, count in values:
                matrix[gt_index[gt_key], pred_index[pred_key]] = count
            rows, columns = linear_sum_assignment(-matrix)
            total += int(matrix[rows, columns].sum())
        return total

    def metrics(self):
        idtp = self._idtp()
        idfp = self.pred - idtp
        idfn = self.gt - idtp
        matched_ratios = [self.track_matched[key] / count for key, count in self.track_total.items()]
        return {
            "gt_objects": self.gt,
            "pred_objects": self.pred,
            "true_positives": self.tp,
            "false_positives": self.fp,
            "false_negatives": self.fn,
            "id_switches": self.id_switches,
            "fragmentations": self.fragmentations,
            "mota": 1.0 - (self.fp + self.fn + self.id_switches) / self.gt if self.gt else 0.0,
            "motp": self.iou_sum / self.tp if self.tp else 0.0,
            "precision": self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0,
            "recall": self.tp / self.gt if self.gt else 0.0,
            "idf1": 2.0 * idtp / (2 * idtp + idfp + idfn) if 2 * idtp + idfp + idfn else 0.0,
            "idtp": idtp,
            "idfp": idfp,
            "idfn": idfn,
            "mostly_tracked": sum(value >= 0.8 for value in matched_ratios),
            "mostly_lost": sum(value <= 0.2 for value in matched_ratios),
            "num_gt_tracks": len(matched_ratios),
        }


def _evaluate_sequences(converted_root, prediction_root, thresholds, selected_sequences=None):
    manifest = read_json(converted_root / "manifest.json")
    overall = Accumulator()
    by_class = defaultdict(Accumulator)
    by_sequence = {}
    allowed = set(selected_sequences) if selected_sequences else None
    for entry in manifest["sequences"]:
        sequence_id = entry["sequence_id"]
        if allowed is not None and sequence_id not in allowed:
            continue
        gt_frames = list(read_jsonl(converted_root / entry["path"]))
        pred_frames = list(read_jsonl(prediction_root / f"{sequence_id}.jsonl"))
        if len(gt_frames) != len(pred_frames):
            raise ValueError(f"序列{sequence_id}帧数不一致")
        sequence_accumulator = Accumulator()
        for gt_frame, pred_frame in zip(gt_frames, pred_frames):
            classes = sorted(
                {value["class_name"] for value in gt_frame["objects"]}
                | {value["class_name"] for value in pred_frame["objects"]}
            )
            for class_name in classes:
                gt_objects = [value for value in gt_frame["objects"] if value["class_name"] == class_name]
                pred_objects = [value for value in pred_frame["objects"] if value["class_name"] == class_name]
                threshold = float(thresholds[class_name])
                overall.add_frame(sequence_id, class_name, gt_objects, pred_objects, threshold)
                by_class[class_name].add_frame(sequence_id, class_name, gt_objects, pred_objects, threshold)
                sequence_accumulator.add_frame(sequence_id, class_name, gt_objects, pred_objects, threshold)
        by_sequence[sequence_id] = sequence_accumulator.metrics()
    summary = {
        "protocol": "路侧虚拟激光雷达坐标系逐帧3D IoU匹配",
        "overall": overall.metrics(),
        "by_class": {key: value.metrics() for key, value in sorted(by_class.items())},
        "iou_thresholds": thresholds,
    }
    return summary, by_sequence


def evaluate(converted_root, prediction_root, output_root, thresholds, selected_sequences=None):
    summary, by_sequence = _evaluate_sequences(
        converted_root, prediction_root, thresholds, selected_sequences
    )
    write_json(output_root / "metrics_summary.json", summary)
    write_json(output_root / "metrics_by_sequence.json", by_sequence)
    return summary, by_sequence
