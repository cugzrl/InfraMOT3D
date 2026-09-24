import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

from inframot3d.config import load_config
from inframot3d.geometry import iou_3d
from inframot3d.io import read_json, read_jsonl, write_json


THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60)
CLASSES = ("Car", "Van", "Bus", "Truck", "Pedestrian", "Cyclist", "Motorcyclist", "Barrowlist")


def _rows(path):
    return list(read_jsonl(path))


def _by_class(objects):
    grouped = defaultdict(list)
    for item in objects:
        grouped[item["class_name"]].append(item)
    return grouped


def _metrics(tp, fp, fn):
    gt = tp + fn
    predicted = tp + fp
    precision = tp / predicted if predicted else 0.0
    recall = tp / gt if gt else 0.0
    upper = 1.0 - (fp + fn) / gt if gt else 0.0
    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "GT": int(gt),
        "Precision": precision,
        "Recall": recall,
        "MOTA_upper": upper,
    }


def _prepare(frames, class_name):
    prepared = []
    for gt_group, det_group in frames:
        gt_objects = gt_group.get(class_name, [])
        pred_objects = det_group.get(class_name, [])
        scores = np.asarray([float(item["score"]) for item in pred_objects], dtype=np.float64)
        matrix = None
        if gt_objects and pred_objects:
            matrix = np.empty((len(gt_objects), len(pred_objects)), dtype=np.float64)
            for row, gt_value in enumerate(gt_objects):
                for column, pred_value in enumerate(pred_objects):
                    matrix[row, column] = iou_3d(gt_value["box"], pred_value["box"])
        prepared.append((len(gt_objects), scores, matrix))
    return prepared


def _count_prepared(prepared, threshold, iou):
    tp = fp = fn = 0
    for gt_count, scores, matrix in prepared:
        if scores.size:
            keep = np.flatnonzero(scores >= threshold)
        else:
            keep = np.zeros(0, dtype=int)
        if gt_count == 0 and keep.size == 0:
            continue
        if matrix is None or keep.size == 0 or gt_count == 0:
            fn += gt_count
            fp += int(keep.size)
            continue
        sub = matrix[:, keep]
        rows, columns = linear_sum_assignment(-sub)
        matched = int(np.count_nonzero(sub[rows, columns] >= iou))
        tp += matched
        fn += gt_count - matched
        fp += int(keep.size) - matched
    return _metrics(tp, fp, fn)


def _pick(rows):
    return max(rows, key=lambda row: (row["MOTA_upper"], row["Precision"], row["threshold"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ab3dmot_centerpoint.yaml")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--output", default="outputs/threshold_sweep/sweep_train.json")
    args = parser.parse_args()
    if args.split != "train":
        raise SystemExit("阈值扫描只允许使用train")
    config = load_config(args.config)
    root = config["_root"]
    split_ids = set(read_json(root / config["split_file"])[args.split])
    detection_root = Path(config["input"]["detection_root"])
    if not detection_root.is_absolute():
        detection_root = root / detection_root
    converted_root = Path(config["project"]["converted_root"])
    manifest = read_json(converted_root / "manifest.json")
    iou_table = {name: float(value) for name, value in config["evaluation"]["iou_thresholds"].items()}
    frames = []
    for entry in manifest["sequences"]:
        sequence_id = entry["sequence_id"]
        if sequence_id not in split_ids:
            continue
        detections = _rows(detection_root / ("%s.jsonl" % sequence_id))
        ground_truth = _rows(converted_root / entry["path"])
        if len(detections) != len(ground_truth):
            raise ValueError("帧数不一致 %s" % sequence_id)
        for det_row, gt_row in zip(detections, ground_truth):
            if det_row["frame_id"] != gt_row["frame_id"]:
                raise ValueError("帧未对齐 %s" % sequence_id)
            frames.append((_by_class(gt_row["objects"]), _by_class(det_row["objects"])))
    print("扫描序列帧%d" % len(frames))
    per_class = {}
    chosen = {}
    prepared = {}
    for class_name in CLASSES:
        prepared[class_name] = _prepare(frames, class_name)
        rows = []
        for threshold in THRESHOLDS:
            row = {"threshold": threshold, "class_name": class_name}
            row.update(_count_prepared(prepared[class_name], threshold, iou_table[class_name]))
            rows.append(row)
            print(
                "%s %.2f TP %d FP %d FN %d P %.4f R %.4f MOTA_upper %.4f"
                % (class_name, threshold, row["TP"], row["FP"], row["FN"], row["Precision"], row["Recall"], row["MOTA_upper"])
            )
        best = _pick(rows)
        chosen[class_name] = best["threshold"]
        per_class[class_name] = {"rows": rows, "selected": best["threshold"]}
    global_rows = []
    for threshold in THRESHOLDS:
        row = {"threshold": threshold}
        total = {"TP": 0, "FP": 0, "FN": 0, "GT": 0}
        for class_name in CLASSES:
            part = _count_prepared(prepared[class_name], threshold, iou_table[class_name])
            for key in total:
                total[key] += part[key]
        row.update(_metrics(total["TP"], total["FP"], total["FN"]))
        global_rows.append(row)
        print(
            "global %.2f TP %d FP %d FN %d P %.4f R %.4f MOTA_upper %.4f"
            % (threshold, row["TP"], row["FP"], row["FN"], row["Precision"], row["Recall"], row["MOTA_upper"])
        )
    payload = {
        "split": args.split,
        "detection_root": str(detection_root),
        "thresholds": list(THRESHOLDS),
        "criterion": "MOTA_upper",
        "per_class": per_class,
        "global": global_rows,
        "score_thresholds": chosen,
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    write_json(output, payload)
    threshold_path = root / "configs" / "centerpoint_score_thresholds.yaml"
    threshold_path.write_text(
        yaml.safe_dump({"score_thresholds": chosen}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print("已写入 %s" % threshold_path)


if __name__ == "__main__":
    main()
