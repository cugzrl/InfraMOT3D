import numpy as np

from inframot3d.geometry import iou_3d


def evaluate_frames(frames, class_names, iou_thresholds, score_threshold):
    metrics = {}
    for name in class_names:
        gt_by_frame = []
        pred_by_frame = []
        for frame in frames:
            gt_boxes = [
                item["box"] for item in frame["gt"] if item["class_name"] == name
            ]
            predictions = [item for item in frame["pred"] if item["class_name"] == name]
            predictions.sort(key=lambda item: float(item["score"]), reverse=True)
            gt_by_frame.append(gt_boxes)
            pred_by_frame.append(predictions)
        metrics[name] = _class_metrics(gt_by_frame, pred_by_frame, float(iou_thresholds[name]), float(score_threshold))
    return metrics


def _class_metrics(gt_by_frame, pred_by_frame, iou_threshold, score_threshold):
    num_gt = sum(len(boxes) for boxes in gt_by_frame)
    records = []
    for frame_index, predictions in enumerate(pred_by_frame):
        gt_boxes = gt_by_frame[frame_index]
        matched = set()
        for prediction in predictions:
            hit = False
            best_index = -1
            best_iou = 0.0
            for gt_index, gt_box in enumerate(gt_boxes):
                if gt_index in matched:
                    continue
                value = iou_3d(prediction["box"], gt_box)
                if value > best_iou:
                    best_iou = value
                    best_index = gt_index
            if best_index >= 0 and best_iou >= iou_threshold:
                matched.add(best_index)
                hit = True
            records.append((float(prediction["score"]), hit))
    records.sort(key=lambda item: item[0], reverse=True)
    tp_flags = np.array([1.0 if item[1] else 0.0 for item in records], dtype=np.float64)
    operating = [item for item in records if item[0] >= score_threshold]
    num_pred = len(operating)
    tp = sum(1 for item in operating if item[1])
    precision = tp / num_pred if num_pred else 0.0
    recall = tp / num_gt if num_gt else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "ap": float(_average_precision(tp_flags, num_gt)),
        "num_gt": int(num_gt),
        "num_pred": int(num_pred),
        "iou_threshold": float(iou_threshold),
    }


def _average_precision(tp_flags, num_gt):
    if num_gt == 0 or len(tp_flags) == 0:
        return 0.0
    tp = np.cumsum(tp_flags)
    fp = np.cumsum(1.0 - tp_flags)
    recall = tp / num_gt
    precision = tp / np.maximum(tp + fp, 1e-9)
    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[0.0], precision, [0.0]])
    for index in range(precision.size - 1, 0, -1):
        precision[index - 1] = max(precision[index - 1], precision[index])
    change = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[change + 1] - recall[change]) * precision[change + 1]))
