import copy
import pickle

import numpy as np
import torch

from ...ops.iou3d_nms import iou3d_nms_utils
from ..dataset import DatasetTemplate


class V2XSeqDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.infos = []
        self.include_data(self.mode)

    def _log(self, message):
        if self.logger is not None:
            self.logger.info(message)
        else:
            print(message)

    def include_data(self, mode):
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            path = self.root_path / info_path
            if not path.exists():
                continue
            with open(path, "rb") as stream:
                self.infos.extend(pickle.load(stream))
        self._log("V2X-Seq样本数%d" % len(self.infos))

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs
        return len(self.infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)
        info = copy.deepcopy(self.infos[index])
        lidar_idx = info["point_cloud"]["lidar_idx"]
        points = np.load(self.root_path / "points" / ("%s.npy" % lidar_idx))
        input_dict = {"frame_id": lidar_idx, "points": points[:, :4]}
        annos = info.get("annos")
        if annos is not None:
            names = annos["name"]
            boxes = annos["gt_boxes_lidar"]
            if len(names) == 0:
                names = np.array([], dtype=str)
                boxes = np.zeros((0, 7), dtype=np.float32)
            input_dict.update({"gt_names": names, "gt_boxes": boxes})
        return self.prepare_data(data_dict=input_dict)

    def evaluation(self, det_annos, class_names, **kwargs):
        iou_thresholds = kwargs.get("iou_thresholds")
        if iou_thresholds is None:
            iou_thresholds = {name: 0.5 if name in ("Car", "Van", "Bus", "Truck") else 0.25 for name in class_names}
        score_threshold = float(kwargs.get("score_threshold", 0.1))
        gt_annos = [info["annos"] for info in self.infos]
        metrics = evaluate_annos(det_annos, gt_annos, class_names, iou_thresholds, score_threshold)
        lines = ["class precision recall ap num_gt num_pred"]
        flat = {}
        for name in class_names:
            item = metrics[name]
            lines.append(
                "%s %.4f %.4f %.4f %d %d" % (
                    name, item["precision"], item["recall"], item["ap"], item["num_gt"], item["num_pred"]
                )
            )
            flat["%s/precision" % name] = float(item["precision"])
            flat["%s/recall" % name] = float(item["recall"])
            flat["%s/ap" % name] = float(item["ap"])
        return "\n".join(lines), flat


def evaluate_annos(det_annos, gt_annos, class_names, iou_thresholds, score_threshold):
    metrics = {}
    for name in class_names:
        gt_by_frame = []
        pred_by_frame = []
        for gt_anno, det_anno in zip(gt_annos, det_annos):
            gt_mask = gt_anno["name"] == name
            gt_boxes = gt_anno["gt_boxes_lidar"][gt_mask] if gt_anno["gt_boxes_lidar"].size else np.zeros((0, 7), np.float32)
            det_mask = det_anno["name"] == name
            det_boxes = det_anno["boxes_lidar"][det_mask] if det_anno["boxes_lidar"].size else np.zeros((0, 7), np.float32)
            det_scores = det_anno["score"][det_mask] if det_anno["score"].size else np.zeros((0,), np.float32)
            gt_by_frame.append(np.asarray(gt_boxes[:, :7], dtype=np.float32))
            pred_by_frame.append((np.asarray(det_boxes[:, :7], dtype=np.float32), np.asarray(det_scores, dtype=np.float32)))
        metrics[name] = _class_metrics(gt_by_frame, pred_by_frame, float(iou_thresholds[name]), score_threshold)
    return metrics


def _class_metrics(gt_by_frame, pred_by_frame, iou_threshold, score_threshold):
    num_gt = int(sum(len(boxes) for boxes in gt_by_frame))
    scored = []
    for frame_index, (boxes, scores) in enumerate(pred_by_frame):
        if len(boxes) == 0:
            continue
        gt_boxes = gt_by_frame[frame_index]
        ious = _iou_matrix(boxes, gt_boxes) if len(gt_boxes) else None
        for index in np.argsort(-scores):
            scored.append((float(scores[index]), frame_index, int(index), ious))
    matched = [set() for _ in gt_by_frame]
    tp_flags = []
    for _, frame_index, box_index, ious in scored:
        if ious is None:
            tp_flags.append(0.0)
            continue
        order = np.argsort(-ious[box_index])
        hit = False
        for gt_index in order:
            if ious[box_index, gt_index] < iou_threshold or int(gt_index) in matched[frame_index]:
                continue
            matched[frame_index].add(int(gt_index))
            hit = True
            break
        tp_flags.append(1.0 if hit else 0.0)
    pairs = sorted(zip(scored, tp_flags), key=lambda item: item[0][0], reverse=True)
    tp_flags = np.asarray([item[1] for item in pairs], dtype=np.float32)
    operating = [item[1] for item in pairs if item[0][0] >= score_threshold]
    num_pred = len(operating)
    tp = float(np.sum(operating)) if operating else 0.0
    precision = tp / num_pred if num_pred else 0.0
    recall = tp / num_gt if num_gt else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "ap": float(_average_precision(tp_flags, num_gt)),
        "num_gt": num_gt,
        "num_pred": int(num_pred),
        "iou_threshold": float(iou_threshold),
    }


def _iou_matrix(boxes_a, boxes_b):
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    with torch.no_grad():
        ious = iou3d_nms_utils.boxes_iou3d_gpu(
            torch.from_numpy(np.ascontiguousarray(boxes_a)).cuda(),
            torch.from_numpy(np.ascontiguousarray(boxes_b)).cuda(),
        )
    return ious.cpu().numpy()


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
