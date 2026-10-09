"""检测候选的 PR、固定 FP 召回、分段召回，用于判断分数是否真的变好"""

import numpy as np

from inframot3d.tcpn.matching import MATCH_IOU, in_range, iou3d_matrix


def greedy_flags(scores, det_boxes, gt_boxes, iou_thr=MATCH_IOU):
    """按分数从高到低贪心匹配，与常规 AP 一致，返回 TP 标记和对应 GT"""
    order = np.argsort(-np.asarray(scores))
    flags = np.zeros(len(scores), dtype=np.int8)
    hit_gt = -np.ones(len(scores), dtype=np.int64)
    if len(scores) == 0 or len(gt_boxes) == 0:
        return flags, hit_gt
    iou = iou3d_matrix(det_boxes, gt_boxes)
    used = np.zeros(len(gt_boxes), dtype=bool)
    for index in order:
        row = np.where(used, -1.0, iou[index])
        best = int(np.argmax(row))
        if row[best] >= iou_thr:
            used[best] = True
            flags[index] = 1
            hit_gt[index] = best
    return flags, hit_gt


class PRAccumulator:
    def __init__(self, iou_thr=MATCH_IOU):
        self.iou_thr = iou_thr
        self.scores = []
        self.flags = []
        self.dist = []
        self.raw = []
        self.num_gt = 0
        self.gt_dist = []
        self.frames = 0

    def add(self, scores, det_boxes, gt_boxes, raw_scores=None):
        det_boxes = np.asarray(det_boxes, dtype=np.float32).reshape(-1, 7)
        gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 7)
        scores = np.asarray(scores, dtype=np.float64)
        keep = in_range(det_boxes)
        det_boxes, scores = det_boxes[keep], scores[keep]
        raw = scores if raw_scores is None else np.asarray(raw_scores, dtype=np.float64)[keep]
        gt_boxes = gt_boxes[in_range(gt_boxes)]
        flags, _ = greedy_flags(scores, det_boxes, gt_boxes, self.iou_thr)
        self.scores.append(scores)
        self.flags.append(flags)
        self.raw.append(raw)
        self.dist.append(np.hypot(det_boxes[:, 0], det_boxes[:, 1]))
        self.gt_dist.append(np.hypot(gt_boxes[:, 0], gt_boxes[:, 1]))
        self.num_gt += len(gt_boxes)
        self.frames += 1

    def arrays(self):
        cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0)
        return cat(self.scores), cat(self.flags), cat(self.dist), cat(self.raw), cat(self.gt_dist)

    def summary(self, fp_budgets=(0.1, 0.25, 0.5, 1.0, 2.0)):
        scores, flags, _, _, _ = self.arrays()
        order = np.argsort(-scores, kind="stable")
        tp = np.cumsum(flags[order])
        fp = np.cumsum(1 - flags[order])
        recall = tp / max(self.num_gt, 1)
        precision = tp / np.maximum(tp + fp, 1)
        ap = _ap(recall, precision)
        out = {"AP": float(ap), "num_gt": int(self.num_gt), "num_det": int(len(scores)), "frames": int(self.frames)}
        for budget in fp_budgets:
            limit = budget * self.frames
            idx = np.searchsorted(fp, limit, side="right") - 1
            out["recall@fp%.2f" % budget] = float(recall[idx]) if idx >= 0 else 0.0
        out["max_recall"] = float(recall[-1]) if len(recall) else 0.0
        return out

    def recall_at(self, threshold):
        scores, flags, _, _, _ = self.arrays()
        keep = scores >= threshold
        return {
            "threshold": float(threshold),
            "recall": float(flags[keep].sum() / max(self.num_gt, 1)),
            "precision": float(flags[keep].sum() / max(keep.sum(), 1)),
            "fp_per_frame": float((1 - flags[keep]).sum() / max(self.frames, 1)),
        }

    def curve(self, points=101):
        scores, flags, _, _, _ = self.arrays()
        order = np.argsort(-scores, kind="stable")
        tp = np.cumsum(flags[order])
        fp = np.cumsum(1 - flags[order])
        recall = tp / max(self.num_gt, 1)
        precision = tp / np.maximum(tp + fp, 1)
        if len(order) == 0:
            return {"recall": [], "precision": [], "fp_per_frame": []}
        pick = np.unique(np.linspace(0, len(order) - 1, points).astype(int))
        return {
            "recall": recall[pick].tolist(),
            "precision": precision[pick].tolist(),
            "fp_per_frame": (fp[pick] / max(self.frames, 1)).tolist(),
            "score": scores[order][pick].tolist(),
        }


def _ap(recall, precision):
    if len(recall) == 0:
        return 0.0
    mrec = np.concatenate([[0.0], recall, [recall[-1]]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def ece(prob, label, bins=10):
    prob = np.asarray(prob, dtype=np.float64)
    label = np.asarray(label, dtype=np.float64)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob >= lo) & (prob < hi if hi < 1 else prob <= hi)
        if mask.any():
            total += mask.mean() * abs(prob[mask].mean() - label[mask].mean())
    return float(total)
