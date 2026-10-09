"""候选检测与 GT 的离线匹配，只在训练侧或最终离线评估中使用

规则与 V2X-Seq 官方跟踪评估保持一致
- 车辆大类 Car Van Bus Truck 合并
- 3D IoU 不低于 0.25 才可匹配
- 一对一匹配按 IoU 最大化的匈牙利算法
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from inframot3d.geometry import iou_3d

VEHICLES = ("Car", "Van", "Bus", "Truck")
MATCH_IOU = 0.25
IGNORE_IOU = 0.1
IGNORE_CENTER = 1.5
OFFICIAL_RANGE = (0.0, -39.68, 100.0, 39.68)

STATUS_TP = 0
STATUS_DUP = 1
STATUS_IGNORE = 2
STATUS_BG = 3
STATUS_NAMES = ("tp", "dup", "ignore", "bg")

_GPU_IOU = None


def _gpu_iou():
    global _GPU_IOU
    if _GPU_IOU is None:
        try:
            import torch
            from pcdet.ops.iou3d_nms import iou3d_nms_utils

            if torch.cuda.is_available():
                _GPU_IOU = iou3d_nms_utils.boxes_iou3d_gpu
            else:
                _GPU_IOU = False
        except Exception:
            _GPU_IOU = False
    return _GPU_IOU


def to_pcdet(boxes):
    """x y z yaw l w h 转 OpenPCDet 的 x y z l w h yaw"""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 7)
    return boxes[:, [0, 1, 2, 4, 5, 6, 3]]


def iou3d_matrix(boxes_a, boxes_b):
    boxes_a = np.asarray(boxes_a, dtype=np.float32).reshape(-1, 7)
    boxes_b = np.asarray(boxes_b, dtype=np.float32).reshape(-1, 7)
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    fn = _gpu_iou()
    if fn:
        import torch

        a = torch.from_numpy(to_pcdet(boxes_a)).cuda()
        b = torch.from_numpy(to_pcdet(boxes_b)).cuda()
        return fn(a, b).cpu().numpy().astype(np.float32)
    out = np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    reach = np.hypot(boxes_a[:, 4], boxes_a[:, 5])[:, None] / 2 + np.hypot(boxes_b[:, 4], boxes_b[:, 5])[None, :] / 2
    dist = np.linalg.norm(boxes_a[:, None, :2] - boxes_b[None, :, :2], axis=-1)
    for i, j in zip(*np.where(dist < reach)):
        out[i, j] = iou_3d(boxes_a[i].tolist(), boxes_b[j].tolist())
    return out


def in_range(boxes, margin=0.0):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 7)
    x0, y0, x1, y1 = OFFICIAL_RANGE
    return (
        (boxes[:, 0] >= x0 - margin)
        & (boxes[:, 0] <= x1 + margin)
        & (boxes[:, 1] >= y0 - margin)
        & (boxes[:, 1] <= y1 + margin)
    )


def label_candidates(det_boxes, gt_boxes):
    """返回每个候选的状态、匹配 GT 下标、匹配 IoU 与最大 IoU

    tp      一对一匹配上的候选
    dup     IoU 达标但对应 GT 已被其他候选占用，属于重复框
    ignore  0.1 到 0.25 的模糊定位，或中心距离 1.5 m 内但 IoU 不足
    bg      与全部车辆 GT 都不相交
    """
    det_boxes = np.asarray(det_boxes, dtype=np.float32).reshape(-1, 7)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 7)
    count = len(det_boxes)
    status = np.full(count, STATUS_BG, dtype=np.int64)
    gt_index = -np.ones(count, dtype=np.int64)
    match_iou = np.zeros(count, dtype=np.float32)
    max_iou = np.zeros(count, dtype=np.float32)
    nearest_gt = -np.ones(count, dtype=np.int64)
    if count == 0 or len(gt_boxes) == 0:
        return status, gt_index, match_iou, max_iou, nearest_gt
    iou = iou3d_matrix(det_boxes, gt_boxes)
    max_iou = iou.max(axis=1)
    nearest_gt = iou.argmax(axis=1)
    cost = np.where(iou >= MATCH_IOU, -iou, 1.0)
    rows, cols = linear_sum_assignment(cost)
    for row, col in zip(rows, cols):
        if iou[row, col] >= MATCH_IOU:
            status[row] = STATUS_TP
            gt_index[row] = col
            match_iou[row] = iou[row, col]
    center = np.linalg.norm(det_boxes[:, None, :2] - gt_boxes[None, :, :2], axis=-1)
    for row in range(count):
        if status[row] == STATUS_TP:
            continue
        if max_iou[row] >= MATCH_IOU:
            status[row] = STATUS_DUP
            nearest_gt[row] = int(np.argmax(iou[row]))
        elif max_iou[row] >= IGNORE_IOU or center[row].min() < IGNORE_CENTER:
            status[row] = STATUS_IGNORE
            nearest_gt[row] = int(np.argmin(center[row]))
    return status, gt_index, match_iou, max_iou, nearest_gt


def vehicle_items(objects):
    return [item for item in objects if item["class_name"] in VEHICLES]
