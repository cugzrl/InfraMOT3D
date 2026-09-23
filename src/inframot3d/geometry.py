import math

import numpy as np
from shapely.geometry import MultiPoint, Polygon


def wrap_angle(value):
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def bev_corners(box):
    x, y, _, yaw, length, width, _ = box
    local = np.array(
        [
            [length / 2.0, width / 2.0],
            [-length / 2.0, width / 2.0],
            [-length / 2.0, -width / 2.0],
            [length / 2.0, -width / 2.0],
        ],
        dtype=float,
    )
    cosine, sine = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return local @ rotation.T + np.array([x, y])


def iou_3d(box_a, box_b):
    poly_a = Polygon(bev_corners(box_a))
    poly_b = Polygon(bev_corners(box_b))
    intersection_area = poly_a.intersection(poly_b).area if poly_a.intersects(poly_b) else 0.0
    z_a_min, z_a_max = box_a[2] - box_a[6] / 2.0, box_a[2] + box_a[6] / 2.0
    z_b_min, z_b_max = box_b[2] - box_b[6] / 2.0, box_b[2] + box_b[6] / 2.0
    intersection_height = max(0.0, min(z_a_max, z_b_max) - max(z_a_min, z_b_min))
    intersection = intersection_area * intersection_height
    volume_a = box_a[4] * box_a[5] * box_a[6]
    volume_b = box_b[4] * box_b[5] * box_b[6]
    union = volume_a + volume_b - intersection
    return intersection / union if union > 0.0 else 0.0


def giou_3d(box_a, box_b):
    poly_a = Polygon(bev_corners(box_a))
    poly_b = Polygon(bev_corners(box_b))
    intersection_area = poly_a.intersection(poly_b).area if poly_a.intersects(poly_b) else 0.0
    z_a_min, z_a_max = box_a[2] - box_a[6] / 2.0, box_a[2] + box_a[6] / 2.0
    z_b_min, z_b_max = box_b[2] - box_b[6] / 2.0, box_b[2] + box_b[6] / 2.0
    intersection_height = max(0.0, min(z_a_max, z_b_max) - max(z_a_min, z_b_min))
    intersection = intersection_area * intersection_height
    volume_a = box_a[4] * box_a[5] * box_a[6]
    volume_b = box_b[4] * box_b[5] * box_b[6]
    union = volume_a + volume_b - intersection
    iou = intersection / union if union > 0.0 else 0.0
    points = np.vstack([bev_corners(box_a), bev_corners(box_b)])
    cover_area = MultiPoint(points).convex_hull.area
    cover_height = max(z_a_max, z_b_max) - min(z_a_min, z_b_min)
    cover = cover_area * cover_height
    return iou - (cover - union) / cover if cover > 0.0 else iou


def center_distance(box_a, box_b):
    return float(np.linalg.norm(np.asarray(box_a[:3]) - np.asarray(box_b[:3])))
