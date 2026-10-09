"""把固定路侧高精地图变到虚拟激光雷达坐标系，供查询采样使用"""

import json
import re
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

PAIR = re.compile(r"\(([-\d.eE]+),\s*([-\d.eE]+)\)")
VEHICLE_RANGE = 8.0


def _parse_xy(items):
    points = []
    for item in items:
        found = PAIR.findall(item if isinstance(item, str) else "")
        if found:
            points.append((float(found[0][0]), float(found[0][1])))
    return np.asarray(points, dtype=np.float64)


class LaneContext:
    def __init__(self, map_dir):
        self.map_dir = Path(map_dir)
        self._cache = {}

    def _index(self, intersection):
        if intersection in self._cache:
            return self._cache[intersection]
        path = self.map_dir / ("%s.json" % intersection)
        payload = json.loads(path.read_text(encoding="utf-8"))
        centers = []
        tangents = []
        for lane in payload["LANE"].values():
            pts = _parse_xy(lane.get("centerline", []))
            if len(pts) < 2:
                continue
            delta = np.diff(pts, axis=0)
            delta = np.vstack([delta, delta[-1]])
            delta = delta / np.clip(np.linalg.norm(delta, axis=1, keepdims=True), 1e-6, None)
            centers.append(pts)
            tangents.append(delta)
        xy = np.concatenate(centers, axis=0)
        tangent = np.concatenate(tangents, axis=0)
        self._cache[intersection] = (cKDTree(xy), tangent)
        return self._cache[intersection]

    def tangent_lidar(self, intersection, rotation, translation, xy):
        """返回车道切向和是否落在车道附近

        xy 是虚拟激光雷达平面坐标
        切向同样转到该坐标系
        距离中心线超过 8 m 时不使用车道方向
        """
        tree, tangent = self._index(intersection)
        local = np.asarray(xy, dtype=np.float64)
        if local.ndim == 1:
            local = local.reshape(1, 2)
        zeros = np.zeros((len(local), 1), dtype=np.float64)
        world = (rotation @ np.concatenate([local, zeros], axis=1).T).T + translation.reshape(1, 3)
        dist, index = tree.query(world[:, :2], k=1)
        world_tangent = tangent[index]
        lidar_tangent = (rotation[:2, :2].T @ world_tangent.T).T
        valid = dist <= VEHICLE_RANGE
        lidar_tangent = lidar_tangent * valid[:, None]
        return lidar_tangent.astype(np.float32), valid.astype(np.float32), dist.astype(np.float32)
