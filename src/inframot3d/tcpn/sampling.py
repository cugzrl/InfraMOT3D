"""地图用法 A：候选 BEV 采样方式，五种方式采样点数都为 25

box：候选航向对齐 5x5 网格（与模型内置网格相同）
axis：坐标轴对齐 5x5 网格
random：固定随机偏移，范围与网格相同，随候选航向旋转
lane：按最近车道切向旋转的 5x5 网格，无可用车道时退化为 box
laneline：沿最近车道中心线 -6 m 到 6 m 每 0.5 m 一个点，无可用车道时退化为 box
"""

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from inframot3d.perception.lane_context import _parse_xy

PATTERNS = ("box", "axis", "random", "lane", "laneline")
GRID = np.array([(u, v) for u in (-2.4, -1.2, 0.0, 1.2, 2.4) for v in (-1.6, -0.8, 0.0, 0.8, 1.6)], np.float32)
RANDOM = np.random.default_rng(0).uniform([-2.4, -1.6], [2.4, 1.6], size=(25, 2)).astype(np.float32)
LINE = np.arange(-6.0, 6.01, 0.5)
LANE_RADIUS = 4.0
LANE_ANGLE = np.deg2rad(30.0)


def _rot(offset, yaw):
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    return np.stack([c * offset[None, :, 0] - s * offset[None, :, 1], s * offset[None, :, 0] + c * offset[None, :, 1]], -1)


class LaneSampler:
    """车道几何在世界坐标系，逐帧外参转到虚拟激光雷达坐标系"""

    def __init__(self, root):
        root = Path(root)
        self.map_dir = root / "data/v2x_seq_maps"
        self.data_root = root / "data/v2x-seq-infrastructure"
        self.frames = {}
        for item in json.loads((self.data_root / "data_info.json").read_text()):
            self.frames[(str(item["sequence_id"]), str(item["frame_id"]))] = item
        self.maps = {}
        self.poses = {}

    def _map(self, loc):
        if loc not in self.maps:
            payload = json.loads((self.map_dir / ("%s.json" % loc)).read_text(encoding="utf-8"))
            lanes, pts, owner = {}, [], []
            for lid, lane in payload["LANE"].items():
                if lane.get("lane_type") == "BIKING":
                    continue
                xy = _parse_xy(lane.get("centerline", []))
                if len(xy) < 2:
                    continue
                lanes[lid] = {"xy": xy, "succ": [s for s in lane.get("successors", []) if s != "None"], "pred": [p for p in lane.get("predecessors", []) if p != "None"]}
                pts.append(xy)
                owner += [(lid, i) for i in range(len(xy))]
            self.maps[loc] = (lanes, cKDTree(np.concatenate(pts)), owner)
        return self.maps[loc]

    def _pose(self, sequence_id, frame_id):
        key = (str(sequence_id), str(frame_id))
        if key not in self.poses:
            info = self.frames[key]
            pose = json.loads((self.data_root / info["calib_virtuallidar_to_world_path"]).read_text())
            rot = np.asarray(pose["rotation"], np.float64)
            self.poses[key] = (rot, np.asarray(pose["translation"], np.float64).reshape(3), info["intersection_loc"])
        return self.poses[key]

    def _polyline(self, lanes, lid):
        lane = lanes[lid]
        parts = []
        if lane["pred"] and lane["pred"][0] in lanes:
            parts.append(lanes[lane["pred"][0]]["xy"][:-1])
        start = sum(len(p) for p in parts)
        parts.append(lane["xy"])
        if lane["succ"] and lane["succ"][0] in lanes:
            parts.append(lanes[lane["succ"][0]]["xy"][1:])
        return np.concatenate(parts), start

    def lane_geometry(self, sequence_id, frame_id, cbox):
        """返回每个候选的车道切向（雷达系）、车道中心线采样点（雷达系）和是否有效"""
        rot, trans, loc = self._pose(sequence_id, frame_id)
        lanes, tree, owner = self._map(loc)
        n = len(cbox)
        yaw = np.array(cbox[:, 3], np.float64)
        lane_yaw = yaw.copy()
        line = np.zeros((n, len(LINE), 2), np.float32)
        valid = np.zeros(n, bool)
        if n == 0:
            return lane_yaw.astype(np.float32), line, valid
        local = np.concatenate([cbox[:, :2], np.zeros((n, 1))], axis=1)
        world = (rot @ local.T).T + trans
        yaw_off = np.arctan2(rot[1, 0], rot[0, 0])
        for i, hits in enumerate(tree.query_ball_point(world[:, :2], LANE_RADIUS)):
            best, cost = None, 1e9
            for k in hits:
                lid, j = owner[k]
                xy = lanes[lid]["xy"]
                a = min(j, len(xy) - 2)
                d = xy[a + 1] - xy[a]
                diff = (np.arctan2(d[1], d[0]) - (yaw[i] + yaw_off) + np.pi / 2) % np.pi - np.pi / 2
                if abs(diff) > LANE_ANGLE:
                    continue
                c = np.hypot(*(xy[j] - world[i, :2])) + 4.0 * abs(diff)
                if c < cost:
                    best, cost = (lid, j), c
            if best is None:
                continue
            poly, start = self._polyline(lanes, best[0])
            seg = np.hypot(*np.diff(poly, axis=0).T)
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            # 候选在中心线上的投影弧长
            j = start + best[1]
            lo, hi = max(j - 1, 0), min(j + 1, len(poly) - 1)
            ab = poly[hi] - poly[lo]
            t = np.clip(np.dot(world[i, :2] - poly[lo], ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
            s0 = cum[lo] + t * (cum[hi] - cum[lo])
            s = np.clip(s0 + LINE, 0, cum[-1])
            wx, wy = np.interp(s, cum, poly[:, 0]), np.interp(s, cum, poly[:, 1])
            pts = np.stack([wx, wy, np.full_like(wx, world[i, 2])], -1)
            line[i] = ((rot.T @ (pts - trans).T).T)[:, :2]
            tangent_world = np.arctan2(ab[1], ab[0])
            lane_yaw[i] = tangent_world - yaw_off
            valid[i] = True
        return lane_yaw.astype(np.float32), line, valid


def candidate_sites(pattern, cbox, lane=None):
    """返回 [N,25,2] 雷达系采样点"""
    xy = cbox[:, None, :2].astype(np.float32)
    yaw = cbox[:, 3].astype(np.float32)
    if pattern == "box":
        return xy + _rot(GRID, yaw)
    if pattern == "axis":
        return xy + GRID[None]
    if pattern == "random":
        return xy + _rot(RANDOM, yaw)
    lane_yaw, line, valid = lane
    if pattern == "lane":
        return xy + _rot(GRID, np.where(valid, lane_yaw, yaw))
    if pattern == "laneline":
        box = xy + _rot(GRID, yaw)
        return np.where(valid[:, None, None], line, box).astype(np.float32)
    raise ValueError(pattern)
