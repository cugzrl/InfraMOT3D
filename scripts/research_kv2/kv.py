"""改进后的运动 Key-Value 检索

每一项消融只改一个因素
CV 残差把历史未来相对其自身匀速的偏差加到当前匀速上，可靠才修正
"""

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from common import (
    angle_diff,
    blend_futures,
    constant_velocity,
    future_pattern,
    interpolate_future,
    parse_centerline,
)

BASE_VARIANT = {"name": "base", "hist": "hist", "gate": 60.0, "penalty": "region", "yaw": False}
MODE_NAME = {0: "static", 1: "straight", 2: "left", 3: "right"}
MODE_CODE = {name: code for code, name in MODE_NAME.items()}
KEY_STEPS = 5


def hist_times(count, dt):
    return (np.arange(count, dtype=np.float64) - (count - 1)) * float(dt)


def sample_hist5(hist, dt):
    # 11 点历史线性插值到 1 秒内均匀 5 点，含当前点
    hist = np.asarray(hist, dtype=np.float64)
    count = hist.shape[-2]
    pos = np.linspace(0.0, count - 1, KEY_STEPS)
    low = np.floor(pos).astype(np.int64)
    high = np.minimum(low + 1, count - 1)
    weight = (pos - low).reshape((1,) * (hist.ndim - 2) + (KEY_STEPS, 1))
    return hist[..., low, :] * (1.0 - weight) + hist[..., high, :] * weight


def cv_path(hist, dt, steps):
    return constant_velocity(np.asarray(hist, dtype=np.float64), dt)[:steps]


def ca_path(hist, dt, steps):
    from common import constant_acceleration

    return constant_acceleration(np.asarray(hist, dtype=np.float64), dt)[:steps]


def ctrv_path(hist, dt, steps):
    # 当前坐标系以前向为 x，角速度来自最近 0.5 s 与再前 0.5 s 的位移方向
    hist = np.asarray(hist, dtype=np.float64)
    mid = hist.shape[0] // 2
    now = hist[-1] - hist[mid]
    prev = hist[mid] - hist[max(0, mid - 5)]
    speed = float(np.linalg.norm(now) / 0.5)
    heading = float(np.arctan2(now[1], now[0])) if np.linalg.norm(now) > 0.2 else 0.0
    if np.linalg.norm(prev) > 0.2 and np.linalg.norm(now) > 0.2:
        yaw_rate = float(angle_diff(heading, np.arctan2(prev[1], prev[0])) / 0.5)
    else:
        yaw_rate = 0.0
    times = (np.arange(1, steps + 1, dtype=np.float64)) * float(dt)
    if abs(yaw_rate) < 1e-3:
        path = np.stack([speed * np.cos(heading) * times, speed * np.sin(heading) * times], axis=1)
    else:
        path = np.stack(
            [
                speed / yaw_rate * (np.sin(heading + yaw_rate * times) - np.sin(heading)),
                speed / yaw_rate * (-np.cos(heading + yaw_rate * times) + np.cos(heading)),
            ],
            axis=1,
        )
    return path.astype(np.float64)


def mode_of_future(future, config):
    future = np.asarray(future, dtype=np.float64)
    if future.ndim != 2 or not np.isfinite(future[-1]).all():
        return -1
    coarse = future_pattern(future, config)
    if coarse == "static":
        return MODE_CODE["static"]
    if coarse == "straight":
        return MODE_CODE["straight"]
    return MODE_CODE["left"] if float(future[-1, 1]) >= 0.0 else MODE_CODE["right"]


def speed_tag(future, speed, config):
    future = np.asarray(future, dtype=np.float64)
    if future.shape[0] < 20 or not np.isfinite(future[9]).all() or not np.isfinite(future[-1]).all():
        return "na"
    later = float(np.linalg.norm(future[-1] - future[9]) / 1.0)
    delta = later - float(speed)
    limit = float(config["motion"]["speed_change_m_s"])
    if delta > limit:
        return "accel"
    if delta < -limit:
        return "decel"
    return "cruise"


def batch_cv(hist, dt, steps):
    mid = hist.shape[1] // 2
    velocity = (hist[:, -1, :] - hist[:, mid, :]) / 0.5
    times = (np.arange(1, steps + 1, dtype=np.float64)) * float(dt)
    return velocity[:, None, :] * times[None, :, None]


def batch_mode(future, config):
    motion = config["motion"]
    end = future[:, -1, :]
    distance = np.linalg.norm(end, axis=1)
    heading = np.abs(np.arctan2(end[:, 1], end[:, 0]))
    turn = (heading > np.deg2rad(float(motion["turn_yaw_deg"]))) | (np.abs(end[:, 1]) > float(motion["turn_lateral_m"]))
    mode = np.full(len(future), MODE_CODE["straight"], dtype=np.int8)
    mode[distance < float(motion["static_disp_m"])] = MODE_CODE["static"]
    left = turn & (end[:, 1] >= 0.0)
    right = turn & (end[:, 1] < 0.0)
    mode[left] = MODE_CODE["left"]
    mode[right] = MODE_CODE["right"]
    bad = ~np.isfinite(end).all(axis=1)
    mode[bad] = -1
    return mode


class LaneGraph:
    """车道 ID 与一跳后继，不用 STRAIGHT 或 LEFT 这类粗标签"""

    def __init__(self, map_root):
        self.map_root = Path(map_root)
        self._cache = {}

    def _load(self, intersection):
        if intersection in self._cache:
            return self._cache[intersection]
        raw = json.loads((self.map_root / ("%s.json" % intersection)).read_text(encoding="utf-8"))
        lanes = {}
        for lane_id, lane in raw["LANE"].items():
            if lane.get("lane_type") == "BIKING":
                continue
            points = parse_centerline(lane.get("centerline", []))
            if len(points) < 2:
                continue
            step_length = float(np.median(np.linalg.norm(np.diff(points, axis=0), axis=1)))
            step = max(1, int(round(2.0 / max(step_length, 1e-3))))
            succ = [str(item) for item in lane.get("successors", []) if str(item) not in ("", "None")]
            lanes[str(lane_id)] = {"xy": points[::step], "succ": succ}
        names = sorted(lanes)
        code = {name: index for index, name in enumerate(names)}
        xy = []
        owner = []
        matrix = np.zeros((len(names), len(names)), dtype=bool)
        for name in names:
            lane = lanes[name]
            xy.append(lane["xy"])
            owner.append(np.full(len(lane["xy"]), code[name], dtype=np.int32))
            for item in lane["succ"]:
                if item in code:
                    matrix[code[name], code[item]] = True
        points = np.concatenate(xy, axis=0)
        self._cache[intersection] = {
            "tree": cKDTree(points),
            "owner": np.concatenate(owner),
            "succ": matrix,
        }
        return self._cache[intersection]

    def code_at(self, intersection, world_xy, radius):
        graph = self._load(intersection)
        point = np.asarray(world_xy, dtype=np.float64).reshape(1, 2)
        dist, index = graph["tree"].query(point, k=1)
        if float(dist[0]) > float(radius):
            return -1
        return int(graph["owner"][int(index[0])])

    def penalty(self, intersection, query_code, bank_codes, other_penalty, succ_penalty):
        bank_codes = np.asarray(bank_codes, dtype=np.int32)
        penalty = np.zeros(len(bank_codes), dtype=np.float64)
        if int(query_code) < 0:
            return penalty
        graph = self._load(intersection)
        known = bank_codes >= 0
        penalty[known] = float(other_penalty)
        same = known & (bank_codes == int(query_code))
        penalty[same] = 0.0
        if known.any():
            linked = graph["succ"][int(query_code)]
            codes = bank_codes.copy()
            codes[~known] = 0
            neighbor = known & ~same & (linked[codes] | graph["succ"][codes, int(query_code)])
            penalty[neighbor] = float(succ_penalty)
        return penalty


def augment_bank(bank, intersection, lane_graph, config):
    hist = np.asarray(bank["hist"], dtype=np.float64)
    future = np.asarray(bank["future"], dtype=np.float64)
    dt = float(config["dt_s"])
    steps = int(round(float(config["future_s"]) / dt))
    cv = batch_cv(hist, dt, steps)
    radius = float(config["retrieval"]["lane_radius_m"])
    lanes = np.array([lane_graph.code_at(intersection, point, radius) for point in bank["origin"]], dtype=np.int32)
    out = {key: np.asarray(bank[key]) for key in bank}
    out["hist5"] = sample_hist5(hist, dt).astype(np.float32)
    out["residual"] = (future[:, :steps] - cv).astype(np.float32)
    out["mode"] = batch_mode(future, config)
    out["lane"] = lanes
    return out


def _sequence_mask(bank, allowed):
    names = [str(name) for name in bank["sequence_names"]]
    allowed_ids = [names.index(str(name)) for name in allowed if str(name) in names]
    if not allowed_ids:
        return np.zeros(len(bank["class_id"]), dtype=bool)
    return np.isin(bank["sequence_id"], np.asarray(allowed_ids, dtype=bank["sequence_id"].dtype))


def select_neighbors(bank, query, allowed, variant, config, lane_graph):
    # 同类别、允许序列、空间门内，每条独立轨迹只留一个最近样本
    spec = config["retrieval"]
    names = [str(name) for name in bank["class_names"]]
    if query["class_name"] not in names:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    class_id = names.index(query["class_name"])
    mask = (bank["class_id"] == class_id) & _sequence_mask(bank, allowed)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return indices, np.zeros(0, dtype=np.float64)
    spatial = np.linalg.norm(bank["origin"][indices] - query["origin"], axis=1)
    keep = spatial <= float(variant["gate"])
    indices = indices[keep]
    spatial = spatial[keep]
    if indices.size == 0:
        return indices, np.zeros(0, dtype=np.float64)
    hist_name = variant["hist"]
    hist = np.asarray(bank[hist_name][indices], dtype=np.float64)
    query_hist = np.asarray(query[hist_name], dtype=np.float64)
    motion = np.sqrt(np.mean((hist - query_hist) ** 2, axis=(1, 2))) / float(spec["motion_sigma_m"])
    speed = np.abs(bank["speed"][indices] - query["speed"]) / float(spec["speed_sigma_m_s"])
    heading = np.abs(angle_diff(bank["heading"][indices], query["heading"])) / np.deg2rad(float(spec["heading_sigma_deg"]))
    cost = spatial / float(spec["spatial_sigma_m"]) + motion + speed + heading
    if variant["yaw"]:
        yaw = np.abs(angle_diff(bank["yaw_rate"][indices], query["yaw_rate"])) / np.deg2rad(float(spec["yaw_sigma_deg_s"]))
        cost = cost + yaw
    if variant["penalty"] == "region":
        region_names = bank["region_names"]
        bank_region = region_names[bank["region"][indices]]
        differ = (bank_region != query["region"]) & (bank_region != "OFFROAD") & (query["region"] != "OFFROAD")
        cost = cost + differ.astype(np.float64) * float(spec["region_penalty"])
    else:
        cost = cost + lane_graph.penalty(
            query["intersection"],
            int(query["lane"]),
            bank["lane"][indices],
            spec["region_penalty"],
            spec["successor_penalty"],
        )
    order = np.argsort(cost, kind="mergesort")
    chosen = []
    chosen_cost = []
    seen = set()
    limit = int(spec["top_k"])
    for position in order:
        uid = int(bank["uid"][indices[position]])
        if uid in seen:
            continue
        seen.add(uid)
        chosen.append(int(indices[position]))
        chosen_cost.append(float(cost[position]))
        if len(chosen) >= limit:
            break
    return np.asarray(chosen, dtype=np.int64), np.asarray(chosen_cost, dtype=np.float64)


def mode_blend(futures, costs, modes):
    # 只平均与 Top-1 相同未来模式的候选，不把左转和右转合成一条
    if len(futures) == 0:
        return None
    keep = np.asarray(modes) == int(modes[0])
    return blend_futures(np.asarray(futures)[keep], np.asarray(costs)[keep])


def residual_forecast(cv, residuals, costs, modes, config):
    # 残差定义在各自朝向坐标里，再加到当前目标自己的匀速上
    if len(residuals) == 0:
        return cv, False
    if float(costs[0]) > float(config["retrieval"]["residual_cost_max"]):
        return cv, False
    keep = np.asarray(modes) == int(modes[0])
    chosen = np.asarray(residuals, dtype=np.float64)[keep]
    chosen_cost = np.asarray(costs, dtype=np.float64)[keep]
    if len(chosen) >= 2:
        spread = float(np.linalg.norm(np.std(chosen[:, -1, :], axis=0)))
        if spread > float(config["retrieval"]["residual_endpoint_std_m"]):
            return cv, False
    correction = blend_futures(chosen, chosen_cost)
    return cv + correction, True


def oracle_index(candidates, future, by_fde):
    error = np.linalg.norm(candidates - future, axis=-1)
    score = error[:, -1] if by_fde else error.mean(axis=1)
    return int(np.argmin(score))


def pointwise_error(candidates, future):
    # 每个时刻单独取最小误差，会把不同轨迹拼起来，只作为 Oracle
    error = np.linalg.norm(np.asarray(candidates) - np.asarray(future), axis=-1)
    return error.min(axis=0)


def local_to_lidar(local_xy, heading, rotation, lidar_xy):
    from common import from_local

    delta_world = from_local(np.asarray(local_xy, dtype=np.float64), np.zeros(2), heading)
    delta_lidar = rotation[:2, :2].T @ delta_world
    return np.asarray(lidar_xy, dtype=np.float64) + delta_lidar


def at_time(path, tau, dt):
    return interpolate_future(path, tau, dt)
