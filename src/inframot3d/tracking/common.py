import math

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

from inframot3d.geometry import center_distance, giou_3d, iou_3d, wrap_angle


def _time_scale(raw):
    magnitude = abs(int(raw))
    if magnitude >= 10**14:
        return 1.0e-6
    if magnitude >= 10**11:
        return 1.0e-3
    return 1.0


def elapsed_seconds(timestamp, origin):
    """相对首帧的秒。V2X-Seq 原始时间是 Unix 微秒，先做整数差再换算。"""
    if isinstance(timestamp, float) and abs(timestamp) < 1.0e11:
        if origin is None:
            return 0.0, float(timestamp)
        return float(timestamp) - float(origin), origin
    raw = int(timestamp)
    if origin is None:
        return 0.0, raw
    return (raw - int(origin)) * _time_scale(origin), origin


def class_groups(tracker_config):
    for name, value in tracker_config.items():
        if isinstance(value, dict) and "classes" in value:
            yield name, value


def public_box(box):
    values = [float(value) for value in np.asarray(box, dtype=float).reshape(-1)[:7]]
    if len(values) != 7 or not all(math.isfinite(value) for value in values):
        raise ValueError("跟踪框出现非有限数值")
    return values


def geometry_box(box):
    values = np.asarray(box, dtype=float).reshape(-1).copy()
    values[4:7] = np.maximum(values[4:7], 1.0e-3)
    return values


def _orientation_diff(diff):
    if diff > np.pi / 2.0:
        diff -= np.pi
    if diff < -np.pi / 2.0:
        diff += np.pi
    return float(diff)


def overlap(box_a, box_b, metric):
    left = geometry_box(box_a)
    right = geometry_box(box_b)
    if metric == "giou":
        return float(giou_3d(left, right))
    if metric == "iou":
        return float(iou_3d(left, right))
    raise ValueError(f"不支持的重叠度量{metric}")


def pair_cost(box_a, box_b, metric):
    if metric in {"giou", "iou"}:
        return 1.0 - overlap(box_a, box_b, metric)
    if metric == "center_distance":
        return center_distance(geometry_box(box_a), geometry_box(box_b))
    if metric == "euler":
        diff = geometry_box(box_a) - geometry_box(box_b)
        diff[3] = _orientation_diff(float(diff[3]))
        return float(np.sqrt(np.dot(diff, diff)))
    raise ValueError(f"不支持的匹配度量{metric}")


def _greedy_pairs(cost):
    num_tracks = cost.shape[1]
    used_rows, used_columns = set(), set()
    pairs = []
    for index in np.argsort(cost, axis=None):
        row, column = divmod(int(index), num_tracks)
        if row in used_rows or column in used_columns:
            continue
        used_rows.add(row)
        used_columns.add(column)
        pairs.append((row, column))
    return pairs


def associate(detections, tracks, settings):
    if not detections or not tracks:
        return [], list(range(len(detections))), list(range(len(tracks)))
    metric = settings["metric"]
    threshold = float(settings["threshold"])
    cost = np.empty((len(detections), len(tracks)), dtype=float)
    for row, detection in enumerate(detections):
        for column, track in enumerate(tracks):
            value = pair_cost(detection["box"], track.box, metric)
            cost[row, column] = value if math.isfinite(value) else 1.0e6
    association = settings["association"]
    if association in {"bipartite", "hungarian"}:
        rows, columns = linear_sum_assignment(cost)
        pairs = list(zip(rows.tolist(), columns.tolist()))
    elif association == "greedy":
        pairs = _greedy_pairs(cost)
    else:
        raise ValueError(f"不支持的匹配算法{association}")
    matches = []
    used_rows, used_columns = set(), set()
    for row, column in pairs:
        if cost[row, column] <= threshold:
            matches.append((int(row), int(column)))
            used_rows.add(int(row))
            used_columns.add(int(column))
    unmatched_detections = [index for index in range(len(detections)) if index not in used_rows]
    unmatched_tracks = [index for index in range(len(tracks)) if index not in used_columns]
    return matches, unmatched_detections, unmatched_tracks


class MotionKalman:
    """常速度 Kalman。位置通道的状态转移使用秒为单位的时间差。"""

    def __init__(self, box, timestamp_seconds, motion):
        self.kf = KalmanFilter(dim_x=10, dim_z=7)
        self.kf.F = np.eye(10)
        self.kf.F[0, 7] = 1.0
        self.kf.F[1, 8] = 1.0
        self.kf.F[2, 9] = 1.0
        self.kf.H = np.zeros((7, 10))
        self.kf.H[:7, :7] = np.eye(7)
        self.kf.P[7:, 7:] *= float(motion["velocity_p_scale"])
        self.kf.P *= float(motion["p_scale"])
        self.kf.Q *= float(motion["q_scale"])
        self.kf.R *= float(motion["r_scale"])
        self.kf.B = np.zeros((10, 1))
        self.kf.x[:7, 0] = np.asarray(box, dtype=float).reshape(7)
        self.prev_time = float(timestamp_seconds)
        self.latest_time = float(timestamp_seconds)
        self.state_box = public_box(box)

    def get_prediction(self, timestamp_seconds):
        time_lag = float(timestamp_seconds) - self.prev_time
        self.latest_time = float(timestamp_seconds)
        self.kf.F = np.eye(10)
        self.kf.F[0, 7] = time_lag
        self.kf.F[1, 8] = time_lag
        self.kf.F[2, 9] = time_lag
        prior = np.asarray(self.kf.F @ self.kf.x, dtype=float).reshape(-1)
        box = prior[:7].copy()
        box[3] = wrap_angle(float(box[3]))
        self.state_box = public_box(box)
        return self.state_box

    def commit(self, measurement):
        self.kf.predict()
        self.kf.x[3, 0] = wrap_angle(float(self.kf.x[3, 0]))
        observation = np.asarray(measurement, dtype=float).reshape(-1).copy()
        observation[3] = wrap_angle(float(observation[3]))
        predicted_yaw = float(self.kf.x[3, 0])
        measured_yaw = float(observation[3])
        if np.pi / 2.0 < abs(measured_yaw - predicted_yaw) < np.pi * 1.5:
            predicted_yaw = wrap_angle(predicted_yaw + np.pi)
        if abs(measured_yaw - predicted_yaw) >= np.pi * 1.5:
            predicted_yaw += 2.0 * np.pi if measured_yaw > 0.0 else -2.0 * np.pi
        self.kf.x[3, 0] = predicted_yaw
        self.kf.update(observation)
        self.kf.x[3, 0] = wrap_angle(float(self.kf.x[3, 0]))
        self.prev_time = self.latest_time
        self.state_box = public_box(self.kf.x[:7, 0])
        return self.state_box


class HitManager:
    def __init__(self, frame_index, min_hits, max_age):
        self.time_since_update = 0
        self.hits = 1
        self.age = 0
        self.min_hits = int(min_hits)
        self.max_age = None if max_age is None else int(max_age)
        self.recent_state = 1
        self.state = "alive" if frame_index <= self.min_hits or self.min_hits == 0 else "birth"

    def predict(self):
        self.age += 1
        self.time_since_update += 1

    def update(self, mode, frame_index):
        self.recent_state = int(mode)
        if int(mode) != 0:
            self.time_since_update = 0
            self.hits += 1
        if self.state == "birth":
            if self.hits >= self.min_hits or frame_index <= self.min_hits:
                self.state = "alive"
            elif self._expired():
                self.state = "dead"
        elif self.state == "alive" and self._expired():
            self.state = "dead"

    def _expired(self):
        return self.max_age is not None and self.time_since_update >= self.max_age


class MotionTrack:
    def __init__(self, detection, track_id, frame_index, timestamp_seconds, settings):
        self.settings = settings
        self.track_id = int(track_id)
        self.class_name = detection["class_name"]
        self.score = float(detection.get("score", 1.0))
        self.source_track_id = detection.get("source_track_id")
        self.score_decay = float(settings["motion"]["score_decay"])
        self.commit_modes = {int(value) for value in settings["commit_modes"]}
        self.output_rule = settings["output_rule"]
        self.max_output_age = int(settings.get("max_output_age", 0))
        self.motion = MotionKalman(detection["box"], timestamp_seconds, settings["motion"])
        self.box = list(self.motion.state_box)
        self.life = HitManager(frame_index, settings["min_hits"], settings.get("max_age"))

    def predict(self, timestamp_seconds):
        self.box = self.motion.get_prediction(timestamp_seconds)
        self.life.predict()
        self.score *= self.score_decay
        return self.box

    def update(self, frame_index, mode, detection=None):
        if int(mode) in self.commit_modes:
            if int(mode) == 1:
                measurement = detection["box"]
                self.score = float(detection.get("score", 1.0))
                self.source_track_id = detection.get("source_track_id")
            else:
                measurement = self.box
                self.source_track_id = None
            self.box = self.motion.commit(measurement)
        else:
            self.source_track_id = None
        self.life.update(mode, frame_index)

    def publish(self):
        if self.life.state == "dead":
            return False
        if self.output_rule == "associated_or_birth":
            if self.life.state == "birth":
                return True
            return self.life.state == "alive" and self.life.recent_state == 1
        if self.output_rule == "recent":
            return self.life.state == "alive" and self.life.time_since_update <= self.max_output_age
        raise ValueError(f"不支持的输出规则{self.output_rule}")

    def export(self):
        return {
            "class_name": self.class_name,
            "track_id": self.track_id,
            "score": float(self.score),
            "box": public_box(self.box),
            "source_track_id": self.source_track_id if self.life.time_since_update == 0 else None,
            "time_since_update": int(self.life.time_since_update),
        }


class MultiClassRunner:
    def __init__(self, tracker_config, builder):
        motion = tracker_config.get("motion")
        if not isinstance(motion, dict):
            raise ValueError("tracker.motion 缺失")
        for key in ("p_scale", "velocity_p_scale", "q_scale", "r_scale", "score_decay"):
            if key not in motion:
                raise ValueError(f"tracker.motion.{key} 缺失")
        self.frame_dt = float(tracker_config["frame_dt"])
        if self.frame_dt <= 0.0:
            raise ValueError("frame_dt 必须为正")
        self.score_threshold = float(tracker_config.get("score_threshold", 0.0))
        self.synthetic_time = 0.0
        self.time_origin = None
        self.trackers = {}
        for _, group in class_groups(tracker_config):
            for class_name in group["classes"]:
                settings = dict(group)
                settings["score_threshold"] = self.score_threshold
                settings["motion"] = motion
                self.trackers[class_name] = builder(settings)
        if not self.trackers:
            raise ValueError("tracker 没有类别配置")
        self.next_id = 1

    def update(self, objects, timestamp=None):
        seconds = self._seconds(timestamp)
        grouped = {class_name: [] for class_name in self.trackers}
        for value in objects:
            class_name = value["class_name"]
            if class_name in grouped:
                grouped[class_name].append(value)
        outputs = []
        for class_name in sorted(self.trackers):
            class_outputs, self.next_id = self.trackers[class_name].update(
                grouped[class_name], self.next_id, seconds
            )
            outputs.extend(class_outputs)
        return sorted(outputs, key=lambda value: value["track_id"])

    def _seconds(self, timestamp):
        if timestamp is None:
            current = self.synthetic_time
            self.synthetic_time += self.frame_dt
            return current
        seconds, self.time_origin = elapsed_seconds(timestamp, self.time_origin)
        return seconds
