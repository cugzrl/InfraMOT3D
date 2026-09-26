import copy

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

from inframot3d.geometry import center_distance, giou_3d, wrap_angle


class KalmanBox:
    def __init__(self, detection, track_id):
        self.filter = KalmanFilter(dim_x=10, dim_z=7)
        self.filter.F = np.eye(10)
        self.filter.F[0, 7] = 1.0
        self.filter.F[1, 8] = 1.0
        self.filter.F[2, 9] = 1.0
        self.filter.H = np.zeros((7, 10))
        self.filter.H[:7, :7] = np.eye(7)
        self.filter.P[7:, 7:] *= 1000.0
        self.filter.P *= 10.0
        self.filter.Q[7:, 7:] *= 0.01
        self.filter.x[:7, 0] = np.asarray(detection["box"], dtype=float)
        self.track_id = int(track_id)
        self.class_name = detection["class_name"]
        self.score = float(detection.get("score", 1.0))
        self.source_track_id = detection.get("source_track_id")
        self.hits = 1
        self.age = 0
        self.time_since_update = 0

    def predict(self):
        self.filter.predict()
        self.filter.x[3, 0] = wrap_angle(self.filter.x[3, 0])
        self.age += 1
        self.time_since_update += 1
        return self.box

    def update(self, detection):
        observation = np.asarray(detection["box"], dtype=float).copy()
        predicted_yaw = wrap_angle(float(self.filter.x[3, 0]))
        observed_yaw = wrap_angle(float(observation[3]))
        difference = abs(observed_yaw - predicted_yaw)
        if np.pi / 2.0 < difference < 3.0 * np.pi / 2.0:
            predicted_yaw = wrap_angle(predicted_yaw + np.pi)
        if abs(observed_yaw - predicted_yaw) >= 3.0 * np.pi / 2.0:
            predicted_yaw += 2.0 * np.pi if observed_yaw > 0.0 else -2.0 * np.pi
        self.filter.x[3, 0] = predicted_yaw
        self.filter.update(observation)
        self.filter.x[3, 0] = wrap_angle(self.filter.x[3, 0])
        self.score = float(detection.get("score", 1.0))
        self.source_track_id = detection.get("source_track_id")
        self.hits += 1
        self.time_since_update = 0

    @property
    def box(self):
        return self.filter.x[:7, 0].astype(float).tolist()


def _affinity(detections, tracks, metric):
    matrix = np.empty((len(detections), len(tracks)), dtype=float)
    for row, detection in enumerate(detections):
        for column, track in enumerate(tracks):
            if metric == "giou_3d":
                matrix[row, column] = giou_3d(detection["box"], track.box)
            elif metric == "center_distance":
                matrix[row, column] = -center_distance(detection["box"], track.box)
            else:
                raise ValueError(f"不支持的匹配度量{metric}")
    return matrix


def _associate(detections, tracks, settings, return_debug=False):
    if not detections or not tracks:
        result = ([], list(range(len(detections))), list(range(len(tracks))))
        if not return_debug:
            return result
        threshold = float(settings["threshold"])
        if settings["metric"] == "center_distance":
            threshold = -threshold
        return (*result, {
            "metric": settings["metric"],
            "threshold": threshold,
            "affinity_matrix": [],
            "gate_mask": [],
            "candidate_pairs": [],
        })
    matrix = _affinity(detections, tracks, settings["metric"])
    threshold = float(settings["threshold"])
    if settings["metric"] == "center_distance":
        threshold = -threshold
    matches = []
    if settings["association"] == "hungarian":
        rows, columns = linear_sum_assignment(-matrix)
        candidates = zip(rows.tolist(), columns.tolist())
    elif settings["association"] == "greedy":
        candidates = []
        used_rows, used_columns = set(), set()
        for flat_index in np.argsort(matrix, axis=None)[::-1]:
            row, column = np.unravel_index(flat_index, matrix.shape)
            if row not in used_rows and column not in used_columns:
                candidates.append((int(row), int(column)))
                used_rows.add(int(row))
                used_columns.add(int(column))
    else:
        raise ValueError(f"不支持的匹配算法{settings['association']}")
    for row, column in candidates:
        if matrix[row, column] >= threshold:
            matches.append((row, column))
    matched_detections = {row for row, _ in matches}
    matched_tracks = {column for _, column in matches}
    unmatched_detections = [index for index in range(len(detections)) if index not in matched_detections]
    unmatched_tracks = [index for index in range(len(tracks)) if index not in matched_tracks]
    result = (matches, unmatched_detections, unmatched_tracks)
    if not return_debug:
        return result
    return (*result, {
        "metric": settings["metric"],
        "threshold": threshold,
        "affinity_matrix": matrix.tolist(),
        "gate_mask": (matrix >= threshold).tolist(),
        "candidate_pairs": [[int(row), int(column)] for row, column in candidates],
    })


class ClassTracker:
    def __init__(self, settings):
        self.settings = copy.deepcopy(settings)
        self.tracks = []
        self.frame_count = 0

    def update(self, detections, id_start, debug=False):
        self.frame_count += 1
        for track in self.tracks:
            track.predict()
        pre_tracks = [
            {
                "track_id": int(track.track_id),
                "box": [float(value) for value in track.box],
                "hits": int(track.hits),
                "time_since_update": int(track.time_since_update),
            }
            for track in self.tracks
        ]
        if debug:
            matches, unmatched_detections, _, association_debug = _associate(
                detections, self.tracks, self.settings, return_debug=True
            )
        else:
            matches, unmatched_detections, _ = _associate(detections, self.tracks, self.settings)
        for detection_index, track_index in matches:
            self.tracks[track_index].update(detections[detection_index])
        next_id = id_start
        created = []
        for detection_index in unmatched_detections:
            current_id = next_id
            self.tracks.append(KalmanBox(detections[detection_index], next_id))
            if debug:
                created.append(
                    {
                        "input_index": int(detections[detection_index].get("_debug_index", detection_index)),
                        "track_id": int(current_id),
                    }
                )
            next_id += 1
        outputs = []
        track_states = []
        for track in self.tracks:
            stable = track.hits >= int(self.settings["min_hits"]) or self.frame_count <= int(self.settings["min_hits"])
            alive = track.time_since_update < int(self.settings["max_age"])
            if debug:
                track_states.append(
                    {
                        "track_id": int(track.track_id),
                        "hits": int(track.hits),
                        "time_since_update": int(track.time_since_update),
                        "stable": bool(stable),
                        "alive": bool(alive),
                        "published": bool(stable and alive),
                    }
                )
            if stable and alive:
                outputs.append(
                    {
                        "class_name": track.class_name,
                        "track_id": track.track_id,
                        "score": track.score,
                        "box": track.box,
                        "source_track_id": track.source_track_id if track.time_since_update == 0 else None,
                        "time_since_update": track.time_since_update,
                    }
                )
        self.tracks = [
            track for track in self.tracks if track.time_since_update < int(self.settings["max_age"])
        ]
        if not debug:
            return outputs, next_id
        snapshot = {
            "candidate_detections": [
                {
                    "input_index": int(item.get("_debug_index", index)),
                    "score": float(item.get("score", 1.0)),
                    "box": [float(value) for value in item["box"]],
                }
                for index, item in enumerate(detections)
            ],
            "pre_tracks": pre_tracks,
            "association": association_debug,
            "assignments": [
                {
                    "input_index": int(detections[detection_index].get("_debug_index", detection_index)),
                    "track_id": int(pre_tracks[track_index]["track_id"]),
                }
                for detection_index, track_index in matches
            ],
            "created": created,
            "track_states": track_states,
            "output_track_ids": [int(item["track_id"]) for item in outputs],
        }
        return outputs, next_id, snapshot


class MultiClassAB3DMOT:
    def __init__(self, tracker_config):
        self.class_settings = {}
        for group_name, group in tracker_config.items():
            if group_name in {"name", "score_threshold", "score_thresholds", "score_thresholds_file"}:
                continue
            if not isinstance(group, dict) or "classes" not in group:
                continue
            for class_name in group["classes"]:
                self.class_settings[class_name] = group
        self.trackers = {
            class_name: ClassTracker(settings) for class_name, settings in self.class_settings.items()
        }
        self.score_threshold = float(tracker_config.get("score_threshold", 0.0))
        self.score_thresholds = {
            str(name): float(value) for name, value in (tracker_config.get("score_thresholds") or {}).items()
        }
        self.next_id = 1
        self.debug_enabled = False
        self.last_debug = None

    def _class_threshold(self, class_name):
        if class_name in self.score_thresholds:
            return self.score_thresholds[class_name]
        return self.score_threshold

    def update(self, objects, timestamp=None):
        # 统一接口保留 timestamp。AB3DMOT 的状态转移固定按一帧，不使用时间戳。
        del timestamp
        grouped = {class_name: [] for class_name in self.trackers}
        indexed = []
        for index, value in enumerate(objects):
            current = dict(value)
            if self.debug_enabled:
                current["_debug_index"] = int(index)
            indexed.append(current)
        filtered = []
        for value in indexed:
            class_name = value["class_name"]
            if class_name in grouped and float(value.get("score", 1.0)) >= self._class_threshold(class_name):
                grouped[class_name].append(value)
            elif self.debug_enabled:
                filtered.append(int(value.get("_debug_index", -1)))
        outputs = []
        snapshots = []
        for class_name in sorted(self.trackers):
            if self.debug_enabled:
                class_outputs, self.next_id, snapshot = self.trackers[class_name].update(
                    grouped[class_name], self.next_id, debug=True
                )
                snapshot["class_name"] = class_name
                snapshot["score_threshold"] = float(self._class_threshold(class_name))
                snapshots.append(snapshot)
            else:
                class_outputs, self.next_id = self.trackers[class_name].update(grouped[class_name], self.next_id)
            outputs.extend(class_outputs)
        ordered = sorted(outputs, key=lambda value: value["track_id"])
        if self.debug_enabled:
            self.last_debug = {
                "input_detections": [
                    {
                        "input_index": int(index),
                        "class_name": value["class_name"],
                        "score": float(value.get("score", 1.0)),
                        "box": [float(item) for item in value["box"]],
                    }
                    for index, value in enumerate(objects)
                ],
                "filtered_detection_indices": filtered,
                "groups": snapshots,
                "output_track_ids": [int(value["track_id"]) for value in ordered],
            }
        else:
            self.last_debug = None
        return ordered

    def enable_debug(self, enabled=True):
        self.debug_enabled = bool(enabled)
