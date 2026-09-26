from inframot3d.tracking.common import MotionTrack, MultiClassRunner, associate, overlap, pair_cost


class SurvivalPolicy:
    def __init__(self, settings):
        policy = settings.get("scene_memory") or {}
        self.enabled = bool(policy.get("enabled", False))
        self.mode = str(policy.get("mode", "none"))
        self.delta_age = int(policy.get("delta_age", 0))
        self.cell = float(policy.get("grid_m", 5.0))
        self.bev = policy.get("bev") or {}
        self.hard_cells = {tuple(int(value) for value in item) for item in policy.get("hard_cells", [])}

    def active(self, box):
        if not self.enabled or self.delta_age <= 0:
            return False
        if self.mode == "global":
            return True
        if self.mode not in {"scene", "shuffled"} or not self.hard_cells:
            return False
        x, y = float(box[0]), float(box[1])
        x_min = float(self.bev["x_min"])
        x_max = float(self.bev["x_max"])
        y_min = float(self.bev["y_min"])
        y_max = float(self.bev["y_max"])
        if x < x_min or x > x_max or y < y_min or y > y_max:
            return False
        ix = int((min(x, x_max - 1.0e-6) - x_min) // self.cell)
        iy = int((min(y, y_max - 1.0e-6) - y_min) // self.cell)
        return (iy, ix) in self.hard_cells


class RedundancyModule:

    def __init__(self, settings):
        redundancy = settings["redundancy"]
        self.mode = redundancy["mode"]
        self.metric = settings["metric"]
        self.det_score = float(redundancy["det_score_threshold"])
        self.det_threshold = float(redundancy["det_dist_threshold"])
        if self.mode not in {"default", "mm"}:
            raise ValueError(f"不支持的 redundancy 模式{self.mode}")

    def resolve(self, track, detections):
        if self.mode == "default":
            return 0
        # 低分检测只参与二阶段冗余匹配，阈值含0.01
        candidates = [
            detection
            for detection in detections
            if float(detection.get("score", 1.0)) >= self.det_score
        ]
        if not candidates:
            return 0
        if self.metric in {"giou", "iou"}:
            best = max(overlap(detection["box"], track.box, self.metric) for detection in candidates)
            return 0 if best < self.det_threshold else 3
        best = min(pair_cost(detection["box"], track.box, self.metric) for detection in candidates)
        return 0 if best > self.det_threshold else 3


class SimpleClassTracker:
    def __init__(self, settings):
        self.settings = settings
        if settings.get("max_age") is None:
            raise ValueError("SimpleTrack 需要 max_age")
        self.score_threshold = float(settings["score_threshold"])
        self.redundancy = RedundancyModule(settings)
        self.survival = SurvivalPolicy(settings)
        self.tracks = []
        self.frame_count = 0
        self.triggered_tracks = set()
        self.extended_tracks = set()
        self.pending_recovery = set()
        self.stats = {
            "unmatched_track_frames": 0,
            "memory_trigger_frames": 0,
            "extra_retained_frames": 0,
            "recovered_tracks": 0,
            "expired_after_extension": 0,
        }

    def update(self, detections, id_start, timestamp_seconds, debug=False):
        self.frame_count += 1
        for track in self.tracks:
            track.predict(timestamp_seconds)
        pre_tracks = [
            {
                "track_id": int(track.track_id),
                "box": [float(value) for value in track.box],
                "state": track.life.state,
                "hits": int(track.life.hits),
                "time_since_update": int(track.life.time_since_update),
            }
            for track in self.tracks
        ]
        confident = [
            detection
            for detection in detections
            if float(detection.get("score", 1.0)) >= self.score_threshold
        ]
        if debug:
            matches, unmatched_detections, unmatched_tracks, association_debug = associate(
                confident, self.tracks, self.settings, return_debug=True
            )
        else:
            matches, unmatched_detections, unmatched_tracks = associate(
                confident, self.tracks, self.settings
            )
        matched = {track_index for _, track_index in matches}
        for detection_index, track_index in matches:
            track = self.tracks[track_index]
            if track.track_id in self.pending_recovery:
                self.stats["recovered_tracks"] += 1
                self.pending_recovery.remove(track.track_id)
            track.update(
                self.frame_count, 1, confident[detection_index]
            )
        redundancy = []
        for track_index in unmatched_tracks:
            if track_index in matched:
                continue
            track = self.tracks[track_index]
            mode = self.redundancy.resolve(track, detections)
            if debug:
                redundancy.append({"track_id": int(track.track_id), "mode": int(mode)})
            if mode != 0:
                track.update(self.frame_count, mode)
                continue
            self.stats["unmatched_track_frames"] += 1
            active = self.survival.active(track.box)
            max_age = int(self.settings["max_age"])
            if active:
                max_age += self.survival.delta_age
                self.stats["memory_trigger_frames"] += 1
                self.triggered_tracks.add(track.track_id)
                if track.life.time_since_update >= int(self.settings["max_age"]):
                    self.stats["extra_retained_frames"] += 1
                    self.extended_tracks.add(track.track_id)
                    self.pending_recovery.add(track.track_id)
            track.update(self.frame_count, mode, max_age=max_age)
        next_id = id_start
        created = []
        for detection_index in unmatched_detections:
            current_id = next_id
            self.tracks.append(
                MotionTrack(
                    confident[detection_index],
                    next_id,
                    self.frame_count,
                    timestamp_seconds,
                    self.settings,
                )
            )
            if debug:
                created.append(
                    {
                        "input_index": int(confident[detection_index].get("_debug_index", detection_index)),
                        "track_id": int(current_id),
                    }
                )
            next_id += 1
        dead_ids = {track.track_id for track in self.tracks if track.life.state == "dead"}
        expired = dead_ids & self.pending_recovery
        self.stats["expired_after_extension"] += len(expired)
        self.pending_recovery.difference_update(dead_ids)
        track_states = [
            {
                "track_id": int(track.track_id),
                "state": track.life.state,
                "recent_state": int(track.life.recent_state),
                "time_since_update": int(track.life.time_since_update),
                "published": bool(track.publish()),
            }
            for track in self.tracks
        ]
        self.tracks = [track for track in self.tracks if track.life.state != "dead"]
        outputs = [track.export() for track in self.tracks if track.publish()]
        if not debug:
            return outputs, next_id
        snapshot = {
            "score_threshold": float(self.score_threshold),
            "filtered_detection_indices": [
                int(item.get("_debug_index", index))
                for index, item in enumerate(detections)
                if float(item.get("score", 1.0)) < self.score_threshold
            ],
            "candidate_detections": [
                {
                    "input_index": int(item.get("_debug_index", index)),
                    "score": float(item.get("score", 1.0)),
                    "box": [float(value) for value in item["box"]],
                }
                for index, item in enumerate(confident)
            ],
            "pre_tracks": pre_tracks,
            "association": association_debug,
            "assignments": [
                {
                    "input_index": int(confident[detection_index].get("_debug_index", detection_index)),
                    "track_id": int(pre_tracks[track_index]["track_id"]),
                }
                for detection_index, track_index in matches
            ],
            "redundancy": redundancy,
            "created": created,
            "track_states": track_states,
            "output_track_ids": [int(item["track_id"]) for item in outputs],
        }
        return outputs, next_id, snapshot

    def memory_stats(self):
        output = dict(self.stats)
        output["triggered_tracks"] = len(self.triggered_tracks)
        output["extended_tracks"] = len(self.extended_tracks)
        return output


class MultiClassSimpleTrack:
    def __init__(self, tracker_config):
        self.runner = MultiClassRunner(tracker_config, SimpleClassTracker)
        self.debug_enabled = False

    def update(self, objects, timestamp=None):
        return self.runner.update(objects, timestamp, debug=self.debug_enabled)

    def enable_debug(self, enabled=True):
        self.debug_enabled = bool(enabled)

    @property
    def last_debug(self):
        return self.runner.last_debug

    def memory_stats(self):
        totals = {
            "unmatched_track_frames": 0,
            "memory_trigger_frames": 0,
            "extra_retained_frames": 0,
            "recovered_tracks": 0,
            "expired_after_extension": 0,
            "triggered_tracks": 0,
            "extended_tracks": 0,
        }
        for tracker in self.runner.trackers.values():
            current = tracker.memory_stats()
            for key in totals:
                totals[key] += int(current[key])
        return totals
