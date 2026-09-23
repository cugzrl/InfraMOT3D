from inframot3d.tracking.common import MotionTrack, MultiClassRunner, associate, overlap, pair_cost


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
        candidates = [
            detection
            for detection in detections
            if float(detection.get("score", 1.0)) > self.det_score
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
        self.tracks = []
        self.frame_count = 0

    def update(self, detections, id_start, timestamp_seconds):
        self.frame_count += 1
        for track in self.tracks:
            track.predict(timestamp_seconds)
        confident = [
            detection
            for detection in detections
            if float(detection.get("score", 1.0)) >= self.score_threshold
        ]
        matches, unmatched_detections, unmatched_tracks = associate(
            confident, self.tracks, self.settings
        )
        matched = {track_index for _, track_index in matches}
        for detection_index, track_index in matches:
            self.tracks[track_index].update(
                self.frame_count, 1, confident[detection_index]
            )
        for track_index in unmatched_tracks:
            if track_index in matched:
                continue
            mode = self.redundancy.resolve(self.tracks[track_index], detections)
            self.tracks[track_index].update(self.frame_count, mode)
        next_id = id_start
        for detection_index in unmatched_detections:
            self.tracks.append(
                MotionTrack(
                    confident[detection_index],
                    next_id,
                    self.frame_count,
                    timestamp_seconds,
                    self.settings,
                )
            )
            next_id += 1
        self.tracks = [track for track in self.tracks if track.life.state != "dead"]
        outputs = [track.export() for track in self.tracks if track.publish()]
        return outputs, next_id


class MultiClassSimpleTrack:
    def __init__(self, tracker_config):
        self.runner = MultiClassRunner(tracker_config, SimpleClassTracker)

    def update(self, objects, timestamp=None):
        return self.runner.update(objects, timestamp)
