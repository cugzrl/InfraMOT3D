from inframot3d.tracking.common import MotionTrack, MultiClassRunner, associate


class ImmortalClassTracker:

    def __init__(self, settings):
        self.settings = settings
        self.score_threshold = float(settings["score_threshold"])
        if "max_age" not in settings or settings["max_age"] is not None:
            raise ValueError("ImmortalTracker 的 max_age 必须为 null")
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
        for detection_index, track_index in matches:
            self.tracks[track_index].update(
                self.frame_count, 1, confident[detection_index]
            )
        for track_index in unmatched_tracks:
            self.tracks[track_index].update(self.frame_count, 0)
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


class MultiClassImmortalTracker:
    def __init__(self, tracker_config):
        self.runner = MultiClassRunner(tracker_config, ImmortalClassTracker)

    def update(self, objects, timestamp=None):
        return self.runner.update(objects, timestamp)
