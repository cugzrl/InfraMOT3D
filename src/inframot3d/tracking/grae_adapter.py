import copy
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

from inframot3d.geometry import bev_corners


def ensure_grae(root):
    root = str(Path(root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def yaw_to_quaternion(yaw):
    half = float(yaw) * 0.5
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def _bev_iou(box_a, box_b):
    poly_a = Polygon(bev_corners(box_a))
    poly_b = Polygon(bev_corners(box_b))
    if not poly_a.is_valid:
        poly_a = poly_a.buffer(0)
    if not poly_b.is_valid:
        poly_b = poly_b.buffer(0)
    if poly_a.is_empty or poly_b.is_empty or not poly_a.intersects(poly_b):
        return 0.0
    intersection = poly_a.intersection(poly_b).area
    union = poly_a.area + poly_b.area - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def match_detection_frame(detections, ground_truth, class_to_index):
    # 同类别且BEV IoU大于0才允许匹配，监督id只在这里生成
    count = len(detections)
    tracking_ids = -np.ones(count, dtype=np.int64)
    if count == 0 or not ground_truth:
        return tracking_ids, 0
    gt_boxes = [item["box"] for item in ground_truth]
    gt_names = [item["class_name"] for item in ground_truth]
    cost = np.full((count, len(ground_truth)), 1e6, dtype=np.float64)
    for row, det in enumerate(detections):
        for column, gt_box in enumerate(gt_boxes):
            if det["class_name"] != gt_names[column]:
                continue
            if det["class_name"] not in class_to_index:
                continue
            iou = _bev_iou(det["box"], gt_box)
            if iou > 0.0:
                cost[row, column] = -iou
    rows, columns = linear_sum_assignment(cost)
    matched = 0
    for row, column in zip(rows, columns):
        if cost[row, column] < 1e5:
            tracking_ids[row] = int(ground_truth[column]["track_index"])
            matched += 1
    return tracking_ids, matched


def assign_causal_velocity(frames):
    # 只用已匹配检测的历史中心和dt，不读取GT位置
    history = {}
    for frame in frames:
        translation = frame["translation"]
        timestamp = float(frame["timestamp"])
        velocity = np.zeros((len(translation), 2), dtype=np.float32)
        for index, track_id in enumerate(frame["tracking_id"]):
            track_id = int(track_id)
            if track_id < 0:
                continue
            center = (float(translation[index, 0]), float(translation[index, 1]))
            past = history.setdefault(track_id, [])
            if past:
                previous_time, previous_x, previous_y = past[-1]
                dt = timestamp - previous_time
                if dt > 1e-6:
                    velocity[index, 0] = (center[0] - previous_x) / dt
                    velocity[index, 1] = (center[1] - previous_y) / dt
            past.append((timestamp, center[0], center[1]))
        frame["velocity"] = velocity
    return frames


def detection_records(objects, class_to_index, timestamp_seconds):
    kept = []
    for item in objects:
        box = np.asarray(item["box"], dtype=np.float64)
        score = float(item["score"])
        if item["class_name"] not in class_to_index or box.shape != (7,) or not np.isfinite(box).all() or not np.isfinite(score):
            continue
        kept.append(item)
    translation = np.zeros((len(kept), 3), dtype=np.float32)
    size = np.zeros((len(kept), 3), dtype=np.float32)
    yaw = np.zeros((len(kept),), dtype=np.float32)
    rotation = np.zeros((len(kept), 4), dtype=np.float32)
    classes = np.zeros((len(kept),), dtype=np.int64)
    score = np.zeros((len(kept),), dtype=np.float32)
    for index, item in enumerate(kept):
        box = item["box"]
        translation[index] = [box[0], box[1], box[2]]
        size[index] = [box[4], box[5], box[6]]
        yaw[index] = box[3]
        rotation[index] = yaw_to_quaternion(box[3])
        classes[index] = class_to_index[item["class_name"]]
        score[index] = item["score"]
    return {
        "translation": translation,
        "size": size,
        "yaw": yaw,
        "rotation": rotation,
        "classes": classes,
        "score": score,
        "tracking_id": -np.ones(len(kept), dtype=np.int64),
        "velocity": np.zeros((len(kept), 2), dtype=np.float32),
        "timestamp": float(timestamp_seconds),
    }


def build_clips(sequence_frames, clip_len, clip_stride, score_threshold):
    clips = []
    for frames in sequence_frames:
        kept = []
        for frame in frames:
            mask = frame["score"] >= float(score_threshold)
            if not np.any(mask):
                continue
            item = {key: (value[mask] if isinstance(value, np.ndarray) and len(value) == len(mask) else value) for key, value in frame.items()}
            kept.append(item)
        for start in range(0, max(len(kept) - clip_len + 1, 0), int(clip_stride)):
            clip = copy.deepcopy(kept[start:start + clip_len])
            if len(clip) < clip_len:
                continue
            assign_causal_velocity(clip)
            clips.append(clip)
            reversed_clip = copy.deepcopy(clip[::-1])
            times = [frame["timestamp"] for frame in clip]
            for frame, timestamp in zip(reversed_clip, times):
                frame["timestamp"] = timestamp
            assign_causal_velocity(reversed_clip)
            clips.append(reversed_clip)
    return clips


def _one_hot(classes, num_classes):
    return torch.nn.functional.one_hot(classes.view(-1).long(), int(num_classes))


def spatial_inputs(instance, num_classes):
    score = instance.score
    center = instance.translation
    yaw = instance.rotation
    size = instance.size
    label = _one_hot(instance.classes, num_classes).to(dtype=torch.float32)
    coordinate = torch.cat([center, yaw, size, score, label], dim=-1).to(torch.float32)
    asso_coord = center[:, None, :] - center[None, ...]
    asso_size = size[:, None, :] - size[None, ...]
    asso_yaw = yaw[:, None, :] - yaw[None, ...]
    asso_label = label[:, None, :] - label[None, ...]
    asso_score = score.expand(len(instance), -1, -1)
    spatial_dist = torch.sqrt(torch.norm(center[:, None, :] - center[None, ...], dim=-1))
    spatial = torch.cat([asso_coord, asso_yaw, asso_size, asso_score, asso_label, spatial_dist.unsqueeze(-1)], dim=-1)
    return coordinate, spatial.to(torch.float32), spatial_dist


def temporal_vector(instance, current_time):
    delta = current_time - instance.time
    return torch.cat([instance.translation, instance.rotation, instance.size, instance.score, delta], dim=-1).to(torch.float32)


def _to_instance(frame, device):
    from models.structures import Instances

    meta = {"timestamp": float(frame["timestamp"]), "sample_token": frame.get("sample_token", "")}
    instance = Instances((1, 1), meta)
    count = len(frame["score"])
    instance.set("translation", torch.as_tensor(frame["translation"], dtype=torch.float32, device=device))
    instance.set("size", torch.as_tensor(frame["size"], dtype=torch.float32, device=device))
    instance.set("yaw", torch.as_tensor(frame["yaw"], dtype=torch.float32, device=device).view(count, 1))
    instance.set("rotation", torch.as_tensor(frame["rotation"], dtype=torch.float32, device=device))
    instance.set("velocity", torch.as_tensor(frame["velocity"], dtype=torch.float32, device=device))
    instance.set("score", torch.as_tensor(frame["score"], dtype=torch.float32, device=device).view(count, 1))
    instance.set("classes", torch.as_tensor(frame["classes"], dtype=torch.int64, device=device).view(count, 1))
    instance.set("tracking_id", torch.as_tensor(frame["tracking_id"], dtype=torch.int64, device=device).view(count, 1))
    instance.set("time", torch.full((count, 1), float(frame["timestamp"]), dtype=torch.float64, device=device))
    instance.set("ct", instance.translation[:, :2].clone())
    return instance


def _init_features(model, instance):
    coordinate, spatial, spatial_dist = spatial_inputs(instance, model.num_classes)
    temporal = torch.zeros_like(temporal_vector(instance, instance.time[0, 0]))
    with torch.no_grad():
        coordinate_feature, instance_feature, motion_feature = model(
            coordinate, spatial, spatial_dist, temporal, first_frame=True
        )
    instance.set("coord_features", coordinate_feature)
    instance.set("instance_feature", instance_feature)
    instance.set("motion_feature", motion_feature)
    age = torch.zeros_like(instance.score)
    instance.set("age", age)
    return instance


def train_clip(model, frames, device):
    from models.structures import Instances
    from torchvision.ops import sigmoid_focal_loss

    instances = [_to_instance(frame, device) for frame in frames if len(frame["score"])]
    if len(instances) < 2:
        return None
    model.eval()
    first = _init_features(model, instances[0])
    tracked = copy.deepcopy(first)
    tracked.age = tracked.age + 1
    model.train()
    losses = []
    for frame_id in range(1, len(instances)):
        current = instances[frame_id]
        current.set("age", torch.zeros_like(current.score))
        if len(tracked) == 0:
            tracked = _init_features(model, current)
            tracked.age = tracked.age + 1
            continue
        coordinate, spatial, spatial_dist = spatial_inputs(current, model.num_classes)
        current_time = current.time[0, 0]
        det_info = temporal_vector(current, current_time)
        track_info = temporal_vector(tracked, current_time)
        temporal_info = det_info[:, None, :] - track_info[None, ...]
        temporal_dist = torch.sqrt(torch.norm(current.ct.reshape(1, -1, 2) - tracked.ct.reshape(-1, 1, 2), dim=-1))
        coordinate_feature, instance_feature, motion_feature, _, affinity_scores, _ = model(
            coordinate,
            spatial,
            spatial_dist,
            temporal_info,
            temporal_dist,
            tracked.motion_feature.clone(),
            first_frame=False,
        )
        current.set("coord_features", coordinate_feature)
        current.set("instance_feature", instance_feature)
        current.set("motion_feature", motion_feature)
        same = current.tracking_id[:, 0].view(-1, 1) == tracked.tracking_id[:, 0].view(1, -1)
        target = same.to(dtype=torch.float32)
        invalid_track = tracked.tracking_id[:, 0] < 0
        target[:, invalid_track] = 0.0
        target = target.T
        affinity_losses = []
        for level in range(affinity_scores.shape[0]):
            pred = affinity_scores[level][..., 0]
            affinity_losses.append(sigmoid_focal_loss(pred, target, alpha=0.25, gamma=1.0, reduction="mean"))
        losses.append(sum(affinity_losses))
        with torch.no_grad():
            dt = (current_time - tracked.time[:, 0]).clamp(min=0).unsqueeze(-1).to(tracked.ct.dtype)
            tracked.ct = tracked.ct + tracked.velocity[:, :2] * dt
            affinity = torch.sigmoid(affinity_scores[-1][..., 0]).T
            invalid = current.classes.view(-1, 1) != tracked.classes.view(1, -1)
            cost = affinity + -1e6 * invalid
            import lap

            _, _, columns = lap.lapjv(1 - cost.detach().cpu().numpy(), extend_cost=True, cost_limit=0.95)
            missed = tracked[torch.as_tensor(columns, device=device) < 0] if len(tracked) else tracked
            if len(current) and len(missed):
                tracked = Instances.cat([current, missed], current._img_meta)
            elif len(current):
                tracked = current
            tracked.age = tracked.age + 1
    if not losses:
        return None
    return torch.stack(losses).mean()


class GraeTracker:
    def __init__(self, model, class_names, alpha=0.24, conf_threshold=0.12, age=12):
        self.model = model
        self.class_names = list(class_names)
        self.alpha = float(alpha)
        self.conf_threshold = float(conf_threshold)
        self.age = int(age)
        self.device = next(model.parameters()).device
        self.reset()

    def reset(self):
        self.track = None
        self.next_id = 0
        self.last_time = None
        self.history = {}

    def _objects(self, instance, score_scale=1.0):
        outputs = []
        for index in range(len(instance)):
            translation = instance.translation[index].detach().cpu().numpy()
            size = instance.size[index].detach().cpu().numpy()
            yaw = float(instance.yaw[index].detach().cpu())
            outputs.append(
                {
                    "class_name": self.class_names[int(instance.classes[index])],
                    "track_id": int(instance.instance_inds[index].detach().cpu()),
                    "score": float(instance.score[index].detach().cpu()) * score_scale,
                    "box": [
                        float(translation[0]),
                        float(translation[1]),
                        float(translation[2]),
                        yaw,
                        float(size[0]),
                        float(size[1]),
                        float(size[2]),
                    ],
                }
            )
        return outputs

    def _update_velocity(self, instance, timestamp):
        velocity = instance.velocity.clone()
        for index in range(len(instance)):
            track_id = int(instance.instance_inds[index].detach().cpu())
            if track_id < 0:
                continue
            center = instance.translation[index, :2].detach().cpu().numpy()
            past = self.history.setdefault(track_id, [])
            if past:
                previous_time, previous_x, previous_y = past[-1]
                dt = float(timestamp) - previous_time
                if dt > 1e-6:
                    velocity[index, 0] = (float(center[0]) - previous_x) / dt
                    velocity[index, 1] = (float(center[1]) - previous_y) / dt
            past.append((float(timestamp), float(center[0]), float(center[1])))
        instance.velocity = velocity

    def update(self, objects, timestamp_seconds, sample_token=""):
        from models.structures import Instances
        import lap

        # 推理只读取检测框，不使用source_track_id
        with torch.no_grad():
            return self._update(objects, timestamp_seconds, sample_token, Instances, lap)

    def _update(self, objects, timestamp_seconds, sample_token, Instances, lap):
        frame = detection_records(objects, {name: index for index, name in enumerate(self.class_names)}, timestamp_seconds)
        frame["sample_token"] = sample_token
        outputs = []
        if self.last_time is None:
            self.last_time = float(timestamp_seconds)
        dt = float(timestamp_seconds) - self.last_time
        self.last_time = float(timestamp_seconds)
        if len(frame["score"]) == 0:
            if self.track is not None and len(self.track):
                self.track.ct = self.track.ct + self.track.velocity[:, :2] * dt
                self.track.translation[:, :2] = self.track.ct
                self.track = self.track[self.track.age[:, 0] < self.age]
                self.track.age = self.track.age + 1
            return outputs
        dets = _to_instance(frame, self.device)
        if self.track is None or len(self.track) == 0:
            dets = dets[dets.score[:, 0] > self.conf_threshold]
            if len(dets) == 0:
                return outputs
            dets.velocity = torch.zeros_like(dets.velocity)
            dets.instance_inds = torch.arange(self.next_id, self.next_id + len(dets), device=self.device).view(-1, 1)
            self.next_id += len(dets)
            self._update_velocity(dets, timestamp_seconds)
            dets = _init_features(self.model, dets)
            self.track = copy.deepcopy(dets)
            self.track.age = self.track.age + 1
            return self._objects(dets)
        self.track.ct = self.track.ct + self.track.velocity[:, :2] * dt
        self.track.translation[:, :2] = self.track.ct
        coordinate, spatial, spatial_dist = spatial_inputs(dets, self.model.num_classes)
        current_time = dets.time[0, 0]
        temporal_info = temporal_vector(dets, current_time)[:, None, :] - temporal_vector(self.track, current_time)[None, ...]
        temporal_dist = torch.sqrt(torch.norm(dets.ct.reshape(1, -1, 2) - self.track.ct.reshape(-1, 1, 2), dim=-1))
        coordinate_feature, instance_feature, motion_feature, _, affinity_scores, _ = self.model(
            coordinate,
            spatial,
            spatial_dist,
            temporal_info,
            temporal_dist,
            self.track.motion_feature.clone(),
            first_frame=False,
        )
        dets.set("coord_features", coordinate_feature)
        dets.set("instance_feature", instance_feature)
        dets.set("motion_feature", motion_feature)
        affinity = torch.sigmoid(affinity_scores[-1][..., 0]).T
        distance = torch.sqrt(torch.sum((self.track.ct.reshape(1, -1, 2) - dets.ct.reshape(-1, 1, 2)) ** 2, dim=2))
        invalid = dets.classes.view(-1, 1) != self.track.classes.view(1, -1)
        cost = affinity * 0.5 + torch.exp(-distance) * 0.5
        cost = cost + -1e6 * invalid
        high = dets[dets.score[:, 0] > self.alpha]
        low = dets[dets.score[:, 0] <= self.alpha]
        high_cost = cost[dets.score[:, 0] > self.alpha]
        low_cost = cost[dets.score[:, 0] <= self.alpha]
        if len(high):
            high.instance_inds = torch.full((len(high), 1), -2, dtype=torch.int64, device=self.device)
        if len(low):
            low.instance_inds = torch.full((len(low), 1), -2, dtype=torch.int64, device=self.device)
        if len(high):
            _, _, columns = lap.lapjv(1 - high_cost.detach().cpu().numpy(), extend_cost=True, cost_limit=0.9)
            track_ids = self.track.instance_inds.clone()
            det_ids = high.instance_inds.clone()
            for track_index, det_index in enumerate(columns):
                if det_index >= 0:
                    det_ids[det_index] = track_ids[track_index]
                    confidence = high_cost[det_index, track_index]
                    high.motion_feature[det_index] = self.track.motion_feature[track_index] * (1 - confidence) + high.motion_feature[det_index] * confidence
                    track_ids[track_index] = -2
            high.instance_inds = det_ids
            remain = torch.where(track_ids[:, 0] != -2)[0]
            low_cost = low_cost[:, remain]
            self.track = self.track[track_ids[:, 0] != -2]
        if len(self.track) and len(low):
            _, _, columns = lap.lapjv(1 - low_cost.detach().cpu().numpy(), extend_cost=True, cost_limit=0.8)
            track_ids = self.track.instance_inds.clone()
            det_ids = low.instance_inds.clone()
            for track_index, det_index in enumerate(columns):
                if det_index >= 0:
                    det_ids[det_index] = track_ids[track_index]
                    confidence = low_cost[det_index, track_index]
                    low.motion_feature[det_index] = self.track.motion_feature[track_index] * (1 - confidence) + low.motion_feature[det_index] * confidence
                    track_ids[track_index] = -2
            low.instance_inds = det_ids
            self.track = self.track[track_ids[:, 0] != -2]
        if len(high) and len(low):
            dets = Instances.cat([high, low], high._img_meta)
        elif len(high):
            dets = high
        else:
            dets = low
        matched = dets[dets.instance_inds[:, 0] > -1]
        fresh = dets[dets.instance_inds[:, 0] < 0]
        fresh = fresh[fresh.score[:, 0] > self.conf_threshold]
        if len(fresh):
            fresh.instance_inds = torch.arange(self.next_id, self.next_id + len(fresh), device=self.device).view(-1, 1)
            self.next_id += len(fresh)
        if len(matched) and len(fresh):
            dets = Instances.cat([matched, fresh], matched._img_meta)
        elif len(matched):
            dets = matched
        else:
            dets = fresh
        if len(dets):
            dets.set("age", torch.zeros_like(dets.score))
            self._update_velocity(dets, timestamp_seconds)
        coasted = self.track[self.track.age[:, 0] < 2] if self.track is not None and len(self.track) else None
        if coasted is not None and len(coasted):
            outputs.extend(self._objects(coasted, score_scale=0.1))
        if self.track is not None and len(self.track) and len(dets):
            self.track = Instances.cat([dets, self.track], dets._img_meta)
        elif len(dets):
            self.track = dets
        if self.track is not None and len(self.track):
            self.track = self.track[self.track.age[:, 0] < self.age]
            self.track.age = self.track.age + 1
        if len(dets):
            outputs.extend(self._objects(dets))
        return outputs


def build_model(grae_root, in_channels, layers, num_classes, device):
    ensure_grae(grae_root)
    from models.main import GRAE

    model = GRAE(in_channels=int(in_channels), layers=int(layers), device=str(device), num_classes=int(num_classes))
    return model.to(device)


def load_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()
    return checkpoint
