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


def _slice_frame(frame, mask):
    item = {
        key: (value[mask] if isinstance(value, np.ndarray) and len(value) == len(mask) else value)
        for key, value in frame.items()
    }
    count = int(np.asarray(mask).sum()) if len(mask) else 0
    item["velocity"] = np.zeros((count, 2), dtype=np.float32)
    return item


def build_clips(sequence_frames, clip_len, clip_stride, score_threshold):
    # 空检测帧保留原始时间戳，不把frame10和frame12粘成相邻帧
    clips = []
    for frames in sequence_frames:
        kept = []
        for frame in frames:
            mask = frame["score"] >= float(score_threshold)
            kept.append(_slice_frame(frame, mask))
        for start in range(0, max(len(kept) - clip_len + 1, 0), int(clip_stride)):
            clip = copy.deepcopy(kept[start:start + clip_len])
            if len(clip) < clip_len:
                continue
            clips.append(clip)
            reversed_clip = copy.deepcopy(clip[::-1])
            times = [frame["timestamp"] for frame in clip]
            for frame, timestamp in zip(reversed_clip, times):
                frame["timestamp"] = timestamp
                frame["velocity"] = np.zeros((len(frame["score"]), 2), dtype=np.float32)
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


def _advance_empty(tracked):
    # 空帧只推进寿命，保留最后一次真实观测时间
    if tracked is None or len(tracked) == 0:
        return tracked
    tracked.velocity = torch.zeros_like(tracked.velocity)
    tracked.age = tracked.age + 1
    return tracked


def _association_target(current, tracked):
    same = current.tracking_id[:, 0].view(-1, 1) == tracked.tracking_id[:, 0].view(1, -1)
    valid = (current.tracking_id[:, 0].view(-1, 1) >= 0) & (tracked.tracking_id[:, 0].view(1, -1) >= 0)
    target = (same & valid).to(dtype=torch.float32)
    return target.T


def train_clip(model, frames, device):
    from models.structures import Instances
    from torchvision.ops import sigmoid_focal_loss

    tracked = None
    losses = []
    for frame in frames:
        if len(frame["score"]) == 0:
            tracked = _advance_empty(tracked)
            continue
        current = _to_instance(frame, device)
        current.velocity = torch.zeros_like(current.velocity)
        current.set("age", torch.zeros_like(current.score))
        if tracked is None or len(tracked) == 0:
            model.eval()
            tracked = _init_features(model, current)
            tracked.velocity = torch.zeros_like(tracked.velocity)
            tracked.age = tracked.age + 1
            model.train()
            continue
        model.train()
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
        target = _association_target(current, tracked)
        affinity_losses = []
        for level in range(affinity_scores.shape[0]):
            pred = affinity_scores[level][..., 0]
            affinity_losses.append(sigmoid_focal_loss(pred, target, alpha=-1, gamma=1.0, reduction="mean"))
        losses.append(sum(affinity_losses))
        with torch.no_grad():
            tracked.velocity = torch.zeros_like(tracked.velocity)
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
            tracked.velocity = torch.zeros_like(tracked.velocity)
            tracked.age = tracked.age + 1
    if not losses:
        return None
    return torch.stack(losses).mean()


def _binary_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positive = int(labels.sum())
    negative = int(len(labels) - positive)
    if positive == 0 or negative == 0:
        return None
    from scipy.stats import rankdata

    ranks = rankdata(scores, method="average")
    sum_positive = float(ranks[labels == 1].sum())
    return (sum_positive - positive * (positive + 1) / 2.0) / (positive * negative)


def association_metrics(model, clips, device):
    from models.structures import Instances
    import lap

    model.eval()
    tp = fp = tn = fn = 0
    labels = []
    scores = []
    with torch.no_grad():
        for frames in clips:
            tracked = None
            for frame in frames:
                if len(frame["score"]) == 0:
                    tracked = _advance_empty(tracked)
                    continue
                current = _to_instance(frame, device)
                current.velocity = torch.zeros_like(current.velocity)
                current.set("age", torch.zeros_like(current.score))
                if tracked is None or len(tracked) == 0:
                    tracked = _init_features(model, current)
                    tracked.velocity = torch.zeros_like(tracked.velocity)
                    tracked.age = tracked.age + 1
                    continue
                coordinate, spatial, spatial_dist = spatial_inputs(current, model.num_classes)
                current_time = current.time[0, 0]
                temporal_info = temporal_vector(current, current_time)[:, None, :] - temporal_vector(tracked, current_time)[None, ...]
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
                target = _association_target(current, tracked)
                probability = torch.sigmoid(affinity_scores[-1][..., 0])
                pred_label = probability >= 0.5
                truth = target >= 0.5
                tp += int((pred_label & truth).sum().item())
                fp += int((pred_label & ~truth).sum().item())
                tn += int((~pred_label & ~truth).sum().item())
                fn += int((~pred_label & truth).sum().item())
                labels.append(truth.detach().reshape(-1).cpu().numpy())
                scores.append(probability.detach().reshape(-1).cpu().numpy())
                affinity = probability.T
                invalid = current.classes.view(-1, 1) != tracked.classes.view(1, -1)
                cost = affinity + -1e6 * invalid
                _, _, columns = lap.lapjv(1 - cost.detach().cpu().numpy(), extend_cost=True, cost_limit=0.95)
                missed = tracked[torch.as_tensor(columns, device=device) < 0] if len(tracked) else tracked
                if len(current) and len(missed):
                    tracked = Instances.cat([current, missed], current._img_meta)
                elif len(current):
                    tracked = current
                tracked.velocity = torch.zeros_like(tracked.velocity)
                tracked.age = tracked.age + 1
    positive = tp + fn
    negative = tn + fp
    predicted = tp + fp
    precision = tp / predicted if predicted else 0.0
    recall = tp / positive if positive else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    auc = _binary_auc(np.concatenate(labels) if labels else np.zeros(0), np.concatenate(scores) if scores else np.zeros(0))
    return {
        "positive_accuracy": tp / positive if positive else 0.0,
        "negative_accuracy": tn / negative if negative else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


class GraeTracker:
    def __init__(self, model, class_names, score_thresholds, alpha=0.24, age=12, score_floor=0.01):
        self.model = model
        self.class_names = list(class_names)
        self.score_thresholds = {str(name): float(value) for name, value in score_thresholds.items()}
        # alpha保留官方二阶段代价分界的配置位，新建轨迹由类别high_threshold控制
        self.alpha = float(alpha)
        self.age = int(age)
        self.score_floor = float(score_floor)
        self.device = next(model.parameters()).device
        self.reset()

    def _high_mask(self, instance):
        if len(instance) == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        limits = []
        for class_index in instance.classes.view(-1).tolist():
            name = self.class_names[int(class_index)]
            if name not in self.score_thresholds:
                raise KeyError("缺少类别阈值%s" % name)
            limits.append(self.score_thresholds[name])
        limit = torch.tensor(limits, device=self.device, dtype=instance.score.dtype)
        return instance.score[:, 0] >= limit

    def reset(self):
        self.track = None
        self.next_id = 0
        self.last_time = None

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

    def update(self, objects, timestamp_seconds, sample_token=""):
        from models.structures import Instances
        import lap

        # 推理只读取检测框，不使用source_track_id
        with torch.no_grad():
            return self._update(objects, timestamp_seconds, sample_token, Instances, lap)

    def _update(self, objects, timestamp_seconds, sample_token, Instances, lap):
        # 只接收类别分数和框，低分二阶段仍保留
        cleaned = []
        for item in objects:
            score = float(item.get("score", 1.0))
            if score < self.score_floor:
                continue
            cleaned.append({"class_name": item["class_name"], "score": score, "box": item["box"]})
        objects = cleaned
        frame = detection_records(objects, {name: index for index, name in enumerate(self.class_names)}, timestamp_seconds)
        frame["sample_token"] = sample_token
        outputs = []
        if self.last_time is None:
            self.last_time = float(timestamp_seconds)
        dt = float(timestamp_seconds) - self.last_time
        self.last_time = float(timestamp_seconds)
        if len(frame["score"]) == 0:
            if self.track is not None and len(self.track):
                self.track.velocity = torch.zeros_like(self.track.velocity)
                self.track = self.track[self.track.age[:, 0] < self.age]
                self.track.age = self.track.age + 1
            return outputs
        dets = _to_instance(frame, self.device)
        dets.velocity = torch.zeros_like(dets.velocity)
        if self.track is None or len(self.track) == 0:
            dets = dets[self._high_mask(dets)]
            if len(dets) == 0:
                return outputs
            dets.velocity = torch.zeros_like(dets.velocity)
            dets.instance_inds = torch.arange(self.next_id, self.next_id + len(dets), device=self.device).view(-1, 1)
            self.next_id += len(dets)
            dets = _init_features(self.model, dets)
            self.track = copy.deepcopy(dets)
            self.track.age = self.track.age + 1
            return self._objects(dets)
        self.track.velocity = torch.zeros_like(self.track.velocity)
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
        high_mask = self._high_mask(dets)
        high = dets[high_mask]
        low = dets[~high_mask]
        high_cost = cost[high_mask]
        low_cost = cost[~high_mask]
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
        fresh = fresh[self._high_mask(fresh)]
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
            dets.velocity = torch.zeros_like(dets.velocity)
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
