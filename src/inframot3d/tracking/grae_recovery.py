import torch
from torch import nn


class TrackConditionedRecovery(nn.Module):
    # 用当前检测特征和历史运动特征预测残差关联 logit
    # 最后一层零初始化，起点与原始 GRAE 分数一致

    def __init__(self, channels):
        super().__init__()
        hidden = int(channels)
        self.det_proj = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.ReLU())
        self.trk_proj = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.pair = nn.Sequential(
            nn.Linear(hidden * 2 + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.pair[-1].weight)
        nn.init.zeros_(self.pair[-1].bias)

    def forward(self, det_instance, trk_motion, det_score, distance):
        if det_score.dim() == 1:
            det_score = det_score.view(-1, 1)
        det_feature = self.det_proj(torch.cat([det_instance, det_score], dim=-1))
        trk_feature = self.trk_proj(trk_motion)
        det_count = det_feature.shape[0]
        track_count = trk_feature.shape[0]
        pair = torch.cat(
            [
                det_feature[:, None, :].expand(det_count, track_count, -1),
                trk_feature[None, :, :].expand(det_count, track_count, -1),
                distance.unsqueeze(-1),
            ],
            dim=-1,
        )
        residual = self.pair(pair).squeeze(-1)
        # 高分检测保持原始 GRAE logit，残差只作用在低分检测上
        low = (det_score.view(-1) < 0.1).to(dtype=residual.dtype)
        return residual * low.view(-1, 1)


def _pair_mask(positive, negative, distance, neg_per_pos, min_neg):
    # 只监督已知身份，并限制近处负样本数量，避免远处易负样本主导
    pos_count = int(positive.sum().item())
    budget = max(int(min_neg), int(neg_per_pos) * max(pos_count, 1))
    neg_count = int(negative.sum().item())
    if neg_count > budget:
        flat = distance.masked_fill(~negative, 1.0e6).reshape(-1)
        chosen = torch.topk(flat, budget, largest=False).indices
        kept = torch.zeros_like(flat, dtype=torch.bool)
        kept[chosen] = True
        negative = kept.view_as(distance)
    return positive | negative


def _assign_pairs(cost, cost_limit):
    import lap

    if cost.numel() == 0 or cost.shape[0] == 0 or cost.shape[1] == 0:
        return []
    _, _, columns = lap.lapjv(1.0 - cost.detach().cpu().numpy(), extend_cost=True, cost_limit=float(cost_limit))
    pairs = []
    for track_index, det_index in enumerate(columns):
        if int(det_index) >= 0:
            pairs.append((int(det_index), int(track_index)))
    return pairs


def _birth_mask(instance, class_names, birth_thresholds):
    keep = []
    for class_index, score in zip(instance.classes.view(-1).tolist(), instance.score.view(-1).tolist()):
        name = class_names[int(class_index)]
        keep.append(float(score) >= float(birth_thresholds[name]))
    if not keep:
        return torch.zeros(0, dtype=torch.bool, device=instance.score.device)
    return torch.tensor(keep, dtype=torch.bool, device=instance.score.device)


def advance_track_state(
    current,
    tracked,
    affinity,
    distance,
    class_names,
    birth_thresholds,
    association_alpha,
    age,
    high_cost_limit,
    low_cost_limit,
    invalid=None,
    direct_mask=None,
):
    # 状态更新与推理的两阶段融合关联一致，低分检测可以续上轨迹但不能新建
    from models.structures import Instances

    if invalid is None:
        invalid = current.classes.view(-1, 1) != tracked.classes.view(1, -1)
    blended = affinity * 0.5 + torch.exp(-distance) * 0.5
    weak = current.score[:, 0] < 0.1
    if bool(weak.any()):
        blended = blended.clone()
        blended[weak] = affinity[weak]
    if direct_mask is not None and bool(direct_mask.any()):
        blended = blended.clone()
        blended[direct_mask] = affinity[direct_mask]
    cost = blended + (-1.0e6) * invalid
    high_mask = current.score[:, 0] >= float(association_alpha)
    high_index = torch.where(high_mask)[0]
    low_index = torch.where(~high_mask)[0]
    matched_det = torch.zeros(len(current), dtype=torch.bool, device=current.score.device)
    matched_track = torch.zeros(len(tracked), dtype=torch.bool, device=current.score.device)
    if len(high_index):
        for local_det, track_index in _assign_pairs(cost[high_index], high_cost_limit):
            det_index = int(high_index[local_det])
            confidence = cost[det_index, track_index]
            current.motion_feature[det_index] = (
                tracked.motion_feature[track_index] * (1.0 - confidence) + current.motion_feature[det_index] * confidence
            )
            matched_det[det_index] = True
            matched_track[track_index] = True
    remain = torch.where(~matched_track)[0]
    if len(low_index) and len(remain):
        for local_det, local_track in _assign_pairs(cost[low_index][:, remain], low_cost_limit):
            det_index = int(low_index[local_det])
            track_index = int(remain[local_track])
            confidence = cost[det_index, track_index]
            current.motion_feature[det_index] = (
                tracked.motion_feature[track_index] * (1.0 - confidence) + current.motion_feature[det_index] * confidence
            )
            matched_det[det_index] = True
            matched_track[track_index] = True
    birth = _birth_mask(current, class_names, birth_thresholds) & ~matched_det
    kept = current[matched_det | birth]
    missed = tracked[~matched_track]
    if len(kept) and len(missed):
        tracked = Instances.cat([kept, missed], kept._img_meta)
    elif len(kept):
        tracked = kept
    else:
        tracked = missed
    if tracked is not None and len(tracked):
        tracked = tracked[tracked.age[:, 0] < int(age)]
        tracked.age = tracked.age + 1
    return tracked


def recovery_clip_loss(
    model,
    recovery,
    frames,
    device,
    class_names,
    birth_thresholds,
    association_alpha=0.24,
    age=12,
    high_cost_limit=0.9,
    low_cost_limit=0.8,
    hard_distance=8.0,
    neg_per_pos=4,
    min_neg=8,
):
    from inframot3d.tracking.grae_adapter import _init_features, _to_instance, spatial_inputs, temporal_vector
    from torchvision.ops import sigmoid_focal_loss

    tracked = None
    losses = []
    stats = {"positive": 0, "negative": 0, "low_positive": 0, "frames": 0}
    for frame in frames:
        if len(frame["score"]) == 0:
            if tracked is not None and len(tracked):
                tracked = tracked[tracked.age[:, 0] < int(age)]
                tracked.age = tracked.age + 1
            continue
        current = _to_instance(frame, device)
        current.velocity = torch.zeros_like(current.velocity)
        current.set("age", torch.zeros_like(current.score))
        if tracked is None or len(tracked) == 0:
            model.eval()
            tracked = _init_features(model, current)
            birth = _birth_mask(tracked, class_names, birth_thresholds)
            tracked = tracked[birth]
            if len(tracked) == 0:
                tracked = None
                continue
            tracked.age = tracked.age + 1
            continue
        model.eval()
        with torch.no_grad():
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
            distance = torch.sqrt(torch.sum((tracked.ct.reshape(1, -1, 2) - current.ct.reshape(-1, 1, 2)) ** 2, dim=2))
            grae_logit = affinity_scores[-1][..., 0].transpose(0, 1)
        recovery.train()
        residual = recovery(
            current.instance_feature.detach(),
            tracked.motion_feature.detach(),
            current.score.detach(),
            distance.detach(),
        )
        logit = grae_logit.detach() + residual
        det_id = current.tracking_id[:, 0]
        trk_id = tracked.tracking_id[:, 0]
        # -1 是未知，不参与损失；-2 是远离全部 GT 的低分背景，只作负样本
        known = (det_id.view(-1, 1) >= 0) & (trk_id.view(1, -1) >= 0)
        positive = known & (det_id.view(-1, 1) == trk_id.view(1, -1))
        known_negative = known & ~positive
        background = (det_id.view(-1, 1) == -2) & (trk_id.view(1, -1) >= 0)
        low = current.score[:, 0] < 0.1
        negative = (known_negative | background) & (distance.detach() < float(hard_distance)) & low.view(-1, 1)
        positive = positive & low.view(-1, 1)
        selected = _pair_mask(positive, negative, distance.detach(), neg_per_pos, min_neg)
        if int(selected.sum().item()) > 0 and int(positive.sum().item()) + int(negative.sum().item()) > 0:
            loss = sigmoid_focal_loss(
                logit[selected],
                positive[selected].to(dtype=logit.dtype),
                alpha=-1,
                gamma=2.0,
                reduction="mean",
            )
            losses.append(loss)
            stats["positive"] += int(positive.sum().item())
            stats["negative"] += int((selected & ~positive).sum().item())
            low = current.score[:, 0] < 0.1
            stats["low_positive"] += int((positive & low.view(-1, 1)).sum().item())
            stats["frames"] += 1
        with torch.no_grad():
            tracked = advance_track_state(
                current,
                tracked,
                torch.sigmoid(logit.detach()),
                distance.detach(),
                class_names,
                birth_thresholds,
                association_alpha,
                age,
                high_cost_limit,
                low_cost_limit,
            )
    if not losses:
        return None, stats
    return torch.stack(losses).mean(), stats
