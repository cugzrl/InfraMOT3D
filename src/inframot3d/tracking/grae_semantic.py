import torch
from torch import nn


class VehicleSemanticGate(nn.Module):
    # 只给同一评估大类、细类不同的候选对一个关联门控
    # 零权重和负偏置让初始门控接近关闭，行为回到硬类别约束

    def __init__(self, channels, num_classes):
        super().__init__()
        width = 32
        self.embed = nn.Embedding(int(num_classes), width)
        self.det_proj = nn.Linear(int(channels), width)
        self.trk_proj = nn.Linear(int(channels), width)
        self.gate = nn.Sequential(
            nn.Linear(width * 3 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -4.0)

    def forward(self, det_instance, trk_motion, det_classes, trk_classes, distance):
        det_embed = self.embed(det_classes.view(-1))
        trk_embed = self.embed(trk_classes.view(-1))
        det_feature = self.det_proj(det_instance)
        trk_feature = self.trk_proj(trk_motion)
        det_count = det_feature.shape[0]
        track_count = trk_feature.shape[0]
        pair = torch.cat(
            [
                det_embed[:, None, :].expand(det_count, track_count, -1),
                trk_embed[None, :, :].expand(det_count, track_count, -1),
                (det_feature[:, None, :] - trk_feature[None, :, :]).abs(),
                distance.unsqueeze(-1),
            ],
            dim=-1,
        )
        logit = self.gate(pair).squeeze(-1)
        # 2m 以外的跨类对保持关闭，避免把不同车辆连到一起
        return logit.masked_fill(distance >= 2.0, -20.0)


def semantic_clip_loss(
    model,
    gate,
    frames,
    device,
    class_names,
    birth_thresholds,
    group_table,
    association_alpha=0.24,
    age=12,
    hard_distance=8.0,
    neg_per_pos=4,
    min_neg=8,
):
    from inframot3d.tracking.grae_adapter import _init_features, _to_instance, spatial_inputs, temporal_vector
    from inframot3d.tracking.grae_recovery import _pair_mask, advance_track_state
    from torchvision.ops import sigmoid_focal_loss

    tracked = None
    losses = []
    stats = {"positive": 0, "negative": 0, "frames": 0}
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
            from inframot3d.tracking.grae_recovery import _birth_mask

            tracked = tracked[_birth_mask(tracked, class_names, birth_thresholds)]
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
            grae_affinity = torch.sigmoid(affinity_scores[-1][..., 0].transpose(0, 1))
        gate.train()
        logit = gate(
            current.instance_feature.detach(),
            tracked.motion_feature.detach(),
            current.classes,
            tracked.classes,
            distance.detach(),
        )
        det_group = group_table[current.classes.view(-1)]
        trk_group = group_table[tracked.classes.view(-1)]
        cross = (det_group.view(-1, 1) == trk_group.view(1, -1)) & (
            current.classes.view(-1, 1) != tracked.classes.view(1, -1)
        )
        det_id = current.tracking_id[:, 0]
        trk_id = tracked.tracking_id[:, 0]
        known = (det_id.view(-1, 1) >= 0) & (trk_id.view(1, -1) >= 0)
        close = distance.detach() < 2.0
        positive = cross & known & close & (det_id.view(-1, 1) == trk_id.view(1, -1))
        negative = cross & known & close & (det_id.view(-1, 1) != trk_id.view(1, -1)) & (distance.detach() < float(hard_distance))
        selected = _pair_mask(positive, negative, distance.detach(), neg_per_pos, min_neg) & cross
        affinity = grae_affinity.detach().clone()
        affinity[cross] = torch.sigmoid(logit)[cross]
        det_group_invalid = det_group.view(-1, 1) != trk_group.view(1, -1)
        if int(selected.sum().item()) > 0:
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
            stats["frames"] += 1
        with torch.no_grad():
            tracked = advance_track_state(
                current,
                tracked,
                affinity.detach(),
                distance.detach(),
                class_names,
                birth_thresholds,
                association_alpha,
                age,
                0.9,
                0.8,
                invalid=det_group_invalid,
                direct_mask=cross,
            )
    if not losses:
        return None, stats
    return torch.stack(losses).mean(), stats
