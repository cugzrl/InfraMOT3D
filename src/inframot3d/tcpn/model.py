"""Track-conditioned Perception Network 阶段 B

候选为中心的网络，CenterPoint 已输出的车辆候选是唯一的观测单元
- 历史条件：每个候选对附近在线轨迹做带空位 token 的注意力，得到轨迹支持特征
- 局部 BEV：候选框坐标系下 5x5 采样，再加最近轨迹预测位置处 3x3 采样，查询向量由候选与历史共同决定
- 候选关系：每个候选对 8 个近邻候选做注意力，用来处理重复框与相邻目标竞争
- 输出：观测质量、框残差、关联嵌入，轨迹侧输出持续性与位置修正
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from inframot3d.tcpn.bev_cache import sample_bev

CAND_DIM = 16
TRACK_DIM = 18
PAIR_DIM = 14
REL_DIM = 11
CAND_GRID = [(u, v) for u in (-2.4, -1.2, 0.0, 1.2, 2.4) for v in (-1.6, -0.8, 0.0, 0.8, 1.6)]
TRACK_GRID = [(u, v) for u in (-0.8, 0.0, 0.8) for v in (-0.8, 0.0, 0.8)]
HIST_RADIUS = 6.0
SITE_RADIUS = 4.0
REL_RADIUS = 8.0
REL_K = 8


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(inplace=True), nn.Linear(h, o))


def rotate(vec, yaw, inverse=False):
    c, s = torch.cos(yaw), torch.sin(yaw)
    if inverse:
        s = -s
    x, y = vec[..., 0], vec[..., 1]
    return torch.stack([c * x - s * y, s * x + c * y], dim=-1)


class PairAttention(nn.Module):
    """查询对一组带对特征的键值做多头注意力，额外一个可学习空位，全被遮挡时退化为空位"""

    def __init__(self, hidden, heads=4):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(hidden, hidden)
        self.null_k = nn.Parameter(torch.zeros(hidden))
        self.null_v = nn.Parameter(torch.zeros(hidden))
        self.out = nn.Linear(hidden, hidden)

    def forward(self, query, key, value, mask):
        b, n, m, h = key.shape
        d = h // self.heads
        q = self.q(query).view(b, n, 1, self.heads, d)
        key = torch.cat([self.null_k.view(1, 1, 1, h).expand(b, n, 1, h), key], dim=2).view(b, n, m + 1, self.heads, d)
        value = torch.cat([self.null_v.view(1, 1, 1, h).expand(b, n, 1, h), value], dim=2).view(b, n, m + 1, self.heads, d)
        mask = torch.cat([torch.ones(b, n, 1, dtype=torch.bool, device=mask.device), mask], dim=2)
        logit = (q * key).sum(-1) / math.sqrt(d)
        logit = logit.masked_fill(~mask[..., None], -1e4)
        weight = logit.softmax(dim=2)
        out = (weight[..., None] * value).sum(2).reshape(b, n, h)
        return self.out(out), weight.mean(-1)


class Block(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.attn = PairAttention(hidden)
        self.ffn = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden * 2), nn.ReLU(inplace=True), nn.Linear(hidden * 2, hidden))

    def forward(self, x, key, value, mask):
        out, weight = self.attn(self.norm(x), key, value, mask)
        x = x + out
        return x + self.ffn(x), weight


class TCPN(nn.Module):
    def __init__(self, hidden=128, bev_dim=128, use_hist=True, use_bev=True, use_site=True, use_relation=True):
        super().__init__()
        self.use_hist = bool(use_hist)
        self.use_bev = bool(use_bev)
        self.use_site = bool(use_site) and self.use_hist and self.use_bev
        self.use_relation = bool(use_relation)
        self.cand_enc = mlp(CAND_DIM, hidden, hidden)
        if self.use_hist:
            self.trk_enc = mlp(TRACK_DIM, hidden, hidden)
            self.pair_k = mlp(hidden + PAIR_DIM, hidden, hidden)
            self.pair_v = mlp(hidden + PAIR_DIM, hidden, hidden)
            self.hist = Block(hidden)
            self.back_k = mlp(hidden + PAIR_DIM, hidden, hidden)
            self.back_v = mlp(hidden + PAIR_DIM, hidden, hidden)
            self.track_block = Block(hidden)
            self.exist_head = mlp(hidden, hidden, 1)
            self.pos_head = mlp(hidden, hidden, 2)
            nn.init.zeros_(self.pos_head[-1].weight)
            nn.init.zeros_(self.pos_head[-1].bias)
        if self.use_bev:
            self.bev_proj = nn.Sequential(nn.LayerNorm(bev_dim), nn.Linear(bev_dim, hidden))
            self.site_pe = mlp(3, hidden, hidden)
            self.bev = Block(hidden)
            if self.use_hist:
                self.track_bev = Block(hidden)
        if self.use_relation:
            self.rel_k = mlp(hidden + REL_DIM, hidden, hidden)
            self.rel_v = mlp(hidden + REL_DIM, hidden, hidden)
            self.relation = Block(hidden)
        self.quality = mlp(hidden, hidden, 1)
        self.box = mlp(hidden, hidden, 6)
        self.embed = mlp(hidden, hidden, 64)
        nn.init.zeros_(self.box[-1].weight)
        nn.init.zeros_(self.box[-1].bias)
        self.register_buffer("cand_grid", torch.tensor(CAND_GRID, dtype=torch.float32), persistent=False)
        self.register_buffer("track_grid", torch.tensor(TRACK_GRID, dtype=torch.float32), persistent=False)

    def _bev_tokens(self, bev, center, yaw, grid, kind):
        offset = rotate(grid.view(1, 1, -1, 2), yaw[..., None])
        sites = center[..., None, :] + offset
        feat = sample_bev(bev, sites)
        pe_in = torch.cat([grid.view(1, 1, -1, 2).expand(*sites.shape[:2], -1, -1) / 2.4, torch.full_like(sites[..., :1], float(kind))], dim=-1)
        return self.bev_proj(feat) + self.site_pe(pe_in)

    def forward(self, batch):
        cand = batch["cand_feat"]
        cmask = batch["cand_mask"]
        cxy = batch["cand_xy"]
        cyaw = batch["cand_yaw"]
        x = self.cand_enc(cand)
        out = {}
        if self.use_hist:
            tmask = batch["trk_mask"]
            t = self.trk_enc(batch["trk_feat"])
            pair = batch["pair_feat"]
            near = batch["pair_dist"] < HIST_RADIUS
            pmask = near & tmask[:, None, :] & cmask[:, :, None]
            tk = t[:, None].expand(-1, x.shape[1], -1, -1)
            pin = torch.cat([tk, pair], dim=-1)
            x, w = self.hist(x, self.pair_k(pin), self.pair_v(pin), pmask)
            out["hist_weight"] = w
        if self.use_bev:
            bev = batch["bev"]
            if "cand_sites" in batch:
                # 地图用法 A：外部给定的 25 个采样点，位置编码用候选坐标系下的偏移
                sites = batch["cand_sites"]
                rel = rotate(sites - cxy[..., None, :], cyaw[..., None], inverse=True) / 2.4
                pe_in = torch.cat([rel.clamp(-3, 3), torch.zeros_like(rel[..., :1])], dim=-1)
                tokens = self.bev_proj(sample_bev(bev, sites)) + self.site_pe(pe_in)
            else:
                tokens = self._bev_tokens(bev, cxy, cyaw, self.cand_grid, 0)
            tok_mask = cmask[..., None].expand(-1, -1, tokens.shape[2])
            if self.use_site:
                nearest = batch["site_index"]
                valid = nearest >= 0
                gather = nearest.clamp(min=0)
                site_xy = torch.gather(batch["trk_xy"], 1, gather[..., None].expand(-1, -1, 2))
                site_yaw = torch.gather(batch["trk_yaw"], 1, gather)
                site = self._bev_tokens(bev, site_xy, site_yaw, self.track_grid, 1)
                tokens = torch.cat([tokens, site], dim=2)
                tok_mask = torch.cat([tok_mask, (valid & cmask)[..., None].expand(-1, -1, site.shape[2])], dim=2)
            x, _ = self.bev(x, tokens, tokens, tok_mask)
        if self.use_relation:
            idx = batch["rel_index"]
            rmask = batch["rel_mask"]
            b, n, k = idx.shape
            nb = torch.gather(x, 1, idx.reshape(b, n * k, 1).expand(-1, -1, x.shape[-1])).view(b, n, k, -1)
            rin = torch.cat([nb, batch["rel_feat"]], dim=-1)
            x, _ = self.relation(x, self.rel_k(rin), self.rel_v(rin), rmask)
        out["quality"] = self.quality(x).squeeze(-1)
        out["box"] = self.box(x)
        out["embed"] = F.normalize(self.embed(x), dim=-1)
        out["cand_hidden"] = x
        if self.use_hist:
            pair_t = batch["pair_feat_t"]
            ck = x[:, None].expand(-1, t.shape[1], -1, -1)
            back = torch.cat([ck, pair_t], dim=-1)
            bmask = pmask.transpose(1, 2)
            t, _ = self.track_block(t, self.back_k(back), self.back_v(back), bmask)
            if self.use_bev:
                site = self._bev_tokens(batch["bev"], batch["trk_xy"], batch["trk_yaw"], self.track_grid, 1)
                t, _ = self.track_bev(t, site, site, tmask[..., None].expand(-1, -1, site.shape[2]))
            out["exist"] = self.exist_head(t).squeeze(-1)
            out["trk_pos"] = self.pos_head(t)
            out["trk_hidden"] = t
        return out


def focal_bce(logit, target, valid, gamma=2.0):
    if not bool(valid.any()):
        return logit.sum() * 0.0
    logit = logit[valid]
    target = target[valid]
    ce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    p = torch.sigmoid(logit)
    pt = p * target + (1 - p) * (1 - target)
    return ((1 - pt).pow(gamma) * ce).sum() / max(float((target > 0.5).sum()), 1.0)


def box_target(cand_box, gt_box):
    """候选坐标系下的残差，航向按 pi 周期折到 [-pi/2, pi/2)"""
    d = rotate(gt_box[..., :2] - cand_box[..., :2], cand_box[..., 3], inverse=True)
    dyaw = gt_box[..., 3] - cand_box[..., 3]
    dyaw = (dyaw + math.pi / 2) % math.pi - math.pi / 2
    dsize = torch.log(gt_box[..., 4:7].clamp(min=0.1) / cand_box[..., 4:7].clamp(min=0.1))
    return torch.cat([d, dyaw[..., None] / 0.3, dsize / 0.2], dim=-1)


def apply_box(cand_box, delta):
    out = cand_box.clone()
    out[..., :2] = cand_box[..., :2] + rotate(delta[..., :2], cand_box[..., 3])
    out[..., 3] = cand_box[..., 3] + delta[..., 2] * 0.3
    out[..., 4:7] = cand_box[..., 4:7] * torch.exp(delta[..., 3:6] * 0.2)
    return out


def loss_fn(out, batch, weights=None):
    weights = weights or {}
    q = batch["cand_quality"]
    cmask = batch["cand_mask"]
    loss_q = focal_bce(out["quality"], q.clamp(min=0), cmask & (q >= 0))
    tp = cmask & (q > 0.5)
    if bool(tp.any()):
        target = box_target(batch["cand_box"], batch["cand_gt_box"])
        loss_box = F.smooth_l1_loss(out["box"][tp], target[tp], beta=0.1)
    else:
        loss_box = out["box"].sum() * 0.0
    total = loss_q + weights.get("box", 0.5) * loss_box
    logs = {"q": float(loss_q.detach()), "box": float(loss_box.detach())}
    if "exist" in out:
        e = batch["trk_exist"]
        valid = batch["trk_mask"] & (e >= 0)
        if bool(valid.any()):
            loss_e = F.binary_cross_entropy_with_logits(out["exist"][valid], e[valid])
        else:
            loss_e = out["exist"].sum() * 0.0
        pv = batch["trk_mask"] & (batch["trk_pos_valid"] > 0)
        if bool(pv.any()):
            loss_p = F.smooth_l1_loss(out["trk_pos"][pv], (batch["trk_pos_target"] - batch["trk_xy"])[pv], beta=0.1)
        else:
            loss_p = out["trk_pos"].sum() * 0.0
        total = total + weights.get("exist", 0.5) * loss_e + weights.get("pos", 0.5) * loss_p
        logs.update({"exist": float(loss_e.detach()), "pos": float(loss_p.detach())})
    logs["total"] = float(total.detach())
    return total, logs
