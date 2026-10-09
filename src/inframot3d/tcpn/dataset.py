"""逐帧数据集，特征只由候选、在线轨迹与当前 BEV 构成"""

import pickle
import random
from pathlib import Path

import numpy as np
import torch

from inframot3d.tcpn.bev_cache import BevStore
from inframot3d.tcpn.matching import STATUS_BG
from inframot3d.tcpn.model import HIST_RADIUS, REL_K, REL_RADIUS, SITE_RADIUS
from inframot3d.tcpn.sampling import candidate_sites
from inframot3d.tcpn.track_labels import TRACK_STATES

MEAN_SIZE = np.array([4.5, 1.9, 1.6], np.float32)


def _logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def _onehot(cls, n=4):
    out = np.zeros((len(cls), n), np.float32)
    out[np.arange(len(cls)), np.clip(cls, 0, n - 1)] = 1.0
    return out


def cand_features(box, score, cls):
    return np.concatenate(
        [
            box[:, 0:1] / 100.0,
            box[:, 1:2] / 40.0,
            box[:, 2:3] / 3.0,
            np.sin(box[:, 3:4]),
            np.cos(box[:, 3:4]),
            np.log(np.clip(box[:, 4:7], 0.1, None) / MEAN_SIZE),
            _onehot(cls),
            _logit(score)[:, None] / 4.0,
            score[:, None],
            np.sqrt(score)[:, None],
            np.hypot(box[:, 0], box[:, 1])[:, None] / 100.0,
        ],
        axis=1,
    ).astype(np.float32)


def track_features(trk):
    box = trk["box"]
    return np.concatenate(
        [
            trk["pred"][:, 0:1] / 100.0,
            trk["pred"][:, 1:2] / 40.0,
            box[:, 2:3] / 3.0,
            np.sin(box[:, 3:4]),
            np.cos(box[:, 3:4]),
            np.log(np.clip(box[:, 4:7], 0.1, None) / MEAN_SIZE),
            _onehot(trk["cls"]),
            trk["score_last"][:, None],
            trk["score_mean"][:, None],
            np.log1p(trk["gap"] * 10.0)[:, None] / 3.0,
            np.log1p(trk["hits"])[:, None] / 3.0,
            trk["vel"] / 10.0,
        ],
        axis=1,
    ).astype(np.float32)


def _relative(src_xy, src_yaw, dst_xy):
    d = dst_xy[None, :, :] - src_xy[:, None, :]
    c, s = np.cos(src_yaw)[:, None], np.sin(src_yaw)[:, None]
    du = c * d[..., 0] + s * d[..., 1]
    dv = -s * d[..., 0] + c * d[..., 1]
    return du, dv, np.hypot(du, dv)


def _angle(a, b):
    diff = b[None, :] - a[:, None]
    return np.stack([np.sin(diff), np.cos(diff), np.sin(2 * diff), np.cos(2 * diff)], axis=-1)


def pair_features(cbox, cscore, ccls, trk):
    """轨迹相对候选的特征 [N,M,14] 与候选相对轨迹的特征 [M,N,14]"""
    n, m = len(cbox), len(trk["pred"])
    if n == 0 or m == 0:
        return np.zeros((n, m, 14), np.float32), np.zeros((m, n, 14), np.float32), np.full((n, m), 1e3, np.float32)
    cxy, cyaw = cbox[:, :2], cbox[:, 3]
    txy, tyaw = trk["pred"], trk["box"][:, 3]
    du, dv, dist = _relative(cxy, cyaw, txy)
    ang = _angle(cyaw, tyaw)
    lr = np.log(np.clip(trk["box"][None, :, 4], 0.1, None) / np.clip(cbox[:, None, 4], 0.1, None))
    wr = np.log(np.clip(trk["box"][None, :, 5], 0.1, None) / np.clip(cbox[:, None, 5], 0.1, None))
    c, s = np.cos(cyaw)[:, None], np.sin(cyaw)[:, None]
    vel = trk["vel"]
    v_along = c * vel[None, :, 0] + s * vel[None, :, 1]
    v_lat = -s * vel[None, :, 0] + c * vel[None, :, 1]
    gap = np.broadcast_to(np.log1p(trk["gap"] * 10.0)[None, :] / 3.0, (n, m))
    hits = np.broadcast_to(np.log1p(trk["hits"])[None, :] / 3.0, (n, m))
    sdiff = trk["score_last"][None, :] - cscore[:, None]
    ct = np.concatenate(
        [np.stack([du / 4, dv / 4, dist / 4], -1), ang, np.stack([lr, wr, gap, hits, sdiff, v_along / 10, v_lat / 10], -1)], -1
    ).astype(np.float32)
    du2, dv2, _ = _relative(txy, tyaw, cxy)
    ang2 = _angle(tyaw, cyaw)
    same = (ccls[None, :] == trk["cls"][:, None]).astype(np.float32)
    tc = np.concatenate(
        [
            np.stack([du2 / 4, dv2 / 4, dist.T / 4], -1),
            ang2,
            np.stack([-lr.T, -wr.T, np.broadcast_to(cscore[None, :], (m, n)), np.broadcast_to(_logit(cscore)[None, :] / 4, (m, n)), -sdiff.T, same, gap.T], -1),
        ],
        -1,
    ).astype(np.float32)
    return ct, tc, dist.astype(np.float32)


def relation_features(cbox, cscore, ccls):
    n = len(cbox)
    k = REL_K
    index = np.zeros((n, k), np.int64)
    mask = np.zeros((n, k), bool)
    feat = np.zeros((n, k, 11), np.float32)
    if n <= 1:
        return index, mask, feat
    du, dv, dist = _relative(cbox[:, :2], cbox[:, 3], cbox[:, :2])
    np.fill_diagonal(dist, 1e3)
    kk = min(k, n - 1)
    order = np.argsort(dist, axis=1)[:, :kk]
    rows = np.arange(n)[:, None]
    index[:, :kk] = order
    mask[:, :kk] = dist[rows, order] < REL_RADIUS
    ang = _angle(cbox[:, 3], cbox[:, 3])[rows, order]
    lr = np.log(np.clip(cbox[order, 4], 0.1, None) / np.clip(cbox[:, None, 4], 0.1, None))
    wr = np.log(np.clip(cbox[order, 5], 0.1, None) / np.clip(cbox[:, None, 5], 0.1, None))
    sd = cscore[order] - cscore[:, None]
    same = (ccls[order] == ccls[:, None]).astype(np.float32)
    feat[:, :kk] = np.concatenate(
        [np.stack([du[rows, order] / 4, dv[rows, order] / 4, dist[rows, order] / 4], -1), ang, np.stack([lr, wr, sd, same], -1)], -1
    )
    return index, mask, feat


def perturb_tracks(trk, sample, rng):
    """真实跟踪误差增强：丢弃、位置扰动、注入背景伪轨迹，伪轨迹持续性标签为 0"""
    m = len(trk["pred"])
    keep = np.array([rng.random() > 0.1 for _ in range(m)], bool) if m else np.zeros(0, bool)
    out = {k: v[keep].copy() for k, v in trk.items()}
    jitter = np.array([rng.random() < 0.5 for _ in range(len(out["pred"]))], bool)
    noise = np.array([[rng.gauss(0, 0.3), rng.gauss(0, 0.3)] for _ in range(len(out["pred"]))], np.float32).reshape(-1, 2)
    out["pred"] = out["pred"] + noise * jitter[:, None]
    bg = np.where((sample["cand_status"] == STATUS_BG) & (sample["cand_score"] >= 0.05))[0]
    fakes = [int(i) for i in bg if rng.random() < 0.15][:2]
    if fakes:
        f = len(fakes)
        box = sample["cand_box"][fakes]
        add = {
            "box": box,
            "pred": box[:, :2].copy(),
            "vel": np.zeros((f, 2), np.float32),
            "gap": np.full(f, 0.1, np.float32),
            "hits": np.array([rng.choice([1, 2, 3]) for _ in fakes], np.float32),
            "score_last": sample["cand_score"][fakes],
            "score_mean": sample["cand_score"][fakes],
            "cls": sample["cand_cls"][fakes],
            "state": np.full(f, TRACK_STATES.index("false"), np.int64),
            "target": -np.ones(f, np.int64),
            "exist": np.zeros(f, np.float32),
            "pos_target": np.zeros((f, 2), np.float32),
            "pos_valid": np.zeros(f, np.float32),
        }
        out = {k: np.concatenate([out[k], add[k]], axis=0).astype(out[k].dtype) for k in out}
    return out


class FrameDataset(torch.utils.data.Dataset):
    def __init__(self, sample_root, bev_root, sequences, query="grae", perturb=False, seed=0, need_bev=True, pattern=None, sampler=None):
        self.samples = []
        for sequence_id in sequences:
            with (Path(sample_root) / ("%s.pkl" % sequence_id)).open("rb") as stream:
                self.samples.extend(pickle.load(stream))
        self.bev = BevStore(bev_root) if need_bev else None
        self.sites = None
        if pattern:
            self.sites = [frame_sites(pattern, s["cand_box"], sampler, s["sequence_id"], s["frame_id"]) for s in self.samples]
        self.query = query
        self.perturb = perturb
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        trk = sample["trk"] if self.query == "grae" else sample["gt_trk"]
        if self.perturb:
            rng = random.Random(hash((self.seed, self.epoch, index)) & 0xFFFFFFFF)
            trk = perturb_tracks(trk, sample, rng)
        item = make_item(sample["cand_box"], sample["cand_score"], sample["cand_cls"], trk)
        item.update(
            {
                "cand_gt_box": sample["cand_gt_box"],
                "cand_quality": sample["cand_quality"],
                "trk_exist": trk["exist"],
                "trk_pos_target": trk["pos_target"],
                "trk_pos_valid": trk["pos_valid"],
                "trk_state": trk["state"],
                "trk_target": trk["target"],
                "index": index,
            }
        )
        if self.bev is not None:
            item["bev"] = np.asarray(self.bev.frame(sample["sequence_id"], sample["frame_id"]))
        if self.sites is not None:
            item["cand_sites"] = self.sites[index]
        return item


def frame_sites(pattern, cbox, sampler, sequence_id, frame_id):
    """训练与推理共用的采样点构造"""
    lane = sampler.lane_geometry(sequence_id, frame_id, cbox) if pattern in ("lane", "laneline") else None
    return candidate_sites(pattern, cbox, lane)


def make_item(cbox, cscore, ccls, trk):
    """训练与推理共用的输入构造，保证特征来源一致"""
    ct, tc, dist = pair_features(cbox, cscore, ccls, trk)
    site = -np.ones(len(cbox), np.int64)
    if len(trk["pred"]) and len(cbox):
        nearest = dist.argmin(axis=1)
        ok = dist[np.arange(len(cbox)), nearest] < SITE_RADIUS
        site[ok] = nearest[ok]
    rel_index, rel_mask, rel_feat = relation_features(cbox, cscore, ccls)
    n, m = len(cbox), len(trk["pred"])
    return {
        "cand_feat": cand_features(cbox, cscore, ccls),
        "cand_xy": cbox[:, :2],
        "cand_yaw": cbox[:, 3],
        "cand_box": cbox,
        "cand_score": cscore,
        "cand_gt_box": np.ones((n, 7), np.float32),
        "cand_quality": -np.ones(n, np.float32),
        "trk_feat": track_features(trk) if m else np.zeros((0, 18), np.float32),
        "trk_xy": trk["pred"],
        "trk_yaw": trk["box"][:, 3] if m else np.zeros(0, np.float32),
        "trk_exist": -np.ones(m, np.float32),
        "trk_pos_target": np.zeros((m, 2), np.float32),
        "trk_pos_valid": np.zeros(m, np.float32),
        "trk_state": -np.ones(m, np.int64),
        "trk_target": -np.ones(m, np.int64),
        "pair_feat": ct,
        "pair_feat_t": tc,
        "pair_dist": dist,
        "site_index": site,
        "rel_index": rel_index,
        "rel_mask": rel_mask,
        "rel_feat": rel_feat,
        "index": 0,
    }


def collate(items):
    b = len(items)
    n = max(1, max(len(it["cand_xy"]) for it in items))
    m = max(1, max(len(it["trk_xy"]) for it in items))
    k = items[0]["rel_index"].shape[1] if items[0]["rel_index"].ndim == 2 else REL_K

    def pad(key, shape, dtype, fill=0):
        out = np.full((b,) + shape, fill, dtype=dtype)
        for i, it in enumerate(items):
            v = it[key]
            if v.size == 0:
                continue
            sl = tuple(slice(0, s) for s in v.shape)
            out[(i,) + sl] = v
        return torch.from_numpy(out)

    batch = {
        "cand_feat": pad("cand_feat", (n, 16), np.float32),
        "cand_xy": pad("cand_xy", (n, 2), np.float32),
        "cand_yaw": pad("cand_yaw", (n,), np.float32),
        "cand_box": pad("cand_box", (n, 7), np.float32, 1.0),
        "cand_gt_box": pad("cand_gt_box", (n, 7), np.float32, 1.0),
        "cand_quality": pad("cand_quality", (n,), np.float32, -1.0),
        "cand_score": pad("cand_score", (n,), np.float32),
        "trk_feat": pad("trk_feat", (m, 18), np.float32),
        "trk_xy": pad("trk_xy", (m, 2), np.float32, -1e3),
        "trk_yaw": pad("trk_yaw", (m,), np.float32),
        "trk_exist": pad("trk_exist", (m,), np.float32, -1.0),
        "trk_pos_target": pad("trk_pos_target", (m, 2), np.float32),
        "trk_pos_valid": pad("trk_pos_valid", (m,), np.float32),
        "trk_state": pad("trk_state", (m,), np.int64, -1),
        "trk_target": pad("trk_target", (m,), np.int64, -1),
        "pair_feat": pad("pair_feat", (n, m, 14), np.float32),
        "pair_feat_t": pad("pair_feat_t", (m, n, 14), np.float32),
        "pair_dist": pad("pair_dist", (n, m), np.float32, 1e3),
        "site_index": pad("site_index", (n,), np.int64, -1),
        "rel_index": pad("rel_index", (n, k), np.int64),
        "rel_mask": pad("rel_mask", (n, k), bool, False),
        "rel_feat": pad("rel_feat", (n, k, 11), np.float32),
        "index": torch.tensor([it["index"] for it in items]),
    }
    batch["cand_mask"] = torch.zeros(b, n, dtype=torch.bool)
    batch["trk_mask"] = torch.zeros(b, m, dtype=torch.bool)
    for i, it in enumerate(items):
        batch["cand_mask"][i, : len(it["cand_xy"])] = True
        batch["trk_mask"][i, : len(it["trk_xy"])] = True
    if "bev" in items[0]:
        batch["bev"] = torch.from_numpy(np.stack([it["bev"] for it in items]))
    if "cand_sites" in items[0]:
        batch["cand_sites"] = pad("cand_sites", (n, 25, 2), np.float32)
    return batch


def to_device(batch, device):
    out = {}
    for key, value in batch.items():
        if key == "bev":
            out[key] = value.to(device, non_blocking=True).float()
        else:
            out[key] = value.to(device, non_blocking=True)
    return out
