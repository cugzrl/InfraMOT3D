"""地图用法 B：运动预测是否从车道获益

GT 轨迹过去 1 s → 未来 0.5 / 1 / 2 s 位置，世界坐标系
预测器：匀速、沿车道直行优先传播、车道多分支最小误差（上界）、MLP 无地图、MLP 有地图
分组：直行 / 左转 / 右转 / 静止，路口内 / 路口外
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from shapely.prepared import prep

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from inframot3d.io import read_json, read_jsonl
from inframot3d.perception.lane_context import _parse_xy

VEHICLES = {"Car", "Van", "Bus", "Truck"}
HORIZONS = (0.5, 1.0, 2.0)
PAST = np.round(np.arange(-1.0, 0.001, 0.1), 3)
MAX_GAP = 0.3
STRIDE = 2
BRANCH_PTS = 15
BRANCH_STEP = 2.0
MAP_RADIUS = 160.0


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class LaneGraph:
    def __init__(self, path, center):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.lanes = {}
        for lid, lane in payload["LANE"].items():
            if lane.get("lane_type") == "BIKING":
                continue
            pts = _parse_xy(lane.get("centerline", []))
            if len(pts) < 2 or np.hypot(*(pts - center).T).min() > MAP_RADIUS:
                continue
            self.lanes[lid] = {"pts": pts, "succ": [s for s in lane.get("successors", []) if s != "None"], "turn": lane.get("turn_direction", "NONE"), "inter": bool(lane.get("is_intersection", False))}
        xy, owner = [], []
        for lid, lane in self.lanes.items():
            xy.append(lane["pts"])
            owner += [(lid, i) for i in range(len(lane["pts"]))]
        self.tree = cKDTree(np.concatenate(xy))
        self.owner = owner
        polys = [Polygon(_parse_xy(j["polygon"])) for j in payload["JUNCTION"].values() if len(j.get("polygon", [])) >= 3]
        polys = [p.buffer(0) for p in polys if np.hypot(*(np.array(p.centroid.coords[0]) - center)) < MAP_RADIUS + 50]
        self.junction = prep(unary_union(polys)) if polys else None

    def in_junction(self, xy):
        return bool(self.junction is not None and self.junction.contains(Point(float(xy[0]), float(xy[1]))))

    def match(self, xy, heading):
        idx = self.tree.query_ball_point(xy, 3.0)
        best, cost = None, 1e9
        for k in idx:
            lid, i = self.owner[k]
            pts = self.lanes[lid]["pts"]
            j = min(i, len(pts) - 2)
            d = pts[j + 1] - pts[j]
            diff = abs(wrap(np.arctan2(d[1], d[0]) - heading))
            if diff > np.deg2rad(45):
                continue
            c = np.hypot(*(pts[i] - xy)) + 3.0 * diff
            if c < cost:
                best, cost = (lid, i), c
        return best

    def branches(self, lid, i, length=40.0, max_branch=6):
        out = []
        stack = [(lid, self.lanes[lid]["pts"][i:], 0)]
        while stack and len(out) < max_branch:
            cur, pts, depth = stack.pop()
            seg = np.hypot(*np.diff(pts, axis=0).T).sum() if len(pts) > 1 else 0.0
            succ = [s for s in self.lanes[cur]["succ"] if s in self.lanes]
            if seg >= length or not succ or depth >= 6:
                out.append(pts)
                continue
            for s in succ:
                stack.append((s, np.vstack([pts, self.lanes[s]["pts"][1:]]), depth + 1))
        return out


def along(path, s):
    seg = np.hypot(*np.diff(path, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.clip(s, 0.0, cum[-1])
    x = np.interp(s, cum, path[:, 0])
    y = np.interp(s, cum, path[:, 1])
    k = np.clip(np.searchsorted(cum, s) - 1, 0, len(seg) - 1)
    d = path[k + 1] - path[k]
    return np.stack([x, y], -1), np.arctan2(d[..., 1], d[..., 0]), cum[-1]


def heading_change(path):
    d0 = path[1] - path[0]
    _, h, total = along(path, np.array([min(30.0, np.hypot(*np.diff(path, axis=0).T).sum())]))
    return wrap(h[0] - np.arctan2(d0[1], d0[0]))


def load_tracks(sequence_id, paths, info_by_seq):
    rows = list(read_jsonl(ROOT / "data/converted/v2x_seq_infrastructure" / paths[sequence_id]))
    frames = info_by_seq[sequence_id]
    tracks = defaultdict(list)
    for row in rows:
        info = frames[str(row["frame_id"])]
        # 外参逐帧读取，部分序列中途有约 0.003 rad 的更新
        pose = read_json(ROOT / "data/v2x-seq-infrastructure" / info["calib_virtuallidar_to_world_path"])
        rot = np.asarray(pose["rotation"], dtype=np.float64)
        trans = np.asarray(pose["translation"], dtype=np.float64).reshape(3)
        yaw_off = np.arctan2(rot[1, 0], rot[0, 0])
        t = int(row["timestamp"]) / 1e6
        for obj in row["objects"]:
            if obj["class_name"] not in VEHICLES:
                continue
            b = obj["box"]
            w = rot @ np.array([b[0], b[1], b[2]]) + trans
            tracks[obj["source_track_id"]].append((t, w[0], w[1], wrap(b[3] + yaw_off)))
    out = {}
    for tid, items in tracks.items():
        items.sort()
        out[tid] = np.array(items)
    return out, info["intersection_loc"], np.round(trans[:2], -1)


def interp_track(arr, times):
    t = arr[:, 0]
    if times.min() < t[0] - 1e-3 or times.max() > t[-1] + 1e-3:
        return None
    inside = (t >= times.min() - MAX_GAP) & (t <= times.max() + MAX_GAP)
    tt = t[inside]
    if len(tt) < 2 or np.diff(tt).max() > MAX_GAP:
        return None
    return np.stack([np.interp(times, t, arr[:, 1]), np.interp(times, t, arr[:, 2])], -1)


def build_samples(sequences, paths, info_by_seq, maps):
    samples = []
    avail = defaultdict(lambda: [0, 0])
    for sequence_id in sequences:
        tracks, loc, center = load_tracks(sequence_id, paths, info_by_seq)
        key = (loc, tuple(np.round(center, 0)))
        if key not in maps:
            maps[key] = LaneGraph(ROOT / "data/v2x_seq_maps" / ("%s.json" % loc), center)
        graph = maps[key]
        for tid, arr in tracks.items():
            for k in range(0, len(arr), STRIDE):
                t0 = arr[k, 0]
                past = interp_track(arr, t0 + PAST)
                if past is None:
                    continue
                fut = {}
                for h in HORIZONS:
                    p = interp_track(arr, np.array([t0, t0 + h]))
                    avail[h][1] += 1
                    if p is not None:
                        avail[h][0] += 1
                        fut[h] = p[1]
                if not fut:
                    continue
                v = (past[-1] - past[-6]) / 0.5
                speed = float(np.hypot(*v))
                yaw = arr[k, 3]
                if speed > 1.0 and np.cos(np.arctan2(v[1], v[0]) - yaw) < 0:
                    yaw = wrap(yaw + np.pi)
                heading = np.arctan2(v[1], v[0]) if speed > 1.0 else yaw
                samples.append({"seq": sequence_id, "loc": loc, "graph": key, "past": past, "fut": fut, "v": v, "speed": speed, "heading": heading, "junction": graph.in_junction(past[-1])})
    return samples, {h: a / max(b, 1) for h, (a, b) in avail.items()}, {h: b for h, (a, b) in avail.items()}


def to_local(xy, origin, heading):
    c, s = np.cos(heading), np.sin(heading)
    d = np.asarray(xy) - origin
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)


def to_world(local, origin, heading):
    c, s = np.cos(heading), np.sin(heading)
    return origin + np.stack([c * local[..., 0] - s * local[..., 1], s * local[..., 0] + c * local[..., 1]], -1)


def lane_features(sample, maps):
    graph = maps[sample["graph"]]
    p0 = sample["past"][-1]
    m = graph.match(p0, sample["heading"])
    feats = np.zeros((3, BRANCH_PTS * 2 + 1), np.float32)
    preds_straight = None
    preds_all = []
    if m is None:
        return feats.reshape(-1), None, [], False
    branches = graph.branches(*m)
    typed = {}
    for br in branches:
        if len(br) < 2:
            continue
        hc = heading_change(br)
        kind = 0 if abs(hc) < np.deg2rad(20) else (1 if hc > 0 else 2)
        if kind not in typed or abs(hc) < abs(typed[kind][1]):
            typed[kind] = (br, hc)
        proj, _, _ = along(br, np.array([0.0]))
        off = p0 - proj[0]
        pts, _, _ = along(br, np.array([sample["speed"] * h for h in HORIZONS]))
        preds_all.append(pts + off)
    for kind, (br, _) in typed.items():
        pts, _, _ = along(br, np.arange(1, BRANCH_PTS + 1) * BRANCH_STEP)
        feats[kind, :-1] = (to_local(pts, p0, sample["heading"]) / 20.0).reshape(-1)
        feats[kind, -1] = 1.0
    if typed:
        br = typed[min(typed, key=lambda k: abs(typed[k][1]))][0]
        proj, _, _ = along(br, np.array([0.0]))
        pts, _, _ = along(br, np.array([sample["speed"] * h for h in HORIZONS]))
        preds_straight = pts + (p0 - proj[0])
    return feats.reshape(-1), preds_straight, preds_all, True


def maneuver(sample):
    if 2.0 not in sample["fut"]:
        return "unknown"
    p0 = sample["past"][-1]
    d = sample["fut"][2.0] - p0
    if np.hypot(*d) < 1.0:
        return "static"
    local = to_local(sample["fut"][2.0], p0, sample["heading"])
    ang = np.arctan2(local[1], local[0])
    if abs(ang) < np.deg2rad(10):
        return "straight"
    return "left" if ang > 0 else "right"


def feature_matrix(samples, maps, use_map):
    rows, lane_ok = [], []
    for s in samples:
        local = to_local(s["past"], s["past"][-1], s["heading"]) / 10.0
        x = [local.reshape(-1), [s["speed"] / 10.0, float(s["junction"])]]
        if use_map:
            x.append(s["lane_feat"])
        rows.append(np.concatenate(x).astype(np.float32))
    return np.stack(rows)


def targets(samples):
    y = np.zeros((len(samples), len(HORIZONS), 2), np.float32)
    m = np.zeros((len(samples), len(HORIZONS)), bool)
    for i, s in enumerate(samples):
        for j, h in enumerate(HORIZONS):
            if h in s["fut"]:
                y[i, j] = to_local(s["fut"][h], s["past"][-1], s["heading"])
                m[i, j] = True
    return y, m


def train_mlp(x, y, m, seed, epochs=60):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 256), torch.nn.ReLU(), torch.nn.Linear(256, 256), torch.nn.ReLU(), torch.nn.Linear(256, len(HORIZONS) * 2))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    xt, yt, mt = map(torch.from_numpy, (x, y, m))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(xt), generator=g)
        for k in range(0, len(xt), 512):
            b = perm[k : k + 512]
            pred = net(xt[b]).view(-1, len(HORIZONS), 2)
            err = torch.linalg.norm(pred - yt[b], dim=-1)
            loss = (err * mt[b]).sum() / mt[b].sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return net


def summarize(errors, groups):
    out = {}
    for name, err in errors.items():
        out[name] = {}
        for gname, mask_fn in groups.items():
            out[name][gname] = {}
            for j, h in enumerate(HORIZONS):
                e = np.array([err[i][j] for i in range(len(err)) if mask_fn(i) and err[i][j] is not None])
                out[name][gname]["%.1fs" % h] = {"n": int(len(e)), "mean": float(e.mean()) if len(e) else None, "median": float(np.median(e)) if len(e) else None, "miss2m": float((e > 2.0).mean()) if len(e) else None}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    manifest = read_json(ROOT / "data/converted/v2x_seq_infrastructure/manifest.json")
    paths = {e["sequence_id"]: e["path"] for e in manifest["sequences"]}
    info_by_seq = {}
    for item in read_json(ROOT / "data/v2x-seq-infrastructure/data_info.json"):
        info_by_seq.setdefault(str(item["sequence_id"]), {})[str(item["frame_id"])] = item
    split = read_json(ROOT / "configs/datasets/v2xseq_sequence_split.json")
    maps = {}
    train, avail_train, n_train = build_samples(split["train"], paths, info_by_seq, maps)
    val, avail_val, n_val = build_samples(split["val"], paths, info_by_seq, maps)
    for s in train + val:
        s["lane_feat"], s["pred_lane"], s["pred_branches"], s["lane_ok"] = lane_features(s, maps)
    errors = {"cv": [], "lane_straight": [], "lane_oracle_min": []}
    for s in val:
        p0 = s["past"][-1]
        row_cv, row_ls, row_lo = [], [], []
        for j, h in enumerate(HORIZONS):
            if h not in s["fut"]:
                row_cv.append(None), row_ls.append(None), row_lo.append(None)
                continue
            gt = s["fut"][h]
            cv = p0 + s["v"] * h
            row_cv.append(float(np.hypot(*(cv - gt))))
            row_ls.append(float(np.hypot(*(s["pred_lane"][j] - gt))) if s["pred_lane"] is not None else float(np.hypot(*(cv - gt))))
            cands = [cv] + [b[j] for b in s["pred_branches"]]
            row_lo.append(float(min(np.hypot(*(c - gt)) for c in cands)))
        errors["cv"].append(row_cv)
        errors["lane_straight"].append(row_ls)
        errors["lane_oracle_min"].append(row_lo)
    ytr, mtr = targets(train)
    yva, mva = targets(val)
    for use_map in (False, True):
        xtr = feature_matrix(train, maps, use_map)
        xva = feature_matrix(val, maps, use_map)
        mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
        for seed in args.seeds:
            net = train_mlp((xtr - mu) / sd, ytr, mtr, seed)
            with torch.no_grad():
                pred = net(torch.from_numpy((xva - mu) / sd)).view(-1, len(HORIZONS), 2).numpy()
            err = np.linalg.norm(pred - yva, axis=-1)
            errors["mlp_%s_s%d" % ("map" if use_map else "nomap", seed)] = [[float(err[i, j]) if mva[i, j] else None for j in range(len(HORIZONS))] for i in range(len(val))]
    man = [maneuver(s) for s in val]
    groups = {"all": lambda i: True}
    for kind in ("straight", "left", "right", "static"):
        groups[kind] = (lambda k: lambda i: man[i] == k)(kind)
    groups["in_junction"] = lambda i: val[i]["junction"]
    groups["out_junction"] = lambda i: not val[i]["junction"]
    for kind in ("straight", "left", "right"):
        groups["%s_in_junction" % kind] = (lambda k: lambda i: man[i] == k and val[i]["junction"])(kind)
    groups["lane_matched"] = lambda i: val[i]["lane_ok"]
    result = {
        "future_label_availability": {"train": {str(h): v for h, v in avail_train.items()}, "val": {str(h): v for h, v in avail_val.items()}, "anchors_train": {str(h): v for h, v in n_train.items()}, "anchors_val": {str(h): v for h, v in n_val.items()}},
        "samples": {"train": len(train), "val": len(val)},
        "lane_match_rate_val": float(np.mean([s["lane_ok"] for s in val])),
        "maneuver_counts_val": {k: man.count(k) for k in set(man)},
        "junction_rate_val": float(np.mean([s["junction"] for s in val])),
        "errors": summarize(errors, groups),
    }
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "errors"}, indent=1))
    for name in errors:
        print(name, {g: result["errors"][name][g]["2.0s"]["mean"] for g in ("all", "straight", "left", "right", "in_junction")})


if __name__ == "__main__":
    main()
