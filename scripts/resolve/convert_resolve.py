"""RESOLVE 转换：会话级划分、三档分辨率共享同一帧与同一 GT 集合、GT 数据库

复用官方旋转角、框解析与类别映射，原始数据只读
"""

import argparse
import collections
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party" / "OpenPCDet"))

from pcdet.datasets.resolve.resolve_dataset import RES_DIRS, ROT_DEG, load_resolve_points
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu

# 与官方 create_sunlakes_data_v3.py 相同的类别重命名
RENAME = {"van": "barrier", "construction vehicle": "construction_vehicle", "golf cart": "traffic_cone"}


def parse_frame(frame):
    labels = [l for l in frame["labels"] if l["drawType"] == "box3d"]
    if not labels:
        return None
    pts = np.array([l["points"] for l in labels], dtype=np.float64)
    theta = np.deg2rad(ROT_DEG)
    rz = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    locs = pts[:, :3] @ rz.T
    yaw = pts[:, 5] + theta
    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
    boxes = np.concatenate([locs, pts[:, 6:9], yaw[:, None]], axis=1).astype(np.float32)
    raw = np.array([l["label"] for l in labels], dtype=object)
    names = np.array([RENAME.get(n, n) for n in raw], dtype=object)
    ids = np.array([int(l.get("id", -1)) for l in labels], dtype=np.int64)
    count = np.array([int(l.get("pointsCount") if l.get("pointsCount") is not None else -1) for l in labels], dtype=np.int64)
    return boxes, names, raw, ids, count


def box_point_index(points, boxes):
    if len(boxes) == 0:
        return np.full(len(points), -1, dtype=np.int64)
    p = torch.from_numpy(points[:, :3]).float().cuda()[None]
    b = torch.from_numpy(boxes).float().cuda()[None]
    return points_in_boxes_gpu(p, b)[0].long().cpu().numpy()


def session_split(sessions, every):
    ordered = sorted(sessions)
    val = [s for i, s in enumerate(ordered) if i % every == every // 2]
    return [s for s in ordered if s not in val], val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="/home/kemove/devdata1/zrl/dataset/RESOLVE")
    parser.add_argument("--output", default="data/resolve")
    parser.add_argument("--split-file", default="configs/research/resolve_session_split.json")
    parser.add_argument("--every", type=int, default=5)
    parser.add_argument("--no-db", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sessions", nargs="*")
    args = parser.parse_args()
    raw_root = Path(args.raw_root)
    out = ROOT / args.output
    frames = json.load(open(raw_root / "HSQJN_3.json", encoding="utf-8"))
    local = {p.name for p in raw_root.iterdir() if p.is_dir()}
    if args.sessions:
        local &= set(args.sessions)
    by_session = collections.defaultdict(list)
    missing = collections.Counter()
    for frame in frames:
        session, _, name = frame["info"].split("/")[-3:]
        if session not in local:
            missing["session_not_local"] += 1
            continue
        token = name[:-4]
        if not all((raw_root / session / d / name).exists() for d in RES_DIRS.values()):
            missing["resolution_file_missing"] += 1
            continue
        by_session[session].append((int(frame["frameIndex"]), token, frame))
    if args.limit:
        by_session = {s: by_session[s] for s in sorted(by_session)[: args.limit]}
    train_s, val_s = session_split(by_session, args.every)
    split = {
        "rule": "本地会话按时间排序，每 %d 个会话取 1 个作为 val，同一会话的所有传感器与分辨率在同一划分" % args.every,
        "train": train_s,
        "val": val_s,
        "frames": {s: len(v) for s, v in by_session.items()},
        "dates": {s: s[8:18] for s in by_session},
        "skipped": dict(missing),
    }
    (ROOT / args.split_file).write_text(json.dumps(split, indent=1))
    db_infos = {res: collections.defaultdict(list) for res in RES_DIRS}
    check = {"pointsCount_vs_128": [], "pointsCount_vs_128_yaw90": []}
    stats = {res: {"points": [], "gt_with_points": 0} for res in RES_DIRS}
    infos = {"train": [], "val": []}
    sample_idx = 0
    for session in sorted(by_session):
        part = "train" if session in train_s else "val"
        for frame_index, token, frame in sorted(by_session[session], key=lambda x: x[0]):
            parsed = parse_frame(frame)
            if parsed is None:
                continue
            boxes, names, raw, ids, count = parsed
            num_points = {}
            for res, d in RES_DIRS.items():
                points = load_resolve_points(raw_root / session / d / (token + ".bin"))
                index = box_point_index(points, boxes)
                num_points[res] = np.bincount(index[index >= 0], minlength=len(boxes)).astype(np.int64)
                stats[res]["points"].append(len(points))
                if res == "128":
                    rot = boxes.copy()
                    rot[:, 6] += np.pi / 2
                    alt = box_point_index(points, rot)
                    alt = np.bincount(alt[alt >= 0], minlength=len(boxes))
                    valid = count > 0
                    check["pointsCount_vs_128"].extend(zip(count[valid].tolist(), num_points[res][valid].tolist()))
                    check["pointsCount_vs_128_yaw90"].extend(zip(count[valid].tolist(), alt[valid].tolist()))
                if part == "train" and not args.no_db:
                    db_dir = out / ("gt_database_%s" % res)
                    db_dir.mkdir(parents=True, exist_ok=True)
                    for k in np.where(num_points[res] > 0)[0]:
                        obj = points[index == k].copy()
                        obj[:, :3] -= boxes[k, :3]
                        rel = "gt_database_%s/%d_%s_%d.bin" % (res, sample_idx, names[k], k)
                        obj.astype(np.float32).tofile(str(out / rel))
                        db_infos[res][names[k]].append({"name": names[k], "path": rel, "image_idx": sample_idx, "gt_idx": int(k), "box3d_lidar": boxes[k], "num_points_in_gt": int(len(obj)), "difficulty": 0})
                stats[res]["gt_with_points"] += int((num_points[res] > 0).sum())
            infos[part].append({
                "frame_id": "%s/%s" % (session, token),
                "session": session,
                "token": token,
                "timestamp": int(token.split("-")[0]) + int(token.split("-")[1]) / 1e9,
                "frame_index": frame_index,
                "gt_boxes": boxes,
                "gt_names": names,
                "gt_names_raw": raw,
                "gt_ids": ids,
                "num_points": num_points,
            })
            sample_idx += 1
    (out / "infos").mkdir(parents=True, exist_ok=True)
    for part, rows in infos.items():
        with open(out / "infos" / ("resolve_infos_%s.pkl" % part), "wb") as stream:
            pickle.dump(rows, stream)
    if not args.no_db:
        for res, d in db_infos.items():
            with open(out / ("resolve_dbinfos_train_%s.pkl" % res), "wb") as stream:
                pickle.dump(dict(d), stream)
    a = np.array(check["pointsCount_vs_128"], dtype=np.float64)
    b = np.array(check["pointsCount_vs_128_yaw90"], dtype=np.float64)
    report = {
        "sessions": {"train": len(train_s), "val": len(val_s)},
        "frames": {k: len(v) for k, v in infos.items()},
        "skipped": dict(missing),
        "points_per_frame": {res: float(np.mean(s["points"])) for res, s in stats.items()},
        "gt_with_points": {res: s["gt_with_points"] for res, s in stats.items()},
        "gt_total": int(sum(len(i["gt_boxes"]) for v in infos.values() for i in v)),
        "pointsCount_check": {
            "pairs": int(len(a)),
            "exact_match_ratio": float(np.mean(a[:, 0] == a[:, 1])) if len(a) else None,
            "median_abs_diff": float(np.median(np.abs(a[:, 0] - a[:, 1]))) if len(a) else None,
            "median_abs_diff_yaw90": float(np.median(np.abs(b[:, 0] - b[:, 1]))) if len(b) else None,
            "corr": float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]) if len(a) > 2 else None,
            "corr_yaw90": float(np.corrcoef(b[:, 0], b[:, 1])[0, 1]) if len(b) > 2 else None,
        },
        "class_counts": dict(collections.Counter(n for v in infos.values() for i in v for n in i["gt_names"])),
    }
    (out / "conversion_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
