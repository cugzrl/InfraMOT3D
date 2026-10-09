"""RESOLVE 单会话读取、时间对齐、框坐标与跨分辨率点云对照

不使用 V2X-Seq 检测权重，原始数据只读
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party" / "OpenPCDet"))

from pcdet.datasets.resolve.resolve_dataset import RES_DIRS, ROT_DEG, load_resolve_points, rotation_matrix

RAW = Path("/home/kemove/devdata1/zrl/dataset/RESOLVE")
RENAME = {"van": "barrier", "construction vehicle": "construction_vehicle", "golf cart": "traffic_cone"}
VEH = {"car", "truck", "bus", "trailer", "construction_vehicle"}


def parse_boxes(frame):
    labels = [l for l in frame["labels"] if l["drawType"] == "box3d"]
    if not labels:
        return np.zeros((0, 7), np.float32), np.array([], dtype=object), np.zeros(0, np.int64)
    pts = np.array([l["points"] for l in labels], dtype=np.float64)
    rz = rotation_matrix()
    locs = pts[:, :3] @ rz.T
    yaw = (pts[:, 5] + np.deg2rad(ROT_DEG) + np.pi) % (2 * np.pi) - np.pi
    boxes = np.concatenate([locs, pts[:, 6:9], yaw[:, None]], axis=1).astype(np.float32)
    names = np.array([RENAME.get(l["label"], l["label"]) for l in labels], dtype=object)
    ids = np.array([int(l.get("id", -1)) for l in labels], np.int64)
    return boxes, names, ids


def points_in_box(points, box):
    dx, dy = points[:, 0] - box[0], points[:, 1] - box[1]
    c, s = np.cos(-box[6]), np.sin(-box[6])
    u, v = c * dx - s * dy, s * dx + c * dy
    return (np.abs(u) <= box[3] / 2) & (np.abs(v) <= box[4] / 2) & (np.abs(points[:, 2] - box[2]) <= box[5] / 2)


def draw_box(ax, box, color):
    c, s = np.cos(box[6]), np.sin(box[6])
    corners = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1], [1, 1]], dtype=np.float64) * np.array([box[3], box[4]]) / 2
    xy = corners @ np.array([[c, s], [-s, c]]) + box[:2]
    ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=0.8)


def main():
    session = sys.argv[1] if len(sys.argv) > 1 else "rosbag2_2025_10_17-09_29_02"
    out = ROOT / "outputs/research/roadside_joint_perception_v2/figures/resolve"
    out.mkdir(parents=True, exist_ok=True)
    frames = json.loads((RAW / "HSQJN_3.json").read_text(encoding="utf-8"))
    rows = []
    for frame in frames:
        sess, _, name = frame["info"].split("/")[-3:]
        if sess != session:
            continue
        token = name[:-4]
        missing = [res for res, d in RES_DIRS.items() if not (RAW / session / d / (token + ".bin")).exists()]
        rows.append({"token": token, "frame": frame, "missing": missing})
    aligned = [r for r in rows if not r["missing"]]
    report = {
        "session": session,
        "labeled_frames": len(rows),
        "aligned_128_64_16": len(aligned),
        "missing_any_res": len(rows) - len(aligned),
        "cameras": [p.name for p in (RAW / session).iterdir() if p.name.startswith("axis")],
        "calib": [p.name for p in (RAW / session / "calibration_txt").iterdir()] if (RAW / session / "calibration_txt").exists() else [],
    }
    if not aligned:
        raise SystemExit("会话没有三档分辨率对齐的帧 %s" % session)
    mid = aligned[len(aligned) // 2]
    boxes, names, ids = parse_boxes(mid["frame"])
    clouds = {res: load_resolve_points(RAW / session / d / (mid["token"] + ".bin")) for res, d in RES_DIRS.items()}
    stats = {}
    for res, pts in clouds.items():
        n_in = [int(points_in_box(pts, b).sum()) for b in boxes]
        stats[res] = {
            "points": int(len(pts)),
            "xy_median": [float(np.median(pts[:, 0])), float(np.median(pts[:, 1]))],
            "z_median": float(np.median(pts[:, 2])),
            "gt_with_points": int(sum(v > 0 for v in n_in)),
            "vehicle_points_median": float(np.median([n_in[i] for i, n in enumerate(names) if n in VEH] or [0])),
        }
    report["sample_token"] = mid["token"]
    name_counts = {}
    if len(names):
        uniq, cnt = np.unique(names, return_counts=True)
        name_counts = {str(k): int(v) for k, v in zip(uniq, cnt)}
    report["gt"] = {"n": int(len(boxes)), "names": name_counts, "unique_ids": int(len(set(ids.tolist())))}
    report["lidar"] = stats
    # 相邻帧 id 连续性
    ids_by_frame = []
    for r in aligned:
        _, _, fid = parse_boxes(r["frame"])
        ids_by_frame.append(set(fid.tolist()))
    persist = []
    for a, b in zip(ids_by_frame, ids_by_frame[1:]):
        persist.append(len(a & b) / max(len(a), 1))
    report["id_persist_mean"] = float(np.mean(persist)) if persist else None
    (out / ("%s_stats.json" % session)).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharex=True, sharey=True)
    for ax, res in zip(axes, ("128", "64", "16")):
        pts = clouds[res]
        ax.scatter(pts[:, 0], pts[:, 1], s=0.15, c=np.clip(pts[:, 2], -3, 1), cmap="viridis", linewidths=0)
        for box, name in zip(boxes, names):
            draw_box(ax, box, "red" if name in VEH else "orange")
        ax.set_title("%s-line n=%d" % (res, len(pts)))
        ax.set_aspect("equal")
        ax.set_xlim(-40, 80)
        ax.set_ylim(-40, 80)
    fig.suptitle("%s  %s" % (session, mid["token"]), fontsize=9)
    fig.tight_layout()
    fig.savefig(out / ("%s_bev.png" % session), dpi=140)
    plt.close(fig)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
