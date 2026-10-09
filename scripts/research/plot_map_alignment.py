"""把一个路口的点云、检测框和车道画在同一虚拟激光雷达坐标系"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from inframot3d.io import read_json, read_jsonl
from inframot3d.perception.lane_context import LaneContext, _parse_xy

ROOT = Path("/home/kemove/devdata1/zrl/InfraMOT3D")
VEHICLES = {"Car", "Van", "Bus", "Truck"}


def main():
    out = ROOT / "outputs/research/roadside_joint_perception/figures"
    out.mkdir(parents=True, exist_ok=True)
    sequence_id = "0000"
    frames = list(read_jsonl(ROOT / "data/converted/v2x_seq_infrastructure/sequences" / ("%s.jsonl" % sequence_id)))
    frame = frames[0]
    info = next(
        item
        for item in read_json(ROOT / "data/v2x-seq-infrastructure/data_info.json")
        if str(item["sequence_id"]) == sequence_id and str(item["frame_id"]) == str(frame["frame_id"])
    )
    pose = read_json(ROOT / "data/v2x-seq-infrastructure" / info["calib_virtuallidar_to_world_path"])
    rotation = np.asarray(pose["rotation"], dtype=np.float64)
    translation = np.asarray(pose["translation"], dtype=np.float64).reshape(3)
    lanes = json.loads((ROOT / "data/v2x_seq_maps" / ("%s.json" % info["intersection_loc"])).read_text())
    polylines = []
    for lane in lanes["LANE"].values():
        world = _parse_xy(lane.get("centerline", []))
        if len(world) < 2:
            continue
        local = (rotation[:2, :2].T @ (world - translation[:2]).T).T
        if np.hypot(local[:, 0], local[:, 1]).min() > 80:
            continue
        polylines.append(local)
    points = np.load(ROOT / "data/centerpoint_v2xseq/points" / ("%s_%s.npy" % (sequence_id, frame["frame_id"])))
    keep = (points[:, 0] < 80) & (np.abs(points[:, 1]) < 40)
    cloud = points[keep][::8]
    context = LaneContext(ROOT / "data/v2x_seq_maps")
    centers = np.array([[item["box"][0], item["box"][1]] for item in frame["objects"] if item["class_name"] in VEHICLES])
    _, _, dist = context.tangent_lidar(info["intersection_loc"], rotation, translation, centers)
    dets = list(read_jsonl(ROOT / "outputs/centerpoint/detections/0000.jsonl"))[0]
    det_xy = [item["box"][:2] for item in dets["objects"] if item["class_name"] in VEHICLES and item["score"] >= 0.1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(cloud[:, 0], cloud[:, 1], s=0.2, c="0.75", linewidths=0)
    for line in polylines:
        ax.plot(line[:, 0], line[:, 1], color="tab:green", linewidth=0.6, alpha=0.8)
    ax.scatter(centers[:, 0], centers[:, 1], c="tab:blue", s=18, label="GT")
    if det_xy:
        det_xy = np.asarray(det_xy)
        ax.scatter(det_xy[:, 0], det_xy[:, 1], c="tab:orange", s=12, marker="x", label="CenterPoint")
    # 连续 10 帧同一辆车
    tracks = {}
    for item in frames[:10]:
        for obj in item["objects"]:
            if obj["class_name"] in VEHICLES:
                tracks.setdefault(obj["source_track_id"], []).append(obj["box"][:2])
    longest = max(tracks.values(), key=len)
    longest = np.asarray(longest)
    ax.plot(longest[:, 0], longest[:, 1], color="tab:red", linewidth=1.5, label="track")
    ax.set_xlim(0, 80)
    ax.set_ylim(-40, 40)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("%s frame %s median lane distance %.2f m" % (info["intersection_loc"], frame["frame_id"], float(np.median(dist))))
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "map_alignment_0000.png", dpi=140)
    print("median", float(np.median(dist)), "p90", float(np.percentile(dist, 90)), "lanes", len(polylines))


if __name__ == "__main__":
    main()
