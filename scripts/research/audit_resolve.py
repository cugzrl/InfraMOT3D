"""清点本地 RESOLVE 解压序列、分辨率和标注覆盖"""

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/home/kemove/devdata1/zrl/dataset/RESOLVE")
OUT = Path("/home/kemove/devdata1/zrl/InfraMOT3D/outputs/research/roadside_joint_perception")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sessions = []
    for path in sorted(ROOT.iterdir()):
        if not path.is_dir():
            continue
        sensors = {}
        stamps = {}
        for child in sorted(path.iterdir()):
            if not child.is_dir():
                continue
            files = [item for item in child.iterdir() if item.is_file()]
            sensors[child.name] = {
                "files": len(files),
                "bytes": sum(item.stat().st_size for item in files),
            }
            if files and child.name.endswith("_sync") and "axis" not in child.name:
                stamps[child.name] = {item.name for item in files}
        shared = None
        if stamps:
            names = list(stamps)
            shared = len(set.intersection(*stamps.values())) if len(names) > 1 else len(next(iter(stamps.values())))
        sessions.append({"name": path.name, "sensors": sensors, "shared_lidar_names": shared})

    payload = json.loads((ROOT / "HSQJN_3.json").read_text())
    by_seq = Counter()
    classes = Counter()
    frames = defaultdict(set)
    tracks = defaultdict(set)
    point_counts = []
    for item in payload:
        info = item.get("info", "")
        parts = info.split("/")
        seq = next((part for part in parts if part.startswith("rosbag2_")), "unknown")
        by_seq[seq] += 1
        frames[seq].add(item.get("frameIndex"))
        for label in item.get("labels", []):
            classes[label.get("label")] += 1
            tracks[seq].add(label.get("id"))
            if label.get("pointsCount") is not None:
                point_counts.append(int(label["pointsCount"]))
    local_names = {item["name"] for item in sessions}
    annotation = {
        "file": "HSQJN_3.json",
        "frames": len(payload),
        "sequences": len(by_seq),
        "classes": classes.most_common(),
        "per_sequence_frames": {key: by_seq[key] for key in by_seq},
        "local_overlap": sorted(name for name in by_seq if name in local_names),
        "missing_local": sorted(name for name in by_seq if name not in local_names),
        "points_per_box_p50": float(np_percentile(point_counts, 50)),
        "id_is_per_frame": True,
    }
    # 同一 id 是否跨帧
    id_frames = defaultdict(set)
    seq0 = next(iter(by_seq))
    for item in payload:
        if seq0 not in item.get("info", ""):
            continue
        for label in item.get("labels", []):
            id_frames[label.get("id")].add(item.get("frameIndex"))
    multi = sum(len(value) > 1 for value in id_frames.values())
    annotation["example_sequence"] = seq0
    annotation["ids"] = len(id_frames)
    annotation["ids_seen_in_multiple_frames"] = multi
    report = {"sessions": sessions, "annotation": annotation}
    (OUT / "resolve_stats.json").write_text(json.dumps(report))
    print("sessions", len(sessions), "annotated", annotation["sequences"], "local overlap", annotation["local_overlap"])
    print("classes", annotation["classes"][:12])
    print("example", seq0, "ids", len(id_frames), "multi", multi, "frames", by_seq[seq0])
    full = [item for item in sessions if item["sensors"].get("ouster128_sync", {}).get("files", 0) > 50]
    print("extracted with ouster128", [(item["name"], item["shared_lidar_names"]) for item in full])


def np_percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * q / 100))
    return ordered[index]


if __name__ == "__main__":
    main()
