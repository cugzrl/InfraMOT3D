import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CLASS_NAMES = ["Car", "Van", "Bus", "Truck", "Pedestrian", "Cyclist", "Motorcyclist", "Barrowlist"]
INTENSITY_KEYS = ("intensity", "reflectivity", "i")


def load_sequence_split(path):
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {name: [str(item) for item in payload[name]] for name in ("train", "val", "test")}


def read_points(path):
    import open3d as o3d

    cloud = o3d.t.io.read_point_cloud(str(path))
    if "positions" not in cloud.point:
        raise KeyError("PCD缺少positions %s" % path)
    xyz = np.asarray(cloud.point["positions"].numpy(), dtype=np.float32)
    intensity = None
    for key in INTENSITY_KEYS:
        if key in cloud.point:
            intensity = np.asarray(cloud.point[key].numpy(), dtype=np.float32).reshape(-1, 1)
            break
    if intensity is None:
        intensity = np.zeros((xyz.shape[0], 1), dtype=np.float32)
    points = np.concatenate([xyz, intensity], axis=1)
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError("点云形状应为Nx4 %s" % path)
    if not np.isfinite(points).all():
        raise ValueError("点云含NaN或Inf %s" % path)
    return points


def label_to_box(value):
    dims = value["3d_dimensions"]
    loc = value["3d_location"]
    return np.array(
        [
            float(loc["x"]),
            float(loc["y"]),
            float(loc["z"]),
            float(dims["l"]),
            float(dims["w"]),
            float(dims["h"]),
            float(value["rotation"]),
        ],
        dtype=np.float32,
    )


def openpcdet_to_box(box):
    values = [float(item) for item in box[:7]]
    x, y, z, dx, dy, dz, heading = values
    return [x, y, z, heading, dx, dy, dz]


def _convert_one(task):
    source_root, output_root, frame, sequence_id, frame_index = task
    import json

    source_root = Path(source_root)
    output_root = Path(output_root)
    frame_id = str(frame["frame_id"])
    lidar_idx = "%s_%s" % (sequence_id, frame_id)
    point_path = output_root / "points" / ("%s.npy" % lidar_idx)
    if not point_path.exists():
        points = read_points(source_root / frame["pointcloud_path"])
        np.save(point_path, points)
    labels = json.loads((source_root / frame["label_lidar_std_path"]).read_text(encoding="utf-8"))
    names = []
    boxes = []
    for value in labels:
        class_name = value["type"]
        if class_name not in CLASS_NAMES:
            raise ValueError("未知类别%s" % class_name)
        names.append(class_name)
        boxes.append(label_to_box(value))
    info = {
        "sequence_id": sequence_id,
        "frame_index": int(frame_index),
        "frame_id": frame_id,
        "timestamp": int(frame["pointcloud_timestamp"]),
        "point_cloud": {"num_features": 4, "lidar_idx": lidar_idx},
        "annos": {
            "name": np.array(names),
            "gt_boxes_lidar": np.stack(boxes).astype(np.float32) if boxes else np.zeros((0, 7), dtype=np.float32),
        },
    }
    return info


def _center_balance(source_root, frames, limit=30):
    below = []
    above = []
    for frame in frames:
        if len(below) >= limit:
            break
        import json

        labels = json.loads((source_root / frame["label_lidar_std_path"]).read_text(encoding="utf-8"))
        if not labels:
            continue
        points = read_points(source_root / frame["pointcloud_path"])
        value = labels[0]
        box = label_to_box(value)
        yaw = float(box[6])
        local = points[:, :2] - box[:2]
        cosine, sine = np.cos(yaw), np.sin(yaw)
        rotated = np.stack(
            [local[:, 0] * cosine + local[:, 1] * sine, -local[:, 0] * sine + local[:, 1] * cosine],
            axis=1,
        )
        inside = (
            (np.abs(rotated[:, 0]) <= box[3] / 2)
            & (np.abs(rotated[:, 1]) <= box[4] / 2)
            & (np.abs(points[:, 2] - box[2]) <= box[5] / 2 + 0.3)
        )
        selected = points[inside, 2]
        if selected.size < 20:
            continue
        below.append(float(np.mean(selected < box[2])))
        above.append(float(np.mean(selected > box[2])))
    if not below:
        return None
    return {"below_mean": float(np.mean(below)), "above_mean": float(np.mean(above)), "num_boxes": len(below)}


def convert_dataset(source_root, output_root, split_file, workers=8):
    import json
    import multiprocessing as mp

    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    frames = json.loads((source_root / "data_info.json").read_text(encoding="utf-8"))
    split = load_sequence_split(split_file)
    grouped = defaultdict(list)
    for frame in frames:
        grouped[str(frame["sequence_id"])].append(frame)
    known = set(split["train"]) | set(split["val"]) | set(split["test"])
    missing = sorted(set(grouped) - known)
    extra = sorted(known - set(grouped))
    if missing or extra:
        raise ValueError("序列划分与数据不一致 missing=%s extra=%s" % (missing[:5], extra[:5]))
    for name in ("points", "infos", "ImageSets", "metadata"):
        (output_root / name).mkdir(parents=True, exist_ok=True)

    tasks = []
    sequence_split = {}
    for sequence_id, sequence_frames in grouped.items():
        sequence_frames = sorted(
            sequence_frames,
            key=lambda item: (int(item["pointcloud_timestamp"]), int(item["frame_id"])),
        )
        for name in ("train", "val", "test"):
            if sequence_id in split[name]:
                sequence_split[sequence_id] = name
        for frame_index, frame in enumerate(sequence_frames):
            tasks.append((str(source_root), str(output_root), frame, sequence_id, frame_index))

    sample_frames = [task[2] for task in tasks if sequence_split[task[3]] == "train"][:40]
    balance = _center_balance(source_root, sample_frames)
    context = mp.get_context("spawn")
    infos = []
    with context.Pool(workers) as pool:
        for index, info in enumerate(pool.imap(_convert_one, tasks, chunksize=16)):
            infos.append(info)
            if index % 500 == 0:
                print("已转换%d/%d" % (index, len(tasks)))
    infos.sort(key=lambda item: (item["sequence_id"], item["frame_index"]))
    by_split = {name: [] for name in ("train", "val", "test")}
    class_counts = Counter()
    for info in infos:
        by_split[sequence_split[info["sequence_id"]]].append(info)
        class_counts.update(info["annos"]["name"].tolist())
        if not np.isfinite(info["annos"]["gt_boxes_lidar"]).all():
            raise ValueError("标注含NaN或Inf %s" % info["frame_id"])
    counts = {}
    for name, rows in by_split.items():
        with open(output_root / "infos" / ("v2x_seq_infos_%s.pkl" % name), "wb") as stream:
            pickle.dump(rows, stream)
        ids = [item["point_cloud"]["lidar_idx"] for item in rows]
        (output_root / "ImageSets" / ("%s.txt" % name)).write_text(
            "\n".join(ids) + ("\n" if ids else ""),
            encoding="utf-8",
        )
        sequences = sorted({item["sequence_id"] for item in rows})
        counts[name] = {"sequences": len(sequences), "frames": len(rows)}
        print("%s序列%d 帧%d" % (name, counts[name]["sequences"], counts[name]["frames"]))
    metadata = {
        "coordinate_system": "virtual_lidar",
        "point_fields": ["x", "y", "z", "intensity"],
        "point_dtype": "float32",
        "box_order": ["x", "y", "z", "dx", "dy", "dz", "heading"],
        "box_mapping": {"dx": "length", "dy": "width", "dz": "height", "heading": "yaw"},
        "z_definition": "标注z与OpenPCDet框中心一致，不额外平移",
        "z_center_check": balance,
        "classes": CLASS_NAMES,
        "class_counts": dict(class_counts),
        "counts": counts,
        "split_file": str(Path(split_file).resolve()),
    }
    (output_root / "metadata" / "conversion.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return counts
