import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from inframot3d.detection.v2x_seq_converter import CLASS_NAMES

MAJOR_CLASSES = ["Car", "Van", "Bus", "Truck", "Pedestrian", "Cyclist", "Motorcyclist"]


def _inside_mask(values, point_cloud_range):
    low = np.asarray(point_cloud_range[:3], dtype=np.float32)
    high = np.asarray(point_cloud_range[3:], dtype=np.float32)
    return np.all((values >= low) & (values <= high), axis=1)


_GENERATOR = None
_RANGE = None


def _init_worker(voxel_size, point_cloud_range, max_points, max_voxels):
    global _GENERATOR, _RANGE
    from spconv.utils import Point2VoxelCPU3d

    _RANGE = [float(value) for value in point_cloud_range]
    _GENERATOR = Point2VoxelCPU3d(
        vsize_xyz=[float(value) for value in voxel_size],
        coors_range_xyz=_RANGE,
        num_point_features=4,
        max_num_points_per_voxel=int(max_points),
        max_num_voxels=int(max_voxels),
    )


def _voxel_count(points):
    import cumm.tensorview as tv

    _, coordinates, _ = _GENERATOR.point_to_voxel(tv.from_numpy(np.ascontiguousarray(points)))
    return int(coordinates.shape[0])


def _scan_one(task):
    path, count_voxels = task
    points = np.load(path)
    intensity = points[:, 3] if len(points) else np.zeros((0,), dtype=np.float32)
    inside = int(_inside_mask(points[:, :3], _RANGE).sum()) if len(points) else 0
    voxels = _voxel_count(points) if count_voxels else None
    return {
        "points": int(len(points)),
        "inside": inside,
        "intensity_min": float(intensity.min()) if intensity.size else None,
        "intensity_max": float(intensity.max()) if intensity.size else None,
        "intensity_sum": float(intensity.sum()) if intensity.size else 0.0,
        "intensity_count": int(intensity.size),
        "voxels": voxels,
    }


def gt_retention(infos, point_cloud_range):
    total = Counter()
    kept = Counter()
    for info in infos:
        boxes = info["annos"]["gt_boxes_lidar"]
        names = info["annos"]["name"]
        if len(boxes) == 0:
            continue
        mask = _inside_mask(boxes[:, :3], point_cloud_range)
        for name, flag in zip(names, mask):
            total[str(name)] += 1
            kept[str(name)] += int(flag)
    rows = []
    for name in CLASS_NAMES:
        count = int(total[name])
        keep = int(kept[name])
        ratio = keep / count if count else 1.0
        rows.append({"class_name": name, "kept": keep, "total": count, "ratio": ratio})
    center_total = int(sum(total.values()))
    center_kept = int(sum(kept.values()))
    return rows, center_kept, center_total


def _print_retention(split_name, rows, center_kept, center_total):
    ratio = center_kept / center_total if center_total else 1.0
    print("%s GT中心范围内 %d / %d %.2f%%" % (split_name, center_kept, center_total, ratio * 100.0))
    for row in rows:
        print("%s %s %d / %d %.2f%%" % (split_name, row["class_name"], row["kept"], row["total"], row["ratio"] * 100.0))
        if row["class_name"] in MAJOR_CLASSES and row["total"] > 0 and row["ratio"] < 0.99:
            print("warning %s %s GT保留率 %.2f%% 低于99%%" % (split_name, row["class_name"], row["ratio"] * 100.0))


def summarize_dataset(output_root, point_cloud_range, voxel_size, max_points, max_voxels, workers=8):
    import multiprocessing as mp

    output_root = Path(output_root)
    tasks = []
    split_infos = {}
    for split_name in ("train", "val"):
        with open(output_root / "infos" / ("v2x_seq_infos_%s.pkl" % split_name), "rb") as stream:
            infos = pickle.load(stream)
        split_infos[split_name] = infos
        for info in infos:
            tasks.append(
                (
                    str(output_root / "points" / ("%s.npy" % info["point_cloud"]["lidar_idx"])),
                    split_name == "train",
                )
            )
    context = mp.get_context("spawn")
    scanned = []
    with context.Pool(
        workers,
        initializer=_init_worker,
        initargs=(voxel_size, point_cloud_range, max_points, max_voxels),
    ) as pool:
        for index, item in enumerate(pool.imap(_scan_one, tasks, chunksize=8)):
            scanned.append(item)
            if index % 500 == 0:
                print("已统计%d/%d" % (index, len(tasks)))
    cursor = 0
    intensity_min = np.inf
    intensity_max = -np.inf
    intensity_sum = 0.0
    intensity_count = 0
    report = {}
    for split_name, infos in split_infos.items():
        part = scanned[cursor:cursor + len(infos)]
        cursor += len(infos)
        points = sum(item["points"] for item in part)
        inside = sum(item["inside"] for item in part)
        for item in part:
            if item["intensity_min"] is None:
                continue
            intensity_min = min(intensity_min, item["intensity_min"])
            intensity_max = max(intensity_max, item["intensity_max"])
            intensity_sum += item["intensity_sum"]
            intensity_count += item["intensity_count"]
        rows, center_kept, center_total = gt_retention(infos, point_cloud_range)
        point_ratio = inside / points if points else 1.0
        print("%s点云范围内比例 %.2f%%" % (split_name, point_ratio * 100.0))
        _print_retention(split_name, rows, center_kept, center_total)
        report[split_name] = {
            "point_ratio": point_ratio,
            "center_kept": center_kept,
            "center_total": center_total,
            "classes": rows,
        }
    if intensity_count:
        print("intensity min %.6f" % intensity_min)
        print("intensity max %.6f" % intensity_max)
        print("intensity mean %.6f" % (intensity_sum / intensity_count))
    train_voxels = np.array([item["voxels"] for item in scanned[:len(split_infos["train"])]], dtype=np.float64)
    if train_voxels.size:
        capped = int(np.sum(train_voxels >= int(max_voxels)))
        print(
            "voxel mean %.1f p95 %.1f p99 %.1f max %.0f" % (
                float(train_voxels.mean()),
                float(np.percentile(train_voxels, 95)),
                float(np.percentile(train_voxels, 99)),
                float(train_voxels.max()),
            )
        )
        print("voxel达到%d上限 %d / %d %.2f%%" % (int(max_voxels), capped, len(train_voxels), capped / len(train_voxels) * 100.0))
        if capped / len(train_voxels) >= 0.01:
            print("warning 大量帧达到voxel上限，后续可考虑提高到80000或100000")
        report["voxels"] = {
            "mean": float(train_voxels.mean()),
            "p95": float(np.percentile(train_voxels, 95)),
            "p99": float(np.percentile(train_voxels, 99)),
            "max": float(train_voxels.max()),
            "capped": capped,
            "frames": int(len(train_voxels)),
            "max_number_of_voxels": int(max_voxels),
        }
    report["intensity"] = {
        "min": None if not intensity_count else float(intensity_min),
        "max": None if not intensity_count else float(intensity_max),
        "mean": None if not intensity_count else float(intensity_sum / intensity_count),
    }
    return report
