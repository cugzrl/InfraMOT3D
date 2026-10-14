"""按折构建运动 KV 库和空间可靠性库"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inframot3d.analysis.scene_difficulty import _match
from inframot3d.evaluation.protocols.v2xseq import V2XSeqProtocol
from inframot3d.io import read_jsonl

from common import (
    LaneIndex,
    assert_bank_isolation,
    build_catalog,
    cell_index,
    forbidden_sequences,
    iter_motion_anchors,
    load_cached_tracks,
    load_config,
    load_hard_regions,
    output_dir,
    project_path,
    save_bank,
    scene_of,
    stack_bank,
    write_csv,
    write_json,
)


def _keep(protocol, item):
    box = [float(value) for value in item["box"]]
    if not np.isfinite(box).all() or min(box[4], box[5], box[6]) <= 0.0:
        return False
    return protocol.inside_range(box)


def _wilson(misses, total):
    if total <= 0:
        return None, None
    z = 1.96
    proportion = misses / total
    denom = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denom
    half = z * np.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denom
    return float(center), float(half)


def build_motion(config, tracks, catalog, lanes, out):
    vehicles = set(config["vehicle_classes"])
    grouped = defaultdict(list)
    used_sequences = defaultdict(set)
    for track in tracks:
        if track["role"] != "inner_train" or track["class_name"] not in vehicles or track["fold"] not in {"A", "B"}:
            continue
        added = False
        for position, sample, future, _ in iter_motion_anchors(track, config, lanes, True):
            grouped[(track["fold"], track["intersection"])].append(
                {
                    "origin": sample["origin"],
                    "heading": sample["heading"],
                    "hist": sample["hist"],
                    "future": future.astype(np.float64),
                    "speed": sample["speed"],
                    "yaw_rate": sample["yaw_rate"],
                    "region": sample["region"],
                    "time": float(track["time"][position]),
                    "class_name": track["class_name"],
                    "sequence_id": track["sequence_id"],
                    "uid": track["uid"],
                }
            )
            added = True
        if added:
            used_sequences[track["fold"]].add(track["sequence_id"])
    motion_dir = out / "kb" / "motion"
    motion_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for (fold_name, intersection), records in sorted(grouped.items()):
        bank = stack_bank(records)
        path = motion_dir / ("fold%s_%s.npz" % (fold_name, intersection))
        save_bank(path, bank)
        for holdout, meta in catalog.items():
            if meta["fold"] == fold_name and meta["role"] == "inner_holdout" and meta["intersection"] == intersection:
                assert_bank_isolation(list(bank["sequence_names"]), holdout, catalog)
        uids = len(set(item["uid"] for item in records))
        summary_rows.append(
            {
                "fold": fold_name,
                "intersection": intersection,
                "samples": len(records),
                "tracks": uids,
                "sequences": len(set(item["sequence_id"] for item in records)),
                "classes": str(dict(Counter(item["class_name"] for item in records))),
                "regions": str(dict(Counter(item["region"] for item in records))),
            }
        )
        print("运动库 %s %s 样本 %d 轨迹 %d" % (fold_name, intersection, len(records), uids), flush=True)
    for fold_name, sequence_ids in used_sequences.items():
        for sequence_id in sequence_ids:
            meta = catalog[sequence_id]
            if meta["split"] != "train" or meta["role"] != "inner_train":
                raise AssertionError("运动库混入非训练序列 %s" % sequence_id)
    write_csv(out / "kb" / "motion_summary.csv", summary_rows)
    return grouped


def _plot_coverage(path, lanes, intersection, records):
    if not records:
        return
    points = np.stack([item["origin"] for item in records])
    bounds = [float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 0].max()), float(points[:, 1].max())]
    road = lanes.centerlines_in_box(intersection, bounds)
    figure, axis = plt.subplots(figsize=(8.0, 8.0))
    if len(road):
        axis.scatter(road[:, 0], road[:, 1], s=1, c="#d0d0d0", linewidths=0)
    axis.scatter(points[::5, 0], points[::5, 1], s=3, c="#1f77b4", linewidths=0, alpha=0.5)
    axis.set_aspect("equal")
    axis.set_title("%s motion anchors %d" % (intersection, len(records)))
    axis.set_xlabel("world x m")
    axis.set_ylabel("world y m")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def build_reliability(config, catalog, out):
    protocol = V2XSeqProtocol(ROOT)
    regions = load_hard_regions(config)
    converted = project_path(config, config["project"]["converted_root"])
    score_limit = float(config["operating_score"])
    min_count = int(config["reliability"]["min_count"])
    buckets = defaultdict(lambda: {"gt": 0, "miss": 0, "errors": []})
    excluded = []
    used = []
    for fold_name in ("A", "B"):
        detection_root = project_path(config, config["fold_detection"][fold_name])
        holdouts = [catalog[item] for item in config_holdouts(catalog, fold_name)]
        for sequence_id, meta in sorted(catalog.items()):
            if meta["fold"] != fold_name or meta["role"] != "inner_train":
                continue
            blocked = False
            for holdout in holdouts:
                if holdout["intersection"] == meta["intersection"] and forbidden_sequences(catalog, holdout["sequence_id"]) & {sequence_id}:
                    blocked = True
            if blocked:
                excluded.append(sequence_id)
                continue
            det_path = detection_root / ("%s.jsonl" % sequence_id)
            if not det_path.is_file():
                raise FileNotFoundError("缺少折外检测 %s" % det_path)
            gt_rows = list(read_jsonl(converted / "sequences" / ("%s.jsonl" % sequence_id)))
            det_rows = {str(row["frame_id"]): row for row in read_jsonl(det_path)}
            scene = scene_of(regions, meta["intersection"])
            for gt_row in gt_rows:
                det_row = det_rows.get(str(gt_row["frame_id"]))
                if det_row is None:
                    raise ValueError("检测缺少帧 %s %s" % (sequence_id, gt_row["frame_id"]))
                gt_by_class = defaultdict(list)
                for item in gt_row["objects"]:
                    if _keep(protocol, item):
                        gt_by_class[item["class_name"]].append(item)
                det_by_class = defaultdict(list)
                for item in det_row["objects"]:
                    if float(item["score"]) < score_limit or not _keep(protocol, item):
                        continue
                    det_by_class[item["class_name"]].append(item)
                for class_name, gt_items in gt_by_class.items():
                    det_items = det_by_class.get(class_name, [])
                    matches = _match(
                        [item["box"] for item in gt_items],
                        [item["box"] for item in det_items],
                        config["match_iou"],
                        config["match_center_gate_m"],
                    )
                    matched = {gt_index: pred_index for gt_index, pred_index, _ in matches}
                    for gt_index, gt_item in enumerate(gt_items):
                        index, _ = cell_index(gt_item["box"][0], gt_item["box"][1], scene["bev"], scene["grid_m"])
                        if index is None:
                            continue
                        key = (fold_name, meta["intersection"], class_name, index[0], index[1])
                        buckets[key]["gt"] += 1
                        if gt_index not in matched:
                            buckets[key]["miss"] += 1
                        else:
                            det_box = det_items[matched[gt_index]]["box"]
                            error = float(np.hypot(det_box[0] - gt_item["box"][0], det_box[1] - gt_item["box"][1]))
                            buckets[key]["errors"].append(error)
            used.append(sequence_id)
            print("可靠性 %s %s" % (fold_name, sequence_id), flush=True)
    rows = []
    for key, value in sorted(buckets.items(), key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3], item[0][4])):
        fold_name, intersection, class_name, iy, ix = key
        total = int(value["gt"])
        misses = int(value["miss"])
        errors = np.asarray(value["errors"], dtype=np.float64)
        center, half = _wilson(misses, total)
        confident = total >= min_count
        rows.append(
            {
                "fold": fold_name,
                "intersection": intersection,
                "class_name": class_name,
                "iy": int(iy),
                "ix": int(ix),
                "n_gt": total,
                "n_miss": misses,
                "n_match": int(errors.size),
                "miss_rate": "" if not confident or center is None else round(misses / total, 4),
                "wilson_center": "" if not confident or center is None else round(center, 4),
                "wilson_half": "" if not confident or half is None else round(half, 4),
                "loc_median_m": "" if errors.size == 0 or not confident else round(float(np.median(errors)), 4),
                "loc_mean_m": "" if errors.size == 0 or not confident else round(float(np.mean(errors)), 4),
                "confident": int(confident),
            }
        )
    write_csv(out / "kb" / "reliability.csv", rows)
    write_json(
        out / "kb" / "reliability_meta.json",
        {
            "operating_score": score_limit,
            "min_count": min_count,
            "detector": config["fold_detection"],
            "used_sequences": used,
            "excluded_overlap_sequences": excluded,
            "note": "A 折 inner_train 使用在 B 上训练的检测器，B 折相反。样本不足的格子不给出漏检率",
        },
    )
    _plot_reliability(out / "kb" / "reliability_maps", rows, regions)
    return rows


def config_holdouts(catalog, fold_name):
    return [sequence_id for sequence_id, meta in catalog.items() if meta["fold"] == fold_name and meta["role"] == "inner_holdout"]


def _plot_reliability(folder, rows, regions):
    folder.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for row in rows:
        if row["class_name"] != "Car":
            continue
        grouped[(row["fold"], row["intersection"])].append(row)
    for (fold_name, intersection), items in grouped.items():
        scene = scene_of(regions, intersection)
        grid = np.full(scene["hard"].shape, np.nan, dtype=np.float64)
        for row in items:
            if int(row["confident"]) != 1 or row["miss_rate"] == "":
                continue
            grid[int(row["iy"]), int(row["ix"])] = float(row["miss_rate"])
        figure, axis = plt.subplots(figsize=(8.0, 4.5))
        image = axis.imshow(grid, origin="lower", vmin=0.0, vmax=1.0, cmap="magma")
        axis.set_title("%s fold %s Car miss rate" % (intersection, fold_name))
        figure.colorbar(image, ax=axis, fraction=0.046)
        figure.tight_layout()
        figure.savefig(folder / ("fold%s_%s_car_miss.png" % (fold_name, intersection)), dpi=120)
        plt.close(figure)


def main():
    config = load_config()
    out = output_dir(config)
    catalog, _, _ = build_catalog(config)
    tracks = load_cached_tracks(out / "cache" / "tracks.pkl")
    lanes = LaneIndex(project_path(config, config["project"]["map_root"]), out / "cache")
    grouped = build_motion(config, tracks, catalog, lanes, out)
    plot_dir = out / "kb" / "coverage"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for (fold_name, intersection), records in grouped.items():
        _plot_coverage(plot_dir / ("fold%s_%s.png" % (fold_name, intersection)), lanes, intersection, records)
    build_reliability(config, catalog, out)
    print("知识库完成", flush=True)


if __name__ == "__main__":
    main()
