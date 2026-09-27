import copy
import math
import shutil
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from inframot3d.analysis.scene_difficulty import _bin_index, _grid_shape, _keep_vehicle, _match, build_scenes
from inframot3d.analysis.scene_survival import _frames
from inframot3d.analysis.tracking_bottleneck import _class_thresholds, _equivalent_rows, _historical_regions, _path
from inframot3d.config import load_config
from inframot3d.evaluation import UnifiedMOTEvaluator
from inframot3d.evaluation.hota import evaluate_hota
from inframot3d.evaluation.protocols.v2xseq import V2XSeqProtocol
from inframot3d.io import read_json, read_jsonl, write_json, write_jsonl
from inframot3d.tracking import create_tracker
from inframot3d.tracking.common import SpatialScoreGate


LABELS = {
    "baseline": "Baseline",
    "global": "Global-Low",
    "scene": "Scene-Aware",
    "reliability": "Reliability memory",
    "shuffle": "Shuffled",
}
COLORS = {
    "baseline": "#6B6B6B",
    "global": "#4C78A8",
    "scene": "#E45756",
    "reliability": "#F58518",
    "shuffle": "#8C8C8C",
}


def _key(name):
    return name.lower().replace("-", "_")


def _cell(config, box):
    shape = _grid_shape(config["bev"], float(config["grid_m"]))
    return _bin_index(float(box[0]), float(box[1]), config["bev"], float(config["grid_m"]), shape)


def _prediction_row(source, objects):
    return {
        "sequence_id": source["sequence_id"],
        "frame_index": source["frame_index"],
        "frame_id": source["frame_id"],
        "timestamp": source["timestamp"],
        "objects": objects,
    }


def _detection_job(payload):
    config, sequence_id, floor = payload
    protocol = V2XSeqProtocol(config["_root"])
    converted = Path(config["project"]["converted_root"])
    gt_rows = list(read_jsonl(converted / "sequences" / (sequence_id + ".jsonl")))
    det_rows = list(read_jsonl(_path(config["_root"], config["detection_root"]) / (sequence_id + ".jsonl")))
    if len(gt_rows) != len(det_rows):
        raise ValueError("检测帧数不一致%s" % sequence_id)
    shape = _grid_shape(config["bev"], float(config["grid_m"]))
    gt_counts = np.zeros(shape, dtype=np.float64)
    records = {key: [] for key in ("frame", "cell", "score", "class", "tp", "gt")}
    for gt_row, det_row in zip(gt_rows, det_rows):
        if int(gt_row["timestamp"]) != int(det_row["timestamp"]):
            raise ValueError("检测帧未对齐%s" % gt_row["frame_id"])
        gt_items = protocol.filter_gt(gt_row["objects"])
        for item in gt_items:
            index = _cell(config, item["box"])
            if index is not None:
                gt_counts[index] += 1
        detections = [
            item
            for item in det_row["objects"]
            if float(item.get("score", 1.0)) >= floor and _keep_vehicle(protocol, item)
        ]
        matches = _match(
            [item["box"] for item in gt_items],
            [item["box"] for item in detections],
            config["match_iou"],
            config["match_center_gate_m"],
        )
        by_detection = {det_index: gt_index for gt_index, det_index, _ in matches}
        for det_index, item in enumerate(detections):
            index = _cell(config, item["box"])
            gt_index = by_detection.get(det_index)
            records["frame"].append(int(gt_row["frame_index"]))
            records["cell"].append(None if index is None else [int(index[0]), int(index[1])])
            records["score"].append(float(item["score"]))
            records["class"].append(item["class_name"])
            records["tp"].append(gt_index is not None)
            records["gt"].append(None if gt_index is None else str(gt_items[gt_index]["source_track_id"]))
    return sequence_id, gt_counts.tolist(), records


def _detection_records(config, sequence_ids, debug_root):
    path = debug_root / "detections.json"
    floor = float(min(config["tau_low"]))
    if path.is_file():
        cached = read_json(path)
        if abs(float(cached["floor"]) - floor) < 1.0e-9:
            print("复用检测匹配缓存", flush=True)
            return cached["gt_counts"], cached["records"]
    gt_counts, records = {}, {}
    with ProcessPoolExecutor(max_workers=int(config["workers"])) as executor:
        futures = [executor.submit(_detection_job, (config, sequence_id, floor)) for sequence_id in sequence_ids]
        for future in as_completed(futures):
            sequence_id, counts, current = future.result()
            gt_counts[sequence_id] = counts
            records[sequence_id] = current
    write_json(path, {"floor": floor, "gt_counts": gt_counts, "records": records})
    return gt_counts, records


def _reliability_cells(config, history_ids, gt_counts, records, width):
    shape = _grid_shape(config["bev"], float(config["grid_m"]))
    gt = np.zeros(shape, dtype=np.float64)
    tp = np.zeros(shape, dtype=np.float64)
    fp = np.zeros(shape, dtype=np.float64)
    high = float(config["reliability_band_high"])
    for sequence_id in history_ids:
        gt += np.asarray(gt_counts[sequence_id], dtype=np.float64)
        current = records[sequence_id]
        for cell, score, positive in zip(current["cell"], current["score"], current["tp"]):
            if cell is None or float(score) >= high:
                continue
            target = tp if positive else fp
            target[cell[0], cell[1]] += 1
    valid = [tuple(int(value) for value in item) for item in np.argwhere(gt >= int(config["min_observations"]))]
    total = float(tp.sum() + fp.sum())
    prior = float(tp.sum()) / total if total > 0 else 0.0
    strength = float(config["prior_strength"])
    # 低分检测的平滑precision，只在历史低分检测多数为真的cell放宽阈值
    precision = (tp + strength * prior) / (tp + fp + strength)
    ranked = sorted(valid, key=lambda item: (float(precision[item]), float(tp[item])), reverse=True)
    return ranked[:width], valid


def _shuffled_cells(valid, width, scene_id, sequence_id, seed):
    if width <= 0 or not valid:
        return []
    token = ("%s:%s" % (scene_id, sequence_id)).encode("utf-8")
    rng = np.random.default_rng(int(seed) + int(zlib.crc32(token)))
    order = rng.permutation(len(valid))[:width]
    return [valid[int(index)] for index in order]


def _memories(config, scenes, protocol, gt_counts, records, debug_root):
    path = debug_root / "memory.json"
    if path.is_file():
        print("复用历史场景记忆", flush=True)
        return read_json(path)
    regions, _ = _historical_regions(config, scenes, config["memory_trackers"], protocol)
    memory = {}
    for scene in scenes:
        for index, target in enumerate(scene["sequences"]):
            if target["split"] != "val":
                continue
            current = regions.get((scene["scene_id"], target["sequence_id"]))
            if current is None:
                continue
            history_ids = [item["sequence_id"] for item in scene["sequences"][:index]]
            hard = sorted(current["hard"])
            reliability, valid = _reliability_cells(config, history_ids, gt_counts, records, len(hard))
            memory[target["sequence_id"]] = {
                "scene_id": scene["scene_id"],
                "history_sequences": history_ids,
                "valid_cells": len(valid),
                "scene": [list(item) for item in hard],
                "reliability": [list(item) for item in reliability],
                "shuffle": {
                    str(seed): [
                        list(item)
                        for item in _shuffled_cells(valid, len(hard), scene["scene_id"], target["sequence_id"], seed)
                    ]
                    for seed in config["shuffle_seeds"]
                },
            }
    write_json(path, memory)
    return memory


def _variant(name, kind, tau, seed=None, normal=None):
    return {"name": name, "kind": kind, "tau": tau, "seed": seed, "normal": normal}


def _global_taus(config):
    values = [float(value) for value in config["tau_low"]] + [float(value) for value in config.get("global_extra_tau", [])]
    return sorted(set(values))


def _stage_one(config):
    output = [_variant("baseline", "baseline", None)]
    for tau in config["tau_low"]:
        text = "%.2f" % float(tau)
        for kind in ("scene", "reliability"):
            output.append(_variant("%s_%s" % (kind, text), kind, float(tau)))
        for seed in config["shuffle_seeds"]:
            output.append(_variant("shuffle%d_%s" % (int(seed), text), "shuffle", float(tau), int(seed)))
    for tau in _global_taus(config):
        output.append(_variant("global_%.2f" % tau, "global", tau))
    return output


def _stage_two(config, normal):
    output = []
    for tau in config["tau_low"]:
        if float(tau) >= normal - 1.0e-9:
            continue
        text = "n%.2f_%.2f" % (normal, float(tau))
        for kind in ("scene", "reliability"):
            output.append(_variant("%s_%s" % (kind, text), kind, float(tau), normal=normal))
        for seed in config["shuffle_seeds"]:
            output.append(_variant("shuffle%d_%s" % (int(seed), text), "shuffle", float(tau), int(seed), normal))
    return output


def _low_cells(variant, memory_entry):
    if memory_entry is None:
        return []
    if variant["kind"] == "shuffle":
        return memory_entry["shuffle"][str(variant["seed"])]
    return memory_entry[variant["kind"]]


def _policy(config, variant, memory_entry):
    if variant["kind"] == "baseline":
        return None
    if variant["kind"] == "global":
        return {"enabled": True, "mode": "global", "low_threshold": variant["tau"]}
    cells = _low_cells(variant, memory_entry)
    return {
        "enabled": bool(cells),
        "mode": "cells",
        "low_threshold": variant["tau"],
        "grid_m": float(config["grid_m"]),
        "bev": dict(config["bev"]),
        "low_cells": cells,
    }


def _reference(setup, normal):
    tracker_config, out_threshold = setup
    thresholds = {str(key): float(value) for key, value in (tracker_config["tracker"].get("score_thresholds") or {}).items()}
    default = float(tracker_config["tracker"].get("score_threshold", 0.0))
    if normal is None:
        return thresholds, default, out_threshold
    # 两级阈值的普通区等价于Global-Low在normal处的设置
    return (
        {key: min(value, normal) for key, value in thresholds.items()},
        min(default, normal),
        min(out_threshold, normal),
    )


def _tracker_setup(config, spec):
    tracker_config = load_config(_path(config["_root"], spec["config"]))
    _class_thresholds(tracker_config)
    baseline = read_json(_path(config["_root"], spec["baseline_metrics"]))
    return tracker_config, float(baseline["best_score_threshold"])


def _track_job(payload):
    config, spec, setup, sequence_id, entry, variant, memory_entry, target = payload
    frames = _frames(config, entry)
    policy = _policy(config, variant, memory_entry)
    thresholds, default, out_threshold = _reference(setup, variant["normal"])
    settings = copy.deepcopy(setup[0]["tracker"])
    settings["score_thresholds"] = thresholds
    settings["score_threshold"] = default
    if policy is not None:
        settings["spatial_score_gate"] = policy
    tracker = create_tracker(settings, config["_root"])
    # 输出阈值与输入阈值用同一张空间阈值图，否则低分轨迹会在输出阈值处被再次删掉
    gate = SpatialScoreGate(policy)
    rows, raw = [], []
    for frame in frames:
        objects = [
            {"class_name": item["class_name"], "score": float(item.get("score", 1.0)), "box": item["box"]}
            for item in frame["objects"]
        ]
        outputs = tracker.update(objects, timestamp=frame["timestamp"])
        if variant["kind"] == "baseline":
            raw.append(_prediction_row(frame, outputs))
        kept = [item for item in outputs if float(item["score"]) >= gate.threshold(item["box"], out_threshold)]
        rows.append(_prediction_row(frame, kept))
    if variant["kind"] == "baseline":
        reference = list(read_jsonl(_path(config["_root"], spec["baseline_prediction_root"]) / (sequence_id + ".jsonl")))
        _equivalent_rows(raw, reference, spec["name"], sequence_id)
    write_jsonl(target, rows)
    return spec["name"], variant["name"], sequence_id


def _run_trackers(config, plan, setups, memory, sequence_ids, debug_root):
    manifest = read_json(Path(config["project"]["converted_root"]) / "manifest.json")
    entries = {entry["sequence_id"]: entry for entry in manifest["sequences"]}
    specs = {spec["name"]: spec for spec in config["trackers"]}
    jobs = []
    for name, variants in plan.items():
        for variant in variants:
            for sequence_id in sequence_ids:
                target = debug_root / "predictions" / _key(name) / variant["name"] / (sequence_id + ".jsonl")
                if target.is_file():
                    continue
                jobs.append((config, specs[name], setups[name], sequence_id, entries[sequence_id], variant, memory.get(sequence_id), target))
    print("待运行tracker任务%d" % len(jobs), flush=True)
    if not jobs:
        return
    done = 0
    with ProcessPoolExecutor(max_workers=int(config["workers"])) as executor:
        futures = [executor.submit(_track_job, job) for job in jobs]
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 200 == 0 or done == len(jobs):
                print("完成tracker任务%d/%d" % (done, len(jobs)), flush=True)


def _hota_job(payload):
    config, tracker_config, prediction_root, sequence_ids = payload
    protocol = V2XSeqProtocol(config["_root"])
    return evaluate_hota(tracker_config, prediction_root, sequence_ids, protocol, 0.0)


def _evaluate(config, plan, setups, sequence_ids, debug_root):
    evaluator = UnifiedMOTEvaluator(config["_root"], "v2xseq")
    results = {}
    pending = []
    for name, variants in plan.items():
        for variant in variants:
            path = debug_root / "evaluation" / _key(name) / (variant["name"] + ".json")
            if path.is_file():
                results[(name, variant["name"])] = read_json(path)
            else:
                pending.append((name, variant, path))
    print("待评估variant%d" % len(pending), flush=True)
    if not pending:
        return results
    with ProcessPoolExecutor(max_workers=min(8, int(config["workers"]))) as executor:
        hota = {}
        for name, variant, _ in pending:
            prediction_root = debug_root / "predictions" / _key(name) / variant["name"]
            hota[(name, variant["name"])] = executor.submit(_hota_job, (config, setups[name][0], prediction_root, sequence_ids))
        for name, variant, path in pending:
            prediction_root = debug_root / "predictions" / _key(name) / variant["name"]
            # 固定目录名复用官方staging目录，输出阈值已逐框写进预测，评估阈值设为0
            output_dir = debug_root / "evaluation" / "_official" / "scene_threshold"
            metrics = evaluator.evaluate(setups[name][0], prediction_root, output_dir, split="val", score_threshold=0.0)
            metrics.update(hota[(name, variant["name"])].result())
            write_json(path, metrics)
            results[(name, variant["name"])] = metrics
            print("完成评估%s %s HOTA %.4f FP %d FN %d" % (name, variant["name"], metrics["HOTA"], metrics["FP"], metrics["FN"]), flush=True)
    shutil.rmtree(debug_root / "evaluation" / "_official", ignore_errors=True)
    return results


def _subset_hota(config, plan, setups, memory_ids, debug_root):
    path = debug_root / "subset_hota.json"
    cached = read_json(path) if path.is_file() else {}
    pending = [
        (name, variant)
        for name, variants in plan.items()
        for variant in variants
        if "%s|%s" % (name, variant["name"]) not in cached
    ]
    if pending:
        with ProcessPoolExecutor(max_workers=int(config["workers"])) as executor:
            futures = {
                executor.submit(
                    _hota_job,
                    (config, setups[name][0], debug_root / "predictions" / _key(name) / variant["name"], memory_ids),
                ): (name, variant["name"])
                for name, variant in pending
            }
            for future in as_completed(futures):
                name, variant_name = futures[future]
                cached["%s|%s" % (name, variant_name)] = future.result()
        write_json(path, cached)
    return {tuple(key.split("|")): value for key, value in cached.items()}


def _is_low(variant, low_set, cell):
    if variant["kind"] == "baseline":
        return False
    if variant["kind"] == "global":
        return True
    return cell is not None and tuple(cell) in low_set


def _region_job(payload):
    config, spec_name, variant, prediction_root, sequence_ids, memory, records, reference = payload
    class_thresholds, default_threshold, out_threshold = reference
    protocol = V2XSeqProtocol(config["_root"])
    converted = Path(config["project"]["converted_root"])
    counts = Counter()
    for sequence_id in sequence_ids:
        memory_entry = memory.get(sequence_id)
        hard = {tuple(item) for item in memory_entry["scene"]} if memory_entry else None
        low_set = {tuple(item) for item in _low_cells(variant, memory_entry)} if variant["kind"] not in {"baseline", "global"} else set()

        def region(cell):
            if hard is None:
                return "none"
            return "hard" if cell is not None and tuple(cell) in hard else "normal"

        gt_rows = list(read_jsonl(converted / "sequences" / (sequence_id + ".jsonl")))
        predictions = {int(row["frame_index"]): row for row in read_jsonl(Path(prediction_root) / (sequence_id + ".jsonl"))}
        assigned = {}
        votes = defaultdict(Counter)
        for gt_row in gt_rows:
            frame_index = int(gt_row["frame_index"])
            gt_items = protocol.filter_gt(gt_row["objects"])
            outputs = [item for item in predictions[frame_index]["objects"] if _keep_vehicle(protocol, item)]
            matches = _match(
                [item["box"] for item in gt_items],
                [item["box"] for item in outputs],
                config["match_iou"],
                config["match_center_gate_m"],
            )
            matched_gt = {gt_index: out_index for gt_index, out_index, _ in matches}
            matched_out = set(matched_gt.values())
            for gt_index, item in enumerate(gt_items):
                name = region(_cell(config, item["box"]))
                counts["gt_" + name] += 1
                if gt_index not in matched_gt:
                    counts["fn_" + name] += 1
                    continue
                gid = str(item["source_track_id"])
                track_id = int(outputs[matched_gt[gt_index]]["track_id"])
                assigned[(frame_index, gid)] = track_id
                votes[gid][track_id] += 1
            for out_index, item in enumerate(outputs):
                if out_index not in matched_out:
                    counts["fp_" + region(_cell(config, item["box"]))] += 1
        dominant = {gid: current.most_common(1)[0][0] for gid, current in votes.items()}
        current = records[sequence_id]
        for frame_index, cell, score, class_name, positive, gid in zip(
            current["frame"], current["cell"], current["score"], current["class"], current["tp"], current["gt"]
        ):
            class_threshold = float(class_thresholds.get(class_name, default_threshold))
            reference_ok = score >= class_threshold and score >= out_threshold
            if _is_low(variant, low_set, cell):
                variant_ok = score >= min(class_threshold, variant["tau"]) and score >= min(out_threshold, variant["tau"])
            else:
                variant_ok = reference_ok
            if not variant_ok or reference_ok:
                continue
            name = region(cell)
            if not positive:
                counts["released_fp"] += 1
                counts["released_fp_" + name] += 1
                continue
            counts["released_tp"] += 1
            counts["released_tp_" + name] += 1
            track_id = assigned.get((int(frame_index), gid))
            if track_id is not None:
                counts["released_covered"] += 1
                counts["released_correct"] += int(track_id == dominant.get(gid))
    return spec_name, variant["name"], dict(counts)


def _region_stats(config, plan, setups, memory, records, sequence_ids, debug_root):
    path = debug_root / "region_stats.json"
    cached = read_json(path) if path.is_file() else {}
    records = {sequence_id: records[sequence_id] for sequence_id in sequence_ids}
    jobs = []
    for name, variants in plan.items():
        for variant in variants:
            if "%s|%s" % (name, variant["name"]) in cached:
                continue
            prediction_root = debug_root / "predictions" / _key(name) / variant["name"]
            jobs.append((config, name, variant, prediction_root, sequence_ids, memory, records, _reference(setups[name], variant["normal"])))
    if jobs:
        with ProcessPoolExecutor(max_workers=int(config["workers"])) as executor:
            futures = [executor.submit(_region_job, job) for job in jobs]
            for future in as_completed(futures):
                spec_name, variant_name, counts = future.result()
                cached["%s|%s" % (spec_name, variant_name)] = counts
        write_json(path, cached)
    return {tuple(key.split("|")): value for key, value in cached.items()}


def _rows(config, plan, results, regions, subset):
    numeric = ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "FP", "FN", "FM")
    rows = {}
    for name, variants in plan.items():
        current = {}
        for variant in variants:
            metrics = results[(name, variant["name"])]
            row = {key: float(metrics[key]) for key in numeric}
            counts = regions.get((name, variant["name"]), {})
            row.update({key: float(value) for key, value in counts.items()})
            row["HOTA_mem"] = float(subset[(name, variant["name"])]["HOTA"])
            row["FP_mem"] = float(counts.get("fp_hard", 0) + counts.get("fp_normal", 0))
            row["FN_mem"] = float(counts.get("fn_hard", 0) + counts.get("fn_normal", 0))
            row.update({"kind": variant["kind"], "tau": variant["tau"], "seed": variant["seed"], "normal": variant["normal"]})
            current[variant["name"]] = row
        groups = defaultdict(list)
        for variant in variants:
            if variant["kind"] == "shuffle":
                suffix = variant["name"].split("_", 1)[1]
                groups[suffix].append(current[variant["name"]])
        for suffix, members in groups.items():
            keys = set().union(*[set(item) for item in members]) - {"kind", "tau", "seed", "normal"}
            merged = {key: float(np.mean([item.get(key, 0.0) for item in members])) for key in keys}
            merged.update(
                {
                    "kind": "shuffle",
                    "tau": members[0]["tau"],
                    "seed": "mean",
                    "normal": members[0]["normal"],
                    "HOTA_std": float(np.std([item["HOTA"] for item in members])),
                }
            )
            current["shuffle_" + suffix] = merged
        rows[name] = current
    return rows


def _family(config, rows, kind, normal=None):
    if kind == "global":
        names = ["global_%.2f" % tau for tau in _global_taus(config)]
        start = rows["baseline"]
    else:
        prefix = kind if normal is None else "%s_n%.2f" % (kind, normal)
        names = [
            "%s_%.2f" % (prefix, float(tau))
            for tau in config["tau_low"]
            if normal is None or float(tau) < normal - 1.0e-9
        ]
        start = rows["baseline"] if normal is None else rows["global_%.2f" % normal]
    points = sorted([rows[name] for name in names if name in rows], key=lambda item: -item["tau"])
    return [start] + points


def _best_global(config, rows):
    best = max(_family(config, rows, "global"), key=lambda item: item["HOTA"])
    return best["tau"]


def _families(config, rows, normal):
    output = {("global", None): _family(config, rows, "global")}
    for kind in ("scene", "reliability", "shuffle"):
        output[(kind, None)] = _family(config, rows, kind)
        if normal is not None:
            output[(kind, normal)] = _family(config, rows, kind, normal)
    return output


def _interp(curve, x_key, x, y_key):
    ordered = sorted(curve, key=lambda item: item[x_key])
    xs = np.asarray([item[x_key] for item in ordered], dtype=np.float64)
    ys = np.asarray([item[y_key] for item in ordered], dtype=np.float64)
    if x < xs.min() or x > xs.max():
        return None
    return float(np.interp(x, xs, ys))


def _matched(reference, family, x_key, fn_key, hota_key):
    output = []
    for point in family[1:]:
        fn = _interp(reference, x_key, point[x_key], fn_key)
        hota = _interp(reference, x_key, point[x_key], hota_key)
        if fn is None:
            continue
        output.append(
            {
                "tau": point["tau"],
                "dFN_pct": 100.0 * (point[fn_key] - fn) / max(fn, 1.0),
                "dHOTA": 100.0 * (point[hota_key] - hota),
            }
        )
    return output


def _matched_summary(matched):
    if not matched:
        return None
    return {
        "points": len(matched),
        "dFN_pct": float(np.mean([item["dFN_pct"] for item in matched])),
        "dHOTA": float(np.mean([item["dHOTA"] for item in matched])),
    }


def _describe(row):
    if row["kind"] == "baseline":
        return "--"
    if row["kind"] == "global":
        return "全局%.2f" % row["tau"]
    if row["normal"] is None:
        return "普通区τ0/低可靠区%.2f" % row["tau"]
    return "普通区%.2f/低可靠区%.2f" % (row["normal"], row["tau"])


def _describe_en(row):
    if row["kind"] == "baseline":
        return ""
    if row["kind"] == "global":
        return "τ=%.2f" % row["tau"]
    if row["normal"] is None:
        return "hard %.2f" % row["tau"]
    return "normal %.2f / hard %.2f" % (row["normal"], row["tau"])


def _best_of(families, kind):
    candidates = [point for (current, _), family in families.items() if current == kind for point in family[1:]]
    return max(candidates, key=lambda item: item["HOTA"])


def _plot_bars(output_root, config, rows, normals):
    import matplotlib.pyplot as plt

    names = [spec["name"] for spec in config["trackers"]]
    kinds = ("baseline", "global", "scene", "shuffle")
    metrics = ("HOTA", "DetA", "AssA", "IDF1", "MOTA")
    figure, axes = plt.subplots(len(names), 2, figsize=(13.5, 4.3 * len(names)), squeeze=False)
    for row_index, name in enumerate(names):
        current = rows[name]
        families = _families(config, current, normals[name])
        selected = [current["baseline"]] + [_best_of(families, kind) for kind in kinds[1:]]
        labels = ["Baseline"] + ["%s (%s)" % (LABELS[kind], _describe_en(item)) for kind, item in zip(kinds[1:], selected[1:])]
        axis = axes[row_index][0]
        x = np.arange(len(metrics))
        width = 0.2
        for index, (kind, item, label) in enumerate(zip(kinds, selected, labels)):
            values = [100.0 * item[key] for key in metrics]
            axis.bar(x + (index - 1.5) * width, values, width, color=COLORS[kind], label=label)
        low = min(100.0 * item[key] for item in selected for key in metrics)
        axis.set_ylim(max(0.0, low - 4.0), 100.0)
        axis.set_xticks(x, metrics)
        axis.set_ylabel("Metric (%)")
        axis.set_title("%s: each method at its best-HOTA threshold (val, 21 seq)" % name)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=7.5, frameon=False, loc="upper left")
        axis = axes[row_index][1]
        deltas = ("FP", "FN", "IDSW")
        x = np.arange(len(deltas))
        for index, (kind, item) in enumerate(zip(kinds[1:], selected[1:])):
            values = [item[key] - current["baseline"][key] for key in deltas]
            bars = axis.bar(x + (index - 1) * 0.26, values, 0.26, color=COLORS[kind], label=LABELS[kind])
            for bar, value in zip(bars, values):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    "%+d" % round(value),
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=8,
                )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(x, ["ΔFP", "ΔFN", "ΔIDS"])
        axis.set_ylabel("Change vs. baseline (count)")
        axis.set_title("%s: errors added / removed vs. baseline" % name)
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=8, frameon=False)
    figure.tight_layout()
    figure.savefig(output_root / "figure_1_core_metrics.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def _plot_tradeoff(output_root, config, rows, normals):
    import matplotlib.pyplot as plt

    names = [spec["name"] for spec in config["trackers"]]
    panels = (("FN_mem", "FN (count)", 1.0), ("HOTA_mem", "HOTA (%)", 100.0))
    figure, axes = plt.subplots(len(panels), len(names), figsize=(6.6 * len(names), 4.6 * len(panels)), squeeze=False)
    for column, name in enumerate(names):
        current = rows[name]
        normal = normals[name]
        families = _families(config, current, normal)
        styles = [
            (("global", None), "Global-Low", COLORS["global"], "-", 2.2),
            (("scene", None), "Scene-Aware (hard only)", COLORS["scene"], ":", 1.4),
            (("shuffle", None), "Shuffled (hard only)", COLORS["shuffle"], ":", 1.4),
        ]
        if normal is not None:
            styles.extend(
                [
                    (("scene", normal), "Scene-Aware on Global %.2f" % normal, COLORS["scene"], "-", 2.2),
                    (("reliability", normal), "Reliability memory on Global %.2f" % normal, COLORS["reliability"], "-", 1.6),
                    (("shuffle", normal), "Shuffled on Global %.2f" % normal, COLORS["shuffle"], "--", 1.8),
                ]
            )
        limit = 1.15 * max(point["FP_mem"] for key, family in families.items() if key[0] != "global" for point in family)
        for row_index, (key, label, scale) in enumerate(panels):
            axis = axes[row_index][column]
            for family_key, text, color, style, width in styles:
                family = families[family_key]
                xs = [item["FP_mem"] for item in family]
                ys = [scale * item[key] for item in family]
                axis.plot(xs, ys, style, marker="o", markersize=3.0, linewidth=width, color=color, label=text)
                if family_key[0] == "global":
                    for item, x, y in zip(family[1:], xs[1:], ys[1:]):
                        if x <= limit:
                            axis.annotate("%.2f" % item["tau"], (x, y), fontsize=6.5, color=color, xytext=(3, 3), textcoords="offset points")
            base = current["baseline"]
            axis.plot([base["FP_mem"]], [scale * base[key]], marker="*", markersize=13, color="black", linestyle="none", label="Baseline")
            axis.set_xlim(0.9 * base["FP_mem"], limit)
            visible = [
                scale * item[key]
                for family_key, _, _, _, _ in styles
                for item in families[family_key]
                if item["FP_mem"] <= limit
            ]
            margin = 0.06 * (max(visible) - min(visible))
            axis.set_ylim(min(visible) - margin, max(visible) + margin)
            axis.set_xlabel("FP (count)")
            axis.set_ylabel(label)
            axis.grid(alpha=0.2)
            if row_index == 0:
                axis.set_title("%s: threshold sweep, 19 val seq with memory" % name)
            if row_index == 0 and column == 0:
                axis.legend(fontsize=7.5, frameon=False)
    figure.tight_layout()
    figure.savefig(output_root / "figure_2_threshold_tradeoff.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def _score_groups(memory, records, sequence_ids):
    groups = {key: [] for key in ("hard_tp", "hard_fp", "normal_tp", "normal_fp")}
    for sequence_id in sequence_ids:
        entry = memory.get(sequence_id)
        if entry is None:
            continue
        hard = {tuple(item) for item in entry["scene"]}
        current = records[sequence_id]
        for cell, score, positive in zip(current["cell"], current["score"], current["tp"]):
            if cell is None:
                continue
            region = "hard" if tuple(cell) in hard else "normal"
            groups["%s_%s" % (region, "tp" if positive else "fp")].append(float(score))
    return {key: np.asarray(value, dtype=np.float64) for key, value in groups.items()}


def _score_stats(groups, low, high):
    output = {}
    for region in ("hard", "normal"):
        tp = groups[region + "_tp"]
        fp = groups[region + "_fp"]
        band_tp = int(np.sum((tp >= low) & (tp < high)))
        band_fp = int(np.sum((fp >= low) & (fp < high)))
        output[region] = {
            "tp": int(tp.size),
            "fp": int(fp.size),
            "tp_median": float(np.median(tp)) if tp.size else float("nan"),
            "fp_median": float(np.median(fp)) if fp.size else float("nan"),
            "band_tp": band_tp,
            "band_fp": band_fp,
            "band_tp_share": band_tp / max(int(tp.size), 1),
            "band_precision": band_tp / max(band_tp + band_fp, 1),
        }
    return output


def _plot_scores(output_root, groups, stats, low, high):
    import matplotlib.pyplot as plt

    bins = np.arange(low, 1.0001, 0.05)
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
    for axis, kind, title in zip(axes, ("tp", "fp"), ("True-positive detections", "False-positive detections")):
        for region, color in (("hard", COLORS["scene"]), ("normal", COLORS["global"])):
            values = groups["%s_%s" % (region, kind)]
            weights = np.full(values.shape, 1.0 / max(values.size, 1))
            axis.hist(
                values,
                bins=bins,
                weights=weights,
                histtype="step",
                linewidth=2.0,
                color=color,
                label="%s (n=%d)" % ("Historical hard" if region == "hard" else "Normal", values.size),
            )
        axis.axvspan(low, high, color="#DDDDDD", alpha=0.35, zorder=0)
        axis.set_xlabel("CenterPoint score")
        axis.set_ylabel("Fraction of detections")
        axis.set_title(title)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=9)
    axes[0].text(
        0.02,
        0.60,
        "score in [%.2f, %.2f):\nhard   %.1f%% of TP, precision %.2f\nnormal %.1f%% of TP, precision %.2f"
        % (
            low,
            high,
            100.0 * stats["hard"]["band_tp_share"],
            stats["hard"]["band_precision"],
            100.0 * stats["normal"]["band_tp_share"],
            stats["normal"]["band_precision"],
        ),
        transform=axes[0].transAxes,
        fontsize=9,
        family="monospace",
    )
    axes[1].set_yscale("log")
    figure.suptitle("Val sequences; hard region built only from earlier sequences (grey band: mostly filtered by baseline)", y=1.02)
    figure.tight_layout()
    figure.savefig(output_root / "figure_3_score_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def _analysis(config, rows, normals):
    output = {}
    for name, current in rows.items():
        normal = normals[name]
        families = _families(config, current, normal)
        reference = families[("global", None)]
        tracker = {"normal": normal, "families": {}}
        for key, family in families.items():
            if key[0] == "global":
                continue
            wins, gaps = 0, []
            control = families.get(("shuffle", key[1]))
            if key[0] != "shuffle" and control is not None:
                for point, other in zip(family[1:], control[1:]):
                    gaps.append(100.0 * (point["HOTA_mem"] - other["HOTA_mem"]))
                    wins += int(point["HOTA_mem"] > other["HOTA_mem"])
            best = max(family[1:], key=lambda item: item["HOTA"])
            tracker["families"]["%s|%s" % key] = {
                "matched": _matched_summary(_matched(reference, family, "FP_mem", "FN_mem", "HOTA_mem")),
                "best": best,
                "best_vs_global": 100.0 * (best["HOTA"] - max(item["HOTA"] for item in reference)),
                "shuffle_wins": wins,
                "shuffle_points": len(gaps),
                "shuffle_gap": float(np.mean(gaps)) if gaps else None,
            }
        output[name] = tracker
    return output


def _go(entry):
    matched = entry["matched"]
    return bool(
        matched
        and matched["dFN_pct"] <= -3.0
        and matched["dHOTA"] >= 0.2
        and entry["best_vs_global"] >= 0.3
        and entry["shuffle_points"]
        and entry["shuffle_wins"] >= math.ceil(0.7 * entry["shuffle_points"])
        and entry["shuffle_gap"] >= 0.2
    )


def _fmt(value, digits=2, signed=False):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "--"
    return (("%%+.%df" if signed else "%%.%df") % digits) % float(value)


def _pct(value):
    return "%.2f" % (100.0 * float(value))


def _summary(config, rows, normals, analysis, stats, memory, sequence_ids):
    names = [spec["name"] for spec in config["trackers"]]
    main = names[0]
    low = float(min(config["tau_low"]))
    high = float(config["reliability_band_high"])
    covered = len([item for item in sequence_ids if item in memory])
    lines = [
        "# Scene-Aware Detection Threshold：Proof-of-Concept",
        "",
        "## 实验问题",
        "",
        "空间自适应地使用低分detection，是否比全局降低detection threshold更好？",
        "",
        "- CenterPoint检测，V2X-Seq val 21个序列，官方Car评估；%s为主，%s看趋势" % (main, "、".join(names[1:])),
        "- 阈值同时作用于tracker输入和逐框输出；association、motion、max_age、lifecycle都不变",
        "- Scene-Aware直接复用上一轮5m三tracker consensus Hard Region（只用目标序列之前的序列，取最高20% cell）",
        "- Shuffled：cell数和阈值完全相同，只随机放置，3个seed平均；Reliability memory：同样cell数，按历史[%.2f, %.2f)低分检测的precision选cell" % (low, high),
        "- 两级阈值：普通区用Global-Low最优阈值，低可靠区再降低，用来检验空间记忆相对最优全局阈值是否还有增量",
        "- **proof-of-concept / oracle historical scene memory**：记忆用了历史序列GT，未使用目标序列GT；%d/%d个val序列有历史" % (covered, len(sequence_ids)),
        "- 注意：官方评估器按整条轨迹平均分过滤，本实验改为逐框施加输出阈值再以阈值0评估，使HOTA与CLEAR基于同一组框；因此Baseline的FP/FN与仓库官方数字不同，HOTA一致",
        "",
        "## 核心结果",
        "",
    ]
    for name in names:
        current = rows[name]
        normal = normals[name]
        families = _families(config, current, normal)
        entries = [("Baseline", current["baseline"])]
        entries.append(("Global-Low", max(families[("global", None)][1:], key=lambda item: item["HOTA"])))
        entries.append(("Scene-Aware（只改低可靠区）", analysis[name]["families"]["scene|None"]["best"]))
        entries.append(("Shuffled（只改随机区）", analysis[name]["families"]["shuffle|None"]["best"]))
        if normal is not None:
            entries.append(("Scene-Aware（叠加在最优Global上）", analysis[name]["families"]["scene|%s" % normal]["best"]))
            entries.append(("Shuffled（叠加在最优Global上）", analysis[name]["families"]["shuffle|%s" % normal]["best"]))
            entries.append(("Reliability memory（叠加在最优Global上）", analysis[name]["families"]["reliability|%s" % normal]["best"]))
        lines.extend(
            [
                "### %s（val 21序列，各方法取自身best-HOTA阈值）" % name,
                "",
                "| Method | 阈值 | HOTA | DetA | AssA | IDF1 | MOTA | IDS | FP | FN |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for label, item in entries:
            lines.append(
                "| %s | %s | %s | %s | %s | %s | %s | %d | %d | %d |"
                % (
                    label,
                    _describe(item),
                    _pct(item["HOTA"]),
                    _pct(item["DetA"]),
                    _pct(item["AssA"]),
                    _pct(item["IDF1"]),
                    _pct(item["MOTA"]),
                    round(item["IDSW"]),
                    round(item["FP"]),
                    round(item["FN"]),
                )
            )
        lines.extend(
            [
                "",
                "新放行的低分检测（相对各自参照：单级相对Baseline，两级相对最优Global）：",
                "",
                "| Method | 放行真检测 | 放行假检测 | 真/假 | 真检测被输出且匹配GT | 其中是该GT主ID | Hard区FN变化 | Normal区FP变化 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for label, item in entries[1:]:
            reference = current["baseline"] if item["normal"] is None else current["global_%.2f" % item["normal"]]
            tp = item.get("released_tp", 0.0)
            fp = item.get("released_fp", 0.0)
            fn_change = item.get("fn_hard", 0.0) - reference.get("fn_hard", 0.0)
            fp_change = item.get("fp_normal", 0.0) - reference.get("fp_normal", 0.0)
            lines.append(
                "| %s | %d | %d | %s | %s%% | %s%% | %+d（%s%%） | %+d（%s%%） |"
                % (
                    label,
                    round(tp),
                    round(fp),
                    _fmt(tp / fp if fp else None),
                    _fmt(100.0 * item.get("released_covered", 0.0) / tp if tp else None, 1),
                    _fmt(100.0 * item.get("released_correct", 0.0) / tp if tp else None, 1),
                    round(fn_change),
                    _fmt(100.0 * fn_change / max(reference.get("fn_hard", 0.0), 1.0), 1, True),
                    round(fp_change),
                    _fmt(100.0 * fp_change / max(reference.get("fp_normal", 0.0), 1.0), 1, True),
                )
            )
        lines.extend(["", "相同FP下与Global-Low曲线比较（19个有记忆的序列，插值到相同FP，负FN更好）：", ""])
        keys = [("scene", None), ("shuffle", None)]
        if normal is not None:
            keys.extend([("scene", normal), ("reliability", normal), ("shuffle", normal)])
        for kind, level in keys:
            entry = analysis[name]["families"]["%s|%s" % (kind, level)]
            matched = entry["matched"]
            label = "%s（%s）" % (LABELS[kind], "只改低可靠区" if level is None else "叠加在Global %.2f上" % level)
            if matched is None:
                lines.append("- %s：FP范围与Global-Low不重叠" % label)
                continue
            text = "- %s：%d点，FN %s%%，HOTA %s" % (label, matched["points"], _fmt(matched["dFN_pct"], 1, True), _fmt(matched["dHOTA"], 2, True))
            if entry["shuffle_points"]:
                text += "；同阈值胜Shuffled %d/%d，HOTA平均差%s" % (entry["shuffle_wins"], entry["shuffle_points"], _fmt(entry["shuffle_gap"], 2, True))
            lines.append(text)
        lines.append("")
    hard = stats["hard"]
    normal_stats = stats["normal"]
    lines.extend(
        [
            "### Hard vs Normal检测分数（Figure 3）",
            "",
            "- 真检测落在低分区间[%.2f, %.2f)的比例：Hard %.1f%%，Normal %.1f%%"
            % (low, high, 100.0 * hard["band_tp_share"], 100.0 * normal_stats["band_tp_share"]),
            "- 该区间precision：Hard %.2f（%d真/%d假），Normal %.2f（%d真/%d假）"
            % (hard["band_precision"], hard["band_tp"], hard["band_fp"], normal_stats["band_precision"], normal_stats["band_tp"], normal_stats["band_fp"]),
            "- 真检测分数中位数Hard %.3f/Normal %.3f，假检测分数中位数Hard %.3f/Normal %.3f"
            % (hard["tp_median"], normal_stats["tp_median"], hard["fp_median"], normal_stats["fp_median"]),
            "",
        ]
    )
    lines.extend(_conclusion(config, rows, normals, analysis, stats))
    return "\n".join(lines) + "\n"


def _conclusion(config, rows, normals, analysis, stats):
    names = [spec["name"] for spec in config["trackers"]]
    main = names[0]
    normal = normals[main]
    current = analysis[main]["families"]
    single = current["scene|None"]
    shuffle_single = current["shuffle|None"]
    stacked = current.get("scene|%s" % normal)
    reliability = current.get("reliability|%s" % normal)
    shuffle_stacked = current.get("shuffle|%s" % normal)
    main_rows = rows[main]
    global_best = max(_family(config, main_rows, "global")[1:], key=lambda item: item["HOTA"])
    baseline = main_rows["baseline"]
    candidates = [item for item in (stacked, reliability) if item is not None]
    go = any(_go(item) for item in candidates)
    trend = []
    for name in names[1:]:
        other = analysis[name]
        entries = [other["families"].get("%s|%s" % (kind, other["normal"])) for kind in ("scene", "reliability")]
        trend.append((name, any(_go(item) for item in entries if item is not None)))
    lines = [
        "## 最关键结论（%s）" % main,
        "",
        "- Global-Low本身就有效：全局%.2f相对Baseline HOTA %s、FN %+d、FP %+d、IDS %+d，说明原输出阈值在逐框输出下偏高"
        % (
            global_best["tau"],
            _fmt(100.0 * (global_best["HOTA"] - baseline["HOTA"]), 2, True),
            round(global_best["FN"] - baseline["FN"]),
            round(global_best["FP"] - baseline["FP"]),
            round(global_best["IDSW"] - baseline["IDSW"]),
        ),
        "- 只改低可靠区：best HOTA相对Global-Low %s；相同FP下FN %s%%、HOTA %s；胜Shuffled %d/%d（HOTA平均差%s）"
        % (
            _fmt(single["best_vs_global"], 2, True),
            _fmt(single["matched"]["dFN_pct"] if single["matched"] else None, 1, True),
            _fmt(single["matched"]["dHOTA"] if single["matched"] else None, 2, True),
            single["shuffle_wins"],
            single["shuffle_points"],
            _fmt(single["shuffle_gap"], 2, True),
        ),
    ]
    if stacked is not None:
        lines.append(
            "- 叠加在最优Global上：Scene-Aware best HOTA相对Global-Low %s，相同FP下FN %s%%、HOTA %s，胜Shuffled %d/%d（%s）；Reliability memory分别为%s、%s%%、%s"
            % (
                _fmt(stacked["best_vs_global"], 2, True),
                _fmt(stacked["matched"]["dFN_pct"] if stacked["matched"] else None, 1, True),
                _fmt(stacked["matched"]["dHOTA"] if stacked["matched"] else None, 2, True),
                stacked["shuffle_wins"],
                stacked["shuffle_points"],
                _fmt(stacked["shuffle_gap"], 2, True),
                _fmt(reliability["best_vs_global"], 2, True),
                _fmt(reliability["matched"]["dFN_pct"] if reliability["matched"] else None, 1, True),
                _fmt(reliability["matched"]["dHOTA"] if reliability["matched"] else None, 2, True),
            )
        )
        lines.append(
            "- Shuffled叠加在最优Global上：相同FP下FN %s%%、HOTA %s"
            % (
                _fmt(shuffle_stacked["matched"]["dFN_pct"] if shuffle_stacked["matched"] else None, 1, True),
                _fmt(shuffle_stacked["matched"]["dHOTA"] if shuffle_stacked["matched"] else None, 2, True),
            )
        )
    lines.append(
        "- Hard区确实更“低分但正确”：真检测落在低分区间的比例%.1f%% vs %.1f%%，precision %.2f vs %.2f；但Hard区低分检测仍是假多真少"
        % (
            100.0 * stats["hard"]["band_tp_share"],
            100.0 * stats["normal"]["band_tp_share"],
            stats["hard"]["band_precision"],
            stats["normal"]["band_precision"],
        )
    )
    lines.extend(
        [
            "- 其他tracker：%s" % "；".join("%s%s判定规则" % (name, "同样满足" if value else "同样不满足") for name, value in trend),
            "",
            "判定规则（数值在首次运行前写入代码，之后未调整）：相对Global-Low曲线，相同FP下FN平均降低≥3%且HOTA≥+0.2，best HOTA≥Global-Low best+0.3，并且≥70%阈值点胜过同条件Shuffled、平均HOTA差≥0.2",
            "",
            "## Go / No-Go",
            "",
        ]
    )
    if go:
        lines.extend(
            [
                "- Spatial score calibration：**Go**",
                "- 是否值得进入正式Scene Memory方法设计：**Yes**",
                "- 下一步最应该建模：不依赖GT的在线cell级检测可靠性（低分检测precision/score偏移），并与全局阈值联合校准，而不是替代全局阈值",
            ]
        )
    else:
        scene_row = single["best"]
        shuffle_row = shuffle_single["best"]
        gains = [item["best_vs_global"] for item in candidates]
        lines.extend(
            [
                "- Spatial score calibration：**No-Go**",
                "- 是否值得进入正式Scene Memory方法设计：**No**（不应沿“空间分数阈值”继续堆模型）",
                "- 原因1：收益几乎全部来自把全局阈值调到合适位置（HOTA %s）；Global-Low曲线基本就是FP–FN的Pareto前沿，只改低可靠区只在紧挨Baseline的第一步更省FP"
                % _fmt(100.0 * (global_best["HOTA"] - baseline["HOTA"]), 2, True),
                "- 原因2：叠加在最优全局阈值之上后，按历史位置放宽的best HOTA增量只有%s～%s，远低于判定线；Hard Region版本同阈值只胜Shuffled %d/%d（HOTA平均差%s）"
                % (
                    _fmt(min(gains), 2, True),
                    _fmt(max(gains), 2, True),
                    stacked["shuffle_wins"] if stacked else 0,
                    stacked["shuffle_points"] if stacked else 0,
                    _fmt(stacked["shuffle_gap"] if stacked else None, 2, True),
                ),
                "- 仍然成立的事实：历史位置确实有信息。只改低可靠区时，Scene-Aware放行检测真/假比%s，Shuffled %s；Hard区FN %+d，Shuffled只有%+d，且Normal区FP不增加；IDS也明显少于Global-Low（%d vs %d）"
                % (
                    _fmt(scene_row.get("released_tp", 0.0) / max(scene_row.get("released_fp", 0.0), 1.0)),
                    _fmt(shuffle_row.get("released_tp", 0.0) / max(shuffle_row.get("released_fp", 0.0), 1.0)),
                    round(scene_row.get("fn_hard", 0.0) - main_rows["baseline"].get("fn_hard", 0.0)),
                    round(shuffle_row.get("fn_hard", 0.0) - main_rows["baseline"].get("fn_hard", 0.0)),
                    round(scene_row["IDSW"]),
                    round(global_best["IDSW"]),
                ),
                "- 但这些信息转化不成跟踪收益：Hard区低分检测precision只有%.2f，真检测总量有限，放宽后新增FP抵消了FN收益；Reliability memory同阈值胜Shuffled %d/%d，但HOTA平均只高%s"
                % (
                    stats["hard"]["band_precision"],
                    reliability["shuffle_wins"] if reliability else 0,
                    reliability["shuffle_points"] if reliability else 0,
                    _fmt(reliability["shuffle_gap"] if reliability else None, 2, True),
                ),
                "- 建议：停止在score calibration上加模型，重新评估Roadside Scene Memory方向；若继续，应面向检测完全缺失或定位偏差的目标，而不是分数阈值",
            ]
        )
    return lines


def run(config):
    output_root = Path(config["project"]["output_root"])
    debug_root = output_root / "debug"
    debug_root.mkdir(parents=True, exist_ok=True)
    protocol = V2XSeqProtocol(config["_root"])
    sequence_ids = [str(value) for value in read_json(_path(config["_root"], config["split_file"]))["val"]]
    scenes, _, _ = build_scenes(config)
    all_ids = [sequence["sequence_id"] for scene in scenes for sequence in scene["sequences"]]
    gt_counts, records = _detection_records(config, all_ids, debug_root)
    memory = _memories(config, scenes, protocol, gt_counts, records, debug_root)
    memory_ids = [sequence_id for sequence_id in sequence_ids if sequence_id in memory]
    setups = {spec["name"]: _tracker_setup(config, spec) for spec in config["trackers"]}
    plan = {spec["name"]: _stage_one(config) for spec in config["trackers"]}
    _run_trackers(config, plan, setups, memory, sequence_ids, debug_root)
    results = _evaluate(config, plan, setups, sequence_ids, debug_root)
    normals = {}
    for name in plan:
        global_rows = {"baseline": results[(name, "baseline")]}
        global_rows.update({"global_%.2f" % tau: dict(results[(name, "global_%.2f" % tau)], tau=tau) for tau in _global_taus(config)})
        global_rows["baseline"] = dict(global_rows["baseline"], tau=None)
        best = _best_global(config, global_rows)
        normals[name] = best if config.get("two_level", False) else None
        if normals[name] is not None:
            plan[name] = plan[name] + _stage_two(config, best)
    print("最优Global阈值%s" % normals, flush=True)
    _run_trackers(config, plan, setups, memory, sequence_ids, debug_root)
    results = _evaluate(config, plan, setups, sequence_ids, debug_root)
    regions = _region_stats(config, plan, setups, memory, records, sequence_ids, debug_root)
    subset = _subset_hota(config, plan, setups, memory_ids, debug_root)
    rows = _rows(config, plan, results, regions, subset)
    analysis = _analysis(config, rows, normals)
    low = float(min(config["tau_low"]))
    high = float(config["reliability_band_high"])
    groups = _score_groups(memory, records, sequence_ids)
    stats = _score_stats(groups, low, high)
    write_json(debug_root / "rows.json", rows)
    write_json(debug_root / "analysis.json", {"normals": normals, "analysis": analysis, "score_stats": stats})
    _plot_bars(output_root, config, rows, normals)
    _plot_tradeoff(output_root, config, rows, normals)
    _plot_scores(output_root, groups, stats, low, high)
    (output_root / "summary.md").write_text(
        _summary(config, rows, normals, analysis, stats, memory, sequence_ids), encoding="utf-8"
    )
    stale = debug_root / "summary_draft.md"
    if stale.is_file():
        stale.unlink()
    print("完成scene threshold实验%s" % output_root, flush=True)
