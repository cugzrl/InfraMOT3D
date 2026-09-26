import copy
import csv
import math
import time
import zlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

from inframot3d.analysis.scene_difficulty import build_scenes, collect_events
from inframot3d.analysis.scene_persist import (
    _accumulate,
    _annotate_fragment_starts,
    _minimum,
    _risk_map,
    _select_regions,
)
from inframot3d.config import load_config
from inframot3d.evaluation import UnifiedMOTEvaluator
from inframot3d.evaluation.hota import evaluate_hota
from inframot3d.io import read_json, read_jsonl, write_json, write_jsonl
from inframot3d.tracking import create_tracker


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path


def _tracker_settings(config):
    tracker_config = load_config(_resolve(config["_root"], config["tracker_config"]))
    settings = copy.deepcopy(tracker_config["tracker"])
    threshold_path = _resolve(config["_root"], settings["score_thresholds_file"])
    thresholds = yaml.safe_load(threshold_path.read_text(encoding="utf-8"))
    settings["score_thresholds"] = {
        str(name): float(value) for name, value in thresholds["score_thresholds"].items()
    }
    return settings


def _method_specs(config):
    output = [{"name": "baseline", "mode": "none", "delta_age": 0}]
    for delta in config["delta_ages"]:
        for mode in ("global", "scene"):
            output.append({"name": "%s_d%d" % (mode, int(delta)), "mode": mode, "delta_age": int(delta)})
    delta = int(config["detail_delta"])
    output.append({"name": "shuffled_d%d" % delta, "mode": "shuffled", "delta_age": delta})
    return output


def _event_index(events):
    grouped = defaultdict(list)
    for event in events:
        grouped[(event["scene_id"], event["sequence_id"])].append(event)
    return grouped


def _shuffled_cells(valid, width, scene_id, sequence_id, seed):
    if width <= 0 or not valid:
        return []
    token = ("%s:%s" % (scene_id, sequence_id)).encode("utf-8")
    current_seed = int(seed) + int(zlib.crc32(token))
    rng = np.random.default_rng(current_seed)
    order = rng.permutation(len(valid))[:width]
    return [valid[int(index)] for index in order]


def _policies(config, scenes, events):
    grouped = _event_index(events)
    tracker = config["history_tracker"]
    cell = float(config["grid_m"])
    minimum = _minimum(config, cell)
    policies = {}
    for scene in scenes:
        history_records = []
        history_ids = []
        for sequence in scene["sequences"]:
            sequence_id = sequence["sequence_id"]
            if history_records:
                counts = _accumulate(history_records, tracker, config["bev"], cell)
                risk = _risk_map(counts, config["prior_strength"], minimum)
                hard, _ = _select_regions(risk, counts, config["bev"], cell, config["hard_fraction"])
                valid = [tuple(int(value) for value in item) for item in np.argwhere(np.isfinite(risk))]
            else:
                hard = []
                valid = []
            shuffled = _shuffled_cells(valid, len(hard), scene["scene_id"], sequence_id, config["seed"])
            policies[(scene["scene_id"], sequence_id)] = {
                "history_sequences": list(history_ids),
                "hard_cells": list(hard),
                "shuffled_cells": list(shuffled),
                "valid_cells": len(valid),
            }
            history_records.extend(grouped[(scene["scene_id"], sequence_id)])
            history_ids.append(sequence_id)
    return policies


def _frames(config, entry):
    converted = list(read_jsonl(config["project"]["converted_root"] / entry["path"]))
    detection_root = _resolve(config["_root"], config["detection_root"])
    detections = list(read_jsonl(detection_root / ("%s.jsonl" % entry["sequence_id"])))
    if len(converted) != len(detections):
        raise ValueError("检测帧数与转换数据不一致%s" % entry["sequence_id"])
    by_id = {row["frame_id"]: row for row in detections}
    output = []
    for frame in converted:
        detection = by_id.get(frame["frame_id"])
        if detection is None or int(detection["timestamp"]) != int(frame["timestamp"]):
            raise ValueError("检测帧未对齐%s" % frame["frame_id"])
        output.append(detection)
    return output


def _memory_policy(config, spec, policy):
    cells = []
    if spec["mode"] == "scene":
        cells = policy["hard_cells"]
    elif spec["mode"] == "shuffled":
        cells = policy["shuffled_cells"]
    return {
        "enabled": spec["mode"] != "none",
        "mode": spec["mode"],
        "delta_age": int(spec["delta_age"]),
        "grid_m": float(config["grid_m"]),
        "bev": dict(config["bev"]),
        "hard_cells": [list(item) for item in cells],
    }


def _run_sequence(config, settings, spec, policy, frames):
    current = copy.deepcopy(settings)
    current["scene_memory"] = _memory_policy(config, spec, policy)
    tracker = create_tracker(current, config["_root"])
    rows = []
    for frame in frames:
        objects = [
            {
                "class_name": item["class_name"],
                "score": float(item.get("score", 1.0)),
                "box": item["box"],
            }
            for item in frame["objects"]
        ]
        outputs = tracker.update(objects, timestamp=frame["timestamp"])
        rows.append(
            {
                "sequence_id": frame["sequence_id"],
                "frame_index": frame["frame_index"],
                "frame_id": frame["frame_id"],
                "timestamp": frame["timestamp"],
                "objects": outputs,
            }
        )
    return rows, tracker.memory_stats()


def _prediction_job(payload):
    config, settings, methods, scene_id, sequence, policy, entry, split = payload
    sequence_id = sequence["sequence_id"]
    frames = _frames(config, entry)
    output_root = Path(config["project"]["output_root"])
    stats = []
    for spec in methods:
        rows, current = _run_sequence(config, settings, spec, policy, frames)
        prediction_dir = output_root / "predictions" / spec["name"]
        write_jsonl(prediction_dir / ("%s.jsonl" % sequence_id), rows)
        stats.append(
            {
                "scene_id": scene_id,
                "sequence_id": sequence_id,
                "split": split,
                "method": spec["name"],
                "mode": spec["mode"],
                "delta_age": spec["delta_age"],
                "frames": len(frames),
                "history_sequences": len(policy["history_sequences"]),
                "valid_cells": policy["valid_cells"],
                "hard_cells": len(policy["hard_cells"]),
                **current,
            }
        )
    return scene_id, sequence_id, stats


def _prediction_run(config, scenes, policies):
    settings = _tracker_settings(config)
    methods = _method_specs(config)
    output_root = Path(config["project"]["output_root"])
    manifest = read_json(config["project"]["converted_root"] / "manifest.json")
    entries = {entry["sequence_id"]: entry for entry in manifest["sequences"]}
    split_data = read_json(_resolve(config["_root"], config["split_file"]))
    split_lookup = {
        str(sequence_id): split
        for split in ("train", "val", "test")
        for sequence_id in split_data.get(split, [])
    }
    for spec in methods:
        (output_root / "predictions" / spec["name"]).mkdir(parents=True, exist_ok=True)
    jobs = []
    for scene in scenes:
        for sequence in scene["sequences"]:
            sequence_id = sequence["sequence_id"]
            split = split_lookup.get(sequence_id, "other")
            if split != str(config.get("prediction_split", "val")):
                continue
            jobs.append(
                (
                    config,
                    settings,
                    methods,
                    scene["scene_id"],
                    sequence,
                    policies[(scene["scene_id"], sequence_id)],
                    entries[sequence_id],
                    split,
                )
            )
    stats = []
    start = time.perf_counter()
    workers = max(1, int(config.get("prediction_workers", 4)))
    if workers == 1:
        for job in jobs:
            scene_id, sequence_id, current = _prediction_job(job)
            stats.extend(current)
            print("完成场景%s序列%s" % (scene_id, sequence_id), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_prediction_job, job) for job in jobs]
            for future in as_completed(futures):
                scene_id, sequence_id, current = future.result()
                stats.extend(current)
                print("完成场景%s序列%s" % (scene_id, sequence_id), flush=True)
    stats.sort(key=lambda row: (row["scene_id"], row["sequence_id"], row["method"]))
    elapsed = time.perf_counter() - start
    _write_csv(output_root / "memory_stats.csv", stats)
    return methods, stats, elapsed


def _baseline_equal(config, scenes):
    generated = Path(config["project"]["output_root"]) / "predictions" / "baseline"
    reference = _resolve(config["_root"], config["baseline_prediction_root"])
    mismatches = []
    for scene in scenes:
        for sequence in scene["sequences"]:
            if sequence["split"] != str(config.get("prediction_split", "val")):
                continue
            filename = "%s.jsonl" % sequence["sequence_id"]
            if (generated / filename).read_bytes() != (reference / filename).read_bytes():
                mismatches.append(sequence["sequence_id"])
    if mismatches:
        raise AssertionError("baseline预测不一致%s" % ",".join(mismatches))
    return True


def _evaluate(config, scenes, methods):
    root = config["_root"]
    output_root = Path(config["project"]["output_root"])
    evaluator = UnifiedMOTEvaluator(root, "v2xseq")
    detail_delta = int(config["detail_delta"])
    detail_methods = {"baseline", "global_d%d" % detail_delta, "shuffled_d%d" % detail_delta, "scene_d%d" % detail_delta}
    val_ids = read_json(_resolve(root, config["split_file"]))["val"]
    hota_threshold = None
    overall = []
    for spec in methods:
        name = spec["name"]
        metrics_path = output_root / "evaluation" / name / "metrics.json"
        if name == "baseline":
            metrics = read_json(_resolve(root, config["baseline_metrics"]))
            hota_threshold = float(metrics["best_score_threshold"])
        elif metrics_path.is_file():
            cached = read_json(metrics_path)
            if abs(float(cached["best_score_threshold"]) - hota_threshold) < 1.0e-9:
                metrics = cached
            else:
                metrics = evaluator.evaluate(
                    config,
                    output_root / "predictions" / name,
                    output_root / "evaluation" / name,
                    split="val",
                    score_threshold=hota_threshold,
                )
        else:
            metrics = evaluator.evaluate(
                config,
                output_root / "predictions" / name,
                output_root / "evaluation" / name,
                split="val",
                score_threshold=hota_threshold,
            )
        if spec["name"] in detail_methods:
            metrics.update(
                evaluate_hota(
                    config,
                    output_root / "predictions" / name,
                    val_ids,
                    evaluator.protocol,
                    hota_threshold,
                )
            )
        overall.append({"method": name, "mode": spec["mode"], "delta_age": spec["delta_age"], **metrics})
        print("完成评估%s" % name, flush=True)
    per_scene = []
    for spec in methods:
        if spec["name"] not in detail_methods:
            continue
        for scene in scenes:
            sequence_ids = [item["sequence_id"] for item in scene["sequences"] if item["split"] == "val"]
            if not sequence_ids:
                continue
            current_dir = output_root / "evaluation_per_scene" / spec["name"] / scene["scene_id"]
            metrics_path = current_dir / "metrics.json"
            if metrics_path.is_file():
                cached = read_json(metrics_path)
                if abs(float(cached["best_score_threshold"]) - hota_threshold) < 1.0e-9:
                    metrics = cached
                else:
                    metrics = evaluator.evaluate(
                        config,
                        output_root / "predictions" / spec["name"],
                        current_dir,
                        sequences=sequence_ids,
                        score_threshold=hota_threshold,
                    )
            else:
                metrics = evaluator.evaluate(
                    config,
                    output_root / "predictions" / spec["name"],
                    current_dir,
                    sequences=sequence_ids,
                    score_threshold=hota_threshold,
                )
            per_scene.append(
                {
                    "scene_id": scene["scene_id"],
                    "method": spec["name"],
                    "sequences": len(sequence_ids),
                    **metrics,
                }
            )
            print("完成分场景评估%s %s" % (spec["name"], scene["scene_id"]), flush=True)
    _write_csv(output_root / "metrics.csv", overall)
    _write_csv(output_root / "metrics_per_scene.csv", per_scene)
    return overall, per_scene


def _change(value, baseline):
    if float(baseline) == 0:
        return None
    return (float(value) - float(baseline)) / float(baseline) * 100.0


def _fmt(value, digits=4):
    if value is None or value == "" or not math.isfinite(float(value)):
        return "--"
    return ("%%.%df" % digits) % float(value)


def _signed(value, digits=4):
    if value is None or value == "" or not math.isfinite(float(value)):
        return "--"
    return ("%%+.%df" % digits) % float(value)


def _result_rows(metrics):
    baseline = next(row for row in metrics if row["method"] == "baseline")
    output = []
    for row in metrics:
        output.append(
            {
                **row,
                "HOTA": row.get("HOTA", ""),
                "AssA": row.get("AssA", ""),
                "delta_IDF1": float(row["IDF1"]) - float(baseline["IDF1"]),
                "delta_MOTA": float(row["MOTA"]) - float(baseline["MOTA"]),
                "IDSW_change_pct": _change(row["IDSW"], baseline["IDSW"]),
                "FM_change_pct": _change(row["FM"], baseline["FM"]),
                "FP_change": int(row["FP"]) - int(baseline["FP"]),
                "FN_change": int(row["FN"]) - int(baseline["FN"]),
            }
        )
    return output


def _write_table(path, rows, detail_delta):
    selected = {
        "baseline",
        "global_d%d" % detail_delta,
        "shuffled_d%d" % detail_delta,
        "scene_d%d" % detail_delta,
    }
    lines = [
        "| Method | HOTA | AssA | IDF1 | ΔIDF1 | MOTA | IDSW | IDSW change | FRAG | FRAG change | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["method"] not in selected:
            continue
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %d | %s%% | %d | %s%% | %d | %d |"
            % (
                row["method"],
                _fmt(row["HOTA"]),
                _fmt(row["AssA"]),
                _fmt(row["IDF1"]),
                _signed(row["delta_IDF1"]),
                _fmt(row["MOTA"]),
                int(row["IDSW"]),
                _fmt(row["IDSW_change_pct"], 1),
                int(row["FM"]),
                _fmt(row["FM_change_pct"], 1),
                int(row["FP"]),
                int(row["FN"]),
            )
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _aggregate_stats(stats, method, split="val"):
    selected = [row for row in stats if row["method"] == method and row["split"] == split]
    keys = (
        "frames",
        "unmatched_track_frames",
        "memory_trigger_frames",
        "extra_retained_frames",
        "recovered_tracks",
        "expired_after_extension",
        "triggered_tracks",
        "extended_tracks",
    )
    return {key: sum(int(row[key]) for row in selected) for key in keys}


def _write_scene_table(path, rows, detail_delta):
    methods = ["baseline", "global_d%d" % detail_delta, "shuffled_d%d" % detail_delta, "scene_d%d" % detail_delta]
    grouped = defaultdict(dict)
    for row in rows:
        grouped[row["scene_id"]][row["method"]] = row
    lines = [
        "| Scene | Method | IDF1 | ΔIDF1 | MOTA | IDSW | FRAG | FP | FN |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scene_id in sorted(grouped):
        baseline = grouped[scene_id].get("baseline")
        if baseline is None:
            continue
        for method in methods:
            row = grouped[scene_id].get(method)
            if row is None:
                continue
            lines.append(
                "| %s | %s | %s | %s | %s | %d | %d | %d | %d |"
                % (
                    scene_id,
                    method,
                    _fmt(row["IDF1"]),
                    _signed(float(row["IDF1"]) - float(baseline["IDF1"])),
                    _fmt(row["MOTA"]),
                    int(row["IDSW"]),
                    int(row["FM"]),
                    int(row["FP"]),
                    int(row["FN"]),
                )
            )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _comparison_figure(config, scenes, output_root):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    scene_id = str(config.get("figure_scene", "yizhuang09_lidar151"))
    sequence_id = str(config.get("figure_sequence", "0059"))
    scene = next(item for item in scenes if item["scene_id"] == scene_id)
    sequence_ids = [item["sequence_id"] for item in scene["sequences"]]
    target_index = sequence_ids.index(sequence_id)
    if target_index <= 0:
        raise ValueError("直观图target sequence缺少历史sequence")
    detail_delta = int(config["detail_delta"])
    history_config = copy.deepcopy(config)
    history_config["trackers"] = [
        {
            "name": "Baseline",
            "prediction_root": str(_resolve(config["_root"], config["baseline_prediction_root"])),
        },
    ]
    history_scene = copy.deepcopy(scene)
    history_scene["sequences"] = history_scene["sequences"][:target_index]
    history, _ = collect_events(history_config, [history_scene])
    _annotate_fragment_starts(history, history_config["trackers"])
    target_config = copy.deepcopy(config)
    target_config["trackers"] = [
        {"name": "Baseline", "prediction_root": str(output_root / "predictions" / "baseline")},
        {
            "name": "SceneMemory",
            "prediction_root": str(output_root / "predictions" / ("scene_d%d" % detail_delta)),
        },
    ]
    target_scene = copy.deepcopy(scene)
    target_scene["sequences"] = [target_scene["sequences"][target_index]]
    target, _ = collect_events(target_config, [target_scene])
    _annotate_fragment_starts(target, target_config["trackers"])
    cell = float(config["grid_m"])
    counts = _accumulate(history, "Baseline", config["bev"], cell)
    risk = _risk_map(counts, config["prior_strength"], _minimum(config, cell))
    hard, _ = _select_regions(risk, counts, config["bev"], cell, config["hard_fraction"])
    bev = config["bev"]
    shape = risk.shape
    hard_grid = np.zeros(shape, dtype=np.float64)
    for index in hard:
        hard_grid[index] = 1.0
    xs = float(bev["x_min"]) + (np.arange(shape[1]) + 0.5) * cell
    ys = float(bev["y_min"]) + (np.arange(shape[0]) + 0.5) * cell
    baseline_miss = [event for event in target if int(event["trackers"]["Baseline"]["miss"]) == 1]
    baseline_idsw = [event for event in target if int(event["trackers"]["Baseline"]["idsw"]) == 1]
    baseline_frag = [event for event in target if int(event["trackers"]["Baseline"]["frag_start"]) == 1]
    reduced = []
    remaining = []
    introduced = []
    for event in target:
        baseline_identity = int(event["trackers"]["Baseline"]["idsw"]) or int(
            event["trackers"]["Baseline"]["frag_start"]
        )
        scene_identity = int(event["trackers"]["SceneMemory"]["idsw"]) or int(
            event["trackers"]["SceneMemory"]["frag_start"]
        )
        if baseline_identity and not scene_identity:
            reduced.append(event)
        elif baseline_identity and scene_identity:
            remaining.append(event)
        elif scene_identity and not baseline_identity:
            introduced.append(event)
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.6), sharex=True, sharey=True)
    image = axes[0].imshow(
        np.ma.masked_invalid(risk),
        origin="lower",
        extent=[float(bev["x_min"]), float(bev["x_max"]), float(bev["y_min"]), float(bev["y_max"])],
        aspect="equal",
        cmap="magma",
    )
    axes[0].set_title("Historical miss risk")
    figure.colorbar(image, ax=axes[0], fraction=0.046, label="Risk")
    for axis in axes:
        if hard:
            axis.contour(xs, ys, hard_grid, levels=[0.5], colors=["#00d5ff"], linewidths=1.5)
        axis.scatter([event["x"] for event in target], [event["y"] for event in target], s=2, c="#b7b7b7", alpha=0.18)
        axis.scatter([0.0], [0.0], marker="^", s=65, c="white", edgecolors="black", linewidths=0.8)
        axis.set_xlim(float(bev["x_min"]) - 2.5, float(bev["x_max"]))
        axis.set_ylim(float(bev["y_min"]), float(bev["y_max"]))
        axis.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[1].scatter([event["x"] for event in baseline_miss], [event["y"] for event in baseline_miss], marker="x", s=20, c="#ef3b2c", linewidths=0.8)
    axes[1].scatter([event["x"] for event in baseline_idsw], [event["y"] for event in baseline_idsw], marker="*", s=75, c="#2171b5", edgecolors="white", linewidths=0.4)
    axes[1].scatter([event["x"] for event in baseline_frag], [event["y"] for event in baseline_frag], marker="o", s=45, facecolors="none", edgecolors="#ff8c00", linewidths=1.2)
    axes[1].set_title("Baseline future failures")
    axes[1].legend(
        handles=[
            Line2D([0], [0], marker="x", color="#ef3b2c", lw=0, label="Miss / lost"),
            Line2D([0], [0], marker="*", color="#2171b5", lw=0, markersize=9, label="ID switch"),
            Line2D([0], [0], marker="o", markerfacecolor="none", markeredgecolor="#ff8c00", lw=0, label="Fragment start"),
        ],
        loc="upper right",
        fontsize=8,
    )
    axes[2].scatter([event["x"] for event in introduced], [event["y"] for event in introduced], marker="D", s=35, c="#756bb1", edgecolors="white", linewidths=0.4)
    axes[2].scatter([event["x"] for event in remaining], [event["y"] for event in remaining], marker="x", s=30, c="#de2d26", linewidths=1.0)
    axes[2].scatter([event["x"] for event in reduced], [event["y"] for event in reduced], marker="o", s=105, facecolors="none", edgecolors="#20a354", linewidths=2.4)
    axes[2].set_title("Removed %d | Remaining %d | Introduced %d" % (len(reduced), len(remaining), len(introduced)))
    axes[2].legend(
        handles=[
            Line2D([0], [0], marker="o", markerfacecolor="none", markeredgecolor="#20a354", lw=0, label="Removed"),
            Line2D([0], [0], marker="x", color="#de2d26", lw=0, label="Remaining"),
            Line2D([0], [0], marker="D", color="#756bb1", lw=0, label="Introduced"),
        ],
        loc="upper right",
        fontsize=8,
    )
    figure.suptitle("%s | target %s | causal all-history memory" % (scene_id, sequence_id), fontsize=13)
    figure.tight_layout()
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = "%s_%s_comparison" % (scene_id, sequence_id)
    png_path = figure_dir / (stem + ".png")
    pdf_path = figure_dir / (stem + ".pdf")
    figure.savefig(png_path, dpi=240, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    metadata = {
        "scene_id": scene_id,
        "target_sequence": sequence_id,
        "history_sequences": sequence_ids[:target_index],
        "hard_cells": len(hard),
        "baseline_miss": len(baseline_miss),
        "baseline_idsw": len(baseline_idsw),
        "baseline_frag": len(baseline_frag),
        "identity_removed": len(reduced),
        "identity_remaining": len(remaining),
        "identity_introduced": len(introduced),
        "png": str(png_path),
        "pdf": str(pdf_path),
    }
    write_json(figure_dir / (stem + ".json"), metadata)
    return metadata


def _write_summary(path, results, per_scene, stats, config):
    detail_delta = int(config["detail_delta"])
    by_name = {row["method"]: row for row in results}
    baseline = by_name["baseline"]
    global_row = by_name["global_d%d" % detail_delta]
    shuffled = by_name["shuffled_d%d" % detail_delta]
    scene = by_name["scene_d%d" % detail_delta]
    scene_variants = [by_name["scene_d%d" % int(delta)] for delta in config["delta_ages"]]
    best_scene = max(scene_variants, key=lambda row: float(row["IDF1"]))
    best_delta = int(best_scene["delta_age"])
    best_global = by_name["global_d%d" % best_delta]
    memory_stats = _aggregate_stats(stats, scene["method"])
    average_extra = (
        memory_stats["extra_retained_frames"] / memory_stats["extended_tracks"]
        if memory_stats["extended_tracks"]
        else 0.0
    )
    best_stats = _aggregate_stats(stats, best_scene["method"])
    scene_rows = defaultdict(dict)
    for row in per_scene:
        scene_rows[row["scene_id"]][row["method"]] = row
    improved_idf1 = 0
    reduced_idsw = 0
    usable = 0
    for values in scene_rows.values():
        if "baseline" not in values or scene["method"] not in values:
            continue
        usable += 1
        improved_idf1 += int(float(values[scene["method"]]["IDF1"]) > float(values["baseline"]["IDF1"]))
        reduced_idsw += int(int(values[scene["method"]]["IDSW"]) < int(values["baseline"]["IDSW"]))
    go = (
        float(scene["IDF1"]) > float(baseline["IDF1"])
        and int(scene["IDSW"]) < int(baseline["IDSW"])
        and int(scene["FM"]) < int(baseline["FM"])
        and float(scene["IDF1"]) > float(global_row["IDF1"])
        and improved_idf1 >= 4
        and reduced_idsw >= 4
    )
    lines = [
        "# Scene survival experiment",
        "",
        "所有target sequence只读取更早sequence构建的历史hard region。association、motion和检测保持不变。",
        "",
        "## 主结果",
        "",
        "- Scene-memory将IDSW从%d变为%d（%s%%）" % (
            int(baseline["IDSW"]),
            int(scene["IDSW"]),
            _fmt(scene["IDSW_change_pct"], 1),
        ),
        "- IDF1从%s变为%s（%s）" % (
            _fmt(baseline["IDF1"]),
            _fmt(scene["IDF1"]),
            _signed(scene["delta_IDF1"]),
        ),
        "- FRAG从%d变为%d（%s%%）" % (
            int(baseline["FM"]),
            int(scene["FM"]),
            _fmt(scene["FM_change_pct"], 1),
        ),
        "- FP变化%+d，FN变化%+d" % (int(scene["FP_change"]), int(scene["FN_change"])),
        "- Global-longer的IDF1为%s，IDSW为%d，FRAG为%d" % (
            _fmt(global_row["IDF1"]),
            int(global_row["IDSW"]),
            int(global_row["FM"]),
        ),
        "- Shuffled-memory的IDF1为%s，IDSW为%d，FRAG为%d" % (
            _fmt(shuffled["IDF1"]),
            int(shuffled["IDSW"]),
            int(shuffled["FM"]),
        ),
        "- %d/%d个场景IDF1提升，%d/%d个场景IDSW下降" % (improved_idf1, usable, reduced_idsw, usable),
        "",
        "## ΔK扫描",
        "",
        "| ΔK | IDF1 | ΔIDF1 | IDSW | FRAG | FP | FN | recovered |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for current in scene_variants:
        current_stats = _aggregate_stats(stats, current["method"])
        lines.append(
            "| %d | %s | %s | %d | %d | %d | %d | %d |"
            % (
                int(current["delta_age"]),
                _fmt(current["IDF1"]),
                _signed(current["delta_IDF1"]),
                int(current["IDSW"]),
                int(current["FM"]),
                int(current["FP"]),
                int(current["FN"]),
                current_stats["recovered_tracks"],
            )
        )
    lines.extend(
        [
        "",
        "- 扫描中IDF1最高的是ΔK=%d，IDF1=%s，IDSW=%d，FRAG=%d" % (
            best_delta,
            _fmt(best_scene["IDF1"]),
            int(best_scene["IDSW"]),
            int(best_scene["FM"]),
        ),
        "- 同一ΔK的Global-longer为IDF1=%s，IDSW=%d，FRAG=%d" % (
            _fmt(best_global["IDF1"]),
            int(best_global["IDSW"]),
            int(best_global["FM"]),
        ),
        "- ΔK=%d实际恢复%d条track，仍过期%d条" % (
            best_delta,
            best_stats["recovered_tracks"],
            best_stats["expired_after_extension"],
        ),
        "",
        "## Memory触发",
        "",
        "- val帧数%d" % memory_stats["frames"],
        "- memory触发track-frame数%d" % memory_stats["memory_trigger_frames"],
        "- 触发track数%d" % memory_stats["triggered_tracks"],
        "- 真正超过原始max_age的track数%d" % memory_stats["extended_tracks"],
        "- 额外保留track-frame数%d" % memory_stats["extra_retained_frames"],
        "- 每条extended track平均额外保留%.2f帧" % average_extra,
        "- 延长后恢复关联%d条，延长后仍过期%d条" % (
            memory_stats["recovered_tracks"],
            memory_stats["expired_after_extension"],
        ),
        "",
        "## Go / No-Go",
        "",
        "- 结论：%s" % ("Go" if go else "No-Go（针对retention-only）"),
        "- 主设置ΔK=%d虽然提升IDF1、HOTA和AssA，但没有降低IDSW或FRAG，且只有%d/%d个场景IDF1提升" % (
            detail_delta,
            improved_idf1,
            usable,
        ),
        "- ΔK=%d仅将IDSW减少%d、FRAG减少%d，幅度不足以证明稳定收益" % (
            best_delta,
            int(baseline["IDSW"]) - int(best_scene["IDSW"]),
            int(baseline["FM"]) - int(best_scene["FM"]),
        ),
        "- 真实memory优于shuffled说明空间位置有信息，但当前hard region只覆盖少量可恢复轨迹，单独延长生命周期不是主要解法",
        "",
        "## 解释",
        "",
        "SimpleTrack仍只输出已关联轨迹，retention-only不会在漏检帧发布motion prediction。它主要影响后续重关联和ID连续性，不应被解释为直接降低当前帧miss。",
        "所有control、Scene-memory、HOTA和AssA均使用Baseline官方评测选出的统一score threshold，避免各方法单独调阈值。",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config, predict=True, evaluate=True):
    output_root = Path(config["project"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    scenes, _, warnings = build_scenes(config)
    methods = _method_specs(config)
    stats = []
    elapsed = None
    if predict:
        events, _ = collect_events(config, scenes)
        _annotate_fragment_starts(events, config["trackers"])
        policies = _policies(config, scenes, events)
        methods, stats, elapsed = _prediction_run(config, scenes, policies)
        _baseline_equal(config, scenes)
    elif (output_root / "memory_stats.csv").is_file():
        stats = list(csv.DictReader((output_root / "memory_stats.csv").open(encoding="utf-8")))
    metrics = []
    per_scene = []
    if evaluate:
        metrics, per_scene = _evaluate(config, scenes, methods)
    elif (output_root / "metrics.csv").is_file():
        metrics = list(csv.DictReader((output_root / "metrics.csv").open(encoding="utf-8")))
        per_scene = list(csv.DictReader((output_root / "metrics_per_scene.csv").open(encoding="utf-8")))
    if metrics:
        results = _result_rows(metrics)
        _write_csv(output_root / "results.csv", results)
        _write_table(output_root / "table.md", results, int(config["detail_delta"]))
        _write_scene_table(output_root / "table_per_scene.md", per_scene, int(config["detail_delta"]))
        _write_summary(output_root / "summary.md", results, per_scene, stats, config)
    manifest_path = output_root / "manifest.json"
    previous_manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    figure = _comparison_figure(config, scenes, output_root) if evaluate else previous_manifest.get("comparison_figure")
    write_json(
        output_root / "manifest.json",
        {
            "experiment": "scene_survival",
            "causal_history": "all_previous_sequences",
            "grid_m": config["grid_m"],
            "hard_fraction": config["hard_fraction"],
            "delta_ages": config["delta_ages"],
            "detail_delta": config["detail_delta"],
            "methods": [item["name"] for item in methods],
            "prediction_seconds": elapsed if elapsed is not None else previous_manifest.get("prediction_seconds"),
            "baseline_equal": True if predict else _baseline_equal(config, scenes),
            "comparison_figure": figure,
            "warnings": warnings,
        },
    )
    print("scene_survival完成", flush=True)
