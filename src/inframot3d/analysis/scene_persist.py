import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from inframot3d.analysis.scene_difficulty import (
    MAP_ROLE,
    _bin_index,
    _grid_shape,
    build_scenes,
    collect_events,
)
from inframot3d.io import write_json


COUNT_KEYS = ("gt", "miss", "idsw", "frag")


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


def _round(value):
    if value is None or not math.isfinite(float(value)):
        return ""
    return round(float(value), 6)


def _minimum(config, cell):
    values = config["min_observations"]
    if isinstance(values, dict):
        key = str(int(cell)) if float(cell).is_integer() else str(float(cell))
        if key in values:
            return int(values[key])
        key = "%.1f" % float(cell)
        if key in values:
            return int(values[key])
        raise KeyError("min_observations缺少网格%s" % cell)
    return int(values)


def _empty_counts(shape):
    return {key: np.zeros(shape, dtype=np.float64) for key in COUNT_KEYS}


def _annotate_fragment_starts(events, trackers):
    names = [item["name"] for item in trackers]
    groups = defaultdict(list)
    for event in events:
        groups[(event["sequence_id"], event["gt_id"])].append(event)
        for name in names:
            event["trackers"][name]["frag_start"] = 0
    for values in groups.values():
        ordered = sorted(values, key=lambda item: (int(item["timestamp"]), int(item["frame_index"])))
        for name in names:
            matched_before = False
            gap_start = None
            for event in ordered:
                if int(event["trackers"][name]["miss"]) == 0:
                    if matched_before and gap_start is not None:
                        gap_start["trackers"][name]["frag_start"] = 1
                    matched_before = True
                    gap_start = None
                elif matched_before and gap_start is None:
                    gap_start = event


def _event_index(events):
    grouped = defaultdict(list)
    for event in events:
        grouped[(event["scene_id"], event["sequence_id"])].append(event)
    return grouped


def _accumulate(records, tracker, bev, cell):
    shape = _grid_shape(bev, cell)
    counts = _empty_counts(shape)
    for event in records:
        index = _bin_index(event["x"], event["y"], bev, cell, shape)
        if index is None:
            continue
        values = event["trackers"][tracker]
        counts["gt"][index] += 1
        counts["miss"][index] += int(values["miss"])
        counts["idsw"][index] += int(values["idsw"])
        counts["frag"][index] += int(values["frag_start"])
    return counts


def _risk_map(counts, prior_strength, minimum):
    gt = counts["gt"]
    total_gt = float(gt.sum())
    global_rate = float(counts["miss"].sum() / total_gt) if total_gt > 0 else 0.0
    risk = (counts["miss"] + float(prior_strength) * global_rate) / (gt + float(prior_strength))
    risk = risk.astype(np.float64)
    risk[gt < int(minimum)] = np.nan
    return risk


def _cell_features(indices, counts, bev, cell):
    values = {}
    for index in indices:
        iy, ix = index
        x = float(bev["x_min"]) + (ix + 0.5) * float(cell)
        y = float(bev["y_min"]) + (iy + 0.5) * float(cell)
        values[index] = (math.hypot(x, y), math.log1p(float(counts["gt"][index])))
    return values


def _matched_normal(hard, candidates, counts, bev, cell):
    if not hard or not candidates:
        return []
    features = _cell_features(list(hard) + list(candidates), counts, bev, cell)
    ranges = np.asarray([value[0] for value in features.values()], dtype=np.float64)
    logs = np.asarray([value[1] for value in features.values()], dtype=np.float64)
    range_scale = max(float(np.std(ranges)), 1.0)
    log_scale = max(float(np.std(logs)), 0.2)
    remaining = set(candidates)
    chosen = []
    for source in hard:
        if not remaining:
            break
        source_range, source_log = features[source]
        target = min(
            remaining,
            key=lambda item: abs(features[item][0] - source_range) / range_scale
            + abs(features[item][1] - source_log) / log_scale,
        )
        chosen.append(target)
        remaining.remove(target)
    return chosen


def _select_regions(risk, counts, bev, cell, fraction):
    valid = [tuple(int(value) for value in item) for item in np.argwhere(np.isfinite(risk))]
    if len(valid) < 4:
        return [], []
    width = max(1, int(math.ceil(float(fraction) * len(valid))))
    hard = sorted(valid, key=lambda item: float(risk[item]), reverse=True)[:width]
    hard_set = set(hard)
    candidates = [item for item in valid if item not in hard_set]
    normal = _matched_normal(hard, candidates, counts, bev, cell)
    return hard, normal


def _pool(counts, indices):
    result = {key: 0.0 for key in COUNT_KEYS}
    for index in indices:
        for key in COUNT_KEYS:
            result[key] += float(counts[key][index])
    return result


def _rate(events, total, scale=1.0):
    if float(total) <= 0:
        return None
    return float(events) / float(total) * float(scale)


def _risk_ratio(hard_events, hard_total, normal_events, normal_total):
    if float(hard_total) <= 0 or float(normal_total) <= 0:
        return None
    hard_rate = (float(hard_events) + 0.5) / (float(hard_total) + 1.0)
    normal_rate = (float(normal_events) + 0.5) / (float(normal_total) + 1.0)
    return hard_rate / normal_rate


def _correlation(history_risk, future_counts, prior_strength, minimum):
    future_risk = _risk_map(future_counts, prior_strength, minimum)
    mask = np.isfinite(history_risk) & np.isfinite(future_risk)
    count = int(mask.sum())
    if count < 3:
        return count, None
    left = history_risk[mask]
    right = future_risk[mask]
    if np.unique(left).size < 2 or np.unique(right).size < 2:
        return count, None
    return count, float(spearmanr(left, right).statistic)


def _fold_row(scene_id, tracker, cell, history_label, history_ids, target_id, history_records, target_records, config):
    bev = config["bev"]
    minimum = _minimum(config, cell)
    prior = float(config["prior_strength"])
    history_counts = _accumulate(history_records, tracker, bev, cell)
    future_counts = _accumulate(target_records, tracker, bev, cell)
    history_risk = _risk_map(history_counts, prior, minimum)
    hard, normal = _select_regions(history_risk, history_counts, bev, cell, config["hard_fraction"])
    if not hard or not normal:
        return None
    hard_values = _pool(future_counts, hard)
    normal_values = _pool(future_counts, normal)
    common, correlation = _correlation(history_risk, future_counts, prior, minimum)
    hard_identity = hard_values["idsw"] + hard_values["frag"]
    normal_identity = normal_values["idsw"] + normal_values["frag"]
    return {
        "scene_id": scene_id,
        "tracker": tracker,
        "grid_m": float(cell),
        "history_length": str(history_label),
        "history_sequences": ";".join(history_ids),
        "target_sequence": target_id,
        "history_gt": int(history_counts["gt"].sum()),
        "future_gt": int(future_counts["gt"].sum()),
        "valid_history_cells": int(np.isfinite(history_risk).sum()),
        "hard_cells": len(hard),
        "normal_cells": len(normal),
        "common_cells": common,
        "spearman": _round(correlation),
        "hard_gt": int(hard_values["gt"]),
        "normal_gt": int(normal_values["gt"]),
        "hard_miss": int(hard_values["miss"]),
        "normal_miss": int(normal_values["miss"]),
        "hard_idsw": int(hard_values["idsw"]),
        "normal_idsw": int(normal_values["idsw"]),
        "hard_fragment": int(hard_values["frag"]),
        "normal_fragment": int(normal_values["frag"]),
        "hard_miss_rate": _round(_rate(hard_values["miss"], hard_values["gt"])),
        "normal_miss_rate": _round(_rate(normal_values["miss"], normal_values["gt"])),
        "miss_risk_ratio": _round(
            _risk_ratio(hard_values["miss"], hard_values["gt"], normal_values["miss"], normal_values["gt"])
        ),
        "hard_identity_per_1k": _round(_rate(hard_identity, hard_values["gt"], 1000.0)),
        "normal_identity_per_1k": _round(_rate(normal_identity, normal_values["gt"], 1000.0)),
        "identity_risk_ratio": _round(
            _risk_ratio(hard_identity, hard_values["gt"], normal_identity, normal_values["gt"])
        ),
    }


def _folds(scenes, grouped, config):
    rows = []
    trackers = [item["name"] for item in config["trackers"]]
    for scene in scenes:
        sequence_ids = [item["sequence_id"] for item in scene["sequences"]]
        for target_index in range(1, len(sequence_ids)):
            target_id = sequence_ids[target_index]
            target_records = grouped[(scene["scene_id"], target_id)]
            for history_length in config["history_lengths"]:
                if str(history_length).lower() == "all":
                    history_ids = sequence_ids[:target_index]
                    label = "all"
                else:
                    width = int(history_length)
                    if target_index < width:
                        continue
                    history_ids = sequence_ids[target_index - width : target_index]
                    label = str(width)
                history_records = []
                for sequence_id in history_ids:
                    history_records.extend(grouped[(scene["scene_id"], sequence_id)])
                for tracker in trackers:
                    for cell in config["grid_sizes"]:
                        row = _fold_row(
                            scene["scene_id"],
                            tracker,
                            float(cell),
                            label,
                            history_ids,
                            target_id,
                            history_records,
                            target_records,
                            config,
                        )
                        if row is not None:
                            rows.append(row)
    return rows


def _sums(rows):
    keys = (
        "hard_gt",
        "normal_gt",
        "hard_miss",
        "normal_miss",
        "hard_idsw",
        "normal_idsw",
        "hard_fragment",
        "normal_fragment",
    )
    return {key: sum(float(row[key]) for row in rows) for key in keys}


def _summary_values(rows):
    values = _sums(rows)
    hard_identity = values["hard_idsw"] + values["hard_fragment"]
    normal_identity = values["normal_idsw"] + values["normal_fragment"]
    correlations = [float(row["spearman"]) for row in rows if row["spearman"] != ""]
    return {
        "hard_miss_rate": _rate(values["hard_miss"], values["hard_gt"]),
        "normal_miss_rate": _rate(values["normal_miss"], values["normal_gt"]),
        "miss_risk_ratio": _risk_ratio(
            values["hard_miss"], values["hard_gt"], values["normal_miss"], values["normal_gt"]
        ),
        "hard_identity_per_1k": _rate(hard_identity, values["hard_gt"], 1000.0),
        "normal_identity_per_1k": _rate(normal_identity, values["normal_gt"], 1000.0),
        "identity_risk_ratio": _risk_ratio(hard_identity, values["hard_gt"], normal_identity, values["normal_gt"]),
        "spearman_mean": float(np.mean(correlations)) if correlations else None,
        "spearman_median": float(np.median(correlations)) if correlations else None,
    }


def _scene_wins(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["scene_id"]].append(row)
    miss_wins = 0
    identity_wins = 0
    usable = 0
    for current in grouped.values():
        values = _sums(current)
        if values["hard_gt"] <= 0 or values["normal_gt"] <= 0:
            continue
        usable += 1
        hard_miss = values["hard_miss"] / values["hard_gt"]
        normal_miss = values["normal_miss"] / values["normal_gt"]
        if hard_miss > normal_miss:
            miss_wins += 1
        hard_identity = (values["hard_idsw"] + values["hard_fragment"]) / values["hard_gt"]
        normal_identity = (values["normal_idsw"] + values["normal_fragment"]) / values["normal_gt"]
        if hard_identity > normal_identity:
            identity_wins += 1
    return usable, miss_wins, identity_wins


def _bootstrap(rows, repeats, rng):
    by_scene = defaultdict(list)
    for row in rows:
        by_scene[row["scene_id"]].append(row)
    scenes = sorted(by_scene)
    miss = []
    identity = []
    if not scenes:
        return None, None, None, None
    for _ in range(int(repeats)):
        sampled = rng.choice(scenes, size=len(scenes), replace=True)
        current = []
        for scene_id in sampled:
            current.extend(by_scene[str(scene_id)])
        values = _summary_values(current)
        if values["miss_risk_ratio"] is not None:
            miss.append(values["miss_risk_ratio"])
        if values["identity_risk_ratio"] is not None:
            identity.append(values["identity_risk_ratio"])
    miss_ci = np.percentile(miss, [2.5, 97.5]) if miss else [np.nan, np.nan]
    identity_ci = np.percentile(identity, [2.5, 97.5]) if identity else [np.nan, np.nan]
    return float(miss_ci[0]), float(miss_ci[1]), float(identity_ci[0]), float(identity_ci[1])


def _summaries(rows, config):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["tracker"], float(row["grid_m"]), row["history_length"])].append(row)
    order = {str(value).lower(): index for index, value in enumerate(config["history_lengths"])}
    rng = np.random.default_rng(int(config["seed"]))
    output = []
    for key in sorted(grouped, key=lambda item: (item[0], item[1], order.get(item[2], 999))):
        tracker, cell, history_length = key
        current = grouped[key]
        values = _summary_values(current)
        usable_scenes, miss_wins, identity_wins = _scene_wins(current)
        miss_low, miss_high, identity_low, identity_high = _bootstrap(current, config["bootstrap_repeats"], rng)
        output.append(
            {
                "tracker": tracker,
                "grid_m": cell,
                "history_length": history_length,
                "n_scenes": len({row["scene_id"] for row in current}),
                "usable_scenes": usable_scenes,
                "miss_higher_scenes": miss_wins,
                "identity_higher_scenes": identity_wins,
                "n_folds": len(current),
                "hard_gt": int(sum(int(row["hard_gt"]) for row in current)),
                "normal_gt": int(sum(int(row["normal_gt"]) for row in current)),
                "hard_miss_rate": _round(values["hard_miss_rate"]),
                "normal_miss_rate": _round(values["normal_miss_rate"]),
                "miss_risk_ratio": _round(values["miss_risk_ratio"]),
                "miss_rr_ci_low": _round(miss_low),
                "miss_rr_ci_high": _round(miss_high),
                "hard_identity_per_1k": _round(values["hard_identity_per_1k"]),
                "normal_identity_per_1k": _round(values["normal_identity_per_1k"]),
                "identity_risk_ratio": _round(values["identity_risk_ratio"]),
                "identity_rr_ci_low": _round(identity_low),
                "identity_rr_ci_high": _round(identity_high),
                "spearman_mean": _round(values["spearman_mean"]),
                "spearman_median": _round(values["spearman_median"]),
            }
        )
    return output


def _wilson(events, total):
    if total <= 0:
        return 0.0, 0.0, 0.0
    value = float(events) / float(total)
    z = 1.96
    denominator = 1.0 + z * z / total
    center = (value + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(value * (1.0 - value) / total + z * z / (4.0 * total * total)) / denominator
    return value, max(0.0, center - radius), min(1.0, center + radius)


def _bar(axis, labels, events, totals, scale, ylabel, colors):
    values = []
    lower = []
    upper = []
    for event, total in zip(events, totals):
        value, low, high = _wilson(float(event), float(total))
        values.append(value * scale)
        lower.append((value - low) * scale)
        upper.append((high - value) * scale)
    positions = np.arange(len(labels))
    axis.bar(positions, values, color=colors, width=0.62, yerr=np.asarray([lower, upper]), capsize=4)
    axis.set_xticks(positions, labels)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    for position, value in zip(positions, values):
        axis.text(position, value, "%.2f" % value, ha="center", va="bottom", fontsize=9)


def _figure(scene, grouped, tracker, cell, config, output_root):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    sequence_ids = [item["sequence_id"] for item in scene["sequences"]]
    cut = max(1, min(len(sequence_ids) - 1, int(math.floor(len(sequence_ids) * float(config["figure_history_ratio"])))))
    history_ids = sequence_ids[:cut]
    future_ids = sequence_ids[cut:]
    history_records = []
    future_records = []
    for sequence_id in history_ids:
        history_records.extend(grouped[(scene["scene_id"], sequence_id)])
    for sequence_id in future_ids:
        future_records.extend(grouped[(scene["scene_id"], sequence_id)])
    bev = config["bev"]
    history_counts = _accumulate(history_records, tracker, bev, cell)
    future_counts = _accumulate(future_records, tracker, bev, cell)
    risk = _risk_map(history_counts, config["prior_strength"], _minimum(config, cell))
    hard, normal = _select_regions(risk, history_counts, bev, cell, config["hard_fraction"])
    if not hard or not normal:
        return None
    hard_values = _pool(future_counts, hard)
    normal_values = _pool(future_counts, normal)
    shape = risk.shape
    figure, axes = plt.subplots(1, 3, figsize=(14.6, 4.35), gridspec_kw={"width_ratios": [1.8, 0.8, 0.8]})
    axis = axes[0]
    image = axis.imshow(
        np.ma.masked_invalid(risk),
        origin="lower",
        extent=[float(bev["x_min"]), float(bev["x_max"]), float(bev["y_min"]), float(bev["y_max"])],
        aspect="equal",
        cmap="magma",
    )
    hard_grid = np.zeros(shape, dtype=np.float64)
    for index in hard:
        hard_grid[index] = 1.0
    xs = float(bev["x_min"]) + (np.arange(shape[1]) + 0.5) * float(cell)
    ys = float(bev["y_min"]) + (np.arange(shape[0]) + 0.5) * float(cell)
    axis.contour(xs, ys, hard_grid, levels=[0.5], colors=["#00ffff"], linewidths=1.5)
    axis.scatter([event["x"] for event in future_records], [event["y"] for event in future_records], s=2, c="#b7b7b7", alpha=0.18)
    misses = [event for event in future_records if int(event["trackers"][tracker]["miss"]) == 1]
    switches = [event for event in future_records if int(event["trackers"][tracker]["idsw"]) == 1]
    fragments = [event for event in future_records if int(event["trackers"][tracker]["frag_start"]) == 1]
    if misses:
        axis.scatter([item["x"] for item in misses], [item["y"] for item in misses], marker="x", s=20, c="#ff3b30", linewidths=0.8)
    if switches:
        axis.scatter([item["x"] for item in switches], [item["y"] for item in switches], marker="*", s=70, c="#2f7ed8", edgecolors="white", linewidths=0.4)
    if fragments:
        axis.scatter(
            [item["x"] for item in fragments],
            [item["y"] for item in fragments],
            marker="o",
            s=45,
            facecolors="none",
            edgecolors="#ff9f0a",
            linewidths=1.2,
        )
    axis.scatter([0.0], [0.0], marker="^", s=80, c="white", edgecolors="black", linewidths=0.8)
    axis.set_xlim(float(bev["x_min"]) - 2.5, float(bev["x_max"]))
    axis.set_ylim(float(bev["y_min"]), float(bev["y_max"]))
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_title("Historical risk with future failures")
    figure.colorbar(image, ax=axis, fraction=0.046, label="Historical miss risk")
    legend = [
        Line2D([0], [0], color="#00ffff", lw=1.5, label="Historical hard region"),
        Line2D([0], [0], marker="x", color="#ff3b30", lw=0, label="Future miss"),
        Line2D([0], [0], marker="*", color="#2f7ed8", lw=0, markersize=9, label="Future ID switch"),
        Line2D([0], [0], marker="o", markerfacecolor="none", markeredgecolor="#ff9f0a", lw=0, label="Future fragment start"),
    ]
    axis.legend(handles=legend, loc="upper left", fontsize=8, framealpha=0.9)
    colors = ["#d62728", "#8c8c8c"]
    _bar(
        axes[1],
        ["Hard", "Matched\nnormal"],
        [hard_values["miss"], normal_values["miss"]],
        [hard_values["gt"], normal_values["gt"]],
        100.0,
        "Future miss rate (%)",
        colors,
    )
    _bar(
        axes[2],
        ["Hard", "Matched\nnormal"],
        [hard_values["idsw"] + hard_values["frag"], normal_values["idsw"] + normal_values["frag"]],
        [hard_values["gt"], normal_values["gt"]],
        1000.0,
        "Identity failures / 1k GT",
        colors,
    )
    title = "%s | %s | %.0f m grid" % (scene["scene_id"], tracker, cell)
    figure.suptitle(title, fontsize=13)
    figure.tight_layout()
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = "%s_%s_%sm" % (scene["scene_id"], tracker.lower().replace("-", ""), int(cell))
    png_path = figure_dir / (stem + ".png")
    pdf_path = figure_dir / (stem + ".pdf")
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    metadata = {
        "map_role": MAP_ROLE,
        "scene_id": scene["scene_id"],
        "tracker": tracker,
        "grid_m": cell,
        "history_sequences": history_ids,
        "future_sequences": future_ids,
        "hard_cells": [list(item) for item in hard],
        "normal_cells": [list(item) for item in normal],
        "hard_future_gt": int(hard_values["gt"]),
        "normal_future_gt": int(normal_values["gt"]),
        "hard_future_miss": int(hard_values["miss"]),
        "normal_future_miss": int(normal_values["miss"]),
        "hard_future_identity": int(hard_values["idsw"] + hard_values["frag"]),
        "normal_future_identity": int(normal_values["idsw"] + normal_values["frag"]),
    }
    write_json(figure_dir / (stem + ".json"), metadata)
    return str(png_path.relative_to(output_root))


def _fmt(value, digits=3):
    if value == "" or value is None:
        return "--"
    return ("%%.%df" % digits) % float(value)


def _write_table(path, summaries, tracker):
    rows = [row for row in summaries if row["tracker"] == tracker]
    lines = [
        "| Grid | History | Scenes | Folds | Hard miss | Normal miss | Miss RR (95% CI) | Hard identity/1k | Normal identity/1k | Identity RR (95% CI) | Spearman |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        miss_ci = "%s (%s-%s)" % (
            _fmt(row["miss_risk_ratio"], 2),
            _fmt(row["miss_rr_ci_low"], 2),
            _fmt(row["miss_rr_ci_high"], 2),
        )
        identity_ci = "%s (%s-%s)" % (
            _fmt(row["identity_risk_ratio"], 2),
            _fmt(row["identity_rr_ci_low"], 2),
            _fmt(row["identity_rr_ci_high"], 2),
        )
        lines.append(
            "| %.0f m | %s | %d | %d | %.2f%% | %.2f%% | %s | %s | %s | %s | %s |"
            % (
                float(row["grid_m"]),
                row["history_length"],
                int(row["n_scenes"]),
                int(row["n_folds"]),
                float(row["hard_miss_rate"]) * 100.0,
                float(row["normal_miss_rate"]) * 100.0,
                miss_ci,
                _fmt(row["hard_identity_per_1k"], 2),
                _fmt(row["normal_identity_per_1k"], 2),
                identity_ci,
                _fmt(row["spearman_mean"], 3),
            )
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(path, scenes, rows, summaries, figures, config):
    tracker = config["primary_tracker"]
    primary = [
        row
        for row in summaries
        if row["tracker"] == tracker
        and float(row["grid_m"]) == float(config["primary_grid"])
        and row["history_length"] == "all"
    ]
    lines = [
        "# Scene persistence experiment",
        "",
        "历史sequence用于构建离线风险图，目标sequence只用于未来评估。hard区域选择不读取目标sequence。",
        "",
        "## 数据",
        "",
        "- 场景数%d" % len(scenes),
        "- fold数%d" % len(rows),
        "- 主tracker%s" % tracker,
        "- 主网格%.0fm" % float(config["primary_grid"]),
        "- hard比例%.0f%%" % (float(config["hard_fraction"]) * 100.0),
        "",
        "## 主结果",
        "",
    ]
    if primary:
        row = primary[0]
        lines.extend(
            [
                "- future hard miss %.2f%%，matched normal miss %.2f%%" % (
                    float(row["hard_miss_rate"]) * 100.0,
                    float(row["normal_miss_rate"]) * 100.0,
                ),
                "- miss risk ratio %s，scene bootstrap 95%% CI %s-%s" % (
                    _fmt(row["miss_risk_ratio"], 2),
                    _fmt(row["miss_rr_ci_low"], 2),
                    _fmt(row["miss_rr_ci_high"], 2),
                ),
                "- identity risk ratio %s，scene bootstrap 95%% CI %s-%s" % (
                    _fmt(row["identity_risk_ratio"], 2),
                    _fmt(row["identity_rr_ci_low"], 2),
                    _fmt(row["identity_rr_ci_high"], 2),
                ),
                "- %d/%d个场景的future miss更高，%d/%d个场景的future identity failure更高" % (
                    int(row["miss_higher_scenes"]),
                    int(row["usable_scenes"]),
                    int(row["identity_higher_scenes"]),
                    int(row["usable_scenes"]),
                ),
            ]
        )
    lines.extend(["", "## 图", ""])
    lines.extend(["- `%s`" % item for item in figures])
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "该实验只验证离线空间持久性，输出不能在val或test推理时读取。fragment位置取连续丢失的第一帧。",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config):
    output_root = Path(config["project"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    scenes, metadata, warnings = build_scenes(config)
    events, _ = collect_events(config, scenes)
    _annotate_fragment_starts(events, config["trackers"])
    grouped = _event_index(events)
    rows = _folds(scenes, grouped, config)
    summaries = _summaries(rows, config)
    _write_csv(output_root / "folds.csv", rows)
    _write_csv(output_root / "table.csv", summaries)
    primary_tracker = config["primary_tracker"]
    _write_table(output_root / "table.md", summaries, primary_tracker)
    requested = set(config.get("figure_scenes") or [scene["scene_id"] for scene in scenes])
    figures = []
    for scene in scenes:
        if scene["scene_id"] not in requested:
            continue
        path = _figure(scene, grouped, primary_tracker, float(config["primary_grid"]), config, output_root)
        if path is not None:
            figures.append(path)
    _write_summary(output_root / "summary.md", scenes, rows, summaries, figures, config)
    write_json(
        output_root / "manifest.json",
        {
            "map_role": MAP_ROLE,
            "experiment": "scene_persist",
            "scenes": [scene["scene_id"] for scene in scenes],
            "trackers": [item["name"] for item in config["trackers"]],
            "grid_sizes": config["grid_sizes"],
            "history_lengths": config["history_lengths"],
            "warnings": warnings,
            "metadata_rows": len(metadata),
            "fold_rows": len(rows),
            "figures": figures,
        },
    )
    print("scene_persist完成 fold %d 图 %d" % (len(rows), len(figures)), flush=True)
