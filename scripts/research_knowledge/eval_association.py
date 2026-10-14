"""固定 GRAE 候选集合上的离线关联区分

GT ID 只用于标记正负样本，不进入候选分数
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inframot3d.analysis.grae_geometry_reliability import IdentityBook, _equivalent, _load_tracker, label_from_ids
from inframot3d.analysis.scene_difficulty import _keep_vehicle, _match
from inframot3d.evaluation.protocols.v2xseq import V2XSeqProtocol
from inframot3d.io import read_jsonl

from common import (
    LaneIndex,
    assert_bank_isolation,
    blend_futures,
    build_catalog,
    cell_index,
    forbidden_sequences,
    from_local,
    interpolate_future,
    load_bank,
    load_cached_tracks,
    load_config,
    load_hard_regions,
    make_sample,
    output_dir,
    project_path,
    region_name,
    sample_window,
    scene_of,
    select_topk,
    world_pose,
    write_csv,
    write_json,
)

METHODS = ("grae", "cv", "kv", "reliability", "oracle")
DEPLOYABLE = {"grae", "cv", "kv", "reliability"}


def _auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    keep = np.isfinite(scores)
    labels = labels[keep]
    scores = scores[keep]
    if labels.size == 0 or int(labels.min()) == int(labels.max()):
        return None
    return float(roc_auc_score(labels, scores))


def _gap_name(seconds, edges):
    for left, right in zip(edges[:-1], edges[1:]):
        if left <= seconds < right:
            return "%g-%g" % (left, right)
    return "na"


def _read_reliability(path):
    import csv

    table = {}
    with Path(path).open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if int(row["confident"]) != 1 or row["miss_rate"] == "":
                continue
            key = (row["fold"], row["intersection"], row["class_name"], int(row["iy"]), int(row["ix"]))
            table[key] = float(row["miss_rate"])
    return table


def _gt_lookup(tracks):
    grouped = defaultdict(list)
    for track in tracks:
        grouped[(track["sequence_id"], track["track_id"])].append(track)
    return grouped


def _oracle_xy(grouped, sequence_id, gt_id, stamp):
    if gt_id is None:
        return None
    best = None
    best_dt = 0.15
    for track in grouped.get((sequence_id, str(gt_id)), []):
        times = track["time"]
        index = int(np.argmin(np.abs(times - stamp)))
        dt = abs(float(times[index] - stamp))
        if dt <= best_dt:
            best_dt = dt
            best = track["lidar_xy"][index]
    return None if best is None else np.asarray(best, dtype=np.float64)


def _two_stage(score, high_mask, class_ok, track_ids, input_indices, high_limit, low_limit):
    import lap

    count_det, count_track = score.shape
    assigned = {}
    if count_det == 0 or count_track == 0:
        return assigned
    valid = class_ok & np.isfinite(score)
    remain = np.ones(count_track, dtype=bool)
    high_idx = np.flatnonzero(np.asarray(high_mask, dtype=bool))
    if len(high_idx):
        sub = np.where(valid[high_idx], score[high_idx], 0.0)
        _, _, columns = lap.lapjv(1.0 - sub, extend_cost=True, cost_limit=float(high_limit))
        for track_index, det_local in enumerate(columns):
            if det_local >= 0:
                assigned[int(track_ids[track_index])] = int(input_indices[high_idx[det_local]])
                remain[track_index] = False
    low_idx = np.flatnonzero(~np.asarray(high_mask, dtype=bool))
    remain_idx = np.flatnonzero(remain)
    if len(low_idx) and len(remain_idx):
        block = score[np.ix_(low_idx, remain_idx)]
        ok = valid[np.ix_(low_idx, remain_idx)]
        sub = np.where(ok, block, 0.0)
        _, _, columns = lap.lapjv(1.0 - sub, extend_cost=True, cost_limit=float(low_limit))
        for local_track, det_local in enumerate(columns):
            if det_local >= 0:
                track_index = int(remain_idx[local_track])
                assigned[int(track_ids[track_index])] = int(input_indices[low_idx[det_local]])
    return assigned


def _kv_prediction(history, now, bank, allowed, config, lanes, rotation):
    if bank is None or len(history) < 2 or history[-1]["class_name"] not in set(config["vehicle_classes"]):
        return None
    pseudo = {
        "time": np.asarray([item["time"] for item in history], dtype=np.float64),
        "world_xy": np.stack([item["world_xy"] for item in history]),
        "world_yaw": np.unwrap([item["world_yaw"] for item in history]),
        "lidar_xy": np.stack([item["lidar_xy"] for item in history]),
        "intersection": history[-1]["intersection"],
    }
    anchor = float(pseudo["time"][-1])
    hist_index, hist_ok, _, _ = sample_window(
        pseudo, anchor, float(config["history_s"]), float(config["future_s"]), float(config["dt_s"]), float(config["sample_tolerance_s"])
    )
    if not np.all(hist_ok) or int(hist_index[-1]) != len(history) - 1:
        return None
    sample = make_sample(pseudo, len(history) - 1, hist_index, lanes, config)
    query = {
        "origin": sample["origin"],
        "heading": sample["heading"],
        "hist": sample["hist"],
        "speed": sample["speed"],
        "region": sample["region"],
        "class_name": history[-1]["class_name"],
    }
    chosen, costs = select_topk(bank, query, allowed, config)
    if chosen.size == 0:
        return None
    future = blend_futures(bank["future"][chosen], costs)
    local = interpolate_future(future, now - anchor, float(config["dt_s"]))
    if local is None:
        return None
    delta_world = from_local(local, np.zeros(2), sample["heading"])
    delta_lidar = rotation[:2, :2].T @ delta_world
    return history[-1]["lidar_xy"] + delta_lidar


def _append_case(bucket, item, limit, better):
    bucket.append(item)
    bucket.sort(key=lambda row: row["gain"], reverse=better)
    del bucket[limit:]


def replay_sequence(config, tracker, protocol, catalog, banks, reliability, hard, gt_tracks, lanes, sequence_id, cases):
    meta = catalog[sequence_id]
    converted = project_path(config, config["project"]["converted_root"])
    detection_root = project_path(config, config["detection_root"])
    gt_rows = list(read_jsonl(converted / "sequences" / ("%s.jsonl" % sequence_id)))
    det_rows = list(read_jsonl(detection_root / ("%s.jsonl" % sequence_id)))
    bank = banks.get("fold%s_%s" % (meta["fold"], meta["intersection"]))
    banned = forbidden_sequences(catalog, sequence_id)
    allowed = [] if bank is None else [str(name) for name in bank["sequence_names"] if str(name) not in banned]
    if bank is not None:
        assert_bank_isolation(allowed, sequence_id, catalog)
    scene = scene_of(hard, meta["intersection"])
    edges = [float(item) for item in config["association"]["gap_edges_s"]]
    alpha = float(config["association"]["alpha"])
    high_limit = float(config["association"]["high_cost_limit"])
    low_limit = float(config["association"]["low_cost_limit"])
    tracker.reset()
    tracker.association_mode = "fusion"
    tracker.high_cost_limit = high_limit
    tracker.low_cost_limit = low_limit
    tracker.enable_debug(True)
    book = IdentityBook()
    history = defaultdict(list)
    pairs = []
    tracks = []
    predictions = []
    for gt_row, det_row in zip(gt_rows, det_rows):
        if str(det_row["frame_id"]) != str(gt_row["frame_id"]):
            raise ValueError("帧未对齐 %s" % sequence_id)
        now = int(det_row["timestamp"]) / 1e6
        cleaned = [
            {"class_name": item["class_name"], "score": float(item["score"]), "box": [float(value) for value in item["box"]]}
            for item in det_row["objects"]
        ]
        outputs = tracker.update(cleaned, now, det_row["frame_id"])
        predictions.append(
            {
                "sequence_id": sequence_id,
                "frame_index": det_row["frame_index"],
                "frame_id": det_row["frame_id"],
                "timestamp": det_row["timestamp"],
                "objects": outputs,
            }
        )
        snapshot = tracker.last_debug
        group = snapshot["groups"][0]
        candidates = group.get("candidate_detections", [])
        pre_tracks = group.get("pre_tracks", [])
        association = group.get("association", {})
        assignments = group.get("assignments", [])
        created = group.get("created", [])
        gt_items = protocol.filter_gt(gt_row["objects"])
        indexed = [(index, item) for index, item in enumerate(det_row["objects"]) if _keep_vehicle(protocol, item)]
        matches = _match(
            [item["box"] for item in gt_items],
            [item["box"] for _, item in indexed],
            config["match_iou"],
            config["match_center_gate_m"],
        )
        id_counts = Counter(str(item["source_track_id"]) for item in gt_items)
        det_gt = {}
        for gt_index, local_index, _ in matches:
            gid = str(gt_items[gt_index]["source_track_id"])
            if id_counts[gid] > 1:
                continue
            det_gt[int(indexed[local_index][0])] = gid
        pre_ids = [int(item["track_id"]) for item in pre_tracks]
        owners = defaultdict(list)
        for track_id in pre_ids:
            gt_id, conflict, _ = book.state(track_id)
            if gt_id is not None and not conflict:
                owners[gt_id].append(track_id)
        shared = {track_id for values in owners.values() if len(values) > 1 for track_id in values}
        if candidates and pre_tracks and association.get("affinity_matrix"):
            learned = np.asarray(association["affinity_matrix"], dtype=np.float64)
            det_xy = np.asarray([[item["box"][0], item["box"][1]] for item in candidates], dtype=np.float64)
            det_scores = np.asarray([float(det_row["objects"][int(item["input_index"])]["score"]) for item in candidates])
            input_indices = np.asarray([int(item["input_index"]) for item in candidates], dtype=np.int64)
            track_ids = np.asarray(pre_ids, dtype=np.int64)
            class_ok = np.zeros(learned.shape, dtype=bool)
            zero_xy = np.zeros((len(pre_tracks), 2), dtype=np.float64)
            cv_xy = np.zeros_like(zero_xy)
            kv_xy = np.full_like(zero_xy, np.nan)
            oracle_track = np.full_like(zero_xy, np.nan)
            covered = np.zeros(len(pre_tracks), dtype=bool)
            oracle_ok = np.zeros(len(pre_tracks), dtype=bool)
            gap_s = np.zeros(len(pre_tracks), dtype=np.float64)
            for track_pos, track in enumerate(pre_tracks):
                track_id = int(track["track_id"])
                zero_xy[track_pos] = track["predicted_xy"]
                past = history.get(track_id, [])
                if len(past) >= 2:
                    dt = past[-1]["time"] - past[-2]["time"]
                    velocity = (past[-1]["lidar_xy"] - past[-2]["lidar_xy"]) / dt if dt > 1e-3 else np.zeros(2)
                    cv_xy[track_pos] = past[-1]["lidar_xy"] + velocity * (now - past[-1]["time"])
                elif past:
                    cv_xy[track_pos] = past[-1]["lidar_xy"]
                else:
                    cv_xy[track_pos] = zero_xy[track_pos]
                if past:
                    gap_s[track_pos] = now - past[-1]["time"]
                prediction = _kv_prediction(past, now, bank, allowed, config, lanes, meta["rotation"])
                if prediction is not None:
                    kv_xy[track_pos] = prediction
                    covered[track_pos] = True
                gt_id, conflict, _ = book.state(track_id)
                if not conflict:
                    oracle = _oracle_xy(gt_tracks, sequence_id, gt_id, now)
                    if oracle is not None:
                        oracle_track[track_pos] = oracle
                        oracle_ok[track_pos] = True
                for det_pos, detection in enumerate(candidates):
                    class_ok[det_pos, track_pos] = int(detection["class_index"]) == int(track["class_index"])
            dist_zero = np.linalg.norm(det_xy[:, None, :] - zero_xy[None, :, :], axis=-1)
            dist_cv = np.linalg.norm(det_xy[:, None, :] - cv_xy[None, :, :], axis=-1)
            dist_kv = dist_zero.copy()
            kv_valid = np.isfinite(kv_xy).all(axis=1)
            if np.any(kv_valid):
                dist_kv[:, kv_valid] = np.linalg.norm(det_xy[:, None, :] - kv_xy[None, kv_valid, :], axis=-1)
            dist_oracle = np.full_like(dist_zero, np.nan)
            if np.any(oracle_ok):
                dist_oracle[:, oracle_ok] = np.linalg.norm(det_xy[:, None, :] - oracle_track[None, oracle_ok, :], axis=-1)
            fusion_zero = 0.5 * learned + 0.5 * np.exp(-dist_zero)
            scores = {
                "grae": fusion_zero,
                "cv": 0.5 * learned + 0.5 * np.exp(-dist_cv),
                "kv": 0.5 * learned + 0.5 * np.exp(-dist_kv),
                "oracle": 0.5 * learned + 0.5 * np.exp(-dist_oracle),
            }
            reliability_weight = np.ones(len(candidates), dtype=np.float64)
            for det_pos, detection in enumerate(candidates):
                raw = det_row["objects"][int(detection["input_index"])]
                index, _ = cell_index(raw["box"][0], raw["box"][1], scene["bev"], scene["grid_m"])
                if index is None:
                    continue
                rate = reliability.get((meta["fold"], meta["intersection"], raw["class_name"], index[0], index[1]))
                if rate is not None:
                    reliability_weight[det_pos] = 1.0 - rate
            scores["reliability"] = fusion_zero * reliability_weight[:, None]
            high_mask = det_scores >= alpha
            assigned = {
                name: _two_stage(value, high_mask, class_ok, track_ids, input_indices, high_limit, low_limit)
                for name, value in scores.items()
            }
            labels = np.full(learned.shape, -1, dtype=np.int8)
            for det_pos, detection in enumerate(candidates):
                raw_index = int(detection["input_index"])
                for track_pos, track_id in enumerate(track_ids):
                    if not class_ok[det_pos, track_pos]:
                        continue
                    gt_id, conflict, _ = book.state(int(track_id))
                    label, _ = label_from_ids(
                        det_gt.get(raw_index), gt_id, conflict, int(track_id) in shared
                    )
                    if label == "positive":
                        labels[det_pos, track_pos] = 1
                    elif label == "negative":
                        labels[det_pos, track_pos] = 0
            for track_pos, track_id in enumerate(track_ids):
                known = class_ok[:, track_pos] & (labels[:, track_pos] >= 0)
                positive = class_ok[:, track_pos] & (labels[:, track_pos] == 1)
                negative = class_ok[:, track_pos] & (labels[:, track_pos] == 0)
                if not np.any(positive):
                    continue
                positive_inputs = set(int(value) for value in input_indices[positive])
                area = region_name(hard, meta["intersection"], float(zero_xy[track_pos, 0]), float(zero_xy[track_pos, 1]))
                row = {
                    "sequence_id": sequence_id,
                    "frame_id": str(det_row["frame_id"]),
                    "track_id": int(track_id),
                    "region": area,
                    "gap": _gap_name(float(gap_s[track_pos]), edges),
                    "covered": int(covered[track_pos]),
                    "oracle_ok": int(oracle_ok[track_pos]),
                    "has_neg": int(np.any(negative)),
                    "pos_dist_grae": float(dist_zero[positive, track_pos].min()),
                    "pos_dist_cv": float(dist_cv[positive, track_pos].min()),
                    "pos_dist_kv": float(dist_kv[positive, track_pos].min()),
                    "pos_dist_oracle": "" if not oracle_ok[track_pos] else float(np.nanmin(dist_oracle[positive, track_pos])),
                }
                for name in METHODS:
                    values = scores[name][:, track_pos]
                    finite = class_ok[:, track_pos] & np.isfinite(values)
                    if not np.any(finite):
                        row["top1_%s" % name] = 0
                    else:
                        winner = int(np.argmax(np.where(finite, values, -np.inf)))
                        row["top1_%s" % name] = int(labels[winner, track_pos] == 1)
                    choice = assigned[name].get(int(track_id))
                    row["hit_%s" % name] = int(choice in positive_inputs)
                    if np.any(positive) and np.any(negative) and np.isfinite(values[positive]).any() and np.isfinite(values[negative]).any():
                        row["sep_%s" % name] = int(np.nanmax(values[positive]) > np.nanmax(values[negative]))
                        row["margin_%s" % name] = float(np.nanmax(values[positive]) - np.nanmax(values[negative]))
                    else:
                        row["sep_%s" % name] = ""
                        row["margin_%s" % name] = ""
                tracks.append(row)
                if row["covered"] and row["has_neg"] and row["top1_grae"] != row["top1_kv"]:
                    gain = row["pos_dist_grae"] - row["pos_dist_kv"]
                    kind = "success" if row["top1_kv"] == 1 and row["top1_grae"] == 0 else "failure"
                    _append_case(
                        cases[kind],
                        {
                            "kind": kind,
                            "gain": gain,
                            "sequence_id": sequence_id,
                            "frame_id": str(det_row["frame_id"]),
                            "intersection": meta["intersection"],
                            "region": area,
                            "gap": row["gap"],
                            "det_xy": det_xy.tolist(),
                            "labels": labels[:, track_pos].tolist(),
                            "zero_xy": zero_xy[track_pos].tolist(),
                            "cv_xy": cv_xy[track_pos].tolist(),
                            "kv_xy": kv_xy[track_pos].tolist(),
                            "grae_score": scores["grae"][:, track_pos].tolist(),
                            "kv_score": scores["kv"][:, track_pos].tolist(),
                        },
                        6,
                        kind == "success",
                    )
                for det_pos in np.flatnonzero(known):
                    pairs.append(
                        {
                            "label": int(labels[det_pos, track_pos]),
                            "sequence_id": sequence_id,
                            "region": area,
                            "gap": row["gap"],
                            "covered": int(covered[track_pos]),
                            "oracle_ok": int(oracle_ok[track_pos]),
                            "grae": float(scores["grae"][det_pos, track_pos]),
                            "cv": float(scores["cv"][det_pos, track_pos]),
                            "kv": float(scores["kv"][det_pos, track_pos]),
                            "reliability": float(scores["reliability"][det_pos, track_pos]),
                            "oracle": float(scores["oracle"][det_pos, track_pos]) if np.isfinite(scores["oracle"][det_pos, track_pos]) else None,
                            "dist_grae": float(dist_zero[det_pos, track_pos]),
                            "dist_cv": float(dist_cv[det_pos, track_pos]),
                            "dist_kv": float(dist_kv[det_pos, track_pos]),
                            "dist_oracle": None if not np.isfinite(dist_oracle[det_pos, track_pos]) else float(dist_oracle[det_pos, track_pos]),
                        }
                    )
        for item in list(assignments) + list(created):
            raw_index = int(item["input_index"])
            source = det_row["objects"][raw_index]
            gid = det_gt.get(raw_index)
            world, world_yaw = world_pose(meta["rotation"], meta["translation"], source["box"][:3], source["box"][3])
            history[int(item["track_id"])].append(
                {
                    "time": now,
                    "lidar_xy": np.asarray(source["box"][:2], dtype=np.float64),
                    "world_xy": world[:2].astype(np.float64),
                    "world_yaw": world_yaw,
                    "class_name": source["class_name"],
                    "intersection": meta["intersection"],
                }
            )
            book.commit(item["track_id"], gid, source["box"][:2], None, det_row["timestamp"])
    baseline = list(read_jsonl(project_path(config, config["baseline_prediction_root"]) / ("%s.jsonl" % sequence_id)))
    same = True
    try:
        _equivalent(predictions, baseline, sequence_id)
    except AssertionError as exc:
        same = False
        print("预测与已发布 GRAE 不一致 %s %s" % (sequence_id, exc), flush=True)
    return pairs, tracks, same


def _subset(pairs, tracks, sequence_ok, region, gap, covered_only):
    def pair_ok(item):
        if not sequence_ok(item["sequence_id"]):
            return False
        if region != "all" and item["region"] != region:
            return False
        if gap != "all" and item["gap"] != gap:
            return False
        if covered_only and not item["covered"]:
            return False
        return True

    def track_ok(item):
        if not sequence_ok(item["sequence_id"]):
            return False
        if region != "all" and item["region"] != region:
            return False
        if gap != "all" and item["gap"] != gap:
            return False
        if covered_only and not item["covered"]:
            return False
        return True

    return [item for item in pairs if pair_ok(item)], [item for item in tracks if track_ok(item)]


def _hard_auc(pair_subset, method):
    # 4 m 来自既有关联距离分箱，只保留近处负样本，避免远处负样本把 AUC 顶满
    chosen = []
    for item in pair_subset:
        if method == "oracle" and not item["oracle_ok"]:
            continue
        if item["label"] == 1 or (item["dist_grae"] is not None and float(item["dist_grae"]) <= 4.0):
            chosen.append(item)
    labels = [item["label"] for item in chosen]
    scores = [item[method] if item[method] is not None else np.nan for item in chosen]
    return _auc(labels, scores), len(chosen), int(sum(labels))


def _row(name, method, pair_subset, track_subset, deployable, covered_only):
    labels = [item["label"] for item in pair_subset]
    scores = [item[method] if item[method] is not None else np.nan for item in pair_subset]
    if method == "oracle":
        labels = [item["label"] for item in pair_subset if item["oracle_ok"]]
        scores = [item["oracle"] if item["oracle"] is not None else np.nan for item in pair_subset if item["oracle_ok"]]
    usable = [item for item in track_subset if method != "oracle" or item["oracle_ok"]]
    sep_values = [item["sep_%s" % method] for item in usable if item["sep_%s" % method] != ""]
    margin_values = [item["margin_%s" % method] for item in usable if item.get("margin_%s" % method) not in ("", None)]
    top_values = [item["top1_%s" % method] for item in usable]
    grae_hit = [item["hit_grae"] for item in usable]
    method_hit = [item["hit_%s" % method] for item in usable]
    corrections = int(sum(not left and right for left, right in zip(grae_hit, method_hit)))
    new_errors = int(sum(left and not right for left, right in zip(grae_hit, method_hit)))
    dist_name = {"grae": "pos_dist_grae", "cv": "pos_dist_cv", "kv": "pos_dist_kv", "reliability": "pos_dist_grae", "oracle": "pos_dist_oracle"}[method]
    pos_dist = [item[dist_name] for item in usable if item.get(dist_name) not in ("", None)]
    hard_auc, hard_n, hard_pos = _hard_auc(pair_subset, method)
    return {
        "slice": name,
        "method": method,
        "deployable": int(deployable and method in DEPLOYABLE),
        "covered_only": int(covered_only),
        "n_pos": int(sum(labels)),
        "n_neg": int(len(labels) - sum(labels)),
        "auc": "" if _auc(labels, scores) is None else round(_auc(labels, scores), 4),
        "hard_auc": "" if hard_auc is None else round(hard_auc, 4),
        "hard_n": hard_n,
        "hard_pos": hard_pos,
        "separation": "" if not sep_values else round(float(np.mean(sep_values)), 4),
        "margin": "" if not margin_values else round(float(np.mean(margin_values)), 4),
        "top1": "" if not top_values else round(float(np.mean(top_values)), 4),
        "n_tracks": len(usable),
        "corrections": corrections,
        "new_errors": new_errors,
        "pos_dist_median": "" if not pos_dist else round(float(np.median(pos_dist)), 4),
    }


def aggregate(pairs, tracks, edges):
    rows = []
    gap_names = ["%g-%g" % (left, right) for left, right in zip(edges[:-1], edges[1:])]
    slices = [("all", lambda _item: True, "all", "all")]
    slices.append(("no_0066", lambda item: item != "0066", "all", "all"))
    for region in ("normal", "hard", "unknown"):
        slices.append((region, lambda item: item != "0066", region, "all"))
    for gap in gap_names:
        slices.append((gap, lambda item: item != "0066", "all", gap))
    for name, sequence_ok, region, gap in slices:
        for covered_only in (False, True):
            pair_subset, track_subset = _subset(pairs, tracks, sequence_ok, region, gap, covered_only)
            for method in METHODS:
                rows.append(_row(name, method, pair_subset, track_subset, True, covered_only))
    return rows


def _plot_cases(folder, cases):
    folder.mkdir(parents=True, exist_ok=True)
    for kind, items in cases.items():
        for index, item in enumerate(items):
            figure, axis = plt.subplots(figsize=(6.0, 6.0))
            xy = np.asarray(item["det_xy"], dtype=np.float64)
            labels = np.asarray(item["labels"])
            if np.any(labels == 0):
                axis.scatter(xy[labels == 0, 0], xy[labels == 0, 1], c="#d62728", s=28, label="negative")
            if np.any(labels == 1):
                axis.scatter(xy[labels == 1, 0], xy[labels == 1, 1], c="#2ca02c", s=36, label="positive")
            axis.scatter([item["zero_xy"][0]], [item["zero_xy"][1]], marker="x", c="black", s=60, label="GRAE")
            axis.scatter([item["cv_xy"][0]], [item["cv_xy"][1]], marker="+", c="#1f77b4", s=70, label="CV")
            if np.isfinite(item["kv_xy"]).all():
                axis.scatter([item["kv_xy"][0]], [item["kv_xy"][1]], marker="*", c="#ff7f0e", s=90, label="KV")
            axis.set_aspect("equal")
            axis.set_title("%s %s %s %s" % (kind, item["sequence_id"], item["frame_id"], item["region"]))
            axis.legend(loc="best", fontsize=8)
            figure.tight_layout()
            figure.savefig(folder / ("%s_%02d.png" % (kind, index)), dpi=120)
            plt.close(figure)


def main():
    config = load_config()
    out = output_dir(config) / "association"
    out.mkdir(parents=True, exist_ok=True)
    catalog, folds, _ = build_catalog(config)
    holdout = [str(item) for fold_name in ("A", "B") for item in folds["inner_holdout"][fold_name]]
    analysis = yaml.safe_load(project_path(config, config["grae_config"]).read_text(encoding="utf-8"))
    analysis["_root"] = ROOT
    analysis["device"] = config["association"]["device"]
    analysis["tracker_config"] = config["tracker_config"]
    analysis["checkpoint"] = config["checkpoint"]
    tracker, _, _ = _load_tracker(analysis)
    protocol = V2XSeqProtocol(ROOT)
    banks = {path.stem: load_bank(path) for path in (output_dir(config) / "kb" / "motion").glob("*.npz")}
    reliability = _read_reliability(output_dir(config) / "kb" / "reliability.csv")
    hard = load_hard_regions(config)
    gt_tracks = _gt_lookup(load_cached_tracks(output_dir(config) / "cache" / "tracks.pkl"))
    lanes = LaneIndex(project_path(config, config["project"]["map_root"]), output_dir(config) / "cache")
    pairs = []
    tracks = []
    matches = {}
    cases = {"success": [], "failure": []}
    for sequence_id in holdout:
        seq_pairs, seq_tracks, same = replay_sequence(
            config, tracker, protocol, catalog, banks, reliability, hard, gt_tracks, lanes, sequence_id, cases
        )
        pairs.extend(seq_pairs)
        tracks.extend(seq_tracks)
        matches[sequence_id] = same
        print("关联 %s 候选 %d 轨迹 %d" % (sequence_id, len(seq_pairs), len(seq_tracks)), flush=True)
    edges = [float(item) for item in config["association"]["gap_edges_s"]]
    rows = aggregate(pairs, tracks, edges)
    write_csv(out / "association_metrics.csv", rows)
    write_json(out / "baseline_match.json", matches)
    write_json(out / "cases.json", cases)
    _plot_cases(out / "cases", cases)
    print("关联完成", flush=True)


if __name__ == "__main__":
    main()
