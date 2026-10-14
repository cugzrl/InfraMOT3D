"""真实困难关联和受控缺测上的离线重打分

检测、网络权重和候选集合在真实数据上保持不变
GT 只用于标注和 Oracle，不进入候选分数
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
sys.path.insert(0, str(ROOT / "scripts" / "research_knowledge"))
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
    world_pose,
    write_csv,
    write_json,
)
from kv import BASE_VARIANT, LaneGraph, at_time, cv_path, local_to_lidar, mode_blend, residual_forecast, select_neighbors

METHODS = ("grae", "cv", "kv", "kv_mode", "kv_res", "oracle")
DEPLOYABLE = {"grae", "cv", "kv", "kv_mode", "kv_res"}


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


def _motion(history, now, bank, allowed, config, lanes, lane_graph, rotation):
    # 检索失败或不可靠时，残差方法退回匀速，绝对 KV 退回零速度
    empty = {"kv": None, "kv_mode": None, "kv_res": None, "applied": False, "covered": False}
    if bank is None or len(history) < 2 or history[-1]["class_name"] not in set(config["vehicle_classes"]):
        return empty
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
        return empty
    if len(np.unique(hist_index)) != len(hist_index):
        return empty
    sample = make_sample(pseudo, len(history) - 1, hist_index, lanes, config)
    query = {
        "origin": sample["origin"],
        "heading": sample["heading"],
        "hist": sample["hist"],
        "speed": sample["speed"],
        "yaw_rate": sample["yaw_rate"],
        "region": sample["region"],
        "class_name": history[-1]["class_name"],
        "intersection": history[-1]["intersection"],
        "lane": -1,
    }
    chosen, costs = select_neighbors(bank, query, allowed, BASE_VARIANT, config, lane_graph)
    if chosen.size == 0:
        return empty
    tau = now - anchor
    dt = float(config["dt_s"])
    steps = int(round(float(config["future_s"]) / dt))
    found = dict(empty)
    found["covered"] = True
    blend = blend_futures(bank["future"][chosen], costs)
    local = at_time(blend, tau, dt)
    if local is not None:
        found["kv"] = local_to_lidar(local, sample["heading"], rotation, history[-1]["lidar_xy"])
    mixed = mode_blend(bank["future"][chosen], costs, bank["mode"][chosen])
    local = None if mixed is None else at_time(mixed, tau, dt)
    if local is not None:
        found["kv_mode"] = local_to_lidar(local, sample["heading"], rotation, history[-1]["lidar_xy"])
    base = cv_path(sample["hist"], dt, steps)
    corrected, applied = residual_forecast(base, bank["residual"][chosen], costs, bank["mode"][chosen], config)
    local = at_time(corrected, tau, dt)
    if local is not None and applied:
        found["kv_res"] = local_to_lidar(local, sample["heading"], rotation, history[-1]["lidar_xy"])
        found["applied"] = True
    return found


def _frame_map(protocol, gt_row, det_row, config):
    gt_items = protocol.filter_gt(gt_row["objects"])
    indexed = [(index, item) for index, item in enumerate(det_row["objects"]) if _keep_vehicle(protocol, item)]
    matches = _match(
        [item["box"] for item in gt_items],
        [item["box"] for _, item in indexed],
        config["match_iou"],
        config["match_center_gate_m"],
    )
    counts = Counter(str(item["source_track_id"]) for item in gt_items)
    mapping = {}
    pairs = []
    for gt_index, local_index, _ in matches:
        gid = str(gt_items[gt_index]["source_track_id"])
        if counts[gid] > 1:
            continue
        det_index = int(indexed[local_index][0])
        mapping[gid] = det_index
        pairs.append((gt_index, det_index))
    return gt_items, mapping, pairs


def plan_drops(gt_rows, det_rows, protocol, config):
    series = defaultdict(list)
    for frame_index, (gt_row, det_row) in enumerate(zip(gt_rows, det_rows)):
        _, mapping, _ = _frame_map(protocol, gt_row, det_row, config)
        for gid, det_index in mapping.items():
            series[gid].append((frame_index, det_index))
    drops = defaultdict(set)
    reserved = defaultdict(set)
    episodes = []
    for gid, items in series.items():
        for duration in config["association"]["dropout_frames"]:
            duration = int(duration)
            for start in range(2, len(items) - duration):
                window = items[start : start + duration]
                recover = items[start + duration]
                previous = items[start - 1]
                if window[0][0] != previous[0] + 1 or recover[0] != window[-1][0] + 1:
                    continue
                if any(window[offset][0] != window[0][0] + offset for offset in range(duration)):
                    continue
                if any(det_index in drops[frame_index] or det_index in reserved[frame_index] for frame_index, det_index in window):
                    continue
                if recover[1] in drops[recover[0]]:
                    continue
                for frame_index, det_index in window:
                    drops[frame_index].add(det_index)
                reserved[recover[0]].add(recover[1])
                episodes.append(
                    {
                        "gt_id": gid,
                        "duration_frames": duration,
                        "duration_s": round(duration * float(config["dt_s"]), 1),
                        "recover_frame": int(recover[0]),
                        "recover_det": int(recover[1]),
                    }
                )
                break
    return drops, episodes


def _passed_objects(det_row, dropped):
    objects = []
    index_map = []
    for index, item in enumerate(det_row["objects"]):
        if index in dropped:
            continue
        objects.append(
            {"class_name": item["class_name"], "score": float(item["score"]), "box": [float(value) for value in item["box"]]}
        )
        index_map.append(index)
    return objects, index_map


def replay_sequence(
    config,
    tracker,
    protocol,
    catalog,
    banks,
    hard,
    gt_tracks,
    lanes,
    lane_graph,
    sequence_id,
    drops,
    episodes,
    source,
):
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
    edges = [float(item) for item in config["association"]["gap_edges_s"]]
    alpha = float(config["association"]["alpha"])
    high_limit = float(config["association"]["high_cost_limit"])
    low_limit = float(config["association"]["low_cost_limit"])
    margin_limit = float(config["association"]["ambiguous_margin"])
    near_limit = float(config["association"]["near_negative_m"])
    tracker.reset()
    tracker.association_mode = "fusion"
    tracker.high_cost_limit = high_limit
    tracker.low_cost_limit = low_limit
    tracker.enable_debug(True)
    book = IdentityBook()
    history = defaultdict(list)
    owners_seen = defaultdict(set)
    pairs = []
    tracks = []
    predictions = []
    events = Counter()
    recoveries = []
    episode_at = defaultdict(list)
    for episode in episodes:
        episode_at[int(episode["recover_frame"])].append(episode)
    for frame_index, (gt_row, det_row) in enumerate(zip(gt_rows, det_rows)):
        if str(det_row["frame_id"]) != str(gt_row["frame_id"]):
            raise ValueError("帧未对齐 %s" % sequence_id)
        now = int(det_row["timestamp"]) / 1e6
        dropped = drops.get(frame_index, set())
        passed, index_map = _passed_objects(det_row, dropped)
        original_of = {local: original for local, original in enumerate(index_map)}
        outputs = tracker.update(passed, now, det_row["frame_id"])
        if source == "real":
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
        gt_items, mapping, _ = _frame_map(protocol, gt_row, det_row, config)
        events["gt_vehicle"] += len(gt_items)
        events["gt_matched"] += len(mapping)
        events["gt_miss"] += len(gt_items) - len(mapping)
        filtered = set(int(item) for item in snapshot.get("filtered_detection_indices") or [])
        # 过滤下标相对于送入跟踪器的列表
        filtered_original = {original_of[item] for item in filtered if item in original_of}
        for det_index in mapping.values():
            if det_index in filtered_original or det_index in dropped:
                events["gt_filtered"] += int(det_index in filtered_original)
        det_gt = {}
        for gid, det_index in mapping.items():
            if det_index in dropped:
                continue
            if det_index not in index_map:
                continue
            det_gt[index_map.index(det_index)] = gid
        pre_ids = [int(item["track_id"]) for item in pre_tracks]
        owners = defaultdict(list)
        for track_id in pre_ids:
            gt_id, conflict, _ = book.state(track_id)
            if gt_id is not None and not conflict:
                owners[gt_id].append(track_id)
        shared = {track_id for values in owners.values() if len(values) > 1 for track_id in values}
        assigned = {}
        if candidates and pre_tracks and association.get("affinity_matrix"):
            learned = np.asarray(association["affinity_matrix"], dtype=np.float64)
            gate = np.asarray(association.get("gate_mask") or [], dtype=bool)
            det_xy = np.asarray([[item["box"][0], item["box"][1]] for item in candidates], dtype=np.float64)
            det_scores = np.asarray([float(passed[int(item["input_index"])]["score"]) for item in candidates])
            input_indices = np.asarray([int(item["input_index"]) for item in candidates], dtype=np.int64)
            track_ids = np.asarray(pre_ids, dtype=np.int64)
            class_ok = np.zeros(learned.shape, dtype=bool)
            zero_xy = np.zeros((len(pre_tracks), 2), dtype=np.float64)
            cv_xy = np.zeros_like(zero_xy)
            kv_xy = np.full_like(zero_xy, np.nan)
            mode_xy = np.full_like(zero_xy, np.nan)
            res_xy = np.full_like(zero_xy, np.nan)
            oracle_track = np.full_like(zero_xy, np.nan)
            covered = np.zeros(len(pre_tracks), dtype=bool)
            applied = np.zeros(len(pre_tracks), dtype=bool)
            oracle_ok = np.zeros(len(pre_tracks), dtype=bool)
            gap_s = np.zeros(len(pre_tracks), dtype=np.float64)
            for track_pos, track in enumerate(pre_tracks):
                track_id = int(track["track_id"])
                zero_xy[track_pos] = track["predicted_xy"]
                past = history.get(track_id, [])
                if len(past) >= 2:
                    step = past[-1]["time"] - past[-2]["time"]
                    velocity = (past[-1]["lidar_xy"] - past[-2]["lidar_xy"]) / step if step > 1e-3 else np.zeros(2)
                    cv_xy[track_pos] = past[-1]["lidar_xy"] + velocity * (now - past[-1]["time"])
                elif past:
                    cv_xy[track_pos] = past[-1]["lidar_xy"]
                else:
                    cv_xy[track_pos] = zero_xy[track_pos]
                if past:
                    gap_s[track_pos] = now - past[-1]["time"]
                motion = _motion(past, now, bank, allowed, config, lanes, lane_graph, meta["rotation"])
                covered[track_pos] = motion["covered"]
                applied[track_pos] = motion["applied"]
                if motion["kv"] is not None:
                    kv_xy[track_pos] = motion["kv"]
                if motion["kv_mode"] is not None:
                    mode_xy[track_pos] = motion["kv_mode"]
                res_xy[track_pos] = motion["kv_res"] if motion["applied"] and motion["kv_res"] is not None else cv_xy[track_pos]
                gt_id, conflict, _ = book.state(track_id)
                if not conflict:
                    oracle = _oracle_xy(gt_tracks, sequence_id, gt_id, now)
                    if oracle is not None:
                        oracle_track[track_pos] = oracle
                        oracle_ok[track_pos] = True
                for det_pos, detection in enumerate(candidates):
                    class_ok[det_pos, track_pos] = int(detection["class_index"]) == int(track["class_index"])
            dist = {name: np.linalg.norm(det_xy[:, None, :] - zero_xy[None, :, :], axis=-1) for name in ("grae", "kv", "kv_mode")}
            dist["cv"] = np.linalg.norm(det_xy[:, None, :] - cv_xy[None, :, :], axis=-1)
            dist["kv_res"] = np.linalg.norm(det_xy[:, None, :] - res_xy[None, :, :], axis=-1)
            dist["oracle"] = np.full((len(candidates), len(pre_tracks)), np.nan)
            valid_kv = np.isfinite(kv_xy).all(axis=1)
            valid_mode = np.isfinite(mode_xy).all(axis=1)
            if np.any(valid_kv):
                dist["kv"][:, valid_kv] = np.linalg.norm(det_xy[:, None, :] - kv_xy[None, valid_kv, :], axis=-1)
            if np.any(valid_mode):
                dist["kv_mode"][:, valid_mode] = np.linalg.norm(det_xy[:, None, :] - mode_xy[None, valid_mode, :], axis=-1)
            if np.any(oracle_ok):
                dist["oracle"][:, oracle_ok] = np.linalg.norm(det_xy[:, None, :] - oracle_track[None, oracle_ok, :], axis=-1)
            scores = {
                "grae": 0.5 * learned + 0.5 * np.exp(-dist["grae"]),
                "cv": 0.5 * learned + 0.5 * np.exp(-dist["cv"]),
                "kv": 0.5 * learned + 0.5 * np.exp(-dist["kv"]),
                "kv_mode": 0.5 * learned + 0.5 * np.exp(-dist["kv_mode"]),
                "kv_res": 0.5 * learned + 0.5 * np.exp(-dist["kv_res"]),
                "oracle": 0.5 * learned + 0.5 * np.exp(-dist["oracle"]),
            }
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
                    label, _ = label_from_ids(det_gt.get(raw_index), gt_id, conflict, int(track_id) in shared)
                    if label == "positive":
                        labels[det_pos, track_pos] = 1
                    elif label == "negative":
                        labels[det_pos, track_pos] = 0
            for track_pos, track_id in enumerate(track_ids):
                positive = class_ok[:, track_pos] & (labels[:, track_pos] == 1)
                negative = class_ok[:, track_pos] & (labels[:, track_pos] == 0)
                if not np.any(positive):
                    continue
                events["positive_tracks"] += 1
                if gate.ndim == 2 and gate.shape == labels.shape and np.any(positive & ~gate[:, track_pos]):
                    events["positive_gated"] += 1
                positive_inputs = set(int(value) for value in input_indices[positive])
                area = region_name(hard, meta["intersection"], float(zero_xy[track_pos, 0]), float(zero_xy[track_pos, 1]))
                values = scores["grae"][:, track_pos]
                margin = ""
                if np.any(negative) and np.isfinite(values[positive]).any() and np.isfinite(values[negative]).any():
                    margin = float(np.nanmax(values[positive]) - np.nanmax(values[negative]))
                near = bool(np.any(negative & (dist["grae"][:, track_pos] <= near_limit)))
                ambiguous = margin != "" and margin < margin_limit
                events["near_tracks"] += int(near)
                events["ambiguous_tracks"] += int(ambiguous)
                row = {
                    "source": source,
                    "sequence_id": sequence_id,
                    "frame_id": str(det_row["frame_id"]),
                    "frame_index": frame_index,
                    "track_id": int(track_id),
                    "region": area,
                    "gap": _gap_name(float(gap_s[track_pos]), edges),
                    "covered": int(covered[track_pos]),
                    "applied": int(applied[track_pos]),
                    "oracle_ok": int(oracle_ok[track_pos]),
                    "has_neg": int(np.any(negative)),
                    "near": int(near),
                    "ambiguous": int(ambiguous),
                    "margin_grae": margin,
                }
                for name in METHODS:
                    row["pos_dist_%s" % name] = ""
                    if name == "oracle" and not oracle_ok[track_pos]:
                        continue
                    if np.any(positive):
                        row["pos_dist_%s" % name] = float(np.nanmin(dist[name][positive, track_pos]))
                    finite = class_ok[:, track_pos] & np.isfinite(scores[name][:, track_pos])
                    if not np.any(finite):
                        row["top1_%s" % name] = 0
                    else:
                        winner = int(np.argmax(np.where(finite, scores[name][:, track_pos], -np.inf)))
                        row["top1_%s" % name] = int(labels[winner, track_pos] == 1)
                    choice = assigned[name].get(int(track_id))
                    row["hit_%s" % name] = int(choice in positive_inputs)
                    if margin != "" and name != "grae":
                        current = scores[name][:, track_pos]
                        if np.isfinite(current[positive]).any() and np.isfinite(current[negative]).any():
                            row["margin_%s" % name] = float(np.nanmax(current[positive]) - np.nanmax(current[negative]))
                        else:
                            row["margin_%s" % name] = ""
                    elif name == "grae":
                        row["margin_grae_score"] = margin
                if not row["hit_grae"]:
                    events["grae_assignment_miss"] += 1
                tracks.append(row)
                for det_pos in np.flatnonzero(class_ok[:, track_pos] & (labels[:, track_pos] >= 0)):
                    pairs.append(
                        {
                            "source": source,
                            "label": int(labels[det_pos, track_pos]),
                            "sequence_id": sequence_id,
                            "region": area,
                            "gap": row["gap"],
                            "near": int(near),
                            "ambiguous": int(ambiguous),
                            "covered": int(covered[track_pos]),
                            "applied": int(applied[track_pos]),
                            "oracle_ok": int(oracle_ok[track_pos]),
                            "grae_hit": row["hit_grae"],
                            **{name: float(scores[name][det_pos, track_pos]) if np.isfinite(scores[name][det_pos, track_pos]) else None for name in METHODS},
                            **{"dist_%s" % name: None if not np.isfinite(dist[name][det_pos, track_pos]) else float(dist[name][det_pos, track_pos]) for name in METHODS},
                        }
                    )
        for episode in episode_at.get(frame_index, []):
            recover_det = int(episode["recover_det"])
            if recover_det not in index_map:
                events["recover_missing"] += 1
                continue
            passed_index = index_map.index(recover_det)
            gt_id = episode["gt_id"]
            owner = None
            for track_id in pre_ids:
                state, conflict, _ = book.state(track_id)
                if not conflict and state == gt_id:
                    owner = track_id
                    break
            record = dict(episode)
            record["source"] = "synthetic"
            record["sequence_id"] = sequence_id
            record["owner_alive"] = int(owner is not None)
            if owner is None:
                events["recover_lost_track"] += 1
            else:
                events["recover_alive"] += 1
                for name in METHODS:
                    choice = assigned.get(name, {}).get(int(owner))
                    record["hit_%s" % name] = int(choice == passed_index)
            recoveries.append(record)
        for item in list(group.get("assignments", [])) + list(group.get("created", [])):
            raw_index = int(item["input_index"])
            if raw_index < 0 or raw_index >= len(passed):
                continue
            source_obj = passed[raw_index]
            gid = det_gt.get(raw_index)
            previous, conflict, _ = book.state(int(item["track_id"]))
            if gid is not None and previous is not None and not conflict and str(previous) != str(gid):
                events["id_switch"] += 1
            world, world_yaw = world_pose(meta["rotation"], meta["translation"], source_obj["box"][:3], source_obj["box"][3])
            history[int(item["track_id"])].append(
                {
                    "time": now,
                    "lidar_xy": np.asarray(source_obj["box"][:2], dtype=np.float64),
                    "world_xy": world[:2].astype(np.float64),
                    "world_yaw": world_yaw,
                    "class_name": source_obj["class_name"],
                    "intersection": meta["intersection"],
                }
            )
            if gid is not None:
                owners_seen[gid].add(int(item["track_id"]))
            book.commit(item["track_id"], gid, source_obj["box"][:2], None, det_row["timestamp"])
    events["fragmented_gt"] = sum(1 for value in owners_seen.values() if len(value) > 1)
    same = None
    if source == "real":
        baseline = list(read_jsonl(project_path(config, config["baseline_prediction_root"]) / ("%s.jsonl" % sequence_id)))
        same = True
        try:
            _equivalent(predictions, baseline, sequence_id)
        except AssertionError as exc:
            same = False
            print("预测与已发布 GRAE 不一致 %s %s" % (sequence_id, exc), flush=True)
    return pairs, tracks, events, recoveries, same


def _hard_auc(pair_subset, method):
    chosen = []
    for item in pair_subset:
        if method == "oracle" and not item["oracle_ok"]:
            continue
        dist = item.get("dist_grae")
        if item["label"] == 1 or (dist is not None and float(dist) <= 4.0):
            chosen.append(item)
    labels = [item["label"] for item in chosen]
    scores = [item[method] if item[method] is not None else np.nan for item in chosen]
    return _auc(labels, scores)


def _row(name, source, method, pair_subset, track_subset):
    labels = []
    scores = []
    for item in pair_subset:
        if method == "oracle" and not item["oracle_ok"]:
            continue
        labels.append(item["label"])
        scores.append(item[method] if item[method] is not None else np.nan)
    usable = [item for item in track_subset if method != "oracle" or item["oracle_ok"]]
    margins = []
    for item in usable:
        key = "margin_grae" if method == "grae" else "margin_%s" % method
        if item.get(key) not in ("", None):
            margins.append(float(item[key]))
    top_values = [item["top1_%s" % method] for item in usable if "top1_%s" % method in item]
    grae_hit = [item["hit_grae"] for item in usable]
    method_hit = [item["hit_%s" % method] for item in usable]
    dist_name = "pos_dist_%s" % method
    pos_dist = [float(item[dist_name]) for item in usable if item.get(dist_name) not in ("", None)]
    hard_auc = _hard_auc(pair_subset, method)
    return {
        "slice": name,
        "source": source,
        "method": method,
        "deployable": int(method in DEPLOYABLE),
        "n_pos": int(sum(labels)),
        "n_neg": int(len(labels) - sum(labels)),
        "auc": "" if _auc(labels, scores) is None else round(_auc(labels, scores), 4),
        "hard_auc": "" if hard_auc is None else round(hard_auc, 4),
        "margin": "" if not margins else round(float(np.mean(margins)), 4),
        "top1": "" if not top_values else round(float(np.mean(top_values)), 4),
        "n_tracks": len(usable),
        "corrections": int(sum(not left and right for left, right in zip(grae_hit, method_hit))),
        "new_errors": int(sum(left and not right for left, right in zip(grae_hit, method_hit))),
        "pos_dist_median": "" if not pos_dist else round(float(np.median(pos_dist)), 4),
        "applied_rate": "" if not usable else round(float(np.mean([item["applied"] for item in usable])), 4),
        "lost_tracks": "",
    }


def aggregate(pairs, tracks):
    rows = []
    slices = {
        "all": lambda item: item["sequence_id"] != "0066",
        "ambiguous": lambda item: item["sequence_id"] != "0066" and int(item.get("ambiguous", 0)) == 1,
        "near": lambda item: item["sequence_id"] != "0066" and int(item.get("near", 0)) == 1,
        "grae_fail": lambda item: item["sequence_id"] != "0066" and int(item.get("grae_hit", item.get("hit_grae", 1))) == 0,
        "hard_region": lambda item: item["sequence_id"] != "0066" and item.get("region") == "hard",
        "gap_0.5_1": lambda item: item["sequence_id"] != "0066" and item.get("gap") == "0.5-1",
        "gap_1_100": lambda item: item["sequence_id"] != "0066" and item.get("gap") == "1-100",
    }
    real_pairs = [item for item in pairs if item["source"] == "real"]
    real_tracks = [item for item in tracks if item["source"] == "real"]
    for name, predicate in slices.items():
        pair_subset = [item for item in real_pairs if predicate(item)]
        track_subset = [item for item in real_tracks if predicate(item)]
        for method in METHODS:
            rows.append(_row(name, "real", method, pair_subset, track_subset))
    return rows


def aggregate_recovery(recoveries):
    rows = []
    durations = sorted({item["duration_s"] for item in recoveries})
    for duration in durations:
        subset = [item for item in recoveries if item["duration_s"] == duration and item["owner_alive"]]
        lost = [item for item in recoveries if item["duration_s"] == duration and not item["owner_alive"]]
        for method in METHODS:
            hits = [item["hit_%s" % method] for item in subset if "hit_%s" % method in item]
            grae = [item["hit_grae"] for item in subset if "hit_grae" in item]
            rows.append(
                {
                    "slice": "dropout_%s" % duration,
                    "source": "synthetic",
                    "method": method,
                    "deployable": int(method in DEPLOYABLE),
                    "n_pos": len(hits),
                    "n_neg": "",
                    "auc": "",
                    "hard_auc": "",
                    "margin": "",
                    "top1": "" if not hits else round(float(np.mean(hits)), 4),
                    "n_tracks": len(hits),
                    "corrections": int(sum(not left and right for left, right in zip(grae, hits))),
                    "new_errors": int(sum(left and not right for left, right in zip(grae, hits))),
                    "pos_dist_median": "",
                    "applied_rate": "",
                    "lost_tracks": len(lost),
                }
            )
    return rows


def _plot(folder, tracks):
    folder.mkdir(parents=True, exist_ok=True)
    interesting = [item for item in tracks if item["source"] == "real" and item["sequence_id"] != "0066" and item["near"]]
    interesting.sort(key=lambda item: float(item["margin_grae"]) if item["margin_grae"] != "" else 99.0)
    for index, item in enumerate(interesting[:4]):
        figure, axis = plt.subplots(figsize=(6.0, 4.2))
        names = list(METHODS)
        vals = [float(item["pos_dist_%s" % name]) if item.get("pos_dist_%s" % name) not in ("", None) else np.nan for name in names]
        axis.bar(names, vals)
        axis.set_ylabel("positive distance m")
        axis.set_title("%s %s gap %s" % (item["sequence_id"], item["frame_id"], item["gap"]))
        figure.tight_layout()
        figure.savefig(folder / ("near_%02d.png" % index), dpi=120)
        plt.close(figure)


def main():
    config = load_config(Path(__file__).resolve().parent / "config.yaml")
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
    hard = load_hard_regions(config)
    gt_tracks = _gt_lookup(load_cached_tracks(project_path(config, config["source_tracks"])))
    lanes = LaneIndex(project_path(config, config["project"]["map_root"]), project_path(config, config["lane_cache"]))
    lane_graph = LaneGraph(project_path(config, config["project"]["map_root"]))
    pairs = []
    tracks = []
    recoveries = []
    events = Counter()
    matches = {}
    for sequence_id in holdout:
        converted = project_path(config, config["project"]["converted_root"])
        detection_root = project_path(config, config["detection_root"])
        gt_rows = list(read_jsonl(converted / "sequences" / ("%s.jsonl" % sequence_id)))
        det_rows = list(read_jsonl(detection_root / ("%s.jsonl" % sequence_id)))
        seq_pairs, seq_tracks, seq_events, _, same = replay_sequence(
            config, tracker, protocol, catalog, banks, hard, gt_tracks, lanes, lane_graph, sequence_id, {}, [], "real"
        )
        pairs.extend(seq_pairs)
        tracks.extend(seq_tracks)
        events.update(seq_events)
        matches[sequence_id] = same
        print("真实关联 %s 轨迹 %d" % (sequence_id, len(seq_tracks)), flush=True)
        drops, episodes = plan_drops(gt_rows, det_rows, protocol, config)
        syn_pairs, syn_tracks, syn_events, syn_recover, _ = replay_sequence(
            config, tracker, protocol, catalog, banks, hard, gt_tracks, lanes, lane_graph, sequence_id, drops, episodes, "synthetic"
        )
        recoveries.extend(syn_recover)
        for key, value in syn_events.items():
            events["synthetic_%s" % key] += value
        print("缺测 %s 回合 %d" % (sequence_id, len(syn_recover)), flush=True)
    rows = aggregate(pairs, tracks) + aggregate_recovery(recoveries)
    write_csv(out / "association_metrics.csv", rows)
    write_json(out / "hard_events.json", dict(events))
    write_json(out / "baseline_match.json", matches)
    _plot(out / "cases", tracks)
    print("关联完成", flush=True)


if __name__ == "__main__":
    main()
