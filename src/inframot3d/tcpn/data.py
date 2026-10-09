"""把检测、GRAE 回放与离线标签整理成逐帧训练样本

模型输入只来自检测框、检测分数、在线轨迹的历史观测与时间戳
status、gt_box、identity 只进入标签字段
"""

import numpy as np

from inframot3d.tcpn.bev_cache import CELL, HEIGHT, WIDTH, X0, Y0
from inframot3d.tcpn.matching import STATUS_BG, STATUS_DUP, STATUS_TP
from inframot3d.tcpn.track_labels import TRACK_STATES, gt_queries, label_sequence

CLASS_INDEX = {"Car": 0, "Van": 1, "Bus": 2, "Truck": 3}
SPEED_LIMIT = 40.0
X_RANGE = (X0, X0 + WIDTH * CELL)
Y_RANGE = (Y0, Y0 + HEIGHT * CELL)
EXIST_POS = ("correct", "unobserved")
EXIST_NEG = ("terminated", "false", "dup_track")


def _inside(xy):
    xy = np.asarray(xy).reshape(-1, 2)
    return (xy[:, 0] >= X_RANGE[0]) & (xy[:, 0] < X_RANGE[1]) & (xy[:, 1] >= Y_RANGE[0]) & (xy[:, 1] < Y_RANGE[1])


def track_arrays(tracks, t_now):
    """轨迹特征原料：最近观测框、恒速预测位置、速度、观测间隔、命中次数与分数"""
    rows = []
    for track in tracks:
        history = track.get("history") or [{"box": track["box"], "score": track["score"], "t": t_now - 100000}]
        last = history[-1]
        velocity = np.zeros(2)
        if len(history) >= 2:
            prev = history[-2]
            dt = (last["t"] - prev["t"]) / 1e6
            if dt > 1e-3:
                velocity = (np.asarray(last["box"][:2]) - np.asarray(prev["box"][:2])) / dt
                speed = np.linalg.norm(velocity)
                if speed > SPEED_LIMIT:
                    velocity = velocity * SPEED_LIMIT / speed
        gap = max((t_now - last["t"]) / 1e6, 0.0)
        pred = np.asarray(last["box"][:2]) + velocity * gap
        scores = [h["score"] for h in history]
        rows.append(
            {
                "box": np.asarray(last["box"], dtype=np.float32),
                "pred": pred.astype(np.float32),
                "vel": velocity.astype(np.float32),
                "gap": float(gap),
                "hits": float(track.get("hits", len(history))),
                "score_last": float(scores[-1]),
                "score_mean": float(np.mean(scores)),
                "cls": int(track.get("class_index", 0)),
            }
        )
    return rows


def pack_tracks_online(tracks, t_now):
    """推理用，只含特征原料，不含任何标签"""
    rows = track_arrays(tracks, t_now)
    m = len(rows)
    out = {
        "box": np.array([r["box"] for r in rows], np.float32).reshape(m, 7),
        "pred": np.array([r["pred"] for r in rows], np.float32).reshape(m, 2),
        "vel": np.array([r["vel"] for r in rows], np.float32).reshape(m, 2),
        "gap": np.array([r["gap"] for r in rows], np.float32),
        "hits": np.array([r["hits"] for r in rows], np.float32),
        "score_last": np.array([r["score_last"] for r in rows], np.float32),
        "score_mean": np.array([r["score_mean"] for r in rows], np.float32),
        "cls": np.array([r["cls"] for r in rows], np.int64),
    }
    keep = _inside(out["pred"]) if m else np.zeros(0, bool)
    return {k: v[keep] for k, v in out.items()}, keep


def _pack_tracks(tracks, t_now, cand_count):
    rows = track_arrays(tracks, t_now)
    m = len(rows)
    out = {
        "box": np.zeros((m, 7), np.float32),
        "pred": np.zeros((m, 2), np.float32),
        "vel": np.zeros((m, 2), np.float32),
        "gap": np.zeros(m, np.float32),
        "hits": np.zeros(m, np.float32),
        "score_last": np.zeros(m, np.float32),
        "score_mean": np.zeros(m, np.float32),
        "cls": np.zeros(m, np.int64),
        "state": np.zeros(m, np.int64),
        "target": -np.ones(m, np.int64),
        "exist": -np.ones(m, np.float32),
        "pos_target": np.zeros((m, 2), np.float32),
        "pos_valid": np.zeros(m, np.float32),
    }
    for j, (row, track) in enumerate(zip(rows, tracks)):
        for key in ("box", "pred", "vel", "gap", "hits", "score_last", "score_mean", "cls"):
            out[key][j] = row[key]
        state = track["state"]
        out["state"][j] = TRACK_STATES.index(state)
        target = int(track.get("target", -1))
        out["target"][j] = target if 0 <= target < cand_count else -1
        if state in EXIST_POS:
            out["exist"][j] = 1.0
        elif state in EXIST_NEG:
            out["exist"][j] = 0.0
        if track.get("gt_box") is not None and state in EXIST_POS:
            out["pos_target"][j] = np.asarray(track["gt_box"][:2], np.float32)
            out["pos_valid"][j] = 1.0
    keep = _inside(out["pred"]) if m else np.zeros(0, bool)
    return {key: value[keep] for key, value in out.items()}, keep


def _gt_history(gt_rows, index, track_id, depth=5):
    history = []
    for back in range(depth, 0, -1):
        k = index - back
        if k < 0:
            continue
        for item in gt_rows[k]["objects"]:
            if str(item.get("source_track_id")) == track_id:
                history.append({"box": item["box"], "score": 1.0, "t": int(gt_rows[k]["timestamp"])})
    return history


def build_samples(sequence_id, det_rows, gt_rows, states):
    frames = label_sequence(det_rows, gt_rows, states)
    samples = []
    for index, frame in enumerate(frames):
        cand = frame["cand"]
        boxes = cand["boxes"]
        keep = _inside(boxes[:, :2]) if len(boxes) else np.zeros(0, bool)
        local = np.where(keep)[0]
        remap = -np.ones(len(boxes), np.int64)
        remap[local] = np.arange(len(local))
        status = cand["status"][local]
        iou = cand["iou"][local]
        max_iou = cand["max_iou"][local]
        quality = np.where(status == STATUS_TP, 1.0, np.where(status == STATUS_BG, 0.0, -1.0)).astype(np.float32)
        # 重复框与对应 TP 的 IoU 接近时无法判定谁该保留，置为 ignore，否则作为负样本
        gt_ids = [cand["gt_id"][k] for k in local]
        nearest = cand["nearest"][local]
        tp_iou = {}
        for k, src in enumerate(local):
            if status[k] == STATUS_TP:
                tp_iou[int(cand["gt_index"][src])] = iou[k]
        for k in range(len(local)):
            if status[k] == STATUS_DUP:
                owner = tp_iou.get(int(nearest[k]))
                quality[k] = -1.0 if owner is None or max_iou[k] >= owner - 0.05 else 0.0
        gt_box = np.zeros((len(local), 7), np.float32)
        for k, g in enumerate(gt_ids):
            if status[k] == STATUS_TP:
                gt_box[k] = cand["gt_boxes"][cand["gt_ids"].index(g)]
        cls = np.array([CLASS_INDEX[c] for c in np.asarray(cand["classes"], dtype=object)[local]], np.int64) if len(local) else np.zeros(0, np.int64)
        t_now = frame["timestamp"]
        tracks = []
        for track in frame["tracks"]:
            track = dict(track)
            if track["target"] >= 0:
                track["target"] = int(remap[track["target"]])
            tracks.append(track)
        grae, _ = _pack_tracks(tracks, t_now, len(local))
        gt_tracks = []
        if index > 0:
            for track in gt_queries(gt_rows[index - 1], cand):
                track = dict(track)
                track["target"] = int(remap[track["target"]]) if track["target"] >= 0 else -1
                track["history"] = _gt_history(gt_rows, index, track["identity"])
                track["hits"] = len(track["history"])
                track["class_index"] = CLASS_INDEX.get(_gt_class(gt_rows[index - 1], track["identity"]), 0)
                if track["state"] in EXIST_POS:
                    g = track["identity"]
                    if g in cand["gt_ids"]:
                        track["gt_box"] = cand["gt_boxes"][cand["gt_ids"].index(g)].tolist()
                gt_tracks.append(track)
        gt_pack, _ = _pack_tracks(gt_tracks, t_now, len(local))
        samples.append(
            {
                "sequence_id": sequence_id,
                "frame_id": frame["frame_id"],
                "timestamp": t_now,
                "cand_box": boxes[local],
                "cand_score": cand["scores"][local],
                "cand_cls": cls,
                "cand_input_index": np.asarray(cand["input_index"], np.int64)[local] if len(local) else np.zeros(0, np.int64),
                "cand_status": status,
                "cand_quality": quality,
                "cand_iou": iou,
                "cand_gt_box": gt_box,
                "cand_gt_id": [g for g in gt_ids],
                "gt_boxes": cand["gt_boxes"],
                "gt_ids": cand["gt_ids"],
                "trk": grae,
                "gt_trk": gt_pack,
                "det_role": [frame["det_role"][k] for k in local],
            }
        )
    return samples


def _gt_class(row, track_id):
    for item in row["objects"]:
        if str(item.get("source_track_id")) == track_id:
            return item["class_name"]
    return "Car"
